"""event_horizon's sessions without a GPU: the rules (budget, gates, keep rule, parity,
verdict), the three sessions end to end on a fake shell, the bucket round trip and the
upload hook, the fingerprinted shutdown, and --list on a machine with no nvidia-smi."""

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(tmp_path, monkeypatch):
    for k, v in {"NVME": tmp_path / "nvme", "LOBBOT_JOBS": tmp_path / "nvme" / "jobs",
                 "HORIZON_LOGS": tmp_path / "logs", "HORIZON_RUN": "t1"}.items():
        monkeypatch.setenv(k, str(v))
    for name in ("event_horizon", "blackhole"):
        sys.modules.pop(name, None)
    m = importlib.import_module("event_horizon")
    monkeypatch.setattr(m, "physical_cores", lambda: 6)
    return m


def fake_bucket(m, tmp_path, monkeypatch, login=True):
    root = tmp_path / "bucket"

    def where(x: str) -> Path:
        return root / x[len("bucket://"):] if x.startswith("bucket://") else Path(x)

    def copy(src, dst):
        s, d = where(src), where(dst)
        if not s.exists():
            return False
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        return True

    monkeypatch.setattr(m, "evroc_copy", copy)
    monkeypatch.setattr(m, "evroc_ok", lambda: login)
    return root / "mugge-library" / "horizon" / "t1"


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


TEST_IDS = [f"t{i:03d}" for i in range(182)]
DEV_IDS = [f"d{i:03d}" for i in range(150)]


def bench(m):
    """The fetched problem sets' records, the downloaded original."""
    m.CODEBENCH.mkdir(parents=True, exist_ok=True)
    meta = {"sha": "lcbsha", "windows": {
        "test": {"since": "2025-01-01", "until": "2025-04-30", "ids": TEST_IDS, "ids_sha256": ids_sha(TEST_IDS)},
        "dev": {"since": "2024-08-01", "until": "2024-12-31", "ids": DEV_IDS, "ids_sha256": ids_sha(DEV_IDS)}}}
    (m.CODEBENCH / "livecodebench.meta.json").write_text(json.dumps(meta))
    (m.CODEBENCH / "livecodebench.jsonl").write_text("{}\n")
    (m.CODEBENCH / "codecontests.meta.json").write_text(json.dumps({"sha": "ccsha", "ids_sha256": "ccids"}))
    (m.CODEBENCH / "codecontests.jsonl").write_text("{}\n")
    m.HF_DIR.mkdir(parents=True)
    (m.HF_DIR / "config.json").write_text("{}")
    (m.NVME / "download.log").write_text("DOWNLOADS_DONE\n")


def passes(pid: str, sample: int) -> bool:
    """Every model: 0/K on a fifth of the problems, every other answer on a fifth, all on the rest."""
    i = int(pid[1:])
    return False if i % 5 == 0 else sample % 2 == 0 if i % 5 == 1 else True


def arg(cmd: str, flag: str) -> Path:
    return Path(re.search(rf"{flag} (\S+)", cmd)[1])


class Shell:
    """Stands in for runner.sh: records each command and writes what the real one would."""

    def __init__(self, m):
        self.m, self.calls = m, []

    def __call__(self, step, cmd, env=None):
        self.calls.append((cmd, env or {}))
        m, w = self.m, self.write
        if "--only eval" in cmd:
            self.eval(arg(cmd, "--job"))
        elif "stages.budget" in cmd:
            d = arg(cmd, "--job")
            rep = json.loads((d / "out" / "eval.json").read_text())
            rows = [json.loads(x) for x in (d / "work" / "code_eval" / f"{d.name}.jsonl").read_text().splitlines()]
            first = [r for r in rows if r["sample"] == 0]
            rep["candidates"][0]["code"]["budgets"] = {"8192": {"mean_pass@1": sum(r["passed"] for r in first) / len(first)}}
            w(d / "out" / "eval.json", rep)
            w(d / ".done" / "budgets", {})
        elif "thinking_report.py" in cmd:
            w(arg(cmd, "--out"), {})
        elif "smoke render" in cmd:
            w(arg(cmd, "--traces"), "")
        elif "llama-quantize --pure" in cmd:
            for t in ("q2_k", "q4_k"):
                w(m.smoke_dir() / f"{t}.gguf", "x")
        elif "smoke map" in cmd:
            w(arg(cmd, "--out"), {})
        elif "vocab smoke" in cmd:
            w(Path(cmd.split("> ")[-1].strip()), {"pass": True})
        elif "gen_command(" in cmd:
            t = Path(re.search(r"python - > (\S+)/gen\.sh", cmd)[1])
            target = float(re.search(r"traces\.jsonl'\), ([\d.e+]+)", cmd)[1])
            out = t / "traces.jsonl"
            w(out, (out.read_text() if out.exists() else "") + json.dumps({"id": t.name, "tokens": [1]}) + "\n")
            w(t / "traces.count", {"tokens": target, "target": target, "done": True, "tok_s": 20000})
        elif "stages.vocab run" in cmd:
            hf = arg(cmd, "--out")
            w(hf / "vocab_index.json", {"kept": [0, 1], "tokenizer_sha256": "toksha"})
            w(hf / "config.json", {})
            w(Path(f"{hf}.report.json"), {"trim": True, "rows": 49152, "attempts": [
                {"size": 49152, "pass": True, "model": {"dropped_mass": {"mean": 1e-5, "p999": 1e-3},
                                                        "nll_per_byte": {"rise": 0.001}},
                 "llama_tokenize": {"pass": True}}]})
        elif "stages.dense gguf" in cmd:
            w(arg(cmd, "--bf16"), "bf16")
        elif "stages.dense imatrix" in cmd:
            w(arg(cmd, "--job") / "work" / "imatrix.gguf", "imatrix")
        elif "stages.horizon run" in cmd:
            p = arg(cmd, "--out")
            w(p / "map.json", {"types": {"blk.0.ffn_down.weight": "q2_k", "blk.0.attn_k.weight": "q4_k"},
                               "bytes": 1000, "sha256": "mapsha"})
            w(p / "horizon.json", {"bytes": 1000, "gb": 1e-6})
            w(p / "ptq" / "q2_k.gguf", "q2")
            w(p / "ladder.txt", "text")
        elif "stages.qat block" in cmd:
            b = arg(cmd, "--out")
            w(b / "block.json", {})
            w(b / "state" / "qat.json", {"phase": "block"})
        elif "stages.qat export" in cmd:
            state, out = arg(cmd, "--state"), arg(cmd, "--out")
            w(out / "model.gguf", hashlib.sha256(str(state).encode()).hexdigest()[:10] * 100)  # 1000 bytes
            w(out / "export.json", {"exact": {"w": True}, "bytes": 1000, "gb": 1e-6, "types": {"q2_k": 1, "q4_k": 1}})
            w(out / "vllm" / "config.json", {})
            w(out / "model-hf" / "config.json", {})
        elif "stages.qat e2e" in cmd:
            out = arg(cmd, "--out")
            w(out / "e2e.json", {"tokens": 327680, "minutes": 2.0, "steps": 20})
            w(out / "state" / "qat.json", {"phase": "e2e"})
        elif "stages.qat onpolicy" in cmd:
            o = arg(cmd, "--work")
            final = o / "round2" / "train" / "state"
            w(final / "qat.json", {})
            ok = {"kl": {"new": 0.9, "kept": 1.0, "drop": 0.1, "ok": True}, "loop": {"rise": 0.0, "se": 0.01, "ok": True},
                  "cap": {"rise": 0.0, "se": 0.01, "ok": True}, "pass": {"diff": 0.01, "se": 0.01, "ok": True},
                  "problems": {"n": 128, "ok": True}}
            flat = {**ok, "kl": {"new": 0.99, "kept": 0.9, "drop": 0.01, "ok": False}}
            w(o / "state.json", {"done": True, "trained": 3, "final": str(final), "rounds": [
                {"round": 1, "kept": True, "rules": ok}, {"round": 2, "kept": False, "rules": flat},
                {"round": 3, "kept": False, "rules": flat}]})
            w(o / "onpolicy.json", {"final": str(final), "trained": 3})
        elif "convert_hf_to_gguf.py" in cmd:
            w(Path(re.search(r"--outfile (\S+)\.part", cmd)[1] + ".gguf"), "bf16")
        elif "stages.horizon ladder" in cmd:
            w(arg(cmd, "--out"), [{"gguf": str(arg(cmd, "--gguf")), "kld": 0.004, "top1": 0.99}])
        elif "stages.vocab tokcheck" in cmd:
            w(Path(re.search(r"> (\S+tokenize\.json)", cmd)[1]), {"pass": True})
        elif "stages.gguf_build vllm" in cmd:
            w(arg(cmd, "--out") / "config.json", {})
        elif "event_horizon.py rescore" in cmd:
            d, name = arg(cmd, "--job"), re.search(r"--name (\S+)", cmd)[1]
            src = d / "work" / "code_eval" / f"{name}.jsonl"
            w(d / "work" / "code_eval" / f"{name}-allcases.jsonl", src.read_text())

    def write(self, p: Path, d) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d if isinstance(d, str) else json.dumps(d))

    def eval(self, d: Path) -> None:
        """What eval writes: the graded rows, the effective config, provenance, eval.json."""
        cfg = json.loads((d / "config.json").read_text())
        ids = cfg.get("code_eval_ids") or TEST_IDS
        by_id = cfg.get("code_eval_samples_by_id") or {}
        rows = [{"suite": "livecodebench", "id": i, "sample": s, "seed": cfg.get("code_eval_seed_offset", 0) + s,
                 "passed": passes(i, s), "finish": "stop", "tokens": 900, "reasoning_chars": 4000}
                for i in ids for s in range(by_id.get(i, cfg["code_eval_samples"]))]
        name = d.name
        self.write(d / "work" / "code_eval" / f"{name}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        self.write(d / "work" / "config.effective.json", {**cfg, "llama_cpp": "/mnt/nvme/llama.cpp"})
        self.write(d / "work" / "provenance.json", {"gpu": ["NVIDIA L40S, 46068, 550.1"], "vllm_venv": {"vllm": "0.31.0"},
                                                    "llama_cpp": "af50f2f", "sandbox_workers": 4, "nproc": 8,
                                                    "git_head": "abc"})
        self.write(d / "out" / "eval.json", {"candidates": [{"name": name, "code": {
            "suites": {"livecodebench": {"ids_sha256": ids_sha(ids)}}, "tok_s_total": 1500.0}}]})
        self.write(d / ".done" / "eval", {})


def kinds(calls) -> list[str]:
    out = []
    for c, _ in calls:
        k = ("eval" if "--only eval" in c else "budget" if "stages.budget" in c else "thinking" if "thinking_report" in c
             else "gen" if "gen_command(" in c else "setup" if "setup_vm.sh" in c else "rescore" if "rescore --job" in c
             else re.search(r"(stages\.\w+ \w+|smoke \w+|llama-quantize|convert_hf_to_gguf)", c)[1])
        out.append(k)
    return out


def run_steps(m, session):
    for s in m.steps(session)[2:]:  # setup and the restore: tested on their own
        s.log = m.LOGS / "x.log"
        if not s.done():
            s.run(s)


# ---------- the rules ----------

def test_budget_preflight_and_gate0(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    assert m.GOAL_PTS_PER_GB == 40
    assert m.budget_gb(52.3) == 1.172  # 0.85 x 52.3 / 40 = 1.111: the floor
    assert m.budget_gb(56.0) == pytest.approx(1.19)
    assert m.budget_gb(70.0) == 1.25
    assert m.gate0(51.9)["pass"] is False and m.gate0(52.0)["pass"] is True
    assert m.gate0(56.0)["budget_gb"] == pytest.approx(1.19)
    assert m.preflight(1_172_000_000, 1.172, 52.3)["pass"]
    over = m.preflight(1_180_000_000, 1.172, 52.3)
    assert not over["pass"] and "within the budget" in over["why"]
    assert not m.preflight(1_300_000_000, 1.5, 51.0)["pass"]  # 40 x 1.3 = 52 > R0
    assert not m.preflight(1_000_000_000, 1.172, 60.0, types_ok=False)["pass"]


def test_k_allocation_by_the_originals_outcome(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    outs = {"a": [False] * 4, "b": [True, False, False, False], "c": [True] * 4}
    assert m.k_allocation(outs) == {"a": 2, "b": 8, "c": 4}
    with pytest.raises(ValueError):
        m.k_allocation({"a": [True] * 3})


def test_fixed_set_se_lower_bound_and_verdict(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    outs = {"x": [True, False], "y": [True] * 8, "z": [True, True, False, False]}
    R, se = m.score(outs), m.fixed_se(outs)
    assert R == pytest.approx(100 * 2 / 3)
    assert se == pytest.approx(100 * math.sqrt(0.25 / 1 + 0 + 0.25 / 3) / 3)
    assert m.fixed_se({"x": [True]}) is None  # one answer: no within-problem spread
    ok = {"a check": True}
    assert m.verdict(R, se, 1.2, ok) == "point estimate only"  # 55.6/GB, lower bound 29/GB
    assert m.verdict(52.0, 0.8, 1.25, ok) == "reached"  # (52 - 1.3) / 1.25 = 40.6
    assert m.verdict(52.0, 2.0, 1.25, ok) == "point estimate only"  # (52 - 3.3) / 1.25 = 39.0
    assert m.verdict(45.0, 0.8, 1.25, ok) == "not reached"
    assert m.verdict(60.0, 0.1, 1.0, {"0 errored answers": False}).startswith("blocked (0 errored answers")


def test_screen_gates_bootstrap_and_paired(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    orig = {f"p{i}": [i % 3 != 0] * 2 for i in range(30)}
    half = {k: [v[0], False] for k, v in orig.items()}
    g = m.screen_gate("gate1", 56.0, half, orig, 1.2, m.GATE1_MIN_PTS_PER_GB)
    assert g["retention"] == pytest.approx(0.5) and g["predicted_pts_per_gb"] == pytest.approx(56 * 0.5 / 1.2, abs=1e-3)
    assert not g["pass"]
    assert m.screen_gate("gate2", 56.0, orig, orig, 1.2, m.GATE2_MIN_PTS_PER_GB)["pass"]  # 46.7/GB
    lo, hi = m.bootstrap(orig)
    assert lo < m.score(orig) < hi and m.bootstrap(orig) == (lo, hi)  # seeded
    diff, se, n = m.paired(half, orig)
    assert n == 30 and diff == pytest.approx(-100 * 20 / 30 / 2) and se > 0


def test_round_rule_and_the_audit_of_the_trainers_decisions(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    good = {"kl": {"new": 0.94, "kept": 1.0}, "loop": {"rise": 0.02, "se": 0.01}, "cap": {"rise": -0.01, "se": 0.01},
            "pass": {"diff": -0.02, "se": 0.01}, "problems": {"n": 128}}
    assert all(m.round_rule(good).values())
    assert not m.round_rule({**good, "kl": {"new": 0.96, "kept": 1.0}})["kl"]  # a 4% drop
    assert not m.round_rule({**good, "loop": {"rise": 0.03, "se": 0.01}})["loop"]
    assert not m.round_rule({**good, "pass": {"diff": -0.03, "se": 0.01}})["pass"]
    bad = {**good, "kl": {"new": 1.0, "kept": 1.0}}
    assert m.rounds_over([True, False]) is None and m.rounds_over([True, False, False])
    assert m.rounds_over([True] * 4)
    fine = {"done": True, "rounds": [{"round": 1, "kept": True, "rules": good},
                                     {"round": 2, "kept": False, "rules": bad}, {"round": 3, "kept": False, "rules": bad}]}
    assert m.audit_rounds(fine) == []
    wrong = {"done": True, "rounds": [{"round": 1, "kept": True, "rules": bad}]}
    probs = m.audit_rounds(wrong)
    assert any("rule says drop" in p for p in probs) and any("no stop rule" in p for p in probs)
    late = {"done": True, "rounds": fine["rounds"] + [{"round": 4, "kept": False, "rules": bad}]}
    assert any("after the stop rule" in p for p in m.audit_rounds(late))


def test_parity_rule(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    mp = {"types": {"a": "q2_k", "b": "q4_k", "n": "f32"}, "bytes": 1_000_000}
    ex = {"exact": {"a": True, "b": True}, "bytes": 1_004_000, "types": {"q2_k": 1, "q4_k": 1}}
    kld, tok = {"kld": 0.008, "top1": 0.985}, {"pass": True}
    assert m.parity(ex, mp, kld, tok)["pass"]
    assert "size" in m.parity({**ex, "bytes": 1_006_000}, mp, kld, tok)["why"]
    assert "bit-exact" in m.parity({**ex, "exact": {"a": False}}, mp, kld, tok)["why"]
    assert "type" in m.parity({**ex, "types": {"q2_k": 2}}, mp, kld, tok)["why"]
    assert "KLD" in m.parity(ex, mp, {"kld": 0.02, "top1": 0.99}, tok)["why"]
    assert "top-1" in m.parity(ex, mp, {"kld": 0.002, "top1": 0.97}, tok)["why"]
    assert "tokenize" in m.parity(ex, mp, kld, {"pass": False})["why"]


def test_shares_and_settings_and_provenance(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    rows = [{"finish": "stop"}, {"finish": "length", "forced": True, "reasoning_chars": 9},
            {"finish": "error", "error": "x"}, {"reason": "case_timeout", "finish": "stop", "answer_cut": True}]
    sh = m.shares(rows)
    assert sh["error"] == sh["cap"] == sh["forced"] == sh["case_timeout"] == sh["answer_cut"] == 0.25
    a, b = tmp_path / "a", tmp_path / "b"
    for d, extra, gpu in ((a, {"code_eval_seed_offset": 1000, "code_eval_samples_by_id": {"x": 8}}, "NVIDIA L40S, 1, 2"),
                          (b, {"code_eval_seed_offset": 0}, "NVIDIA L40S, 1, 3")):
        (d / "work").mkdir(parents=True)
        (d / "work" / "config.effective.json").write_text(json.dumps({"code_eval_max_tokens": 16384, **extra}))
        (d / "work" / "provenance.json").write_text(json.dumps({"gpu": [gpu], "nproc": 8, "git_head": str(d)}))
    assert m.settings_diff(a, b) == []  # only what differs between readings by design
    block, other = m.provenance_diff(a, b)
    assert block == [] and other and other[0].startswith("git_head")
    (a / "work" / "config.effective.json").write_text(json.dumps({"code_eval_max_tokens": 8192}))
    (a / "work" / "provenance.json").write_text(json.dumps({"gpu": ["NVIDIA B200, 1, 2"], "nproc": 8}))
    assert m.settings_diff(a, b) == ["code_eval_max_tokens: 8192 vs 16384"]
    assert m.provenance_diff(a, b)[0][0].startswith("gpu")


def test_template_check(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    markup = ["<|start|>", "<|end|>", "<think>"]

    def chat(msgs, default="", gen="<|start|>assistant\n<think>"):
        out = "" if any(x["role"] == "system" for x in msgs) else default
        return out + "".join(f"<|start|>{x['role']}\n{x['content']}<|end|>\n" for x in msgs) + gen

    assert m.template_problems(chat, "Answer in code.", "Sum two ints.", markup) == []
    dflt = m.template_problems(lambda x: chat(x, "<|start|>system\nYou are a helpful model.<|end|>\n"),
                               "Answer in code.", "Sum two ints.", markup)
    assert any("default system text" in p for p in dflt)
    switch = m.template_problems(lambda x: chat(x, gen="<|start|>assistant\n" + ("" if len(x) > 1 else "<think>")),
                                 "Answer in code.", "Sum two ints.", markup)
    assert any("generation prompt differs" in p for p in switch)


# ---------- machines, GPU and the command line ----------

def fake_smi(tmp_path, used: int, name: str = "NVIDIA L40S", mib: int = 46068) -> Path:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    smi = b / "nvidia-smi"
    smi.write_text("#!/bin/bash\ncase \"$*\" in\n"
                   f"  *query-gpu=memory.used*) echo {used};;\n"
                   f"  *query-gpu=name,memory.total*) echo '{name}, {mib}';;\n"
                   "  *query-compute-apps*) echo 'pid, process_name, used_memory [MiB]'; "
                   "echo '4242, python3, 39000 MiB';;\nesac\n")
    smi.chmod(0o755)
    return b


def test_gpu_free_fails_naming_the_processes_that_hold_it(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    env = {**os.environ, "PATH": f"{fake_smi(tmp_path, 40000)}:/usr/bin:/bin"}
    r = subprocess.run(["bash", "-c", m.gpu_free(tries=2, every=0)], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 1 and "4242, python3" in r.stdout and "40000 MiB" in r.stdout
    fake_smi(tmp_path, 100)
    r = subprocess.run(["bash", "-c", m.gpu("echo ran")], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "ran"
    assert m.gpu("x").startswith("set -e\n" + m.GPU_FREE)


def test_machine_check(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    monkeypatch.setenv("PATH", f"{fake_smi(tmp_path, 0)}:/usr/bin:/bin")
    assert m.machine_problem("measure") is None and "B200" in m.machine_problem("train")
    fake_smi(tmp_path, 0, "NVIDIA B200", 183359)
    assert m.machine_problem("train") is None and "L40S" in m.machine_problem("read")
    monkeypatch.setenv("HORIZON_ANY_GPU", "1")
    assert m.machine_problem("read") is None
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert "no GPU" in m.machine_problem("measure")


def test_list_needs_no_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no nvidia-smi, no evroc
    m = load(tmp_path, monkeypatch)
    assert m.main(["--list"]) == 0
    out = capsys.readouterr().out
    for s in ("session measure:", "session train:", "session read:", "GPU smoke (a)-(h)", "record reading, 182 x K_i",
              "on-policy session 1 of up to 5 (round 1)", "on-policy session 5 of up to 5 (judges the last round)",
              "estimates"):
        assert s in out
    assert "original reference reading, 182 x 4 (Gate 0) ~ 4h06m" in out  # 728 answers at the measured rate + 8k
    assert m.main(["--list", "--session", "read"]) == 0
    assert "session measure" not in capsys.readouterr().out
    assert m.main(["summary"]) == 0 and "not read yet" in capsys.readouterr().out
    assert m.main([]) == 2


def test_shutdown_is_refused_without_a_login(tmp_path, monkeypatch, capsys):
    m = load(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "evroc_ok", lambda: False)
    monkeypatch.setattr(m, "run_plan", lambda *a, **k: pytest.fail("must not start"))
    assert m.main(["--session", "read", "--shutdown"]) == 2
    assert "Refusing to start" in capsys.readouterr().out


# ---------- the bucket ----------

def test_upload_and_restore_round_trip(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    monkeypatch.setattr(m, "BIG_BYTES", 100)
    d = m.W / "thing-abc123"
    (d / "sub").mkdir(parents=True)
    (d / "result.json").write_text('{"r": 1}')
    (d / "sub" / "big.bin").write_bytes(b"b" * 500)
    (d / "cache").mkdir()
    (d / "cache" / "skip.npy").write_text("rebuildable")
    (d / "x.part").write_text("partial download")
    art = m.dir_art("thing", lambda: d, ("train",), ("result.json",), ("cache",))
    assert m.upload_art(art)
    base = root / "thing" / "abc123"
    man = json.loads((base / "manifest.json").read_text())
    rel = {f["path"] for f in man["files"]}
    assert rel == {"jobs/event-horizon/thing-abc123/result.json", "jobs/event-horizon/thing-abc123/sub/big.bin"}
    assert (base / "f" / "jobs/event-horizon/thing-abc123/sub/big.bin").stat().st_size == 500
    assert (base / "small.tar").exists() and man["complete"] is True and m.in_sync(art)
    shutil.rmtree(d)
    shutil.rmtree(m.W / ".uploads")
    assert m.restore_art(art, "measure") == "results"  # another session: only the light files
    assert (d / "result.json").exists() and not (d / "sub" / "big.bin").exists()
    assert m.restore_art(art, "train") == "all"
    assert (d / "sub" / "big.bin").read_bytes() == b"b" * 500 and m.in_sync(art)
    assert m.restore_art(art, "train") is None  # in sync: nothing to get


def test_hook_uploads_after_each_step_and_once_at_the_end(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    import runner
    a, b = m.W / "made-aaa111", m.W / "half-bbb222"

    def make(s):
        s.log.write_text("made\n")
        a.mkdir(parents=True)
        (a / "out.json").write_text("{}")

    def fail(s):
        s.log.write_text("failing\n")
        b.mkdir(parents=True)
        (b / "partial.jsonl").write_text("{}\n")
        raise RuntimeError("boom")

    plan = [m.step("measure", "make", {"": 1}, make, lambda: False, m.dir_art("made", lambda: a, ("measure",))),
            m.step("measure", "fail", {"": 1}, fail, lambda: False, m.dir_art("half", lambda: b, ("measure",)))]
    logs = m.LOGS / "measure"
    logs.parent.mkdir(parents=True)
    code = runner.run_plan(plan, lambda: "s", logs, ["run"], after_step=m.make_hook("measure"))
    assert code == 1
    assert json.loads((root / "made" / "aaa111" / "manifest.json").read_text())["complete"] is True
    assert json.loads((root / "half" / "bbb222" / "manifest.json").read_text())["complete"] is False
    assert (root / "summary-measure.txt").exists()
    assert {p.name for p in (root / "logs" / "measure").iterdir()} == {"01-make.log", "02-fail.log"}


def test_shutdown_only_after_exit_0_with_the_uploaded_fingerprint(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    calls = []
    monkeypatch.setattr(m.os, "system", calls.append)
    s = m.Step("x", {"": 1}, None)
    m.run_upload(s, "measure")
    marks = json.loads((m.W / ".uploaded").read_text())  # on the NVMe: wiped with the results it describes
    assert set(marks) == {"measure"} and m.uploaded("measure")
    m.shutdown_after(1, "measure")
    assert calls == []
    m.shutdown_after(0, "measure")
    assert calls == ["sudo shutdown -h +5"]
    (m.CODEBENCH / "livecodebench.jsonl").write_text("changed since the upload\n")
    m.shutdown_after(0, "measure")
    assert calls == ["sudo shutdown -h +5"] and not m.uploaded("measure")


def test_restore_needs_a_login_and_the_earlier_sessions(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch, login=False)
    s = m.Step("x", {"": 1}, None)
    m.run_restore(s, "measure")  # the first session can start without one
    with pytest.raises(RuntimeError, match="no evroc login"):
        m.run_restore(s, "train")
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    with pytest.raises(RuntimeError, match="gate0 has not been decided"):
        m.run_restore(s, "train")
    with pytest.raises(RuntimeError, match="choice.json"):
        m.run_restore(s, "read")


# ---------- the three sessions on a fake shell ----------

def test_three_sessions_end_to_end(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    shell = Shell(m)
    monkeypatch.setattr(m, "sh", shell)
    m.run_setup(m.Step("x", {"": 1}, None))
    assert shell.calls[0][0].startswith("SKIP_EXTRAS=1 ") and "STUDENT= " in shell.calls[0][0]

    shell.calls.clear()
    run_steps(m, "measure")
    k = kinds(shell.calls)
    assert k == ["stages.dense gguf",
                 "smoke template", "smoke render", "eval", "llama-quantize", "smoke pack", "smoke map", "stages.qat block",
                 "stages.vocab smoke", "smoke text", "stages.horizon ladder",
                 "eval", "budget", "thinking",  # original reference reading, Gate 0
                 "eval", "budget", "thinking",  # original dev
                 "gen", "stages.vocab run", "stages.vocab register",
                 "stages.dense gguf", "stages.dense calib", "stages.dense imatrix",
                 "stages.horizon run",
                 "stages.qat block", "stages.qat export",
                 "eval", "budget", "thinking"]  # Block-AP dev screen, Gate 1
    gpu_jobs = [c for c, _ in shell.calls if any(x in c for x in ("--only eval", "stages.budget", "stages.qat",
                                                                  "horizon run", "vocab run", "dense imatrix"))]
    assert gpu_jobs and all(c.startswith("set -e\n" + m.GPU_FREE) for c in gpu_jobs)
    evals = [e for c, e in shell.calls if "--only eval" in c]
    assert all(e["LOBBOT_VLLM_SEQS"] == "96" and e["LOBBOT_SANDBOX_WORKERS"] == "4" for e in evals)
    gen = next(c for c, _ in shell.calls if "gen_command(" in c)
    assert "seqs=160, samples=2" in gen and "8000000.0" in gen
    thinking = [c for c, _ in shell.calls if "thinking_report" in c]
    assert "--ref original-ref-" in thinking[0] and "--ref original-dev-" in thinking[2] and "--json" in thinking[0]
    smoke = json.loads((m.smoke_dir() / "smoke.json").read_text())
    assert smoke["ok"] and set(smoke) >= {"a+d", "b", "c", "e", "f", "g", "h"}

    g0 = m.gate("gate0")
    assert g0["pass"] and g0["R0"] == pytest.approx(100 * (37 * 0.5 + 108) / 182) and g0["budget_gb"] == 1.25
    kal = m.k_alloc()
    assert kal["t000"] == 2 and kal["t001"] == 8 and kal["t002"] == 4 and len(kal) == 182
    assert m.gate("preflight")["pass"] and m.gate("gate1")["pass"]
    ref_cfg = json.loads((m.reading_dir(m.READINGS["original-ref"]) / "config.json").read_text())
    assert ref_cfg["code_eval_samples"] == 4 and ref_cfg["lcb_since"] == "2025-01-01" and ref_cfg["vllm_seqs"] == 96
    assert "size_gb_override" not in ref_cfg and "code_eval_ids" not in ref_cfg
    dev_cfg = json.loads((m.reading_dir(m.READINGS["blockap-dev"]) / "config.json").read_text())
    assert len(dev_cfg["code_eval_ids"]) == 128 and dev_cfg["code_eval_seed_offset"] == 50
    assert list(dev_cfg["size_gb_override"].values()) == [1e-6]  # the GGUF's bytes, not the bf16 checkpoint's
    assert (root / "reading-original-ref").exists()  # uploaded by the last step
    assert m.uploaded("measure") and m.vocab_sha() == "toksha"

    shell.calls.clear()  # a new VM for session train: the restore finds everything, then registers the trim
    m.run_restore(m.Step("x", {"": 1}, None), "train")
    assert kinds(shell.calls) == ["stages.vocab register"]

    shell.calls.clear()
    run_steps(m, "train")
    assert kinds(shell.calls) == ["eval", "stages.qat e2e", "gen", "stages.qat e2e", "stages.qat onpolicy",
                                  "stages.qat export", "stages.vocab register", "convert_hf_to_gguf",
                                  "stages.horizon ladder", "stages.vocab tokcheck"]
    gen = next(c for c, _ in shell.calls if "gen_command(" in c)
    assert "seqs=384" in gen and "24000000.0" in gen
    traces24 = m.traces_file("train").read_text().splitlines()
    assert len(traces24) == 2  # the 8M rows, then gen resumed on the copy
    assert json.loads((m.onpolicy_dir() / "audit.json").read_text())["ok"]
    assert m.gate("parity")["pass"]
    c = m.choice()
    assert c["gguf_sha256"] == hashlib.sha256((m.final_dir() / "model.gguf").read_bytes()).hexdigest()
    assert c["record"]["answers"] == 37 * 2 + 37 * 8 + 108 * 4 and c["record"]["seed_offset"] == 1000
    assert m.minutes()["e2e"] == pytest.approx(min(40e6 / (327680 / 120) / 60, 60) + 5)

    shell.calls.clear()
    run_steps(m, "read")
    assert kinds(shell.calls) == ["stages.gguf_build vllm", "eval", "budget", "thinking",
                                  "eval", "budget", "thinking", "eval", "thinking", "rescore", "rescore"]
    rec = json.loads((m.reading_dir(m.READINGS["final-record"]) / "config.json").read_text())
    assert rec["code_eval_samples_by_id"] == kal and rec["code_eval_seed_offset"] == 1000
    llama = json.loads((m.reading_dir(m.READINGS["final-llama"]) / "config.json").read_text())
    assert list(llama["eval_candidates"].values()) == [str(m.final_gguf())] and llama["code_eval_budgets"] == []
    out = m.summary()
    assert "Verdict (pre-registered in choice.json, one look): REACHED" in out
    assert "final-record (the verdict): 182 problems, 802 answers" in out
    assert "8k re-scored" in out and "all cases" in out and "R/40" in out and "shares:" in out
    assert "settings: same as original-ref" in out
    assert all(ok for ok in m.record_checks().values())

    shell.calls.clear()
    for session in m.SESSIONS:
        run_steps(m, session)
    assert shell.calls == []  # done work is not redone

    # A new choice is a new run: choice.json is never rewritten with another file.
    (m.final_dir() / "model.gguf.sha256").write_text("other\n")
    with pytest.raises(RuntimeError, match="already pre-registers"):
        m.run_choice(m.Step("x", {"": 1}, None))


def test_onpolicy_runs_one_session_per_step_and_uploads_each(tmp_path, monkeypatch):
    """S2.5 on a fake trainer that keeps round 1 and drops rounds 2 and 3: one qat onpolicy
    call per session (--until-session k), the work dir uploaded after each, the audit
    when the stop rule ends the rounds in session 4, session 5 never run."""
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    import runner
    O = m.W / "onpolicy-abc123"
    monkeypatch.setattr(m, "onpolicy_dir", lambda: O)
    monkeypatch.setattr(m, "e2e_dir", lambda: m.W / "e2e-def456")
    ok = {"kl": {"new": 0.9, "kept": 1.0, "drop": 0.1, "ok": True}, "loop": {"rise": 0.0, "se": 0.01, "ok": True},
          "cap": {"rise": 0.0, "se": 0.01, "ok": True}, "pass": {"diff": 0.01, "se": 0.01, "ok": True},
          "problems": {"n": 128, "ok": True}}
    flat = {**ok, "kl": {"new": 0.99, "kept": 0.9, "drop": 0.01, "ok": False}}
    keep = {1: True, 2: False, 3: False}
    calls, seen = [], []

    def trainer(step, cmd, env=None):
        """qat onpolicy's sessions: session k judges round k-1, then trains round k."""
        until = int(re.search(r"--until-session (\d+)", cmd)[1])
        calls.append(until)
        st = json.loads((O / "state.json").read_text()) if (O / "state.json").exists() else \
            {"session": 1, "trained": 0, "rounds": [], "done": False, "final": None}
        while not st["done"] and st["session"] <= until:
            if st["trained"]:
                r = st["trained"]
                st["rounds"].append({"round": r, "kept": keep[r], "rules": ok if keep[r] else flat})
            kept = [r["kept"] for r in st["rounds"]]
            if len(kept) >= m.ROUNDS_MAX or kept[-2:] == [False, False]:
                st.update(done=True, final=str(O / "round1" / "train" / "state"))
                (O / "onpolicy.json").write_text(json.dumps({"final": st["final"], "trained": st["trained"]}))
            else:
                st.update(trained=st["trained"] + 1, session=st["session"] + 1)
        O.mkdir(parents=True, exist_ok=True)
        (O / "state.json").write_text(json.dumps(st))

    upload = m.upload_art

    def recorded(art, complete=True):
        if art.slug == "onpolicy":
            seen.append(json.loads((O / "state.json").read_text())["session"])
        return upload(art, complete)

    monkeypatch.setattr(m, "sh", trainer)
    monkeypatch.setattr(m, "upload_art", recorded)
    plan = [s for s in m.steps("train") if s.name.startswith("on-policy session")]
    assert len(plan) == m.ONPOLICY_SESSIONS == 5
    logs = m.LOGS / "train"
    logs.parent.mkdir(parents=True)
    assert runner.run_plan(plan, lambda: "s", logs, ["run"], after_step=m.make_hook("train")) == 0
    assert calls == [1, 2, 3, 4]  # the stop rule fired in session 4: session 5 is done without a call
    assert seen[:4] == [2, 3, 4, 4]  # each session's state reaches the bucket as it ends
    audit = json.loads((O / "audit.json").read_text())
    assert audit["ok"] and [r["kept"] for r in audit["rounds"]] == [True, False, False]
    assert all(s.state == "done" for s in plan) and m.onpolicy_done()
    assert json.loads((root / "onpolicy" / "abc123" / "manifest.json").read_text())["complete"] is True

    # A session that ends with the state where it was is a failure, not a silent skip.
    st = json.loads((O / "state.json").read_text())
    (O / "state.json").write_text(json.dumps({**st, "done": False, "session": 3}))
    monkeypatch.setattr(m, "sh", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="left session 3 unfinished"):
        m.run_onpolicy(m.Step("x", {"": 1}, None), 3)
    assert not m.onpolicy_session_done(3) and m.onpolicy_session_done(2)


class FloorShell(Shell):
    """horizon run on a budget under its cheapest map: floor.json and exit 3, nothing swept."""

    def __call__(self, step, cmd, env=None):
        if "stages.horizon run" not in cmd:
            return super().__call__(step, cmd, env)
        self.calls.append((cmd, env or {}))
        self.write(arg(cmd, "--out") / "floor.json", {"floor_bytes": 2_000_000_000, "floor_gb": 2.0,
                                                      "budget_bytes": 1_250_000_000, "over": True})


def test_a_budget_under_the_cheapest_map_fails_the_preflight(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    shell = FloorShell(m)
    monkeypatch.setattr(m, "sh", shell)
    skipped = []
    for s in m.steps("measure")[2:]:
        s.log = m.LOGS / "x.log"
        if not s.done():
            try:
                s.run(s)
            except m.Skip:
                skipped.append(s.name)
    assert skipped == ["Block-AP and its export", m.READINGS["blockap-dev"].title]
    pf = m.gate("preflight")
    assert pf["floor"] and not pf["pass"] and pf["gb"] == 2.0 and "within the budget" in pf["why"]
    assert m.ptq_done() and m.uploaded("measure")  # the upload step is not behind the gate
    assert "Preflight: cheapest possible map 2.000 GB" in m.summary()
    assert kinds(shell.calls)[-1] == "stages.horizon run"  # no Block-AP, no dev screen

    # horizon's exit 3 is how the floor is reported; any other failure still fails the step.
    cmd = next(c for c, _ in shell.calls if "stages.horizon run" in c).splitlines()[-1]
    for code, ok in ((3, True), (1, False), (0, True)):
        line = re.sub(r"^python -m stages\.horizon run .*?(?= \|\| )", f"(exit {code})", cmd)
        r = subprocess.run(["bash", "-c", f"set -e\n{line}\necho after"], capture_output=True, text=True)
        assert (r.returncode == 0 and "after" in r.stdout) is ok, (code, line)


# ---------- the command lines against the real CLIs ----------

class Parsed(Exception):
    """Raised once the real CLI has parsed its arguments: nothing after that runs."""


def cli_calls(cmd: str) -> list[tuple[str, list[str]]]:
    """Each python CLI call in a shell command: (module or script, its arguments up to the
    next shell operator). `python -` (a heredoc) is not one."""
    out = []
    for line in cmd.replace("$gen", "4242").splitlines():
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        try:
            toks = list(lex)
        except ValueError:  # a line of a heredoc's python
            continue
        for i, t in enumerate(toks[:-1]):
            if t not in ("python", "python3") or toks[i + 1] == "-":
                continue
            j = i + 3 if toks[i + 1] == "-m" else i + 2
            args = []
            for x in toks[j:]:
                if not set(x) - set("();<>|&"):
                    break
                args.append(x)
            out.append((toks[j - 1], args))
    return out


def parse_in_real_cli(target: str, args: list[str], monkeypatch, capsys) -> argparse.Namespace:
    """The target's own main() up to its parse_args: unknown flags, missing required ones,
    bad types and choices fail here as they would on the VM, and so does a flag that
    argparse would only take as an abbreviation."""
    real, got = argparse.ArgumentParser.parse_args, {}

    def parse(self, a=None, namespace=None):
        got["ns"] = real(self, a, namespace)
        flags = {o for p in [self, *(c for x in self._actions if isinstance(x, argparse._SubParsersAction)
                                      for c in x.choices.values())] for x in p._actions for o in x.option_strings}
        short = [x for x in args if x.startswith("--") and x.split("=")[0] not in flags]
        assert not short, f"{target}: {short} only match as abbreviations"
        raise Parsed

    with monkeypatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy: the module is imported already
        mp.setattr(argparse.ArgumentParser, "parse_args", parse)
        mp.setattr(signal, "signal", lambda *a: None)
        mp.setattr(sys, "argv", [target, *args])
        try:
            if target.endswith(".py"):
                runpy.run_path(str(ROOT / target), run_name="__main__")
            else:
                runpy.run_module(target, run_name="__main__", alter_sys=False)
        except Parsed:
            return got["ns"]
        except SystemExit as e:
            pytest.fail(f"{target} {' '.join(args)}\nexit {e.code}: {capsys.readouterr().err[-800:]}")
    pytest.fail(f"{target} {' '.join(args)} never parsed its arguments")


def test_every_command_line_parses_in_the_stage_it_calls(tmp_path, monkeypatch, capsys):
    """The three sessions on the fake shell, then each python command line they ran (and
    the gen script's) through the real CLI's argument parser."""
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    shell = Shell(m)
    monkeypatch.setattr(m, "sh", shell)
    run_steps(m, "measure")
    m.run_restore(m.Step("x", {"": 1}, None), "train")
    run_steps(m, "train")
    run_steps(m, "read")
    cmds = [c for c, _ in shell.calls]
    for c in [c for c in cmds if "gen_command(" in c]:  # the script gen.sh is made from
        script = re.search(r"<<'EOF'\n(.*?)\nEOF", c, re.S)[1]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(script, {})
        cmds.append(buf.getvalue())
    calls = [x for c in cmds for x in cli_calls(c)]
    outside = {Path(t).name for t, _ in calls if Path(t).is_absolute()}
    assert outside == {"convert_hf_to_gguf.py"}  # llama.cpp's own, not in this repo
    calls = [(t, a) for t, a in calls if not Path(t).is_absolute()]
    seen = set()
    for target, args in calls:
        ns = parse_in_real_cli(target, args, monkeypatch, capsys)
        seen.add(f"{target} {getattr(ns, 'cmd', None) or args[0] if args else ''}".strip())
    assert {"stages.dense gguf", "stages.dense calib", "stages.dense imatrix", "stages.vocab smoke",
            "stages.vocab run", "stages.vocab register", "stages.vocab tokcheck", "stages.horizon ladder",
            "stages.horizon run", "stages.qat block", "stages.qat export", "stages.qat e2e", "stages.qat onpolicy",
            "stages.gguf_build vllm", "stages.ternary gen", "stages.budget --job", "pipeline.py --job",
            "scripts/event_horizon.py rescore"} <= seen
    assert any(t == "scripts/thinking_report.py" for t, _ in calls)
    assert any(t == "scripts/event_horizon.py" and a[:1] == ["smoke"] for t, a in calls)
