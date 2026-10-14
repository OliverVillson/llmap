"""event_horizon's sessions without a GPU: the rules (budget, gates, keep rule, parity,
verdict), the three sessions and session all end to end on a fake shell, the bucket round
trip and the upload hook, the fingerprinted shutdown, and --list on a machine with no nvidia-smi."""

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
import tarfile
import tempfile
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


def bench(m, lcb_sha="lcbsha"):
    """The fetched problem sets' records, the downloaded original."""
    m.CODEBENCH.mkdir(parents=True, exist_ok=True)
    meta = {"sha": lcb_sha, "windows": {
        "test": {"since": "2025-01-01", "until": "2025-04-30", "ids": TEST_IDS, "ids_sha256": ids_sha(TEST_IDS)},
        "dev": {"since": "2024-08-01", "until": "2024-12-31", "ids": DEV_IDS, "ids_sha256": ids_sha(DEV_IDS)}}}
    (m.CODEBENCH / "livecodebench.meta.json").write_text(json.dumps(meta))
    (m.CODEBENCH / "livecodebench.jsonl").write_text("{}\n")
    (m.CODEBENCH / "codecontests.meta.json").write_text(json.dumps({"sha": "ccsha", "ids_sha256": "ccids"}))
    (m.CODEBENCH / "codecontests.jsonl").write_text("{}\n")
    (m.CODEBENCH / "apps.meta.json").write_text(json.dumps({"sha": "appssha", "ids_sha256": "appsids"}))
    (m.CODEBENCH / "apps.jsonl").write_text("{}\n")
    granite(m)
    (m.NVME / "download.log").write_text("DOWNLOADS_DONE\n")


COMMIT = "c0ffee" * 6 + "abcd"


def safetensors_header(p: Path, shapes: dict) -> None:
    """A safetensors file's header (8-byte length, then the JSON), without its data."""
    head = json.dumps({k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}).encode()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(len(head).to_bytes(8, "little") + head)


def granite_shapes(m, layers=40) -> dict:
    shapes = {"model.embed_tokens.weight": [100352, 2560], "lm_head.weight": [100352, 2560], "model.norm.weight": [2560]}
    for i in range(layers):
        shapes |= {f"model.layers.{i}.{s}.weight": [8, 8] for s in m.BASE_LINEARS}
        shapes |= {f"model.layers.{i}.{n}.weight": [2560] for n in ("input_layernorm", "post_attention_layernorm")}
    return shapes


def granite(m, d=None, shapes=None, **cfg) -> Path:
    """What `hf download` leaves for the base: its config, its weights' headers and the
    commit it read, in .cache/huggingface/download/<file>.metadata."""
    d = d or m.HF_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**m.BASE_CONFIG, **cfg}))
    safetensors_header(d / "model.safetensors", granite_shapes(m) if shapes is None else shapes)
    meta = d / ".cache" / "huggingface" / "download" / "config.json.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(f"{COMMIT}\netag\n1760000000.0\n")
    return d


def passes(pid: str, sample: int) -> bool:
    """Every model: 0/K on a fifth of the problems, every other answer on a fifth, all on the rest."""
    i = int(pid[1:])
    return False if i % 5 == 0 else sample % 2 == 0 if i % 5 == 1 else True


def arg(cmd: str, flag: str) -> Path:
    return Path(re.search(rf"{flag} (\S+)", cmd)[1])


L40S = "NVIDIA L40S, 46068 MiB, 550.1"
B200 = "NVIDIA B200, 183359 MiB, 580.95"


class Shell:
    """Stands in for runner.sh: records each command and writes what the real one would,
    its readings' provenance on `gpu`."""

    def __init__(self, m, gpu=L40S):
        self.m, self.calls, self.gpu = m, [], gpu

    def passed(self, name: str, pid: str, sample: int) -> bool:
        """Whether answer `sample` to problem pid passes in the reading (job) `name`."""
        return passes(pid, sample)

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
            p0 = sum(r["passed"] for r in first) / len(first)  # budget.long_run: 16k on the 8k re-score's answers
            rep["candidates"][0]["code"]["budgets"] = {"8192": {"mean_pass@1": 0.9 * p0}, "16384": {"mean_pass@1": p0}}
            w(d / "out" / "eval.json", rep)
            w(d / ".done" / "budgets", {})
        elif "thinking_report.py" in cmd:
            w(arg(cmd, "--out"), {})
        elif "smoke render" in cmd:
            w(arg(cmd, "--traces"), "")
        elif "smoke template" in cmd:
            w(arg(cmd, "--out"), {"problems": [], "default_system_text": "You are a helpful model", "accepted": False})
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
            w(out / "e2e.json", {"tokens": 327680, "minutes": 2.0, "steps": 20, "peak_gb": 151.2, "tok_s": 2900.0})
            w(out / "state" / "qat.json", {"phase": "e2e"})
        elif "stages.qat onpolicy" in cmd and "--until-session 1" in cmd and "--rounds 1 " in cmd:  # S2.2's smoke
            o, rd = arg(cmd, "--work"), arg(cmd, "--work") / "round1"
            w(rd / "train" / "train.json", {"steps": 4, "minutes": 0.4, "peak_gb": 163.0})
            row = lambda n: json.dumps({"id": "c", "prompt": [1] * 100, "tokens": [2] * n, "passed": False}) + "\n"
            w(rd / "held.labelled.jsonl", row(16000) + row(900))
            w(rd / "rollouts.labelled.jsonl", row(22000) + row(3000))
            w(o / "state.json", {"session": 2, "trained": 1, "done": False, "rounds": []})
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
        name = d.name
        rows = [{"suite": "livecodebench", "id": i, "sample": s, "seed": cfg.get("code_eval_seed_offset", 0) + s,
                 "passed": self.passed(name, i, s), "finish": "stop", "tokens": 900, "reasoning_chars": 4000}
                for i in ids for s in range(by_id.get(i, cfg["code_eval_samples"]))]
        self.write(d / "work" / "code_eval" / f"{name}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        self.write(d / "work" / "config.effective.json", {**cfg, "llama_cpp": "/mnt/nvme/llama.cpp"})
        self.write(d / "work" / "provenance.json", {"gpu": [self.gpu], "vllm_venv": {"vllm": "0.31.0"},
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


# What each session's steps run on the fake shell, in order, with nothing done before.
MEASURE_KINDS = ["stages.dense gguf",
                 "smoke template", "smoke render", "eval", "llama-quantize", "smoke pack", "smoke map", "stages.qat block",
                 "stages.vocab smoke", "smoke text", "stages.horizon ladder",
                 "eval", "budget", "thinking",  # original reference reading, Gate 0
                 "eval", "budget", "thinking",  # original dev
                 "gen", "stages.vocab run", "stages.vocab register",
                 "stages.dense gguf", "stages.dense calib", "stages.dense imatrix",
                 "stages.horizon run",
                 "stages.qat block", "stages.qat export",
                 "eval", "budget", "thinking"]  # Block-AP dev screen, Gate 1
TRAIN_KINDS = ["eval", "stages.qat e2e", "stages.qat onpolicy",  # S2.2: rates, on-policy smoke
               "gen", "stages.qat e2e", "stages.qat onpolicy",
               "stages.qat export", "stages.vocab register", "convert_hf_to_gguf",
               "stages.horizon ladder", "stages.vocab tokcheck"]
READ_KINDS = ["stages.gguf_build vllm", "eval", "budget", "thinking",  # final dev screen, Gate 2
              "eval", "budget", "thinking", "eval", "thinking", "rescore", "rescore"]


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

    assert m.template_problems(chat, "Answer in code.", "Sum two ints.", markup) == ([], "")
    # A default system text only when none is given (Granite's templates) is reported, not
    # a failure: every prompt the run sends carries codebench.SYSTEM.
    probs, dflt = m.template_problems(lambda x: chat(x, "<|start|>system\nYou are a helpful model.<|end|>\n"),
                                      "Answer in code.", "Sum two ints.", markup)
    assert probs == [] and dflt == "You are helpful model"
    switch, _ = m.template_problems(lambda x: chat(x, gen="<|start|>assistant\n" + ("" if len(x) > 1 else "<think>")),
                                    "Answer in code.", "Sum two ints.", markup)
    assert any("generation prompt differs" in p for p in switch)
    # ... and text next to our system message still fails
    dated = lambda msgs: chat([{**x, "content": "Today is 2026-10-13.\n" + x["content"]} if x["role"] == "system"
                               else x for x in msgs])
    extra, _ = m.template_problems(dated, "Answer in code.", "Sum two ints.", markup)
    assert any("next to our system message" in p and "Today" in p for p in extra)


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
    """Readings run on an L40S or a B200 (RECORD_SEQS has each one's value); training, and
    so session all, needs a B200's memory; another GPU reads only with HORIZON_ANY_GPU=1."""
    m = load(tmp_path, monkeypatch)
    monkeypatch.setenv("PATH", f"{fake_smi(tmp_path, 0)}:/usr/bin:/bin")
    assert m.machine_problem("measure") is None and m.machine_problem("read") is None
    assert "B200" in m.machine_problem("train") and "B200" in m.machine_problem("all")
    fake_smi(tmp_path, 0, "NVIDIA B200", 183359)
    for s in ("measure", "train", "read", "all"):  # a B200 is accepted for readings
        assert m.machine_problem(s) is None, s
    fake_smi(tmp_path, 0, "NVIDIA H100 80GB HBM3", 81559)
    assert "L40S or a B200" in m.machine_problem("read") and "B200" in m.machine_problem("train")
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


def test_shutdown_only_after_exit_0_with_the_uploaded_fingerprint(tmp_path, monkeypatch, capsys):
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    calls = []
    monkeypatch.setattr(m.os, "system", lambda c: calls.append(c) or 0)
    s = m.Step("x", {"": 1}, None)
    m.run_upload(s, "measure")
    marks = json.loads((m.W / ".uploaded").read_text())  # on the NVMe: wiped with the results it describes
    assert set(marks) == {"measure"} and m.uploaded("measure")
    m.shutdown_after(1, "measure")
    assert calls == []
    m.shutdown_after(0, "measure")
    assert calls == ["sudo shutdown -h +5"]
    assert "Shutting the VM down in 5 minutes" in capsys.readouterr().out
    monkeypatch.setattr(m.os, "system", lambda c: calls.append(c) or 256)  # sudo refused
    m.shutdown_after(0, "measure")
    out = capsys.readouterr().out
    assert "SHUTDOWN FAILED" in out and "STILL UP" in out and "Shutting the VM down" not in out
    monkeypatch.setattr(m.os, "system", lambda c: calls.append(c) or 0)
    calls.clear()
    (m.CODEBENCH / "livecodebench.jsonl").write_text("changed since the upload\n")
    m.shutdown_after(0, "measure")
    assert calls == [] and not m.uploaded("measure")


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


class SetupShell:
    """setup_vm.sh on a new VM: a fresh fetch of the suites unless SKIP_SUITES=1 (LiveCodeBench's
    main has moved since session 1: another commit, another file; the other suites are
    fetched when their meta is missing), and this VM's freeze of both venvs. Records what
    CONSTRAINTS held when it ran. A suite in `fail` does not download (setup_vm.sh warns and
    goes on)."""

    def __init__(self, m, freeze="torch==2.14.0\n", fail=()):
        self.m, self.freeze, self.calls, self.constraints, self.fail = m, freeze, [], {}, set(fail)

    def __call__(self, step, cmd, env=None):
        self.calls.append(cmd)
        m = self.m
        cons = re.search(r"CONSTRAINTS=(\S*) ", cmd)[1]
        self.constraints = {f.name: f.read_text() for f in Path(cons).glob("*.txt")} if cons else {}
        if "SKIP_SUITES=1 " not in cmd:
            m.CODEBENCH.mkdir(parents=True, exist_ok=True)
            for s in m.SUITES:
                if s in self.fail or (s != "livecodebench" and (m.CODEBENCH / f"{s}.meta.json").exists()):
                    continue
                fresh = json.dumps({"id": f"new {s}"}).encode() + b"\n"
                (m.CODEBENCH / f"{s}.jsonl").write_bytes(fresh)
                meta = json.loads((m.CODEBENCH / f"{s}.meta.json").read_text()) \
                    if (m.CODEBENCH / f"{s}.meta.json").exists() else {}
                (m.CODEBENCH / f"{s}.meta.json").write_text(json.dumps(
                    {**meta, "sha": "lcbsha-NEW" if s == "livecodebench" else f"{s}-NEW",
                     "file_sha256": hashlib.sha256(fresh).hexdigest()}))
        m.venvs_dir().mkdir(parents=True, exist_ok=True)
        for v in m.VENVS:
            (m.venvs_dir() / f"{v}.txt").write_text(self.freeze)


def new_vm(m):
    shutil.rmtree(m.NVME)
    m.NVME.mkdir(parents=True)


def test_a_new_vm_gets_the_runs_code_suites_from_the_bucket_never_a_fresh_fetch(tmp_path, monkeypatch):
    """orchestration#0: session 1's suites are the run's. A later VM's setup skips the fetch
    and restores the bucket's copy once the NVMe is mounted, checked against the manifest's
    sha256 and the metas'; a fresh fetch is never marked as the pinned copy."""
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    data = b'{"id": "session 1"}\n'
    (m.CODEBENCH / "livecodebench.jsonl").write_bytes(data)
    meta = json.loads((m.CODEBENCH / "livecodebench.meta.json").read_text())
    (m.CODEBENCH / "livecodebench.meta.json").write_text(json.dumps({**meta, "file_sha256": hashlib.sha256(data).hexdigest()}))
    art = m.codebench_art()
    assert not m.upload_art(art, complete=False)  # a partial pin is never uploaded
    assert not (root / "codebench").exists()
    assert m.upload_art(art)  # the setup step's hook in session 1
    man = json.loads((root / "codebench" / "pinned" / "manifest.json").read_text())
    assert all(len(f["sha256"]) == 64 for f in man["files"])

    new_vm(m)  # session train's VM
    shell = SetupShell(m)
    monkeypatch.setattr(m, "sh", shell)
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert "SKIP_SUITES=1 " in shell.calls[0]
    assert m.lcb_meta()["sha"] == "lcbsha" and (m.CODEBENCH / "livecodebench.jsonl").read_bytes() == data
    assert m.in_sync(art) and m.upload_art(art)  # the hook after setup: already the bucket's
    assert m.restore_art(art, "train") is None

    # A fresh fetch (the bucket was out of reach at setup) is refused, not marked in sync,
    # and the restore step then brings back the pinned copy.
    new_vm(m)
    pinned = m.codebench_pinned
    monkeypatch.setattr(m, "codebench_pinned", lambda tmp: False)
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert "SKIP_SUITES= " in shell.calls[-1] and m.lcb_meta()["sha"] == "lcbsha-NEW"
    with pytest.raises(RuntimeError, match="not the ones .* pins"):
        m.upload_art(art)
    assert not m.in_sync(art) and m.restore_art(art, "train") == "all"
    assert m.lcb_meta()["sha"] == "lcbsha" and (m.CODEBENCH / "livecodebench.jsonl").read_bytes() == data
    assert m.codebench_problems() == []

    # A file that is not what its meta recorded: the restore step puts the pinned copy back
    # (a size match is not enough), and a copy it cannot repair stops the run.
    (m.CODEBENCH / "livecodebench.jsonl").write_bytes(b"other bytes, same len\n")
    assert m.codebench_problems() == ["livecodebench.jsonl is not the file its meta records (sha256)"]
    with pytest.raises(RuntimeError, match="session train cannot start"):
        m.run_restore(m.Step("x", {"": 1}, None), "train")
    assert (m.CODEBENCH / "livecodebench.jsonl").read_bytes() == data
    restore = m.restore_art
    monkeypatch.setattr(m, "restore_art", lambda art, session: None)
    (m.CODEBENCH / "livecodebench.jsonl").write_bytes(b"other bytes, same len\n")
    with pytest.raises(RuntimeError, match="not whole: livecodebench.jsonl is not the file"):
        m.run_restore(m.Step("x", {"": 1}, None), "train")
    monkeypatch.setattr(m, "restore_art", restore)
    # ... and so does a bucket copy whose JSONL is not its meta's.
    monkeypatch.setattr(m, "codebench_pinned", pinned)
    new_vm(m)
    bad = tmp_path / "bad.meta.json"
    bad.write_text(json.dumps({**meta, "file_sha256": "0" * 64}))
    with tarfile.open(root / "codebench" / "pinned" / "small.tar", "a") as t:  # extracted last, so it wins
        t.add(bad, arcname="codebench/livecodebench.meta.json")
    with pytest.raises(RuntimeError, match="do not check out"):
        m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert "SKIP_SUITES=1 " in shell.calls[-1]


def test_setup_vm_reruns_keep_the_venvs_the_first_run_built():
    """A failed setup step is re-run on the same VM (seen on the B200 2026-10-13): uv refuses
    to create a venv over an existing one unless told to keep it."""
    lines = [l for l in (ROOT / "scripts" / "setup_vm.sh").read_text().splitlines() if re.match(r"\s*uv venv\b", l)]
    assert len(lines) == 2 and all("--allow-existing" in l for l in lines)


@pytest.mark.parametrize("lost", ["livecodebench", "apps"])
def test_a_partial_first_fetch_is_never_pinned_and_the_rerun_fetches_the_rest(tmp_path, monkeypatch, lost):
    """round 2, orchestration: setup_vm.sh only warns when a suite does not download. The
    first setup must not pin what it got: its step fails, the codebench art refuses to pin
    (whatever the hook says), and the re-run fetches the missing suite and pins all three.
    A missing APPS, which no check used to see, is caught the same way."""
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    m.NVME.mkdir(parents=True)
    shell = SetupShell(m, fail=[lost])
    monkeypatch.setattr(m, "sh", shell)
    with pytest.raises(RuntimeError, match=f"not whole, so nothing is pinned: no {lost}.meta.json"):
        m.run_setup(m.Step("VM setup", {"": 1}, None))
    art = m.codebench_art()
    assert art.unfit() == [f"no {lost}.meta.json"]
    assert not m.upload_art(art, complete=True) and not m.upload_art(art, complete=False)  # the hook, either way
    assert not (root / "codebench").exists() and not m.setup_done()
    with pytest.raises(RuntimeError, match=f"not whole: no {lost}.meta.json"):
        m.run_restore(m.Step("restore", {"": 1}, None), "measure")

    shell.fail.clear()  # the re-run: nothing is pinned, so setup_vm.sh fetches again
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert "SKIP_SUITES= " in shell.calls[-1] and m.codebench_problems() == []
    assert m.upload_art(art)
    man = json.loads((root / "codebench" / "pinned" / "manifest.json").read_text())
    assert sorted(f["path"] for f in man["files"]) == sorted(f"codebench/{s}.{x}" for s in m.SUITES
                                                              for x in ("jsonl", "meta.json"))
    new_vm(m)  # the next session's VM restores all three
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert "SKIP_SUITES=1 " in shell.calls[-1] and m.codebench_problems() == [] and m.in_sync(art)


def test_the_first_setups_venv_freeze_constrains_every_later_vm(tmp_path, monkeypatch):
    """orchestration#1: both venvs are frozen across sessions. The first setup's freeze is
    uploaded once; a later VM installs with it as constraints and never replaces it."""
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    m.NVME.mkdir(parents=True)
    shell = SetupShell(m, freeze="torch==2.14.0\ntransformers==5.17.0\n")
    monkeypatch.setattr(m, "sh", shell)
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert shell.constraints == {} and "CONSTRAINTS= " in shell.calls[0]
    assert {f.name for f in (root / "venvs").iterdir()} == {"venv-vllm.txt", "venv-train.txt"}

    new_vm(m)
    shell.freeze = "torch==2.14.0\ntransformers==5.17.1\n"  # what this VM would install on its own
    m.run_setup(m.Step("VM setup", {"": 1}, None))
    assert f"UV_EXCLUDE_NEWER={m.UV_EXCLUDE_NEWER} " in shell.calls[1]
    assert shell.constraints == {"venv-vllm.txt": "torch==2.14.0\ntransformers==5.17.0\n",
                                 "venv-train.txt": "torch==2.14.0\ntransformers==5.17.0\n"}
    assert (root / "venvs" / "venv-train.txt").read_text().endswith("5.17.0\n")  # the first setup's wins
    assert not m.pin_venvs()
    assert not list(Path(tempfile.gettempdir()).glob("event-horizon-setup-*"))  # the lookup dir is gone


def test_settings_block_on_the_harness_namespace_and_report_the_rest(tmp_path, monkeypatch):
    """orchestration#2: a Config field added between sessions outside the harness (a QAT
    knob, say) is reported only; one inside it, or a changed value, blocks. So does a venv
    version or the suites' records in provenance."""
    m = load(tmp_path, monkeypatch)
    a, b = tmp_path / "a", tmp_path / "b"
    base = {**m.EVAL, "vllm_seqs": m.RECORD_SEQS["L40S"], "code_eval_dir": "/mnt/nvme/codebench", "heal_lr": 1e-4}
    for d in (a, b):
        (d / "work").mkdir(parents=True)
        (d / "work" / "provenance.json").write_text(json.dumps({"gpu": ["NVIDIA L40S, 1, 2"], "eval_venv": {"torch": "2.14.0"}}))
    write = lambda d, cfg: (d / "work" / "config.effective.json").write_text(json.dumps(cfg))
    write(a, {**base, "qat_new_knob": 3, "heal_lr": 2e-4, "eval_model_fingerprint": "x", "code_eval_seed_offset": 1000})
    write(b, {**base, "eval_model_fingerprint": "y"})
    assert m.settings_diff(a, b) == []
    assert m.settings_reported(a, b) == ["heal_lr: 0.0002 vs 0.0001", "qat_new_knob: 3 vs '(none)'"]
    write(a, {**base, "eval_new_knob": 1, "lcb_until": "2025-05-31", "vllm_seqs": 64})
    write(b, {**base, "lcb_until": "2025-04-30"})
    assert m.settings_diff(a, b) == ["eval_new_knob: 1 vs '(none)'", "lcb_until: '2025-05-31' vs '2025-04-30'",
                                     "vllm_seqs: 64 vs 96"]
    assert m.provenance_diff(a, b) == ([], [])
    (a / "work" / "provenance.json").write_text(json.dumps({"gpu": ["NVIDIA L40S, 1, 2"], "eval_venv": {"torch": "2.14.1"},
                                                            "code_suites": {"livecodebench": {"sha": "new"}}}))
    block, _ = m.provenance_diff(a, b)
    assert [x.split(":")[0] for x in block] == ["eval_venv", "code_suites"]


def test_choice_json_is_one_per_run_whatever_moves_upstream(tmp_path, monkeypatch):
    """orchestration#3: the choice art has one key per run. A new VM after a change upstream
    of final_dir restores the run's choice.json, and pre-registering another GGUF is refused."""
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    shell = Shell(m)
    monkeypatch.setattr(m, "sh", shell)
    run_steps(m, "measure")
    m.run_restore(m.Step("x", {"": 1}, None), "train")
    run_steps(m, "train")
    first = m.choice()["gguf_sha256"]
    assert [p.name for p in (root / "choice").iterdir()] == ["run"]
    new_vm(m)
    bench(m)
    monkeypatch.setitem(m.ONPOLICY, "fresh", 200)  # every key from onpolicy_dir on moves
    m.run_restore(m.Step("x", {"": 1}, None), "train")
    assert m.choice()["gguf_sha256"] == first
    with pytest.raises(RuntimeError, match="already pre-registers"):
        run_steps(m, "train")
    assert [p.name for p in (root / "choice").iterdir()] == ["run"]
    assert json.loads(next((root / "choice" / "run").glob("manifest.json")).read_text())["complete"]
    # Nothing is pre-registered without the bucket to check it against.
    monkeypatch.setattr(m, "evroc_ok", lambda: False)
    m.CHOICE.unlink()
    with pytest.raises(RuntimeError, match="checked against the bucket"):
        m.run_choice(m.Step("x", {"": 1}, None))
    assert not m.CHOICE.exists()


def test_every_key_below_the_traces_carries_the_held_out_split(tmp_path, monkeypatch):
    """traces-opd#0's follow-up: the split is read from stages/traces.py (this script is
    stdlib only), and another HELD_FRAC or SPLIT_RULE moves vocab, imatrix, PTQ and
    Block-AP (and everything keyed by their names) but not the traces, which gen writes
    whole whatever the split."""
    m = load(tmp_path, monkeypatch)
    T = pytest.importorskip("stages.traces")
    assert m.split_rule() == {"held_frac": T.HELD_FRAC, "rule": T.SPLIT_RULE}
    bench(m)
    for f, v in (("traces_sha", lambda session="measure": f"traces-{session}"), ("vocab_sha", lambda: "toksha"),
                 ("imatrix_sha", lambda: "imsha"), ("gate", lambda name: {"budget_gb": 1.25}),
                 ("type_map", lambda: {"sha256": "mapsha"})):
        monkeypatch.setattr(m, f, v)
    keys = ("traces_dir", "vocab_dir", "imatrix_dir", "ptq_dir", "blockap_dir", "rates_dir", "e2e_dir", "onpolicy_dir")
    names = lambda: {k: getattr(m, k)().name for k in keys}
    now = names()
    for other in ({"held_frac": 0.02, "rule": T.SPLIT_RULE}, {"held_frac": T.HELD_FRAC, "rule": T.SPLIT_RULE + 1}):
        monkeypatch.setattr(m, "split_rule", lambda other=other: other)
        moved = names()
        assert moved["traces_dir"] == now["traces_dir"]
        assert all(moved[k] != now[k] for k in keys[1:]), (other, moved, now)


def test_a_stopped_download_is_started_again_a_bounded_number_of_times(tmp_path, monkeypatch):
    """orchestration#4: no DOWNLOADS_DONE and no hf process: run_download starts setup_vm.sh's
    download again (same revision, appended to the log), up to DOWNLOAD_RESTARTS times."""
    m = load(tmp_path, monkeypatch)
    m.NVME.mkdir(parents=True)
    (m.NVME / "download.log").write_text("HTTP 502\n")
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))  # pgrep: none
    cmds = []

    def restart(step, cmd, env=None):
        cmds.append(cmd)
        granite(m)
        with open(m.NVME / "download.log", "a") as f:
            f.write("DOWNLOADS_DONE\n")

    monkeypatch.setattr(m, "sh", restart)
    m.run_download(m.Step("x", {"": 1}, None))
    assert len(cmds) == 1 and m.download_done()
    assert f"hf download {m.MODEL} --revision main --local-dir {m.HF_DIR} && echo DOWNLOADS_DONE" in cmds[0]
    assert f">> {m.NVME / 'download.log'}" in cmds[0] and "setsid nohup" in cmds[0]

    (m.NVME / "download.log").write_text("HTTP 502\n")
    cmds.clear()
    monkeypatch.setattr(m, "sh", lambda step, cmd, env=None: cmds.append(cmd))  # it keeps failing
    with pytest.raises(RuntimeError, match="stopped 4 times.*Start it by hand"):
        m.run_download(m.Step("x", {"": 1}, None))
    assert len(cmds) == m.DOWNLOAD_RESTARTS == 3


def test_the_base_must_be_the_llama_style_granite_every_stage_assumes(tmp_path, monkeypatch):
    """verdict-quant-training: the architecture guard right after the download."""
    m = load(tmp_path, monkeypatch)
    d = granite(m, tmp_path / "ok")
    assert m.base_problems(d) == []
    assert any("architectures" in p for p in m.base_problems(granite(m, tmp_path / "a",
                                                                     architectures=["GraniteMoeHybridForCausalLM"])))
    assert any("layer_types" in p for p in m.base_problems(granite(m, tmp_path / "l", layer_types=["attention"] * 40)))
    for k, v in (("num_hidden_layers", 64), ("hidden_size", 4096), ("intermediate_size", 12800),
                 ("num_key_value_heads", 4), ("vocab_size", 49159), ("tie_word_embeddings", True)):
        assert any(p.startswith(k) for p in m.base_problems(granite(m, tmp_path / k, **{k: v}))), k
    shapes = granite_shapes(m)
    hybrid = {**{k: v for k, v in shapes.items() if ".mlp." not in k},
              **{f"model.layers.{i}.shared_mlp.input_linear.weight": [16384, 2560] for i in range(40)}}
    probs = m.base_problems(granite(m, tmp_path / "h", shapes=hybrid))
    assert any("120 expected projection weights missing" in p for p in probs)
    assert any("2-D layer weights the stages do not map" in p and "shared_mlp" in p for p in probs)
    no_head = {k: v for k, v in shapes.items() if k != "lm_head.weight"}
    assert any("lm_head" in p for p in m.base_problems(granite(m, tmp_path / "t", shapes=no_head)))
    # a sharded checkpoint: the index names the shards, their headers the shapes
    s = tmp_path / "sharded"
    granite(m, s)
    (s / "model.safetensors").unlink()
    keys = sorted(shapes)
    safetensors_header(s / "model-00001-of-00002.safetensors", {k: shapes[k] for k in keys[::2]})
    safetensors_header(s / "model-00002-of-00002.safetensors", {k: shapes[k] for k in keys[1::2]})
    (s / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: f"model-0000{1 + i % 2}-of-00002.safetensors" for i, k in enumerate(keys)}}))
    assert m.base_problems(s) == []
    # the download step: a wrong base fails it, and is never done
    bench(m)
    assert m.download_done() and m.download_commit() == COMMIT
    granite(m, architectures=["GraniteMoeHybridForCausalLM"])
    assert not m.download_done()
    with pytest.raises(RuntimeError, match="not the base every stage assumes"):
        m.run_download(m.Step("x", {"": 1}, None))


def test_the_all_cases_rescore_grades_a_few_problems_at_a_time(tmp_path, monkeypatch):
    """orchestration#9: items are built per chunk of problems and dropped once graded; a
    chunk the grader cannot write is recorded as grader_error, not the end of the step."""
    m = load(tmp_path, monkeypatch)
    from stages import codebench, sandbox
    job = tmp_path / "job"
    (job / "work" / "code_eval").mkdir(parents=True)
    (job / "work" / "config.effective.json").write_text(json.dumps({"code_eval_dir": str(tmp_path), "lcb_since": "x"}))
    rows = [{"suite": "livecodebench", "id": f"p{i}", "sample": k, "answer": "a", "passed": True}
            for i in range(5) for k in range(2)]
    (job / "work" / "code_eval" / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    probs = [{"id": f"p{i}", "tests": [], "language": "python"} for i in range(5)]
    monkeypatch.setattr(codebench, "load_suite", lambda *a, **k: probs)
    monkeypatch.setattr(codebench, "job", lambda p, answer, code=None: ("python", p["id"], ""))
    seen = []

    def run_many(items, timeout=0):
        seen.append([it[1] for it in items])
        if "p2" in seen[-1]:
            raise OSError(28, "No space left on device")
        return [sandbox.Result(True, "") for _ in items]

    monkeypatch.setattr(sandbox, "run_many", run_many)
    assert m.rescore_main(["--job", str(job), "--name", "r", "--chunk", "2"]) == 0
    assert seen == [["p0", "p0", "p1", "p1"], ["p2", "p2", "p3", "p3"], ["p4", "p4"]]
    out = [json.loads(x) for x in (job / "work" / "code_eval" / "r-allcases.jsonl").read_text().splitlines()]
    assert [(x["id"], x["sample"]) for x in out] == [(r["id"], r["sample"]) for r in rows]  # the reading's order
    assert {x["id"] for x in out if x["reason"] == "grader_error"} == {"p2", "p3"}
    assert all(x["passed"] for x in out if x["id"] in ("p0", "p1", "p4"))


# ---------- the three sessions on a fake shell ----------

def test_three_sessions_end_to_end(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    root = fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    shell = Shell(m)
    monkeypatch.setattr(m, "sh", shell)
    m.run_setup(m.Step("x", {"": 1}, None))
    setup = shell.calls[0][0]
    assert setup.startswith("SKIP_EXTRAS=1 ") and "STUDENT= " in setup and "TEACHER_REVISION=main " in setup
    assert f"UV_EXCLUDE_NEWER={m.UV_EXCLUDE_NEWER} " in setup  # both venvs as PyPI stood then, every session
    assert "SKIP_SUITES= " in setup and "CONSTRAINTS= " in setup  # an empty bucket: fetch, nothing to constrain

    shell.calls.clear()
    run_steps(m, "measure")
    assert kinds(shell.calls) == MEASURE_KINDS
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
    assert smoke["b"]["default_system_text"] == "You are a helpful model"  # reported, not a failure
    assert all(e["TMPDIR"] == str(m.NVME / "tmp") and e["VLLM_CACHE_ROOT"].startswith(str(m.NVME)) for e in evals)
    assert all(e["LOBBOT_EVAL_BASE_REVISION"] == "main" and e["LOBBOT_EVAL_BASE_COMMIT"] == COMMIT for e in evals)
    # the training text's guess, re-based on the original's reading: 8M at 1500 tok/s x the fp8 KV gain
    assert m.GUESS["gen-measure"] == 130 and m.minutes()["gen-measure"] == pytest.approx(8e6 / 3000 / 60 + 5)

    g0 = m.gate("gate0")
    assert g0["pass"] and g0["R0"] == pytest.approx(100 * (37 * 0.5 + 108) / 182) and g0["budget_gb"] == 1.25
    kal = m.k_alloc()
    assert kal["t000"] == 2 and kal["t001"] == 8 and kal["t002"] == 4 and len(kal) == 182
    assert m.gate("preflight")["pass"] and m.gate("gate1")["pass"]
    ref_cfg = json.loads((m.reading_dir(m.READINGS["original-ref"]) / "config.json").read_text())
    assert ref_cfg["code_eval_samples"] == 4 and ref_cfg["lcb_since"] == "2025-01-01" and ref_cfg["vllm_seqs"] == 96
    assert "size_gb_override" not in ref_cfg and "code_eval_ids" not in ref_cfg
    # eval keys resumed answers by content: the original's id, commit and weight sizes
    assert ref_cfg["eval_model_fingerprint"].startswith(f"{m.MODEL}@{COMMIT}:")
    dev_cfg = json.loads((m.reading_dir(m.READINGS["blockap-dev"]) / "config.json").read_text())
    assert len(dev_cfg["code_eval_ids"]) == 128 and dev_cfg["code_eval_seed_offset"] == 50
    assert list(dev_cfg["size_gb_override"].values()) == [1e-6]  # the GGUF's bytes, not the bf16 checkpoint's
    assert dev_cfg["eval_model_fingerprint"] == m.model_of("blockap")[1]  # the export's sha256
    assert "eval_model_fingerprint" in m.PER_READING
    assert (root / "reading-original-ref").exists()  # uploaded by the last step
    assert m.uploaded("measure") and m.vocab_sha() == "toksha"

    shell.calls.clear()  # a new VM for session train: the restore finds everything, then registers the trim
    m.run_restore(m.Step("x", {"": 1}, None), "train")
    assert kinds(shell.calls) == ["stages.vocab register"]

    shell.calls.clear()
    run_steps(m, "train")
    assert kinds(shell.calls) == TRAIN_KINDS
    gen = next(c for c, _ in shell.calls if "gen_command(" in c)
    assert "seqs=384" in gen and "24000000.0" in gen
    rate_e2e, smoke_op, op = (c for c, _ in shell.calls if "stages.qat e2e" in c or "stages.qat onpolicy" in c
                              if not ("--max-minutes 60" in c))
    assert "--freeze-steps 0" in rate_e2e and f"--out {m.rates_dir()}/qat" in rate_e2e
    assert f"--state {m.rates_dir()}/qat/state" in smoke_op and "--until-session 1" in smoke_op
    assert "--rounds 1 --fresh 2 --rollouts 2 --held 2 --max-minutes 5 --rollout-tokens-cap 0" in smoke_op
    assert "--max-minutes 25 --rollout-tokens-cap 8000000 " in op  # each session's caps
    rates = json.loads((m.rates_dir() / "rates.json").read_text())
    assert rates["qat_tok_s"] == 2900.0  # qat's own rate from step 1 on, not tokens / minutes (2731)
    assert rates["qat_peak_gb"] == 151.2 and rates["onpolicy"] == {"rollouts": 4, "steps": 4, "minutes": 0.4,
                                                                   "peak_gb": 163.0, "longest_row": 22100}
    # on-policy guesses: fresh rollouts capped at 8M tokens, training at 25 min, + 10 per session
    qat, roll = 2900.0, 20000
    held = 128 * 2 * 8000
    assert m.minutes()["onpolicy"] == pytest.approx(4 * ((8e6 + held) / roll / 60 + min(16e6 / qat / 60, 25) + 10))
    assert m.minutes()["onpolicy-judge"] == pytest.approx(held / roll / 60 + 10)
    with monkeypatch.context() as mp:  # a cap under the guess: qat holds it within one problem's rollouts
        mp.setitem(m.ONPOLICY, "rollout_tokens_cap", 4_000_000)
        fresh = 4e6 + 4 * m.LCB_TOKENS
        assert m.minutes()["onpolicy"] == pytest.approx(4 * ((fresh + held) / roll / 60 + min(2 * fresh / qat / 60, 25) + 10))
    traces24 = m.traces_file("train").read_text().splitlines()
    assert len(traces24) == 2  # the 8M rows, then gen resumed on the copy
    assert json.loads((m.onpolicy_dir() / "audit.json").read_text())["ok"]
    assert m.gate("parity")["pass"]
    c = m.choice()
    assert c["gguf_sha256"] == hashlib.sha256((m.final_dir() / "model.gguf").read_bytes()).hexdigest()
    assert c["record"]["answers"] == 37 * 2 + 37 * 8 + 108 * 4 and c["record"]["seed_offset"] == 1000
    assert m.minutes()["e2e"] == pytest.approx(min(40e6 / 2900.0 / 60, 60) + 5)

    shell.calls.clear()
    run_steps(m, "read")
    assert kinds(shell.calls) == READ_KINDS
    rec = json.loads((m.reading_dir(m.READINGS["final-record"]) / "config.json").read_text())
    assert rec["code_eval_samples_by_id"] == kal and rec["code_eval_seed_offset"] == 1000
    llama = json.loads((m.reading_dir(m.READINGS["final-llama"]) / "config.json").read_text())
    assert list(llama["eval_candidates"].values()) == [str(m.final_gguf())] and llama["code_eval_budgets"] == []
    out = m.summary()
    assert "Verdict (pre-registered in choice.json, one look): REACHED" in out
    assert "final-record (the verdict): 182 problems, 802 answers" in out
    assert "8k re-scored" in out and "all cases" in out and "R/40" in out and "shares:" in out
    p0 = 100 * sum(passes(i, 0) for i in TEST_IDS) / len(TEST_IDS)  # the first answer of each problem
    assert f"8k re-scored {0.9 * p0:.1f} vs 16k {p0:.1f} on the same answers (the first per problem)" in out
    assert f"  R {100 * (37 * 0.5 + 108) / 182:.1f}  " in out  # the headline stays over every answer
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


# ---------- one B200: session all ----------

class GateShell(Shell):
    """The fake shell with one gate failing: the original's reference reading at R0 50
    (Gate 0), or every answer of a dev screen wrong (Gate 1 after Block-AP, Gate 2 after
    the final model)."""

    def __init__(self, m, stop, gpu=B200):
        super().__init__(m, gpu)
        self.stop = stop

    def passed(self, name, pid, sample):
        if self.stop == "gate0" and name.startswith("original-ref-"):
            return int(pid[1:]) % 2 == 0  # 91 of the 182 problems: R0 50.0
        if name.startswith({"gate1": "blockap-dev-", "gate2": "final-dev-"}.get(self.stop, "-")):
            return False
        return super().passed(name, pid, sample)


def one_b200(m, tmp_path, monkeypatch, shell=None):
    """A B200 VM for session all: the bucket, the problem sets and the base, an nvidia-smi
    that names a B200, the fake shell (its readings' provenance on the B200), and a
    shutdown that is only recorded. Returns (bucket root, shell, shutdowns, restores)."""
    root = fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    monkeypatch.setenv("PATH", f"{fake_smi(tmp_path, 0, 'NVIDIA B200', 183359)}:/usr/bin:/bin")
    shell = shell or Shell(m, B200)
    monkeypatch.setattr(m, "sh", shell)
    downs, restores = [], []
    monkeypatch.setattr(m.os, "system", lambda c: downs.append(c) or 0)
    real = m.run_restore
    monkeypatch.setattr(m, "run_restore", lambda s, session: restores.append(session) or real(s, session))
    return root, shell, downs, restores


def test_all_runs_the_three_sessions_on_one_b200(tmp_path, monkeypatch, capsys):
    """--session all on a fake shell: measure's, train's and read's commands back to back,
    with the setup, the restore and the download once (train's own smoke and rates step
    still runs); every reading on the B200 with its RECORD_SEQS, the original's reading and
    the record agreeing by construction; the upload after every step; the verdict; the
    later readings' guesses re-based on the reference reading; and a re-run redoes nothing."""
    m = load(tmp_path, monkeypatch)
    root, shell, downs, restores = one_b200(m, tmp_path, monkeypatch)
    names = [s.name for s in m.steps("all")]
    assert len(names) == len(set(names)) == 29
    assert names == ([s.name for s in m.steps("measure")][:-1] + [s.name for s in m.steps("train")][3:-1]
                     + [s.name for s in m.steps("read")][2:])
    for once in ("VM setup", "restore from the bucket", "Granite 4.2 3B download", "GPU smoke (a)-(h)",
                 "B200 smoke and rates", "upload, then stop"):
        assert names.count(once) == 1, once

    assert m.main(["--session", "all"]) == 0
    assert restores == ["all"]  # train's and read's restores are not run
    assert kinds(shell.calls) == ["setup", *MEASURE_KINDS, *TRAIN_KINDS, *READ_KINDS]
    evals = [e for c, e in shell.calls if "--only eval" in c or "stages.budget" in c]
    assert len(evals) == 1 + 6 + 5 + 1  # smoke (c), six readings, five re-scorings, S2.2's vLLM smoke
    assert all(e["LOBBOT_VLLM_SEQS"] == "192" and e["LOBBOT_SANDBOX_WORKERS"] == "4" for e in evals)
    for r in m.READINGS.values():  # the B200's RECORD_SEQS in every reading
        d = m.reading_dir(r)
        assert json.loads((d / "config.json").read_text())["vllm_seqs"] == m.RECORD_SEQS["B200"] == 192, r.name
        assert json.loads((d / "work" / "config.effective.json").read_text())["vllm_seqs"] == 192, r.name
    gens = [c for c, _ in shell.calls if "gen_command(" in c]
    assert len(gens) == 2 and all("seqs=384, samples=2" in c for c in gens)  # gen on the B200, both times
    rec, ref = m.READINGS["final-record"], m.READINGS["original-ref"]
    assert m.reading_config(rec, "L40S")["vllm_seqs"] == 192 == m.record_seqs("L40S")  # the reference's, wherever
    assert m.settings_diff(m.reading_dir(rec), m.reading_dir(ref)) == [] and m.off_gpu() == []
    c = m.choice()
    assert c["harness"]["vllm_seqs"] == 192 and c["gpu"] == "NVIDIA B200" and c["harness_sha"] == m.h(m.harness_settings())
    assert all(m.gate(g)["pass"] for g in ("gate0", "preflight", "gate1", "parity", "gate2"))
    assert all(m.record_checks().values())
    out = capsys.readouterr().out
    assert "Verdict (pre-registered in choice.json, one look): REACHED" in out and "STOPPED" not in out
    assert "vLLM with 192 in flight on the NVIDIA B200, every reading on that one GPU" in out
    assert "Finished in" in out and downs == []  # no --shutdown
    assert m.uploaded("all") and (root / "summary-all.txt").exists()
    for slug in ("codebench", "smoke", "reading-original-ref", "reading-original-dev", "traces", "vocab", "imatrix",
                 "ptq", "blockap", "reading-blockap-dev", "rates", "traces24", "e2e", "onpolicy", "final", "choice",
                 "reading-final-dev", "reading-final-record", "reading-final-llama", "rescore"):
        man = next((root / slug).glob("*/manifest.json"))
        assert json.loads(man.read_text())["complete"], slug
    # the readings after the reference re-based on its answers a minute: 1500 tok/s over 900 tokens an answer
    assert m.reference_rate() == pytest.approx(100.0) and "100.0 answers a minute" in m.measured("B200")
    assert m.reading_minutes(m.READINGS["final-dev"], "B200")["eval"] == pytest.approx(256 / 100 + m.READ_BOOT_MIN)
    assert m.reading_minutes(rec, "B200")["eval"] == pytest.approx(802 / 100 + m.READ_BOOT_MIN)
    assert m.minutes("B200")["gen-measure"] == pytest.approx(8e6 / 3000 / 60 + 5)

    shell.calls.clear()  # the same command again: nothing is redone (the setup checks itself, the trim is registered)
    assert m.main(["--session", "all"]) == 0
    assert kinds(shell.calls) == ["setup", "stages.vocab register"] and restores == ["all", "all"]


@pytest.mark.parametrize("stop", ["gate0", "gate1", "gate2"])
def test_a_gate_stop_on_one_b200_is_a_clean_finish_that_powers_off(tmp_path, monkeypatch, capsys, stop):
    """A failed gate in session all skips every step after it but the upload; the run
    exits 0, the summary says where and why it stopped, everything is in the bucket as it
    is on the disk, and --shutdown powers the B200 off under the fingerprint rule."""
    m = load(tmp_path, monkeypatch)
    root, shell, downs, _ = one_b200(m, tmp_path, monkeypatch, GateShell(m, stop))
    assert m.main(["--session", "all", "--shutdown"]) == 0
    ran = {"gate0": MEASURE_KINDS[:14], "gate1": MEASURE_KINDS, "gate2": [*MEASURE_KINDS, *TRAIN_KINDS, *READ_KINDS[:4]]}
    assert kinds(shell.calls) == ["setup", *ran[stop]]
    assert not m.gate(stop)["pass"] and all(m.gate(g)["pass"] for g in ("gate0", "gate1", "gate2")[:int(stop[-1])])
    plan = [s.name for s in m.steps("all")]
    at = plan.index(next(r.title for r in m.READINGS.values() if r.gate == stop))
    out = capsys.readouterr().out
    assert out.count(f"skipped: {stop} did not pass") == len(plan) - at - 2  # every step after it but the upload
    label = f"STOPPED at Gate {stop[-1]}, after the "
    assert label in out and label in (root / "summary-all.txt").read_text() and "Finished in" in out
    assert m.uploaded("all") and all(m.in_sync(a) for a in m.session_arts("all") if a.key() and a.files())
    assert downs == ["sudo shutdown -h +5"] and "Shutting the VM down in 5 minutes" in out


def test_list_all_needs_no_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no nvidia-smi, no evroc
    m = load(tmp_path, monkeypatch)
    assert m.main(["--list", "--session", "all"]) == 0
    out = capsys.readouterr().out
    assert "session all: on the B200" in out and "session measure" not in out
    assert "at 4x the L40S's measured 728 answers in 3h54m, an ESTIMATE" in out and "B200 estimates" in out
    lines = [x for x in out.splitlines() if re.match(r"\s*\d+\. ", x)]
    assert len(lines) == 29 and sum(" VM setup " in x for x in lines) == 1
    assert "original reference reading, 182 x 4 (Gate 0) ~ 1h04m" in out  # 728 answers at 4x the L40S's rate + 6
    assert "record reading, 182 x K_i              ~ 1h16m" in out  # ~880 answers until the K_i file exists
    assert "total ~13h46m (guesses)" in out


def test_a_reading_on_another_gpu_blocks_the_verdict_and_is_refused(tmp_path, monkeypatch):
    """The harness is the GPU of the original's reference reading: a dev screen, the record
    or the llama-server check on another GPU name blocks the verdict, and a machine with
    another GPU is refused before it reads (HORIZON_ANY_GPU=1 reads anyway)."""
    m = load(tmp_path, monkeypatch)
    one_b200(m, tmp_path, monkeypatch)
    assert m.main(["--session", "all"]) == 0
    check = "every reading on the original's GPU"
    assert m.harness_gpu() == "NVIDIA B200" and m.record_checks()[check]
    for name in ("final-dev", "final-record", "final-llama"):
        p = m.reading_dir(m.READINGS[name]) / "work" / "provenance.json"
        prov = json.loads(p.read_text())
        p.write_text(json.dumps({**prov, "gpu": [L40S]}))
        assert m.off_gpu() == [f"{name}: gpu: ['NVIDIA L40S'] vs ['NVIDIA B200']"]
        checks = m.record_checks()
        assert not checks[check] and m.verdict(60.0, 1.0, 1.0, checks).startswith("blocked (")
        assert check in m.verdict(60.0, 1.0, 1.0, checks)
        out = m.summary()
        assert "Verdict (pre-registered in choice.json, one look): BLOCKED" in out
        assert "not on the original reference reading's GPU (gpu: ['NVIDIA L40S'] vs ['NVIDIA B200'])" in out
        p.write_text(json.dumps(prov))
    assert all(m.record_checks().values())

    monkeypatch.setenv("PATH", f"{fake_smi(tmp_path, 0)}:/usr/bin:/bin")  # an L40S
    assert "ran on NVIDIA B200; this GPU is NVIDIA L40S" in m.gpu_mismatch()
    assert any("this GPU is NVIDIA L40S" in x for x in m.session_inputs("read"))
    assert m.session_inputs("train") == []  # training reads nothing
    with pytest.raises(RuntimeError, match="final-dev is not read here"):
        m.run_reading(m.Step("x", {"": 1}, None), m.READINGS["final-dev"], "L40S")
    monkeypatch.setenv("HORIZON_ANY_GPU", "1")
    assert m.gpu_mismatch() is None


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
    calls, seen, cmds = [], [], []

    def trainer(step, cmd, env=None):
        """qat onpolicy's sessions: session k judges round k-1, then trains round k."""
        until = int(re.search(r"--until-session (\d+)", cmd)[1])
        calls.append(until)
        cmds.append(cmd)
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
    # every session passes its caps (qat keeps the last values given, so each call restates them)
    assert all(f"--max-minutes 25 --rollout-tokens-cap {250 * 4 * 8000} " in c for c in cmds)
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


class FallbackShell(Shell):
    """horizon run whose verified map bought nothing over the floor (gain_check's "no
    better": the floor ships, map.json and horizon.json say so), and a Block-AP that found
    fewer windows than asked."""

    def __call__(self, step, cmd, env=None):
        super().__call__(step, cmd, env)
        if "stages.horizon run" in cmd:
            p = arg(cmd, "--out")
            hz = json.loads((p / "horizon.json").read_text())
            gain = {"pred": 0.02, "real": 0.001, "se": 0.002, "ratio": 0.05, "verdict": "no better",
                    "why": "the map's KL 0.39900 is within 2 SE (0.00400) of the floor's 0.40000"}
            fb = {"why": gain["why"], "rejected": {"sha256": "rejected", "bytes": 1_240_000_000}}
            self.write(p / "horizon.json", {**hz, "gain": gain, "fallback": fb, "warnings": []})
        elif "stages.qat block" in cmd:
            self.write(arg(cmd, "--out") / "block.json", {"asked": 2048, "windows": 1500, "warnings": [
                "1500 of the 2048 windows asked: the traces have too few rows long enough"]})


def test_a_map_that_fell_back_to_the_floor_and_block_ap_warnings_are_shown(tmp_path, monkeypatch, capsys):
    """quant-formats#2's follow-up: horizon.json's gain, fallback and warnings are copied into
    preflight.json and printed above the bar ("!" lines), and so are block.json's warnings;
    the summary carries both."""
    m = load(tmp_path, monkeypatch)
    fake_bucket(m, tmp_path, monkeypatch)
    bench(m)
    monkeypatch.setattr(m, "sh", FallbackShell(m))
    run_steps(m, "measure")
    pf = m.gate("preflight")
    assert pf["pass"] and pf["gain"]["verdict"] == "no better" and pf["warnings"] == []
    assert pf["fallback"]["rejected"]["sha256"] == "rejected"
    shown = capsys.readouterr().out
    assert "!!! PTQ MAP FELL BACK TO THE FLOOR: the map's KL 0.39900 is within 2 SE" in shown
    assert "rejected 1.240 GB map" in shown and "0.001 measured, 0.02 predicted, SE 0.002: no better" in shown
    assert "! Block-AP warning: 1500 of the 2048 windows asked" in shown
    out = m.summary()
    assert "  !!! PTQ MAP FELL BACK TO THE FLOOR" in out and "  ! Block-AP warning: 1500 of the 2048" in out
    # a short gain ships the map with a warning line of its own
    lines = m.ptq_lines({"gain": {"verdict": "short", "real": 0.004, "pred": 0.02, "se": 0.001}, "fallback": None,
                         "warnings": ["the map gains 0.00400 of the 0.02000 predicted (20%, under 50%)"]})
    assert lines[0] == "! PTQ map warning: the map gains 0.00400 of the 0.02000 predicted (20%, under 50%)"
    assert not any(x.startswith("!!!") for x in lines)


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


@pytest.mark.parametrize("rc", [0, 3])
def test_server_scripts_end_with_their_last_commands_status_and_stop_the_server(tmp_path, monkeypatch, rc):
    """The smoke's and gen's wrappers under set -e: a stub vLLM that never exits, a stub
    curl that says healthy, and a last command that exits rc. The EXIT trap's KILL of a
    group already gone used to turn rc 0 into exit 1 (step 5's (a+d) on the B200)."""
    from stages.ternary import gen_command
    m = load(tmp_path, monkeypatch)
    stub = tmp_path / "bin"
    stub.mkdir()
    pidfile = tmp_path / "server.pid"
    (stub / "py").write_text(f"#!/bin/bash\necho $$ > {pidfile}\nexec sleep 300\n")
    (stub / "curl").write_text("#!/bin/bash\nexit 0\n")
    (stub / "python").write_text(f"#!/bin/bash\nexit {rc}\n")  # gen's last command
    for f in stub.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "LOBBOT_VLLM_PY": str(stub / "py"), "PATH": f"{stub}:{os.environ['PATH']}"}
    env.pop("LOBBOT_VLLM_ATTN_BACKEND", None)
    scripts = {"smoke": "set -e\n" + m.vllm_script(tmp_path / "hf", 8199, tmp_path / "vllm.log", f"exit {rc}"),
               "gen": gen_command(str(tmp_path / "hf"), "/b", tmp_path / "w" / "traces.jsonl", 8e6, port=8199)}
    for name, script in scripts.items():
        pidfile.unlink(missing_ok=True)
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False)
        assert r.returncode == rc, (name, r.stdout, r.stderr)
        pid = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
