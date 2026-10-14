"""scripts/centipede.py without a VM: the whole plan walked with a stand-in for every
stage command (each writes what its real stage writes), so the step order, the paths one
step hands the next, the done checks, the pick, the uploads and the summary are checked;
the agent helpers (pool, round) run for real."""

import importlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TASK = "Write a Python function `double(x)` that returns twice x."
TESTS = "assert double(2) == 4"
GOOD = "prog.py\n```python\ndef double(x):\n    return 2 * x\n```"
FIX1 = {"sft": 0.50, "round": 0.60, "granite-3b": 0.30, "granite-30b": 0.70, "agent": 0.58}
MUGGE = {"sft": 45.0, "round": 55.0, "granite-3b": 20.0, "granite-30b": 70.0, "agent": 45.0}  # of 20 good runs


def bench_tar(path: Path) -> None:
    """A dev-set tarball as scp'd to the VM: bench projects at its top level."""
    src = path.parent / "bench-src"
    for name in ("ledger-svc", "logpipe"):
        write(src / name / "plan.json", {"bench": {"canary": "MUGGE-BENCH-x", "split": "dev"}, "tickets": []})
    subprocess.run(["tar", "czf", str(path), "-C", str(src), "ledger-svc", "logpipe"], check=True)


@pytest.fixture()
def cp(tmp_path, monkeypatch):
    monkeypatch.setenv("NVME", str(tmp_path / "nvme"))
    monkeypatch.setenv("LOBBOT_JOBS", str(tmp_path / "nvme" / "jobs"))
    monkeypatch.setenv("CENTIPEDE_LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    sys.modules.pop("centipede", None)
    m = importlib.import_module("centipede")
    for d in (m.STUDENT_DIR, m.TEACHER_DIR):
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"architectures": ["GraniteForCausalLM"]}))
    (m.STUDENT_DIR / "tokenizer.json").write_text("{}")
    (m.NVME / "download.log").write_text("DOWNLOADS_DONE\n")
    monkeypatch.setattr(m, "setup_missing", lambda: [])
    monkeypatch.setattr(m, "images_ok", lambda: True)
    bench_tar(m.BENCH_TAR)
    return m


def write(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(d if isinstance(d, str) else json.dumps(d))


def fake_stages(m, commands: list[str]):
    """sh(step, cmd): checks cmd is valid bash, then does what the stage it names would."""
    from stages import agent

    def sh(step, cmd, env=None):
        assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0, cmd
        main = cmd.splitlines()[-1].split(" || ")[0]
        commands.append(main)
        a = shlex.split(main)
        if "setup_muggebench.sh" in main:
            assert a[-5:] == list(m.SANDBOX_IMAGES)
        elif a[0] == "tar":
            subprocess.run(a, check=True)
        elif a[:2] == ["cd", "mugge"] and "bench/run.ts" in a:
            assert "--fake" in a and a[a.index("--sandbox") + 1] == "docker"
            bench = Path(a[a.index("--bench") + 1])
            assert sorted(d.name for d in bench.iterdir()) == ["ledger-svc", "logpipe", "spike"]
            out = Path(a[a.index("--out") + 1])
            write(out / "report.json", {"score": 95.0, "tickets": 3})
            write(out / "results.jsonl", "\n".join(json.dumps({"key": k, "solved": k != "logpipe/lp-1"})
                                                    for k in ("ledger-svc/l-1", "logpipe/lp-1", "spike/c-hash")))
        elif "pipeline.py" in a:
            d = Path(a[a.index("--job") + 1])
            cfg = json.loads((d / "config.json").read_text())
            if a[-1] == "data":
                write(d / "data" / "train.jsonl", json.dumps({"messages": [{"role": "user", "content": d.name},
                                                                           {"role": "assistant", "content": "x"}]}) + "\n")
                agent.write_jsonl(d / "work" / "data_inputs.jsonl", [{"input": f"{TASK} ({d.name})", "tests": TESTS}])
                write(d / "data" / "stats.json", {"train": 1, "harness_write_rows": 1, "fix_rows": 0})
            elif cfg["code_eval_suites"] == ["agent-train"]:
                assert not cfg["mugge_bench_dir"]  # the round's answers are no eval
                pool = agent.read_jsonl(Path(cfg["code_eval_dir"]) / "agent-train.jsonl")
                agent.write_jsonl(d / "work" / "code_eval" / "sft.jsonl", [
                    {"suite": "agent-train", "id": p["id"], "sample": 0, "passed": True, "answer": GOOD,
                     "reasoning": "Twice x."} for p in pool])
                assert cfg["code_eval_budgets"] and cfg["code_eval_samples"] == m.ROUND_SAMPLES
            else:
                assert cfg["code_eval_format"] == "harness" and cfg["code_eval_fix"]
                assert cfg["mugge_bench_dir"] == str(m.BENCH) and (m.BENCH / "spike" / "plan.json").exists()
                write(d / "out" / "eval.json", {"candidates": [
                    {"name": n, "size_gb": 1.0, "code": {"mean_pass@1": FIX1[n] - 0.1, "mean_fix@1": FIX1[n],
                                                         "suites": {"c-set": {"pass@1": 0.4, "fix@1": 0.5}}},
                     "mugge_bench": {"score": 1.0, "tickets": 3, "seeds": 3}}  # the runner's, broken tickets in
                    for n in cfg["eval_candidates"]]})
                for n in cfg["eval_candidates"]:  # 20 runs of a good ticket, 5 of one no answer passes
                    good = round(MUGGE[n] / 5)
                    write(d / "work" / "mugge_bench" / n / "results.jsonl", "\n".join(
                        [json.dumps({"key": "spike/c-hash", "solved": k < good}) for k in range(20)]
                        + [json.dumps({"key": "logpipe/lp-1", "solved": False})] * 5))
            write(d / ".done" / a[-1], "{}")
        elif a[:3] == ["python", "-m", "stages.agent"] and a[3] in ("pool", "round"):
            agent.main(a[3:])
        elif a[:3] == ["python", "-m", "stages.agent"] and a[3] == "traces":
            out = Path(a[a.index("--out") + 1])
            write(out, "")
            write(out.with_suffix(".json"), {"rows": 1})
        elif a[:3] == ["python", "-m", "stages.heal"]:
            write(Path(a[a.index("--job") + 1]) / "work" / "dense" / "config.json",
                  {"architectures": ["GraniteForCausalLM"]})
        elif a[:4] == ["python", "-m", "stages.dense", "gguf"]:
            write(Path(a[a.index("--bf16") + 1]), "GGUF")
        elif a[:4] == ["python", "-m", "stages.dense", "imatrix"]:
            write(Path(a[a.index("--job") + 1]) / "work" / "imatrix.gguf", "GGUF")
        elif a[:4] == ["python", "-m", "stages.dense", "calib"]:
            pass
        elif a[:4] == ["python", "-m", "stages.horizon", "run"]:
            out = Path(a[a.index("--out") + 1])
            assert float(a[a.index("--budget-gb") + 1]) == m.BUDGET_GB
            write(out / "ptq" / "q2_k.gguf", "GGUF")
            write(out / "map.json", {"bytes": 1_400_000_000, "types": {}})
        elif a[:4] == ["python", "-m", "stages.qat", "block"]:
            assert Path(a[a.index("--gguf") + 1]).parent == m.C / "ptq" / "ptq"
            write(Path(a[a.index("--out") + 1]) / "block.json", {})
        elif a[:4] == ["python", "-m", "stages.qat", "e2e"]:
            assert a[a.index("--state") + 1] == str(m.C / "blockap" / "state")
            write(Path(a[a.index("--out") + 1]) / "e2e.json", {})
        elif a[:4] == ["python", "-m", "stages.qat", "export"]:
            assert a[a.index("--state") + 1] == str(m.C / "e2e" / "state") and "--no-vllm" in a
            out = Path(a[a.index("--out") + 1])
            write(out / "model.gguf", "G" * 1000)
            write(out / "export.json", {"exact": {"blk.0.ffn_up.weight": True}})
        elif a[:4] == ["python", "-m", "stages.agent", "serve-check"]:
            assert a[a.index("--slots") + 1] == "40" and a[a.index("--kv") + 1] == "q8_0"
            write(Path(a[a.index("--out") + 1]), {"slots": 40, "ctx": 16384, "kv": "q8_0", "requests": 40,
                                                  "completed": 40, "wall_s": 60.0, "tok_s_all": 4000.0,
                                                  "tok_s_per_agent_median": 100.0, "wrote_file": 40, "passed": 30,
                                                  "gpu_mib": {"before": 0, "peak": 31000}})
        else:
            raise AssertionError(f"unexpected command: {main}")

    return sh


def test_the_whole_plan_with_stand_in_stages(cp, monkeypatch):
    from runner import run_plan

    commands, copies = [], []
    monkeypatch.setattr(cp, "sh", fake_stages(cp, commands))
    monkeypatch.setattr(cp, "evroc_ok", lambda: True)
    monkeypatch.setattr(cp, "evroc_copy", lambda src, dst: copies.append((src, dst)) or not src.startswith("bucket://"))
    assert run_plan(cp.steps(), cp.summary, cp.LOGS, ["run"], after_step=cp.upload) == 0
    # the order the work happens in
    kinds = [re.sub(r" --.*", "", c) for c in commands]
    assert "setup_muggebench.sh" in commands[0] and "bench/run.ts" in commands[2]  # first (the VM is set up)
    assert kinds.index("python pipeline.py") < kinds.index("python -m stages.heal")  # base eval, data, then SFT
    assert sum(c.endswith("--only data") for c in commands) == 4
    assert [c for c in kinds if c.startswith("python -m stages.qat")] == [
        "python -m stages.qat block", "python -m stages.qat e2e", "python -m stages.qat export"]
    # the round model won on the Mugge score, so it is the agent, and it was trained on the round's rows too
    pick = json.loads(cp.PICK.read_text())
    assert pick["agent"] == "round" and pick["by"] == "mugge" and pick["dir"] == str(cp.sft_model("round"))
    rows = cp.sft_rows("round").read_text().splitlines()
    assert len(rows) == 4 + 4  # one teacher row per language, and a passing round answer to each task
    assert any("reasoning_content" in r for r in rows)
    assert (cp.sft_model("round") / "tokenizer.json").exists()  # the 3B's own tokenizer files
    # uploads: results after every step, the teacher's rows, the round's tasks and the GGUF once each
    sent = [dst.rsplit("/", 1)[1] for src, dst in copies if not src.startswith("bucket://")]
    assert sent.count("teacher-rows.jsonl") == sent.count("agent-train.jsonl") == sent.count("agent.gguf") == 1
    assert sent.count("muggebench-dev.tar.gz") == 1  # for later VMs, outside the run's folder
    assert sent.count("results.tar") >= len(cp.steps()) - 3
    assert all(max(src, dst, key=lambda x: x.startswith("bucket://")).startswith("bucket://mugge-library/centipede/cp1/")
               for src, dst in copies if dst != cp.BENCH_KEY)
    s = cp.summary()
    assert "SFT + round" in s and "60.0" in s and "40 agents x 16384 tokens" in s and "31000 MiB" in s
    assert " 55.0 " in s and "MuggeBench: 3 tickets" in s and "(not: logpipe/lp-1)" in s
    # a re-run finds every step done
    commands.clear()
    assert run_plan(cp.steps(), cp.summary, cp.LOGS, ["run"]) == 0 and commands == []


def test_a_fresh_vm_restores_the_teachers_rows_and_skips_the_data(cp, monkeypatch):
    got = []

    def copy(src, dst):
        if src.startswith("bucket://"):
            got.append(src)
            Path(dst).write_text("{}\n")
            return True
        return True

    monkeypatch.setattr(cp, "evroc_ok", lambda: True)
    monkeypatch.setattr(cp, "evroc_copy", copy)
    cp.run_restore(None)
    assert got == [cp.bucket("teacher-rows.jsonl"), cp.bucket("agent-train.jsonl")]
    assert all(cp.data_done(lang) for lang in cp.LANGS) and cp.merged()


def test_the_list_and_the_time(cp, capsys):
    assert cp.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == len(cp.steps()) + 1 and "40 agents on one llama-server" in out
    assert sum(s.total for s in cp.steps()) < 13 * 60


def test_no_dev_set_on_the_vm_or_in_the_bucket_stops_the_run_before_it_starts(cp, monkeypatch, capsys):
    cp.BENCH_TAR.unlink()
    monkeypatch.setattr(cp, "evroc_ok", lambda: True)
    monkeypatch.setattr(cp, "evroc_copy", lambda src, dst: False)
    monkeypatch.setattr(cp, "run_plan", lambda *a, **k: pytest.fail("the run started"))
    assert cp.main([]) == 2
    assert "scp ~/Downloads/muggebench-dev.tar.gz evroc:~/" in capsys.readouterr().out
