"""blackhole's runner without a GPU: what runs beside what, the repair adapter reaching the
last score, and the paired interval in the summary."""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(tmp_path, monkeypatch):
    for k, v in {"NVME": tmp_path, "LOBBOT_JOBS": tmp_path / "jobs", "EXP05_VRAM_GB": "46",
                 "BLACKHOLE_LOGS": tmp_path / "logs"}.items():
        monkeypatch.setenv(k, str(v))
    for m in ("exp05", "blackhole"):
        sys.modules.pop(m, None)
    return importlib.import_module("blackhole")


def answers(job: Path, name: str, rates: dict[str, int]):
    d = job / "work" / "code_eval"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"suite": "livecodebench", "id": p, "sample": s, "passed": s < k} for p, k in rates.items() for s in range(4)]
    (d / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")


def test_one_job_at_a_time_all_scored_on_vllm(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    calls = []

    def sh(step, cmd, env=None):
        calls.append((cmd, env or {}))
        if "--only eval" in cmd:
            (Path(cmd.split("--job ")[1].split()[0]) / ".done" / "eval").write_text("{}")
        elif "stages.ternary gen" in cmd:
            m.W.mkdir(parents=True, exist_ok=True)
        elif "dense calib" in cmd:
            (m.e5.BASE / "work" / "calib-chat.txt").write_text("x")
        elif "dense imatrix" in cmd:
            (m.e5.BASE / "work" / "imatrix.gguf").write_text("x")
        elif "dense quantize" in cmd:
            assert "--recipe bh15" in cmd
            (m.QUANT_JOB / ".done" / "quantize").write_text("{}")
        elif "--export" in cmd:
            out = Path(cmd.split("--export ")[1].split()[0])
            out.mkdir(parents=True)
            (out / "config.json").write_text("{}")
        elif "stages.qheal" in cmd:
            m.ADAPTER.mkdir(parents=True)
            (m.ADAPTER / "adapter_config.json").write_text("{}")

    monkeypatch.setattr(m, "sh", sh)
    s = m.Step("x", {"": 1}, None)
    plan = m.steps()[3:-1]
    for step in plan:
        step.run(s)
        assert step.done(), step.name
    kinds = ["eval" if "--only eval" in c else "export" if "--export" in c else "gen" if "ternary gen" in c
             else c.split()[2] for c, _ in calls]
    assert kinds == ["eval", "gen", "stages.dense", "stages.dense", "stages.dense",
                     "export", "eval", "stages.qheal", "export", "eval"]
    assert all(e == {"LOBBOT_VLLM_GPU_UTIL": "0.9"} for c, e in calls if "--only eval" in c)
    exports = [c for c, _ in calls if "--export" in c]
    assert "--adapter" not in exports[0] and f"--adapter {m.ADAPTER}" in exports[1]
    for key in m.JOBS:
        cfg = json.loads((m.JOBS[key] / "config.json").read_text())
        assert cfg["eval_candidates"] == {key: str(m.HF[key])} and cfg["code_eval_samples"] == 4
        assert "eval_lora" not in cfg
    calls.clear()
    for step in plan:
        if not step.done():
            step.run(s)
    assert calls == []  # done work is not redone


def test_summary_gives_the_share_kept_with_a_paired_interval(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    rates = {f"p{i}": (4 if i % 3 else 2) for i in range(60)}  # the original
    answers(m.JOBS["original"], "original", rates)
    answers(m.JOBS["bh15+repair"], "bh15+repair", {p: (k - 1 if i % 10 == 0 else k) for i, (p, k) in enumerate(rates.items())})
    m.QUANT.parent.mkdir(parents=True)
    m.QUANT.write_bytes(b"x" * 1000)
    m.HEAL.mkdir(parents=True)
    m.LORA.write_bytes(b"x" * 100)
    share, lo, hi = m.share_kept(m.per_problem(m.JOBS["bh15+repair"], "bh15+repair"), m.per_problem(m.JOBS["original"], "original"))
    assert 0.95 < share < 0.99 and lo < share < hi <= 1.0
    out = m.summary()
    assert "bh15+repair" in out and "kept" in out and "95% interval" in out


def test_shutdown_only_once_the_results_are_uploaded(tmp_path, monkeypatch):
    bh = load(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(bh.os, "system", calls.append)
    bh.LOGS.mkdir(parents=True)
    bh.shutdown_after()
    assert calls == []  # the VM's disk would be wiped with the results on it
    (bh.LOGS / ".uploaded").write_text("x")
    bh.shutdown_after()
    assert calls == ["sudo shutdown -h +5"]
