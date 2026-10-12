"""blackhole's runner without a GPU: what runs beside what, the repair adapter reaching the
last score, and the paired interval in the summary."""

import importlib
import json
import sys
import threading
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


def test_phases_share_the_gpu_and_the_adapter_is_scored(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    calls, lock = [], threading.Lock()

    def sh(step, cmd, env=None):
        with lock:
            calls.append((cmd, (env or {}).get("LOBBOT_EVAL_PORT")))
        if "--only eval" in cmd:
            (Path(cmd.split("--job ")[1].split()[0]) / ".done" / "eval").write_text("{}")
        elif "dense calib" in cmd:
            (m.e5.BASE / "work" / "calib-chat.txt").write_text("x")
        elif "dense imatrix" in cmd:
            (m.e5.BASE / "work" / "imatrix.gguf").write_text("x")
        elif "dense quantize" in cmd:
            assert "--recipe bh15" in cmd
            (m.QUANT_JOB / ".done" / "quantize").write_text("{}")
        elif "stages.qheal" in cmd:
            m.HEAL.mkdir(parents=True, exist_ok=True)
            m.LORA.write_bytes(b"x")

    monkeypatch.setattr(m, "sh", sh)
    s = m.Step("x", {"": 1}, None)
    m.run_phase1(s)
    assert m.phase1_done()
    side = [c for c, p in calls if "--only eval" not in c]
    assert "stages.ternary gen" in side[0] and "dense calib" in side[1] and "dense quantize" in side[3]
    assert [p for c, p in calls if "--only eval" in c] == ["8091"]
    assert [p for c, p in calls if "dense calib" in c] == ["8092"]
    calls.clear()
    m.run_phase2(s)
    assert m.phase2_done() and any("stages.qheal" in c for c, _ in calls)
    m.steps()[5].run(s)
    cfg = json.loads((m.JOBS["bh15+repair"] / "config.json").read_text())
    assert cfg["eval_lora"] == str(m.LORA) and cfg["code_eval_samples"] == 4
    assert cfg["eval_candidates"] == {"bh15+repair": str(m.QUANT)}
    calls.clear()
    m.run_phase1(s)
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
