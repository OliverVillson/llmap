"""exp05's runner without a GPU: the two phases run their work at once on separate
llama-server ports, and the quantized recipes wait for the importance matrix."""

import importlib
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(tmp_path, monkeypatch):
    monkeypatch.setenv("NVME", str(tmp_path))
    monkeypatch.setenv("LOBBOT_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("EXP05_LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("EXP05_VRAM_GB", "48")
    sys.modules.pop("exp05", None)
    return importlib.import_module("exp05")


def fake_sh(m, calls, lock):
    """Records each command with its port and marks the work it would do as done."""
    def sh(step, cmd, env=None):
        with lock:
            calls.append((cmd.split()[2] if cmd.startswith("python -m") else cmd.split()[1], (env or {}).get("LOBBOT_EVAL_PORT")))
        job = Path(cmd.split("--job ")[1].split()[0])
        if "dense calib" in cmd:
            (m.BASE / "work" / "calib-chat.txt").write_text("x")
        elif "dense imatrix" in cmd:
            (m.BASE / "work" / "imatrix.gguf").write_text("x")
        elif "dense quantize" in cmd:
            (job / ".done" / "quantize").write_text("{}")
        elif "--only eval" in cmd:
            (job / ".done" / "eval").write_text("{}")
        elif "stages.budget" in cmd:
            (job / ".done" / "budgets").write_text("{}")
    return sh


def test_phases_share_the_gpu_on_two_ports(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    calls, lock = [], threading.Lock()
    monkeypatch.setattr(m, "sh", fake_sh(m, calls, lock))
    m.run_phase1(m.Step("p1", {"": 1}, m.run_phase1))
    assert m.phase1_done() and not m.phase2_done()
    bf16 = [c for c in calls if c[1] == m.PORTS[0]]
    assert bf16 == [("pipeline.py", "8091"), ("stages.budget", "8091")]
    side = [c[0] for c in calls if c[1] != m.PORTS[0]]
    assert side == ["stages.dense"] * 5  # calib, imatrix, q4, m28, m22
    calls.clear()
    m.run_phase2(m.Step("p2", {"": 1}, m.run_phase2))
    assert m.phase2_done()
    assert sorted(p for _, p in calls) == ["8091"] * 4 + ["8092"] * 2  # q4 then m22; m28
    assert m.RECIPES == ["bf16", "q4", "m28", "m22"] and m.MAX_TOKENS == 16384 and m.BUDGETS == [8192]
