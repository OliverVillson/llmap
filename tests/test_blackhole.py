"""blackhole's runner without a GPU: the steps' commands, in order, and the shared GPU in phase 1."""

import importlib
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(tmp_path, monkeypatch):
    for k, v in {"NVME": tmp_path, "LOBBOT_JOBS": tmp_path / "jobs", "EXP05_VRAM_GB": "180",
                 "BLACKHOLE_LOGS": tmp_path / "logs"}.items():
        monkeypatch.setenv(k, str(v))
    for m in ("exp05", "blackhole"):
        sys.modules.pop(m, None)
    return importlib.import_module("blackhole")


def test_steps_run_in_order_and_resume(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    assert subprocess.run(["bash", "-n", "-c", m.gen_command()]).returncode == 0
    calls, lock = [], threading.Lock()

    def sh(step, cmd, env=None):
        with lock:
            calls.append((cmd, env))
        if "--only eval" in cmd:
            job = Path(cmd.split("--job ")[1].split()[0])
            (job / ".done" / "eval").write_text("{}")
        elif "ternary kl" in cmd:
            (m.W / "kl-snap.json").write_text('{"kl": 1.0}')
        elif "ternary block" in cmd:
            (m.W / "block").mkdir(parents=True)
            (m.W / "block" / "block.json").write_text("{}")
        elif "ternary e2e" in cmd:
            (m.W / "e2e").mkdir(parents=True)
            (m.W / "e2e" / "e2e.json").write_text('{"kl": [{"step": 0, "kl": 0.5}], "tokens": 1, "size_gb": 0.9, "size_gb_vocab24k": 0.73}')

    monkeypatch.setattr(m, "sh", sh)
    s = m.Step("x", {"": 1}, None)
    m.run_phase1(s)
    assert m.phase1_done()
    base = next(e for c, e in calls if "--only eval" in c)
    assert base == {"LOBBOT_VLLM_GPU_UTIL": "0.45"}  # shares the GPU with the training-text server
    assert any("stages.ternary gen" in c for c, _ in calls)
    for run in (m.run_snap, m.run_block, m.run_e2e, m.run_score):
        run(s)
    order = [c.split()[3] if "stages.ternary" in c else "eval" for c, _ in calls if "set -e" not in c]
    assert order == ["eval", "kl", "block", "e2e", "eval"]
    e2e = next(c for c, _ in calls if "ternary e2e" in c)
    assert "--student " + str(m.W / "block" / "latent") in e2e and "--length 8192 --batch 2" in e2e
    assert str(m.W / "e2e" / "snapped") in (m.FINAL_JOB / "config.json").read_text()
    calls.clear()
    m.run_phase1(s)
    assert calls == []  # done work is not redone
    assert "10.0x smaller" in m.summary()
