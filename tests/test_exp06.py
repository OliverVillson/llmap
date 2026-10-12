"""exp06's OPD command without a GPU: valid bash, teacher on GPU 0, student and trainer on GPU 1."""

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_opd_command(tmp_path, monkeypatch):
    monkeypatch.setenv("NVME", str(tmp_path))
    monkeypatch.setenv("LOBBOT_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("EXP05_VRAM_GB", "48")
    monkeypatch.setenv("EXP06_LOGS", str(tmp_path / "logs"))
    for m in ("exp05", "exp06"):
        sys.modules.pop(m, None)
    m = importlib.import_module("exp06")
    cmd = m.opd_command()
    assert subprocess.run(["bash", "-n", "-c", cmd]).returncode == 0
    lines = cmd.splitlines()
    assert lines[0] == "set -e"
    teacher = next(l for l in lines if "--served-model-name teacher" in l)
    student = next(l for l in lines if "--served-model-name student" in l)
    trainer = next(l for l in lines if "stages.opd" in l)
    assert teacher.startswith("CUDA_VISIBLE_DEVICES=0 ") and "--quantization fp8" in teacher and teacher.endswith("& teacher=$!")
    assert student.startswith("CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True ") and "--enable-lora" in student
    assert trainer.startswith("CUDA_VISIBLE_DEVICES=1 ") and "--max-tokens 8192" in trainer
    assert lines.index(trainer) > max(i for i, l in enumerate(lines) if l.startswith("until curl"))
    assert sum(s.total for s in m.steps()) < 4 * 60
