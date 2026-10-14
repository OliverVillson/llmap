"""blackhole is retired: it imports without a GPU, keeps only the paired share-kept
interval Event Horizon's summary uses, and points its command at event_horizon.py."""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no nvidia-smi: importing must not need one
    sys.modules.pop("blackhole", None)
    return importlib.import_module("blackhole")


def answers(job: Path, name: str, rates: dict[str, int]):
    d = job / "work" / "code_eval"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"suite": "livecodebench", "id": p, "sample": s, "passed": s < k} for p, k in rates.items() for s in range(4)]
    rows.append({"suite": "codecontests", "id": "other", "sample": 0, "passed": True})
    (d / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")


def test_share_kept_is_a_paired_interval(tmp_path, monkeypatch):
    bh = load(tmp_path, monkeypatch)
    rates = {f"p{i}": (4 if i % 3 else 2) for i in range(60)}
    answers(tmp_path / "ref", "ref", rates)
    answers(tmp_path / "new", "new", {p: (k - 1 if i % 10 == 0 else k) for i, (p, k) in enumerate(rates.items())})
    ref, new = bh.per_problem(tmp_path / "ref", "ref"), bh.per_problem(tmp_path / "new", "new")
    assert len(ref) == 60 and "other" not in ref and ref["p0"] == 0.5
    share, lo, hi = bh.share_kept(new, ref)
    assert 0.95 < share < 0.99 and lo < share < hi <= 1.0
    assert bh.share_kept(new, ref) == (share, lo, hi)  # seeded
    assert bh.share_kept({}, ref) is None and bh.per_problem(tmp_path / "none", "x") == {}


def test_the_old_route_is_gone_and_the_command_points_on(tmp_path, monkeypatch, capsys):
    bh = load(tmp_path, monkeypatch)
    for gone in ("LIMIT_GB", "LORA", "HEAL", "steps", "shutdown_after", "e5", "LOGS"):
        assert not hasattr(bh, gone), gone
    assert "event_horizon.py" in bh.__doc__ and "retired" in bh.__doc__
    assert bh.main() == 2
    assert "scripts/event_horizon.py" in capsys.readouterr().err
