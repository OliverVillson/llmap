"""The one place the train harness touches bhdata, BlackHole's kit library (stdlib Python, in
the BlackHole project's folder at blackhole/data/bhdata). Everything bhdata-shaped goes through
here, so when its interface moves only this file changes.

bhdata was not reachable from the thread that wrote this file; it is written against the
interface BlackHole documented (15 Oct 2026):

  python3 -m bhdata.llmap export --out <dir>
      <dir>/train-bench/      MuggeBench-format projects, ~6K tickets (plan.json, project/,
                              solutions/, hidden/; bench/load.ts reads them)
      <dir>/heldout-bench/    the same, ~2K tickets, split from train-bench by kit
      <dir>/extra_rows.jsonl  free mutation rows (fix, test, review), {"messages": [...], "meta"?}
  bhdata.decontam             its own MuggeBench decontamination
  bhdata.wire.REVIEW_SYSTEM   the proposed lead-review system prompt (BH0.2; review rows are
                              not worker calls, so BH0.1 sets them aside)

Set BHDATA to the bhdata package dir (the folder holding llmap.py) when it is not in one of
CANDIDATES. Extra export arguments go through BHDATA_EXPORT_ARGS (split like a shell would).
"""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CANDIDATES = (
    "/mnt/project-files/blackhole/data/bhdata",
    "~/blackhole/data/bhdata",
    "~/bhdata",
)


@dataclass
class Export:
    train_bench: Path
    heldout_bench: Path
    extra_rows: Path


def locate() -> Path | None:
    """The bhdata package dir, or None."""
    for c in ([os.environ["BHDATA"]] if os.environ.get("BHDATA") else []) + list(CANDIDATES):
        p = Path(c).expanduser()
        if (p / "__init__.py").exists() or (p / "llmap.py").exists():
            return p.resolve()
    return None


def layout(out: str | Path) -> Export:
    out = Path(out)
    return Export(out / "train-bench", out / "heldout-bench", out / "extra_rows.jsonl")


def export(out: str | Path, *, args: list[str] | None = None) -> Export:
    """Runs bhdata's export into out; returns where its three outputs are. Raises with the
    reason when bhdata is not here or its output is not the documented shape."""
    pkg = locate()
    if pkg is None:
        raise FileNotFoundError("bhdata not found: set BHDATA to its package dir (BlackHole's blackhole/data/bhdata)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    extra = args if args is not None else shlex.split(os.environ.get("BHDATA_EXPORT_ARGS", ""))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(pkg.parent), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    cmd = [sys.executable, "-m", f"{pkg.name}.llmap", "export", "--out", str(out), *extra]
    print(f"[bhdata] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    e = layout(out)
    missing = [str(p) for p in (e.train_bench, e.heldout_bench) if not p.is_dir()]
    if missing:
        raise RuntimeError(f"bhdata export wrote no {', '.join(missing)}; its interface may have changed (see bhdata_io.py)")
    if not e.extra_rows.exists():
        print(f"[bhdata] no {e.extra_rows}: no free rows this time", flush=True)
    return e


def decontam_module():
    """bhdata.decontam when bhdata is here, else None."""
    pkg = locate()
    if pkg is None:
        return None
    sys.path.insert(0, str(pkg.parent))
    try:
        return importlib.import_module(f"{pkg.name}.decontam")
    except ImportError:
        return None


def bhdata_decontam(rows: list[dict]) -> tuple[list[dict], int]:
    """rows bhdata's own decontam keeps, and how many it dropped. Its function name is not
    documented; the first of filter_rows, clean, drop that exists is used, else nothing."""
    mod = decontam_module()
    fn = next((getattr(mod, n) for n in ("filter_rows", "clean", "drop") if mod and hasattr(mod, n)), None)
    if fn is None:
        return rows, 0
    kept = fn(rows)
    kept = kept[0] if isinstance(kept, tuple) else kept
    return list(kept), len(rows) - len(kept)


def review_system() -> str | None:
    """bhdata.wire.REVIEW_SYSTEM, the proposed lead-review prompt, when bhdata is here."""
    pkg = locate()
    if pkg is None:
        return None
    sys.path.insert(0, str(pkg.parent))
    try:
        return getattr(importlib.import_module(f"{pkg.name}.wire"), "REVIEW_SYSTEM", None)
    except ImportError:
        return None
