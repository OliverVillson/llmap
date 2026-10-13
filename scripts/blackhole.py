#!/usr/bin/env python3
"""Blackhole is retired; scripts/event_horizon.py replaces it. What is left here is the
paired share-kept interval its summary used, which Event Horizon's summary still reports.

Why it was retired (plan: /mnt/project-files/plan/event-horizon-qat.md, audit:
/mnt/project-files/plan/blackhole-audit.md): its route was llama.cpp's imatrix formats
(~1.45 GB) with a ~0.05 GB LoRA repair on top, judged against a 1.5 GB limit and 95% of
the original's score. The goal is now LiveCodeBench points per GB of the shipped file,
at least 40, and that file is the whole artifact: at ~1.5 GB it would need 58-60 points
where the original reads ~52, so no adapter or 1.5 GB format can reach it. Its runner
also uploaded only at the very end, marked the upload in the home dir (which outlives
the NVMe it described), and judged a share kept rather than a lower bound on the
reading itself; Event Horizon fixes those.

    python3 scripts/event_horizon.py --list    # the three sessions and their steps
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def per_problem(d: Path, name: str) -> dict[str, float]:
    """Each LiveCodeBench problem's pass rate over its answers, from a job's graded rows."""
    p = d / "work" / "code_eval" / f"{name}.jsonl"
    if not p.exists():
        return {}
    got: dict[str, list[bool]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("suite") == "livecodebench":
                got.setdefault(r["id"], []).append(bool(r.get("passed")))
    return {k: sum(v) / len(v) for k, v in got.items()}


def share_kept(model: dict[str, float], ref: dict[str, float], n: int = 4000, seed: int = 0
               ) -> tuple[float, float, float] | None:
    """(share of the reference's score kept, 95% interval low, high) over the problems both
    answered, by resampling problems (paired: each draw keeps a problem's two scores together)."""
    ids = sorted(set(model) & set(ref))
    if not ids or not sum(ref[i] for i in ids):
        return None
    share = sum(model[i] for i in ids) / sum(ref[i] for i in ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        pick = [rng.choice(ids) for _ in ids]
        r = sum(ref[i] for i in pick)
        if r:
            draws.append(sum(model[i] for i in pick) / r)
    draws.sort()
    return share, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def main() -> int:
    print("blackhole is retired: its 1.5 GB route cannot reach 40 points per GB. Use scripts/event_horizon.py "
          "(--list shows its sessions; --session measure starts on the L40S).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
