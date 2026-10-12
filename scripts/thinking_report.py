#!/usr/bin/env python3
"""Why a compressed model loses more with thinking on: per-answer breakdown of a
thinking eval against its full model, from the code eval's per-answer rows.

    python3 scripts/thinking_report.py exp02-jobs.tar             # every model in the tar
    python3 scripts/thinking_report.py jobs/ --ref ref-think --suite livecodebench

Reads work/code_eval/<name>.jsonl under every job (a directory or a tar of one,
like the bucket's exp02/exp02-jobs.tar). Per model: pass rate, median thinking
tokens, the share that hit the token cap, the share that needed a forced answer
and how many of those passed, test timeouts, and loops (a repetitive end of
thinking). Against --ref, on the problems both answered:
  - lost: the ref solved it and the model did not, split by how the model failed
    (cap hit, stopped inside its thinking, test timeout, wrong answer)
  - tokens: the model's thinking length over the ref's on problems both solved
Stdlib only, so it runs on a laptop.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tarfile
import zlib
from collections import Counter
from pathlib import Path

# A thinking whose last 1500 chars compress below this ratio is going in circles.
LOOP_RATIO = 0.2


def load(src: Path, suite: str) -> dict[str, dict[str, dict]]:
    """{model: {problem id: row}} for the first sample of each problem in `suite`."""
    found: dict[str, list[dict]] = {}

    def add(name: str, lines) -> None:
        if "/work/code_eval/" not in name or not name.endswith(".jsonl"):
            return
        rows = [json.loads(l) for l in lines if l.strip()]
        found[Path(name).stem] = [r for r in rows if r.get("suite") == suite and r.get("sample", 0) == 0]

    if src.is_dir():
        for p in sorted(src.rglob("*.jsonl")):
            add(p.as_posix(), p.read_text().splitlines())
    else:
        with tarfile.open(src) as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".jsonl"):
                    add(m.name, tar.extractfile(m).read().decode().splitlines())
    return {k: {r["id"]: r for r in v} for k, v in found.items() if v}


def looped(row: dict) -> bool:
    tail = (row.get("reasoning_tail") or row.get("reasoning") or "")[-1500:]
    return len(tail) >= 500 and len(zlib.compress(tail.encode())) / len(tail.encode()) < LOOP_RATIO


def how_failed(row: dict) -> str:
    if row.get("finish") == "error":
        return "error"
    if row.get("finish") == "length" or row.get("forced") == "length":
        return "cap hit"
    if row.get("forced") == "stop":
        return "stopped in thinking"
    if row.get("reason") == "timeout":
        return "test timeout"
    if row.get("reason") == "compile_error":
        return "did not run"
    return "wrong answer"


def stats(rows: dict[str, dict]) -> dict:
    rs = list(rows.values())
    n = len(rs)
    share = lambda f: round(100 * sum(map(f, rs)) / n, 1)
    forced = [r for r in rs if r.get("forced")]
    toks = [r["tokens"] for r in rs if r.get("tokens")]
    return {
        "n": n,
        "pass%": share(lambda r: r.get("passed")),
        "tokens_median": round(statistics.median(toks)) if toks else None,
        "cap%": share(lambda r: r.get("finish") == "length"),
        "forced%": share(lambda r: bool(r.get("forced"))),
        "forced_passed": f"{sum(bool(r.get('passed')) for r in forced)}/{len(forced)}",
        "timeout%": share(lambda r: r.get("reason") == "timeout"),
        "loop%": share(looped),
    }


def compare(model: dict[str, dict], ref: dict[str, dict]) -> dict:
    both = model.keys() & ref.keys()
    lost = [model[i] for i in both if ref[i].get("passed") and not model[i].get("passed")]
    gained = sum(1 for i in both if model[i].get("passed") and not ref[i].get("passed"))
    ratios = [model[i]["tokens"] / ref[i]["tokens"] for i in both
              if model[i].get("passed") and ref[i].get("passed") and model[i].get("tokens") and ref[i].get("tokens")]
    return {
        "shared": len(both),
        "lost": len(lost),
        "gained": gained,
        "lost_by": dict(Counter(map(how_failed, lost)).most_common()),
        "lost_looped": sum(map(looped, lost)),
        "tokens_vs_ref": round(statistics.median(ratios), 2) if ratios else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="a jobs directory or a tar of one")
    ap.add_argument("--ref", default="ref-think", help="the full model's eval name (default ref-think)")
    ap.add_argument("--suite", default="livecodebench")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    models = load(a.src, a.suite)
    if not models:
        print(f"no {a.suite} rows under {a.src}", file=sys.stderr)
        return 1
    ref = models.get(a.ref)
    out = {name: {**stats(rows), **({"vs_ref": compare(rows, ref)} if ref and name != a.ref else {})}
           for name, rows in sorted(models.items())}
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    cols = ["n", "pass%", "tokens_median", "cap%", "forced%", "forced_passed", "timeout%", "loop%"]
    w = max(map(len, out)) + 2
    print("model".ljust(w) + "  ".join(f"{c:>13}" for c in cols))
    for name, s in out.items():
        print(name.ljust(w) + "  ".join(f"{str(s[c]):>13}" for c in cols))
    if ref is None:
        print(f"\n(no '{a.ref}' eval found, so no comparison; pass --ref)")
    for name, s in out.items():
        v = s.get("vs_ref")
        if v:
            print(f"\n{name} vs {a.ref} on {v['shared']} shared problems: lost {v['lost']}, gained {v['gained']}, "
                  f"thinking {v['tokens_vs_ref']}x the ref's on problems both solved")
            print(f"  lost by: {v['lost_by']}; {v['lost_looped']} of the lost were looping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
