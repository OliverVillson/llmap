#!/usr/bin/env python3
"""MuggeBench readings as rollouts for scripts/thinking_report.py: one row per thinking
call, so the report can tell looping thinking from long thinking that finishes.

    python3 scripts/mugge_rollouts.py jobs/<job>/work/mugge_bench/<name>        # writes <name>.jsonl beside it
    python3 scripts/mugge_rollouts.py <bench out> --out bh01.jsonl
    python3 scripts/thinking_report.py bh01.jsonl --ref ...

Reads a bench run's results.jsonl, transcripts.jsonl (run.ts --transcripts, which the eval
stage passes) and proxy-calls.jsonl (the eval stage's proxy, bench_answer: each call's
thinking tokens, how it ended and the end of its thinking). Each thinking call becomes
    {id: "<project>/<ticket>#<call>", sample: seed, tokens, finish, forced, retry,
     answer_cut, passed, reasoning_tail, attempt}
in the eval rows' shape thinking_report reads: tokens is the thinking call's completion
count (a retry or forced answer not in), finish how that call ended, forced why its
thinking ended with no answer ("length", "stop", "no_reasoning", or null), retry how it
was answered then ("nothink": asked again with thinking off, as Centipede does; "force"),
and passed whether this call's answer is the one the ticket ended on and the ticket was
solved (an earlier call's answer failed its checks, or the run would have stopped there).

Calls sent with thinking off (a ticket's calls after one ran out, MUGGE_CUT_RETRY) have
no thinking to report and are left out; the count is printed. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def prompt_sha(text: str) -> str:
    """As stages/eval.py prompt_sha (kept stdlib-only here, so it runs on a laptop)."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut by a stop
            pass
    return out


def convert(run: Path) -> tuple[list[dict], dict]:
    """(rows, counts) for one bench run directory."""
    results = {(r["key"], r["seed"]): r for r in jsonl(run / "results.jsonl")}
    transcripts = jsonl(run / "transcripts.jsonl")
    if not transcripts:
        sys.exit(f"no transcripts in {run}: run the bench with --transcripts (the eval stage does)")
    proxy: dict[tuple[str, int | None], list[dict]] = defaultdict(list)
    for p in jsonl(run / "proxy-calls.jsonl"):
        proxy[(p["prompt_sha"], p.get("seed"))].append(p)
    rows, counts = [], {"thinking_calls": 0, "thinking_off_calls": 0, "unmatched": 0}
    for t in transcripts:
        res = results.get((t["key"], t["seed"]), {})
        calls = t.get("calls", [])
        final = t.get("answerAttempt") or res.get("attempts") or (calls[-1].get("attempt") if calls else None)
        for i, c in enumerate(calls):
            prompt = c.get("prompt")
            prompt = prompt if isinstance(prompt, str) else json.dumps(prompt)
            queue = proxy.get((prompt_sha(prompt), t["seed"]))
            p = queue.pop(0) if queue else None
            thinking = p["thinking"] if p else c.get("thinking")
            if thinking is False:
                counts["thinking_off_calls"] += 1
                continue
            if p is None:  # no proxy line: tokens and finish from the engine's own view
                counts["unmatched"] += 1
            counts["thinking_calls"] += 1
            lost = p.get("lost") if p else (c.get("forced") or None)
            if isinstance(lost, str) and lost.startswith("nothink:"):
                lost = lost.split(":", 1)[1]
            finish = p["finish"] if p else ("length" if c.get("cut") else "stop")
            rows.append({
                "id": f"{t['key']}#{i}", "sample": t["seed"], "attempt": c.get("attempt"),
                "tokens": p["thinking_tokens"] if p else c.get("completionTokens"),
                "finish": finish, "forced": lost, "retry": (p.get("retry") if p else None) if lost else None,
                "answer_cut": finish == "length" and not lost,
                "passed": bool(res.get("solved")) and c.get("attempt") == final,
                **({"reasoning_tail": p["reasoning_tail"]} if p and p.get("reasoning_tail") else {}),
            })
    return rows, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a bench run's --out directory (work/mugge_bench/<name>)")
    ap.add_argument("--out", type=Path, help="default: <run>.jsonl beside the run, named after it")
    a = ap.parse_args(argv)
    rows, counts = convert(a.run)
    out = a.out or a.run.parent / f"{a.run.name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lost = sum(1 for r in rows if r["forced"])
    print(f"{out}: {len(rows)} thinking calls ({lost} ended with no answer, "
          f"{sum(1 for r in rows if r['retry'] == 'nothink')} of them asked again without thinking); "
          f"{counts['thinking_off_calls']} thinking-off calls left out"
          + (f"; {counts['unmatched']} calls had no proxy line (no thinking text)" if counts["unmatched"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
