"""MuggeBench runs for the train harness: the command for mugge/bench/run.ts (the engine's
own write -> check -> fix loop), the shards a long run is cut into so it resumes, and the
reading of what it wrote: the score with its interval and verdict, answers per minute,
and the share of calls that came back without a file.

Every run points bench/run.ts at the recording proxy (proxy.py), which owns sampling; the
command still passes the same settings so its own report says what was used.
"""

from __future__ import annotations

import math
import os
import shlex
from collections import defaultdict
from pathlib import Path

from bh.exec import Cmd, load, read_jsonl

MUGGE = Path(__file__).resolve().parents[2] / "mugge"


def projects(bench: Path) -> list[Path]:
    """The bench projects under bench (folders with a plan.json), sorted."""
    return sorted(p for p in Path(bench).iterdir() if (p / "plan.json").exists()) if Path(bench).exists() else []


def shard(bench: Path, out: Path, size: int) -> list[Path]:
    """bench's projects in shards of `size`, each a folder of symlinks to them, so a long
    run is many short ones and a stopped run resumes at the first shard without a report."""
    ps = projects(bench)
    dirs = []
    for i in range(0, len(ps), max(1, size)):
        d = out / f"shard-{i // max(1, size):03d}"
        d.mkdir(parents=True, exist_ok=True)
        for p in ps[i:i + size]:
            link = d / p.name
            if not link.exists():
                os.symlink(p.resolve(), link)
        dirs.append(d)
    return dirs


def command(cfg, bench: Path, out: Path, url: str, name: str, *, seeds: int, concurrency: int,
            transcripts: bool = False, tickets: int = 0, split: str = "", fake: bool = False) -> Cmd:
    """`bun bench/run.ts` with thinking off, a finite top_k and the per-call cap; no Bun
    fetch cap (MUGGE_TIMEOUT_S an hour: a call may wait for a slot, then prefill 10K)."""
    args = ["--bench", str(bench), "--out", str(out), "--url", url, "--model", name, "--name", name,
            "--seeds", str(seeds), "--concurrency", str(concurrency), "--sandbox", cfg.sandbox,
            "--thinking", "on" if cfg.thinking else "off", "--temperature", str(cfg.temperature),
            "--top-p", str(cfg.top_p), "--top-k", str(cfg.top_k), "--max-tokens", str(cfg.max_call_tokens)]
    if transcripts:
        args.append("--transcripts")
    if tickets:
        args += ["--tickets", str(tickets)]
    if split:
        args += ["--split", split]
    if fake:
        args.append("--fake")
    sh = f"cd {shlex.quote(str(MUGGE))} && bun bench/run.ts " + " ".join(shlex.quote(a) for a in args)
    return Cmd(sh, "bench", {"bench": str(bench), "out": str(out), "url": url, "name": name, "seeds": seeds,
                             "transcripts": transcripts, "tickets": tickets, "fake": fake},
               env={"MUGGE_TIMEOUT_S": "3600"})


# ---------- reading a run ----------

def rows(out: Path) -> list[dict]:
    return read_jsonl(Path(out) / "results.jsonl")


def stats(out: Path, broken: set[str] = frozenset()) -> dict:
    """What a run did: tickets, runs, solved share (tickets a reference answer fails are
    left out), calls, calls per minute of wall time, and calls without a file."""
    rs = [r for r in rows(out) if r.get("key") not in broken]
    rep = load(Path(out) / "report.json") or {}
    calls = sum(int(r.get("calls") or 0) for r in rs)
    empty = sum(int(r.get("emptyAnswers") or 0) for r in rs)
    wall_ms = rep.get("wallMs") or 0
    point, low, high = interval(rs)
    return {"tickets": len({r["key"] for r in rs}), "runs": len(rs), "solved": sum(bool(r.get("solved")) for r in rs),
            "score": point, "low": low, "high": high, "calls": calls, "empty_answers": empty,
            "empty_share": round(empty / calls, 4) if calls else None,
            "answers_per_min": round(calls / (wall_ms / 60000), 1) if wall_ms else None,
            "wall_min": round(wall_ms / 60000, 2) if wall_ms else None,
            "completion_tokens": sum(int(r.get("completionTokens") or 0) for r in rs)}


def interval(rs: list[dict]) -> tuple[float | None, float | None, float | None]:
    """Score in % (mean over tickets of each ticket's solved share across its seeds) and a
    95% interval over tickets (normal approximation; 57 tickets give about +-12 points).
    Tickets, not runs, are the sample: seeds of one ticket are not independent."""
    per = defaultdict(list)
    for r in rs:
        per[r["key"]].append(1.0 if r.get("solved") else 0.0)
    xs = [sum(v) / len(v) for v in per.values()]
    if not xs:
        return None, None, None
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 1.0
    return round(100 * m, 1), round(100 * max(0.0, m - half), 1), round(100 * min(1.0, m + half), 1)


def verdict(low: float | None, high: float | None, bar: float) -> str:
    """MEETS when the whole interval is at or over the bar, MISSES when it is all under,
    TOO CLOSE otherwise (muggebench.sh's words). Whether the bar is the point or the low
    end is Oliver's open decision; both are printed."""
    if low is None or high is None:
        return "NO SCORE"
    if low >= bar:
        return "MEETS"
    if high < bar:
        return "MISSES"
    return "TOO CLOSE"


def broken_tickets(preflight_out: Path) -> set[str]:
    """Tickets that fail with their own reference answer: no model can solve them."""
    return {r["key"] for r in rows(preflight_out) if not r.get("solved")}
