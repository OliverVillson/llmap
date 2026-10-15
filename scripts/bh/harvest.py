"""Training rows from runs of the engine's own loop (`bun bench/run.ts --transcripts`).

A row is one real worker call: the system and user messages the engine sent, and the answer
that made the ticket pass, in the canonical answer form (files.ts parseFiles, then
renderFiles in owns order). Per (ticket, seed) only the run's last call can be a row: the
engine stops calling once the checks pass, so that call's answer is the passing one, and when
it was a fix call its prompt holds the real failing check that came before it.

Per ticket, from 2-4 seeds:
  write: the shortest passing first-call answer (one row; the write prompt is the same in
         every seed);
  fix:   the shortest passing fix calls with distinct prompts, at most max_fix;
  write-from-fix (write_from_fix): when no seed passed on its first call, the write prompt
         with the shortest passing fix answer, which passes from the start worktree too.
Then every chosen row is checked again in the sandbox (check.py: acceptance and hidden
tests) and dropped if it fails.

Never a row: answers cut at max_tokens or forced after the thinking (the call's cut/forced
flags; without them, an answer that used its whole cap counts as cut), answers that do not
write every owned file (parser drops), fix calls after a failed model call, tickets whose
reference answer fails (broken), and tickets decontam flags. Drops are counted, and a warning
is printed when more than DROP_WARN of all answers drop for length or parsing.

Also here, for round 2: repair_requests turns the student's failed held-out runs into the fix
call the engine would make next (its files as it left them, the check's real output), for
the teacher to answer; repair_rows keeps the teacher's shortest passing repair per ticket
(own-repair rows). The student's own passes go through harvest() as RFT rows.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import check, render

DROP_WARN = 0.20


def read_jsonl(p: str | Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write_jsonl(p: str | Path, rows: list[dict]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(p)


def is_cut(call: dict) -> bool:
    """The engine's cut flag (PR #10 / the per-attempt log: cut, or answer_cut), the bench
    proxy's forced flag, else an answer that used its whole cap."""
    for k in ("cut", "answer_cut", "forced", "truncated"):
        if call.get(k):
            return True
    n, cap = call.get("completionTokens"), call.get("maxTokens")
    return bool(n and cap and n >= cap)


@dataclass
class Stats:
    runs: int = 0
    answers: int = 0  # every model call in the transcripts
    cut: int = 0  # of those, cut at the cap or forced
    parser: int = 0  # of those, missing an owned file after parsing
    solved: int = 0  # (ticket, seed) runs that passed
    candidates: Counter = field(default_factory=Counter)  # passing last calls, by kind
    dropped: Counter = field(default_factory=Counter)  # why a passing last call made no row
    rows: Counter = field(default_factory=Counter)  # rows written, by kind
    tickets_with_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    def drop_share(self) -> float:
        return (self.cut + self.parser) / self.answers if self.answers else 0.0

    def as_dict(self) -> dict:
        return {"runs": self.runs, "answers": self.answers, "cut_or_forced": self.cut, "parser_drops": self.parser,
                "drop_share": round(self.drop_share(), 4), "solved_runs": self.solved,
                "candidates": dict(self.candidates), "dropped": dict(self.dropped), "rows": dict(self.rows),
                "tickets_with_rows": self.tickets_with_rows, "warnings": self.warnings}


def _warn(stats: Stats, msg: str) -> None:
    stats.warnings.append(msg)
    print(f"[harvest] WARNING: {msg}", flush=True)


def load_runs(runs: list[str | Path]) -> list[tuple[dict, list[dict]]]:
    """[(result row, its transcript calls)] over every run dir (results.jsonl + transcripts.jsonl)."""
    out = []
    for d in map(Path, runs):
        results = {(r["key"], r["seed"]): r for r in read_jsonl(d / "results.jsonl")}
        trans = read_jsonl(d / "transcripts.jsonl")
        if results and not trans:
            raise FileNotFoundError(f"{d} has results but no transcripts.jsonl; run the bench with --transcripts")
        for t in trans:
            r = results.get((t["key"], t["seed"]))
            if r is not None:
                out.append((r, t.get("calls") or []))
    return out


def _meta(tk: dict, **kw) -> dict:
    return {"key": tk["key"], "split": tk["split"], "theme": tk["theme"], "language": tk["language"], "role": tk["role"], **kw}


def harvest(bench: str | Path, runs: list[str | Path], *, source: str = "teacher", broken: set[str] = frozenset(),
            excluded: set[str] = frozenset(), max_fix: int = 2, write_from_fix: bool = True, sandbox: str = "local",
            workers: int | None = None, timeout: float = check.TIMEOUT) -> tuple[list[dict], Stats]:
    """(rows, stats) from bench runs over the tickets in bench. Rows are
    {"messages": [system, user, assistant], "meta": {...}}."""
    tks = render.tickets(bench)
    stats = Stats()
    pairs = load_runs(runs)
    stats.runs = len(pairs)
    system = render.system_prompt()

    # every answer, canonicalised by the engine's parser: parser drops and canonical forms
    flat = [(r, i, c) for r, calls in pairs if r["key"] in tks for i, c in enumerate(calls)]
    canon = render.call([{"op": "canonical", "answer": c.get("answer") or "", "owns": tks[r["key"]]["owns"]}
                         for r, _, c in flat])
    canon_of = {(r["key"], r["seed"], i): x for (r, i, _), x in zip(flat, canon)}
    stats.answers = len(flat)
    for (r, i, c), x in zip(flat, canon):
        if is_cut(c):
            stats.cut += 1
        elif x["parsed"]["missing"]:
            stats.parser += 1

    # the start worktree's write prompt per ticket: the drift check and the write-from-fix prompt
    keys = sorted({r["key"] for r, calls in pairs if r["key"] in tks and r.get("solved") and calls})
    writes = render.call([{"op": "write", "project": tks[k]["dir"], "ticket": tks[k]["id"]} for k in keys])
    write_prompt = {k: w["messages"][1]["content"] for k, w in zip(keys, writes)}

    cands: dict[str, list[dict]] = defaultdict(list)
    for r, calls in pairs:
        key = r["key"]
        if key not in tks:
            stats.dropped["not_in_bench"] += 1
            continue
        if not r.get("solved") or not calls:
            continue
        stats.solved += 1
        i = len(calls) - 1
        c, x = calls[i], canon_of[(key, r["seed"], i)]
        prompt = c.get("prompt") or ""
        kind = render.kind_of(prompt)
        stats.candidates[kind] += 1
        why = ("broken_reference" if key in broken else "decontam" if key in excluded else "cut" if is_cut(c)
               else "parser" if x["parsed"]["missing"] or x["parsed"]["extra"]
               else "system_drift" if c.get("system") not in (None, system)
               else "after_model_error" if kind == "fix" and render.failed_command(prompt) == "(model call)"
               else "prompt_drift" if (prompt != write_prompt[key] if kind == "write"
                                       else not prompt.startswith(write_prompt[key].split(render.PACK_END)[0]))
               else "")
        if why:
            stats.dropped[why] += 1
            continue
        cands[key].append({"kind": kind, "prompt": prompt, "content": x["content"], "files": x["parsed"]["files"],
                           "seed": r["seed"], "tokens": c.get("completionTokens") or 0})
    drift = stats.dropped["prompt_drift"] + stats.dropped["system_drift"]
    if drift:
        _warn(stats, f"{drift} passing calls do not match what this engine renders for the same ticket: the run "
                     f"used another engine version; re-run it with this checkout")

    chosen: list[tuple[dict, dict]] = []  # (ticket, row draft)
    for key, cs in sorted(cands.items()):
        tk = tks[key]
        writes_ = sorted((c for c in cs if c["kind"] == "write"), key=lambda c: len(c["content"]))
        fixes = sorted((c for c in cs if c["kind"] == "fix"), key=lambda c: len(c["content"]))
        if writes_:
            chosen.append((tk, {**writes_[0], "kind": "write"}))
        elif fixes and write_from_fix:
            chosen.append((tk, {**fixes[0], "kind": "write-from-fix", "prompt": write_prompt[key]}))
        seen = set()
        for c in fixes:
            if len(seen) < max_fix and c["prompt"] not in seen:
                seen.add(c["prompt"])
                chosen.append((tk, c))

    results = check.check_many([check.Item(tk["dir"], tk["id"], c["files"], key=tk["key"]) for tk, c in chosen],
                               tks, workers=workers, timeout=timeout, kind=sandbox)
    rows, with_rows = [], set()
    for (tk, c), res in zip(chosen, results):
        if not res.ok:
            stats.dropped[f"sandbox_{res.stage}"] += 1
            continue
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": c["prompt"]},
                                  {"role": "assistant", "content": c["content"]}],
                     "meta": _meta(tk, kind=c["kind"], source=source, seed=c["seed"])})
        stats.rows[c["kind"]] += 1
        with_rows.add(tk["key"])
    stats.tickets_with_rows = len(with_rows)
    if stats.drop_share() > DROP_WARN:
        _warn(stats, f"{stats.cut + stats.parser} of {stats.answers} answers ({stats.drop_share():.0%}) dropped: "
                     f"{stats.cut} cut at the cap or forced, {stats.parser} missing an owned file after parsing")
    print(f"[harvest] {source}: {sum(stats.rows.values())} rows ({dict(stats.rows)}) from {stats.solved} passing runs "
          f"of {stats.runs}; {stats.cut} cut, {stats.parser} parser drops of {stats.answers} answers; "
          f"dropped {dict(stats.dropped)}", flush=True)
    return rows, stats


# ---------- round 2: own-repair ----------

def repair_requests(bench: str | Path, runs: list[str | Path], *, broken: set[str] = frozenset(),
                    excluded: set[str] = frozenset(), sandbox: str = "local", workers: int | None = None,
                    timeout: float = check.TIMEOUT) -> tuple[list[dict], dict]:
    """The fix calls the engine would make next on the student's failed runs: [{"id", "key",
    "seed", "messages": [system, user]}], and counts. The student's files are rebuilt by
    replaying its answers over the start worktree (each call's owned files, as applyOutput
    writes them); the acceptance commands run on them; the first failing one becomes the
    fix call, rendered from that very worktree. A run whose files pass acceptance (it failed
    hidden tests only) has no visible failure to repair and is skipped."""
    tks = render.tickets(bench)
    counts: Counter = Counter()
    items = []
    for r, calls in load_runs(runs):
        key = r["key"]
        if r.get("solved") or key not in tks or not calls:
            continue
        if key in broken or key in excluded:
            counts["broken_or_excluded"] += 1
            continue
        owns = tks[key]["owns"]
        files: dict[str, str] = {}
        parsed = render.call([{"op": "parse", "answer": c.get("answer") or "", "owns": owns} for c in calls])
        for p in parsed:
            files.update({k: v for k, v in p["files"].items() if k in owns})
        if not files:
            counts["no_files"] += 1  # nothing written yet: a write call, not a repair
            continue
        items.append((r, check.Item(tks[key]["dir"], tks[key]["id"], files, key=key)))
    res = check.check_many([it for _, it in items], tks, kind=sandbox, workers=workers, timeout=timeout, keep=True)
    reqs = []
    try:
        todo = [(r, it, x) for (r, it), x in zip(items, res) if x.stage == "acceptance"]
        counts["hidden_only"] = sum(1 for x in res if x.ok or x.stage == "hidden")
        rendered = render.call([{"op": "fix", "project": it.project, "ticket": it.ticket, "worktree": x.worktree,
                                 "failed": {"command": x.command, "output": x.output, "timedOut": x.timed_out, "raw": True}}
                                for r, it, x in todo])
        for (r, it, x), rep in zip(todo, rendered):
            reqs.append({"id": f"{r['key']}#s{r['seed']}", "key": r["key"], "seed": r["seed"], "messages": rep["messages"]})
    finally:
        for x in res:
            if x.worktree:
                shutil.rmtree(x.worktree, ignore_errors=True)
    counts["requests"] = len(reqs)
    print(f"[repair] {len(reqs)} fix calls for the teacher; {dict(counts)}", flush=True)
    return reqs, dict(counts)


def repair_rows(bench: str | Path, requests: list[dict], answers: list[dict], *, sandbox: str = "local",
                workers: int | None = None, timeout: float = check.TIMEOUT) -> tuple[list[dict], dict]:
    """Own-repair rows: per student failure, the teacher's shortest answer to its fix call that
    passes acceptance and hidden tests. answers: [{"id", "answers": [{"content", "cut"?,
    "completionTokens"?, "maxTokens"?}]}] (data.answer_requests)."""
    tks = render.tickets(bench)
    by_id = {a["id"]: a.get("answers") or [] for a in answers}
    counts: Counter = Counter()
    flat = []
    for q in requests:
        for a in by_id.get(q["id"], []):
            counts["answers"] += 1
            if is_cut(a):
                counts["cut"] += 1
                continue
            flat.append((q, a))
    canon = render.call([{"op": "canonical", "answer": a.get("content") or "", "owns": tks[q["key"]]["owns"]} for q, a in flat])
    ok = []
    for (q, a), x in zip(flat, canon):
        if x["parsed"]["missing"] or x["parsed"]["extra"]:
            counts["parser"] += 1
        else:
            ok.append((q, x))
    res = check.check_many([check.Item(tks[q["key"]]["dir"], tks[q["key"]]["id"], x["parsed"]["files"], key=q["key"])
                            for q, x in ok], tks, kind=sandbox, workers=workers, timeout=timeout)
    best: dict[str, tuple[dict, dict]] = {}
    for (q, x), r in zip(ok, res):
        if not r.ok:
            counts[f"sandbox_{r.stage}"] += 1
        elif q["id"] not in best or len(x["content"]) < len(best[q["id"]][1]["content"]):
            best[q["id"]] = (q, x)
    rows = [{"messages": [*q["messages"][:2], {"role": "assistant", "content": x["content"]}],
             "meta": _meta(tks[q["key"]], kind="fix", source="own-repair", seed=q["seed"])}
            for q, x in (best[k] for k in sorted(best))]
    counts["rows"] = len(rows)
    if counts["answers"] and (counts["cut"] + counts["parser"]) / counts["answers"] > DROP_WARN:
        print(f"[repair] WARNING: {counts['cut'] + counts['parser']} of {counts['answers']} teacher answers dropped "
              f"({counts['cut']} cut, {counts['parser']} parser)", flush=True)
    print(f"[repair] {len(rows)} own-repair rows; {dict(counts)}", flush=True)
    return rows, dict(counts)
