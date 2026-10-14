#!/usr/bin/env python3
"""Why a compressed model loses more with thinking on: per-answer breakdown of a
thinking eval against its full model, from the code eval's per-answer rows or from
a file of rollouts.

    python3 scripts/thinking_report.py exp02-jobs.tar             # every model in the tar
    python3 scripts/thinking_report.py jobs/ --ref ref-think --suite livecodebench
    python3 scripts/thinking_report.py jobs/eh-original jobs/eh-q2 --ref original --json --out q2-thinking.json
    python3 scripts/thinking_report.py held-rollouts.jsonl --tokenizer W/model --json

Reads work/code_eval/<name>.jsonl under every job (a directory or a tar of one,
like the bucket's exp02/exp02-jobs.tar). A reading that stopped before its tests ran
has only <name>.partial.jsonl: that is read instead, with no pass rate. A .jsonl
given by name (or .jsonl.gz) is read as it is, so a file of rollouts works too: rows
with token ids and a finish, {id, tokens: [ids], finish, forced, passed} as
stages/traces.py writes them, named after the file. Every sample of a problem counts.

Per model: pass rate, median thinking tokens, and the shares of answers that
  - hit the token cap (cap%),
  - needed a forced answer because the thinking never closed (forced%), and how many
    of those passed,
  - were cut by the cap after the thinking closed (answer_cut%),
  - ran out of test time (timeout%: the whole run; case_timeout%: one case),
  - loop: a thinking whose end goes in circles (loop%; loop_checked counts the answers
    whose thinking was there to check).
Against --ref, answer by answer (same problem, same sample):
  - lost: the ref solved it and the model did not, split by how the model failed
    (cap hit, answer cut, stopped inside its thinking, test timeout, wrong answer)
  - tokens: the model's thinking length over the ref's on answers both got right
--json prints the same as JSON and --out writes it to a file, so a run can keep it
beside each reading. Stdlib only, so it runs on a laptop; --tokenizer (for the loop
check on rollouts, see looped) needs transformers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import tarfile
import zlib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A thinking whose last 1500 chars compress below this ratio is going in circles.
LOOP_RATIO = 0.2
# Token ids, with no tokenizer to turn them back into text: a window of about 1500
# chars of thinking, its ids packed 3 bytes each. Measured on a 14k byte-level BPE,
# this cut agreed with the text one on 99% of 800 windows (prose, code, made-up loops)
# and no prose or code window went below 0.32; ids spread over a 262k vocab moved the
# best cut to 0.25.
LOOP_IDS = 400
LOOP_RATIO_IDS = 0.22
# What closes a thinking, in either chat format (as stages/opd.py looks for them).
CLOSE_TAGS = ("</think>", "<channel|>")


# ---------- reading ----------

def _key(r: dict, seen: Counter) -> tuple[str, int]:
    """(problem id, sample): eval rows carry their sample; rollouts are numbered in
    file order within their problem."""
    pid = str(r.get("id"))
    if "sample" in r:
        return pid, int(r["sample"])
    seen[pid] += 1
    return pid, seen[pid] - 1


def _rows(lines, suite: str) -> dict[tuple[str, int], dict]:
    """{(id, sample): row} for `suite` (rollout rows have no suite and are all kept).
    A partial file can hold an answer twice (an errored one, asked again): the last
    one that did not error wins."""
    out: dict[tuple[str, int], dict] = {}
    seen: Counter = Counter()
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line cut by a stop
            continue
        if r.get("suite", suite) != suite:
            continue
        k = _key(r, seen)
        if k not in out or not errored(r):
            out[k] = r
    return out


def load(src: Path | list[Path], suite: str) -> dict[str, dict[tuple[str, int], dict]]:
    """{model: {(problem id, sample): row}} for every answer in `suite`, from jobs
    directories, tars of them and .jsonl files. A model's graded <name>.jsonl wins over
    its <name>.partial.jsonl."""
    found: dict[str, tuple[bool, dict]] = {}  # model: (from a partial file, rows)

    def add(name: str, lines, given: bool = False) -> None:
        if not name.endswith(".jsonl") or not (given or "/work/code_eval/" in name):
            return
        stem = Path(name).name[: -len(".jsonl")]
        partial = stem.endswith(".partial")
        model = stem[: -len(".partial")] if partial else stem
        if partial and model in found and not found[model][0]:
            return
        rows = _rows(lines, suite)
        if rows:
            found[model] = (partial, rows)

    for s in [src] if isinstance(src, (str, Path)) else src:
        s = Path(s)
        if s.is_dir():
            for p in sorted(s.rglob("*.jsonl")):
                add(p.as_posix(), p.read_text().splitlines())
        elif s.name.endswith(".jsonl"):
            add(s.as_posix(), s.read_text().splitlines(), given=True)
        elif s.name.endswith(".jsonl.gz"):  # rollouts as uploaded
            add(s.as_posix()[:-3], gzip.decompress(s.read_bytes()).decode().splitlines(), given=True)
        else:
            with tarfile.open(s) as tar:
                for m in tar:
                    if m.isfile() and m.name.endswith(".jsonl"):
                        add(m.name, tar.extractfile(m).read().decode().splitlines())
    return {k: rows for k, (_, rows) in found.items()}


# ---------- one answer ----------

def errored(row: dict) -> bool:
    return row.get("finish") == "error" or "error" in row


def n_tokens(row: dict) -> int | None:
    """Completion tokens: an eval row's count (thinking in, a forced answer not), or
    the length of a rollout's ids (the forced answer in)."""
    t = row.get("tokens")
    return len(t) if isinstance(t, list) else t


def answer_cut(row: dict) -> bool:
    """The cap cut the answer after the thinking closed. The eval marks it; older eval
    rows and rollouts show it as a cap hit that needed no forced answer."""
    if "answer_cut" in row:
        return bool(row["answer_cut"])
    thought = isinstance(row.get("tokens"), list) or (row.get("reasoning_chars") or 0) > 0
    return row.get("finish") == "length" and not row.get("forced") and thought


def thinking(row: dict, tok=None) -> str | None:
    """The thinking text, or None when the row does not hold it. Eval rows keep its
    end (reasoning_tail) on forced answers, or all of it (reasoning) when the eval kept
    reasoning; rollout ids are decoded with tok: the text before the closing tag,
    without the forced answer's early-stop line."""
    if "reasoning_tail" in row or "reasoning" in row:
        return row.get("reasoning_tail") or row.get("reasoning") or ""
    ids = row.get("tokens")
    if tok is None or not isinstance(ids, list):
        return None
    text = tok.decode(ids, skip_special_tokens=False)
    cut = [i for i in (text.find(t) for t in CLOSE_TAGS) if i >= 0]
    text = text[:min(cut)] if cut else text
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from stages.eval import EARLY_STOP
    stop = EARLY_STOP.strip()
    return (text[:text.rfind(stop)] if stop in text else text).strip()  # stripped, as the eval keeps it


def looped_ids(ids: list[int]) -> bool:
    """Whether any LOOP_IDS-id window of the row (stepping half a window) compresses
    below LOOP_RATIO_IDS. Without a tokenizer the end of the thinking is not known, so
    this looks at the whole row: a loop the model got out of counts too."""
    packed = b"".join((t & 0xFFFFFF).to_bytes(3, "little") for t in ids)
    step = LOOP_IDS // 2
    for i in range(0, max(1, len(ids) - step), step):
        if min(LOOP_IDS, len(ids) - i) < LOOP_IDS // 3:  # too short to tell, as 500 of the text's 1500 chars
            continue
        w = packed[3 * i: 3 * (i + LOOP_IDS)]
        if len(zlib.compress(w)) / len(w) < LOOP_RATIO_IDS:
            return True
    return False


def loop_check(row: dict, tok=None) -> str | None:
    """How looped() checks the row's thinking: "text", "ids", or None (it holds none)."""
    ids = isinstance(row.get("tokens"), list)
    if "reasoning_tail" in row or "reasoning" in row or (ids and tok is not None):
        return "text"
    return "ids" if ids else None


def looped(row: dict, tok=None) -> bool:
    """A thinking going in circles: its last 1500 chars compress below LOOP_RATIO.
    Rollout rows hold token ids: with tok they are decoded and checked the same way;
    without, by looped_ids."""
    text = thinking(row, tok)
    if text is None:
        ids = row.get("tokens")
        return isinstance(ids, list) and looped_ids(ids)
    tail = text[-1500:]
    return len(tail) >= 500 and len(zlib.compress(tail.encode())) / len(tail.encode()) < LOOP_RATIO


def how_failed(row: dict) -> str:
    if errored(row):
        return "error"
    if answer_cut(row):
        return "answer cut"
    if row.get("finish") == "length" or row.get("forced") == "length":
        return "cap hit"
    if row.get("forced") == "stop":
        return "stopped in thinking"
    if row.get("reason") in ("timeout", "case_timeout"):
        return "test timeout"
    if row.get("reason") == "compile_error":
        return "did not run"
    return "wrong answer"


# ---------- per model ----------

def stats(rows: dict, tok=None) -> dict:
    rs = list(rows.values())
    n = len(rs)
    share = lambda f: round(100 * sum(map(f, rs)) / n, 1) if n else None
    graded = [r for r in rs if r.get("passed") is not None]  # not yet tested: a partial file
    forced = [r for r in graded if r.get("forced")]
    toks = [t for t in map(n_tokens, rs) if t]
    how = Counter(loop_check(r, tok) for r in rs)
    how.pop(None, None)
    return {
        "n": n,
        "pass%": round(100 * sum(bool(r["passed"]) for r in graded) / len(graded), 1) if graded else None,
        "tokens_median": round(statistics.median(toks)) if toks else None,
        "cap%": share(lambda r: r.get("finish") == "length"),
        "forced%": share(lambda r: bool(r.get("forced"))),
        "forced_passed": f"{sum(bool(r['passed']) for r in forced)}/{len(forced)}",
        "answer_cut%": share(answer_cut),
        "timeout%": share(lambda r: r.get("reason") == "timeout"),
        "case_timeout%": share(lambda r: r.get("reason") == "case_timeout"),
        "error%": share(errored),
        "loop%": share(lambda r: looped(r, tok)),
        "loop_checked": sum(how.values()),
        "loop_on": "+".join(sorted(how)) or None,
    }


def compare(model: dict, ref: dict, tok=None) -> dict:
    both = model.keys() & ref.keys()
    lost = [model[i] for i in both if ref[i].get("passed") and model[i].get("passed") is False]
    gained = sum(1 for i in both if model[i].get("passed") and ref[i].get("passed") is False)
    ratios = [n_tokens(model[i]) / n_tokens(ref[i]) for i in both
              if model[i].get("passed") and ref[i].get("passed") and n_tokens(model[i]) and n_tokens(ref[i])]
    return {
        "shared": len(both),
        "lost": len(lost),
        "gained": gained,
        "lost_by": dict(Counter(map(how_failed, lost)).most_common()),
        "lost_looped": sum(looped(r, tok) for r in lost),
        "tokens_vs_ref": round(statistics.median(ratios), 2) if ratios else None,
    }


def report(src: Path | list[Path], ref: str = "ref-think", suite: str = "livecodebench", tok=None) -> dict:
    """{model: stats, plus vs_ref against the ref model when it is among them}."""
    models = load(src, suite)
    base = models.get(ref)
    return {name: {**stats(rows, tok), **({"vs_ref": compare(rows, base, tok)} if base and name != ref else {})}
            for name, rows in sorted(models.items())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, nargs="+", help="jobs directories, tars of them, or .jsonl files")
    ap.add_argument("--ref", default="ref-think", help="the full model's eval name (default ref-think)")
    ap.add_argument("--suite", default="livecodebench")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--out", type=Path, help="also write the JSON to this file")
    ap.add_argument("--tokenizer", help="a tokenizer directory, to check rollouts' thinking as text")
    a = ap.parse_args(argv)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = report(a.src, a.ref, a.suite, tok)
    if not out:
        print(f"no {a.suite} rows under {' '.join(map(str, a.src))}", file=sys.stderr)
        return 1
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    cols = ["n", "pass%", "tokens_median", "cap%", "forced%", "forced_passed", "answer_cut%", "timeout%", "loop%"]
    w = max(map(len, out)) + 2
    print("model".ljust(w) + "  ".join(f"{c:>13}" for c in cols))
    for name, s in out.items():
        print(name.ljust(w) + "  ".join(f"{str(s[c]):>13}" for c in cols))
    if a.ref not in out:
        print(f"\n(no '{a.ref}' eval found, so no comparison; pass --ref)")
    for name, s in out.items():
        v = s.get("vs_ref")
        if v:
            print(f"\n{name} vs {a.ref} on {v['shared']} shared answers: lost {v['lost']}, gained {v['gained']}, "
                  f"thinking {v['tokens_vs_ref']}x the ref's on answers both got right")
            print(f"  lost by: {v['lost_by']}; {v['lost_looped']} of the lost were looping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
