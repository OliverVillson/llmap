"""Download the code eval suites into one directory of JSONL files.

    uv run --with datasets --with huggingface_hub python scripts/fetch_codebench.py --out /mnt/nvme/codebench
    ... --only livecodebench codecontests apps     # what Event Horizon uses (setup_vm.sh SKIP_EXTRAS=1)

Writes <out>/<suite>.jsonl in the problem shape stages/codebench.py reads:
  multipl-e-{py,js,ts,cpp}  nuprl/MultiPL-E HumanEval translations (Python
                            falls back to openai/openai_humaneval)
  livecodebench             livecodebench/code_generation_lite release_v6 (problems from
                            2023-05 to 2025-04) with every test case; stages/eval.py scores
                            a date window of it (Config.lcb_since, lcb_until) on the first
                            Config.lcb_max_cases cases
  codecontests              deepmind/code_contests train split (Codeforces, AtCoder and
                            others, all from before 2022): training problems for the
                            data stage's contest rows (Config.data_contest_source) and
                            the traces, never scored
  apps                      codeparrot/apps call-based problems (Codewars and LeetCode,
                            2021 or earlier) as LiveCodeBench functional rows: starter
                            code in the prompt and {"func", "cases"} tests, the format of
                            the LeetCode part of the scored window, which CodeContests
                            lacks; training only
  general-calib             general English text (WikiText-103) for REAP calibration
                            in experiment 01's r50mix-gen ablation (Config.reap_calib_path)
and, for livecodebench, codecontests and apps, <out>/<suite>.meta.json once the JSONL is
complete: the dataset commit it was read at, the case cap, sha256s of the file and of
its problem ids (ids_sha256), and what was dropped and why. livecodebench's also lists
the ids, with their sha256, of each WINDOWS entry: the dev window that selection uses and
the test window that is reported, so every reading can check it scored the same set.

Pinning. Every Hugging Face read names a revision (REVISIONS). A branch name is resolved
to its commit once per fetch and every file is read at that commit, which the meta
records. The VM's NVMe is wiped when it stops, so the fetched suites go to the bucket
once and are restored from there (setup_vm.sh SKIP_SUITES=1 skips the fetch); otherwise
setup_vm.sh fetches only the suites that have no meta file.

Contamination. A CodeContests or APPS problem that shares a 13-word run with any
LiveCodeBench statement (stages/data.py _shingles, the dedupe the data stage and opd
apply again on load) is dropped here, so livecodebench is fetched first.

APPS stores a call-based case's expected value either inside a one-item list or bare,
and its own grader accepts both. Ours compares with ==, as lcb_runner does, so each
problem's cases are tried unwrapped first, then as given, and kept in the first form
on which one of its reference solutions passes every case in our sandbox, through the
harness that grades answers (codebench.job). A problem none passes is dropped: its
tests would mislabel the traces' `passed`.

Run it once per VM; eval reads the files offline.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pickle
import re
import sys
import tempfile
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # stages.data (dedupe), stages.codebench and sandbox (APPS check); stdlib only

MULTIPLE = {"py": "python", "js": "js", "ts": "ts", "cpp": "cpp"}
LCB_REPO = "livecodebench/code_generation_lite"
CC_REPO = "deepmind/code_contests"
APPS_REPO = "codeparrot/apps"
# The revision each dataset is read at; resolve() turns a branch into its commit, and
# each fetch records that commit as "sha" in <suite>.meta.json (Config.code_eval_dir),
# for the job's provenance. Event Horizon fetches once, in session 1, and every later VM
# restores that copy from the bucket instead of fetching (scripts/event_horizon.py
# run_setup); a commit pinned here keeps a fetch after that (a lost bucket copy, another
# experiment) on the same files.
# TODO(after Event Horizon's session 1): replace each "main" with the "sha" its
# <suite>.meta.json recorded (the summary's "Code suites at" line), so a later fetch reads
# the same files; test_revisions_are_branches_or_full_commits checks the form.
REVISIONS = {LCB_REPO: "main", CC_REPO: "main", APPS_REPO: "main"}
# release_v6: the files lcb_runner's --release_version release_v6 reads (1055 problems).
# Later releases add files; naming these keeps the set fixed even if the pin moves.
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")
# Inclusive date windows whose problem ids livecodebench.meta.json records: dev for
# selection (S1.4 and the screens), test for the reported reading.
WINDOWS = {"dev": ("2024-08-01", "2024-12-31"), "test": ("2025-01-01", "2025-04-30")}
SUITES = ("multipl-e", "livecodebench", "codecontests", "apps", "general-calib")  # in fetch order
APPS_SOLUTIONS = 3  # reference solutions tried per APPS problem


def resolve(repo: str) -> str:
    """The commit REVISIONS[repo] names, so one fetch reads every file at one commit."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo, revision=REVISIONS[repo]).sha


def ids_sha256(ids) -> str:
    """sha256 of the problem ids, sorted, one per line: the same set gives the same hash
    whatever order it was read in."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def multipl_e_row(lang: str, r: dict) -> dict:
    m = re.search(r"candidate\s*=\s*([\w:]+)", r["tests"])
    tests = r["tests"]
    # C++ completions stop before the closing brace, so the tests open with it.
    # Answers here are whole functions; codebench.assemble closes body-only ones.
    if tests.lstrip().startswith("}"):
        tests = tests.lstrip()[1:]
    return {"id": r["name"], "language": lang, "prompt": r["prompt"], "stub": r["prompt"],
            "entry": m.group(1) if m else "", "tests": tests}


def fetch_multipl_e(out: Path) -> None:
    from datasets import load_dataset

    for short, lang in MULTIPLE.items():
        try:
            ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{short}", split="test")
            rows = [multipl_e_row(lang, r) for r in ds]
        except Exception as e:
            if short != "py":
                raise
            print(f"MultiPL-E humaneval-py unavailable ({e}); using openai/openai_humaneval")
            ds = load_dataset("openai/openai_humaneval", split="test")
            rows = [{"id": r["task_id"], "language": "python", "prompt": r["prompt"], "stub": r["prompt"],
                     "entry": r["entry_point"], "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in ds]
        write(out / f"multipl-e-{short}.jsonl", rows)


class _StrOnly(pickle.Unpickler):
    """LiveCodeBench pickles a plain JSON string; refuse anything else."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def decode_cases(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        raw = _StrOnly(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
        return json.loads(raw)


def functional_tests(func: str, cases: list[dict]) -> dict:
    """A LeetCode-style problem's tests as data; stages/codebench.py builds the harness,
    with its time limit per case, at eval time."""
    return {"func": func, "cases": [[c["input"], c["output"]] for c in cases]}


def lcb_row(r: dict, max_cases: int = 0) -> dict | None:
    """A LiveCodeBench problem with its public cases first, then the private ones; the
    first max_cases of them, every one when max_cases is 0."""
    public = decode_cases(r["public_test_cases"])
    cases = public + decode_cases(r["private_test_cases"])
    if not cases:
        return None
    if max_cases:
        cases = cases[:max_cases]
    prompt = r["question_content"].strip()
    if cases[0].get("testtype") == "functional":
        func = json.loads(r.get("metadata") or "{}").get("func_name")
        if not func:
            return None
        prompt += f"\n\nUse this starter code:\n```python\n{r['starter_code'].rstrip()}\n```"
        tests = functional_tests(func, cases)
    else:
        tests = [{"input": c["input"], "output": c["output"]} for c in cases]
    return {"id": f"{r['platform']}/{r['question_id']}", "language": "python", "prompt": prompt, "stub": "",
            "entry": "", "tests": tests, "date": str(r["contest_date"])[:10], "difficulty": r.get("difficulty"),
            "public": min(len(public), len(cases))}  # the first cases are the examples in the problem text


def fetch_livecodebench(out: Path, max_cases: int = 0) -> dict:
    """release_v6 at its pinned commit, sorted by date, every case unless max_cases. Each
    row is spooled to disk as it is decoded: all of a problem's cases can run to hundreds
    of MB, and the whole release would not fit in memory."""
    from huggingface_hub import hf_hub_download, list_repo_files

    (out / "livecodebench.meta.json").unlink(missing_ok=True)
    sha = resolve(LCB_REPO)
    have = set(list_repo_files(LCB_REPO, repo_type="dataset", revision=sha))
    if missing := [f for f in LCB_FILES if f not in have]:
        raise RuntimeError(f"{LCB_REPO} at {sha} has no {', '.join(missing)}")
    index, seen, read, n = [], set(), 0, {"no_cases_or_func": 0, "duplicate": 0}
    with tempfile.TemporaryFile(dir=out) as spool:
        for f in LCB_FILES:
            with open(hf_hub_download(LCB_REPO, f, repo_type="dataset", revision=sha), encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    read += 1
                    row = lcb_row(json.loads(line), max_cases)
                    if not row:
                        n["no_cases_or_func"] += 1
                    elif row["id"] in seen:
                        n["duplicate"] += 1
                    else:
                        seen.add(row["id"])
                        data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                        index.append((row["date"], row["id"], isinstance(row["tests"], dict), spool.tell(), len(data)))
                        spool.write(data)
        index.sort()

        def lines():
            for *_, at, size in index:
                spool.seek(at)
                yield spool.read(size)

        info = write_lines(out / "livecodebench.jsonl", lines())
    windows = {}
    for name, (since, until) in WINDOWS.items():
        sel = [(i, fn) for d, i, fn, *_ in index if since <= d <= until]
        ids = sorted(i for i, _ in sel)
        windows[name] = {"since": since, "until": until, "n": len(ids), "functional": sum(fn for _, fn in sel),
                         "ids_sha256": ids_sha256(ids), "ids": ids}
    meta = {"repo": LCB_REPO, "revision": REVISIONS[LCB_REPO], "sha": sha, "files": list(LCB_FILES),
            "max_cases": max_cases, **info, "ids_sha256": ids_sha256(i for _, i, *_ in index),
            "read": read, "dropped": n, "windows": windows}
    write_meta(out, "livecodebench", meta)
    if index:
        print(f"livecodebench: {index[0][0]} .. {index[-1][0]}; "
              + ", ".join(f"{k} {w['since']}..{w['until']}: {w['n']} ({w['functional']} functional)"
                          for k, w in windows.items()))
    return meta


def lcb_shingles(out: Path) -> set[int]:
    """The 13-word shingles of every LiveCodeBench statement in out, read a row at a time
    (the file carries every test case)."""
    from stages.data import _shingles

    p = out / "livecodebench.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: fetch livecodebench first, the dedupe needs it")
    seen: set[int] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen |= _shingles(json.loads(line)["prompt"])
    return seen


# deepmind/code_contests' source codes
CC_SOURCES = {1: "codechef", 2: "codeforces", 3: "hackerearth", 4: "codejam", 5: "atcoder", 6: "aizu"}
CC_MAX_CASE_CHARS = 20000  # cases with a longer input or output are dropped, to keep the file small


def codecontests_row(r: dict, max_cases: int = 0) -> dict | None:
    """A deepmind/code_contests problem as a stdin/stdout problem: examples first, then the
    private and generated cases, up to max_cases (all when 0)."""
    def cases(t):
        return [{"input": i, "output": o} for i, o in zip(t["input"], t["output"])
                if len(i) <= CC_MAX_CASE_CHARS and len(o) <= CC_MAX_CASE_CHARS]

    public = cases(r["public_tests"])
    tests = public + cases(r["private_tests"]) + cases(r["generated_tests"])
    if max_cases:
        tests = tests[:max_cases]
    if len(tests) < 2 or not r["description"].strip():
        return None
    source = CC_SOURCES.get(r.get("source"), "other")
    return {"id": f"{source}/{r['name'].split('.')[0].strip()}", "language": "python",
            "prompt": r["description"].strip(), "stub": "", "entry": "", "tests": tests,
            "difficulty": r.get("difficulty"), "public": min(len(public), len(tests))}


def fetch_codecontests(out: Path, max_cases: int = 50, n: int = 6000) -> dict:
    """Up to n problems of the train split that share no 13-word run with LiveCodeBench,
    streamed so the multi-GB split is not all downloaded."""
    from datasets import load_dataset
    from stages.data import _shingles

    (out / "codecontests.meta.json").unlink(missing_ok=True)
    lcb = lcb_shingles(out)
    sha = resolve(CC_REPO)
    rows, seen, overlap = [], set(), 0
    for r in load_dataset(CC_REPO, split="train", streaming=True, revision=sha):
        row = codecontests_row(r, max_cases)
        if not row or row["id"] in seen:
            continue
        seen.add(row["id"])
        if _shingles(row["prompt"]) & lcb:
            overlap += 1
            continue
        rows.append(row)
        if len(rows) >= n:
            break
    info = write(out / "codecontests.jsonl", rows)
    meta = {"repo": CC_REPO, "revision": REVISIONS[CC_REPO], "sha": sha, "max_cases": max_cases, **info,
            "ids_sha256": ids_sha256(r["id"] for r in rows), "dropped": {"lcb_overlap": overlap}}
    write_meta(out, "codecontests", meta)
    print(f"codecontests: {overlap} problems dropped for a 13-word run in common with LiveCodeBench")
    return meta


def apps_cases(io_: dict, unwrap: bool, max_cases: int = 0) -> list[dict] | None:
    """An APPS call-based problem's cases in LiveCodeBench's functional form: each argument
    as JSON on its own line, the expected value as JSON (taken out of its one-item list
    when unwrap); None when an input is not a list of arguments."""
    cases = []
    for args, want in zip(io_.get("inputs") or [], io_.get("outputs") or []):
        if not isinstance(args, list):
            return None
        cases.append({"input": "\n".join(json.dumps(a) for a in args), "output": json.dumps(want[0] if unwrap else want)})
    return cases[:max_cases] if max_cases else cases


def apps_rows(r: dict, split: str, max_cases: int = 0) -> list[dict]:
    """One APPS (or TACO: same fields) call-based problem as LiveCodeBench functional rows,
    one per reading of its expected values, unwrapped first (see the module docstring);
    [] for a stdin problem, or one whose starter code does not define its function."""
    try:
        io_ = json.loads(r.get("input_output") or "{}")
    except json.JSONDecodeError:
        return []
    func, starter, question = io_.get("fn_name"), (r.get("starter_code") or "").rstrip(), (r.get("question") or "").strip()
    if not func or not question or not re.search(rf"\bdef\s+{re.escape(func)}\b", starter):
        return []
    outs = io_.get("outputs") or []
    forms = [True, False] if outs and all(isinstance(o, list) and len(o) == 1 for o in outs) else [False]
    prompt = f"{question}\n\nUse this starter code:\n```python\n{starter}\n```"
    rows = []
    for unwrap in forms:
        if cases := apps_cases(io_, unwrap, max_cases):
            rows.append({"id": f"apps/{split}/{r.get('problem_id', r.get('id'))}", "language": "python",
                         "prompt": prompt, "stub": "", "entry": "", "tests": functional_tests(func, cases),
                         "difficulty": r.get("difficulty"), "public": 0, "url": r.get("url") or ""})
    return rows


def verified(cands: list[list[dict]], solutions: list[list[str]]) -> list[dict | None]:
    """For each problem, the first of its rows (cands[i]) on which one of its reference
    solutions passes every case, run as answers are graded (codebench.job in the
    sandbox); None when none does."""
    from stages import codebench, sandbox

    items, owner = [], []
    for i, (rows, sols) in enumerate(zip(cands, solutions)):
        for k, row in enumerate(rows):
            for code in sols:
                items.append(codebench.job(row, "", code=code))
                owner.append((i, k))
    print(f"apps: running {len(items)} reference solutions in the sandbox", flush=True)
    passed = {o for o, res in zip(owner, sandbox.run_many(items)) if res.passed}
    return [next((row for k, row in enumerate(rows) if (i, k) in passed), None) for i, rows in enumerate(cands)]


def fetch_apps(out: Path, max_cases: int = 50) -> dict:
    """APPS's call-based problems from every split (all from 2021 or earlier, none scored)
    that share no 13-word run with LiveCodeBench, each in the form one of its reference
    solutions passes; a problem whose statement repeats an earlier one's is kept once."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from stages.data import _shingles

    (out / "apps.meta.json").unlink(missing_ok=True)
    lcb = lcb_shingles(out)
    sha = resolve(APPS_REPO)
    files = sorted(f for f in list_repo_files(APPS_REPO, repo_type="dataset", revision=sha) if f.endswith(".jsonl"))
    if not files:
        raise RuntimeError(f"{APPS_REPO} at {sha} has no .jsonl files")
    cands, sols, texts = [], [], set()
    n = {"call_based": 0, "duplicate": 0, "lcb_overlap": 0, "no_solution": 0}
    for f in files:
        with open(hf_hub_download(APPS_REPO, f, repo_type="dataset", revision=sha), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows = apps_rows(r, Path(f).stem, max_cases)
                if not rows:
                    continue
                n["call_based"] += 1
                text = " ".join(re.findall(r"[a-z0-9]+", r["question"].lower()))
                if text in texts:
                    n["duplicate"] += 1
                    continue
                texts.add(text)
                if _shingles(rows[0]["prompt"]) & lcb:
                    n["lcb_overlap"] += 1
                    continue
                try:
                    s = json.loads(r.get("solutions") or "[]")
                except json.JSONDecodeError:
                    s = []
                s = [c for c in s if isinstance(c, str) and c.strip()][:APPS_SOLUTIONS]
                if not s:
                    n["no_solution"] += 1
                    continue
                cands.append(rows)
                sols.append(s)
    keep = verified(cands, sols)
    rows = [r for r in keep if r]
    n["reference_failed"] = len(keep) - len(rows)
    if not rows:
        raise RuntimeError(f"no APPS problem passed its reference solutions ({n}); is python3 on PATH?")
    as_given = sum(1 for r, c in zip(keep, cands) if r is not None and len(c) > 1 and r is c[1])
    info = write(out / "apps.jsonl", rows)
    meta = {"repo": APPS_REPO, "revision": REVISIONS[APPS_REPO], "sha": sha, "files": files, "max_cases": max_cases,
            **info, "ids_sha256": ids_sha256(r["id"] for r in rows), "dropped": n, "outputs_as_given": as_given}
    write_meta(out, "apps", meta)
    print(f"apps: {len(rows)} of {n['call_based']} call-based problems kept; dropped {n}")
    return meta


def fetch_general_calib(out: Path, n: int = 2000) -> None:
    """Wikipedia articles of at least ~200 words, one {"text"} row each."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rows, doc = [], []
    for line in ds["text"]:
        if line.startswith(" = ") and not line.startswith(" = = "):  # a new article
            text = "".join(doc).strip()
            if len(text.split()) >= 200:
                rows.append({"text": text})
                if len(rows) >= n:
                    break
            doc = []
        doc.append(line)
    write(out / "general-calib.jsonl", rows)


def write_lines(path: Path, lines) -> dict:
    """JSONL lines (bytes) to path through a .part file renamed at the end, so an interrupted
    fetch leaves no partial suite; the rows, bytes and sha256 of what was written."""
    tmp = path.with_name(path.name + ".part")
    h, rows, size = hashlib.sha256(), 0, 0
    with open(tmp, "wb") as f:
        for line in lines:
            f.write(line)
            h.update(line)
            rows, size = rows + 1, size + len(line)
    os.replace(tmp, path)
    print(f"{path}: {rows} problems")
    return {"rows": rows, "bytes": size, "file_sha256": h.hexdigest()}


def write(path: Path, rows: list[dict]) -> dict:
    return write_lines(path, ((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))


def write_meta(out: Path, suite: str, meta: dict) -> None:
    """<suite>.meta.json, written after the suite's JSONL is complete: setup_vm.sh takes it
    to mean the suite is fetched."""
    p = out / f"{suite}.meta.json"
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps({"suite": suite, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta},
                              indent=1) + "\n")
    os.replace(tmp, p)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", choices=SUITES, help="fetch these suites only (default: all)")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="test cases kept per LiveCodeBench problem; 0 (the default) keeps them all, and the "
                         "eval reads the first Config.lcb_max_cases")
    ap.add_argument("--train-max-cases", type=int, default=50,
                    help="test cases kept per CodeContests or APPS problem (they only label training text); 0 keeps all")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only or SUITES)
    if "multipl-e" in only:
        fetch_multipl_e(out)
    if "livecodebench" in only:
        fetch_livecodebench(out, args.max_cases)
    if "codecontests" in only:
        fetch_codecontests(out, args.train_max_cases)
    if "apps" in only:
        fetch_apps(out, args.train_max_cases)
    if "general-calib" in only:
        fetch_general_calib(out)


if __name__ == "__main__":
    main()
