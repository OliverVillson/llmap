"""Code benchmarks for execution-based eval: loading, prompting, scoring.

Every suite is a JSONL file of problems in one shape:
  id        unique within the suite
  language  python | javascript | typescript | c | cpp (stages/sandbox.py)
  prompt    what the model is asked (a function stub with its docstring, or a
            full problem statement)
  stub      code the answer completes (MultiPL-E prompt); "" for none
  entry     name of the function the tests call ("" for stdin programs)
  tests     test code appended to the answer (a string), or a list of
            {"input", "output"} stdin/stdout cases (Python only; assemble()
            turns them into a test harness, since the sandbox runs answer + tests)
  date      optional ISO date the problem was published (LiveCodeBench)

Suites (Config.code_eval_suites):
  multipl-e-py, multipl-e-js, multipl-e-ts, multipl-e-cpp
                 HumanEval translated by MultiPL-E, 161 problems each
  c-set          our hand-written C problems (benchmarks/c-set.jsonl)
  livecodebench  LiveCodeBench problems (Python) published on or after
                 Config.lcb_since, so they are newer than the base model
  heldout        the job's own held-out rows that carry a language and tests
                 (code TaskSpecs, see common/taskspec.py)
Downloaded suites live in Config.code_eval_dir; scripts/fetch_codebench.py
writes them there. c-set ships with the repo.

pass@k uses the unbiased estimator from the Codex paper (Chen et al. 2021):
with n samples per problem of which c pass, pass@k = 1 - C(n-c, k) / C(n, k).
"""

from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLED = {"c-set": REPO / "benchmarks" / "c-set.jsonl"}
LANGUAGE_NAMES = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript", "c": "C", "cpp": "C++",
                  "asm": "x86-64 assembly"}
ALIASES = {"py": "python", "js": "javascript", "ts": "typescript", "c++": "cpp", "cxx": "cpp"}
FENCE = {"python": "python", "javascript": "javascript", "typescript": "typescript", "c": "c", "cpp": "cpp", "asm": "asm"}
SYSTEM = ("You are an expert programmer. Answer with one complete, correct code block "
          "and nothing else: no explanation, no example usage, no tests.")
# Imports LiveCodeBench prepends for LeetCode-style answers, which assume them.
PY_PRELUDE = ("from typing import *\nfrom collections import *\nfrom functools import *\n"
              "from itertools import *\nfrom heapq import *\nfrom bisect import *\n"
              "import math, string, re, sys, random\n")


def normalize_language(language: str) -> str:
    lang = ALIASES.get(language.lower(), language.lower())
    if lang not in LANGUAGE_NAMES:
        raise ValueError(f"unsupported language {language!r}; expected one of {', '.join(LANGUAGE_NAMES)}")
    return lang


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"pass@{k} needs at least {k} samples per problem, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def suite_path(name: str, bench_dir: str | Path) -> Path:
    return BUNDLED.get(name) or Path(bench_dir) / f"{name}.jsonl"


def load_suite(name: str, bench_dir: str | Path, since: str | None = None, limit: int | None = None) -> list[dict]:
    p = suite_path(name, bench_dir)
    if not p.exists():
        raise FileNotFoundError(f"code suite {name} not found at {p}; run scripts/fetch_codebench.py")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if since:
        rows = [r for r in rows if not r.get("date") or r["date"][:10] >= since]
    for r in rows:
        r["language"] = normalize_language(r["language"])
        r["suite"] = name
    return rows[:limit] if limit else rows


def heldout_problems(held: list[dict]) -> list[dict]:
    """Held-out rows of a code TaskSpec job that can be executed."""
    out = []
    for i, h in enumerate(held):
        if h.get("language") and h.get("tests"):
            out.append({"id": f"heldout/{i}", "suite": "heldout", "language": normalize_language(h["language"]),
                        "prompt": h["input"], "stub": h.get("stub", ""), "entry": h.get("entry", ""),
                        "tests": h["tests"], "raw_prompt": True, "system": h.get("system", "")})
    return out


def build_prompt(p: dict) -> str:
    if p.get("raw_prompt"):
        return p["prompt"]
    lang, name = p["language"], LANGUAGE_NAMES[p["language"]]
    if p.get("stub"):
        return (f"Complete the following {name} function. Reply with the whole function, including its "
                f"signature and any imports or includes it needs, in one ```{FENCE[lang]} code block.\n\n"
                f"```{FENCE[lang]}\n{p['prompt'].rstrip()}\n```")
    if isinstance(p["tests"], list):
        return (f"{p['prompt'].rstrip()}\n\nWrite a complete {name} program that reads the input from "
                f"stdin and writes the answer to stdout. Reply with the program in one ```{FENCE[lang]} code block.")
    return f"{p['prompt'].rstrip()}\n\nReply in {name}, in one ```{FENCE[lang]} code block."


def extract_code(answer: str) -> str:
    """The longest fenced code block, else the whole answer (minus a stray fence)."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    blocks = re.findall(r"```[\w+#.-]*[ \t]*\n(.*?)```", answer, flags=re.S)
    if blocks:
        return max(blocks, key=len).rstrip()
    return re.sub(r"^```[\w+#.-]*\s*\n|\n?```\s*$", "", answer.strip("\n").rstrip())  # keeps a body's indent


def ticket(p: dict, code: str = ""):
    """A problem as a Mugge ticket (stages/harness.py), for code_eval_format "harness":
    the problem is the context, the answer file is the one owned file and the sandbox's
    build and run are the acceptance commands. code is the file as it is now."""
    from stages import harness, sandbox

    lang, name = p["language"], LANGUAGE_NAMES[p["language"]]
    src, cmds, where = sandbox.layout(lang)
    if p.get("raw_prompt"):
        task = p["prompt"]
    elif p.get("stub"):
        task = f"Complete the following {name} function.\n\n```{FENCE[lang]}\n{p['prompt'].rstrip()}\n```"
    elif isinstance(p["tests"], list):
        task = f"{p['prompt'].rstrip()}\n\nWrite a complete {name} program that reads the input from stdin and writes the answer to stdout."
        where = f"{src} is run on the test inputs and its output is compared with the expected output."
    else:
        task = p["prompt"].rstrip()
    first = next((l.strip() for l in task.splitlines() if l.strip() and not l.startswith("```")), "Write the code")
    pid = str(p["id"]) if str(p["id"]).startswith(p["suite"]) else f"{p['suite']}-{p['id']}"
    return harness.Ticket(id=re.sub(r"[^\w.-]+", "-", pid), title=(f"Implement {p['entry']}" if p.get("entry") else first)[:80],
                          context=f"{task.rstrip()}\n\n{where}", owns=[src], acceptance=cmds,
                          current={src: code} if code else {})


def harness_code(p: dict, answer: str) -> str:
    """The owned file out of a harness-format answer; a lone fenced block also counts."""
    from stages import harness, sandbox

    files, _ = harness.parse_files(answer)
    src = sandbox.layout(p["language"])[0]
    if src in files:
        return files[src]
    return next(iter(files.values())) if len(files) == 1 else extract_code(answer)


def harness_fix_text(p: dict, code: str, failed) -> str:
    """Mugge's fix call for a failed harness-format answer."""
    from stages import harness, sandbox

    _, cmds, _ = sandbox.layout(p["language"])
    cmd = cmds[0] if failed.reason == "compile_error" and len(cmds) > 1 else cmds[-1]
    out = harness.tail(failed.output or "", FIX_OUTPUT_CHARS) or ("(timed out)" if failed.reason == "timeout" else "(no output)")
    return harness.fix_text(ticket(p, code), cmd, out)


FIX_OUTPUT_CHARS = 2000
_FAILED = {"compile_error": "It did not build:", "timeout": "It ran out of time:"}


def fix_prompt(p: dict, answer: str, failed) -> str:
    """One repair turn, the way Mugge's harness asks for a fix: the task again, the
    code as it is now and what the failing run printed, in one fresh message (no
    chat history). The model answers with the whole code again. failed is the
    sandbox Result of the first answer."""
    lang = FENCE[p["language"]]
    out = (failed.output or "").strip()
    if len(out) > FIX_OUTPUT_CHARS:
        out = "…" + out[-FIX_OUTPUT_CHARS:]
    return (f"{build_prompt(p)}\n\n## Your code as it is now\n```{lang}\n{extract_code(answer)}\n```\n\n"
            f"## Fix\n{_FAILED.get(failed.reason, 'It failed its tests:')}\n{out or '(no output)'}\n\n"
            f"Reply with the whole fixed code in one ```{lang} code block.")


def add_fix_scores(suites: dict, problems: list[dict], first: list, fixed: list) -> None:
    """fix@1 per suite: the share of problems whose first answer passed, or whose one
    fix did. first and fixed hold one sandbox Result (or None) per problem."""
    for name, s in suites.items():
        if "pass@1" not in s:
            continue
        idx = [j for j, p in enumerate(problems) if p["suite"] == name]
        ok = sum(first[j].passed or bool(fixed[j] and fixed[j].passed) for j in idx)
        s["fix@1"] = round(ok / len(idx), 4)
        s["fixed"] = sum(bool(fixed[j] and fixed[j].passed) for j in idx)


def mean_fix1(suites: dict) -> float | None:
    vals = [s["fix@1"] for s in suites.values() if "fix@1" in s]
    return round(sum(vals) / len(vals), 4) if vals else None


_PREAMBLE = re.compile(r"^\s*(#include\b|using namespace\b|from \S+ import\b|import \S|const \w+ = require\()")


def _strip_main(code: str, lang: str) -> str:
    """Drop the answer's own entry point when the tests bring theirs."""
    if lang in ("c", "cpp"):
        m = re.search(r"^\s*(int|void)\s+main\s*\(", code, flags=re.M)
        return code[:m.start()].rstrip() if m else code
    if lang == "python":
        m = re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", code, flags=re.M)
        return code[:m.start()].rstrip() if m else code
    return code


def assemble(p: dict, answer: str, code: str | None = None) -> tuple[str, str]:
    """(code, tests) for sandbox.run_tests: the answer's code (or code, when the caller
    already took it out of the answer), completed with the stub when the model only
    wrote a body and with the stub's imports/includes up front; stdin/stdout cases
    become a Python harness around the answer."""
    lang = p["language"]
    code = extract_code(answer) if code is None else code
    tests = p["tests"]
    if isinstance(tests, str) and (lang == "python" or re.search(r"\bmain\s*\(", tests)):
        code = _strip_main(code, lang)  # the tests are the entry point (stdin programs keep theirs)
    stub, entry = p.get("stub") or "", p.get("entry") or ""
    if stub:
        defines = entry and re.search(rf"\b{re.escape(entry)}\s*[(<=:]", code)
        if not defines:
            code = stub.rstrip("\n") + "\n" + code  # a body-only completion
            if lang != "python":  # close the function if the model stopped inside it
                code += "\n}" * max(0, code.count("{") - code.count("}"))
        elif lang in ("c", "cpp") and "typedef" in stub and "typedef" not in code:
            code = stub.rstrip("\n") + "\n" + code  # the types the stub declares; repeated prototypes are fine
        else:
            pre = [l for l in stub.splitlines() if _PREAMBLE.match(l) and l.strip() not in code]
            if pre:
                code = "\n".join(pre) + "\n" + code
    if p.get("suite") == "livecodebench" and lang == "python" and isinstance(tests, str):
        code = PY_PRELUDE + code
    if not isinstance(tests, str):
        return stdio_harness(code, tests), ""
    return code, tests


def stdio_harness(code: str, cases: list[dict]) -> str:
    """A Python program that runs `code` once per case with that case's stdin and
    fails unless its stdout matches (trailing whitespace ignored per line)."""
    data = json.dumps([[c["input"], c["output"]] for c in cases])
    return (
        "import io as _io, json as _json, sys as _sys\n"
        f"_SRC = {code!r}\n"
        f"_CASES = _json.loads({data!r})\n"
        "_norm = lambda s: [l.rstrip() for l in s.strip().splitlines()]\n"
        "_real_out = _sys.stdout\n"
        "for _i, (_inp, _want) in enumerate(_CASES):\n"
        "    _sys.stdin = _io.TextIOWrapper(_io.BytesIO(_inp.encode()), encoding='utf-8')\n"
        "    _buf = _io.BytesIO()\n"
        "    _out = _io.TextIOWrapper(_buf, encoding='utf-8', write_through=True)\n"
        "    _sys.stdout = _out\n"
        "    try:\n"
        "        exec(compile(_SRC, 'answer.py', 'exec'), {'__name__': '__main__'})\n"
        "    except SystemExit as _e:\n"
        "        if _e.code not in (None, 0):\n"
        "            raise\n"
        "    finally:\n"
        "        _out.flush()\n"
        "        _sys.stdout = _real_out\n"
        "    _got = _buf.getvalue().decode('utf-8', 'replace')\n"
        "    if _norm(_got) != _norm(_want):\n"
        "        raise SystemExit(f'case {_i}: expected {_want[:200]!r}, got {_got[:200]!r}')\n")


CANNOT_RUN = {"missing_toolchain", "unsupported_language"}


def summarize(problems: list[dict], results: list[list], ks: list[int]) -> dict:
    """Per-suite pass@k from per-problem lists of sandbox Results."""
    suites: dict[str, dict] = {}
    for p, rs in zip(problems, results):
        s = suites.setdefault(p["suite"], {"language": p["language"], "n": 0, "_pk": {k: 0.0 for k in ks}, "status": {}})
        if s["language"] != p["language"]:
            s["language"] = "mixed"
        s["n"] += 1
        c = sum(r.passed for r in rs)
        for k in ks:
            s["_pk"][k] += pass_at_k(len(rs), c, k)
        for r in rs:
            status = "pass" if r.passed else (r.reason or "test_failed")
            s["status"][status] = s["status"].get(status, 0) + 1
    out = {}
    for name, s in suites.items():
        pk = {f"pass@{k}": round(v / s["n"], 4) for k, v in s.pop("_pk").items()}
        if set(s["status"]) <= CANNOT_RUN:  # no toolchain for it: report, don't score as 0
            out[name] = {**s, "skipped": ", ".join(sorted(s["status"]))}
        else:
            out[name] = {**s, **pk}
    return out


def mean_pass1(suites: dict) -> float | None:
    """Unweighted mean pass@1 over suites, so each language counts the same."""
    vals = [s["pass@1"] for s in suites.values() if "pass@1" in s]
    return round(sum(vals) / len(vals), 4) if vals else None


def ref_suites(ref_report: dict) -> dict:
    """Per-suite code results of the reference model in another job's eval.json:
    the candidate named "ref", else that job's winner."""
    cands = {c["name"]: c for c in ref_report.get("candidates", [])}
    c = cands.get("ref") or cands.get(ref_report.get("winner"))
    return ((c or {}).get("code") or {}).get("suites", {})


def share_of_ref(suites: dict, ref: dict) -> dict:
    """pass@1 as a share of the reference per suite, plus their mean and minimum
    (experiment 01's bar looks at both)."""
    out = {}
    for name, s in suites.items():
        r = ref.get(name, {}).get("pass@1")
        if "pass@1" in s and r:
            out[name] = round(s["pass@1"] / r, 4)
    if out:
        vals = list(out.values())
        out = {"suites": out, "mean": round(sum(vals) / len(vals), 4), "min": round(min(vals), 4)}
    return out
