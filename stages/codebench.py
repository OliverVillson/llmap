"""Code benchmarks for execution-based eval: loading, prompting, scoring.

Every suite is a JSONL file of problems in one shape:
  id        unique within the suite
  language  python | javascript | typescript | c | cpp (stages/sandbox.py)
  prompt    what the model is asked (a function stub with its docstring, or a
            full problem statement)
  stub      code the answer completes (MultiPL-E prompt); "" for none
  entry     name of the function the tests call ("" for stdin programs)
  tests     test code appended to the answer (a string), a list of
            {"input", "output"} stdin/stdout cases, or {"func", "cases"} for a
            LeetCode-style Solution method (both Python only; job() turns them
            into a test harness with a time limit per case, since the sandbox runs
            answer + tests)
  date      optional ISO date the problem was published (LiveCodeBench)

Problems whose tests are cases (the last two shapes: LiveCodeBench, CodeContests)
are graded the way lcb_runner (LiveCodeBench's evaluation/testing_util.py) grades
them, so our reading of an answer is the official one:
  - its import_string prelude, verbatim, plus the 50000-digit int limit, before
    every answer, stdin programs included (bare inf, gcd, deepcopy, defaultdict;
    recursion to 50000; big ints)
  - stdout compared line by line after stripping both sides, then token by token
    as Decimal, with no tolerance (0.5 == 0.500000, 1/3 at 16 digits != at 18)
  - a functional answer's Solution().func when its code has `class Solution`, else
    its top-level func
  - an exit, sys.exit(1) included, ends a stdin program; its output is still graded
  - the last complete fenced block is the code (extract_code)
  - a program that runs main() in a thread and returns is not waited for: what the
    thread prints after that is lost and the case fails, as in lcb_runner
Where we differ from lcb_runner, checked by running both on the same answers
(tests/test_codebench.py, LCB_PARITY): lcb_runner patches builtins.open with the
case input, but its own prelude has already bound the real open in the answer's
namespace, so there open(0) reads the runner's own stdin; we give the case input
to open(0), open('/dev/stdin') and open(sys.stdin.fileno()), as it means to.
Its stdin stand-in cannot be iterated (`for line in sys.stdin`) and its
sys.stdin.buffer.readline() returns the first line every time, and it runs a
stdin program inside a function, so `global` names fail; all of these pass here.
The cases go to the sandbox as a cases.json file next to the program (job()), so
large test data never hits the sandbox's file size limit.

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
import os
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
# Seconds each LiveCodeBench test case may run, as in LiveCodeBench's own runner; a
# problem's cases share one process, so its time limit is this times the cases.
CASE_TIMEOUT = float(os.environ.get("LOBBOT_LCB_CASE_TIMEOUT", 6))
# lcb_runner's import_string (evaluation/testing_util.py), verbatim: it runs before
# every answer there, functional and stdin alike, and answers lean on it.
LCB_IMPORTS = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
               "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
               "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
               "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
               "from builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
               "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
               "import statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
               "import json\nsys.setrecursionlimit(50000)\n")
# lcb_runner also lifts the int-to-str limit for its whole run
# (compute_code_generation_metrics.py), so answers and expected outputs may have
# more than 4300 digits.
_DIGITS = "if hasattr(sys, 'set_int_max_str_digits'):\n    sys.set_int_max_str_digits(50000)\n"
PY_PRELUDE = LCB_IMPORTS + _DIGITS
# The file job() hands the sandbox with a problem's cases, which its harness reads.
CASES_FILE = "cases.json"


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


def load_suite(name: str, bench_dir: str | Path, since: str | None = None, limit: int | None = None, *,
               until: str | None = None, ids=None, fields: tuple[str, ...] | None = None) -> list[dict]:
    """A suite's problems, read one line at a time and filtered row by row: a LiveCodeBench
    fetch with every case stored is gigabytes, too much to read whole. Dated rows outside
    since..until are skipped (undated ones kept), only `ids` when given, only `fields` of
    each row when given (with its language; e.g. ("id", "prompt") for a dedupe that needs
    no tests), at most `limit` rows."""
    p = suite_path(name, bench_dir)
    if not p.exists():
        raise FileNotFoundError(f"code suite {name} not found at {p}; run scripts/fetch_codebench.py")
    want = None if ids is None else set(ids)
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            date = (r.get("date") or "")[:10]
            if date and ((since and date < since) or (until and date > until)):
                continue
            if want is not None and r.get("id") not in want:
                continue
            if fields is not None:
                r = {k: r[k] for k in (*fields, "language") if k in r}
            r["language"] = normalize_language(r["language"])
            r["suite"] = name
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


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


def last_block(text: str) -> str | None:
    """The last complete fenced code block in text, None when there is none. As in
    lcb_runner, a fence is any line with ``` in it and the code is the lines between
    two fences; fences pair up in order, so an opening fence with no close (an answer
    cut off at the token cap) does not count, where lcb_runner would take the text
    before it."""
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return None
    end = len(fences) - len(fences) % 2
    return "\n".join(lines[fences[end - 2] + 1:fences[end - 1]])


def extract_code(answer: str) -> str:
    """The last complete fenced code block, else the whole answer (minus a stray fence)."""
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    block = last_block(answer)
    if block is not None:
        return block.rstrip()
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

    src = sandbox.layout(p["language"])[0]
    files, _ = harness.parse_files(answer, [src])
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


def examples_only(p: dict) -> dict:
    """The problem with only the example cases its text shows (LiveCodeBench's public
    tests, the first p["public"] cases), as a coder can check before handing in."""
    k, tests = p["public"], p["tests"]
    return {**p, "tests": {**tests, "cases": tests["cases"][:k]} if isinstance(tests, dict) else tests[:k]}


def add_pick_scores(suites: dict, problems: list[dict], results: list[list], examples: list[list]) -> None:
    """pick@examples per suite: of a problem's samples, the first that passes the examples
    in its text is handed in (the first sample when none does), as Mugge's loop would.
    Only problems with example cases count. results and examples hold one entry per sample."""
    picked: dict[str, list[bool]] = {}
    for p, rs, ex in zip(problems, results, examples):
        if p.get("public") and len(ex) == len(rs):
            picked.setdefault(p["suite"], []).append(rs[next((i for i, ok in enumerate(ex) if ok), 0)].passed)
    for name, got in picked.items():
        if "pass@1" in suites.get(name, {}):
            suites[name]["pick@examples"] = round(sum(got) / len(got), 4)


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


def job(p: dict, answer: str, code: str | None = None) -> tuple[str, str, str, float | None, dict]:
    """(language, code, tests, time limit, files): one sandbox.run_many item for the
    answer. Case-shaped tests travel as files[CASES_FILE], which the sandbox writes
    next to the program and the harness reads, so the program stays small however
    much test data a problem has. code is the answer's code when the caller already
    took it out (see assemble)."""
    code, tests, cases = _assemble(p, answer, code, embed=False)
    files = {CASES_FILE: json.dumps(cases)} if cases is not None else {}
    return p["language"], code, tests, time_limit(p), files


def assemble(p: dict, answer: str, code: str | None = None) -> tuple[str, str]:
    """(code, tests) for sandbox.run_tests: the answer's code (or code, when the caller
    already took it out of the answer), completed with the stub when the model only
    wrote a body and with the stub's imports/includes up front; stdin/stdout cases
    become a Python harness around the answer. For callers that pass no files: the
    cases are written into the harness, which job() avoids."""
    code, tests, _ = _assemble(p, answer, code, embed=True)
    return code, tests


def _assemble(p: dict, answer: str, code: str | None, embed: bool) -> tuple[str, str, list | None]:
    """(code, tests, cases): cases is what the harness reads from CASES_FILE, None when
    the tests are not cases or embed wrote them into the harness."""
    lang = p["language"]
    code = extract_code(answer) if code is None else code
    tests = p["tests"]
    if isinstance(tests, dict) or isinstance(tests, str) and (lang == "python" or re.search(r"\bmain\s*\(", tests)):
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
    if isinstance(tests, str):
        return code, tests, None
    if isinstance(tests, dict):
        cases = tests["cases"]
        harness = functional_harness(tests["func"], cases if embed else None, solution="class Solution" in code)
        return _GUARD + PY_PRELUDE + code, harness, None if embed else cases
    cases = [[c["input"], c["output"]] for c in tests]
    return stdio_harness(code, cases if embed else None), "", None if embed else cases


def time_limit(p: dict) -> float | None:
    """The sandbox time limit for a problem whose cases run in one harness: CASE_TIMEOUT
    per case plus start-up; None for any other problem (the suite's limit applies)."""
    tests = p.get("tests")
    cases = tests if isinstance(tests, list) else tests.get("cases") if isinstance(tests, dict) else None
    return CASE_TIMEOUT * len(cases) + 10 if cases else None


# Head of both harnesses. Per-case limit: an itimer raises _CaseTimeout, a
# BaseException, so an answer's own `except Exception` cannot swallow it.
_HEAD = (
    "import io as _io, json as _json, signal as _signal, sys as _sys\n"
    "if hasattr(_sys, 'set_int_max_str_digits'):\n"
    "    _sys.set_int_max_str_digits(50000)\n"
    "class _CaseTimeout(BaseException):\n"
    "    pass\n"
    "def _alarm(*_):\n"
    "    raise _CaseTimeout()\n"
    "_signal.signal(_signal.SIGALRM, _alarm)\n")
# os._exit would end a harness with exit code 0 before the remaining cases are
# checked, so from here on it raises SystemExit, as sys.exit does.
_NO_EXIT = (
    "import os as _os\n"
    "def _exit(status=0):\n"
    "    raise SystemExit(status)\n"
    "_os._exit, _real_exit = _exit, _os._exit\n")
# Goes before a functional answer, whose code runs at the top level of the program:
# an exit there would skip the tests with exit code 0, so the run fails instead.
# The harness disarms it once it starts.
_GUARD = _NO_EXIT + (
    "import atexit as _atexit\n"
    "def _exited():\n"
    "    _os.write(2, b'the answer ended before its tests ran\\n')\n"
    "    _real_exit(1)\n"
    "_atexit.register(_exited)\n")


def _load_cases(cases: list | None) -> str:
    """Harness line that sets _CASES: from CASES_FILE, or written in when cases are given."""
    if cases is None:
        return f"with _io.open({CASES_FILE!r}, encoding='utf-8') as _fh:\n    _CASES = _json.load(_fh)\n"
    return f"_CASES = _json.loads({json.dumps(cases)!r})\n"


def functional_harness(func: str, cases: list | None = None, solution: bool = True) -> str:
    """Tests for a LeetCode-style answer: Solution().func(*args) per case (the top-level
    func when solution is False: lcb_runner looks for `class Solution` in the code),
    each within CASE_TIMEOUT seconds. A case's input is one JSON value per line. The
    cases come from CASES_FILE unless given. It goes after the code _assemble builds,
    which starts with _GUARD and PY_PRELUDE."""
    target = f"getattr(Solution(), {func!r})" if solution else f"globals()[{func!r}]"
    return (
        _HEAD + "_atexit.unregister(_exited)\n" + _load_cases(cases) +
        f"_f = {target}\n"
        "for _i, (_inp, _out) in enumerate(_CASES):\n"
        "    _args = [_json.loads(_l) for _l in _inp.split('\\n') if _l.strip()]\n"
        "    _want = _json.loads(_out)\n"
        f"    _signal.setitimer(_signal.ITIMER_REAL, {CASE_TIMEOUT})\n"
        "    try:\n"
        "        _got = _f(*_args)\n"
        "    except _CaseTimeout:\n"
        f"        raise SystemExit(f'case {{_i}}: over the {CASE_TIMEOUT:g} s time limit')\n"
        "    except SystemExit:\n"
        "        raise SystemExit(f'case {_i}: the answer exited')\n"
        "    finally:\n"
        "        _signal.setitimer(_signal.ITIMER_REAL, 0)\n"
        "    _got = list(_got) if isinstance(_got, tuple) else _got\n"
        "    if not _got == _want:\n"
        "        raise SystemExit(f'case {_i}: expected {repr(_want)[:200]}, got {repr(_got)[:200]}')\n")


def stdio_harness(code: str, cases: list | None = None) -> str:
    """A Python program that runs `code`, after PY_PRELUDE, once per case with that
    case's input on sys.stdin and open(0), and fails unless its stdout matches the
    expected output the way lcb_runner's grade_stdio compares them, within
    CASE_TIMEOUT seconds. cases are [input, output] pairs; they come from CASES_FILE
    unless given. A thread the program starts is not waited for (lcb_runner does not
    either)."""
    return (
        _HEAD + _NO_EXIT + "import ast as _ast, builtins as _bi\nfrom decimal import Decimal as _Decimal\n" +
        _load_cases(cases) +
        # One module, prelude first (so a late __future__ import fails, as in
        # lcb_runner), with the answer's own line numbers in its tracebacks.
        f"_tree = _ast.parse({PY_PRELUDE!r})\n"
        f"_tree.body += _ast.parse({code!r}, 'answer.py').body\n"
        "_CODE = compile(_tree, 'answer.py', 'exec')\n"
        "_real_out, _real_open = _sys.stdout, _bi.open\n"
        "class _Stdin(_io.TextIOWrapper):\n"
        "    def fileno(self):  # so open(sys.stdin.fileno()) is open(0)\n"
        "        return 0\n"
        "def _open(file, mode='r', *args, **kwargs):\n"
        "    if file in (0, '/dev/stdin'):\n"
        "        return _io.BytesIO(_inp.encode()) if 'b' in mode else _io.StringIO(_inp)\n"
        "    return _real_open(file, mode, *args, **kwargs)\n"
        "def _lines(s):\n"
        "    return [l.strip() for l in s.strip().split('\\n')]\n"
        "def _decimals(line):\n"
        "    try:\n"
        "        return [_Decimal(t) for t in line.split()]\n"
        "    except Exception:\n"
        "        return None\n"
        "def _same(got, want):  # lcb_runner's grade_stdio\n"
        "    got, want = _lines(got), _lines(want)\n"
        "    if len(got) != len(want):\n"
        "        return False\n"
        "    for g, w in zip(got, want):\n"
        "        if g != w:\n"
        "            dg, dw = _decimals(g), _decimals(w)\n"
        "            if dg is None or dw is None or dg != dw:\n"
        "                return False\n"
        "    return True\n"
        "for _i, (_inp, _want) in enumerate(_CASES):\n"
        "    _sys.stdin = _Stdin(_io.BytesIO(_inp.encode()), encoding='utf-8')\n"
        "    _buf = _io.BytesIO()\n"
        "    _out = _io.TextIOWrapper(_buf, encoding='utf-8', write_through=True)\n"
        "    _sys.stdout, _bi.open = _out, _open\n"
        f"    _signal.setitimer(_signal.ITIMER_REAL, {CASE_TIMEOUT})\n"
        "    try:\n"
        "        exec(_CODE, {'__name__': '__main__'})\n"
        "    except SystemExit:\n"
        "        pass\n"
        "    except _CaseTimeout:\n"
        f"        raise SystemExit(f'case {{_i}}: over the {CASE_TIMEOUT:g} s time limit')\n"
        "    finally:\n"
        "        _signal.setitimer(_signal.ITIMER_REAL, 0)\n"
        "        _out.flush()\n"
        "        _sys.stdout, _bi.open = _real_out, _real_open\n"
        "    _got = _buf.getvalue().decode('utf-8', 'replace')\n"
        "    if not _same(_got, _want):\n"
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
