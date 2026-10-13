import base64
import json
import os
import pickle
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from stages import codebench as cb
from stages import sandbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_codebench as fc  # noqa: E402

needs = lambda *tools: pytest.mark.skipif(any(not shutil.which(t) for t in tools), reason=f"needs {', '.join(tools)}")


def run(p, answer):
    return sandbox.run_many([cb.job(p, answer)])[0]


def test_pass_at_k():
    assert cb.pass_at_k(1, 1, 1) == 1.0 and cb.pass_at_k(1, 0, 1) == 0.0
    assert cb.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert cb.pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)  # C(7,5)/C(10,5)
    assert cb.pass_at_k(4, 2, 3) == 1.0  # fewer failures than k
    with pytest.raises(ValueError):
        cb.pass_at_k(1, 1, 5)


def test_extract_code_takes_the_last_complete_block():
    """As lcb_runner does: the last block, not the longest; an opening fence that is
    never closed (an answer cut off at the cap) does not count. stages/data.py takes
    code out of teacher answers the same way."""
    from stages import data

    for extract in (cb.extract_code, data.extract_code):
        assert extract("Here:\n```python\nx = 1\n```\nand\n```\ny = 22222\n```") == "y = 22222"
        assert extract("```python\nlong_answer = 1\n```\nUsage:\n```python\nf()\n```") == "f()"
        assert extract("```python\nx = 1\n```\nNow a test:\n```python\nassert f(") == "x = 1"
        assert extract("<think>```python\nscratch = 1\n```</think>```js\nf()\n```") == "f()"
        assert extract("Here it is: ```c\nint f(void) { return 1; }\n```") == "int f(void) { return 1; }"
    assert cb.extract_code("```python\n    return x\n```") == "    return x"  # a body keeps its indent
    assert cb.extract_code("def f(): pass") == "def f(): pass"
    assert cb.extract_code("```python\ndef f(): pass") == "def f(): pass"  # no complete block: the answer
    assert cb.last_block("```python\nx = 1") is None and cb.last_block("```\n```") == ""


@needs("cc")
def test_c_set_reference_solutions_pass_and_stubs_fail():
    rows = cb.load_suite("c-set", "/nonexistent")
    assert len(rows) >= 10 and all(r["language"] == "c" and r["entry"] for r in rows)
    ok = sandbox.run_many([("c", r["canonical"], r["tests"]) for r in rows])
    assert [r.reason for r in ok if not r.passed] == []
    # The canonical answer as a model reply, fenced, without the stub's types or includes.
    for r in rows:
        body = r["canonical"].split("\n", r["canonical"].count("#include"))[-1]
        assert run(r, f"```c\n{body}\n```").passed, r["id"]
    stub_only = sandbox.run_many([("c", r["stub"], r["tests"]) for r in rows])
    assert not any(r.passed for r in stub_only)


MPE_PY = {"suite": "multipl-e-py", "language": "python", "entry": "add",
          "prompt": "from typing import List\n\ndef add(xs: List[int]) -> int:\n    \"\"\"Sum.\"\"\"\n",
          "tests": "def check(candidate):\n    assert candidate([1, 2]) == 3\n\ncheck(add)\n"}
MPE_PY["stub"] = MPE_PY["prompt"]


@needs("python3")
def test_python_whole_function_body_only_and_main_block():
    assert run(MPE_PY, "```python\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n```").passed  # import comes from the stub
    assert run(MPE_PY, "    return sum(xs)\n").passed
    assert run(MPE_PY, "```python\ndef add(xs):\n    return sum(xs)\n\nif __name__ == '__main__':\n    input()\n```").passed
    assert not run(MPE_PY, "```python\ndef add(xs):\n    return 0\n```").passed


@needs("node")
def test_javascript_body_only_gets_closed():
    p = {"suite": "multipl-e-js", "language": "javascript", "entry": "add",
         "prompt": "function add(a, b){\n", "stub": "function add(a, b){\n",
         "tests": "const assert = require('node:assert');\nassert.deepEqual(add(1, 2), 3);\n"}
    assert run(p, "  return a + b;\n").passed
    assert run(p, "```javascript\nfunction add(a, b) { return a + b; }\n```").passed


@needs("g++")
def test_cpp_includes_and_main_are_handled():
    stub = "#include<assert.h>\n#include<bits/stdc++.h>\nlong add(long a, long b) {\n"
    p = {"suite": "multipl-e-cpp", "language": "cpp", "entry": "add", "prompt": stub, "stub": stub,
         "tests": "int main() {\n    auto candidate = add;\n    assert(candidate(1, 2) == 3);\n}\n"}
    code, _ = cb.assemble(p, "```cpp\nlong add(long a, long b) { return a + b; }\nint main() { return 0; }\n```")
    assert code.startswith("#include<assert.h>\n#include<bits/stdc++.h>\n") and "main" not in code
    code, _ = cb.assemble(p, "    return a + b;")
    assert code.rstrip().endswith("}") and code.count("{") == code.count("}")
    assert run(p, "```cpp\nlong add(long a, long b) { return a + b; }\nint main() { return 0; }\n```").passed
    assert run(p, "    return a + b;").passed
    assert not run(p, "    return a - b;").passed


@needs("python3")
def test_stdin_problems_run_through_a_harness():
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "Add two numbers.",
         "tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10"}]}
    assert "stdin" in cb.build_prompt(p)
    good = "```python\nimport sys\n\ndef main():\n    a, b = map(int, sys.stdin.buffer.read().split())\n    print(a + b)\n\nif __name__ == '__main__':\n    main()\n```"
    assert run(p, good).passed
    assert run(p, "```python\na, b = map(int, input().split())\nprint(a + b)\nexit(0)\n```").passed
    bad = run(p, "```python\nprint(3)\n```")
    assert not bad.passed and "case 1" in bad.output


@needs("python3")
def test_cases_travel_as_a_file_and_assemble_still_writes_them_in():
    """job() hands the cases to the sandbox as cases.json, so the program stays small;
    assemble() keeps its (code, tests) for callers that pass no files."""
    big = "9" * 100_000
    stdio = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "Echo.",
             "tests": [{"input": big + "\n", "output": big + "\n"}, {"input": "7\n", "output": "7\n"}]}
    func = {**stdio, "tests": fc.functional_tests("f", [{"input": "[1, 2]", "output": "3"}])}
    lang, code, tests, limit, files = cb.job(stdio, "```python\nprint(input())\n```")
    assert (lang, tests, limit) == ("python", "", cb.time_limit(stdio)) and len(code) < 10_000
    assert json.loads(files[cb.CASES_FILE]) == [[big + "\n", big + "\n"], ["7\n", "7\n"]]
    assert sandbox.run_many([(lang, code, tests, limit, files)])[0].passed
    assert not sandbox.run_tests(lang, code, tests).passed  # without its cases.json the harness cannot run
    assert sandbox.run_tests("python", *cb.assemble(stdio, "```python\nprint(input())\n```")).passed
    answer = "```python\ndef f(xs):\n    return sum(xs)\n```"
    assert json.loads(cb.job(func, answer)[4][cb.CASES_FILE]) == [["[1, 2]", "3"]]
    assert run(func, answer).passed and sandbox.run_tests("python", *cb.assemble(func, answer)).passed
    assert cb.job(MPE_PY, "    return sum(xs)\n")[3:] == (None, {})  # tests that are code need no files


@needs("python3")
def test_livecodebench_functional_tests():
    tests = fc.functional_tests("twoSum", [{"input": "[2, 7, 11]\n9", "output": "[0, 1]"}])
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "x", "tests": tests}
    ans = ("```python\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
           "        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen:\n"
           "                return [seen[target - x], i]\n            seen[x] = i\n```")
    assert run(p, ans).passed  # List comes from the LiveCodeBench prelude
    assert not run(p, ans.replace("[seen[target - x], i]", "[i]")).passed
    demo = ans.replace("```python\n", "```python\nimport sys\n") .replace(
        "\n```", "\n\nif __name__ == '__main__':\n    print(Solution().twoSum(*eval(sys.stdin.read())))\n```")
    assert run(p, demo).passed  # the answer's own main block is dropped; the tests are the entry point


@needs("python3")
def test_livecodebench_limits_each_case_not_the_whole_problem(monkeypatch):
    """Each case gets CASE_TIMEOUT, as in LiveCodeBench's runner: three cases of ~0.6 s
    pass under a 1 s limit even though together they take longer; a stuck case fails
    at its own limit, and an answer's `except Exception` cannot swallow it."""
    monkeypatch.setattr(cb, "CASE_TIMEOUT", 1.0)
    slow = "```python\nimport time\ntime.sleep(0.6)\nprint(int(input()) * 2)\n```"
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "Double it.",
         "tests": [{"input": f"{i}\n", "output": f"{2 * i}\n"} for i in range(3)]}
    assert cb.time_limit(p) == 13.0
    assert sandbox.run_many([("python", *cb.assemble(p, slow), cb.time_limit(p))], timeout=1.5)[0].passed
    assert sandbox.run_many([cb.job(p, slow)], timeout=1.5)[0].passed
    stuck = "```python\ntry:\n    while True:\n        pass\nexcept Exception:\n    print(0)\n```"
    r = sandbox.run_many([cb.job(p, stuck)])[0]
    assert not r.passed and r.reason == "case_timeout" and "case 0: over the 1 s time limit" in r.output
    f = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "x",
         "tests": fc.functional_tests("f", [{"input": "1", "output": "1"}])}
    spin = "```python\nclass Solution:\n    def f(self, x):\n        while True:\n            pass\n```"
    r = sandbox.run_many([cb.job(f, spin)])[0]
    assert not r.passed and r.reason == "case_timeout" and "over the 1 s time limit" in r.output


# LiveCodeBench parity: each answer as graded by lcb_runner's own run_test (LiveCodeBench
# evaluation/testing_util.py at commit 28fef95, run as its check_correctness runs it, with
# the runner's stdin at /dev/null) and by ours: (name, code, cases, func or None for a stdin
# program, lcb_runner passes, we pass). The rows where the two differ are the differences
# stages/codebench.py documents. test_lcb_runner_agrees_with_the_parity_table re-runs the
# lcb_runner column against a LiveCodeBench checkout.
ADD = [("1 2\n", "3\n"), ("5 5\n", "10\n")]
BIG = "1" + "0" * 4999  # a 5000-digit int, past Python's default 4300-digit str limit
LCB_PARITY = [
    # how a stdin program reads its input
    ("input()", "a, b = map(int, input().split())\nprint(a + b)", ADD, None, True, True),
    ("sys.stdin.readline", "import sys\na, b = map(int, sys.stdin.readline().split())\nprint(a + b)", ADD, None, True, True),
    ("sys.stdin.read", "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)", ADD, None, True, True),
    ("sys.stdin.buffer.read", "import sys\na, b = map(int, sys.stdin.buffer.read().split())\nprint(a + b)", ADD, None, True, True),
    ("sys.stdin.readlines", "import sys\nprint(sum(int(l) for l in sys.stdin.readlines() if l.strip()))",
     [("1\n2\n", "3\n")], None, True, True),
    ("input = sys.stdin.readline", "import sys\ninput = sys.stdin.readline\nprint(int(input()) + int(input()))",
     [("1\n2\n", "3\n")], None, True, True),
    ("from sys import stdin", "from sys import stdin\na, b = map(int, stdin.readline().split())\nprint(a + b)", ADD, None, True, True),
    ("bare stdin from the prelude", "a, b = map(int, stdin.readline().split())\nprint(a + b)", ADD, None, True, True),
    ("builtins.open(0)", "import builtins\na, b = map(int, builtins.open(0).read().split())\nprint(a + b)", ADD, None, True, True),
    ("open(0).read", "a, b = map(int, open(0).read().split())\nprint(a + b)", ADD, None, False, True),
    ("open(0).readline", "a, b = map(int, open(0).readline().split())\nprint(a + b)", ADD, None, False, True),
    ("open('/dev/stdin')", "a, b = map(int, open('/dev/stdin').read().split())\nprint(a + b)", ADD, None, False, True),
    ("open(sys.stdin.fileno())", "import sys\na, b = map(int, open(sys.stdin.fileno()).read().split())\nprint(a + b)",
     ADD, None, False, True),
    ("for line in sys.stdin", "import sys\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)",
     ADD, None, False, True),
    ("sys.stdin.buffer.readline twice", "import sys\nprint(int(sys.stdin.buffer.readline()) + int(sys.stdin.buffer.readline()))",
     [("1\n2\n", "3\n")], None, False, True),
    # how a stdin program is run
    ("main guard", "def main():\n    a, b = map(int, input().split())\n    print(a + b)\nif __name__ == '__main__':\n    main()",
     ADD, None, True, True),
    ("global statement", "n = 0\ndef f():\n    global n\n    n += 1\na, b = map(int, input().split())\nf()\nprint(a + b + n - 1)",
     ADD, None, False, True),
    ("sys.exit(0) after printing", "import sys\nprint(sum(map(int, input().split())))\nsys.exit(0)", ADD, None, True, True),
    ("sys.exit(1) after printing", "import sys\nprint(sum(map(int, input().split())))\nsys.exit(1)", ADD, None, True, True),
    ("exit('bye') after printing", "print(sum(map(int, input().split())))\nexit('bye')", ADD, None, True, True),
    ("no import of defaultdict", "d = defaultdict(int)\nd[0] += sum(map(int, input().split()))\nprint(d[0])", ADD, None, True, True),
    ("recursion 3000 deep", "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nprint(f(int(input())))",
     [("3000\n", "3000\n")], None, True, True),
    ("@cache recursion 3000 deep", "from functools import cache\n@cache\ndef f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n"
     "print(f(int(input())))", [("3000\n", "3000\n")], None, True, True),
    ("5000-digit output", "print(10 ** 4999)", [("", BIG + "\n")], None, True, True),
    ("late __future__ import", "from __future__ import annotations\nprint(3)", [("", "3\n")], None, False, False),
    ("threaded main, joined", "import threading\ndef main():\n    print(sum(map(int, input().split())))\n"
     "threading.stack_size(1 << 26)\nt = threading.Thread(target=main)\nt.start()\nt.join()", ADD, None, True, True),
    ("threaded main, not joined", "import threading, time\ndef main():\n    a, b = map(int, input().split())\n"
     "    time.sleep(0.3)\n    print(a + b)\nthreading.stack_size(1 << 26)\nthreading.Thread(target=main).start()",
     ADD, None, False, False),
    # how stdout is compared
    ("0.500000 for 0.5", "print('0.500000')", [("", "0.5\n")], None, True, True),
    ("1.50 for 1.5", "print('1.50')", [("", "1.5\n")], None, True, True),
    ("1e3 for 1000", "print('1e3')", [("", "1000\n")], None, True, True),
    ("-0 for 0", "print('-0')", [("", "0\n")], None, True, True),
    ("'1  2' for '1 2'", "print('1  2')", [("", "1 2\n")], None, True, True),
    ("leading space on line 2", "print('a')\nprint(' b')", [("", "a\nb\n")], None, True, True),
    ("trailing blank lines", "print(3)\nprint()\nprint()", [("", "3\n")], None, True, True),
    ("1/3 at 16 digits for 18", "print('0.3333333333333333')", [("", "0.333333333333333333\n")], None, False, False),
    ("'a  b' for 'a b'", "print('a  b')", [("", "a b\n")], None, False, False),
    ("Yes for yes", "print('Yes')", [("", "yes\n")], None, False, False),
    ("a blank line between", "print(1)\nprint()\nprint(2)", [("", "1\n2\n")], None, False, False),
    ("wrong answer", "print(4)", [("1 2\n", "3\n")], None, False, False),
    # functional (LeetCode-style) answers
    ("class Solution", "class Solution:\n    def f(self, x: List[int]) -> int:\n        return sum(x)", [("[1, 2]", "3")], "f", True, True),
    ("top-level function", "def f(x):\n    return sum(x)", [("[1, 2]", "3")], "f", True, True),
    ("bare inf, gcd, deepcopy", "class Solution:\n    def f(self, x):\n        y = deepcopy(x)\n        return gcd(y[0], y[1]) if inf > 0 else 0",
     [("[4, 6]", "2")], "f", True, True),
    ("bare comb, isqrt, ascii_lowercase", "class Solution:\n    def f(self, n):\n        return comb(n, 2) + isqrt(n) + len(ascii_lowercase)",
     [("4", "34")], "f", True, True),
    ("@cache recursion 3000 deep (functional)", "class Solution:\n    def f(self, n):\n        @cache\n        def g(k):\n"
     "            return 0 if k == 0 else 1 + g(k - 1)\n        return g(n)", [("3000", "3000")], "f", True, True),
    ("5000-digit result", "class Solution:\n    def f(self, n):\n        return 10 ** n", [("4999", BIG)], "f", True, True),
    ("tuple for a list", "class Solution:\n    def f(self, n):\n        return (n, n)", [("1", "[1, 1]")], "f", True, True),
    ("main guard (functional)", "class Solution:\n    def f(self, n):\n        return n\nif __name__ == '__main__':\n"
     "    print(Solution().f(int(input())))", [("1", "1")], "f", True, True),
    ("0.5000001 for 0.5 (functional)", "class Solution:\n    def f(self, n):\n        return 0.5000001", [("1", "0.5")], "f", False, False),
]


def lcb_problem(cases, func=None):
    tests = fc.functional_tests(func, [{"input": i, "output": o} for i, o in cases]) if func else \
        [{"input": i, "output": o} for i, o in cases]
    return {"id": "atcoder/x", "suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "x",
            "tests": tests}


@needs("python3")
def test_lcb_parity_table():
    """Our verdict on every answer in LCB_PARITY. The C22 list is in it: bare
    inf/gcd/deepcopy, @cache recursion 3000 deep, a stdin answer that never imports
    defaultdict, a 5000-digit result, 0.5 vs 0.500000 and '1  2' vs '1 2' pass; 1/3
    at 16 vs 18 digits and 'a  b' vs 'a b' still fail; open(0) and a top-level
    function pass; a threaded main is not waited for."""
    items = [cb.job(lcb_problem(cases, func), f"```python\n{code}\n```") for _, code, cases, func, _, _ in LCB_PARITY]
    wrong = {row[0]: (r.passed, r.output[-300:]) for row, r in zip(LCB_PARITY, sandbox.run_many(items))
             if r.passed != row[5]}
    assert not wrong


LCB_RUN = """
import json, multiprocessing, sys
sys.path.insert(0, sys.argv[1])
sys.set_int_max_str_digits(50000)  # as compute_code_generation_metrics.py does
from lcb_runner.evaluation.testing_util import run_test


def child(sample, code, conn):
    res, _ = run_test(sample, test=code, timeout=6)
    conn.send([r is True for r in res])  # a failed case is a negative error code


if __name__ == "__main__":
    ctx, verdicts = multiprocessing.get_context("fork"), []
    for code, io in json.load(open(sys.argv[2])):
        recv, send = ctx.Pipe(False)
        p = ctx.Process(target=child, args=({"input_output": json.dumps(io)}, code, send))
        p.start()
        got = recv.recv() if recv.poll(7 * len(io["inputs"]) + 5) else []
        p.join(1)
        if p.is_alive():
            p.kill()
        verdicts.append(bool(got) and all(got))
    print(json.dumps(verdicts))
"""


@pytest.mark.skipif(not os.environ.get("LOBBOT_LCB_RUNNER"), reason="set LOBBOT_LCB_RUNNER to a LiveCodeBench checkout")
def test_lcb_runner_agrees_with_the_parity_table(tmp_path):
    """lcb_runner's own verdicts on LCB_PARITY, so the table's lcb_runner column (and
    with it our differences) stays true for the checkout at LOBBOT_LCB_RUNNER."""
    pytest.importorskip("numpy")  # testing_util imports it
    rows = []
    for _, code, cases, func, _, _ in LCB_PARITY:
        io = {"inputs": [i for i, _ in cases], "outputs": [o for _, o in cases], **({"fn_name": func} if func else {})}
        rows.append((code, io))
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    (tmp_path / "lcb_run.py").write_text(LCB_RUN)
    p = subprocess.run([sys.executable, "-I", str(tmp_path / "lcb_run.py"), os.environ["LOBBOT_LCB_RUNNER"],
                        str(tmp_path / "rows.json")], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    verdicts = json.loads(p.stdout.strip().splitlines()[-1])
    assert dict(zip([r[0] for r in LCB_PARITY], verdicts)) == {r[0]: r[4] for r in LCB_PARITY}


@needs("python3")
def test_threaded_main_is_not_waited_for_as_in_lcb_runner():
    """A program that starts main() in a thread and returns ends its case there: the
    thread's output misses the comparison and the case fails, as under lcb_runner
    (the answers that join their thread pass). Kept for parity, not fixed."""
    p = lcb_problem([("1 2\n", "3\n")])
    main = "import threading, time\ndef main():\n    a, b = map(int, input().split())\n    time.sleep(0.3)\n    print(a + b)\n"
    r = run(p, f"```python\n{main}threading.Thread(target=main).start()\n```")
    assert not r.passed and r.reason == "test_failed" and "case 0: expected '3\\n', got ''" in r.output
    assert run(p, f"```python\n{main}t = threading.Thread(target=main)\nt.start()\nt.join()\n```").passed


@needs("python3")
def test_exits_never_skip_the_remaining_cases():
    """An exit ends a stdin program's case and its output is still compared (lcb_runner
    catches SystemExit too); os._exit counts as one, so it cannot end the harness with
    exit code 0 before every case is checked. A functional answer that exits fails."""
    p = lcb_problem(ADD)
    assert run(p, "```python\nimport os\nprint(sum(map(int, input().split())))\nos._exit(0)\n```").passed
    assert not run(p, "```python\nimport os\nos._exit(0)\n```").passed
    f = lcb_problem([("[1, 2]", "3")], "f")
    r = run(f, "```python\nimport sys\nsys.exit(0)\ndef f(x):\n    return sum(x)\n```")
    assert not r.passed and "the answer ended before its tests ran" in r.output
    r = run(f, "```python\nimport os\nos._exit(0)\ndef f(x):\n    return sum(x)\n```")
    assert not r.passed and "the answer ended before its tests ran" in r.output
    for body in ("exit()", "import os; os._exit(0)"):
        r = run(f, f"```python\nclass Solution:\n    def f(self, x):\n        {body}\n```")
        assert not r.passed and "case 0: the answer exited" in r.output


@needs("python3")
def test_open_mock_hands_out_the_case_input_only():
    """open(0) and friends read the case's input, text or bytes, fresh for each case;
    any other file opens for real."""
    p = lcb_problem([("1 2\n", "3\n"), ("5 5\n", "10\n")])
    assert run(p, "```python\nprint(sum(map(int, open(0, 'rb').read().split())))\n```").passed
    assert run(p, "```python\nwith open(0) as f:\n    print(sum(int(t) for line in f for t in line.split()))\n```").passed
    assert run(p, "```python\nimport json\njson.load(open('cases.json'))\nprint(sum(map(int, input().split())))\n```").passed
    assert not run(p, "```python\nopen('missing.txt')\nprint(sum(map(int, input().split())))\n```").passed


def test_decode_cases_refuses_pickled_objects():
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
    assert fc.decode_cases(blob) == cases
    assert fc.decode_cases(json.dumps(cases)) == cases
    evil = base64.b64encode(zlib.compress(pickle.dumps(os.getcwd))).decode()
    with pytest.raises(pickle.UnpicklingError):
        fc.decode_cases(evil)


def test_multipl_e_row_drops_the_closing_brace_tests_start_with():
    r = fc.multipl_e_row("cpp", {"name": "HumanEval_0_add", "prompt": "long add() {\n",
                                 "tests": "}\nint main() {\n    auto candidate = add;\n}\n"})
    assert r["entry"] == "add" and r["tests"].startswith("\nint main")


def test_lcb_row_shapes():
    base = {"question_content": "Q", "platform": "atcoder", "question_id": "abc1", "contest_date": "2026-06-01T00:00:00",
            "starter_code": "", "metadata": "{}", "private_test_cases": "[]"}
    r = fc.lcb_row({**base, "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "stdin"}])}, 50)
    assert r["tests"] == [{"input": "1", "output": "1"}] and r["date"] == "2026-06-01" and r["public"] == 1
    r = fc.lcb_row({**base, "starter_code": "class Solution:\n    def f(self, x):", "metadata": '{"func_name": "f"}',
                    "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "functional"}])}, 50)
    assert r["tests"] == {"func": "f", "cases": [["1", "1"]]} and "starter code" in r["prompt"]


def test_codecontests_row_keeps_examples_first_and_drops_huge_cases():
    t = lambda ins, outs: {"input": ins, "output": outs}
    r = fc.codecontests_row({"name": "1000_A. Sum", "description": "Add them.", "source": 2, "difficulty": 7,
                             "public_tests": t(["1 2\n"], ["3\n"]), "private_tests": t(["2 2\n", "x" * 30000], ["4\n", "y"]),
                             "generated_tests": t(["5 5\n"], ["10\n"])}, 50)
    assert r["id"] == "codeforces/1000_A" and r["public"] == 1 and r["prompt"] == "Add them."
    assert [c["input"] for c in r["tests"]] == ["1 2\n", "2 2\n", "5 5\n"]
    assert fc.codecontests_row({"name": "x", "description": "d", "public_tests": t(["1"], ["1"]),
                                "private_tests": t([], []), "generated_tests": t([], [])}, 50) is None


def test_load_suite_filters_by_date_and_limit(tmp_path):
    rows = [{"id": str(i), "language": "py", "prompt": "", "tests": "", "date": d}
            for i, d in enumerate(["2025-12-01", "2026-05-01", "2026-07-01"])]
    (tmp_path / "livecodebench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = cb.load_suite("livecodebench", tmp_path, since="2026-05-01")
    assert [r["id"] for r in got] == ["1", "2"] and got[0]["language"] == "python"
    assert len(cb.load_suite("livecodebench", tmp_path, limit=1)) == 1
    with pytest.raises(FileNotFoundError):
        cb.load_suite("multipl-e-js", tmp_path)


def test_heldout_rows_from_code_taskspecs():
    held = [{"input": "write f", "reference": "x", "tests": "assert f()", "language": "javascript"},
            {"input": "text task", "reference": "y"}]
    probs = cb.heldout_problems(held)
    assert len(probs) == 1 and probs[0]["language"] == "javascript" and cb.build_prompt(probs[0]) == "write f"


def test_summarize_skips_suites_without_a_toolchain_and_shares_of_ref():
    R = sandbox.Result
    probs = [{"suite": "a", "language": "c"}, {"suite": "a", "language": "c"}, {"suite": "b", "language": "cpp"}]
    res = [[R(True, ""), R(False, "test_failed")], [R(False, "timeout"), R(False, "timeout")],
           [R(False, "unsupported_language"), R(False, "unsupported_language")]]
    s = cb.summarize(probs, res, [1, 2])
    assert s["a"]["pass@1"] == 0.25 and s["a"]["pass@2"] == 0.5 and s["a"]["status"]["timeout"] == 2
    assert "pass@1" not in s["b"] and s["b"]["skipped"] == "unsupported_language"
    assert cb.mean_pass1(s) == 0.25
    share = cb.share_of_ref({"a": {"pass@1": 0.4}, "c": {"pass@1": 0.9}, "d": {"pass@1": 0.5}},
                            {"a": {"pass@1": 0.5}, "c": {"pass@1": 0.9}})
    assert share == {"suites": {"a": 0.8, "c": 1.0}, "mean": 0.9, "min": 0.8}
    ref = {"winner": "lobbot-moe", "candidates": [{"name": "ref", "code": {"suites": {"a": {"pass@1": 1}}}}]}
    assert cb.ref_suites(ref) == {"a": {"pass@1": 1}}


@pytest.mark.parametrize("variant", ["ref", "fp8", "r25q4", "r50mix", "r50mix-gen"])
def test_code10x_configs_load_with_job(tmp_path, variant):
    from stages._util import Job

    (tmp_path / "config.json").write_text((ROOT / "configs/code10x" / f"{variant}.json").read_text())
    cfg = Job(tmp_path).config
    assert cfg.teacher == "Qwen/Qwen3.6-35B-A3B" and cfg.code_eval_suites
    assert all(s == "heldout" or cb.suite_path(s, cfg.code_eval_dir).name == f"{s}.jsonl" for s in cfg.code_eval_suites)
    if variant in ("ref", "fp8"):
        assert list(cfg.eval_candidates) == [variant]
    else:
        assert "heldout" in cfg.code_eval_suites and not cfg.dense_fallback
        assert (cfg.reap_calib == "general") == (variant == "r50mix-gen") and (cfg.reap_calib_path != "") == (variant == "r50mix-gen")
    assert (variant == "ref") == (cfg.code_eval_ref == "")


def test_code10x_dry_run_reports_share_of_ref(tmp_path):
    """setup writes all five jobs; eval in dry run gives pass@1 per suite, each
    variant as a share of ref, and both the job report and the cross-variant
    table print it."""
    from lobbot.report import render

    jobs = tmp_path / "jobs"
    env = {**os.environ, "LOBBOT_DRY_RUN": "1", "LOBBOT_JOBS": str(jobs)}
    sh = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True, text=True)
    p = sh("scripts/code10x.py", "setup", "--taskspec", "examples/python-utils.code.taskspec.json")
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((jobs / "code10x-r25q4/taskspec.json").read_text())["target"]["max_size_gb"] == 16.0
    for v in ("ref", "r50mix"):
        cfg = json.loads((jobs / f"code10x-{v}/config.json").read_text())
        cfg["code_eval_dir"] = str(tmp_path / "bench")  # dry run: suites need not exist
        cfg["code_eval_suites"] = ["c-set"]
        (jobs / f"code10x-{v}/config.json").write_text(json.dumps(cfg))
    p = sh("pipeline.py", "--job", str(jobs / "code10x-ref"), "--only", "eval")
    assert p.returncode == 0, p.stdout + p.stderr
    p = sh("pipeline.py", "--job", str(jobs / "code10x-r50mix"))
    assert p.returncode == 0, p.stdout + p.stderr

    rep = json.loads((jobs / "code10x-r50mix/out/eval.json").read_text())
    assert rep["score_method"] == "pass@1"
    moe = next(c for c in rep["candidates"] if c["name"] == "lobbot-moe")
    assert moe["code"]["suites"]["c-set"]["pass@1"] == pytest.approx(0.8)
    assert moe["code"]["share_of_ref"]["mean"] == pytest.approx(1.0)

    text = "\n".join(render("code10x-r50mix", {"eval": rep, "taskspec": {}, "config": {}}, color=False, width=100))
    assert "Code tests (pass@1, share of ref)" in text and "80% (100% of ref)" in text

    p = sh("scripts/code10x.py", "report")
    assert "r50mix" in p.stdout and "pass (≥90%" in p.stdout and "not run" in p.stdout, p.stdout + p.stderr


@needs("cc")
def test_code_eval_runs_answers_and_writes_per_answer_rows(tmp_path, monkeypatch):
    """eval.code_eval against a fake server: perfect answers on one problem,
    stubs on the other; sampled pass@k when code_eval_samples > 1."""
    from stages import eval as ev
    from stages._util import Job

    job = Job(tmp_path)
    job.config.code_eval_samples, job.config.code_eval_k = 2, [1, 2]
    probs = cb.load_suite("c-set", "/nonexistent", limit=2)
    calls = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False, on_answer=None, infos=None):
        calls.append((system, temperature, seed))
        good = "```c\n" + probs[0]["canonical"] + "\n```"
        return [good if seed == 0 else "```c\nint nope;\n```", "no code here"], 50.0

    monkeypatch.setattr(ev, "generate", fake_generate)
    out = ev.code_eval(job, "cand", probs, "task system")
    s = out["suites"]["c-set"]
    assert s["n"] == 2 and s["pass@1"] == 0.25 and s["pass@2"] == 0.5
    assert out["tok_s_vm"] == 50.0 and out["samples_per_problem"] == 2
    assert {c[0] for c in calls} == {cb.SYSTEM} and {c[2] for c in calls} == {0, 1}
    rows = [json.loads(l) for l in (tmp_path / "work/code_eval/cand.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and sum(r["passed"] for r in rows) == 1


@needs("cc")
def test_code_eval_fix_turn_gets_the_failure_and_scores_fix_at_1(tmp_path, monkeypatch):
    """With code_eval_fix, a failed greedy answer gets one more call that shows the
    code and the failing output; fix@1 counts first passes plus fixes."""
    from stages import eval as ev
    from stages._util import Job

    job = Job(tmp_path)
    job.config.code_eval_fix = True
    probs = cb.load_suite("c-set", "/nonexistent", limit=3)
    good = lambda p: "```c\n" + p["canonical"] + "\n```"
    calls = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False, on_answer=None, infos=None):
        calls.append(prompts)
        for _ in prompts:
            on_answer()
        if len(calls) == 1:  # first answers: 0 passes, 1 and 2 do not build
            return [good(probs[0]), "```c\nint broken(\n```", "```c\nint nope;\n```"], None
        return [good(probs[1]), "```c\nint still_nope;\n```"], None  # the fix turn

    events = []
    monkeypatch.setattr(ev, "generate", fake_generate)
    monkeypatch.setattr(ev, "emit", lambda stage, **kw: events.append(kw))
    out = ev.code_eval(job, "cand", probs, "task system")
    s = out["suites"]["c-set"]
    assert s["pass@1"] == round(1 / 3, 4) and s["fix@1"] == round(2 / 3, 4) and s["fixed"] == 1
    assert out["mean_fix@1"] == s["fix@1"]
    fix_prompts = calls[1]
    assert len(fix_prompts) == 2 and "int broken(" in fix_prompts[0] and "It did not build:" in fix_prompts[0]
    assert "## Fix" in fix_prompts[1] and "int nope;" in fix_prompts[1]
    failed = type("R", (), {"reason": "test_failed", "output": "x" * 5000})()
    text = cb.fix_prompt(probs[0], "```c\nint f;\n```", failed)
    assert "It failed its tests:\n…" + "x" * cb.FIX_OUTPUT_CHARS in text and "x" * 2001 not in text
    rows = [json.loads(l) for l in (tmp_path / "work/code_eval/cand.jsonl").read_text().splitlines()]
    assert [r.get("fix_passed") for r in rows] == [None, True, False]
    assert [(e["answered"], e["total"]) for e in events if "answered" in e][-1] == (5, 5)


def test_code_eval_thinking_samples_and_asks_for_thinking(tmp_path, monkeypatch):
    """Thinking loops under greedy decoding, so a thinking eval samples at 0.6 with
    a fixed seed per sample, one sample or several, and passes thinking through."""
    from stages import eval as ev
    from stages._util import Job

    job = Job(tmp_path)
    job.config.code_eval_thinking, job.config.code_eval_max_tokens = True, 24576
    probs = cb.load_suite("c-set", "/nonexistent", limit=1)
    calls = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False, on_answer=None, infos=None):
        calls.append((max_tokens, temperature, seed, thinking))
        on_answer()
        return [""], None

    events = []
    monkeypatch.setattr(ev, "generate", fake_generate)
    monkeypatch.setattr(ev, "emit", lambda stage, **kw: events.append(kw))
    ev.code_eval(job, "cand", probs, "task system")
    assert calls == [(24576, 0.6, 0, True)]
    # Progress for the Mac runner's bar: 0 of 1 before answering, 1 of 1 after.
    assert [(e["answered"], e["total"]) for e in events if "answered" in e] == [(0, 1), (1, 1)]
    calls.clear()
    job.config.code_eval_samples = 2
    ev.code_eval(job, "cand", probs, "task system")
    assert sorted(calls) == [(24576, 0.6, 0, True), (24576, 0.6, 1, True)]


def test_examples_only_and_pick_at_examples():
    """A sample that passes the examples shown in the problem is the one handed in;
    with none, the first sample is."""
    R = sandbox.Result
    stdio = {"suite": "livecodebench", "language": "python", "public": 1,
             "tests": [{"input": "1", "output": "1"}, {"input": "2", "output": "2"}]}
    func = {**stdio, "tests": {"func": "f", "cases": [["1", "1"], ["2", "2"]]}}
    assert cb.examples_only(stdio)["tests"] == [{"input": "1", "output": "1"}]
    assert cb.examples_only(func)["tests"] == {"func": "f", "cases": [["1", "1"]]}
    probs = [stdio, func, {**stdio, "public": 0}]
    results = [[R(False, "test_failed"), R(True, "")], [R(False, "test_failed"), R(True, "")],
               [R(True, ""), R(True, "")]]
    examples = [[False, True], [False, False], []]
    suites = {"livecodebench": {"pass@1": 0.5}}
    cb.add_pick_scores(suites, probs, results, examples)
    assert suites["livecodebench"]["pick@examples"] == 0.5  # 1 of the 2 problems with examples


def test_generate_sends_thinking_flag(monkeypatch):
    import httpx
    from stages import eval as ev

    sent = []

    class R:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "```py\nx = 1\n```"}}]}

    monkeypatch.setattr(httpx, "post", lambda url, timeout, json: sent.append((timeout, json)) or R())
    ev.generate("sys", ["a"], 24576, 0.6, 0, thinking=True)
    ev.generate("sys", ["b"], 2048)
    (t1, on), (t2, off) = sent
    assert on["chat_template_kwargs"] == {"enable_thinking": True} and on["top_k"] == 20 and t1 >= 24576 / 4
    assert off["chat_template_kwargs"] == {"enable_thinking": False} and "top_k" not in off and t2 == 600


def test_eval_context_fits_code_answer_cap(monkeypatch):
    from stages import eval as ev
    from stages._util import Config

    monkeypatch.delenv("LOBBOT_EVAL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LOBBOT_EVAL_CTX", raising=False)
    cfg = Config(code_eval_suites=["livecodebench"], code_eval_max_tokens=24576, data_answer_max_tokens=2048)
    assert ev.limits(cfg) == (2048, 24576 + 4096)
    cfg.code_eval_thinking = True  # room for the answer after a thinking that used it all
    assert ev.limits(cfg) == (2048, 24576 + ev.ANSWER_TOKENS + 4096)


def write_gguf_header(path, kvs):
    """A GGUF with only a header: kvs is [(key, value type, packed value)]."""
    import struct

    s = lambda b: struct.pack("<Q", len(b)) + b
    body = b"".join(s(k.encode()) + struct.pack("<I", t) + v for k, t, v in kvs)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body)


def test_experts_override_names_the_files_architecture(tmp_path):
    import struct

    from stages import eval as ev
    from stages._util import Config

    gguf = tmp_path / "m.gguf"
    s = lambda b: struct.pack("<Q", len(b)) + b
    write_gguf_header(gguf, [
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("general.tags", 9, struct.pack("<IQ", 8, 2) + s(b"a") + s(b"bc")),  # arrays are skipped too
        ("general.architecture", 8, s(b"qwen35moe")),
    ])
    assert ev.gguf_arch(str(gguf)) == "qwen35moe"
    assert "--override-kv" not in ev.server_args(Config(), str(gguf))
    args = ev.server_args(Config(eval_experts_used=6), str(gguf))
    assert args[args.index("--override-kv") + 1] == "qwen35moe.expert_used_count=int:6"


def test_split_reasoning_handles_split_and_inline_tags():
    from stages.eval import split_reasoning

    assert split_reasoning({"content": "x = 1", "reasoning_content": "hmm"}) == ("hmm", "x = 1")
    assert split_reasoning({"content": "<think>hmm</think>\n\nx = 1"}) == ("hmm", "x = 1")
    assert split_reasoning({"content": "hmm</think>x = 1"}) == ("hmm", "x = 1")  # template opened it
    assert split_reasoning({"content": "<think>hmm, still going"}) == ("hmm, still going", "")
    assert split_reasoning({"content": None, "reasoning_content": "hmm"}) == ("hmm", "")
    assert split_reasoning({"content": "", "reasoning_content": "hmm</think>\n\nx = 1"}) == ("hmm", "x = 1")


def test_thinking_without_an_answer_is_closed_and_answered(monkeypatch):
    """A thinking that ends with no answer gets </think> appended (plus Qwen's
    early-stop line when it ran out of budget) and the model answers on top."""
    import httpx
    from stages import eval as ev

    finish = {"a": "stop", "b": "length"}
    sent = []

    class R:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body

    def post(url, timeout, json):
        sent.append((url.rsplit("/", 1)[1], json))
        if url.endswith("/chat/completions"):
            return R({"choices": [{"finish_reason": finish[json["messages"][1]["content"]],
                                   "message": {"content": "", "reasoning_content": "plan it"}}],
                      "usage": {"completion_tokens": 900}})
        if url.endswith("/apply-template"):
            assert json["chat_template_kwargs"] == {"enable_thinking": True}
            return R({"prompt": "<|im_start|>assistant\n<think>\n"})
        return R({"content": "```py\nx = 1\n```", "tokens_predicted": 12})

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(ev, "SLOTS", 1)
    infos = []
    answers, _ = ev.generate("sys", ["a", "b"], 24576, 0.6, 0, thinking=True, infos=infos)
    assert answers == ["```py\nx = 1\n```"] * 2
    assert [(i["forced"], i["tokens"], i["forced_tokens"]) for i in infos] == [("stop", 900, 12), ("length", 900, 12)]
    stop, length = [j["prompt"] for name, j in sent if name == "completion"]
    assert stop == "<|im_start|>assistant\n<think>\nplan it\n</think>\n\n"
    assert length == "<|im_start|>assistant\n<think>\nplan it" + ev.EARLY_STOP + "\n</think>\n\n"


def test_thinking_off_never_forces_an_answer(monkeypatch):
    import httpx
    from stages import eval as ev

    class R:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}

    urls = []
    monkeypatch.setattr(httpx, "post", lambda url, timeout, json: urls.append(url) or R())
    infos = []
    assert ev.generate("sys", ["a"], 4096, infos=infos)[0] == [""]
    assert len(urls) == 1 and infos[0]["forced"] is None and infos[0]["finish"] == "length"
