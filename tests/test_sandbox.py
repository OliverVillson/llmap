import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stages import codebench, sandbox

ROOT = Path(__file__).resolve().parents[1]

C_TESTS = "#include <assert.h>\nint main(void) { assert(add(2, 3) == 5); return 0; }"
TS_TESTS = "if (add(2, 3) !== 5) throw new Error('add');"
JS_TESTS = "if (add(2, 3) !== 5) throw new Error('add');"

# language, passing answer, answer that fails the tests, answer that does not build, tests
CASES = [
    ("python", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a - b",
     "def add(a, b)\n    return a + b", "assert add(2, 3) == 5"),
    ("javascript", "function add(a, b) { return a + b; }", "function add(a, b) { return a - b; }",
     "function add(a, b) { return a + b;", JS_TESTS),
    ("typescript", "function add(a: number, b: number): number { return a + b; }",
     "function add(a: number, b: number): number { return a - b; }",
     "function add(a: number, b: number): string { return a + b; }", TS_TESTS),
    ("c", "int add(int a, int b) { return a + b; }", "int add(int a, int b) { return a - b; }",
     "int add(int a, int b) { return a + b }", C_TESTS),
    ("cpp", "#include <string>\nint add(int a, int b) { return a + b; }",
     "int add(int a, int b) { return a - b; }", "int add(int a, int b) { return a + b }",
     "#include <cassert>\nint main() { assert(add(2, 3) == 5); return 0; }"),
]
TOOL = {"python": "python3", "javascript": "node", "typescript": "tsc", "c": "cc", "cpp": "g++"}


def need(lang):
    if not shutil.which(TOOL[lang]):
        pytest.skip(f"{TOOL[lang]} not installed")


@pytest.mark.parametrize("lang,good,wrong,broken,tests", CASES, ids=[c[0] for c in CASES])
def test_pass_fail_and_build_error(lang, good, wrong, broken, tests):
    need(lang)
    ok = sandbox.run_tests(lang, good, tests)
    assert ok.passed and ok.reason == "", ok.output
    bad = sandbox.run_tests(lang, wrong, tests)
    assert not bad.passed and bad.reason == "test_failed"
    assert sandbox.run_tests(lang, broken, tests).reason == "compile_error"


def test_c_answer_without_tested_function_fails_to_build():
    need("c")
    assert sandbox.run_tests("c", "int sub(int a, int b) { return a - b; }", C_TESTS).reason == "compile_error"


def test_timeout_kills_runaway_code():
    r = sandbox.run_tests("python", "while True:\n    pass", "", timeout=1)
    assert not r.passed and r.reason == "timeout"


def test_timeout_kills_child_processes():
    code = "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nwhile True:\n    pass"
    assert sandbox.run_tests("python", code, "", timeout=1).reason == "timeout"


def test_runs_in_a_fresh_dir_with_a_scrubbed_env(monkeypatch):
    """The dir holds the program alone: the build check writes no .pyc (py_compile
    would, as big as the program, under the run's file size limit)."""
    monkeypatch.setenv("SECRET_TOKEN", "x")
    code = "import os\nassert 'SECRET_TOKEN' not in os.environ\nassert os.listdir('.') == ['prog.py'], os.listdir('.')"
    assert sandbox.run_tests("python", code, "").passed


def test_data_files_sit_next_to_the_program():
    code = "assert sorted(__import__('os').listdir('.')) == ['a.bin', 'b.txt', 'prog.py']\n" \
           "assert open('a.bin', 'rb').read() == b'\\x00\\xff'\nassert open('b.txt', encoding='utf-8').read() == 'é'"
    assert sandbox.run_tests("python", code, "", files={"a.bin": b"\x00\xff", "b.txt": "é"}).passed
    for bad in ("prog.py", "../x", "sub/x", ""):
        with pytest.raises(ValueError):
            sandbox.run_tests("python", "pass", "", files={bad: "x"})


def test_70_mb_of_cases_passes():
    """A problem with 70 MB of test cases (36 AtCoder-size cases of 2e5 numbers): the
    cases go in cases.json, which this process writes, so neither the run's 64 MiB
    file size limit nor the build sees them. Written into the program, they made
    every answer a compile_error."""
    rng = random.Random(0)
    line = " ".join(str(rng.randrange(10 ** 9)) for _ in range(200_000))
    cases = [{"input": f"{line} {i}\n", "output": f"{sum(map(int, line.split())) + i}\n"} for i in range(36)]
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "Sum.", "tests": cases}
    item = codebench.job(p, "```python\nimport sys\nprint(sum(map(int, sys.stdin.buffer.read().split())))\n```")
    assert len(item[4][codebench.CASES_FILE]) > 70e6 and len(item[1]) < 10_000
    r = sandbox.run_many([item])[0]
    assert r.passed, r.output


def test_case_timeout_is_its_own_reason():
    """A harness that stops on one case's own limit says so on its last line; any
    other failure stays test_failed, and running past the whole limit is timeout."""
    out = "raise SystemExit('case 3: over the 6 s time limit')"
    assert sandbox.run_tests("python", out, "").reason == "case_timeout"
    assert sandbox.run_tests("python", "raise SystemExit('case 3: expected 1, got 2')", "").reason == "test_failed"
    assert sandbox.run_tests("python", "print('case 3: over the 6 s time limit')", "").passed


@pytest.mark.skipif(sandbox._wrapper() != ("unshare", "-rn"), reason="no unprivileged network namespace here")
def test_no_network():
    code = "import socket\ns = socket.socket()\ns.settimeout(2)\ntry:\n    s.connect(('1.1.1.1', 80))\nexcept OSError:\n    pass\nelse:\n    raise SystemExit('network reachable')"
    assert sandbox.run_tests("python", code, "").passed


def test_unknown_language():
    assert sandbox.run_tests("rust", "", "").reason == "unsupported_language"


def test_run_many_keeps_order():
    items = [("python", f"x = {i}", f"assert x == {i if i % 2 else -1}") for i in range(6)]
    assert [r.passed for r in sandbox.run_many(items, workers=3)] == [False, True] * 3


def test_run_many_takes_time_limits_and_files():
    items = [("python", "import time\ntime.sleep(2)", "", None), ("python", "import time\ntime.sleep(2)", "", 5),
             ("python", "assert open('d').read() == 'x'", "", None, {"d": "x"}), ("python", "pass", "", 5, None)]
    assert [r.reason for r in sandbox.run_many(items, timeout=1)] == ["timeout", "", "", ""]


def test_workers_default_to_physical_cores_minus_two():
    cores = sandbox.physical_cores()
    assert 1 <= cores <= (os.cpu_count() or 1)
    env = {k: v for k, v in os.environ.items() if k != "LOBBOT_SANDBOX_WORKERS"}
    get = lambda env: subprocess.run([sys.executable, "-c", "from stages import sandbox; print(sandbox.WORKERS)"],
                                     cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert get(env) == str(max(1, cores - 2))
    assert get({**env, "LOBBOT_SANDBOX_WORKERS": "7"}) == "7"


ASM_OK = ".intel_syntax noprefix\n    .text\n    .globl add\nadd:\n    lea eax, [rdi + rsi]\n    ret\n"
ASM_TESTS = "#include <assert.h>\nint add(int, int);\nint main(void) { assert(add(2, 3) == 5); return 0; }"


def test_asm_links_against_c_tests():
    need("c")
    assert sandbox.run_tests("asm", ASM_OK, ASM_TESTS).passed
    assert sandbox.run_tests("asm", ASM_OK.replace("lea eax, [rdi + rsi]", "mov eax, edi"), ASM_TESTS).reason == "test_failed"
    assert sandbox.run_tests("asm", "add:\n    bogus eax\n", ASM_TESTS).reason == "compile_error"
