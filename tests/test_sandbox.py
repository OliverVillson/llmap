import shutil

import pytest

from stages import sandbox

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
    monkeypatch.setenv("SECRET_TOKEN", "x")
    code = "import os\nassert 'SECRET_TOKEN' not in os.environ\nassert set(os.listdir('.')) <= {'prog.py', '__pycache__'}"
    assert sandbox.run_tests("python", code, "").passed


@pytest.mark.skipif(sandbox._wrapper() != ("unshare", "-rn"), reason="no unprivileged network namespace here")
def test_no_network():
    code = "import socket\ns = socket.socket()\ns.settimeout(2)\ntry:\n    s.connect(('1.1.1.1', 80))\nexcept OSError:\n    pass\nelse:\n    raise SystemExit('network reachable')"
    assert sandbox.run_tests("python", code, "").passed


def test_unknown_language():
    assert sandbox.run_tests("rust", "", "").reason == "unsupported_language"


def test_run_many_keeps_order():
    items = [("python", f"x = {i}", f"assert x == {i if i % 2 else -1}") for i in range(6)]
    assert [r.passed for r in sandbox.run_many(items, workers=3)] == [False, True] * 3
