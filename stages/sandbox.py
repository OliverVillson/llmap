"""Run code plus its tests in a throwaway sandbox (code TaskSpecs).

One rule for every language: the program is the answer, a blank line, then the
tests, in one file. It passes when it builds and exits 0 within the timeout.

  python      prog.py   py_compile, then python3 prog.py     (tests use assert)
  javascript  prog.js   node --check, then node prog.js      (tests throw on failure)
  typescript  prog.ts   tsc --strict ... then node prog.js   (type errors are compile errors)
  c           prog.c    cc -std=c11 -O1 prog.c -lm; ./prog   (tests own main(), use assert.h)
  cpp         prog.cpp  g++ -std=c++17 -O1 prog.cpp; ./prog  (same as C; for eval suites)
  asm         prog.s + tests.c   cc prog.s tests.c; ./prog   (the one exception: x86-64 GNU as
                        answer, C tests with main() in their own file, linked together)

Isolation per run: a fresh temp dir, a scrubbed environment, rlimits on CPU time,
memory and file size, its own process group (killed whole on timeout), and no
network through `unshare -rn` when the kernel allows unprivileged user namespaces.
LOBBOT_SANDBOX_WRAP replaces that wrapper with your own prefix (e.g. an nsjail or
bwrap command line; "" for none).
This is a guard against buggy and runaway code, not a hardened jail for hostile
code: on the VM it runs teacher answers to our own tasks.

Knobs: LOBBOT_SANDBOX_TIMEOUT [10] seconds per build and per run,
LOBBOT_SANDBOX_MEM_MB [1024], LOBBOT_SANDBOX_WORKERS [cpu count].
Used by stages/data.py (keep only passing teacher answers) and stages/eval.py (pass@k).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

TIMEOUT = float(os.environ.get("LOBBOT_SANDBOX_TIMEOUT", 10))
MEM_MB = int(os.environ.get("LOBBOT_SANDBOX_MEM_MB", 1024))
WORKERS = int(os.environ.get("LOBBOT_SANDBOX_WORKERS", 0)) or os.cpu_count() or 4
MAX_OUTPUT = 4000

# language -> (source file, build argv or None, run argv)
_LANG = {
    "python": ("prog.py", ["python3", "-m", "py_compile", "prog.py"], ["python3", "prog.py"]),
    "javascript": ("prog.js", ["node", "--check", "prog.js"], ["node", "prog.js"]),
    "typescript": ("prog.ts", ["tsc", "--strict", "--target", "es2022", "--module", "commonjs",
                               "--lib", "es2022,dom", "--skipLibCheck", "prog.ts"],
                   ["node", "prog.js"]),
    "c": ("prog.c", ["cc", "-std=c11", "-O1", "-o", "prog", "prog.c", "-lm"], ["./prog"]),
    # eval only (MultiPL-E C++); TaskSpec languages stay the four above
    "cpp": ("prog.cpp", ["g++", "-std=c++17", "-O1", "-o", "prog", "prog.cpp"], ["./prog"]),
    "asm": ("prog.s", ["cc", "-std=c11", "-O1", "-Wa,--noexecstack", "-o", "prog", "prog.s", "tests.c", "-lm"],
            ["./prog"]),
}
# Languages whose tests go in their own file (name) instead of after the answer.
_SPLIT_TESTS = {"asm": "tests.c"}
LANGUAGES = tuple(_LANG)


@dataclass
class Result:
    passed: bool
    reason: str  # "" | compile_error | test_failed | timeout | unsupported_language | missing_toolchain
    output: str = ""


def program(language: str, code: str, tests: str) -> str:
    return f"{code.rstrip()}\n\n{tests.strip()}\n"


def run_tests(language: str, code: str, tests: str, timeout: float = TIMEOUT) -> Result:
    if language not in _LANG:
        return Result(False, "unsupported_language", language)
    src, build, run = _LANG[language]
    for argv in (build, run):
        if argv and argv[0] != "./prog" and not shutil.which(argv[0]):
            return Result(False, "missing_toolchain", argv[0])
    with tempfile.TemporaryDirectory(prefix="lobbot-sbx-") as d:
        if language in _SPLIT_TESTS:
            files = {src: code.rstrip() + "\n", _SPLIT_TESTS[language]: tests.strip() + "\n"}
        else:
            files = {src: program(language, code, tests)}
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        if build:
            rc, out = _exec(build, d, timeout)
            if rc is None:
                return Result(False, "timeout", out)
            if rc != 0:
                return Result(False, "compile_error", out)
        rc, out = _exec(run, d, timeout)
        if rc is None:
            return Result(False, "timeout", out)
        return Result(rc == 0, "" if rc == 0 else "test_failed", out)


def run_many(items: list[tuple[str, str, str]], workers: int | None = None,
             timeout: float = TIMEOUT) -> list[Result]:
    """run_tests over (language, code, tests) triples in parallel; results keep input order."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers or WORKERS) as ex:
        return list(ex.map(lambda it: run_tests(*it, timeout=timeout), items))


# --------------------------------------------------------------------------- process isolation

@lru_cache(maxsize=1)
def _wrapper() -> tuple[str, ...]:
    custom = os.environ.get("LOBBOT_SANDBOX_WRAP")
    if custom is not None:
        return tuple(shlex.split(custom))
    if shutil.which("unshare"):
        try:  # unprivileged user + network namespace: loopback only, no outside network
            ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return ("unshare", "-rn")
    return ()


def _env(d: str) -> dict:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": d, "TMPDIR": d,
            "LANG": "C.UTF-8", "NODE_OPTIONS": "", "PYTHONDONTWRITEBYTECODE": "1"}


def _limits(timeout: float):
    def apply():
        import resource
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        # Node and V8 reserve large virtual ranges up front, so RLIMIT_AS would kill them;
        # cap the data segment instead, which still stops runaway heap growth.
        mem = MEM_MB << 20
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
        os.setsid()
    return apply


def _exec(argv: list[str], d: str, timeout: float) -> tuple[int | None, str]:
    """(exit code, output tail); exit code None on timeout."""
    p = subprocess.Popen([*_wrapper(), *argv], cwd=d, env=_env(d), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=_limits(timeout))
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        _kill(p)
        out, _ = p.communicate()
        rc = None
    text = out.decode("utf-8", "replace")
    return rc, text[-MAX_OUTPUT:]


def _kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, 9)  # the whole group: forks and the program under the wrapper
    except ProcessLookupError:
        pass
