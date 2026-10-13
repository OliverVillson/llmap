"""Run code plus its tests in a throwaway sandbox (code TaskSpecs).

One rule for every language: the program is the answer, a blank line, then the
tests, in one file. It passes when it builds and exits 0 within the timeout.

  python      prog.py   compile(), then python3 prog.py      (tests use assert)
  javascript  prog.js   node --check, then node prog.js      (tests throw on failure)
  typescript  prog.ts   tsc --strict ... then node prog.js   (type errors are compile errors)
  c           prog.c    cc -std=c11 -O1 prog.c -lm; ./prog   (tests own main(), use assert.h)
  cpp         prog.cpp  g++ -std=c++17 -O1 prog.cpp; ./prog  (same as C; for eval suites)
  asm         prog.s + tests.c   cc prog.s tests.c; ./prog   (the one exception: x86-64 GNU as
                        answer, C tests with main() in their own file, linked together)

A run may also get data files (run_tests' files), written next to the program by
this process, so the program's file size limit does not apply to them: test cases
that would make the program too big go there (stages/codebench.py job()).

Isolation per run: a fresh temp dir, a scrubbed environment, rlimits on CPU time,
memory and file size, its own process group (killed whole on timeout), and no
network through `unshare -rn` when the kernel allows unprivileged user namespaces.
LOBBOT_SANDBOX_WRAP replaces that wrapper with your own prefix (e.g. an nsjail or
bwrap command line; "" for none).
This is a guard against buggy and runaway code, not a hardened jail for hostile
code: on the VM it runs teacher answers to our own tasks.

Knobs: LOBBOT_SANDBOX_TIMEOUT [10] seconds per build and per run,
LOBBOT_SANDBOX_MEM_MB [4096, or RAM / workers when that is less; the run's data
segment, see default_mem_mb], LOBBOT_SANDBOX_WORKERS [physical cores - 2]. Time
limits are wall clock, so runs sharing a hyperthread with another run (or with a
model server) time out sooner; one run per physical core, with two cores left for
the rest of the machine, keeps a slow but correct answer's verdict independent of
load.
Used by stages/data.py (keep only passing teacher answers) and stages/eval.py (pass@k).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache


def physical_cores() -> int:
    """Physical cores this process may run on: hyperthreads of one core count once.
    Without Linux's topology files each CPU counts as a core."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cpus = range(os.cpu_count() or 1)
    cores = set()
    for c in cpus:
        topo = f"/sys/devices/system/cpu/cpu{c}/topology"
        try:
            with open(f"{topo}/physical_package_id") as pkg, open(f"{topo}/core_id") as core:
                cores.add((pkg.read().strip(), core.read().strip()))
        except OSError:
            cores.add(("cpu", str(c)))
    return len(cores) or 1


def ram_mb() -> int | None:
    """This machine's physical memory in MB, None when the OS does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >> 20
    except (AttributeError, ValueError, OSError):
        return None


def default_mem_mb(workers: int) -> int:
    """MEM_MB when LOBBOT_SANDBOX_MEM_MB is unset: 4096, the 4 GB lcb_runner's
    reliability_guard comment names (it then passes no limit at all), so a correct
    LiveCodeBench answer that needs over 1 GB passes here as it does there; less
    when this machine's RAM split over `workers` runs at once is less, so a full
    pool of runaway programs cannot take the machine down."""
    ram = ram_mb()
    return min(MEM_CAP_MB, ram // workers) if ram else MEM_CAP_MB


TIMEOUT = float(os.environ.get("LOBBOT_SANDBOX_TIMEOUT", 10))
WORKERS = int(os.environ.get("LOBBOT_SANDBOX_WORKERS", 0)) or max(1, physical_cores() - 2)
MEM_CAP_MB = 4096
MEM_MB = int(os.environ.get("LOBBOT_SANDBOX_MEM_MB", 0)) or default_mem_mb(WORKERS)
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
# What python's build step runs: the same syntax check as py_compile (which prompts
# show, see layout), without the .pyc py_compile always writes. That file is as big
# as the program, so a program carrying a lot of data failed its build on the file
# size limit, and it left a __pycache__ in the run's dir.
_PY_CHECK = ["python3", "-c", "import sys, traceback\n"
             "try:\n"
             "    compile(open('prog.py', 'rb').read(), 'prog.py', 'exec')\n"
             "except Exception as e:\n"
             "    sys.exit(''.join(traceback.format_exception_only(type(e), e)))\n"]
# The line codebench's harnesses end with when one case runs past its own time limit.
_CASE_TIMEOUT = re.compile(r"^case \d+: over the [\d.]+ s time limit$", re.M)


@dataclass
class Result:
    passed: bool
    reason: str  # "" | compile_error | test_failed | case_timeout | timeout | unsupported_language | missing_toolchain
    output: str = ""


def program(language: str, code: str, tests: str) -> str:
    return f"{code.rstrip()}\n\n{tests.strip()}\n"


def layout(language: str) -> tuple[str, list[str], str]:
    """(file the answer is saved as, the commands that check it, where the tests are),
    for prompts that show a run the way Mugge's harness does (stages/harness.py)."""
    src, build, run = _LANG[language]
    where = (f"The tests are in {_SPLIT_TESTS[language]}, built and linked with {src}." if language in _SPLIT_TESTS
             else f"The tests are appended to {src} when these run.")
    return src, [" ".join(c) for c in (build, run) if c], where


def run_tests(language: str, code: str, tests: str, timeout: float = TIMEOUT,
              files: dict[str, bytes | str] | None = None) -> Result:
    """Build and run code plus tests. files are data the program reads, by name from
    its working dir (bytes, or str written as UTF-8); this process writes them, so
    the run's file size limit does not apply. A run that fails on one test case's
    own time limit (codebench's harnesses) gets reason case_timeout; one that runs
    past timeout, timeout."""
    if language not in _LANG:
        return Result(False, "unsupported_language", language)
    src, build, run = _LANG[language]
    for argv in (build, run):
        if argv and argv[0] != "./prog" and not shutil.which(argv[0]):
            return Result(False, "missing_toolchain", argv[0])
    if language == "python":
        build = _PY_CHECK
    with tempfile.TemporaryDirectory(prefix="lobbot-sbx-") as d:
        if language in _SPLIT_TESTS:
            sources = {src: code.rstrip() + "\n", _SPLIT_TESTS[language]: tests.strip() + "\n"}
        else:
            sources = {src: program(language, code, tests)}
        for name in files or {}:
            if name in sources or name != os.path.basename(name) or name in ("", ".", ".."):
                raise ValueError(f"data file {name!r} must be a plain file name other than {', '.join(sources)}")
        for name, data in {**sources, **(files or {})}.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(data.encode() if isinstance(data, str) else data)
        if build:
            rc, out = _exec(build, d, timeout)
            if rc is None:
                return Result(False, "timeout", out)
            if rc != 0:
                return Result(False, "compile_error", out)
        rc, out = _exec(run, d, timeout)
        if rc is None:
            return Result(False, "timeout", out)
        if rc != 0 and _CASE_TIMEOUT.search(out):
            return Result(False, "case_timeout", out)
        return Result(rc == 0, "" if rc == 0 else "test_failed", out)


def run_many(items: list[tuple], workers: int | None = None,
             timeout: float = TIMEOUT) -> list[Result]:
    """run_tests over (language, code, tests) triples in parallel; results keep input order.
    An item may add a fourth element, its own time limit, which wins over timeout when
    set, and a fifth, its data files (codebench.job() gives all five)."""
    if not items:
        return []

    def one(it: tuple) -> Result:
        limit = it[3] if len(it) > 3 and it[3] else timeout
        return run_tests(*it[:3], timeout=limit, files=it[4] if len(it) > 4 else None)

    with ThreadPoolExecutor(max_workers=workers or WORKERS) as ex:
        return list(ex.map(one, items))


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
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
        except (ValueError, OSError):  # macOS can refuse it; CPU time and file size still apply
            pass
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
    text = out.decode("utf-8", "replace").replace(d + os.sep, "")  # "prog.py", not the temp dir path
    return rc, text[-MAX_OUTPUT:]


def _kill(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, 9)  # the whole group: forks and the program under the wrapper
    except ProcessLookupError:
        pass
