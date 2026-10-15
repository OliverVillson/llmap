"""Running the train harness's commands: each through bash with .env.vm sourced and its
output in the step's log (runner.sh, plus a process handle the GPU watchdog can stop),
or, in a dry run, checked for bash syntax and handed to the mock that writes what the
real command would (scripts/bh/mock.py). The bucket is evroc, or a local folder.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from runner import REPO, Step, on_line  # puts the repo on sys.path

from common.progress import parse  # noqa: E402


@dataclass
class Cmd:
    """One command of a stage: the shell line, the kind of command it is (the mock's
    key), and what it is about (paths and settings the mock reads; never the shell)."""
    sh: str
    kind: str
    args: dict = field(default_factory=dict)
    gpu: bool = False  # a long GPU step: the watchdog checks the GPU works in its first minutes
    env: dict = field(default_factory=dict)
    # gpu steps that read on the CPU first (sft.py tokenizes every row): the watch window
    # starts at the command's first progress event instead of its launch
    watch_from_event: bool = False
    timeout_min: float = 0.0  # stopped (StepTimeout) after this many minutes; 0: no limit


class StepTimeout(RuntimeError):
    """A command stopped at its time limit (the session clock's share for it)."""


class GpuIdle(RuntimeError):
    """A long GPU step whose GPU stayed idle for the watch window."""


WATCH_EVERY = 15.0  # seconds between the GPU watchdog's readings


def gpu_reading() -> tuple[int, int] | None:
    """(utilization %, memory used MiB) of GPU 0, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        u, m = out.splitlines()[0].split(",")
        return int(u), int(m)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Exec:
    def __init__(self, cfg, dry: bool = False, mock: Callable[["Cmd"], None] | None = None,
                 reading: Callable[[], tuple[int, int] | None] = gpu_reading):
        self.cfg, self.dry, self.mock, self.reading = cfg, dry, mock, reading
        self.commands: list[Cmd] = []  # every command run, in order (the dry run's record)
        self.notes: list[str] = []  # guard warnings, for the summary
        self.on_warn = None  # called with each warning (bh/live.py keeps them for watch)

    def warn(self, msg: str) -> None:
        line = f"WARNING: {msg}"
        self.notes.append(msg)
        print(line, flush=True)
        if self.on_warn:
            self.on_warn(msg)

    def run(self, step: Step | None, cmd: Cmd) -> None:
        self.commands.append(cmd)
        if self.dry:
            r = subprocess.run(["bash", "-n", "-c", cmd.sh], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"not valid bash: {cmd.sh}\n{r.stderr}")
            if step and step.log:
                with open(step.log, "a") as log:
                    log.write(f"\n===== {time.strftime('%F %T')} (dry run) $ {cmd.sh}\n")
            if self.mock:
                self.mock(cmd)
            return
        self._sh(step, cmd)

    def _sh(self, step: Step | None, cmd: Cmd) -> None:
        full = f"cd {shlex.quote(str(REPO))} && {{ [ -f .env.vm ] && source .env.vm; true; }} && {cmd.sh}"
        log = open(step.log, "a") if step and step.log else open(os.devnull, "w")
        with log:
            log.write(f"\n===== {time.strftime('%F %T')} $ {cmd.sh}\n")
            log.flush()
            proc = subprocess.Popen(["bash", "-c", full], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, errors="replace", env={**os.environ, **cmd.env}, start_new_session=True)
            idle = threading.Event()
            stop = threading.Event()
            armed = threading.Event()
            if not cmd.watch_from_event:
                armed.set()
            if cmd.gpu:
                threading.Thread(target=self._watch, args=(proc, idle, stop, armed, cmd.kind), daemon=True).start()
            timer, fired = None, threading.Event()
            if cmd.timeout_min > 0:
                timer = threading.Timer(cmd.timeout_min * 60, lambda: (fired.set(), _killpg(proc)))
                timer.daemon = True
                timer.start()
            tail: list[str] = []
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if step:
                        on_line(step, line)
                    if not armed.is_set() and parse(line) is not None:
                        armed.set()
                    if line.strip():
                        tail = (tail + [line.rstrip()])[-15:]
                code = proc.wait()
            except BaseException:
                _killpg(proc)
                raise
            finally:
                stop.set()
                if timer:
                    timer.cancel()
        if fired.is_set() and code != 0 and not idle.is_set():
            raise StepTimeout(f"{cmd.kind}: stopped at its {cmd.timeout_min:.1f}-minute limit")
        if idle.is_set():
            raise GpuIdle(f"{cmd.kind}: the GPU stayed idle for the first {self.cfg.gpu_watch_minutes:g} minutes "
                          f"(utilization under 5% the whole time), so it was stopped; is it running on the CPU?")
        if code != 0:
            raise RuntimeError(f"`{cmd.sh}` failed:\n" + "\n".join("    " + t for t in tail[-8:]))

    def _watch(self, proc: subprocess.Popen, idle: threading.Event, stop: threading.Event, armed: threading.Event,
               what: str) -> None:
        """The GPU-busy check: readings every 15 s over the step's first minutes (from its
        first progress event when it reads on the CPU first); if every one shows the GPU
        under 5% busy, warn, and stop the step when gpu_idle_abort."""
        while not armed.wait(min(5.0, WATCH_EVERY)):
            if stop.is_set():
                return
        window, seen = self.cfg.gpu_watch_minutes * 60, []
        t0 = time.monotonic()
        while time.monotonic() - t0 < window:
            if stop.wait(WATCH_EVERY):
                return
            r = self.reading()
            if r is None:
                return  # no nvidia-smi: nothing to judge by
            seen.append(r[0])
        if seen and max(seen) < 5:
            self.warn(f"{what}: GPU under 5% busy for {len(seen)} readings over {window / 60:g} min")
            if self.cfg.gpu_idle_abort:
                idle.set()
                _killpg(proc)


def _killpg(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=30)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


# ---------- the bucket ----------

class Bucket:
    """bucket://mugge-library/blackhole/<run>/<name> through the evroc CLI, or a local
    folder (BH_BUCKET=/some/dir: the dry run, or a laptop without evroc)."""

    def __init__(self, root: str, run: str):
        self.root, self.run = root.rstrip("/"), run
        self.local = not root.startswith("bucket://")

    def key(self, name: str) -> str:
        return f"{self.root}/blackhole/{self.run}/{name}"

    def ok(self) -> bool:
        if self.local:
            return True
        from runner import evroc_ok
        return evroc_ok()

    def put(self, src: Path, name: str) -> bool:
        if self.local:
            dst = Path(self.key(name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            tmp.replace(dst)
            return True
        return _evroc(str(src), self.key(name), tries=3)

    def get(self, name: str, dst: Path) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        if self.local:
            src = Path(self.key(name))
            if not src.exists():
                return False
            shutil.copyfile(src, tmp)
        elif not _evroc(self.key(name), str(tmp), tries=1) or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        return True


def _evroc(src: str, dst: str, tries: int) -> bool:
    if not shutil.which("evroc"):
        return False
    for k in range(tries):
        ok = all(subprocess.run(["evroc", "storage", "bucket", *a], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                check=False).returncode == 0
                 for a in (["get-s3-credentials"], ["copy", "--from", src, "--to", dst]))
        if ok:
            return True
        time.sleep(5 * (k + 1))
    return False


# ---------- small file helpers ----------

def load(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def dump(p: Path, d) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def read_jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def write_jsonl(p: Path, rows) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
