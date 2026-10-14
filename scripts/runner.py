"""The one-command experiment runner: steps in order, a progress bar with the time
left, logs per step, and resume (a step whose done() holds is skipped). Used by
the scripts/exp*.py and scripts/blackhole*.py runs; stdlib only, for the system python3.

run_plan takes an optional upload hook, called after every step that finishes and
once more however the run ends (a failed step, Ctrl-C, or the last step done), so
results reach the bucket as they are made: the NVMe they are written to is wiped
when the VM stops, and a run that breaks at step 6 used to leave steps 1-5 there only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.progress import parse  # noqa: E402

WIDTH = 30
TTY = sys.stdout.isatty()


class Skip(Exception):
    """A step that cannot run now; the rest of the run goes on."""


class Done(Exception):
    """A step that was already done when it started (a restore before it finished it).
    It counts as done, and its ~0 s are left out of the speed that scales the time left:
    counted, a restored 4 h reading once halved every guess after it."""


MIN_TIMED_S = 30  # a step that took less says nothing about the guesses (it found its work done)


@dataclass
class Step:
    name: str
    minutes: dict[str, float]  # time guess per part: pipeline stages, or {"": n}
    run: Callable[["Step"], None]
    done: Callable[[], bool] = lambda: False
    progress: Callable[[], float | None] = lambda: None  # for steps that send no events
    # The time guess again (minutes' shape), read when a step starts: a guess an earlier step
    # re-bases (a measured rate, say) reaches the bar of the run that measured it.
    refresh: Callable[[], dict[str, float]] | None = None
    # live state
    state: str = "waiting"  # waiting, running, done, skipped, failed, stopped (Ctrl-C)
    part: str = ""
    frac: float = 0.0  # of the current part
    finished: set = field(default_factory=set)
    started: float = 0.0
    part_started: float = 0.0
    # [answered at the part's first event, its time, answered now, total] for a part that
    # reports answers: the bar shows answered/total and the rate counts only answers made
    # since that first event (an eval resumed from 600 of 728 once showed ~1 min left).
    counts: list | None = None
    log: Path | None = None

    @property
    def total(self) -> float:
        return sum(self.minutes.values())



def read(p: Path) -> dict:
    return json.loads(p.read_text())


def write(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")



def evroc_ok() -> bool:
    if not shutil.which("evroc"):
        return False
    r = subprocess.run("evroc storage bucket get-s3-credentials", shell=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode:
        print(f"evroc get-s3-credentials -> exit {r.returncode}: {r.stderr.strip()[-600:]}", file=sys.stderr)
    return r.returncode == 0



# ---------- running commands ----------

state_lock = threading.RLock()  # the step state and the terminal


def above(text: str) -> None:
    """Prints a line above the bar (the bar is redrawn on the next tick)."""
    with state_lock:
        sys.stdout.write(("\r\033[K" if TTY else "") + text + "\n")
        sys.stdout.flush()


def sh(step: Step, cmd: str, env: dict | None = None) -> None:
    """Runs cmd in bash with .env.vm sourced; its progress events move the bar."""
    full = f"cd {REPO} && {{ [ -f .env.vm ] && source .env.vm; true; }} && {cmd}"
    with open(step.log, "a") as log:
        log.write(f"\n===== {time.strftime('%F %T')} $ {cmd}\n")
        log.flush()
        proc = subprocess.Popen(["bash", "-c", full], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace", env={**os.environ, **(env or {})})
        tail: list[str] = []
        for line in proc.stdout:
            log.write(line)
            log.flush()
            on_line(step, line)
            if line.strip() and parse(line) is None:
                tail = (tail + [line.rstrip()])[-15:]
        if proc.wait() != 0:
            raise RuntimeError(f"`{cmd}` failed:\n" + "\n".join("    " + t for t in tail[-8:]))


def on_line(step: Step, line: str) -> None:
    e = parse(line)
    if e is None:
        return
    stage, status = e["stage"], e["status"]
    with state_lock:
        if stage in step.minutes and stage not in step.finished:
            # A step's eventless part ("": an export, say) comes before its pipeline stages,
            # so the first event of one ends it, and its guessed minutes leave the time left.
            if step.part == "" and "" in step.minutes:
                step.finished.add("")
            if status in ("done", "skipped"):
                step.finished.add(stage)
                if status == "done":
                    took = hm(time.time() - step.part_started) if step.part == stage else ""
                    above(f"    {stage} done{f' in {took}' if took else ''}: {e.get('msg', '')}")
                step.part, step.frac = "", 0.0
                return
            if step.part != stage:
                step.part, step.frac, step.part_started, step.counts = stage, 0.0, time.time(), None
            if "answered" in e and e.get("total"):
                step.frac = max(step.frac, 0.1 + 0.85 * e["answered"] / e["total"])
                # A count that goes back is a new one (the next candidate in the same stage).
                if step.counts is None or e["answered"] < step.counts[2]:
                    step.counts = [e["answered"], time.time(), e["answered"], e["total"]]
                else:
                    step.counts[2:] = [e["answered"], e["total"]]
            elif e.get("pct") is not None:
                step.frac = max(step.frac, e["pct"] / 100)
        if status == "error":
            above(f"    {stage} error: {e.get('msg', '')}")


# ---------- the bar ----------

def hm(seconds: float) -> str:
    if seconds < 60:
        return "<1m"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def clock_fill(steps: list[Step]) -> None:
    """Moves the running steps that are in their eventless part: by their own progress
    measure, else by the clock against that part's guess. A step whose pipeline is
    sending events keeps the progress those set (the clock once overwrote an eval's
    answered/total every second: 600 of 728 answers showed as 4%)."""
    now = time.time()
    for s in steps:
        if s.state == "running" and s.part == "" and "" in s.minutes and "" not in s.finished:
            p = s.progress()
            s.frac = p if p is not None else min(0.95, (now - s.part_started) / max(1e-6, s.minutes[""] * 60))


class Run:
    def __init__(self, steps: list[Step]):
        self.steps, self.t0 = steps, time.time()
        self.done_est = 0.0  # minutes of guesses finished in this run
        self.done_real = 0.0  # and how long they really took

    def fraction_and_left(self) -> tuple[float, float]:
        total = sum(s.total for s in self.steps)
        speed = min(3.0, max(0.5, self.done_real / self.done_est)) if self.done_est > 10 else 1.0
        got, left = 0.0, 0.0
        for s in self.steps:
            if s.state in ("done", "skipped"):
                got += s.total
                continue
            for part, mins in s.minutes.items():
                if s.state == "running" and part in s.finished:
                    got += mins
                elif s.state == "running" and part == s.part:
                    frac = s.frac
                    got += mins * frac
                    ran = time.time() - (s.part_started or s.started)
                    c = s.counts
                    if c:  # answers left at the rate since the part's first event
                        since = time.time() - c[1]
                        if c[2] > c[0] and since > 120:
                            left += (c[3] - c[2]) * since / (c[2] - c[0])
                        else:
                            left += mins * 60 * speed * (1 - frac)
                    elif frac >= 0.05 and ran > 120:
                        left += ran / frac * (1 - frac)
                    else:
                        left += mins * 60 * speed * (1 - frac)
                else:
                    left += mins * 60 * speed
        return got / total, left

    def line(self) -> str:
        frac, left = self.fraction_and_left()
        n = len(self.steps)
        cur = next((s for s in self.steps if s.state == "running"), None)
        what = ""
        if cur:
            i = self.steps.index(cur) + 1
            c = cur.counts
            part = (f" · {cur.part} {c[2]}/{c[3]}" if c else f" · {cur.part} {cur.frac:.0%}") if cur.part else ""
            what = f"  | {i}/{n} {cur.name}{part}"
        filled = int(WIDTH * frac)
        eta = "done" if frac >= 1 else f"~{hm(left)} left" if left >= 60 else "<1m left"
        return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {frac:.0%}  {hm(time.time() - self.t0)} in, {eta}{what}"


def refresh_minutes(plan: list[Step]) -> None:
    """Each step not yet done or running takes its guess again from its refresh (a guess
    that fails to read keeps the old one)."""
    for s in plan:
        if s.refresh is not None and s.state == "waiting":
            try:
                new = s.refresh()
            except Exception:  # noqa: BLE001  (a guess is never worth stopping a run for)
                continue
            with state_lock:
                s.minutes = new


def run_plan(plan: list[Step], summary: Callable[[], str], logs: Path, argv: list[str] | None = None,
             *, after_step: Callable[[Step | None, bool], None] | None = None) -> int:
    """Runs the steps in order with the bar; --list prints them, `summary` the results.

    after_step(step, final), when given, uploads what the run has made so far. It is
    called as after_step(s, False) after each step s that finishes in this run, and
    once as after_step(s, True) in a finally when the run ends, however it ends: s is
    the step the run ended in (state "failed", or "stopped" for Ctrl-C), or None when
    no step failed. A hook that raises is reported and the run goes on (the next call
    tries again); its errors are listed at the end and leave the exit code alone, so
    the caller decides what a missing upload means (e.g. no shutdown)."""
    argv = sys.argv if argv is None else argv
    if "--list" in argv:
        for i, s in enumerate(plan, 1):
            parts = ", ".join(f"{k} {v:.0f}m" for k, v in s.minutes.items() if k)
            print(f"{i:2}. {s.name:38} ~{hm(s.total * 60):>6}" + (f"  ({parts})" if parts else "")
                  + ("  [done]" if s.done() else ""))
        print(f"    total ~{hm(sum(s.total for s in plan) * 60)} (guesses)")
        return 0
    if "summary" in argv:
        print(summary())
        return 0

    logs.mkdir(exist_ok=True)
    run = Run(plan)
    for s in plan:
        s.state = "done" if s.done() else "waiting"
    stop = threading.Event()

    def draw():
        while not stop.wait(1):
            with state_lock:
                clock_fill(plan)
                if TTY:
                    sys.stdout.write("\r\033[K" + run.line())
                    sys.stdout.flush()

    def heartbeat():  # without a terminal (e.g. piped to a file), one line every 5 minutes
        while not stop.wait(300):
            above(run.line())

    upload_failed: list[str] = []

    def upload(s: Step | None, final: bool) -> None:
        if after_step is None:
            return
        what = f"after {s.name}" if s and not final else "at the end"
        try:
            after_step(s, final)
        except KeyboardInterrupt:
            if not final:
                raise
            upload_failed.append(f"{what}: stopped with Ctrl-C")
            above(f"  upload {what} stopped with Ctrl-C")
        except Exception as e:  # noqa: BLE001  (the results stay on the disk; the next call retries)
            upload_failed.append(f"{what}: {e}")
            above(f"  upload {what} failed: {e}")

    threading.Thread(target=draw, daemon=True).start()
    if not TTY:
        threading.Thread(target=heartbeat, daemon=True).start()
    skipped, failed = [], None
    at: Step | None = None  # the step the run is in; left set when it fails or is stopped
    try:
        for i, s in enumerate(plan, 1):
            if s.state == "done":
                above(f"✓ {i}. {s.name} (done before)")
                continue
            s.log = logs / f"{i:02d}-{re.sub(r'[^a-z0-9]+', '-', s.name.lower()).strip('-')}.log"
            refresh_minutes(plan)
            with state_lock:
                s.state, s.started, s.part_started = "running", time.time(), time.time()
                s.part = "" if "" in s.minutes else next(iter(s.minutes))
            above(f"▶ {i}. {s.name}  (log: {s.log})")
            at = s
            try:
                s.run(s)
                took = time.time() - s.started
                with state_lock:
                    s.state = "done"
                    if took >= MIN_TIMED_S:
                        run.done_est += s.total
                        run.done_real += took / 60
                above(f"✓ {i}. {s.name} in {hm(took)}")
            except Done as e:
                with state_lock:
                    s.state = "done"
                above(f"✓ {i}. {s.name} ({e or 'done before'})")
            except Skip as e:
                with state_lock:
                    s.state = "skipped"
                skipped.append(f"{s.name}: {e}")
                above(f"– {i}. {s.name} skipped: {e}")
            except Exception as e:  # noqa: BLE001
                with state_lock:
                    s.state = "failed"
                failed = f"{s.name}: {e}"
                above(f"✗ {i}. {s.name} failed: {e}\n  Full log: {s.log}")
                break
            at = None
            if s.state == "done":
                upload(s, False)
    except KeyboardInterrupt:
        above(f"\nStopped during {at.name if at else 'an upload'}. Re-run the same command to carry on from there.")
        return 130
    finally:
        stop.set()
        with state_lock:
            if at is not None and at.state == "running":  # Ctrl-C, or anything else that is not an Exception
                at.state = "stopped"
        upload(at, True)
    time.sleep(0.1)
    if TTY:
        sys.stdout.write("\r\033[K" + run.line() + "\n")
    print("\n" + summary())
    if skipped:
        print("\nSkipped:\n" + "\n".join("  " + x for x in skipped))
    if upload_failed:
        print("\nUploads that failed (the results are still on this disk):\n"
              + "\n".join("  " + x for x in upload_failed))
    if failed:
        print(f"\nStopped at {failed}\nSend Claude this output. Re-running the same command resumes.")
        return 1
    print(f"\nFinished in {hm(time.time() - run.t0)}. Send Claude the table above.")
    return 0
