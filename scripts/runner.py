"""The one-command experiment runner: steps in order, a progress bar with the time
left, logs per step, and resume (a step whose done() holds is skipped). Used by
scripts/exp02.py and scripts/exp03.py; stdlib only, for the system python3.
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


@dataclass
class Step:
    name: str
    minutes: dict[str, float]  # time guess per part: pipeline stages, or {"": n}
    run: Callable[["Step"], None]
    done: Callable[[], bool] = lambda: False
    progress: Callable[[], float | None] = lambda: None  # for steps that send no events
    # live state
    state: str = "waiting"  # waiting, running, done, skipped, failed
    part: str = ""
    frac: float = 0.0  # of the current part
    finished: set = field(default_factory=set)
    started: float = 0.0
    part_started: float = 0.0
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
    return subprocess.run("evroc storage bucket get-s3-credentials", shell=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0



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
            if status in ("done", "skipped"):
                step.finished.add(stage)
                if status == "done":
                    took = hm(time.time() - step.part_started) if step.part == stage else ""
                    above(f"    {stage} done{f' in {took}' if took else ''}: {e.get('msg', '')}")
                step.part, step.frac = "", 0.0
                return
            if step.part != stage:
                step.part, step.frac, step.part_started = stage, 0.0, time.time()
            if "answered" in e and e.get("total"):
                step.frac = max(step.frac, 0.1 + 0.85 * e["answered"] / e["total"])
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
                    if frac >= 0.05 and ran > 120:
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
            part = f" · {cur.part} {cur.frac:.0%}" if cur.part else ""
            what = f"  | {i}/{n} {cur.name}{part}"
        filled = int(WIDTH * frac)
        eta = "done" if frac >= 1 else f"~{hm(left)} left" if left >= 60 else "<1m left"
        return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {frac:.0%}  {hm(time.time() - self.t0)} in, {eta}{what}"


def run_plan(plan: list[Step], summary: Callable[[], str], logs: Path, argv: list[str] | None = None) -> int:
    """Runs the steps in order with the bar; --list prints them, `summary` the results."""
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
                for s in plan:  # steps without progress events: their own measure, else the clock
                    if s.state == "running" and "" in s.minutes:
                        p = s.progress()
                        s.frac = p if p is not None else min(0.95, (time.time() - s.started) / (s.total * 60))
                if TTY:
                    sys.stdout.write("\r\033[K" + run.line())
                    sys.stdout.flush()

    def heartbeat():  # without a terminal (e.g. piped to a file), one line every 5 minutes
        while not stop.wait(300):
            above(run.line())

    threading.Thread(target=draw, daemon=True).start()
    if not TTY:
        threading.Thread(target=heartbeat, daemon=True).start()
    skipped, failed = [], None
    for i, s in enumerate(plan, 1):
        if s.state == "done":
            above(f"✓ {i}. {s.name} (done before)")
            continue
        s.log = logs / f"{i:02d}-{re.sub(r'[^a-z0-9]+', '-', s.name.lower()).strip('-')}.log"
        with state_lock:
            s.state, s.started, s.part_started = "running", time.time(), time.time()
            s.part = "" if "" in s.minutes else next(iter(s.minutes))
        above(f"▶ {i}. {s.name}  (log: {s.log})")
        try:
            s.run(s)
            took = time.time() - s.started
            with state_lock:
                s.state = "done"
                run.done_est += s.total
                run.done_real += took / 60
            above(f"✓ {i}. {s.name} in {hm(took)}")
        except Skip as e:
            with state_lock:
                s.state = "skipped"
            skipped.append(f"{s.name}: {e}")
            above(f"– {i}. {s.name} skipped: {e}")
        except KeyboardInterrupt:
            stop.set()
            above(f"\nStopped during {s.name}. Re-run the same command to carry on from there.")
            return 130
        except Exception as e:  # noqa: BLE001
            with state_lock:
                s.state = "failed"
            failed = f"{s.name}: {e}"
            above(f"✗ {i}. {s.name} failed: {e}\n  Full log: {s.log}")
            break
    stop.set()
    time.sleep(0.1)
    if TTY:
        sys.stdout.write("\r\033[K" + run.line() + "\n")
    print("\n" + summary())
    if skipped:
        print("\nSkipped:\n" + "\n".join("  " + x for x in skipped))
    if failed:
        print(f"\nStopped at {failed}\nSend Claude this output. Re-running the same command resumes.")
        return 1
    print(f"\nFinished in {hm(time.time() - run.t0)}. Send Claude the table above.")
    return 0

