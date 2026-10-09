#!/usr/bin/env python3
"""Draws a progress bar from the pipeline's progress events on stdin.

    python pipeline.py --job J --only eval 2>&1 | python scripts/progress_bar.py J/run.log

Every line also goes to the log file. Stage messages and errors print above the
bar. Events carrying answered/total (the code eval sends them) move the bar, and
the time left is extrapolated from the answers finished so far."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.progress import parse  # noqa: E402

WIDTH = 30
TTY = sys.stdout.isatty()
lock = threading.Lock()
state = {"done": 0, "total": 0, "start": None, "status": "starting"}


def hm(seconds: float) -> str:
    if seconds < 60:
        return "<1m"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def bar() -> str:
    done, total, start = state["done"], state["total"], state["start"]
    if not total:
        return f"{state['status']}..."
    elapsed = time.time() - start
    filled = WIDTH * done // total
    left = "done" if done == total else f"~{hm(elapsed / done * (total - done))} left" if done else "estimating time left"
    return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {done}/{total} problems  {done / total:.0%}  {hm(elapsed)} in, {left}"


def above(text: str) -> None:
    """Prints a line above the bar (the bar is redrawn on the next tick)."""
    sys.stdout.write(("\r\033[K" if TTY else "") + text + "\n")
    sys.stdout.flush()


def read(log) -> None:
    for line in sys.stdin:
        log.write(line)
        log.flush()
        event = parse(line)
        with lock:
            if event is None:
                if any(w in line for w in ("Error", "error", "Traceback")):
                    above(line.rstrip())
            elif "answered" in event:
                if state["start"] is None or event["answered"] == 0:
                    state["start"] = time.time()
                state["done"], state["total"] = event["answered"], event["total"]
                if not TTY:
                    above(bar())
            elif event.get("msg"):
                if not state["total"]:
                    state["status"] = event["msg"]
                above(event["msg"])


def main() -> None:
    with open(sys.argv[1], "a") as log:
        reader = threading.Thread(target=read, args=(log,), daemon=True)
        reader.start()
        while reader.is_alive():
            if TTY:
                with lock:
                    sys.stdout.write("\r\033[K" + bar())
                    sys.stdout.flush()
            reader.join(1)
        if TTY and state["total"]:
            sys.stdout.write("\r\033[K" + bar() + "\n")


if __name__ == "__main__":
    main()
