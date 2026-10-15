"""Runs that outlive the SSH session, and the view to check on them (bh_run.py PROFILE
start | watch | stop; status is the one-shot form).

Per profile, in the logs dir (BH_LOGS, default ~/bh-logs/<run>):
  <profile>-run.json     the run's state: pid, argv, log, the stage it is in and since when,
                         every stage's state, the exit code; rewritten by the run itself
  <profile>-<time>.log   one log per detached run (everything the run prints)
  <profile>-current.log  a link to the newest one

start re-runs bh_run.py with the same arguments in its own session (setsid), stdin closed and
output to the log, so closing the SSH session or the terminal does not touch it (Linux, WSL2,
macOS). stop sends it SIGINT, the same as Ctrl-C: the step in flight stops, the servers stop,
the results are uploaded, and the same command resumes. watch redraws one screen every few
seconds: stage and clock, the latest gate readings, tok/s from the recording proxy's log (and
the engine's own crew status.json when the bench writes one), and the log's tail.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def files(cfg) -> dict[str, Path]:
    d = Path(cfg.logs)
    return {"state": d / f"{cfg.profile}-run.json", "current": d / f"{cfg.profile}-current.log", "dir": d}


def read_state(cfg) -> dict:
    try:
        return json.loads(files(cfg)["state"].read_text())
    except (OSError, ValueError):
        return {}


def record(cfg, **kw) -> dict:
    """Merges kw into the run's state file (written whole, then renamed)."""
    p = files(cfg)["state"]
    p.parent.mkdir(parents=True, exist_ok=True)
    st = {**read_state(cfg), **kw, "updated": time.time()}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n")
    tmp.replace(p)
    return st


def alive(pid) -> bool:
    if not pid:
        return False
    try:  # our own child (start, then stop, in one process) is a zombie until reaped
        if os.waitpid(int(pid), os.WNOHANG)[0] == int(pid):
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running(cfg) -> int | None:
    """The pid of this profile's run when one is going."""
    st = read_state(cfg)
    return st["pid"] if st.get("state") == "running" and alive(st.get("pid")) else None


# ---------- start / stop ----------

def start(cfg, argv: list[str]) -> int:
    """argv (bh_run.py's arguments without `start`) as a detached run."""
    pid = running(cfg)
    if pid:
        print(f"a {cfg.profile} run is already going (pid {pid}): bh_run.py {cfg.profile} watch")
        return 2
    f = files(cfg)
    f["dir"].mkdir(parents=True, exist_ok=True)
    log = f["dir"] / f"{cfg.profile}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(log, "a") as out, open(os.devnull) as nul:
        proc = subprocess.Popen([sys.executable, "-u", str(REPO / "scripts" / "bh_run.py"), *argv], cwd=REPO,
                                stdin=nul, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "BH_RUN_LOG": str(log)})
    f["current"].unlink(missing_ok=True)
    f["current"].symlink_to(log.name)
    record(cfg, state="running", pid=proc.pid, argv=argv, log=str(log), started=time.time(), stage=None, exit=None)
    opts = next((" " + " ".join(argv[i:]) for i, x in enumerate(argv) if x.startswith("--")), "")
    print(f"started {cfg.profile} (pid {proc.pid}); it keeps running when you log out\n"
          f"  log:   {log}\n  watch: python3 scripts/bh_run.py {cfg.profile} watch{opts}\n"
          f"  stop:  python3 scripts/bh_run.py {cfg.profile} stop{opts}")
    return 0


def stop(cfg, wait_s: float = 600) -> int:
    """SIGINT to the run (as Ctrl-C), then wait for it to upload and exit; a second stop
    while it is still stopping sends SIGTERM."""
    st = read_state(cfg)
    pid = st.get("pid")
    if not alive(pid) or st.get("state") != "running":
        print(f"no {cfg.profile} run is going" + (f" (the last one {st.get('state')}, exit {st.get('exit')})"
                                                  if st else ""))
        return 0
    sig = signal.SIGTERM if st.get("stopping") else signal.SIGINT
    record(cfg, stopping=True)
    os.kill(int(pid), sig)
    print(f"sent {'SIGTERM' if sig == signal.SIGTERM else 'SIGINT (Ctrl-C)'} to {pid}; waiting for the upload and exit")
    deadline = time.monotonic() + wait_s
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(pid):
        print(f"still stopping after {wait_s:.0f}s (an upload, probably); `stop` again forces it")
        return 1
    argv = st.get("argv") or [cfg.profile]
    print(f"stopped; `python3 scripts/bh_run.py {' '.join([argv[0], 'start', *argv[1:]])}` resumes it")
    return 0


# ---------- watch ----------

def _tail(p: Path, n: int, max_bytes: int = 256_000) -> list[str]:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _newest(work: Path, pattern: str, within_s: float = 600) -> Path | None:
    now, best = time.time(), None
    for p in work.glob(pattern):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if now - m < within_s and (best is None or m > best[0]):
            best = (m, p)
    return best[1] if best else None


def tok_s(work: Path) -> str:
    """Completion tokens a second over the last minute, from the busiest recent proxy log."""
    log = _newest(work, "**/proxy/calls.jsonl")
    if log is None:
        return ""
    now, toks, n, cut = time.time(), 0, 0, 0
    for line in _tail(log, 5000):
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if now - (c.get("t") or 0) <= 60:
            toks += c.get("completion_tokens") or 0
            n += 1
            cut += bool(c.get("cut"))
    return f"{toks / 60:.0f} tok/s over the last minute ({n} calls, {cut} cut) · {log.parent.parent.name}"


def crew(work: Path) -> str:
    """The engine's crew status (status.json, schema centipede.status/1) of the newest bench run."""
    p = _newest(work, "**/status.json", within_s=120)
    if p is None:
        return ""
    try:
        s = json.loads(p.read_text())
        c, t = s["crew"], s["tokS"]
        return (f"crew {c['state']}: {c['agents']['working']} working, {c['agents']['queued']} queued; "
                f"tickets {c['tickets'].get('done', 0)}/{c['tickets']['total']}; {t['total']:.0f} tok/s · {p.parent}")
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def _hm(s: float) -> str:
    m = int(s // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def _event(log: Path | None) -> str:
    """The step's latest progress event (pct and message), e.g. the SFT's update count."""
    for line in reversed(_tail(log, 200) if log else []):
        if line.startswith("{") and '"pct"' in line:
            try:
                e = json.loads(line)
                return f"{e.get('pct', 0):.0f}% {e.get('msg', '')}".strip()
            except ValueError:
                continue
    return ""


def frame(cfg, run) -> str:
    st = read_state(cfg)
    now = time.time()
    if not st:
        return f"no {cfg.profile} run recorded in {files(cfg)['dir']}"
    going = st.get("state") == "running" and alive(st.get("pid"))
    state = st.get("state") if going or st.get("state") != "running" else "gone (no process; it died or the machine rebooted)"
    head = (f"BH{cfg.version} {cfg.profile} · {state} · pid {st.get('pid')} · up {_hm(now - (st.get('started') or now))}"
            f" · {time.strftime('%H:%M:%S')}")
    if cfg.session_minutes and cfg.where == "b200":
        head += f" · session {cfg.session_minutes:g} min, {cfg.session_minutes - run.elapsed_min():.0f} left"
    out = [head]
    stages = st.get("stages") or {}
    if stages:
        mark = {"done": "✓", "running": "▶", "failed": "✗", "skipped": "–", "stopped": "■"}
        out.append("  ".join(f"{mark.get(v, '·')} {k}" for k, v in stages.items()))
    if st.get("stage") and going:
        el = now - (st.get("stage_started") or now)
        line = f"stage {st['stage']}: {_hm(el)} of ~{st.get('stage_minutes', 0):.0f}m"
        ev = _event(Path(st["stage_log"])) if st.get("stage_log") else ""
        out.append(line + (f" · {ev}" if ev else ""))
    work = Path(cfg.work)
    for x in (tok_s(work), crew(work)):
        if x:
            out.append(x)
    if st.get("exit") is not None:
        out.append(f"exit {st['exit']}")
    gates = [x for x in run.summary().splitlines()[1:] if x.strip()]
    if gates:
        out += ["", *gates]
    if st.get("warnings"):
        out += ["", "warnings:"] + [f"  {w[:200]}" for w in st["warnings"][-5:]]
    log = Path(st["log"]) if st.get("log") else None
    if log:
        out += ["", f"--- {log}"] + [x[:160] for x in _tail(log, 12)]
    return "\n".join(out)


def watch(cfg, run, every: float = 3.0) -> int:
    """frame() redrawn in place until Ctrl-C (which stops the watching, not the run)."""
    tty = sys.stdout.isatty()
    try:
        while True:
            text = frame(cfg, run)
            sys.stdout.write(("\x1b[H\x1b[2J" if tty else "\n") + text + "\n")
            sys.stdout.flush()
            if not tty:
                return 0
            time.sleep(every)
    except KeyboardInterrupt:
        print()
        return 0
