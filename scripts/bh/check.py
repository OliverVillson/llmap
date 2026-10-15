"""Sandbox checks of answers before they become rows, and of reference answers before the
teacher runs: a ticket's files go into the worktree the bench starts it in (scaffold plus
its dependencies' reference answers), its acceptance commands run, then its hidden files are
copied in and its hidden commands run, exactly as bench/run.ts grades a ticket. An answer
passes when every command exits 0.

Isolation is stages/sandbox.py's (sandbox="local", the default): no network through
`unshare -rn`, rlimits on CPU time, memory and file size, its own process group killed whole
on timeout, a scrubbed environment. sandbox="docker"|"podman" runs each command in the
bench's toolchain image instead, with the argv of mugge/src/engine/sandbox.ts
ContainerSandbox (--network none), for languages whose toolchain is only in an image.
Command output is stdout then stderr, trimmed, as the engine reads it, so a failing
check's output can go into a fix call unchanged (render.ts trims it the engine's way).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import render

sys.path.insert(0, str(render.REPO))
from stages import sandbox  # noqa: E402

WORK = "/work"  # where the engine's container sandbox mounts the worktree (sandbox.ts)
TIMEOUT = 120.0  # seconds per command: sandbox.ts DEFAULT_TIMEOUT_MS
CHUNK = 128
# SIOCGIFFLAGS / SIOCSIFFLAGS | IFF_UP on lo, for hosts without the ip command
LO_UP = ("python3 -c \"import socket,fcntl,struct as t;s=socket.socket();q=lambda f:t.pack('16sH14s',b'lo',f,b'');"
         "fcntl.ioctl(s,0x8914,q(t.unpack('16sH14s',fcntl.ioctl(s,0x8913,q(0)))[1]|1))\"")  # worktrees on disk at once per check_many chunk


@dataclass
class Item:
    project: str  # bench project dir
    ticket: str  # ticket id
    files: dict[str, str] = field(default_factory=dict)  # the answer's files, written over the start worktree
    reference: bool = False  # check the ticket's own reference answer instead of files
    key: str = ""


@dataclass
class Result:
    ok: bool
    stage: str = ""  # "" when ok | refused | acceptance | hidden | error
    command: str = ""
    output: str = ""
    timed_out: bool = False
    worktree: str | None = None  # kept=True: the worktree as the failing check left it


def container_argv(engine: str, command: str, cwd: str, image: str, prefix: str = "mugge") -> list[str]:
    """sandbox.ts ContainerSandbox.argv, for the same isolation the bench uses."""
    user = ["--userns", "keep-id"] if engine == "podman" else ["--user", f"{os.getuid()}:{os.getgid()}"]
    return [engine, "run", "--rm", "--network", "none", "--memory", "2g", "--cpus", "2", "--pids-limit", "512",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL", *user, "-e", "HOME=/tmp",
            "-v", f"{cwd}:/work", "-w", "/work", f"{prefix}-{image}", "bash", "-c", command]


def run(command: str, cwd: str, image: str, timeout: float = TIMEOUT, kind: str = "local") -> tuple[int, str, bool]:
    """(exit code, stdout+stderr trimmed, timed out)."""
    home = tempfile.mkdtemp(prefix="bh-home-")
    try:
        if kind == "local":
            wrap = sandbox._wrapper()
            # unshare -rn leaves loopback down; tests that serve on 127.0.0.1 need it up
            argv = [*wrap, "bash", "-c", f"{LO_UP} 2>/dev/null\n{command}" if wrap == ("unshare", "-rn") else command]
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": home, "LANG": "C.UTF-8",
                   "NODE_OPTIONS": "", "PYTHONDONTWRITEBYTECODE": "1"}
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, preexec_fn=sandbox._limits(timeout))
        elif kind in ("docker", "podman"):
            p = subprocess.Popen(container_argv(kind, command, cwd, image), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        else:
            raise ValueError(f"sandbox must be local, docker or podman, not {kind}")
        timed_out = False
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            sandbox._kill(p)
            out, err = p.communicate()
        text = (out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip()
        if kind == "local":  # paths as the container sandbox shows them (the worktree is /work there)
            text = text.replace(os.path.realpath(cwd), WORK).replace(cwd, WORK).replace(home, "/tmp")
        return (124 if timed_out else p.returncode), text, timed_out
    finally:
        shutil.rmtree(home, ignore_errors=True)


def _commands(tk: dict, which: str) -> list[str]:
    return list(tk["acceptance"] if which == "acceptance" else tk["hidden"])


def check_many(items: list[Item], tickets: dict[str, dict], *, workers: int | None = None, timeout: float = TIMEOUT,
               kind: str = "local", keep: bool = False) -> list[Result]:
    """One Result per item, in order. tickets: render.tickets(bench) for every bench the
    items come from, by key. keep: leave the worktree of each acceptance failure on disk
    (Result.worktree; the caller removes it) so a fix call can be rendered from it."""
    workers = workers or sandbox.WORKERS
    results: list[Result] = []
    for start in range(0, len(items), CHUNK):
        part = items[start:start + CHUNK]
        root = tempfile.mkdtemp(prefix="bh-check-")
        try:
            dirs = [os.path.join(root, str(i)) for i in range(len(part))]
            render.call([{"op": "prepare", "project": it.project, "ticket": it.ticket, "dest": d, "own": it.reference}
                         for it, d in zip(part, dirs)])
            res: list[Result | None] = [None] * len(part)
            for i, (it, d) in enumerate(zip(part, dirs)):
                tk = tickets[it.key]
                bad = [p for p in it.files if p not in tk["owns"]]
                if bad:
                    res[i] = Result(False, "refused", "(write files)", f"files outside owns: {', '.join(bad)}")
                    continue
                for path, content in it.files.items():
                    dest = Path(d, path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content)

            def stage(i: int, which: str, part=part, dirs=dirs, res=res) -> None:
                it, d = part[i], dirs[i]
                for command in _commands(tickets[it.key], which):
                    code, out, timed = run(command, d, tickets[it.key]["image"], timeout, kind)
                    if code != 0:
                        res[i] = Result(False, which, command, out, timed)
                        return

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda i: stage(i, "acceptance"), [i for i in range(len(part)) if res[i] is None]))
                passed = [i for i in range(len(part)) if res[i] is None]
                if keep:  # the failing worktrees as the engine saw them, before any hidden file
                    for i in range(len(part)):
                        if res[i] is not None and res[i].stage == "acceptance":
                            kept = tempfile.mkdtemp(prefix="bh-kept-")
                            os.rmdir(kept)
                            shutil.move(dirs[i], kept)
                            res[i].worktree = kept
                render.call([{"op": "hidden", "project": part[i].project, "ticket": part[i].ticket, "dest": dirs[i]}
                             for i in passed])
                list(ex.map(lambda i: stage(i, "hidden"), passed))
            results.extend(r if r is not None else Result(True) for r in res)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def check_references(bench: str | Path, *, workers: int | None = None, timeout: float = TIMEOUT,
                     kind: str = "local") -> dict:
    """Every ticket's reference answer through its acceptance and hidden commands. A ticket
    whose reference fails is broken: no row may come from it and the teacher's score leaves
    it out. Returns {"tickets", "passed", "broken": {key: "stage: command\\noutput tail"}}."""
    tks = render.tickets(bench)
    items = [Item(t["dir"], t["id"], reference=True, key=k) for k, t in tks.items()]
    res = check_many(items, tks, workers=workers, timeout=timeout, kind=kind)
    broken = {it.key: f"{r.stage}: {r.command}\n{sandbox_tail(r.output)}" for it, r in zip(items, res) if not r.ok}
    return {"tickets": len(items), "passed": len(items) - len(broken), "broken": broken}


def sandbox_tail(text: str, n: int = 800) -> str:
    return text if len(text) <= n else "…" + text[-n:]
