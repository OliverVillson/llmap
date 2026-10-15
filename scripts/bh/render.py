"""The bridge to the engine's own wire format: every request goes to mugge/bench/render.ts
(`mugge render-call`), which builds the write and fix messages with prompts.ts and parses and
renders answers with files.ts. Nothing in Python builds a prompt or renders an answer, so the
rows cannot drift from what the runtime harness sends. See render.ts for the ops."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MUGGE = REPO / "mugge"
RENDER = MUGGE / "bench" / "render.ts"
CHUNK = 400  # requests per bun process: one start-up for many renders, bounded memory


class RenderError(RuntimeError):
    pass


def bun() -> str:
    return os.environ.get("BUN", "bun")


def call(requests: list[dict], *, strict: bool = True, chunk: int = CHUNK) -> list[dict]:
    """Replies to requests, in order. strict: a reply with an error raises; otherwise it is
    returned as {"id", "error"} for the caller to count."""
    out: list[dict] = []
    for i in range(0, len(requests), chunk):
        part = requests[i:i + chunk]
        text = "".join(json.dumps({**r, "id": i + j}, ensure_ascii=False) + "\n" for j, r in enumerate(part))
        p = subprocess.run([bun(), str(RENDER)], input=text.encode(), capture_output=True, cwd=MUGGE)
        if p.returncode != 0:
            raise RenderError(f"render.ts exited {p.returncode}: {p.stderr.decode(errors='replace')[-2000:]}")
        replies = [json.loads(line) for line in p.stdout.decode().splitlines() if line.strip()]
        if len(replies) != len(part):
            raise RenderError(f"render.ts answered {len(replies)} of {len(part)} requests")
        for req, rep in zip(part, replies):
            rep["id"] = req.get("id")
            if strict and rep.get("error"):
                raise RenderError(f"{req.get('op')} {req.get('ticket', '')}: {rep['error']}")
            out.append(rep)
    return out


def one(request: dict) -> dict:
    return call([request])[0]


def tickets(bench: str | Path) -> dict[str, dict]:
    """Every ticket under a bench dir by key (`<project>/<id>`), as bench/load.ts reads it."""
    return {t["key"]: t for t in one({"op": "tickets", "bench": str(bench)})["tickets"]}


def system_prompt() -> str:
    """The engine's worker system prompt (prompts.ts SYSTEM), read through a render."""
    spike = MUGGE / "spike"
    t = next(iter(tickets(spike).values()))
    return one({"op": "write", "project": t["dir"], "ticket": t["id"]})["messages"][0]["content"]


PACK_END = "## Your files as they are now"
FIX_MARK = "\n\n## Fix\nThis command failed:\n$ "


def kind_of(prompt: str) -> str:
    """write or fix, from the user message (a fix message is the pack plus the Fix section)."""
    return "fix" if FIX_MARK in prompt else "write"


def failed_command(prompt: str) -> str:
    """The command a fix message names ("(write files)", "(model call)" or a check)."""
    i = prompt.rfind(FIX_MARK)
    return prompt[i + len(FIX_MARK):].split("\n", 1)[0] if i >= 0 else ""
