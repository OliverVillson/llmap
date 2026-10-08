"""Mugge's harness calls, for training rows and eval: the same prompts the engine
sends (mugge/src/engine/prompts.ts) and the same answer format (mugge/src/engine/files.ts).
Keep the three in sync; tests/test_harness.py checks the system prompt against prompts.ts.

A write call gets the ticket's context pack; a fix call gets the same pack, with the
files as they are now, plus the failing command's output. Both are one fresh message,
never a chat history. The model answers with each owned file in full, as its path on
one line and its content in a fenced code block, then an optional "Note:" line. Plain files, not
JSON: code escaped inside JSON scores worse (Aider's code-in-JSON test).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SYSTEM = "\n".join([
    "You are a coding specialist working on one ticket of a larger project.",
    'Reply with each file you own in full: its path on one line, then its content in one fenced code block. '
    'You may end with one line "Note: <what you changed>".',
    "Write only the files the ticket owns. Do not change interfaces you only read.",
    "No comments that mention AI, tickets or this process; write code the way the repo already does.",
])

MAX_FILE_CHARS = 20_000  # prompts.ts: one huge file cannot crowd out the rest
FENCE_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".c": "c", ".h": "c", ".cpp": "cpp",
              ".hpp": "cpp", ".s": "asm", ".html": "html", ".css": "css", ".json": "json", ".md": "markdown",
              ".sh": "bash", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml"}


@dataclass
class Ticket:
    id: str
    title: str
    context: str
    owns: list[str]
    acceptance: list[str]
    role: str = "implement"
    # path -> content, as the engine reads them from the worktree
    interfaces: dict[str, str] = field(default_factory=dict)
    reads: dict[str, str] = field(default_factory=dict)
    current: dict[str, str] = field(default_factory=dict)  # the owned files as they are now
    memory: str = ""


def _file_block(path: str, text: str) -> str:
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n…(truncated)"
    return f"--- {path}\n{text}"


def pack_text(t: Ticket) -> str:
    """prompts.ts packText, line for line."""
    return "\n".join([
        f"Ticket: {t.id}",
        f"Title: {t.title}",
        f"Role: {t.role}",
        "",
        t.context,
        "",
        f"Files you own (write each in full): {', '.join(t.owns)}",
        f"Done when these commands pass: {' ; '.join(t.acceptance)}",
        f"\nProject notes:\n{t.memory}" if t.memory else "",
        "",
        "## Interfaces",
        *(_file_block(p, x) for p, x in t.interfaces.items()),
        "",
        "## Files to read",
        *(_file_block(p, x) for p, x in t.reads.items()),
        "",
        "## Your files as they are now",
        *(_file_block(p, t.current.get(p, "")) for p in t.owns),
    ])


def write_messages(t: Ticket) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": pack_text(t)}]


def fix_messages(t: Ticket, command: str, output: str) -> list[dict]:
    """t.current holds the files that failed."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (
        f"{pack_text(t)}\n\n## Fix\nThis command failed:\n$ {command}\n{output}\n\n"
        "Return your files fixed so it passes.")}]


def tail(text: str, n: int = 4000) -> str:
    """The end of a command's output, where the error is (sandbox.ts tail)."""
    text = text.strip()
    return text if len(text) <= n else "…" + text[-n:]


def render_files(files: dict[str, str], note: str = "") -> str:
    """The answer for these files: path line, fenced content, and the note."""
    parts = []
    for path, content in files.items():
        lang = FENCE_LANG.get(path[path.rfind("."):].lower(), "") if "." in path else ""
        runs = [len(m) for m in re.findall(r"^`{3,}", content, re.M)]
        fence = "`" * max(3, max(runs, default=0) + 1)
        parts.append(f"{path}\n{fence}{lang}\n{content.rstrip(chr(10))}\n{fence}")
    if note:
        parts.append(f"Note: {note}")
    return "\n\n".join(parts)


_OPEN = re.compile(r"^(`{3,})[\w+#.-]*\s*$")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _path(line: str) -> str:
    """A file path on its own line, minus markdown around it ("**a.py**", "### `a.py`:")."""
    p = re.sub(r"^#+\s*", "", line.strip()).strip("*`: ")
    return p if p and not re.search(r"\s", p) else ""


def parse_files(raw: str) -> tuple[dict[str, str], str]:
    """(files, note) from an answer. A fenced block counts when the line before it is a
    path; a block without one is not a file. Mirrors files.ts parseFiles."""
    lines = _THINK.sub("", raw).split("\n")
    files, note, i, prev = {}, "", 0, ""
    while i < len(lines):
        line = lines[i]
        m = _OPEN.match(line.strip())
        if m:
            fence, body, i = m.group(1), [], i + 1
            while i < len(lines) and lines[i].strip() != fence:
                body.append(lines[i])
                i += 1
            path = _path(prev)
            if path:
                files[path] = ("\n".join(body) + "\n") if body else ""
            prev, i = "", i + 1
            continue
        if line.strip().lower().startswith("note:"):
            note = line.strip()[5:].strip()
        if line.strip():
            prev = line
        i += 1
    return files, note
