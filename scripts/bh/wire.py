"""What both sides of the train harness agree on about one model call: its key, and the
engine's system prompt (stages/harness.py mirrors mugge/src/engine/prompts.ts, and
tests/test_harness.py holds the two equal)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.harness import SYSTEM  # noqa: E402,F401  (stdlib-only module)


def call_key(user: str, answer: str) -> str:
    """One call's key: its user message and its answer. The bench transcript has both
    (prompt, answer) and so does the recording proxy's log, so the two join on it."""
    return hashlib.sha256(f"{user}\x00{answer}".encode()).hexdigest()[:24]
