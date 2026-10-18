"""Horizontal rules."""

from typing import List

from mdlite.types import BlockResult, Inline


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A rule is a line of three or more of the same character, "-", "*" or "_", with any
    spaces or tabs before, between and after them and nothing else: "---", "* * *", " ___ ".
    Returns ("<hr>\\n", i + 1); None for anything else (also when i is past the end).
    """
    raise NotImplementedError
