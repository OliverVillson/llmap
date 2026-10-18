"""Horizontal rules."""

import re
from typing import List

from mdlite.types import BlockResult, Inline

_RULE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A rule is a line of three or more of the same character, "-", "*" or "_", with any
    spaces or tabs before, between and after them and nothing else: "---", "* * *", " ___ ".
    Returns ("<hr>\\n", i + 1); None for anything else (also when i is past the end).
    """
    if i < len(lines) and _RULE.match(lines[i]):
        return "<hr>\n", i + 1
    return None
