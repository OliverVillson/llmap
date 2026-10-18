"""Paragraphs."""

import re
from typing import List

from mdlite.types import BlockResult, Inline

_STARTERS = re.compile(
    r"^(?:#{1,6}(?:[ \t]|$)"
    r"| {0,3}(?:```|~~~)"
    r"| {0,3}>"
    r"| {0,3}(?:[-*+]|\d+[.)]) "
    r"|[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$)"
)


def starts_block(line: str) -> bool:
    """Whether a line starts some other block and so ends a paragraph above it: a heading
    ("#" to "######" then a space, a tab or the end of the line), a code fence ("```" or "~~~"
    after up to three spaces), a quote (">" after up to three spaces), a list item ("-", "*"
    or "+" then a space, or digits then "." or ")" then a space, after up to three spaces) or
    a rule (as in block_rule: three or more of one of - * _ with only spaces or tabs around).
    """
    return bool(_STARTERS.match(line))


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A paragraph is line i (which must not be blank: empty or only whitespace) and every
    following line up to, not including, the first blank line, the first line for which
    starts_block() is true, or the end. Each line is stripped of whitespace on both ends; they
    are joined with "\\n" and passed to inline() once.

    Returns ("<p>" + inline(joined) + "</p>\\n", index of the first line after it); None when
    line i is blank or past the end.
    """
    if i >= len(lines) or not lines[i].strip():
        return None
    j = i + 1
    while j < len(lines) and lines[j].strip() and not starts_block(lines[j]):
        j += 1
    text = "\n".join(line.strip() for line in lines[i:j])
    return f"<p>{inline(text)}</p>\n", j
