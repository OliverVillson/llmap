"""Paragraphs."""

from typing import List

from mdlite.types import BlockResult, Inline


def starts_block(line: str) -> bool:
    """Whether a line starts some other block and so ends a paragraph above it: a heading
    ("#" to "######" then a space, a tab or the end of the line), a code fence ("```" or "~~~"
    after up to three spaces), a quote (">" after up to three spaces), a list item ("-", "*"
    or "+" then a space, or digits then "." or ")" then a space, after up to three spaces) or
    a rule (as in block_rule: three or more of one of - * _ with only spaces or tabs around).
    """
    raise NotImplementedError


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A paragraph is line i (which must not be blank: empty or only whitespace) and every
    following line up to, not including, the first blank line, the first line for which
    starts_block() is true, or the end. Each line is stripped of whitespace on both ends; they
    are joined with "\\n" and passed to inline() once.

    Returns ("<p>" + inline(joined) + "</p>\\n", index of the first line after it); None when
    line i is blank or past the end.
    """
    raise NotImplementedError
