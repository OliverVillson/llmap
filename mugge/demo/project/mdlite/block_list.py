"""Bullet, numbered and task lists (one level, no nesting)."""

from typing import List

from mdlite.types import BlockResult, Inline


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A list starts at an item line: after up to three spaces, a bullet "-", "*" or "+"
    followed by a space, or a number of 1 to 9 digits followed by "." or ")" and a space.
    The list continues with items of the same kind: the same bullet character, or numbers
    with the same "." or ")". An item's text is the rest of its line, stripped, plus any
    following lines indented by at least two spaces that are not blank and not items
    themselves, each stripped and joined with "\\n". The list ends at a blank line, at a line
    that is neither such an item nor a continuation, or at the end.

    Each item: "<li>{inline(text)}</li>\\n". A task item, text starting with "[ ] " or
    "[x] " (or "[X] "), is '<li class="task"><input type="checkbox" disabled> {inline(rest)}</li>\\n',
    with " checked" after "disabled" for an x. Bullets give "<ul>\\n...</ul>\\n"; numbers give
    "<ol>\\n...</ol>\\n", or '<ol start="{n}">' when the first number n is not 1 (leading zeros
    dropped: "007." starts at 7).

    Returns (html, index of the first line after the list); None when line i is not an item.
    """
    raise NotImplementedError
