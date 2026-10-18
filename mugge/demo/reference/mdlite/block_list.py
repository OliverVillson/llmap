"""Bullet, numbered and task lists (one level, no nesting)."""

import re
from typing import List, Optional, Tuple

from mdlite.types import BlockResult, Inline

_ITEM = re.compile(r"^ {0,3}(?:([-*+])|(\d{1,9})([.)])) (.*)$")
_TASK = re.compile(r"^\[([ xX])\] (.*)$", re.S)


def _item(line: str) -> Optional[Tuple[str, Optional[int], str]]:
    m = _ITEM.match(line)
    if not m:
        return None
    if m.group(1):
        return m.group(1), None, m.group(4)
    return m.group(3), int(m.group(2)), m.group(4)


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
    if i >= len(lines):
        return None
    first = _item(lines[i])
    if not first:
        return None
    marker, start, _ = first
    items: List[str] = []
    j = i
    while j < len(lines):
        it = _item(lines[j])
        if not it or it[0] != marker or (it[1] is None) != (start is None):
            break
        text = [it[2].strip()]
        j += 1
        while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() and not _item(lines[j]):
            text.append(lines[j].strip())
            j += 1
        items.append("\n".join(text))
    out = []
    for text in items:
        t = _TASK.match(text)
        if t:
            checked = " checked" if t.group(1) in "xX" else ""
            out.append(f'<li class="task"><input type="checkbox" disabled{checked}> {inline(t.group(2))}</li>\n')
        else:
            out.append(f"<li>{inline(text)}</li>\n")
    if start is None:
        open_tag, close_tag = "<ul>\n", "</ul>\n"
    else:
        open_tag, close_tag = ("<ol>\n" if start == 1 else f'<ol start="{start}">\n'), "</ol>\n"
    return open_tag + "".join(out) + close_tag, j
