"""Block quotes."""

import re
from typing import List

from mdlite.types import BlockResult, Inline

_QUOTE = re.compile(r"^ {0,3}> ?")


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A block quote is line i and the lines right after it that start with ">" after up to
    three spaces; it ends at the first line that does not. From each line remove the spaces
    before ">", the ">" and one space after it if there is one. The remaining lines form
    paragraphs separated by blank lines (empty or only whitespace); each paragraph's lines are
    stripped, joined with "\\n" and passed to inline() once.

        > Hello
        > *world*
        >
        > Bye

    becomes "<blockquote>\\n<p>{inline('Hello\\n*world*')}</p>\\n<p>{inline('Bye')}</p>\\n</blockquote>\\n".
    A quote with no text at all is "<blockquote>\\n</blockquote>\\n". Returns (html, index of
    the first line after the quote); None when line i does not start with ">".
    """
    j = i
    inner: List[str] = []
    while j < len(lines):
        m = _QUOTE.match(lines[j])
        if not m:
            break
        inner.append(lines[j][m.end():])
        j += 1
    if j == i:
        return None
    paras: List[List[str]] = [[]]
    for line in inner:
        if line.strip():
            paras[-1].append(line.strip())
        elif paras[-1]:
            paras.append([])
    out = ["<blockquote>\n"]
    for p in paras:
        if p:
            out.append(f"<p>{inline(chr(10).join(p))}</p>\n")
    out.append("</blockquote>\n")
    return "".join(out), j
