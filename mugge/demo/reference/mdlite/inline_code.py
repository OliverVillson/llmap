"""Code spans."""

import re
from typing import List, Tuple

_RUN = re.compile(r"`+")


def split_code(text: str) -> List[Tuple[str, str]]:
    """Cut a line into ("text", s) and ("code", s) parts, in order.

    A code span opens with a run of N backticks and closes at the next run of exactly N
    backticks; a longer or shorter run inside does not close it. If the content both begins
    and ends with a space and is not all spaces, one space is stripped from each end. An
    opening run that is never closed is plain text (scanning goes on right after it).
    Neighbouring text is merged into one part, and no part is an empty text part.

        split_code("a `b` c")  -> [("text", "a "), ("code", "b"), ("text", " c")]
        split_code("``a`b``")  -> [("code", "a`b")]
        split_code("`` `x` ``") -> [("code", "`x`")]
        split_code("`open")    -> [("text", "`open")]
        split_code("")         -> []
    """
    parts: List[Tuple[str, str]] = []

    def add_text(s: str) -> None:
        if not s:
            return
        if parts and parts[-1][0] == "text":
            parts[-1] = ("text", parts[-1][1] + s)
        else:
            parts.append(("text", s))

    i = 0
    while i < len(text):
        m = _RUN.search(text, i)
        if not m:
            add_text(text[i:])
            break
        add_text(text[i:m.start()])
        n = len(m.group())
        close = None
        for c in _RUN.finditer(text, m.end()):
            if len(c.group()) == n:
                close = c
                break
        if close is None:
            add_text(m.group())
            i = m.end()
            continue
        content = text[m.end():close.start()]
        if len(content) >= 2 and content[0] == " " and content[-1] == " " and content.strip(" "):
            content = content[1:-1]
        parts.append(("code", content))
        i = close.end()
    return parts
