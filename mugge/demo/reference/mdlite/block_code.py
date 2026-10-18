"""Fenced code blocks."""

import re
from typing import List

from mdlite.types import BlockResult, Inline

_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A fenced code block opens with a line of three or more "`" or three or more "~"
    (after up to three spaces), optionally followed by an info string; the language is the
    info string's first word. It closes at the first later line that is, after up to three
    spaces, a run of the same character at least as long as the opening one, with only
    spaces after it. With no closing line the block runs to the end of the document. A "`"
    fence whose info string contains a "`" is not a fence.

    The content lines are kept as they are (no stripping, no inline()), joined with "\\n",
    with & < > " escaped as mdlite.escape.escape_html does (do not import it: this module
    must not depend on other mdlite modules, so do the four replacements here).

    Returns ('<pre><code class="language-{lang}">{content}\\n</code></pre>\\n', index after the
    closing fence), without the class attribute when there is no language, and with no "\\n"
    before "</code>" when the block has no content lines. None when line i is not an opening fence.

    This ticket also owns mdlite/theme.css: add the rules that style code blocks (see the
    ticket) and keep every rule already in the file.
    """
    if i >= len(lines):
        return None
    m = _OPEN.match(lines[i])
    if not m:
        return None
    fence, info = m.group(1), m.group(2)
    if fence[0] == "`" and "`" in info:
        return None
    lang = info.split()[0] if info.split() else ""
    close = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",} *$")
    j = i + 1
    while j < len(lines) and not close.match(lines[j]):
        j += 1
    body = lines[i + 1:j]
    content = _escape("\n".join(body)) + "\n" if body else ""
    cls = f' class="language-{_escape(lang)}"' if lang else ""
    return f"<pre><code{cls}>{content}</code></pre>\n", min(j + 1, len(lines))
