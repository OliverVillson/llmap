"""Table of contents."""

from typing import List

from mdlite.types import Heading


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_toc(headings: List[Heading], max_level: int = 3) -> str:
    """A nested list of links to the headings of level <= max_level, as a <nav>.

    Nesting follows the order of the headings: a heading deeper than the one before opens one
    nested list (however many levels deeper it is: h1 then h3 nests once); one at the same
    level is a sibling; a shallower one closes lists until it reaches a list whose first
    heading is at its level or shallower, and becomes a sibling there. Link text is the
    heading text with & < > " escaped (as mdlite.escape.escape_html; do not import other
    mdlite modules); the href is "#" + id. One tag per line, a nested list opening right
    after its parent's link:

        <nav class="toc">
        <ul>
        <li><a href="#a">A</a>
        <ul>
        <li><a href="#b">B</a></li>
        </ul>
        </li>
        <li><a href="#c">C</a></li>
        </ul>
        </nav>

    for [(1, "A", "a"), (2, "B", "b"), (1, "C", "c")]. Ends with "\\n". No headings to show:
    "" (the empty string).
    """
    shown = [h for h in headings if h[0] <= max_level]
    if not shown:
        return ""
    out = ['<nav class="toc">', "<ul>"]
    stack = [shown[0][0]]  # the first heading level of each open list
    prev = shown[0][0]
    for k, (level, text, hid) in enumerate(shown):
        if k:
            if level > prev:
                stack.append(level)
            else:
                out[-1] += "</li>"
                while len(stack) > 1 and stack[-1] > level:
                    stack.pop()
                    out += ["</ul>", "</li>"]
        out.append(f'<li><a href="#{hid}">{_escape(text)}</a>')
        if k + 1 < len(shown) and shown[k + 1][0] > level:
            out.append("<ul>")
        prev = level
    out[-1] += "</li>"
    while len(stack) > 1:
        stack.pop()
        out += ["</ul>", "</li>"]
    out += ["</ul>", "</nav>"]
    return "\n".join(out) + "\n"
