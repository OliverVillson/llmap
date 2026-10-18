"""Links, images and autolinks."""

import re
from typing import Callable

_LINK = re.compile(
    r'(?P<img>!)?\[(?P<label>[^\[\]]*)\]\((?P<dest>[^\s()]+)(?:\s+"(?P<title>[^"]*)")?\)'
    r"|<(?P<auto>https?://[^\s<>]+)>"
)


def render_links(text: str, inner: Callable[[str], str], attr: Callable[[str], str]) -> str:
    """Render the links in a piece of raw (unescaped) text.

    - Image: ![alt](src) or ![alt](src "title")
      -> <img src="{attr(src)}" alt="{attr(alt)}"> (with ` title="{attr(title)}"` after alt)
    - Link: [label](href) or [label](href "title")
      -> <a href="{attr(href)}">{inner(label)}</a> (with ` title="{attr(title)}"` after href)
    - Autolink: <http://...> or <https://...> (no spaces, < or > inside)
      -> <a href="{attr(url)}">{inner(url)}</a>

    Labels and alt text contain no "[" or "]"; a destination is non-empty and has no spaces and
    no parentheses; a title is in double quotes after at least one space and has no double
    quote inside. Anything that does not match is plain text. All the plain text between links
    goes through inner() as one piece per gap (never character by character), and inner() is
    never called on an empty string.

        render_links("see [docs](/d) now", str.upper, str) -> 'SEE <a href="/d">DOCS</a> NOW'
    """
    out = []
    pos = 0
    for m in _LINK.finditer(text):
        if m.start() > pos:
            out.append(inner(text[pos:m.start()]))
        if m.group("auto"):
            url = m.group("auto")
            out.append(f'<a href="{attr(url)}">{inner(url)}</a>')
        else:
            title = m.group("title")
            t = f' title="{attr(title)}"' if title is not None else ""
            if m.group("img"):
                out.append(f'<img src="{attr(m.group("dest"))}" alt="{attr(m.group("label"))}"{t}>')
            else:
                label = m.group("label")
                out.append(f'<a href="{attr(m.group("dest"))}"{t}>{inner(label) if label else ""}</a>')
        pos = m.end()
    if pos < len(text):
        out.append(inner(text[pos:]))
    return "".join(out)
