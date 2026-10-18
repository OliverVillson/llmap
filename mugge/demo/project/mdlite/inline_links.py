"""Links, images and autolinks."""

from typing import Callable


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
    raise NotImplementedError
