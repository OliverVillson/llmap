"""The inline pipeline."""

from mdlite.escape import escape_attr, escape_html
from mdlite.inline_code import split_code
from mdlite.inline_emphasis import render_emphasis
from mdlite.inline_links import render_links


def _inner(text: str) -> str:
    return render_emphasis(escape_html(text))


def render_inline(text: str) -> str:
    """Render inline Markdown and escape HTML, as ARCHITECTURE.md's "Inline pipeline" says:
    split_code() first; a code part becomes "<code>" + escape_html(part) + "</code>"; a text
    part goes through render_links(part, inner, escape_attr) with
    inner(t) = render_emphasis(escape_html(t)).

        render_inline('**a** `<b>` [c](/d "e")')
        -> '<strong>a</strong> <code>&lt;b&gt;</code> <a href="/d" title="e">c</a>'
    """
    out = []
    for kind, part in split_code(text):
        if kind == "code":
            out.append(f"<code>{escape_html(part)}</code>")
        else:
            out.append(render_links(part, _inner, escape_attr))
    return "".join(out)
