"""HTML escaping."""


def escape_html(text: str) -> str:
    """Escape text for HTML content.

    Replaces & with &amp;, < with &lt;, > with &gt; and " with &quot;, and nothing else.
    & goes first, so "&lt;" becomes "&amp;lt;" (escaping is not idempotent, on purpose).
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def escape_attr(text: str) -> str:
    """Escape text for a double-quoted HTML attribute value.

    The same as escape_html, plus ' becomes &#39;.
    """
    return escape_html(text).replace("'", "&#39;")
