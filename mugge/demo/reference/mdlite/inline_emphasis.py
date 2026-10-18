"""Strong, emphasis and strikethrough."""

import re


def _body(c: str) -> str:
    edge = r"[^\s" + re.escape(c) + "]"
    return "(" + edge + "(?:.*?" + edge + ")??)"


_STEPS = [
    (re.compile(r"~~" + _body("~") + r"~~"), r"<del>\1</del>"),
    (re.compile(r"\*\*" + _body("*") + r"\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\w)__" + _body("_") + r"__(?!\w)"), r"<strong>\1</strong>"),
    (re.compile(r"\*" + _body("*") + r"\*"), r"<em>\1</em>"),
    (re.compile(r"(?<!\w)_" + _body("_") + r"_(?!\w)"), r"<em>\1</em>"),
]


def render_emphasis(text: str) -> str:
    """Render emphasis in text that is already HTML-escaped (escaping leaves * _ ~ alone).

    In this order, each step over the whole text, left to right, shortest match first:

    1. ~~x~~ -> <del>x</del>
    2. **x** -> <strong>x</strong>, and __x__ -> <strong>x</strong>
    3. *x*   -> <em>x</em>,         and _x_   -> <em>x</em>

    The content x is not empty, does not start or end with whitespace or with the delimiter's
    own character, and may contain markup made by an earlier step. A delimiter made of "_" must not touch a letter, digit or "_"
    on its outer side, so snake_case_names stay as they are; "*" and "~" have no such rule.
    Anything unmatched stays as written.

        render_emphasis("**bold** and *it*")  -> "<strong>bold</strong> and <em>it</em>"
        render_emphasis("***both***")         -> "<em><strong>both</strong></em>"
        render_emphasis("a * b * c")          -> "a * b * c"
        render_emphasis("snake_case_name")    -> "snake_case_name"
    """
    for pattern, repl in _STEPS:
        text = pattern.sub(repl, text)
    return text
