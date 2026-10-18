"""The `---` header at the top of a document."""

import re
from typing import Any, Dict, List, Tuple

_KEY = re.compile(r"^([A-Za-z0-9_-]+):(?:\s(.*)|)$")
_INT = re.compile(r"^-?\d+$")


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _value(v: str) -> Any:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v == "true":
        return True
    if v == "false":
        return False
    if _INT.match(v):
        return int(v)
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        items: List[str] = [_unquote(x.strip()) for x in inner.split(",")] if inner else []
        return items
    return v


def split_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a document into its front matter and its body.

    Front matter is present only when the first line is exactly "---" and a later line is
    exactly "---" (lines may end in "\\r\\n" or "\\n"). Between them, each line is blank, a
    comment (first non-space character "#"), or "key: value" where key is letters, digits,
    "_" or "-" and the value is the rest of the line, stripped. Values:

        "quoted" or 'quoted' -> the string inside the quotes
        true / false         -> True / False (lowercase only)
        an integer like -3   -> int
        [a, "b c", 3]        -> a list of strings, each item stripped and unquoted ([] is empty)
        anything else        -> the string itself (empty value: "")

    A later key replaces an earlier one. Any other line raises ValueError naming its line
    number in the document (the first "---" is line 1), e.g. "front matter line 3: no key".

    Returns (values, body), where body is everything after the closing "---" line. Without
    front matter: ({}, text) with text unchanged.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text
    end = next((k for k in range(1, len(lines)) if lines[k].rstrip("\r\n") == "---"), None)
    if end is None:
        return {}, text
    values: Dict[str, Any] = {}
    for k in range(1, end):
        line = lines[k].rstrip("\r\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = _KEY.match(line)
        if not m:
            raise ValueError(f"front matter line {k + 1}: no key")
        values[m.group(1)] = _value(m.group(2) or "")
    return values, "".join(lines[end + 1:])
