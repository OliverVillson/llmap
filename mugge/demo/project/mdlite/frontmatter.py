"""The `---` header at the top of a document."""

from typing import Any, Dict, Tuple


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
    raise NotImplementedError
