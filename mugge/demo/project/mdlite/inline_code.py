"""Code spans."""

from typing import List, Tuple


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
    raise NotImplementedError
