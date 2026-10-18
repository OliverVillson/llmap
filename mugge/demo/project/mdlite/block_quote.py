"""Block quotes."""

from typing import List

from mdlite.types import BlockResult, Inline


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A block quote is line i and the lines right after it that start with ">" after up to
    three spaces; it ends at the first line that does not. From each line remove the spaces
    before ">", the ">" and one space after it if there is one. The remaining lines form
    paragraphs separated by blank lines (empty or only whitespace); each paragraph's lines are
    stripped, joined with "\\n" and passed to inline() once.

        > Hello
        > *world*
        >
        > Bye

    becomes "<blockquote>\\n<p>{inline('Hello\\n*world*')}</p>\\n<p>{inline('Bye')}</p>\\n</blockquote>\\n".
    A quote with no text at all is "<blockquote>\\n</blockquote>\\n". Returns (html, index of
    the first line after the quote); None when line i does not start with ">".
    """
    raise NotImplementedError
