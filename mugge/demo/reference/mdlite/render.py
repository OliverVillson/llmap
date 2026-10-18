"""The whole document."""

from typing import List, NamedTuple

from mdlite import block_code, block_list, block_para, block_quote, block_rule, block_table
from mdlite.block_heading import HeadingParser
from mdlite.inline import render_inline
from mdlite.slug import Slugger
from mdlite.types import Heading


class Rendered(NamedTuple):
    html: str
    #: Every heading in the document, in order (from HeadingParser.headings).
    headings: List[Heading]


def render(markdown: str) -> Rendered:
    """Render a Markdown document (no front matter; see frontmatter.py) to HTML.

    Normalise "\\r\\n" and "\\r" to "\\n", then split into lines on "\\n". Make one Slugger and
    one HeadingParser for the document. Walk the lines as ARCHITECTURE.md's "The block
    contract" says: skip blank lines (empty or only whitespace); at any other line try, in
    order, block_code.parse, the HeadingParser's parse, block_rule.parse, block_table.parse,
    block_quote.parse, block_list.parse and block_para.parse, each with inline=render_inline,
    and take the first that returns a result. The HTML is the blocks' HTML concatenated
    ("" for an empty document).
    """
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings = HeadingParser(Slugger())
    parsers = [block_code.parse, headings.parse, block_rule.parse, block_table.parse, block_quote.parse, block_list.parse, block_para.parse]
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        for parse in parsers:
            got = parse(lines, i, render_inline)
            if got is not None:
                html, i = got
                out.append(html)
                break
        else:
            raise RuntimeError(f"no block parser took line {i + 1}")
    return Rendered("".join(out), headings.headings)


def to_html(markdown: str) -> str:
    """render(markdown).html"""
    return render(markdown).html
