"""Pipe tables."""

from typing import List

from mdlite.types import BlockResult, Inline


def split_row(line: str) -> List[str]:
    """The cells of one table row: strip the line, drop one leading and one trailing "|",
    split on every "|" not preceded by a backslash, turn each "\\|" into "|" and strip each cell.

        split_row("| a | b \\\\| c |") -> ["a", "b | c"]
        split_row("a|b")             -> ["a", "b"]
    """
    raise NotImplementedError


def parse(lines: List[str], i: int, inline: Inline) -> BlockResult:
    """A table is a header row at line i (a line with at least one "|"), then a delimiter row
    whose cells are each dashes with an optional ":" on either side (":--", "--:", ":-:",
    "---"), and as many delimiter cells as header cells. Body rows follow until a blank line,
    a line without "|", or the end. A body row with fewer cells is padded with empty cells; extra
    cells are dropped. Alignment: ":-" left, "-:" right, ":-:" center, none for plain dashes;
    an aligned cell gets ` style="text-align:{left|right|center}"` on its th and every td.

    Every cell's text goes through inline(). The HTML, one tag per line:

        <table>
        <thead>
        <tr>
        <th>Name</th>
        <th style="text-align:right">Qty</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td>apples</td>
        <td style="text-align:right">3</td>
        </tr>
        </tbody>
        </table>

    with no <tbody> section at all when there are no body rows. Returns (html ending in
    "\\n", index after the last body row); None when line i does not start a table.

    This ticket also owns mdlite/theme.css: add the rules that style tables (see the ticket)
    and keep every rule already in the file.
    """
    raise NotImplementedError
