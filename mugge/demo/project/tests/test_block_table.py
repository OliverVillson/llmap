import re
import unittest
from pathlib import Path

from mdlite.block_table import parse, split_row


def up(s):
    return s.upper()


class SplitRow(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(split_row("| a | b \\| c |"), ["a", "b | c"])
        self.assertEqual(split_row("a|b"), ["a", "b"])
        self.assertEqual(split_row("|  x  |"), ["x"])
        self.assertEqual(split_row("| a | |"), ["a", ""])


class Tables(unittest.TestCase):
    def test_full_table(self):
        lines = ["| Name | Qty |", "|------|----:|", "| apples | 3 |", "| pears | 10 |", "", "after"]
        self.assertEqual(parse(lines, 0, up), ("\n".join([
            "<table>", "<thead>", "<tr>", "<th>NAME</th>", '<th style="text-align:right">QTY</th>', "</tr>", "</thead>",
            "<tbody>",
            "<tr>", "<td>APPLES</td>", '<td style="text-align:right">3</td>', "</tr>",
            "<tr>", "<td>PEARS</td>", '<td style="text-align:right">10</td>', "</tr>",
            "</tbody>", "</table>"]) + "\n", 4))

    def test_alignments_and_no_outer_pipes(self):
        html, end = parse(["a | b | c", ":-- | :-: | ---"], 0, up)
        self.assertEqual(end, 2)
        self.assertIn('<th style="text-align:left">A</th>', html)
        self.assertIn('<th style="text-align:center">B</th>', html)
        self.assertIn("<th>C</th>", html)
        self.assertNotIn("tbody", html)

    def test_padding_and_dropping(self):
        html, end = parse(["|a|b|", "|-|-|", "|1|", "|1|2|3|", "no pipe here"], 0, up)
        self.assertEqual(end, 4)
        self.assertEqual(html.count("<td>"), 4)
        self.assertIn("<tr>\n<td>1</td>\n<td></td>\n</tr>", html)
        self.assertNotIn("3", html)

    def test_not_tables(self):
        cases = [
            ["| a | b |", "| - |"],
            ["| a | b |", "| x | y |"],
            ["no pipes", "---"],
            ["| a |"],
            ["| a | b |", ""],
        ]
        for lines in cases:
            self.assertIsNone(parse(lines, 0, up), lines)


class Theme(unittest.TestCase):
    css = (Path(__file__).resolve().parent.parent / "mdlite" / "theme.css").read_text()

    def test_table_rules(self):
        self.assertRegex(self.css, re.compile(r"(^|\n)table\s*\{[^}]*border-collapse\s*:\s*collapse", re.S))
        self.assertRegex(self.css, re.compile(r"(^|\n)th,\s*td\s*\{[^}]*padding", re.S))

    def test_base_rules_kept(self):
        self.assertRegex(self.css, re.compile(r"(^|\n)body\s*\{[^}]*max-width", re.S))


if __name__ == "__main__":
    unittest.main()
