import unittest

from mdlite.block_heading import HeadingParser
from mdlite.slug import Slugger


def up(s):
    return "[" + s.upper() + "]"


class Headings(unittest.TestCase):
    def setUp(self):
        self.p = HeadingParser(Slugger())

    def parse(self, line, i=0):
        return self.p.parse([line], i, up)

    def test_levels(self):
        self.assertEqual(self.parse("# Hello world"), ('<h1 id="hello-world">[HELLO WORLD]</h1>\n', 1))
        self.assertEqual(self.parse("###### Six"), ('<h6 id="six">[SIX]</h6>\n', 1))

    def test_not_headings(self):
        for line in ["####### Seven", "#tag", "", " # indented", "text # not"]:
            self.assertIsNone(self.parse(line), line)
        self.assertEqual(self.p.headings, [])

    def test_closing_hashes(self):
        self.assertEqual(self.parse("## Title ##"), ('<h2 id="title">[TITLE]</h2>\n', 1))
        self.assertEqual(self.parse("# C#"), ('<h1 id="c">[C#]</h1>\n', 1))
        self.assertEqual(self.parse("#   spaced out   #  "), ('<h1 id="spaced-out">[SPACED OUT]</h1>\n', 1))

    def test_empty_heading(self):
        self.assertEqual(self.parse("##"), ('<h2 id="section">[]</h2>\n', 1))

    def test_tab_separator(self):
        self.assertEqual(self.parse("#\tTabbed"), ('<h1 id="tabbed">[TABBED]</h1>\n', 1))

    def test_ids_are_unique_and_remembered(self):
        lines = ["# Intro", "text", "## Intro", "## *Usage*"]
        self.assertEqual(self.p.parse(lines, 0, up)[1], 1)
        self.assertIsNone(self.p.parse(lines, 1, up))
        self.assertEqual(self.p.parse(lines, 2, up), ('<h2 id="intro-1">[INTRO]</h2>\n', 3))
        self.p.parse(lines, 3, up)
        self.assertEqual(self.p.headings, [(1, "Intro", "intro"), (2, "Intro", "intro-1"), (2, "*Usage*", "usage")])

    def test_index_past_the_end(self):
        self.assertIsNone(self.p.parse(["# a"], 1, up))


if __name__ == "__main__":
    unittest.main()
