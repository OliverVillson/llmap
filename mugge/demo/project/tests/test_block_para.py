import unittest

from mdlite.block_para import parse, starts_block

calls = []


def up(s):
    calls.append(s)
    return s.upper()


class Paragraphs(unittest.TestCase):
    def setUp(self):
        calls.clear()

    def test_one_line(self):
        self.assertEqual(parse(["hello"], 0, up), ("<p>HELLO</p>\n", 1))

    def test_lines_until_blank(self):
        lines = ["  first line ", "second", "", "next"]
        self.assertEqual(parse(lines, 0, up), ("<p>FIRST LINE\nSECOND</p>\n", 2))
        self.assertEqual(calls, ["first line\nsecond"])

    def test_whitespace_line_is_blank(self):
        self.assertEqual(parse(["a", " \t ", "b"], 0, up), ("<p>A</p>\n", 1))
        self.assertIsNone(parse(["a", "   "], 1, up))

    def test_stops_before_other_blocks(self):
        for starter in ["# Head", "```py", "~~~", "> quote", "- item", "* item", "+ item", "12. item", "3) item", "---", "* * *", "##"]:
            self.assertEqual(parse(["text", starter], 0, up), ("<p>TEXT</p>\n", 1), starter)

    def test_does_not_stop(self):
        for line in ["#tag", "-dash", "2.5 apples", "1.item", "a > b", "--", "| a | b |"]:
            self.assertEqual(parse(["text", line], 0, up)[1], 2, line)

    def test_starts_block(self):
        self.assertTrue(starts_block("   > deep"))
        self.assertFalse(starts_block("    > code-indented"))
        self.assertFalse(starts_block(""))

    def test_blank_or_past_end(self):
        self.assertIsNone(parse([""], 0, up))
        self.assertIsNone(parse(["a"], 1, up))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
