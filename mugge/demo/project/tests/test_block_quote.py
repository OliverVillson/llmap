import unittest

from mdlite.block_quote import parse

calls = []


def up(s):
    calls.append(s)
    return s.upper()


class Quotes(unittest.TestCase):
    def setUp(self):
        calls.clear()

    def test_example(self):
        lines = ["> Hello", "> *world*", ">", "> Bye", "after"]
        self.assertEqual(parse(lines, 0, up), ("<blockquote>\n<p>HELLO\n*WORLD*</p>\n<p>BYE</p>\n</blockquote>\n", 4))
        self.assertEqual(calls, ["Hello\n*world*", "Bye"])

    def test_spacing(self):
        self.assertEqual(parse(["   >no space", ">  two   "], 0, up), ("<blockquote>\n<p>NO SPACE\nTWO</p>\n</blockquote>\n", 2))

    def test_ends_at_first_other_line(self):
        self.assertEqual(parse(["> a", "", "> b"], 0, up), ("<blockquote>\n<p>A</p>\n</blockquote>\n", 1))
        self.assertEqual(parse(["> a", "lazy"], 0, up)[1], 1)

    def test_many_blank_lines(self):
        self.assertEqual(parse([">", "> a", ">", ">  ", "> b", ">"], 0, up), ("<blockquote>\n<p>A</p>\n<p>B</p>\n</blockquote>\n", 6))

    def test_empty_quote(self):
        self.assertEqual(parse([">"], 0, up), ("<blockquote>\n</blockquote>\n", 1))
        self.assertEqual(calls, [])

    def test_not_quotes(self):
        for line in ["    > code", "a > b", ""]:
            self.assertIsNone(parse([line], 0, up), repr(line))
        self.assertIsNone(parse(["> a"], 1, up))


if __name__ == "__main__":
    unittest.main()
