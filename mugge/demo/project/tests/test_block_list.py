import unittest

from mdlite.block_list import parse


def up(s):
    return s.upper()


class Lists(unittest.TestCase):
    def test_bullets(self):
        lines = ["- one", "- two", "", "- three"]
        self.assertEqual(parse(lines, 0, up), ("<ul>\n<li>ONE</li>\n<li>TWO</li>\n</ul>\n", 2))

    def test_numbers(self):
        self.assertEqual(parse(["1. a", "2. b", "10. c"], 0, up), ("<ol>\n<li>A</li>\n<li>B</li>\n<li>C</li>\n</ol>\n", 3))

    def test_start_number(self):
        self.assertEqual(parse(["007) x"], 0, up), ('<ol start="7">\n<li>X</li>\n</ol>\n', 1))

    def test_kind_change_ends_list(self):
        self.assertEqual(parse(["- a", "* b"], 0, up)[1], 1)
        self.assertEqual(parse(["1. a", "2) b"], 0, up)[1], 1)
        self.assertEqual(parse(["- a", "1. b"], 0, up)[1], 1)

    def test_continuation_lines(self):
        lines = ["* first", "  more text  ", "    deeper", "* second", "not indented"]
        self.assertEqual(parse(lines, 0, up), ("<ul>\n<li>FIRST\nMORE TEXT\nDEEPER</li>\n<li>SECOND</li>\n</ul>\n", 4))

    def test_tasks(self):
        lines = ["- [ ] todo", "- [x] done", "- [X] also", "- [] plain"]
        self.assertEqual(parse(lines, 0, up), ("<ul>\n"
            '<li class="task"><input type="checkbox" disabled> TODO</li>\n'
            '<li class="task"><input type="checkbox" disabled checked> DONE</li>\n'
            '<li class="task"><input type="checkbox" disabled checked> ALSO</li>\n'
            "<li>[] PLAIN</li>\n</ul>\n", 4))

    def test_indented_markers(self):
        self.assertEqual(parse(["   + a", " + b"], 0, up), ("<ul>\n<li>A</li>\n<li>B</li>\n</ul>\n", 2))

    def test_not_lists(self):
        for line in ["-no space", "1.no space", "    - code", "1234567890. big", "text", ""]:
            self.assertIsNone(parse([line], 0, up), repr(line))
        self.assertIsNone(parse(["- a"], 1, up))


if __name__ == "__main__":
    unittest.main()
