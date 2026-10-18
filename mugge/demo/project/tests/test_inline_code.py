import unittest

from mdlite.inline_code import split_code


class SplitCode(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(split_code("a `b` c"), [("text", "a "), ("code", "b"), ("text", " c")])
        self.assertEqual(split_code("``a`b``"), [("code", "a`b")])
        self.assertEqual(split_code("`` `x` ``"), [("code", "`x`")])

    def test_unclosed_run_is_text(self):
        self.assertEqual(split_code("`open"), [("text", "`open")])
        self.assertEqual(split_code("``` x ``"), [("text", "``` x ``")])

    def test_unclosed_then_closed(self):
        self.assertEqual(split_code("`` a `b`"), [("text", "`` a "), ("code", "b")])

    def test_empty_and_plain(self):
        self.assertEqual(split_code(""), [])
        self.assertEqual(split_code("no code"), [("text", "no code")])

    def test_only_spaces_kept(self):
        self.assertEqual(split_code("`  `"), [("code", "  ")])

    def test_one_side_space_kept(self):
        self.assertEqual(split_code("` a`"), [("code", " a")])

    def test_two_spans(self):
        self.assertEqual(split_code("`a` and `b`"), [("code", "a"), ("text", " and "), ("code", "b")])


if __name__ == "__main__":
    unittest.main()
