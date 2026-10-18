import unittest

from mdlite.inline_emphasis import render_emphasis as em


class Emphasis(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(em("**bold** and *it*"), "<strong>bold</strong> and <em>it</em>")
        self.assertEqual(em("***both***"), "<em><strong>both</strong></em>")
        self.assertEqual(em("a * b * c"), "a * b * c")
        self.assertEqual(em("snake_case_name"), "snake_case_name")

    def test_underscores(self):
        self.assertEqual(em("__strong__ _em_"), "<strong>strong</strong> <em>em</em>")
        self.assertEqual(em("call my_func_name or _this_"), "call my_func_name or <em>this</em>")

    def test_strike(self):
        self.assertEqual(em("~~gone~~ but ~ not ~"), "<del>gone</del> but ~ not ~")

    def test_shortest_match(self):
        self.assertEqual(em("*a* b *c*"), "<em>a</em> b <em>c</em>")
        self.assertEqual(em("**a** **b**"), "<strong>a</strong> <strong>b</strong>")

    def test_nested_markup(self):
        self.assertEqual(em("**bold with *it* inside**"), "<strong>bold with <em>it</em> inside</strong>")
        self.assertEqual(em("~~*x*~~"), "<del><em>x</em></del>")

    def test_unmatched(self):
        self.assertEqual(em("**open"), "**open")
        self.assertEqual(em("2 * 3 = 6"), "2 * 3 = 6")
        self.assertEqual(em("** x**"), "** x**")

    def test_escaped_text(self):
        self.assertEqual(em("&lt;*b*&gt;"), "&lt;<em>b</em>&gt;")

    def test_one_char(self):
        self.assertEqual(em("*a*"), "<em>a</em>")


if __name__ == "__main__":
    unittest.main()
