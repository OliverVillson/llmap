import unittest

from mdlite.toc import render_toc


class Toc(unittest.TestCase):
    def test_example(self):
        self.assertEqual(render_toc([(1, "A", "a"), (2, "B", "b"), (1, "C", "c")]), "\n".join([
            '<nav class="toc">', "<ul>",
            '<li><a href="#a">A</a>', "<ul>", '<li><a href="#b">B</a></li>', "</ul>", "</li>",
            '<li><a href="#c">C</a></li>',
            "</ul>", "</nav>"]) + "\n")

    def test_flat(self):
        self.assertEqual(render_toc([(2, "x & y", "x-y"), (2, "<z>", "z")]),
                         '<nav class="toc">\n<ul>\n<li><a href="#x-y">x &amp; y</a></li>\n<li><a href="#z">&lt;z&gt;</a></li>\n</ul>\n</nav>\n')

    def test_jump_nests_once_and_closes_all(self):
        self.assertEqual(render_toc([(1, "A", "a"), (3, "B", "b"), (3, "C", "c")]), "\n".join([
            '<nav class="toc">', "<ul>",
            '<li><a href="#a">A</a>', "<ul>", '<li><a href="#b">B</a></li>', '<li><a href="#c">C</a></li>', "</ul>", "</li>",
            "</ul>", "</nav>"]) + "\n")

    def test_deep_then_back(self):
        self.assertEqual(render_toc([(1, "A", "a"), (2, "B", "b"), (3, "C", "c"), (2, "D", "d"), (1, "E", "e")]), "\n".join([
            '<nav class="toc">', "<ul>",
            '<li><a href="#a">A</a>', "<ul>",
            '<li><a href="#b">B</a>', "<ul>", '<li><a href="#c">C</a></li>', "</ul>", "</li>",
            '<li><a href="#d">D</a></li>', "</ul>", "</li>",
            '<li><a href="#e">E</a></li>',
            "</ul>", "</nav>"]) + "\n")

    def test_starts_deep(self):
        self.assertEqual(render_toc([(2, "A", "a"), (1, "B", "b")]),
                         '<nav class="toc">\n<ul>\n<li><a href="#a">A</a></li>\n<li><a href="#b">B</a></li>\n</ul>\n</nav>\n')

    def test_max_level(self):
        self.assertEqual(render_toc([(4, "deep", "d")]), "")
        self.assertEqual(render_toc([(1, "A", "a"), (4, "deep", "d")], max_level=1),
                         '<nav class="toc">\n<ul>\n<li><a href="#a">A</a></li>\n</ul>\n</nav>\n')
        self.assertEqual(render_toc([]), "")


if __name__ == "__main__":
    unittest.main()
