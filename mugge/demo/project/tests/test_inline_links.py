import unittest

from mdlite.inline_links import render_links

calls = []


def inner(s):
    calls.append(s)
    return s.upper()


def attr(s):
    return s.replace('"', "&quot;").replace("&", "+")


class RenderLinks(unittest.TestCase):
    def setUp(self):
        calls.clear()

    def test_example(self):
        self.assertEqual(render_links("see [docs](/d) now", inner, attr), 'SEE <a href="/d">DOCS</a> NOW')
        self.assertEqual(calls, ["see ", "docs", " now"])

    def test_link_with_title(self):
        self.assertEqual(
            render_links('[a](http://x.io/?q=1&r=2 "The title")', inner, attr),
            '<a href="http://x.io/?q=1+r=2" title="The title">A</a>',
        )

    def test_image(self):
        self.assertEqual(render_links("![a cat](cat.png)", inner, attr), '<img src="cat.png" alt="a cat">')
        self.assertEqual(calls, [])

    def test_image_with_title(self):
        self.assertEqual(
            render_links('x ![c](c.png "big") y', inner, attr),
            'X <img src="c.png" alt="c" title="big"> Y',
        )

    def test_autolink(self):
        self.assertEqual(
            render_links("go <https://example.com/a_b> now", inner, attr),
            'GO <a href="https://example.com/a_b">HTTPS://EXAMPLE.COM/A_B</a> NOW',
        )

    def test_not_links(self):
        for s in ["[a] (b)", "[a](b c)", "<ftp://x>", "<https://a b>", "[a]()", "plain"]:
            calls.clear()
            self.assertEqual(render_links(s, inner, attr), s.upper(), s)
            self.assertEqual(calls, [s], s)

    def test_two_links_and_no_empty_calls(self):
        self.assertEqual(render_links("[a](1)[b](2)", inner, attr), '<a href="1">A</a><a href="2">B</a>')
        self.assertEqual(calls, ["a", "b"])

    def test_empty(self):
        self.assertEqual(render_links("", inner, attr), "")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
