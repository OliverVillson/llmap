import unittest

from mdlite.escape import escape_attr, escape_html


class EscapeHtml(unittest.TestCase):
    def test_plain_text_is_unchanged(self):
        self.assertEqual(escape_html("hello world *_~`"), "hello world *_~`")

    def test_the_four_characters(self):
        self.assertEqual(escape_html('a < b > c & "d"'), "a &lt; b &gt; c &amp; &quot;d&quot;")

    def test_ampersand_first(self):
        self.assertEqual(escape_html("&lt;"), "&amp;lt;")

    def test_single_quote_is_left(self):
        self.assertEqual(escape_html("it's"), "it's")

    def test_empty(self):
        self.assertEqual(escape_html(""), "")


class EscapeAttr(unittest.TestCase):
    def test_quotes(self):
        self.assertEqual(escape_attr("""it's "x" """), "it&#39;s &quot;x&quot; ")

    def test_like_html_otherwise(self):
        self.assertEqual(escape_attr("a&b<c>"), "a&amp;b&lt;c&gt;")


if __name__ == "__main__":
    unittest.main()
