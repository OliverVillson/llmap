import unittest

from mdlite.inline import render_inline as r


class Inline(unittest.TestCase):
    def test_example(self):
        self.assertEqual(r('**a** `<b>` [c](/d "e")'), '<strong>a</strong> <code>&lt;b&gt;</code> <a href="/d" title="e">c</a>')

    def test_code_is_not_emphasised(self):
        self.assertEqual(r("`*not em*` *em*"), "<code>*not em*</code> <em>em</em>")

    def test_links_inside_text_with_emphasis(self):
        self.assertEqual(r("*see* [the **docs**](https://x.io/?a=1&b=2)"),
                         '<em>see</em> <a href="https://x.io/?a=1&amp;b=2">the <strong>docs</strong></a>')

    def test_escaping(self):
        self.assertEqual(r('a < b & "c" > d'), "a &lt; b &amp; &quot;c&quot; &gt; d")
        self.assertEqual(r("![it's \"x\"](a.png)"), '<img src="a.png" alt="it&#39;s &quot;x&quot;">')

    def test_autolink(self):
        self.assertEqual(r("<https://a.io/x_y_z>"), '<a href="https://a.io/x_y_z">https://a.io/x_y_z</a>')

    def test_emphasis_across_a_link_gap_is_not_joined(self):
        self.assertEqual(r("*a [b](c) d*"), '*a <a href="c">b</a> d*')

    def test_plain(self):
        self.assertEqual(r("plain text"), "plain text")
        self.assertEqual(r(""), "")


if __name__ == "__main__":
    unittest.main()
