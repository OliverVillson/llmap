import unittest

from mdlite.render import render, to_html

DOC = """# Notes

Some *intro* text
over two lines.

## Setup

1. Install
2. Run `make`

```sh
echo "<hi>"
```

> Quoted **bold**

| a | b |
|---|--:|
| 1 | 2 |

* * *

- [x] done
"""

EXPECTED = (
    '<h1 id="notes">Notes</h1>\n'
    "<p>Some <em>intro</em> text\nover two lines.</p>\n"
    '<h2 id="setup">Setup</h2>\n'
    "<ol>\n<li>Install</li>\n<li>Run <code>make</code></li>\n</ol>\n"
    '<pre><code class="language-sh">echo &quot;&lt;hi&gt;&quot;\n</code></pre>\n'
    "<blockquote>\n<p>Quoted <strong>bold</strong></p>\n</blockquote>\n"
    "<table>\n<thead>\n<tr>\n<th>a</th>\n<th style=\"text-align:right\">b</th>\n</tr>\n</thead>\n"
    "<tbody>\n<tr>\n<td>1</td>\n<td style=\"text-align:right\">2</td>\n</tr>\n</tbody>\n</table>\n"
    "<hr>\n"
    '<ul>\n<li class="task"><input type="checkbox" disabled checked> done</li>\n</ul>\n'
)


class Render(unittest.TestCase):
    def test_document(self):
        self.assertEqual(to_html(DOC), EXPECTED)

    def test_headings_are_returned(self):
        self.assertEqual(render(DOC).headings, [(1, "Notes", "notes"), (2, "Setup", "setup")])

    def test_crlf_and_cr(self):
        self.assertEqual(to_html("# A\r\n\r\ntext\rmore"), '<h1 id="a">A</h1>\n<p>text\nmore</p>\n')

    def test_paragraph_then_list_without_blank_line(self):
        self.assertEqual(to_html("intro\n- a\n- b"), "<p>intro</p>\n<ul>\n<li>a</li>\n<li>b</li>\n</ul>\n")

    def test_rule_wins_over_list(self):
        self.assertEqual(to_html("* * *\n* item"), "<hr>\n<ul>\n<li>item</li>\n</ul>\n")

    def test_empty(self):
        self.assertEqual(to_html(""), "")
        self.assertEqual(to_html("\n  \n"), "")

    def test_each_document_has_its_own_ids(self):
        self.assertEqual(to_html("# A"), to_html("# A"))


if __name__ == "__main__":
    unittest.main()
