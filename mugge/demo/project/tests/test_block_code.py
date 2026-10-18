import re
import unittest
from pathlib import Path

from mdlite.block_code import parse


def boom(s):
    raise AssertionError("code is never passed to inline()")


class CodeBlocks(unittest.TestCase):
    def test_backticks_with_language(self):
        lines = ["```python", "def f(x):", "    return x < 1", "```", "after"]
        self.assertEqual(parse(lines, 0, boom), ('<pre><code class="language-python">def f(x):\n    return x &lt; 1\n</code></pre>\n', 4))

    def test_tildes_without_language(self):
        self.assertEqual(parse(["~~~", 'a "b" & c', "~~~"], 0, boom), ("<pre><code>a &quot;b&quot; &amp; c\n</code></pre>\n", 3))

    def test_info_string_first_word(self):
        self.assertEqual(parse(["``` js title=x", "1", "```"], 0, boom)[0], '<pre><code class="language-js">1\n</code></pre>\n')

    def test_closing_must_match(self):
        lines = ["````", "```", "~~~~", "````` ", "x"]
        self.assertEqual(parse(lines, 0, boom), ("<pre><code>```\n~~~~\n</code></pre>\n", 4))

    def test_unclosed_runs_to_end(self):
        self.assertEqual(parse(["```", "a", "", "b"], 0, boom), ("<pre><code>a\n\nb\n</code></pre>\n", 4))

    def test_empty_block(self):
        self.assertEqual(parse(["```", "```"], 0, boom), ("<pre><code></code></pre>\n", 2))

    def test_indent_kept(self):
        self.assertEqual(parse(["  ```", "  x", "   ```"], 0, boom), ("<pre><code>  x\n</code></pre>\n", 3))

    def test_not_fences(self):
        for line in ["``", "``` a`b", "    ```", "text ```", ""]:
            self.assertIsNone(parse([line, "```"], 0, boom), repr(line))
        self.assertIsNone(parse(["```"], 1, boom))


class Theme(unittest.TestCase):
    css = (Path(__file__).resolve().parent.parent / "mdlite" / "theme.css").read_text()

    def test_code_rules(self):
        self.assertRegex(self.css, re.compile(r"(^|\n)pre\s*\{[^}]*overflow-x\s*:\s*auto", re.S))
        self.assertRegex(self.css, re.compile(r"(^|\n)code\s*\{[^}]*font-family", re.S))

    def test_base_rules_kept(self):
        self.assertRegex(self.css, re.compile(r"(^|\n)body\s*\{[^}]*max-width", re.S))


if __name__ == "__main__":
    unittest.main()
