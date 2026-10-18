import unittest

from mdlite.frontmatter import split_front_matter as split


class FrontMatter(unittest.TestCase):
    def test_values(self):
        doc = "---\ntitle: \"Hello: world\"\nsub: 'single'\ndraft: true\npublic: false\nTrue: True\norder: -3\ntags: [a, \"b c\", 3]\nnone: []\nplain: some text \nempty:\n---\n# Body\n"
        meta, body = split(doc)
        self.assertEqual(meta, {"title": "Hello: world", "sub": "single", "draft": True, "public": False, "True": "True",
                                "order": -3, "tags": ["a", "b c", "3"], "none": [], "plain": "some text", "empty": ""})
        self.assertEqual(body, "# Body\n")

    def test_comments_blanks_and_repeats(self):
        meta, body = split("---\n# a comment\n\n  # indented comment\nk: 1\nk: 2\n---\n")
        self.assertEqual(meta, {"k": 2})
        self.assertEqual(body, "")

    def test_crlf(self):
        self.assertEqual(split("---\r\na-b_c: x\r\n---\r\nbody\r\n"), ({"a-b_c": "x"}, "body\r\n"))

    def test_no_front_matter(self):
        for doc in ["# Title\n---\n", "--- \nk: v\n---\n", "---\nk: v\n", "", "----\n---\n"]:
            self.assertEqual(split(doc), ({}, doc), repr(doc))

    def test_errors_name_the_line(self):
        with self.assertRaises(ValueError) as e:
            split("---\nok: 1\nthis is not a key\n---\n")
        self.assertIn("line 3", str(e.exception))
        with self.assertRaises(ValueError) as e:
            split("---\nbad key: 1\n---\n")
        self.assertIn("line 2", str(e.exception))


if __name__ == "__main__":
    unittest.main()
