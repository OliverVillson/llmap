import unittest

from mdlite.block_rule import parse


def boom(s):
    raise AssertionError("a rule has no inline text")


class Rules(unittest.TestCase):
    def test_rules(self):
        for line in ["---", "***", "___", "* * *", " - - - - ", "_____", "\t*\t*\t*"]:
            self.assertEqual(parse(["x", line, "y"], 1, boom), ("<hr>\n", 2), repr(line))

    def test_not_rules(self):
        for line in ["--", "-*-", "--- x", "- item", "", "**bold**", "==="]:
            self.assertIsNone(parse([line], 0, boom), repr(line))

    def test_past_the_end(self):
        self.assertIsNone(parse(["---"], 1, boom))


if __name__ == "__main__":
    unittest.main()
