import unittest

from mdlite.slug import Slugger, slugify


class Slugify(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("  Café -- au  lait "), "café-au-lait")
        self.assertEqual(slugify("snake_case name"), "snake_case-name")

    def test_empty_becomes_section(self):
        self.assertEqual(slugify("!!!"), "section")
        self.assertEqual(slugify(""), "section")

    def test_markup_characters_go(self):
        self.assertEqual(slugify("Using `render()` *fast*"), "using-render-fast")

    def test_tabs_and_digits(self):
        self.assertEqual(slugify("Step\t2:  Build"), "step-2-build")


class SluggerTest(unittest.TestCase):
    def test_repeats_are_numbered(self):
        s = Slugger()
        self.assertEqual([s.slug("Intro"), s.slug("Intro"), s.slug("intro!")], ["intro", "intro-1", "intro-2"])

    def test_skips_taken_candidates(self):
        s = Slugger()
        self.assertEqual(s.slug("Intro 1"), "intro-1")
        self.assertEqual(s.slug("Intro"), "intro")
        self.assertEqual(s.slug("Intro"), "intro-2")

    def test_each_slugger_starts_fresh(self):
        self.assertEqual(Slugger().slug("A"), "a")
        self.assertEqual(Slugger().slug("A"), "a")


if __name__ == "__main__":
    unittest.main()
