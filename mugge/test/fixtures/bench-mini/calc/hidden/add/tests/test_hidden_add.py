import unittest
from calc.add import add


class HiddenAddTest(unittest.TestCase):
    def test_more(self):
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0.5, 0.25), 0.75)
