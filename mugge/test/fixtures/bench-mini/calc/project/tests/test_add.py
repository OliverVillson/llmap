import unittest
from calc.add import add


class AddTest(unittest.TestCase):
    def test_small(self):
        self.assertEqual(add(2, 3), 5)
