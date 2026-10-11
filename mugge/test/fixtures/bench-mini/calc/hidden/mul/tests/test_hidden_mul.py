import unittest
from calc.mul import mul


class HiddenMulTest(unittest.TestCase):
    def test_more(self):
        self.assertEqual(mul(0, 5), 0)
        self.assertEqual(mul(7, -2), -14)
