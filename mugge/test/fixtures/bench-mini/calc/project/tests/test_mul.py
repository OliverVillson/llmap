import unittest
from calc.mul import mul


class MulTest(unittest.TestCase):
    def test_small(self):
        self.assertEqual(mul(3, 4), 12)
