import unittest

from mugge_api import binding


class BindingTest(unittest.TestCase):
    def test_lib_path(self):
        self.assertTrue(binding.LIB_PATH.exists(), f"{binding.LIB_PATH} missing; run make -C libshort lib")

    def test_encode(self):
        self.assertEqual(binding.encode(0), "0")
        self.assertEqual(binding.encode(61), "Z")
        self.assertEqual(binding.encode(62), "10")
        self.assertEqual(binding.encode(123456789), "8m0Kx")
        self.assertEqual(binding.encode(2**64 - 1), "lYGhA16ahyf")

    def test_decode(self):
        self.assertEqual(binding.decode("0"), 0)
        self.assertEqual(binding.decode("10"), 62)
        self.assertEqual(binding.decode("8m0Kx"), 123456789)
        self.assertEqual(binding.decode("lYGhA16ahyf"), 2**64 - 1)

    def test_round_trip(self):
        for n in [1, 7, 62, 3843, 3844, 10**12, 2**40 + 3]:
            self.assertEqual(binding.decode(binding.encode(n)), n)

    def test_encode_rejects_out_of_range(self):
        for n in [-1, 2**64, 1.5, "3"]:
            with self.assertRaises(ValueError, msg=repr(n)):
                binding.encode(n)

    def test_decode_rejects_bad_codes(self):
        for code in ["", "a-b", "lYGhA16ahyg", None, 5]:
            with self.assertRaises(ValueError, msg=repr(code)):
                binding.decode(code)


if __name__ == "__main__":
    unittest.main()
