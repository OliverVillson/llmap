"""Hidden tests for Store (api/API.md)."""
import unittest

from mugge_api.store import Store


class HiddenStoreTest(unittest.TestCase):
    def test_many_ids_in_order(self):
        s = Store()
        for i in range(1, 501):
            self.assertEqual(s.add(f"https://example.com/{i}"), i)
        self.assertEqual(len(s), 500)
        for i in range(1, 501):
            self.assertEqual(s.get(i), f"https://example.com/{i}")
        self.assertIsNone(s.get(0))
        self.assertIsNone(s.get(501))

    def test_readding_does_not_use_an_id(self):
        s = Store()
        self.assertEqual(s.add("https://a.example"), 1)
        self.assertEqual(s.add("https://a.example"), 1)
        self.assertEqual(s.add("https://a.example"), 1)
        self.assertEqual(s.add("https://b.example"), 2)
        self.assertEqual(s.add("https://a.example/"), 3)  # a different string
        self.assertEqual(len(s), 3)

    def test_hits_are_per_id_and_survive_readding(self):
        s = Store()
        a = s.add("https://a.example")
        b = s.add("https://b.example")
        for _ in range(3):
            s.hit(a)
        self.assertEqual(s.hit(b), 1)
        self.assertEqual(s.hits(a), 3)
        self.assertEqual(s.hits(b), 1)
        self.assertEqual(s.add("https://a.example"), a)
        self.assertEqual(s.hits(a), 3)
        self.assertEqual(s.hit(a), 4)
        self.assertEqual(s.hits(b), 1)

    def test_unknown_ids(self):
        s = Store()
        for bad in [0, 1, -1, 2]:
            with self.assertRaises(KeyError):
                s.hit(bad)
            with self.assertRaises(KeyError):
                s.hits(bad)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.add("https://a.example"), 1)
        self.assertEqual(s.hits(1), 0)
        with self.assertRaises(KeyError):
            s.hit(2)
        self.assertEqual(len(s), 1)
        self.assertIsNone(s.get(2))


if __name__ == "__main__":
    unittest.main()
