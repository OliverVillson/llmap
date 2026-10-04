import unittest

from mugge_api.store import Store


class StoreTest(unittest.TestCase):
    def test_ids_count_up_from_one(self):
        s = Store()
        self.assertEqual(len(s), 0)
        self.assertEqual(s.add("https://a.example"), 1)
        self.assertEqual(s.add("https://b.example"), 2)
        self.assertEqual(len(s), 2)

    def test_same_url_same_id(self):
        s = Store()
        a = s.add("https://a.example")
        s.add("https://b.example")
        self.assertEqual(s.add("https://a.example"), a)
        self.assertEqual(len(s), 2)

    def test_get(self):
        s = Store()
        i = s.add("https://a.example")
        self.assertEqual(s.get(i), "https://a.example")
        self.assertIsNone(s.get(99))

    def test_hits(self):
        s = Store()
        i = s.add("https://a.example")
        self.assertEqual(s.hits(i), 0)
        self.assertEqual(s.hit(i), 1)
        self.assertEqual(s.hit(i), 2)
        self.assertEqual(s.hits(i), 2)
        with self.assertRaises(KeyError):
            s.hit(99)
        with self.assertRaises(KeyError):
            s.hits(99)

    def test_stores_are_independent(self):
        a, b = Store(), Store()
        a.add("https://a.example")
        self.assertEqual(len(b), 0)
        self.assertEqual(b.add("https://b.example"), 1)


if __name__ == "__main__":
    unittest.main()
