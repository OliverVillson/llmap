import json
import unittest

from mugge_api.handlers import App


def post(app, path, obj):
    return app.handle("POST", path, json.dumps(obj).encode())


class HandlersTest(unittest.TestCase):
    def setUp(self):
        self.app = App()  # real Store, real libshort binding

    def test_health(self):
        self.assertEqual(self.app.handle("GET", "/health"), (200, {"ok": True}))

    def test_shorten_and_lookup(self):
        status, body = post(self.app, "/shorten", {"url": "https://a.example/x"})
        self.assertEqual((status, body), (201, {"code": "1", "url": "https://a.example/x"}))
        for _ in range(61):
            status, body = post(self.app, "/shorten", {"url": f"https://example.com/{_}"})
        self.assertEqual(body["code"], "10")  # id 62
        self.assertEqual(self.app.handle("GET", "/lookup/1"), (200, {"code": "1", "url": "https://a.example/x", "hits": 1}))
        self.assertEqual(self.app.handle("GET", "/lookup/1?ref=x"), (200, {"code": "1", "url": "https://a.example/x", "hits": 2}))
        self.assertEqual(self.app.handle("GET", "/stats"), (200, {"count": 62}))

    def test_same_url_same_code(self):
        a = post(self.app, "/shorten", {"url": "https://a.example"})
        post(self.app, "/shorten", {"url": "https://b.example"})
        self.assertEqual(post(self.app, "/shorten", {"url": "https://a.example"}), a)

    def test_bad_requests(self):
        self.assertEqual(self.app.handle("POST", "/shorten", b"{nope"), (400, {"error": "invalid json"}))
        self.assertEqual(self.app.handle("POST", "/shorten", b"[1]"), (400, {"error": "invalid json"}))
        self.assertEqual(post(self.app, "/shorten", {"url": "ftp://x"}), (400, {"error": "invalid url"}))
        self.assertEqual(post(self.app, "/shorten", {}), (400, {"error": "invalid url"}))
        self.assertEqual(self.app.handle("GET", "/stats"), (200, {"count": 0}))

    def test_not_found(self):
        self.assertEqual(self.app.handle("GET", "/lookup/5"), (404, {"error": "not found"}))
        self.assertEqual(self.app.handle("GET", "/lookup/a-b"), (404, {"error": "not found"}))
        self.assertEqual(self.app.handle("GET", "/lookup/"), (404, {"error": "not found"}))
        self.assertEqual(self.app.handle("GET", "/nope"), (404, {"error": "not found"}))

    def test_wrong_method(self):
        self.assertEqual(self.app.handle("GET", "/shorten"), (405, {"error": "method not allowed"}))
        self.assertEqual(self.app.handle("POST", "/stats"), (405, {"error": "method not allowed"}))


if __name__ == "__main__":
    unittest.main()
