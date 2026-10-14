"""Hidden tests for App.handle (api/API.md): injected store and codec, routing edges, bodies."""
import json
import unittest

from mugge_api.handlers import App
from mugge_api.store import Store


class FakeCodec:
    """encode(n) -> 'c<n>'; decode accepts only that shape."""

    def encode(self, n):
        return f"c{n}"

    def decode(self, code):
        if not isinstance(code, str) or not code.startswith("c") or not code[1:].isdigit():
            raise ValueError(code)
        return int(code[1:])


def post(app, obj, path="/shorten"):
    return app.handle("POST", path, json.dumps(obj).encode())


class HiddenHandlersTest(unittest.TestCase):
    def test_uses_injected_codec_and_store(self):
        store = Store()
        store.add("https://pre.example")
        app = App(store=store, codec=FakeCodec())
        self.assertIs(app.store, store)
        self.assertEqual(app.handle("GET", "/stats"), (200, {"count": 1}))
        self.assertEqual(post(app, {"url": "https://a.example"}), (201, {"code": "c2", "url": "https://a.example"}))
        self.assertEqual(post(app, {"url": "https://pre.example"}), (201, {"code": "c1", "url": "https://pre.example"}))
        self.assertEqual(app.handle("GET", "/lookup/c1"), (200, {"code": "c1", "url": "https://pre.example", "hits": 1}))
        self.assertEqual(app.handle("GET", "/lookup/c1"), (200, {"code": "c1", "url": "https://pre.example", "hits": 2}))
        self.assertEqual(app.handle("GET", "/lookup/c2?x=1&y=2"), (200, {"code": "c2", "url": "https://a.example", "hits": 1}))
        self.assertEqual(store.hits(1), 2)
        self.assertEqual(app.handle("GET", "/lookup/c9"), (404, {"error": "not found"}))
        self.assertEqual(app.handle("GET", "/lookup/1"), (404, {"error": "not found"}))
        self.assertEqual(app.handle("GET", "/stats"), (200, {"count": 2}))

    def test_unknown_lookup_records_no_hit(self):
        store = Store()
        app = App(store=store)
        post(app, {"url": "https://a.example"})
        self.assertEqual(app.handle("GET", "/lookup/2"), (404, {"error": "not found"}))
        self.assertEqual(store.hits(1), 0)
        self.assertEqual(app.handle("GET", "/lookup/1")[1]["hits"], 1)

    def test_bodies(self):
        app = App()
        for body in [b"", b"null", b"42", b'"https://a.example"', b"\xff\xfe", b"{", b'{"url": "https://a.example"'] :
            self.assertEqual(app.handle("POST", "/shorten", body), (400, {"error": "invalid json"}), body)
        for obj in [{"url": None}, {"url": 5}, {"url": ["https://a.example"]}, {"link": "https://a.example"}, {"url": "https://a .example"}, {"url": ""}]:
            self.assertEqual(post(app, obj), (400, {"error": "invalid url"}), obj)
        self.assertEqual(app.handle("GET", "/stats"), (200, {"count": 0}))
        self.assertEqual(post(app, {"url": "https://a.example", "extra": 1}), (201, {"code": "1", "url": "https://a.example"}))
        self.assertEqual(post(app, {"url": "https://a.example"}, path="/shorten?utm=1"), (201, {"code": "1", "url": "https://a.example"}))
        self.assertEqual(app.handle("GET", "/stats?verbose=1"), (200, {"count": 1}))
        self.assertEqual(app.handle("GET", "/health?x"), (200, {"ok": True}))

    def test_routes(self):
        app = App()
        for method, path in [("PUT", "/health"), ("DELETE", "/stats"), ("PUT", "/shorten"), ("POST", "/health?x=1")]:
            self.assertEqual(app.handle(method, path), (405, {"error": "method not allowed"}), (method, path))
        for path in ["/", "", "/health/", "/healthz", "/stat", "/lookup", "/shorten/x", "/HEALTH"]:
            self.assertEqual(app.handle("GET", path), (404, {"error": "not found"}), path)

    def test_codes_follow_libshort(self):
        app = App()
        codes = [post(app, {"url": f"https://example.com/{i}"})[1]["code"] for i in range(1, 3846)]
        self.assertEqual(codes[61], "10")
        self.assertEqual(codes[3842], "ZZ")
        self.assertEqual(codes[3843], "100")
        self.assertEqual(app.handle("GET", "/lookup/100"), (200, {"code": "100", "url": "https://example.com/3844", "hits": 1}))
        self.assertEqual(app.handle("GET", "/lookup/zzzzzzzzzzzzzz"), (404, {"error": "not found"}))


if __name__ == "__main__":
    unittest.main()
