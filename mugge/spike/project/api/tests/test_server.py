import json
import unittest
import urllib.error
import urllib.request

from mugge_api.server import start


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.server, self.thread = start(port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, obj=None):
        data = None if obj is None else json.dumps(obj).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.headers.get("Content-Type"), json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), json.loads(e.read())

    def test_round_trip(self):
        self.assertEqual(self.request("GET", "/health"), (200, "application/json", {"ok": True}))
        self.assertEqual(self.request("POST", "/shorten", {"url": "https://a.example"}), (201, "application/json", {"code": "1", "url": "https://a.example"}))
        self.assertEqual(self.request("GET", "/lookup/1")[2], {"code": "1", "url": "https://a.example", "hits": 1})
        self.assertEqual(self.request("GET", "/stats")[2], {"count": 1})

    def test_errors_are_json(self):
        self.assertEqual(self.request("POST", "/shorten", {"url": "nope"}), (400, "application/json", {"error": "invalid url"}))
        self.assertEqual(self.request("GET", "/lookup/zzz"), (404, "application/json", {"error": "not found"}))
        self.assertEqual(self.request("POST", "/health", {}), (405, "application/json", {"error": "method not allowed"}))

    def test_daemon_thread(self):
        self.assertTrue(self.thread.daemon)
        self.assertTrue(self.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
