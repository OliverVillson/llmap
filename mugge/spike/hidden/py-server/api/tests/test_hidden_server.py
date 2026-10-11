"""Hidden tests for the HTTP server (api/API.md): injected app, raw request forwarding, headers,
main() on a given port."""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from mugge_api.server import make_server, start

API_DIR = Path(__file__).resolve().parents[1]


class EchoApp:
    def __init__(self):
        self.calls = []

    def handle(self, method, path, body=b""):
        self.calls.append((method, path, body))
        return 202, {"method": method, "path": path, "body": body.decode("utf-8"), "n": len(self.calls)}


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        raw = r.read()
        return r.status, r.getheader("Content-Type"), r.getheader("Content-Length"), raw
    finally:
        conn.close()


class HiddenServerTest(unittest.TestCase):
    def test_make_server_does_not_serve_by_itself(self):
        server = make_server(EchoApp())
        try:
            self.assertIsInstance(server, ThreadingHTTPServer)
            host, port = server.server_address[:2]
            self.assertEqual(host, "127.0.0.1")
            self.assertGreater(port, 0)
        finally:
            server.server_close()

    def test_forwards_method_path_and_body(self):
        app = EchoApp()
        server, thread = start(app)
        port = server.server_address[1]
        try:
            payload = json.dumps({"url": "https://a.example/ü"}).encode()
            status, ctype, clen, raw = request(port, "POST", "/x/y?q=1&r=2", body=payload, headers={"Content-Type": "application/json"})
            self.assertEqual((status, ctype), (202, "application/json"))
            self.assertEqual(int(clen), len(raw))
            self.assertEqual(json.loads(raw), {"method": "POST", "path": "/x/y?q=1&r=2", "body": payload.decode(), "n": 1})
            status, _, _, raw = request(port, "GET", "/anything")
            self.assertEqual(status, 202)
            self.assertEqual(json.loads(raw), {"method": "GET", "path": "/anything", "body": "", "n": 2})
            status, _, _, raw = request(port, "POST", "/empty", body=b"", headers={"Content-Length": "0"})
            self.assertEqual(json.loads(raw)["body"], "")
        finally:
            server.shutdown()
            server.server_close()

    def test_real_app_over_http(self):
        server, thread = start()
        port = server.server_address[1]
        try:
            status, ctype, clen, raw = request(port, "POST", "/shorten", body=b"")
            self.assertEqual((status, ctype, json.loads(raw)), (400, "application/json", {"error": "invalid json"}))
            self.assertEqual(int(clen), len(raw))
            for i in range(1, 63):
                status, _, _, raw = request(port, "POST", "/shorten", body=json.dumps({"url": f"https://e.example/{i}"}).encode())
                self.assertEqual(status, 201)
            self.assertEqual(json.loads(raw), {"code": "10", "url": "https://e.example/62"})
            status, _, _, raw = request(port, "GET", "/lookup/10?from=cli")
            self.assertEqual((status, json.loads(raw)), (200, {"code": "10", "url": "https://e.example/62", "hits": 1}))
            status, ctype, _, raw = request(port, "GET", "/nope")
            self.assertEqual((status, ctype, json.loads(raw)), (404, "application/json", {"error": "not found"}))
        finally:
            server.shutdown()
            server.server_close()

    def test_concurrent_requests(self):
        server, thread = start(EchoApp())
        port = server.server_address[1]
        errors = []

        def worker(i):
            try:
                status, _, _, raw = request(port, "GET", f"/w/{i}")
                if status != 202 or json.loads(raw)["path"] != f"/w/{i}":
                    errors.append(i)
            except Exception as e:  # pragma: no cover
                errors.append(repr(e))

        try:
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            self.assertEqual(errors, [])
        finally:
            server.shutdown()
            server.server_close()

    def test_main_serves_on_the_given_port(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen(
            [sys.executable, "-c", f"from mugge_api.server import main; main([{str(port)!r}])"],
            cwd=API_DIR, env=dict(os.environ), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            deadline = time.time() + 10
            last = None
            while time.time() < deadline:
                if proc.poll() is not None:
                    self.fail(f"main exited early: {proc.stderr.read().decode()[-500:]}")
                try:
                    status, ctype, _, raw = request(port, "GET", "/health")
                    self.assertEqual((status, ctype, json.loads(raw)), (200, "application/json", {"ok": True}))
                    return
                except OSError as e:
                    last = e
                    time.sleep(0.1)
            self.fail(f"main never served on port {port}: {last!r}")
        finally:
            proc.kill()
            proc.wait(5)
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
