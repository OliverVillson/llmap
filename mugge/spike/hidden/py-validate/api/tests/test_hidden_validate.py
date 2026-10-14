"""Hidden tests for is_valid_url (api/API.md)."""
import unittest

from mugge_api.validate import is_valid_url


class HiddenValidateTest(unittest.TestCase):
    def test_accepts(self):
        for url in [
            "http://a",
            "HtTp://example.com",
            "https://example.com:443/path",
            "https://user:pw@example.com/",
            "http://[::1]/",
            "http://bücher.example/straße",
            "https://example.com/" + "x" * (2048 - len("https://example.com/")),
        ]:
            self.assertIs(is_valid_url(url), True, url[:80])

    def test_rejects_without_raising(self):
        for url in [
            "http://[::1",
            "http:example.com",
            "//example.com",
            "http://:80/",
            "http://user@/",
            "https:",
            "httpx://example.com",
            "ws://example.com",
            "file:///etc/passwd",
            " http://example.com",
            "http://example.com ",
            "http://example.com/\x00",
            "http://example.com/\x1f",
            "http://example.com/\r",
            "\x7fhttp://example.com",
            "https://example.com/" + "x" * (5000),
        ]:
            try:
                got = is_valid_url(url)
            except Exception as e:  # never raises
                self.fail(f"is_valid_url({url[:40]!r}) raised {e!r}")
            self.assertIs(got, False, repr(url[:80]))

    def test_non_strings_without_raising(self):
        for v in [b"https://example.com", bytearray(b"http://a"), 3.5, True, object(), {"url": "http://a"}, ("http://a",)]:
            try:
                got = is_valid_url(v)
            except Exception as e:
                self.fail(f"is_valid_url({v!r}) raised {e!r}")
            self.assertIs(got, False, repr(v))


if __name__ == "__main__":
    unittest.main()
