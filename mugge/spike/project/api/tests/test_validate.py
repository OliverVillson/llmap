import unittest

from mugge_api.validate import is_valid_url


class ValidateTest(unittest.TestCase):
    def test_accepts_http_and_https(self):
        for url in ["http://example.com", "https://example.com/a/b?c=d#e", "HTTPS://Example.COM", "http://localhost:8080/x", "https://1.2.3.4/"]:
            self.assertTrue(is_valid_url(url), url)

    def test_rejects_other_schemes_and_missing_host(self):
        for url in ["ftp://example.com", "javascript:alert(1)", "example.com", "http://", "https:///path", "mailto:a@b.c", ""]:
            self.assertFalse(is_valid_url(url), url)

    def test_rejects_whitespace_and_control_chars(self):
        for url in ["http://exa mple.com", "http://example.com/\n", "http://example.com/\tx", "http://example.com/\x7f"]:
            self.assertFalse(is_valid_url(url), repr(url))

    def test_length_limit(self):
        base = "https://example.com/"
        self.assertTrue(is_valid_url(base + "a" * (2048 - len(base))))
        self.assertFalse(is_valid_url(base + "a" * (2049 - len(base))))

    def test_non_strings(self):
        for v in [None, 42, b"http://example.com", ["http://example.com"]]:
            self.assertFalse(is_valid_url(v), repr(v))


if __name__ == "__main__":
    unittest.main()
