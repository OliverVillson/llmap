"""Hidden tests for the libshort ctypes binding (api/API.md, libshort/include/short.h)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mugge_api import binding

API_DIR = Path(__file__).resolve().parents[1]


def ref_encode(n):
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    while True:
        out = alphabet[n % 62] + out
        n //= 62
        if n == 0:
            return out


class HiddenBindingTest(unittest.TestCase):
    def test_boundaries_round_trip(self):
        for k in range(11):
            for n in [62**k - 1, 62**k, 62**k + 1]:
                if 0 <= n < 2**64:
                    code = binding.encode(n)
                    self.assertIs(type(code), str)
                    self.assertEqual(code, ref_encode(n))
                    self.assertEqual(binding.decode(code), n)
        for n in [2**63, 2**64 - 2, 2**32, 2**32 - 1]:
            self.assertEqual(binding.encode(n), ref_encode(n))
            self.assertEqual(binding.decode(ref_encode(n)), n)

    def test_decode_follows_libshort(self):
        self.assertEqual(binding.decode("01"), 1)
        self.assertEqual(binding.decode("Z"), 61)
        self.assertIs(type(binding.decode("Z")), int)

    def test_encode_rejects_non_ints(self):
        for v in [True, False, 2.0, None, [1], -(2**64), 2**65]:
            with self.assertRaises(ValueError, msg=repr(v)):
                binding.encode(v)

    def test_decode_rejects(self):
        for v in [b"1", "é", "1 ", " ", "zzzzzzzzzzzz", "100000000000", ["1"]]:
            with self.assertRaises(ValueError, msg=repr(v)):
                binding.decode(v)

    def _run(self, env_lib, code):
        env = dict(os.environ, MUGGE_LIBSHORT=str(env_lib))
        return subprocess.run([sys.executable, "-c", code], cwd=API_DIR, env=env, capture_output=True, text=True, timeout=30)

    def test_import_never_fails(self):
        r = self._run("/nonexistent/libshort.so", "import mugge_api.binding as b; print(b.LIB_PATH)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "/nonexistent/libshort.so")

    def test_env_override_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "copy.so"
            shutil.copy(binding.LIB_PATH, copy)
            r = self._run(copy, "import mugge_api.binding as b; print(b.encode(3844), b.decode('ZZ'))")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["100", "3843"])


if __name__ == "__main__":
    unittest.main()
