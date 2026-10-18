import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mdlite.__main__ import main

ROOT = Path(__file__).resolve().parent.parent
CSS = (ROOT / "mdlite" / "theme.css").read_text()


def run(argv, stdin=None):
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdin
    sys.stdin = io.StringIO(stdin or "")
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as e:
                code = e.code
    finally:
        sys.stdin = old
    return code, out.getvalue(), err.getvalue()


class Cli(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, text):
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_fragment_to_stdout(self):
        self.assertEqual(run([self.write("a.md", "# Hi\n\n*x*\n")]), (0, '<h1 id="hi">Hi</h1>\n<p><em>x</em></p>\n', ""))

    def test_stdin(self):
        self.assertEqual(run([], stdin="text"), (0, "<p>text</p>\n", ""))
        self.assertEqual(run(["-"], stdin="text")[1], "<p>text</p>\n")

    def test_standalone_with_front_matter(self):
        src = self.write("notes.md", '---\ntitle: "A & B"\nlang: sv\n---\n# Heading\n')
        code, out, _ = run([src, "--standalone"])
        self.assertEqual(code, 0)
        self.assertEqual(out, (
            '<!doctype html>\n<html lang="sv">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>A &amp; B</title>\n<style>\n" + CSS + "</style>\n</head>\n<body>\n"
            '<h1 id="heading">Heading</h1>\n</body>\n</html>\n'))

    def test_title_order(self):
        src = self.write("my-notes.md", "---\ntitle: Meta\n---\n# Head\n")
        self.assertIn("<title>Flag</title>", run([src, "--standalone", "--title", "Flag"])[1])
        self.assertIn("<title>Meta</title>", run([src, "--standalone"])[1])
        self.assertIn("<title>Head</title>", run([self.write("h.md", "# Head\n"), "--standalone"])[1])
        self.assertIn("<title>my-notes</title>", run([self.write("my-notes.md", "text\n"), "--standalone"])[1])
        self.assertIn("<title>Document</title>", run(["--standalone"], stdin="text")[1])
        self.assertIn('<html lang="en">', run(["--standalone"], stdin="text")[1])

    def test_toc(self):
        out = run([self.write("t.md", "# A\n## B\n"), "--toc"])[1]
        self.assertTrue(out.startswith('<nav class="toc">\n<ul>\n<li><a href="#a">A</a>\n<ul>\n'), out)
        self.assertTrue(out.endswith('</nav>\n<h1 id="a">A</h1>\n<h2 id="b">B</h2>\n'), out)
        self.assertEqual(run(["--toc"], stdin="no headings")[1], "<p>no headings</p>\n")

    def test_output_file(self):
        dest = self.tmp / "out.html"
        self.assertEqual(run([self.write("c.md", "é"), "-o", str(dest)]), (0, "", ""))
        self.assertEqual(dest.read_text(encoding="utf-8"), "<p>é</p>\n")

    def test_errors(self):
        missing = str(self.tmp / "nope.md")
        self.assertEqual(run([missing]), (2, "", f"mdlite: cannot read {missing}\n"))
        bad = self.write("bad.md", "---\nok: 1\nnot a key\n---\n")
        self.assertEqual(run([bad]), (2, "", f"mdlite: {bad}: front matter line 3: no key\n"))
        self.assertEqual(run(["--nope"])[0], 2)

    def test_python_dash_m(self):
        p = subprocess.run([sys.executable, "-m", "mdlite", "--standalone", "--toc"], input="# Run\n", capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('<nav class="toc">', p.stdout)
        self.assertIn("<title>Run</title>", p.stdout)


if __name__ == "__main__":
    unittest.main()
