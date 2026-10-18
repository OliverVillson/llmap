"""The command line: python3 -m mdlite [INPUT] [-o OUTPUT] [--standalone] [--toc] [--title TITLE]"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mdlite.escape import escape_attr, escape_html
from mdlite.frontmatter import split_front_matter
from mdlite.render import render
from mdlite.toc import render_toc


def _page(content: str, title: str, lang: str) -> str:
    css = (Path(__file__).resolve().parent / "theme.css").read_text(encoding="utf-8")
    return (
        "<!doctype html>\n"
        f'<html lang="{escape_attr(lang)}">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape_html(title)}</title>\n"
        "<style>\n"
        f"{css}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{content}</body>\n"
        "</html>\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Convert one Markdown file to HTML. argv defaults to sys.argv[1:]; returns the exit code.

    INPUT is a path, or "-" or nothing for stdin; it is read as UTF-8. Its front matter
    (frontmatter.split_front_matter) is split off and the body rendered (render.render).
    --toc puts render_toc(headings) before the content (nothing when it is empty).
    --standalone wraps the content in a whole page, exactly:

        <!doctype html>
        <html lang="{lang}">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <style>
        {the text of mdlite/theme.css, read next to this file, as it is}</style>
        </head>
        <body>
        {content}</body>
        </html>

    (ending in "\\n"), where lang is front matter "lang" or "en", and title, HTML-escaped, is
    --title, else front matter "title", else the first heading's text, else the input file's
    name without its extension, else "Document". Without --standalone the output is the
    content alone. It goes to -o OUTPUT (UTF-8) or to stdout.

    Errors print one line "mdlite: <what>" to stderr and return 2: an input that cannot be
    read ("mdlite: cannot read notes.md"), or front matter that raises ValueError
    ("mdlite: notes.md: front matter line 3: no key"). Bad arguments behave as argparse does
    (exit code 2).
    """
    ap = argparse.ArgumentParser(prog="mdlite", description="Markdown to HTML")
    ap.add_argument("input", nargs="?", default="-")
    ap.add_argument("-o", "--output")
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--toc", action="store_true")
    ap.add_argument("--title")
    args = ap.parse_args(argv)

    name = "stdin" if args.input == "-" else args.input
    try:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
    except OSError:
        print(f"mdlite: cannot read {name}", file=sys.stderr)
        return 2
    try:
        meta, body = split_front_matter(text)
    except ValueError as e:
        print(f"mdlite: {name}: {e}", file=sys.stderr)
        return 2

    doc = render(body)
    content = (render_toc(doc.headings) if args.toc else "") + doc.html
    if args.standalone:
        title = args.title or meta.get("title") or (doc.headings[0][1] if doc.headings else "")
        if not title:
            title = Path(args.input).stem if args.input != "-" else "Document"
        content = _page(content, str(title), str(meta.get("lang") or "en"))
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
