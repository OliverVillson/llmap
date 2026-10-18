"""The command line: python3 -m mdlite [INPUT] [-o OUTPUT] [--standalone] [--toc] [--title TITLE]"""

import sys
from typing import List, Optional

from mdlite.escape import escape_attr, escape_html
from mdlite.frontmatter import split_front_matter
from mdlite.render import render
from mdlite.toc import render_toc


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
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
