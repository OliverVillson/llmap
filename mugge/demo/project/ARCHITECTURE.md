# mdlite

A small Markdown to HTML converter in plain Python 3 (standard library only), with a command
line: `python3 -m mdlite notes.md --standalone --toc -o notes.html`.

## Layout

| File | What it does |
|---|---|
| `mdlite/types.py` | The shared types below. Read-only for every ticket. |
| `mdlite/escape.py` | `escape_html`, `escape_attr` |
| `mdlite/slug.py` | `slugify`, `Slugger` (unique heading ids) |
| `mdlite/inline_code.py` | `split_code`: cuts a line into text and code-span parts |
| `mdlite/inline_links.py` | `render_links`: links, images and autolinks |
| `mdlite/inline_emphasis.py` | `render_emphasis`: strong, em and strikethrough |
| `mdlite/inline.py` | `render_inline`: the inline pipeline built from the four above |
| `mdlite/block_heading.py` | `HeadingParser`: `# Title` lines, with ids, remembered for the TOC |
| `mdlite/block_rule.py` | `parse`: horizontal rules |
| `mdlite/block_para.py` | `parse`: paragraphs |
| `mdlite/block_code.py` | `parse`: fenced code blocks |
| `mdlite/block_quote.py` | `parse`: block quotes |
| `mdlite/block_list.py` | `parse`: bullet, numbered and task lists |
| `mdlite/block_table.py` | `parse`: pipe tables |
| `mdlite/frontmatter.py` | `split_front_matter`: the `---` header of a document |
| `mdlite/toc.py` | `render_toc`: nested table of contents |
| `mdlite/render.py` | `render`: the whole document, block by block |
| `mdlite/__main__.py` | the command line |
| `mdlite/theme.css` | the stylesheet a `--standalone` page embeds |

## The block contract

Every block parser has the same shape (`mdlite.types.BlockParser`):

```python
def parse(lines: list[str], i: int, inline: Inline) -> tuple[str, int] | None
```

`lines` is the document split on `\n` (no line endings), `i` the line to look at. A parser that
does not start a block at line `i` returns `None` and touches nothing. Otherwise it returns the
block's HTML, **always ending in exactly one `\n`**, and the index of the first line after the
block. `inline` renders inline Markdown (emphasis, links, code spans) and escapes HTML; block
parsers call it on text they show and never escape that text themselves. Code blocks are the
exception: their content is shown as it is, escaped, never passed to `inline`.

`HeadingParser(slugger).parse` has the same shape; it is a class because it keeps the ids it gave
out and the list of headings for the table of contents.

`render` skips blank lines (empty or only spaces) between blocks and tries the parsers in this
order at each line: code, heading, rule, table, quote, list, paragraph. The paragraph parser
accepts any non-blank line, so it always comes last.

## Inline pipeline

`render_inline(text)`:

1. `split_code(text)` cuts the text into `("text", s)` and `("code", s)` parts.
2. A code part becomes `<code>` + `escape_html(s)` + `</code>`.
3. A text part goes through `render_links(s, inner, escape_attr)` where
   `inner(t) = render_emphasis(escape_html(t))`: link labels and the text between links get
   escaped first, then emphasis.

Escaping never touches `*`, `_` or `~`, so emphasis can run on escaped text.
