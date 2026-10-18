---
title: mdlite
lang: en
---
# mdlite

A small **Markdown to HTML** converter in plain Python 3, standard library only. It was
built by a Centipede crew: one orchestrator and ten code agents working at the same time,
each on its own ticket, with every ticket checked by its tests and reviewed by the lead.

## Use it

```sh
python3 -m mdlite notes.md --standalone --toc -o notes.html
cat notes.md | python3 -m mdlite > fragment.html
```

| Flag | What it does |
|:-----|:-------------|
| `--standalone` | a whole page, with the theme |
| `--toc` | a table of contents before the text |
| `--title T` | the page title (else front matter, then the first heading) |
| `-o FILE` | write there instead of stdout |

## What it understands

- Headings with ids, paragraphs, rules
- *Emphasis*, **strong**, ~~strikethrough~~ and `code spans`
- [Links](https://example.com "with a title"), images and <https://autolinks.example>
- Fenced code blocks, block quotes, lists and pipe tables
- [x] task lists
- [ ] nested lists (not yet)

> Each part is its own module with its own tests, so ten agents could build it at once.
> Two of them shared a stylesheet; the orchestrator ran them one after the other.

### Modules

1. `escape`, `slug`, `inline_code`, `inline_links`, `inline_emphasis`
2. `block_heading`, `block_rule`, `block_para`, `block_code`, `block_quote`, `block_list`, `block_table`
3. `frontmatter`, `toc`, then `inline`, `render` and the command line

* * *

Run the tests with `python3 -m unittest discover -s tests -t .`
