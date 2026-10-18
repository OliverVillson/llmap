"""Types shared by every mdlite module (see ARCHITECTURE.md)."""

from typing import Callable, List, Optional, Protocol, Tuple

#: Renders inline Markdown (emphasis, links, code spans) and escapes HTML.
Inline = Callable[[str], str]

#: What a block parser returns: the block's HTML (ending in exactly one "\n") and the index of
#: the first line after the block; None when no block of its kind starts at that line.
BlockResult = Optional[Tuple[str, int]]


class BlockParser(Protocol):
    def __call__(self, lines: List[str], i: int, inline: Inline) -> BlockResult: ...


#: One heading as the table of contents needs it: (level 1-6, its text as written, its id).
Heading = Tuple[int, str, str]
