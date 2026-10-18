"""ATX headings."""

from typing import List

from mdlite.slug import Slugger
from mdlite.types import BlockResult, Heading, Inline


class HeadingParser:
    """Parses `# Title` lines and remembers every heading for the table of contents."""

    def __init__(self, slugger: Slugger) -> None:
        self.slugger = slugger
        #: Every heading parsed so far, in order: (level, text as written, id).
        self.headings: List[Heading] = []

    def parse(self, lines: List[str], i: int, inline: Inline) -> BlockResult:
        """A heading is one line: 1 to 6 "#" at the very start, then at least one space or tab
        and the text, or the "#"s alone ("##" is an empty level-2 heading). Spaces and tabs
        around the text are stripped, and so is a closing run of "#" that has a space or tab
        before it ("## Title ##" is "Title"; "# C#" keeps "C#"). Seven "#", or "#" followed
        directly by text ("#tag"), is not a heading.

        Returns ('<h2 id="{id}">{inline(text)}</h2>\\n', i + 1), where id is
        self.slugger.slug(text) of the text as written, and appends (level, text, id) to
        self.headings.
        """
        raise NotImplementedError
