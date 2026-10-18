"""Heading ids."""


def slugify(text: str) -> str:
    """Turn heading text into an id.

    Lowercase it; drop every character that is not a letter, a digit (str.isalnum, so
    non-ASCII letters stay), a space, "-" or "_"; turn each run of whitespace into one "-";
    collapse runs of "-" into one; strip "-" from both ends. An empty result becomes "section".

        slugify("Hello, World!")     -> "hello-world"
        slugify("  Café -- au  lait ") -> "café-au-lait"
        slugify("snake_case name")   -> "snake_case-name"
        slugify("!!!")               -> "section"
    """
    raise NotImplementedError


class Slugger:
    """Gives out unique ids within one document."""

    def __init__(self) -> None:
        raise NotImplementedError

    def slug(self, text: str) -> str:
        """slugify(text), made unique among the ids this Slugger gave out before.

        The first "intro" is "intro", the second "intro-1", the third "intro-2". If a candidate
        is already taken (a heading literally called "Intro 1" took "intro-1"), keep counting.
        """
        raise NotImplementedError
