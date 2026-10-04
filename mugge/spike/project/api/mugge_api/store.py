"""In-memory url store. See api/API.md. STUB: ticket py-store fills this in."""


class Store:
    def add(self, url: str) -> int:
        raise NotImplementedError

    def get(self, id: int) -> str | None:
        raise NotImplementedError

    def hit(self, id: int) -> int:
        raise NotImplementedError

    def hits(self, id: int) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
