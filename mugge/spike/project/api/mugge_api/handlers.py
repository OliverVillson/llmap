"""Request handling. See api/API.md. STUB: ticket py-handlers fills this in."""
from . import binding
from .store import Store
from .validate import is_valid_url


class App:
    def __init__(self, store=None, codec=None):
        self.store = store if store is not None else Store()
        self.codec = codec if codec is not None else binding

    def handle(self, method: str, path: str, body: bytes = b"") -> tuple[int, dict]:
        raise NotImplementedError
