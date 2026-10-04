"""ctypes wrapper over libshort. See api/API.md. STUB: ticket py-binding fills this in."""
import os
from pathlib import Path

LIB_PATH = Path(os.environ.get("MUGGE_LIBSHORT", Path(__file__).resolve().parents[2] / "libshort" / "build" / "libshort.so"))


def encode(n: int) -> str:
    raise NotImplementedError


def decode(code: str) -> int:
    raise NotImplementedError
