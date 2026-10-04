"""HTTP wiring. See api/API.md. STUB: ticket py-server fills this in."""
import sys
import threading
from http.server import ThreadingHTTPServer

from .handlers import App


def make_server(app=None, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    raise NotImplementedError


def start(app=None, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    raise NotImplementedError


def main(argv=None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1:])
