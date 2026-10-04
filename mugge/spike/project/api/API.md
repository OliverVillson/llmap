# mugge_api interface

Python 3.11, standard library only. Run tests from the repo root with
`python3 -m unittest discover -t api -s api/tests -p 'test_<name>.py'`
(binding, handlers and server tests need `make -C libshort lib` first).

## validate.py
- `is_valid_url(url) -> bool`: True only for a `str` of 1..2048 characters with no whitespace
  or control characters (code point < 33 or == 127), whose scheme is `http` or `https`
  (case-insensitive) and which has a non-empty hostname (per `urllib.parse.urlsplit`).
  Anything else, including non-strings, is False. Never raises.

## store.py
- `class Store`: in-memory, pure Python.
  - `add(url: str) -> int`: stores url and returns its id. Ids start at 1 and count up; adding
    a url already stored returns its existing id.
  - `get(id: int) -> str | None`: the url for id, or None.
  - `hit(id: int) -> int`: adds one to id's hit count and returns the new count; raises
    `KeyError` for an unknown id.
  - `hits(id: int) -> int`: the hit count (0 for a new id); raises `KeyError` for an unknown id.
  - `__len__() -> int`: number of stored urls.

## binding.py
ctypes wrapper over `libshort/build/libshort.so` (see `libshort/include/short.h`). The path is
`LIB_PATH` (repo root / `libshort/build/libshort.so`), overridable with env `MUGGE_LIBSHORT`.
The library is loaded lazily on first use and cached, so importing never fails.
- `encode(n: int) -> str`: `short_encode`. Raises `ValueError` if n is not an int in 0..2**64-1.
- `decode(code: str) -> int`: `short_decode`. Raises `ValueError` if the code is not a `str` or
  short_decode returns -1.

## handlers.py
- `class App(store=None, codec=None)`: `store` defaults to a new `Store()`, `codec` to the
  `binding` module (anything with `encode(int) -> str` and `decode(str) -> int`).
  - `handle(method: str, path: str, body: bytes = b"") -> tuple[int, dict]`. A query string
    in `path` is ignored. Routes:
    - `GET /health` -> `200 {"ok": True}`
    - `POST /shorten` with JSON body `{"url": "..."}` -> `201 {"code": codec.encode(id), "url": url}`.
      Body that is not a JSON object -> `400 {"error": "invalid json"}`;
      missing or invalid url (`is_valid_url`) -> `400 {"error": "invalid url"}`.
    - `GET /lookup/<code>` -> records a hit, `200 {"code": code, "url": url, "hits": n}`.
      Undecodable or unknown code -> `404 {"error": "not found"}`.
    - `GET /stats` -> `200 {"count": len(store)}`
    - a known path with the wrong method -> `405 {"error": "method not allowed"}`
    - anything else -> `404 {"error": "not found"}`

## server.py
- `make_server(app=None, host="127.0.0.1", port=0) -> http.server.ThreadingHTTPServer`:
  serves `app` (default `App()`), mapping every GET/POST to `app.handle(method, path, body)`
  (body read using Content-Length) and replying with the status and the dict as JSON,
  `Content-Type: application/json`. Request logging is silenced.
- `start(app=None, host="127.0.0.1", port=0) -> tuple[ThreadingHTTPServer, threading.Thread]`:
  `make_server` then `serve_forever` in a daemon thread. Caller stops it with
  `server.shutdown(); server.server_close()`.
- `main(argv=None) -> None`: serve forever on port `argv[0]` (default 8080).
