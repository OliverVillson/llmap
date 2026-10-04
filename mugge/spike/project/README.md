# short

A tiny polyglot URL shortener, used as the Mugge engine spike's test project.

- `libshort/` (C): base62 codes, FNV-1a hash, a string table and the shortener. `make -C libshort test`
- `api/` (Python, stdlib only): an HTTP API over an in-memory store, using `libshort` through ctypes.
  `make -C libshort lib && python3 -m unittest discover -t api -s api/tests`
- `cli/` (TypeScript, Bun): a command-line client for the API. `bun test ./cli/test && tsc -p cli --noEmit`

Interfaces: `libshort/include/short.h`, `api/API.md`, `cli/src/types.ts`.
Run the API with `python3 -m mugge_api.server 8080` from `api/` and the CLI with `bun cli/bin.ts shorten https://example.com`.
