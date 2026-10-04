// Shared interface of the CLI. Every module in src/ builds on these types; tests in test/.

/** Server used when neither --server nor MUGGE_SERVER is given. */
export const DEFAULT_SERVER = 'http://127.0.0.1:8080';

/** One parsed command. */
export type Command =
  | { kind: 'shorten'; url: string }
  | { kind: 'lookup'; code: string }
  | { kind: 'stats' }
  | { kind: 'help' };

/** Result of parsing argv (program name already removed). */
export interface ParsedArgs {
  command: Command;
  /** From `--server <url>` or `--server=<url>`; undefined when not given. */
  server?: string;
  /** True when `--json` was given. */
  json: boolean;
}

export type ArgsResult = { ok: true; args: ParsedArgs } | { ok: false; error: string };

/** Effective settings after merging flags, environment and defaults. */
export interface Config {
  /** Base URL with no trailing slash, e.g. `http://127.0.0.1:8080`. */
  server: string;
  /** Print raw JSON instead of human-readable lines. */
  json: boolean;
}

/** API responses (see api/API.md). */
export interface ShortenResult {
  code: string;
  url: string;
}
export interface LookupResult {
  code: string;
  url: string;
  hits: number;
}
export interface Stats {
  count: number;
}

export interface FetchInit {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
}
export interface FetchResponse {
  status: number;
  json(): Promise<unknown>;
}
/** The subset of `fetch` the client uses, injectable for tests. */
export type FetchLike = (url: string, init?: FetchInit) => Promise<FetchResponse>;

/** Typed API client (see client.ts). */
export interface Client {
  shorten(url: string): Promise<ShortenResult>;
  lookup(code: string): Promise<LookupResult>;
  stats(): Promise<Stats>;
}

/** Thrown by the client for any response status outside 200..299. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Everything main() touches in the outside world. */
export interface Deps {
  fetch: FetchLike;
  env: Record<string, string | undefined>;
  /** Writes one line to stdout. */
  out(line: string): void;
  /** Writes one line to stderr. */
  err(line: string): void;
}
