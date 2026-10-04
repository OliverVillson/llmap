import type { ApiError, LookupResult, ShortenResult, Stats } from './types.ts';

// Human-readable output lines. STUB: ticket ts-format fills this in.

/** `<code> -> <url>` */
export function formatShorten(r: ShortenResult): string {
  throw new Error('not implemented');
}

/** `<code> -> <url> (<hits> hits)`, with `1 hit` for exactly one. */
export function formatLookup(r: LookupResult): string {
  throw new Error('not implemented');
}

/** `<count> links`, with `1 link` for exactly one. */
export function formatStats(s: Stats): string {
  throw new Error('not implemented');
}

/** `error: <message> (HTTP <status>)` */
export function formatError(e: ApiError): string {
  throw new Error('not implemented');
}

/**
 * Multi-line help text. The first line is exactly `usage: short [--server <url>] [--json] <command>`,
 * followed by one line per command (shorten <url>, lookup <code>, stats, help), each mentioning
 * the command name.
 */
export function usage(): string {
  throw new Error('not implemented');
}
