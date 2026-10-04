import type { ArgsResult } from './types.ts';

/**
 * Parses argv (program name already removed) into a command.
 *
 * Commands: `shorten <url>`, `lookup <code>`, `stats`, `help`. No arguments, `-h` or `--help`
 * anywhere means help. Options may appear anywhere: `--json`, `--server <url>`, `--server=<url>`.
 * Errors (returned as { ok: false, error }), exactly:
 *   `unknown option: <arg>` for any other argument starting with `-`
 *   `--server requires a value` when --server is last or its value is empty
 *   `unknown command: <word>`
 *   `shorten requires a url` / `lookup requires a code`
 *   `unexpected argument: <word>` for anything left over after the command's argument
 *
 * STUB: ticket ts-args fills this in.
 */
export function parseArgs(argv: string[]): ArgsResult {
  throw new Error('not implemented');
}
