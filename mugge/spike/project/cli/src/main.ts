import type { Deps } from './types.ts';

/**
 * Runs the CLI and resolves to the exit code. Steps:
 *  1. parseArgs(argv). On error: err(`error: <error>`), err(usage()), return 2.
 *  2. help: out(usage()), return 0.
 *  3. resolveConfig(deps.env, { server, json }). If it throws: err(`error: <message>`), return 2.
 *  4. createClient(config.server, deps.fetch) and run the command. Print one line with out():
 *     JSON.stringify(result) when config.json, else formatShorten / formatLookup / formatStats.
 *     Return 0.
 *  5. ApiError: err(formatError(e)), return 1. Any other error: err(`error: <message>`), return 1.
 *
 * STUB: ticket ts-main fills this in.
 */
export async function main(argv: string[], deps: Deps): Promise<number> {
  throw new Error('not implemented');
}
