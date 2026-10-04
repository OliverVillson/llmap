import type { Config } from './types.ts';

/**
 * Merges flags over environment over defaults.
 *   server: overrides.server ?? env.MUGGE_SERVER ?? DEFAULT_SERVER, trailing slashes removed.
 *           Must match /^https?:\/\/[^\s\/]+/ (after trimming), else throws
 *           `new Error('invalid server url: <value>')` with the value as given.
 *   json:   overrides.json === true, or env.MUGGE_JSON is '1' or 'true'.
 *
 * STUB: ticket ts-config fills this in.
 */
export function resolveConfig(
  env: Record<string, string | undefined>,
  overrides: { server?: string; json?: boolean },
): Config {
  throw new Error('not implemented');
}
