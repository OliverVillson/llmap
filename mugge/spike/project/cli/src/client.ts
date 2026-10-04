import type { Client, FetchLike } from './types.ts';

/**
 * API client over an injected fetch. `base` may end in slashes; they are dropped.
 *   shorten(url): POST `${base}/shorten`, headers { 'content-type': 'application/json' },
 *                 body JSON.stringify({ url })
 *   lookup(code): GET `${base}/lookup/${encodeURIComponent(code)}`, init { method: 'GET' }
 *   stats():      GET `${base}/stats`, init { method: 'GET' }
 * A 2xx response resolves to its JSON body. Any other status throws
 * `new ApiError(status, body.error)` when the body is JSON with a string `error`,
 * else `new ApiError(status, 'HTTP <status>')`.
 *
 * STUB: ticket ts-client fills this in.
 */
export function createClient(base: string, fetch: FetchLike): Client {
  throw new Error('not implemented');
}
