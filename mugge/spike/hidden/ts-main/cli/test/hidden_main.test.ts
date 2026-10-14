// Hidden tests for main (cli/src/main.ts doc comment).
import { describe, expect, test } from 'bun:test';
import { main } from '../src/main.ts';
import type { Deps, FetchInit } from '../src/types.ts';

function deps(routes: Record<string, [number, unknown]>, env: Record<string, string> = {}) {
  const out: string[] = [];
  const err: string[] = [];
  const calls: { url: string; init?: FetchInit }[] = [];
  const d: Deps = {
    env,
    out: (l) => out.push(l),
    err: (l) => err.push(l),
    fetch: async (url, init) => {
      calls.push({ url, init });
      const [status, body] = routes[url] ?? [404, { error: 'not found' }];
      return {
        status,
        json: async () => {
          if (body instanceof Error) throw body;
          return body;
        },
      };
    },
  };
  return { d, out, err, calls };
}

describe('main (hidden)', () => {
  test('json output is JSON.stringify of the response', async () => {
    const body = { code: '1', url: 'https://a.example/?x="y"' };
    const t = deps({ 'http://127.0.0.1:8080/shorten': [201, body] });
    expect(await main(['--json', 'shorten', 'https://a.example/?x="y"'], t.d)).toBe(0);
    expect(t.out).toEqual([JSON.stringify(body)]);
    expect(t.err).toEqual([]);
    expect(JSON.parse(t.calls[0].init?.body ?? 'null')).toEqual({ url: 'https://a.example/?x="y"' });
  });

  test('MUGGE_JSON from env', async () => {
    const t = deps({ 'http://e:1/stats': [200, { count: 7 }] }, { MUGGE_SERVER: 'http://e:1/', MUGGE_JSON: 'true' });
    expect(await main(['stats'], t.d)).toBe(0);
    expect(t.out).toEqual(['{"count":7}']);
  });

  test('human lookup and stats lines', async () => {
    const t = deps({ 'http://h/lookup/a%2Fb': [200, { code: 'a/b', url: 'https://b', hits: 3 }], 'http://h/stats': [200, { count: 0 }] });
    expect(await main(['lookup', 'a/b', '--server=http://h'], t.d)).toBe(0);
    expect(await main(['--server', 'http://h', 'stats'], t.d)).toBe(0);
    expect(t.out).toEqual(['a/b -> https://b (3 hits)', '0 links']);
    expect(t.calls.map((c) => c.url)).toEqual(['http://h/lookup/a%2Fb', 'http://h/stats']);
  });

  test('--help goes to out, makes no request, ignores a bad server', async () => {
    const t = deps({});
    expect(await main(['stats', '--help'], t.d)).toBe(0);
    expect(await main(['help', '--server', 'nope'], t.d)).toBe(0);
    expect(t.out).toHaveLength(2);
    for (const line of t.out) expect(line.split('\n')[0]).toBe('usage: short [--server <url>] [--json] <command>');
    expect(t.err).toEqual([]);
    expect(t.calls).toHaveLength(0);
  });

  test('usage errors exit 2 before any request', async () => {
    for (const [argv, msg] of [
      [['--verbose', 'stats'], 'unknown option: --verbose'],
      [['shorten'], 'shorten requires a url'],
      [['stats', '--server='], '--server requires a value'],
      [['lookup', 'a', 'b'], 'unexpected argument: b'],
    ] as [string[], string][]) {
      const t = deps({});
      expect(await main(argv, t.d)).toBe(2);
      expect(t.err).toHaveLength(2);
      expect(t.err[0]).toBe(`error: ${msg}`);
      expect(t.err[1].split('\n')[0]).toBe('usage: short [--server <url>] [--json] <command>');
      expect(t.out).toEqual([]);
      expect(t.calls).toHaveLength(0);
    }
  });

  test('config error from env exits 2 without usage', async () => {
    const t = deps({}, { MUGGE_SERVER: 'ftp://x' });
    expect(await main(['lookup', 'q'], t.d)).toBe(2);
    expect(t.err).toEqual(['error: invalid server url: ftp://x']);
    expect(t.out).toEqual([]);
    expect(t.calls).toHaveLength(0);
  });

  test('api errors exit 1', async () => {
    const t = deps({ 'http://127.0.0.1:8080/shorten': [400, { error: 'invalid url' }], 'http://127.0.0.1:8080/stats': [503, new Error('not json')] });
    expect(await main(['shorten', 'nope'], t.d)).toBe(1);
    expect(await main(['stats', '--json'], t.d)).toBe(1);
    expect(t.err).toEqual(['error: invalid url (HTTP 400)', 'error: HTTP 503 (HTTP 503)']);
    expect(t.out).toEqual([]);
  });

  test('other errors exit 1 with their message', async () => {
    const t = deps({});
    t.d.fetch = async () => {
      throw new TypeError('fetch failed');
    };
    expect(await main(['lookup', 'x', '--json'], t.d)).toBe(1);
    expect(t.err).toEqual(['error: fetch failed']);
    expect(t.out).toEqual([]);
  });
});
