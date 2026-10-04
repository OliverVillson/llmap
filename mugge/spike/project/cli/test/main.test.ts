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
      return { status, json: async () => body };
    },
  };
  return { d, out, err, calls };
}

describe('main', () => {
  test('shorten prints a line', async () => {
    const t = deps({ 'http://127.0.0.1:8080/shorten': [201, { code: '1', url: 'https://a.example' }] });
    expect(await main(['shorten', 'https://a.example'], t.d)).toBe(0);
    expect(t.out).toEqual(['1 -> https://a.example']);
    expect(t.err).toEqual([]);
    expect(t.calls[0].init?.method).toBe('POST');
  });

  test('lookup with --server and --json', async () => {
    const t = deps({ 'http://h:1/lookup/5': [200, { code: '5', url: 'https://b', hits: 1 }] });
    expect(await main(['lookup', '5', '--server', 'http://h:1/', '--json'], t.d)).toBe(0);
    expect(JSON.parse(t.out[0])).toEqual({ code: '5', url: 'https://b', hits: 1 });
  });

  test('stats uses env', async () => {
    const t = deps({ 'http://env:2/stats': [200, { count: 1 }] }, { MUGGE_SERVER: 'http://env:2' });
    expect(await main(['stats'], t.d)).toBe(0);
    expect(t.out).toEqual(['1 link']);
  });

  test('help', async () => {
    const t = deps({});
    expect(await main([], t.d)).toBe(0);
    expect(t.out[0].startsWith('usage: short')).toBe(true);
    expect(t.calls).toHaveLength(0);
  });

  test('bad args exit 2 with usage', async () => {
    const t = deps({});
    expect(await main(['nope'], t.d)).toBe(2);
    expect(t.err[0]).toBe('error: unknown command: nope');
    expect(t.err[1].startsWith('usage: short')).toBe(true);
    expect(t.out).toEqual([]);
  });

  test('bad server exits 2', async () => {
    const t = deps({});
    expect(await main(['stats', '--server', 'nope'], t.d)).toBe(2);
    expect(t.err).toEqual(['error: invalid server url: nope']);
  });

  test('api error exits 1', async () => {
    const t = deps({});
    expect(await main(['lookup', 'zz'], t.d)).toBe(1);
    expect(t.err).toEqual(['error: not found (HTTP 404)']);
  });

  test('network error exits 1', async () => {
    const t = deps({});
    t.d.fetch = async () => {
      throw new Error('connection refused');
    };
    expect(await main(['stats'], t.d)).toBe(1);
    expect(t.err).toEqual(['error: connection refused']);
  });
});
