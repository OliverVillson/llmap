import { describe, expect, test } from 'bun:test';
import { createClient } from '../src/client.ts';
import { ApiError, type FetchInit, type FetchLike } from '../src/types.ts';

function fake(status: number, body: unknown) {
  const calls: { url: string; init?: FetchInit }[] = [];
  const fetch: FetchLike = async (url, init) => {
    calls.push({ url, init });
    return {
      status,
      json: async () => {
        if (body instanceof Error) throw body;
        return body;
      },
    };
  };
  return { fetch, calls };
}

describe('createClient', () => {
  test('shorten posts json', async () => {
    const f = fake(201, { code: '1', url: 'https://a.example' });
    const c = createClient('http://h:1//', f.fetch);
    expect(await c.shorten('https://a.example')).toEqual({ code: '1', url: 'https://a.example' });
    expect(f.calls).toHaveLength(1);
    expect(f.calls[0].url).toBe('http://h:1/shorten');
    expect(f.calls[0].init?.method).toBe('POST');
    expect(f.calls[0].init?.headers).toEqual({ 'content-type': 'application/json' });
    expect(JSON.parse(f.calls[0].init?.body ?? 'null')).toEqual({ url: 'https://a.example' });
  });

  test('lookup encodes the code', async () => {
    const f = fake(200, { code: 'a/b', url: 'https://x', hits: 2 });
    expect(await createClient('http://h:1', f.fetch).lookup('a/b')).toEqual({ code: 'a/b', url: 'https://x', hits: 2 });
    expect(f.calls[0].url).toBe('http://h:1/lookup/a%2Fb');
    expect(f.calls[0].init?.method).toBe('GET');
  });

  test('stats', async () => {
    const f = fake(200, { count: 5 });
    expect(await createClient('http://h:1/', f.fetch).stats()).toEqual({ count: 5 });
    expect(f.calls[0].url).toBe('http://h:1/stats');
    expect(f.calls[0].init?.method).toBe('GET');
  });

  test('error status with json error', async () => {
    const f = fake(404, { error: 'not found' });
    const e = await createClient('http://h:1', f.fetch).lookup('zz').catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.status).toBe(404);
    expect(e.message).toBe('not found');
  });

  test('error status without json', async () => {
    const f = fake(502, new Error('not json'));
    const e = await createClient('http://h:1', f.fetch).stats().catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.status).toBe(502);
    expect(e.message).toBe('HTTP 502');
  });

  test('3xx is an error too', async () => {
    const f = fake(302, {});
    const e = await createClient('http://h:1', f.fetch).stats().catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.message).toBe('HTTP 302');
  });
});
