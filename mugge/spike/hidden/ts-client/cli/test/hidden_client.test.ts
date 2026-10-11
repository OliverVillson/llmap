// Hidden tests for createClient (cli/src/client.ts doc comment, api/API.md).
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

const fail = (p: Promise<unknown>) => p.then(() => { throw new Error('expected a rejection'); }, (e) => e);

describe('createClient (hidden)', () => {
  test('exact requests', async () => {
    const f = fake(201, { code: '1', url: 'https://a.example/"q"' });
    const c = createClient('https://h.example/api///', f.fetch);
    await c.shorten('https://a.example/"q"');
    expect(f.calls).toEqual([
      {
        url: 'https://h.example/api/shorten',
        init: { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ url: 'https://a.example/"q"' }) },
      },
    ]);
    const g = fake(200, { count: 0 });
    await createClient('http://h:1', g.fetch).stats();
    expect(g.calls).toEqual([{ url: 'http://h:1/stats', init: { method: 'GET' } }]);
  });

  test('lookup escapes the code', async () => {
    const f = fake(200, { code: 'x', url: 'https://x', hits: 1 });
    const c = createClient('http://h:1/', f.fetch);
    await c.lookup('a b?#&é');
    await c.lookup('../stats');
    expect(f.calls.map((x) => x.url)).toEqual([`http://h:1/lookup/${encodeURIComponent('a b?#&é')}`, 'http://h:1/lookup/..%2Fstats']);
    expect(f.calls[0].init).toEqual({ method: 'GET' });
  });

  test('2xx range resolves to the body as is', async () => {
    for (const status of [200, 201, 204, 299]) {
      const body = { code: 'z', url: 'https://z', hits: status };
      expect(await createClient('http://h', fake(status, body).fetch).lookup('z')).toEqual(body);
    }
  });

  test('statuses outside 2xx reject with ApiError', async () => {
    for (const status of [100, 199, 300, 304, 400, 418, 500, 503]) {
      const e = await fail(createClient('http://h', fake(status, { error: `boom ${status}` }).fetch).stats());
      expect(e).toBeInstanceOf(ApiError);
      expect(e).toBeInstanceOf(Error);
      expect(e.status).toBe(status);
      expect(e.message).toBe(`boom ${status}`);
    }
  });

  test('non-string or missing error falls back to HTTP <status>', async () => {
    for (const body of [{ error: 5 }, { error: { msg: 'x' } }, {}, null, 'oops', [1], { message: 'x' }]) {
      const e = await fail(createClient('http://h', fake(400, body).fetch).shorten('https://a'));
      expect(e).toBeInstanceOf(ApiError);
      expect(e.status).toBe(400);
      expect(e.message).toBe('HTTP 400');
    }
  });

  test('one fetch per call, nothing else', async () => {
    const f = fake(200, { count: 3 });
    const c = createClient('http://h', f.fetch);
    await c.stats();
    await c.stats();
    expect(f.calls).toHaveLength(2);
  });
});
