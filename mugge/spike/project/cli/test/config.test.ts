import { describe, expect, test } from 'bun:test';
import { resolveConfig } from '../src/config.ts';
import { DEFAULT_SERVER } from '../src/types.ts';

describe('resolveConfig', () => {
  test('defaults', () => {
    expect(resolveConfig({}, {})).toEqual({ server: DEFAULT_SERVER, json: false });
  });

  test('env', () => {
    expect(resolveConfig({ MUGGE_SERVER: 'https://s.example/', MUGGE_JSON: '1' }, {})).toEqual({ server: 'https://s.example', json: true });
    expect(resolveConfig({ MUGGE_JSON: 'true' }, {}).json).toBe(true);
    expect(resolveConfig({ MUGGE_JSON: '0' }, {}).json).toBe(false);
  });

  test('flags win over env', () => {
    expect(resolveConfig({ MUGGE_SERVER: 'https://env.example' }, { server: 'http://flag:9///', json: true })).toEqual({
      server: 'http://flag:9',
      json: true,
    });
    expect(resolveConfig({ MUGGE_JSON: '1' }, { json: false }).json).toBe(true);
  });

  test('invalid server', () => {
    expect(() => resolveConfig({}, { server: 'ftp://x' })).toThrow('invalid server url: ftp://x');
    expect(() => resolveConfig({ MUGGE_SERVER: 'localhost:8080' }, {})).toThrow('invalid server url: localhost:8080');
    expect(() => resolveConfig({}, { server: 'http://' })).toThrow('invalid server url: http://');
  });
});
