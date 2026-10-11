// Hidden tests for resolveConfig (cli/src/config.ts doc comment).
import { describe, expect, test } from 'bun:test';
import { resolveConfig } from '../src/config.ts';
import { DEFAULT_SERVER } from '../src/types.ts';

describe('resolveConfig (hidden)', () => {
  test('precedence', () => {
    expect(resolveConfig({ OTHER: 'x' }, {})).toEqual({ server: DEFAULT_SERVER, json: false });
    expect(resolveConfig({ MUGGE_SERVER: 'http://env:1' }, { server: undefined })).toEqual({ server: 'http://env:1', json: false });
    expect(resolveConfig({ MUGGE_SERVER: 'http://env:1' }, { server: 'http://flag:2' }).server).toBe('http://flag:2');
    expect(resolveConfig({}, { server: 'https://only-flag.example' }).server).toBe('https://only-flag.example');
  });

  test('trailing slashes only', () => {
    expect(resolveConfig({}, { server: 'https://h.example/api//' }).server).toBe('https://h.example/api');
    expect(resolveConfig({}, { server: 'http://h////' }).server).toBe('http://h');
    expect(resolveConfig({ MUGGE_SERVER: 'http://10.0.0.1:8080/' }, {}).server).toBe('http://10.0.0.1:8080');
  });

  test('json flag and env values', () => {
    expect(resolveConfig({}, { json: true }).json).toBe(true);
    expect(resolveConfig({ MUGGE_JSON: 'true' }, { json: false }).json).toBe(true);
    for (const v of ['TRUE', 'yes', '', '2', 'false', ' 1']) {
      expect(resolveConfig({ MUGGE_JSON: v }, {}).json).toBe(false);
    }
  });

  test('invalid servers report the value as given', () => {
    expect(() => resolveConfig({}, { server: 'mailto:x' })).toThrow('invalid server url: mailto:x');
    expect(() => resolveConfig({}, { server: 'http:/h' })).toThrow('invalid server url: http:/h');
    expect(() => resolveConfig({}, { server: 'http:// h' })).toThrow('invalid server url: http:// h');
    expect(() => resolveConfig({}, { server: 'http:///path' })).toThrow('invalid server url: http:///path');
    expect(() => resolveConfig({ MUGGE_SERVER: 'https://ok.example' }, { server: 'ftp://flag' })).toThrow('invalid server url: ftp://flag');
    expect(() => resolveConfig({ MUGGE_SERVER: 'h:1/' }, { json: true })).toThrow('invalid server url: h:1/');
    let caught: unknown;
    try {
      resolveConfig({}, { server: 'nope' });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe('invalid server url: nope');
  });
});
