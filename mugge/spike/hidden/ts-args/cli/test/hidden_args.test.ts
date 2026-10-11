// Hidden tests for parseArgs (cli/src/args.ts doc comment, cli/src/types.ts).
import { describe, expect, test } from 'bun:test';
import { parseArgs } from '../src/args.ts';

const ok = (argv: string[]) => {
  const r = parseArgs(argv);
  if (!r.ok) throw new Error(`expected ok for ${JSON.stringify(argv)}, got error: ${r.error}`);
  return r.args;
};
const err = (argv: string[]) => {
  const r = parseArgs(argv);
  if (r.ok) throw new Error(`expected error for ${JSON.stringify(argv)}`);
  return r.error;
};

describe('parseArgs (hidden)', () => {
  test('help wins wherever -h or --help appears', () => {
    expect(ok(['shorten', '--help']).command).toEqual({ kind: 'help' });
    expect(ok(['frobnicate', '-h']).command).toEqual({ kind: 'help' });
    expect(ok(['lookup', 'a', 'b', '--help']).command).toEqual({ kind: 'help' });
    expect(ok(['--help', '--json']).command).toEqual({ kind: 'help' });
  });

  test('server value forms', () => {
    expect(ok(['--server', 'http://h:1', 'stats']).server).toBe('http://h:1');
    expect(ok(['lookup', '--server', 'http://h', 'abc'])).toEqual({ command: { kind: 'lookup', code: 'abc' }, server: 'http://h', json: false });
    expect(ok(['--server=http://h:1/?a=b', 'stats']).server).toBe('http://h:1/?a=b');
    expect(ok(['shorten', 'https://x.example/?q=1&r=2', '--server=https://s'])).toEqual({
      command: { kind: 'shorten', url: 'https://x.example/?q=1&r=2' },
      server: 'https://s',
      json: false,
    });
  });

  test('server key is left out, json is always present', () => {
    const a = ok(['stats', '--json']);
    expect(Object.keys(a).sort()).toEqual(['command', 'json']);
    expect(a.json).toBe(true);
    const b = ok(['lookup', 'q']);
    expect('server' in b).toBe(false);
    expect(b.json).toBe(false);
    expect(ok(['--json', '--json', 'stats']).json).toBe(true);
  });

  test('option errors', () => {
    expect(err(['-x', 'stats'])).toBe('unknown option: -x');
    expect(err(['stats', '--jsonx'])).toBe('unknown option: --jsonx');
    expect(err(['stats', '-'])).toBe('unknown option: -');
    expect(err(['--server', '', 'stats'])).toBe('--server requires a value');
    expect(err(['lookup', 'abc', '--server'])).toBe('--server requires a value');
  });

  test('command errors', () => {
    expect(err(['Stats'])).toBe('unknown command: Stats');
    expect(err(['x', 'y'])).toBe('unknown command: x');
    expect(err(['lookup'])).toBe('lookup requires a code');
    expect(err(['--json', 'shorten'])).toBe('shorten requires a url');
    expect(err(['stats', 'a', 'b'])).toBe('unexpected argument: a');
    expect(err(['shorten', 'https://a', 'https://b', 'c'])).toBe('unexpected argument: https://b');
    expect(err(['help', 'me'])).toBe('unexpected argument: me');
  });
});
