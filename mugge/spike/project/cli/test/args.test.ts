import { describe, expect, test } from 'bun:test';
import { parseArgs } from '../src/args.ts';

const ok = (argv: string[]) => {
  const r = parseArgs(argv);
  if (!r.ok) throw new Error(`expected ok, got error: ${r.error}`);
  return r.args;
};
const err = (argv: string[]) => {
  const r = parseArgs(argv);
  if (r.ok) throw new Error(`expected error for ${JSON.stringify(argv)}`);
  return r.error;
};

describe('parseArgs', () => {
  test('commands', () => {
    expect(ok(['shorten', 'https://a.example'])).toEqual({ command: { kind: 'shorten', url: 'https://a.example' }, json: false });
    expect(ok(['lookup', 'abc'])).toEqual({ command: { kind: 'lookup', code: 'abc' }, json: false });
    expect(ok(['stats'])).toEqual({ command: { kind: 'stats' }, json: false });
    expect(ok(['help'])).toEqual({ command: { kind: 'help' }, json: false });
  });

  test('help forms', () => {
    expect(ok([]).command).toEqual({ kind: 'help' });
    expect(ok(['-h']).command).toEqual({ kind: 'help' });
    expect(ok(['stats', '--help']).command).toEqual({ kind: 'help' });
  });

  test('options anywhere', () => {
    expect(ok(['--json', 'stats'])).toEqual({ command: { kind: 'stats' }, json: true });
    expect(ok(['lookup', 'x', '--server', 'http://h:1'])).toEqual({ command: { kind: 'lookup', code: 'x' }, server: 'http://h:1', json: false });
    expect(ok(['--server=http://h:2', 'shorten', 'https://a', '--json'])).toEqual({
      command: { kind: 'shorten', url: 'https://a' },
      server: 'http://h:2',
      json: true,
    });
  });

  test('server is absent when not given', () => {
    expect(ok(['stats']).server).toBeUndefined();
  });

  test('errors', () => {
    expect(err(['--verbose', 'stats'])).toBe('unknown option: --verbose');
    expect(err(['stats', '--server'])).toBe('--server requires a value');
    expect(err(['--server=', 'stats'])).toBe('--server requires a value');
    expect(err(['frobnicate'])).toBe('unknown command: frobnicate');
    expect(err(['shorten'])).toBe('shorten requires a url');
    expect(err(['lookup', '--json'])).toBe('lookup requires a code');
    expect(err(['stats', 'extra'])).toBe('unexpected argument: extra');
    expect(err(['lookup', 'a', 'b'])).toBe('unexpected argument: b');
  });
});
