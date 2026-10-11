// Hidden tests for the format functions (cli/src/format.ts doc comments).
import { describe, expect, test } from 'bun:test';
import { formatError, formatLookup, formatShorten, formatStats, usage } from '../src/format.ts';
import { ApiError } from '../src/types.ts';

describe('format (hidden)', () => {
  test('shorten keeps code and url verbatim', () => {
    expect(formatShorten({ code: 'lYGhA16ahyf', url: 'https://x.example/a b?c=1#d' })).toBe('lYGhA16ahyf -> https://x.example/a b?c=1#d');
    expect(formatShorten({ code: '0', url: '' })).toBe('0 -> ');
  });

  test('lookup counts', () => {
    expect(formatLookup({ code: '1', url: 'https://a', hits: 2 })).toBe('1 -> https://a (2 hits)');
    expect(formatLookup({ code: '1', url: 'https://a', hits: 11 })).toBe('1 -> https://a (11 hits)');
    expect(formatLookup({ code: '1', url: 'https://a', hits: 21 })).toBe('1 -> https://a (21 hits)');
    expect(formatLookup({ code: '1', url: 'https://a', hits: 1000000 })).toBe('1 -> https://a (1000000 hits)');
  });

  test('stats counts', () => {
    expect(formatStats({ count: 2 })).toBe('2 links');
    expect(formatStats({ count: 11 })).toBe('11 links');
    expect(formatStats({ count: 101 })).toBe('101 links');
  });

  test('error uses message and status', () => {
    expect(formatError(new ApiError(500, 'HTTP 500'))).toBe('error: HTTP 500 (HTTP 500)');
    expect(formatError(new ApiError(400, 'invalid url'))).toBe('error: invalid url (HTTP 400)');
    expect(formatError(new ApiError(405, 'method not allowed'))).toBe('error: method not allowed (HTTP 405)');
  });

  test('usage lists each command on its own line', () => {
    const u = usage();
    const lines = u.split('\n');
    expect(lines[0]).toBe('usage: short [--server <url>] [--json] <command>');
    expect(lines.length).toBeGreaterThanOrEqual(5);
    const rest = lines.slice(1);
    for (const cmd of ['shorten', 'lookup', 'stats', 'help']) {
      expect(rest.some((l) => l.includes(cmd))).toBe(true);
    }
    const which = ['shorten', 'lookup', 'stats', 'help'].map((cmd) => rest.findIndex((l) => new RegExp(`\\b${cmd}\\b`).test(l)));
    expect(new Set(which).size).toBe(4);
    expect(usage()).toBe(u);
  });
});
