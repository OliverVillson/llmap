import { describe, expect, test } from 'bun:test';
import { formatError, formatLookup, formatShorten, formatStats, usage } from '../src/format.ts';
import { ApiError } from '../src/types.ts';

describe('format', () => {
  test('shorten', () => {
    expect(formatShorten({ code: '1', url: 'https://a.example' })).toBe('1 -> https://a.example');
  });

  test('lookup pluralizes hits', () => {
    expect(formatLookup({ code: 'Zz', url: 'https://b', hits: 0 })).toBe('Zz -> https://b (0 hits)');
    expect(formatLookup({ code: 'Zz', url: 'https://b', hits: 1 })).toBe('Zz -> https://b (1 hit)');
    expect(formatLookup({ code: 'Zz', url: 'https://b', hits: 12 })).toBe('Zz -> https://b (12 hits)');
  });

  test('stats pluralizes links', () => {
    expect(formatStats({ count: 0 })).toBe('0 links');
    expect(formatStats({ count: 1 })).toBe('1 link');
    expect(formatStats({ count: 3 })).toBe('3 links');
  });

  test('error', () => {
    expect(formatError(new ApiError(404, 'not found'))).toBe('error: not found (HTTP 404)');
  });

  test('usage', () => {
    const lines = usage().split('\n');
    expect(lines[0]).toBe('usage: short [--server <url>] [--json] <command>');
    const rest = lines.slice(1).join('\n');
    for (const cmd of ['shorten', 'lookup', 'stats', 'help']) expect(rest).toContain(cmd);
  });
});
