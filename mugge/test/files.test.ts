import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseFiles, renderFiles } from '../src/engine/files.ts';

// Shared with llmap's tests/test_harness.py, so the trainer and the engine agree on the format.
const cases = JSON.parse(readFileSync(join(import.meta.dir, 'fixtures/files-cases.json'), 'utf8'));

describe('coder answer format', () => {
  for (const c of cases.parse) {
    test(`parse: ${c.why}`, () => {
      expect(parseFiles(c.answer, c.owns)).toEqual({ files: c.files, note: c.note });
    });
  }
  test('render matches the shared cases and parses back', () => {
    for (const c of cases.render) {
      const text = renderFiles(c.files, c.note);
      expect(text).toBe(c.answer);
      const back = parseFiles(text);
      for (const [p, content] of Object.entries(c.files as Record<string, string>)) expect(back.files[p]).toBe(content.replace(/\n+$/, '') + '\n');
      expect(back.note).toBe(c.note);
    }
  });
});
