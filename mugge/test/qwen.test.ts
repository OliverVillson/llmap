/**
 * The engine on Qwen3.8-27B: thinking inside content (no --reasoning-parser), thinking split out
 * (with it), a call cut while thinking, the per-mode sampling, the preset, and the smoke through
 * the fake model answering the way Qwen does.
 */
import { describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runSmoke } from '../bench/smoke.ts';
import { inferenceFromEnv } from '../src/config.ts';
import { resolveMachine } from '../src/engine/backends.ts';
import { parseFiles, stripThinking } from '../src/engine/files.ts';
import { OpenAICompatible, parseJsonAnswer } from '../src/engine/inference.ts';
import { classifyAnswer } from '../src/engine/telemetry.ts';

describe('thinking inside content', () => {
  test('stripThinking: closed, opened by the template (bare </think>), unclosed, none', () => {
    expect(stripThinking('<think>\nplan\n</think>\n\na.py\n```\nx\n```')).toBe('\n\na.py\n```\nx\n```');
    expect(stripThinking('plan, the template opened it\n</think>\n\nanswer')).toBe('\n\nanswer');
    expect(stripThinking('<think>\nstill thinking when the cap hit')).toBe('');
    expect(stripThinking('a.py\n```\nx\n```')).toBe('a.py\n```\nx\n```');
  });

  test('a file drafted inside the thinking is never written, with or without the opening tag', () => {
    const answer = 'Let me draft:\nscratch.py\n```python\nbad\n```\n</think>\n\nI will write it now.\n\nsrc/a.py\n```python\nok\n```';
    expect(parseFiles(answer, ['src/a.py'])).toEqual({ files: { 'src/a.py': 'ok\n' }, note: '' });
    expect(parseFiles(`<think>\n${answer}`, ['src/a.py']).files).toEqual({ 'src/a.py': 'ok\n' });
    // a prose line before the files is fine: the path is still alone on the line before its fence
    expect(classifyAnswer(answer, ['src/a.py'], parseFiles(answer, ['src/a.py'])).loose).toEqual([]);
    expect(parseJsonAnswer<{ verdict: string }>('weighing it\n</think>\n{"verdict": "approve", "reason": ""}').verdict).toBe('approve');
  });

  test('the client: value is the answer, raw keeps it all; a call cut while thinking with no parser has no answer', async () => {
    const replies = [
      { message: { content: 'thinking\n</think>\n\nsrc/a.py\n```python\nok\n```' }, finish_reason: 'stop' },
      { message: { content: 'src/a.py\n```python\nok\n```', reasoning_content: 'thinking' }, finish_reason: 'stop' },
      { message: { content: 'thinking, with a draft:\nsrc/a.py\n```python\nbad\n```\nand more' }, finish_reason: 'length' },
    ];
    let n = 0;
    const server = Bun.serve({ port: 0, fetch: () => Response.json({ choices: [replies[n++]], usage: { prompt_tokens: 1, completion_tokens: 1 } }) });
    try {
      const inf = new OpenAICompatible({ ...inferenceFromEnv({}), baseUrl: `${server.url}v1` });
      const msgs = [{ role: 'user' as const, content: 'hi' }];
      const a = await inf.text('coder', msgs, 100);
      expect(a.value).toBe('\n\nsrc/a.py\n```python\nok\n```');
      expect(a.raw.startsWith('thinking\n</think>')).toBe(true);
      expect(a.reasoningChars).toBe('thinking\n</think>'.length);
      expect(a.reasoning).toBe('thinking\n');
      const b = await inf.text('coder', msgs, 100);
      expect(b.value).toBe('src/a.py\n```python\nok\n```');
      expect(b.reasoningChars).toBe('thinking'.length);
      expect(b.reasoning).toBe('thinking');
      const c = await inf.text('coder', msgs, 100);
      expect(c.cut).toBe(true);
      expect(c.thinking).toBe(true);
      expect(c.value).toBe('');
      expect(c.reasoning).toBe(c.raw);
      expect(parseFiles(c.value, ['src/a.py']).files).toEqual({});
    } finally {
      server.stop(true);
    }
  });
});

test('the qwen3.8-27b preset: 32 KiB a token at fp8, the recurrent state on top', () => {
  const m = resolveMachine({ name: 'b200', url: 'http://x/v1', preset: 'qwen3.8-27b', cache_type: 'fp8', ctx: 32768, memory_gb: 100, server: 'vllm' });
  expect(m.kvBytesPerToken).toBe(16 * 4 * 256 * 2);
  expect(m.bytesPerSlot).toBe(32768 * 32768 + 153_944_064);
  expect(m.server).toBe('vllm');
  expect(() => resolveMachine({ name: 'x', url: 'http://x/v1', server: 'ollama' as never })).toThrow(/server must be/);
});

test('the smoke meets the bar on the fake model answering the way Qwen3.8 does', async () => {
  const out = mkdtempSync(join(tmpdir(), 'mugge-qwen-smoke-'));
  try {
    const r = await runSmoke({ fake: true, fakeQwen: true, fakeLatencyMs: 5, out, quiet: true, skipRefs: true });
    expect(r.verdict).toBe('MEETS');
    expect(r.passed).toBe(r.tickets);
    expect(r.calls.dropped).toBe(0);
    expect(r.calls.loose).toBe(0);
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}, 120_000);
