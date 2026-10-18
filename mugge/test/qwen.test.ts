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
import { inferenceFromEnv, machinesFromEnv } from '../src/config.ts';
import { BackendPool, resolveMachine } from '../src/engine/backends.ts';
import { parseFiles, stripThinking } from '../src/engine/files.ts';
import { effortsFromEnv, OpenAICompatible, parseJsonAnswer } from '../src/engine/inference.ts';
import { reviewCapFromEnv, reviewThinkingFromEnv } from '../src/engine/review.ts';
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

describe('thinking effort per role', () => {
  test('MUGGE_REASONING_EFFORT: one level, levels per role, or both', () => {
    expect(effortsFromEnv({})).toEqual({ roles: {} });
    expect(effortsFromEnv({ MUGGE_REASONING_EFFORT: 'Medium' })).toEqual({ all: 'medium', roles: {} });
    expect(effortsFromEnv({ MUGGE_REASONING_EFFORT: 'worker=off, lead=xhigh,manager=xhigh' })).toEqual({ roles: { worker: 'off', lead: 'xhigh', manager: 'xhigh' } });
    expect(effortsFromEnv({ MUGGE_REASONING_EFFORT: 'low,manager=xhigh' })).toEqual({ all: 'low', roles: { manager: 'xhigh' } });
    expect(() => effortsFromEnv({ MUGGE_REASONING_EFFORT: 'coder=low' })).toThrow(/worker, lead or manager/);
    expect(() => effortsFromEnv({ MUGGE_REASONING_EFFORT: 'worker=max' })).toThrow(/xhigh, medium, low or off/);
    // off for everything turns thinking off, whatever MUGGE_THINKING says
    expect(inferenceFromEnv({ MUGGE_REASONING_EFFORT: 'off' }).extra).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(inferenceFromEnv({ MUGGE_REASONING_EFFORT: 'worker=medium' }).extra).toEqual({ chat_template_kwargs: { enable_thinking: true } });
  });

  test('each role\'s calls carry its level; a machine\'s own level wins; the cut retry still turns thinking off', async () => {
    const bodies: any[] = [];
    const server = Bun.serve({ port: 0, fetch: async (req) => (bodies.push(await req.json()), Response.json({ choices: [{ message: { content: '{"verdict":"approve","reason":""}' }, finish_reason: 'stop' }], usage: {} })) });
    const url = `${server.url}v1`;
    const env = { MUGGE_REASONING_EFFORT: 'worker=medium,lead=xhigh,manager=low', MUGGE_BACKENDS: JSON.stringify([
      { name: 'w', url, roles: ['worker'], slots: 4 },
      { name: 'l', url, roles: ['lead'], slots: 1 },
      { name: 'm', url, roles: ['manager'], slots: 1, reasoning_effort: 'off' },
    ]) };
    try {
      const pool = BackendPool.fromMachines(machinesFromEnv(env), inferenceFromEnv({ ...env, MUGGE_INFERENCE_URL: url }), env);
      const msgs = [{ role: 'user' as const, content: 'hi' }];
      await pool.forAgent({ role: 'worker' }).text('coder', msgs, 10);
      await pool.forAgent({ role: 'worker' }).text('coder', msgs, 10, undefined, { thinking: false });
      // the review's thinking off is only a default: the lead's level wins
      await pool.forAgent({ role: 'lead' }).complete('lead', msgs, {}, 10, undefined, { thinking: false, thinkingIsDefault: true });
      await pool.forAgent({ role: 'manager' }).complete('architect', msgs, {}, 10);
      const kw = bodies.map((b) => b.chat_template_kwargs);
      expect(kw[0]).toEqual({ enable_thinking: true, reasoning_effort: 'medium' });
      expect(kw[1]).toMatchObject({ enable_thinking: false });
      expect(kw[2]).toEqual({ enable_thinking: true, reasoning_effort: 'xhigh' });
      expect(kw[3]).toEqual({ enable_thinking: false });
      // and the sampling follows: the cut retry and the manager go out with the non-thinking set
      expect(bodies.map((b) => b.temperature)).toEqual([1, 0.7, 1, 0.7]);
      expect(pool.thinksFor('worker', () => false)).toBe(true);
      expect(pool.thinksFor('manager', () => true)).toBe(false);
    } finally {
      server.stop(true);
    }
    expect(reviewThinkingFromEnv({ MUGGE_REASONING_EFFORT: 'lead=xhigh' })).toBe(true);
    expect(reviewThinkingFromEnv({ MUGGE_REASONING_EFFORT: 'lead=xhigh', MUGGE_REVIEW_THINKING: 'off' })).toBe(false);
    expect(reviewThinkingFromEnv({ MUGGE_REASONING_EFFORT: 'lead=off' })).toBe(false);
    expect(reviewCapFromEnv({}, true)).toBe(16_384);
    expect(reviewCapFromEnv({}, false)).toBe(2048);
  });
});
