import { expect, test } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findProjects } from '../bench/load.ts';
import { runBench } from '../bench/run.ts';
import { capReport, SPIKE_DIR } from '../bench/smoke.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { cutRetryFromEnv } from '../src/engine/harness.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { Telemetry, type CallRecord } from '../src/engine/telemetry.ts';

const oneTicket = () => findProjects(SPIKE_DIR).map((p) => ({ ...p, tickets: p.tickets.filter((t) => t.ticket.id === 'c-hash') })).filter((p) => p.tickets.length);

async function overthink(cutRetry?: string) {
  const out = mkdtempSync(join(tmpdir(), 'mugge-cap-'));
  const projects = oneTicket();
  const solutions = { 'c-hash': { files: JSON.parse(readFileSync(join(SPIKE_DIR, 'solutions', 'c-hash.json'), 'utf8')).files, note: '' } };
  const fake = startFakeModel({ solutions, overthink: ['c-hash'] });
  const before = process.env.MUGGE_CUT_RETRY;
  if (cutRetry) process.env.MUGGE_CUT_RETRY = cutRetry;
  else delete process.env.MUGGE_CUT_RETRY;
  try {
    const tel = new Telemetry({ dir: out, warn: null });
    const rows = await runBench({
      projects, sandbox: new LocalSandbox(), seeds: 1, concurrency: 1, out, telemetry: tel, maxTokens: 512,
      inference: () => new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', maxCallTokens: 512, extra: { chat_template_kwargs: { enable_thinking: true } } }),
    });
    tel.close();
    const records: CallRecord[] = readFileSync(join(out, 'calls.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    return { rows, records, sent: fake.calls().map((c) => c.thinking) };
  } finally {
    fake.stop();
    if (before === undefined) delete process.env.MUGGE_CUT_RETRY;
    else process.env.MUGGE_CUT_RETRY = before;
    rmSync(out, { recursive: true, force: true });
  }
}

test('a thinking call that runs out of tokens: the ticket goes on without thinking and passes', async () => {
  const { rows, records, sent } = await overthink();
  expect(sent).toEqual([true, false]);
  expect(rows[0]!.solved).toBe(true);
  expect(records.map((r) => [r.thinking, r.drop, r.clean])).toEqual([[true, 'cut', false], [false, null, true]]);
  const cap = capReport(records, 512, 'on')!;
  expect([cap.outOfTokens, cap.retriedWithoutThinking]).toEqual([1, 1]);
  expect(cap.advice).toContain('a bigger cap will not help'); // the canned answer is short: the fake ran away
}, 60_000);

test('MUGGE_CUT_RETRY=same keeps thinking on: every call runs out and the ticket fails', async () => {
  const { rows, sent } = await overthink('same');
  expect(sent.length).toBe(4);
  expect(sent.every((t) => t === true)).toBe(true);
  expect(rows[0]!.solved).toBe(false);
}, 60_000);

test('MUGGE_CUT_RETRY takes nothink or same only', () => {
  expect(cutRetryFromEnv({})).toBe('nothink');
  expect(cutRetryFromEnv({ MUGGE_CUT_RETRY: 'SAME' })).toBe('same');
  expect(() => cutRetryFromEnv({ MUGGE_CUT_RETRY: 'bigger' })).toThrow('nothink or same');
});

test('cap report: says when the cap is too small, and when it is far bigger than needed', () => {
  const rec = (completionTokens: number, more: Partial<CallRecord> = {}): CallRecord => ({
    v: 1, run: 'r', at: 0, ticket: 't', attempt: 1, kind: 'write', agent: null, team: null, machine: null, model: null,
    promptTokens: 1, completionTokens, reasoningTokens: null, reasoningChars: null, maxTokens: 8192, thinking: null, ms: 1, tokS: 1,
    cut: false, forced: null, files: 1, missing: 0, loose: 0, drop: null, error: null, clean: true, ...more,
  });
  const small = capReport([rec(900), rec(1200), rec(1500)], 8192, 'off')!;
  expect(small.finished).toEqual({ p50: 1200, p90: 1500, max: 1500 });
  expect(small.advice).toContain('--max-tokens 3072');
  const tight = capReport([rec(8192, { cut: true, drop: 'cut' }), rec(5000), rec(4000, { forced: 'length' })], 8192, 'on')!;
  expect(tight.outOfTokens).toBe(2);
  expect(tight.advice).toContain('--max-tokens 16384');
  expect(tight.advice).toContain('For BlackHole');
  const runaway = capReport([rec(8192, { cut: true, drop: 'cut' }), rec(700), rec(900)], 8192, 'off')!;
  expect(runaway.advice).toContain('a bigger cap will not help');
  expect(capReport([rec(0, { error: 'boom' })], 8192, 'on')).toBeNull();
});
