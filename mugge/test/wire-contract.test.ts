/**
 * The worker wire contract: the system prompt and user message of every worker write and fix call
 * must stay byte-identical to the base branch's (claude/project-thread-uidsop), because the
 * specialists are trained on byte-exact copies of them (llmap stages/harness.py). The base
 * branch's harness.ts, prompts.ts, files.ts and what they import are kept verbatim under
 * fixtures/wire-base/ (from `git show origin/claude/project-thread-uidsop:mugge/src/...`), and each
 * test drives both sides with the same inputs and compares the bytes.
 */
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { inferenceFromEnv, machinesFromEnv } from '../src/config.ts';
import { BackendPool } from '../src/engine/backends.ts';
import * as headFiles from '../src/engine/files.ts';
import { runTicket as headRunTicket } from '../src/engine/harness.ts';
import type { ChatMessage, Completion, Inference } from '../src/engine/inference.ts';
import * as head from '../src/engine/prompts.ts';
import { reviewFeedback } from '../src/engine/review.ts';
import type { RunOptions, RunResult, Sandbox } from '../src/engine/sandbox.ts';
import { tail as headTail } from '../src/engine/sandbox.ts';
import type { Plan, Ticket } from '../src/tickets/schema.ts';
import * as baseFiles from './fixtures/wire-base/engine/files.ts';
import { runTicket as baseRunTicket } from './fixtures/wire-base/engine/harness.ts';
import { OpenAICompatible as BaseOpenAICompatible } from './fixtures/wire-base/engine/inference.ts';
import * as base from './fixtures/wire-base/engine/prompts.ts';
import { tail as baseTail } from './fixtures/wire-base/engine/sandbox.ts';

const OWNED_HEADER = '## Your files as they are now';

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), 'mugge-wire-'));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The first byte where a and b differ, with context, or null when they are the same bytes. */
function firstDiff(a: string, b: string): string | null {
  const x = Buffer.from(a, 'utf8'), y = Buffer.from(b, 'utf8');
  if (x.equals(y)) return null;
  let i = 0;
  while (i < x.length && i < y.length && x[i] === y[i]) i++;
  const ctx = (buf: Buffer) => JSON.stringify(buf.subarray(Math.max(0, i - 60), i + 60).toString('utf8'));
  return `byte ${i} (lengths ${x.length} vs ${y.length}):\n  head: ${ctx(x)}\n  base: ${ctx(y)}`;
}

/** Same messages, same roles, same order, same bytes. */
function expectSameMessages(headMsgs: ChatMessage[], baseMsgs: ChatMessage[], what: string) {
  expect(headMsgs.map((m) => m.role)).toEqual(baseMsgs.map((m) => m.role));
  for (let i = 0; i < baseMsgs.length; i++) {
    const d = firstDiff(headMsgs[i]!.content, baseMsgs[i]!.content);
    if (d) throw new Error(`${what}: message ${i} (${baseMsgs[i]!.role}) differs at ${d}`);
  }
  expect(Buffer.from(JSON.stringify(headMsgs)).equals(Buffer.from(JSON.stringify(baseMsgs)))).toBe(true);
}

function commonPrefix(a: string, b: string): number {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

function tk(o: Partial<Ticket> & { id: string }): Ticket {
  return {
    title: `Ticket ${o.id}`, role: 'implement', model: 'coder', priority: 3, depends_on: [], owns: [], reads: [],
    context: 'Implement it.', acceptance: ['true'], image: 'node', budget: { attempts: 4, max_tokens: 60000 }, escalate_to: null, ...o,
  };
}

const BIG = Array.from({ length: 1200 }, (_, i) => `line ${i}: ${'x'.repeat(i % 37)}`).join('\n'); // > 20 000 chars
const LONG_LOG = Array.from({ length: 400 }, (_, i) => `  at frame${i} (src/money.ts:${i}:${i % 9}) ✗ expected ${i} got ${i + 1}`).join('\n');

/** A repo-shaped folder: interfaces, globbable reads, a huge file, unicode and CRLF, owned files existing, empty and missing. */
function seed(name: string): string {
  const wt = join(dir, name);
  const files: Record<string, string> = {
    'src/types.ts': 'export interface Money {\r\n  cents: number; // ÆØÅ — ✓\r\n}\r\n',
    'src/api.h': '#pragma once\nint add(int a, int b);\n',
    'src/money.ts': "import type { Money } from './types.ts';\n\n/** Fenced:\n```ts\nx\n```\n*/\nexport const zero: Money = { cents: 0 };\n",
    'src/empty.ts': '',
    'src/util/a.ts': 'export const a = 1;\n',
    'src/util/b.ts': 'export const b = 2;\n\n\n',
    'src/util/c.js': 'module.exports = 3;\n',
    'lib/x/y.py': 'def y():\n    return "y"\n',
    'lib/z.py': 'Z = 1',
    'docs/notes.md': '# Notes\n\nTabs\there, trailing spaces   \n',
    'big.txt': BIG,
    'node_modules/pkg/index.ts': 'never packed\n',
  };
  for (const [p, c] of Object.entries(files)) {
    mkdirSync(dirname(join(wt, p)), { recursive: true });
    writeFileSync(join(wt, p), c);
  }
  return wt;
}

const PLAN_INTERFACES = ['src/types.ts', 'src/api.h', 'src/money.ts'];

const TICKETS: Array<{ why: string; ticket: Ticket; interfaces: string[]; memory: string }> = [
  { why: 'one owned file that does not exist yet, nothing to read', ticket: tk({ id: 'new', owns: ['src/new.ts'] }), interfaces: [], memory: '' },
  {
    why: 'several owned files (existing, missing, empty), globs, a missing read, an owned read, an interface read, a huge read, notes',
    ticket: tk({
      id: 'full', title: 'Money: add and format', role: 'implement',
      owns: ['src/money.ts', 'src/new.ts', 'src/empty.ts'],
      reads: ['docs/notes.md', 'src/util/*.ts', 'lib/**/*.py', 'missing.txt', 'src/money.ts', 'src/types.ts', 'big.txt', 'node_modules/**'],
      context: 'Add `add(a, b)` and `format(m)`.\n\nKeep cents as integers.  \nEdge: negative amounts.',
      acceptance: ['bun test test/money.test.ts', 'bunx tsc --noEmit'],
    }),
    interfaces: PLAN_INTERFACES, memory: 'Use cents.\nNo floats.',
  },
  {
    why: 'test role, empty context, interfaces only, header text inside the context',
    ticket: tk({ id: 'tst', role: 'test', owns: ['test/money.test.ts'], context: '', reads: ['lib/*.py'], acceptance: ['bun test'] }),
    interfaces: PLAN_INTERFACES, memory: '',
  },
  {
    why: 'context that itself contains the section headers',
    ticket: tk({ id: 'hdr', role: 'fix', owns: ['src/money.ts'], context: `Odd text:\n${OWNED_HEADER}\n## Fix\n--- src/money.ts`, reads: ['big.txt'] }),
    interfaces: ['src/api.h'], memory: 'x',
  },
];

function failures(tail: (s: string, n?: number) => string, ticket: Ticket) {
  return [
    { command: 'bun test', output: '1 fail' },
    { command: 'bun test test/money.test.ts', output: tail(LONG_LOG) },
    { command: 'bunx tsc --noEmit', output: '(timed out)\n' + tail(LONG_LOG) },
    { command: 'make', output: '' },
    { command: '(model call)', output: 'inference 500: {"error":"boom"}' },
    { command: '(write files)', output: `You wrote files you do not own: evil.ts. Write only: ${ticket.owns.join(', ')}.` },
  ];
}

describe('builders: head prompts.ts vs base prompts.ts', () => {
  test('the system prompt is the same bytes', () => {
    expect(firstDiff(head.SYSTEM, base.SYSTEM)).toBeNull();
  });

  test('tail() trims a long failing output the same way on both sides', () => {
    expect(LONG_LOG.length).toBeGreaterThan(3000);
    expect(headTail(LONG_LOG)).toBe(baseTail(LONG_LOG));
    expect(headTail(LONG_LOG, 800)).toBe(baseTail(LONG_LOG, 800));
    expect(headTail(LONG_LOG).length).toBe(3001);
  });

  for (const c of TICKETS) {
    test(`write and fix messages: ${c.why}`, () => {
      const wt = seed(`b-${c.ticket.id}`);
      const plan: Plan = { interfaces: c.interfaces, tickets: [c.ticket] };
      const hp = head.buildPack(wt, plan, c.ticket, c.memory);
      const bp = base.buildPack(wt, plan, c.ticket, c.memory);
      expect(hp).toEqual(bp);
      expect(head.expandReads(wt, c.ticket.reads)).toEqual(base.expandReads(wt, c.ticket.reads));
      expectSameMessages(head.writeMessages(wt, hp), base.writeMessages(wt, bp), `${c.ticket.id} write`);
      const hf = failures(headTail, c.ticket), bf = failures(baseTail, c.ticket);
      for (let i = 0; i < hf.length; i++) {
        expectSameMessages(head.fixMessages(wt, hp, hf[i]!), base.fixMessages(wt, bp, bf[i]!), `${c.ticket.id} fix ${hf[i]!.command}`);
      }
    });
  }

  test('the huge read really is truncated, the glob skips node_modules and .js, the owned read is not repeated', () => {
    const c = TICKETS[1]!;
    const wt = seed('b-shape');
    const msg = head.writeMessages(wt, head.buildPack(wt, { interfaces: c.interfaces, tickets: [c.ticket] }, c.ticket, c.memory))[1]!.content;
    expect(msg).toContain('\n…(truncated)');
    expect(msg).not.toContain('never packed');
    expect(msg).not.toContain('module.exports');
    expect(msg.split('--- src/money.ts\n').length).toBe(2); // only under the owned section
    expect(msg).toContain('\r\n  cents: number; // ÆØÅ — ✓');
  });
});

describe('files.ts: renderFiles and parseFiles on both sides', () => {
  const samples: Array<[Record<string, string>, string]> = [
    [{ 'src/a.ts': 'export const a = 1;\n' }, ''],
    [{ 'src/a.ts': 'x\n\n\n', 'README.md': '# T\n```sh\nmake\n```\n', 'Makefile': 'all:\n\techo ok\n' }, 'fixed the build'],
    [{ 'x.PY': '````\n`````\n', 'empty.txt': '' }, ''],
    [{ 'src/types.ts': 'a\r\nb\r\n', 'ü/ø.json': '{"k": "✓"}' }, 'unicode'],
  ];
  test('renderFiles gives the same bytes', () => {
    for (const [files, note] of samples) {
      expect(firstDiff(headFiles.renderFiles(files, note), baseFiles.renderFiles(files, note))).toBeNull();
    }
  });

  test('parseFiles reads every fixture answer and every rendered answer the same way', () => {
    const cases = JSON.parse(readFileSync(join(import.meta.dir, 'fixtures/files-cases.json'), 'utf8'));
    const answers: Array<[string, string[]]> = [];
    for (const group of Object.values(cases) as any[]) {
      for (const c of group) if (typeof c.answer === 'string') answers.push([c.answer, c.owns ?? []]);
    }
    for (const [files, note] of samples) answers.push([headFiles.renderFiles(files, note), Object.keys(files)]);
    answers.push(['<think>src/a.ts\n```ts\nbad\n```</think>Here is the fixed `src/money.ts`:\n```ts\nok\n```', ['src/money.ts']]);
    expect(answers.length).toBeGreaterThan(10);
    for (const [raw, owns] of answers) {
      expect(headFiles.parseFiles(raw, owns)).toEqual(baseFiles.parseFiles(raw, owns));
      expect(headFiles.parseFiles(raw)).toEqual(baseFiles.parseFiles(raw));
    }
  });
});

// ---- The real call path: runTicket on each side, with a recording model and a scripted sandbox ----

interface Call { kind: 'text' | 'complete'; model: string; messages: ChatMessage[]; maxTokens: number }

class Recorder implements Inference {
  calls: Call[] = [];
  constructor(private answer: (n: number) => string | Error, private reviewAnswer: object = { verdict: 'approve', problems: [], fix: '' }, private onText?: (messages: ChatMessage[]) => void) {}
  async text(model: string, messages: ChatMessage[], maxTokens: number): Promise<Completion<string>> {
    this.calls.push({ kind: 'text', model, messages: structuredClone(messages), maxTokens });
    this.onText?.(messages);
    const a = this.answer(this.calls.filter((c) => c.kind === 'text').length);
    if (a instanceof Error) throw a;
    return { value: a, raw: a, promptTokens: 100, completionTokens: 10, ms: 1 };
  }
  async complete<T>(model: string, messages: ChatMessage[], _schema: object, maxTokens: number): Promise<Completion<T>> {
    this.calls.push({ kind: 'complete', model, messages: structuredClone(messages), maxTokens });
    const raw = JSON.stringify(this.reviewAnswer);
    return { value: this.reviewAnswer as T, raw, promptTokens: 50, completionTokens: 5, ms: 1 };
  }
}

/** Runs the acceptance commands from a script: one entry per run, in order; past the end, everything passes. */
class Scripted implements Sandbox {
  readonly name = 'scripted';
  private n = 0;
  constructor(private script: Array<Partial<RunResult>>) {}
  async run(_command: string, _o: RunOptions): Promise<RunResult> {
    const r = this.script[this.n++] ?? {};
    return { code: 0, out: 'ok', ms: 1, timedOut: false, ...r };
  }
}

interface Scenario {
  why: string;
  ticket: Ticket;
  interfaces: string[];
  memory: string;
  answer: (n: number) => string | Error;
  script: Array<Partial<RunResult>>;
}

const files = (f: Record<string, string>, note = '') => headFiles.renderFiles(f, note);

const SCENARIOS: Scenario[] = [
  {
    why: 'write, long failing check, timed-out check, pass (attempts 1-3)',
    ticket: tk({ id: 'loop', owns: ['src/money.ts', 'src/new.ts'], reads: ['src/util/*.ts', 'big.txt'], acceptance: ['bun test', 'bunx tsc --noEmit'] }),
    interfaces: PLAN_INTERFACES, memory: 'Use cents.',
    answer: (n) => files({ 'src/money.ts': `export const v${n} = ${n};\n`, 'src/new.ts': `export const n = ${n};\n` }, `try ${n}`),
    script: [{ code: 1, out: LONG_LOG }, {}, { code: 124, out: LONG_LOG.slice(0, 5000), timedOut: true }],
  },
  {
    why: 'refused file, model error, escalation to a bigger model, ends failed (attempts 1-4)',
    ticket: tk({ id: 'esc', owns: ['src/new.ts'], budget: { attempts: 2, max_tokens: 60000 }, escalate_to: 'big', acceptance: ['make'] }),
    interfaces: [], memory: '',
    answer: (n) => (n === 1 ? files({ 'src/new.ts': 'a\n', 'evil.ts': 'x\n' }) : n === 2 ? new Error('inference 500: boom') : files({ 'src/new.ts': `b${n}\n` })),
    script: [{ code: 2, out: 'make: *** [all] Error 2' }, { code: 2, out: '' }],
  },
  {
    why: 'an answer with no files, then a fix that passes',
    ticket: tk({ id: 'nofiles', owns: ['src/empty.ts'], reads: ['docs/notes.md'], acceptance: ['bun test'] }),
    interfaces: ['src/api.h'], memory: '',
    answer: (n) => (n === 1 ? 'I cannot do this.' : files({ 'src/empty.ts': 'export {};\n' })),
    script: [{ code: 1, out: 'error: no tests' }],
  },
];

async function runBoth(s: Scenario, headCtx: { review?: boolean; thinking?: boolean } = {}) {
  const plan: Plan = { interfaces: s.interfaces, tickets: [s.ticket] };
  const tag = `${s.ticket.id}-${headCtx.review ? 'r' : ''}${headCtx.thinking ? 't' : ''}-${Math.random().toString(36).slice(2, 8)}`;
  const hm = new Recorder(s.answer), bm = new Recorder(s.answer);
  const hr = await headRunTicket(s.ticket, seed(`h-${tag}`), { plan, inference: hm, sandbox: new Scripted(s.script), emit: () => {}, memory: s.memory, review: false, ...headCtx });
  const br = await baseRunTicket(s.ticket, seed(`b-${tag}`), { plan, inference: bm, sandbox: new Scripted(s.script), emit: () => {}, memory: s.memory });
  return { hm, bm, hr, br };
}

describe('runTicket: the messages the harness actually sends, head vs base', () => {
  for (const s of SCENARIOS) {
    test(s.why, async () => {
      const { hm, bm, hr, br } = await runBoth(s);
      expect(hr.ok).toBe(br.ok);
      expect(hr.attempts).toBe(br.attempts);
      expect(hm.calls.length).toBe(bm.calls.length);
      expect(hm.calls.length).toBeGreaterThanOrEqual(2);
      for (let i = 0; i < bm.calls.length; i++) {
        const h = hm.calls[i]!, b = bm.calls[i]!;
        expect([h.kind, h.model, h.maxTokens]).toEqual([b.kind, b.model, b.maxTokens]);
        expectSameMessages(h.messages, b.messages, `${s.ticket.id} call ${i + 1}`);
      }
      // and each call is exactly what the base builders give for the inputs it had
      expect(hm.calls[0]!.messages[1]!.content).not.toContain('\n\n## Fix\n');
      for (const c of hm.calls.slice(1)) expect(c.messages[1]!.content).toContain('\n\n## Fix\nThis command failed:\n$ ');
    });
  }

  test('MUGGE_THINKING and the thinking switch only change states, not messages', async () => {
    const old = process.env.MUGGE_THINKING;
    try {
      for (const v of ['on', 'off']) {
        process.env.MUGGE_THINKING = v;
        for (const thinking of [undefined, true, false]) {
          const { hm, bm } = await runBoth(SCENARIOS[0]!, { thinking });
          expect(hm.calls.length).toBe(bm.calls.length);
          for (let i = 0; i < bm.calls.length; i++) expectSameMessages(hm.calls[i]!.messages, bm.calls[i]!.messages, `thinking ${v}/${thinking} call ${i + 1}`);
        }
      }
    } finally {
      if (old === undefined) delete process.env.MUGGE_THINKING;
      else process.env.MUGGE_THINKING = old;
    }
  });

  test('with the lead review on, worker calls never carry REVIEW_SYSTEM; the review is its own call', async () => {
    const { hm, bm } = await runBoth(SCENARIOS[0]!, { review: true });
    const text = hm.calls.filter((c) => c.kind === 'text');
    const review = hm.calls.filter((c) => c.kind === 'complete');
    expect(review.length).toBe(1);
    expect(review[0]!.messages[0]!.content).toBe(head.REVIEW_SYSTEM);
    expect(review[0]!.model).toBe('lead');
    for (const c of text) {
      expect(c.messages.length).toBe(2);
      expect(c.messages[0]!.content).toBe(base.SYSTEM);
      for (const m of c.messages) expect(m.content).not.toContain(head.REVIEW_SYSTEM.split('\n')[0]!);
    }
    // an approving review adds no worker call: the worker calls match base exactly
    expect(text.length).toBe(bm.calls.length);
    for (let i = 0; i < bm.calls.length; i++) expectSameMessages(text[i]!.messages, bm.calls[i]!.messages, `review-on call ${i + 1}`);
  });

  test("a review's fix call is a base-format fix message (command '(lead review)')", async () => {
    const s = SCENARIOS[0]!;
    const plan: Plan = { interfaces: s.interfaces, tickets: [s.ticket] };
    const wt = seed(`h-reviewfix-${Math.random().toString(36).slice(2, 8)}`);
    const rv = { verdict: 'problems', problems: ['format() drops the sign', 'add() is a stub'], fix: 'Handle negative cents.' };
    const expected: ChatMessage[][] = [];
    let n = 0;
    const m = new Recorder(s.answer, rv, (msgs) => {
      n++;
      // the base builders' answer for this worktree, as it stands when the call goes out
      const bp = base.buildPack(wt, plan, s.ticket, s.memory);
      if (n === 4) expected.push(base.fixMessages(wt, bp, { command: '(lead review)', output: reviewFeedback(rv) }));
      else expected.push(msgs as ChatMessage[]);
    });
    await headRunTicket(s.ticket, wt, { plan, inference: m, sandbox: new Scripted(s.script), emit: () => {}, memory: s.memory, review: true });
    const text = m.calls.filter((c) => c.kind === 'text');
    expect(text.length).toBe(4); // write, fix, fix, then the review's fix
    expectSameMessages(text[3]!.messages, expected[3]!, 'review fix');
    expect(text[3]!.messages[1]!.content).toContain('\n\n## Fix\nThis command failed:\n$ (lead review)\nThe lead reviewed your passing files and found problems:\n- format() drops the sign');
  });
});

// ---- The request on the wire: settings change other fields, never messages ----

describe('request body: MUGGE_THINKING / MUGGE_MAX_CALL_TOKENS / sampling leave messages alone', () => {
  const bodies: string[] = [];
  let server: ReturnType<typeof Bun.serve>;
  beforeAll(() => {
    server = Bun.serve({
      port: 0,
      async fetch(req) {
        bodies.push(await req.text());
        return Response.json({ choices: [{ message: { content: 'src/new.ts\n```ts\nx\n```' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 1 } });
      },
    });
  });
  afterAll(() => server.stop(true));

  const ENVS: Array<Record<string, string>> = [
    {},
    { MUGGE_THINKING: 'on' },
    { MUGGE_THINKING: 'off' },
    { MUGGE_MAX_CALL_TOKENS: '100' },
    { MUGGE_THINKING: 'on', MUGGE_MAX_CALL_TOKENS: '7', MUGGE_TOP_K: '20', MUGGE_TEMPERATURE: '0.7', MUGGE_PRESET: 'minicpm5-2b' },
  ];

  test('every setting sends the base builders\' messages, serialized the same as base sends them', async () => {
    const c = TICKETS[1]!;
    const wt = seed('wire');
    const plan: Plan = { interfaces: c.interfaces, tickets: [c.ticket] };
    const bp = base.buildPack(wt, plan, c.ticket, c.memory);
    const want = [base.writeMessages(wt, bp), base.fixMessages(wt, bp, { command: 'bun test', output: baseTail(LONG_LOG) })];
    const url = `http://127.0.0.1:${server.port}/v1`;

    // base: its own client, as the base engine sent them
    const baseBodies: string[] = [];
    for (const msgs of want) {
      bodies.length = 0;
      await new BaseOpenAICompatible({ baseUrl: url, defaultModel: 'm' }).text('coder', msgs, 5000);
      baseBodies.push(bodies[0]!);
    }
    const slice = (body: string) => {
      const a = body.indexOf('"messages":');
      return body.slice(a, a + '"messages":'.length + JSON.stringify(JSON.parse(body).messages).length);
    };

    for (const env of ENVS) {
      const full = { MUGGE_INFERENCE_URL: url, MUGGE_MODEL: 'm', ...env };
      const pool = BackendPool.fromMachines(machinesFromEnv(full), inferenceFromEnv(full), full);
      const worker = pool.forAgent({ role: 'worker', agent: 'w1' });
      for (let i = 0; i < want.length; i++) {
        bodies.length = 0;
        const headMsgs = i === 0 ? head.writeMessages(wt, head.buildPack(wt, plan, c.ticket, c.memory)) : head.fixMessages(wt, head.buildPack(wt, plan, c.ticket, c.memory), { command: 'bun test', output: headTail(LONG_LOG) });
        await worker.text('coder', headMsgs, 5000);
        const sent = JSON.parse(bodies[0]!);
        expectSameMessages(sent.messages, want[i]!, `env ${JSON.stringify(env)} ${i ? 'fix' : 'write'}`);
        expect(firstDiff(slice(bodies[0]!), slice(baseBodies[i]!))).toBeNull();
        expect(sent.max_tokens).toBe(env.MUGGE_MAX_CALL_TOKENS ? Number(env.MUGGE_MAX_CALL_TOKENS) : 5000);
        if (env.MUGGE_THINKING) expect(sent.chat_template_kwargs).toEqual({ enable_thinking: env.MUGGE_THINKING === 'on' });
        else expect(sent.chat_template_kwargs).toBeUndefined();
      }
    }
  });
});

// ---- Prefix stability (KV prefix caching): write vs the fix call that follows it ----

describe('prefix stability: a fix message starts with the write message', () => {
  /** Write call and the first fix call of a real loop, with how much of the write message the fix repeats. */
  async function writeThenFix(ticket: Ticket, answer: string, interfaces: string[] = PLAN_INTERFACES) {
    const plan: Plan = { interfaces, tickets: [ticket] };
    const m = new Recorder(() => answer);
    await headRunTicket(ticket, seed(`p-${ticket.id}-${Math.random().toString(36).slice(2, 8)}`), { plan, inference: m, sandbox: new Scripted([{ code: 1, out: 'FAIL money.test.ts\nexpected 3 got 4' }]), emit: () => {}, memory: 'Use cents.', review: false });
    const w = m.calls[0]!.messages, f = m.calls[1]!.messages;
    const W = w[1]!.content, F = f[1]!.content;
    const header = W.indexOf(`\n${OWNED_HEADER}\n`) + 1;
    const shared = commonPrefix(W, F);
    return { w, f, W, F, header, shared };
  }

  test('the system prompt is the same on write and fix', async () => {
    const { w, f } = await writeThenFix(tk({ id: 'sys', owns: ['src/new.ts'] }), files({ 'src/new.ts': 'x\n' }));
    expect(f[0]!.content).toBe(w[0]!.content);
  });

  test('owned file missing before the write: the whole write message is a prefix of the fix message', async () => {
    const r = await writeThenFix(tk({ id: 'pnew', owns: ['src/new.ts'], reads: ['docs/notes.md'] }), files({ 'src/new.ts': 'export const n = 1;\n' }));
    expect(r.header).toBeGreaterThan(0);
    expect(r.F.slice(0, r.header)).toBe(r.W.slice(0, r.header)); // everything before "## Your files as they are now"
    expect(r.W.endsWith(`${OWNED_HEADER}\n--- src/new.ts\n`)).toBe(true);
    expect(r.F.startsWith(r.W)).toBe(true);
    expect(r.shared).toBe(r.W.length);
    expect(r.F.slice(r.W.length).startsWith('export const n = 1;\n')).toBe(true);
    console.log(`[prefix] no owned file yet: write user message ${r.W.length} chars, "${OWNED_HEADER}" at ${r.header}, shared prefix ${r.shared} (= the whole write message); fix continues with the new file content`);
  });

  test('owned file existing and rewritten: the prefix runs through the header into the old file, up to its first changed byte', async () => {
    const r = await writeThenFix(tk({ id: 'pold', owns: ['src/money.ts'], reads: ['docs/notes.md'] }), files({ 'src/money.ts': "import type { Money } from './types.ts';\n\nexport const one: Money = { cents: 100 };\n" }));
    expect(r.F.slice(0, r.header)).toBe(r.W.slice(0, r.header));
    const firstFile = r.header + `${OWNED_HEADER}\n--- src/money.ts\n`.length;
    expect(r.shared).toBeGreaterThanOrEqual(firstFile);
    // the old and new contents share "import type { Money } from './types.ts';\n\n"
    expect(r.shared).toBe(firstFile + "import type { Money } from './types.ts';\n\n".length);
    console.log(`[prefix] owned file rewritten: write user message ${r.W.length} chars, "${OWNED_HEADER}" at ${r.header}, owned content starts at ${firstFile}, shared prefix ${r.shared}; diverges at ${JSON.stringify(r.W.slice(r.shared, r.shared + 20))} (write) vs ${JSON.stringify(r.F.slice(r.shared, r.shared + 20))} (fix)`);
  });

  test('several owned files: an unchanged first file stays in the prefix, the change in the second ends it', async () => {
    const r = await writeThenFix(
      tk({ id: 'pmulti', owns: ['src/util/a.ts', 'src/new.ts'], reads: ['src/util/*.ts'] }),
      files({ 'src/util/a.ts': 'export const a = 1;\n', 'src/new.ts': 'export const n = 2;\n' }),
    );
    expect(r.F.slice(0, r.header)).toBe(r.W.slice(0, r.header));
    expect(r.W.endsWith('--- src/util/a.ts\nexport const a = 1;\n\n--- src/new.ts\n')).toBe(true);
    expect(r.shared).toBe(r.W.length);
    console.log(`[prefix] two owned files (first unchanged, second new): write ${r.W.length} chars, header at ${r.header}, shared ${r.shared} (= the whole write message)`);
  });

  test('without any change to the worktree the fix message is exactly write + "\\n\\n## Fix\\n…"', () => {
    const c = TICKETS[1]!;
    const wt = seed('p-same');
    const pack = head.buildPack(wt, { interfaces: c.interfaces, tickets: [c.ticket] }, c.ticket, c.memory);
    const W = head.writeMessages(wt, pack)[1]!.content;
    const F = head.fixMessages(wt, pack, { command: 'bun test', output: '1 fail' })[1]!.content;
    expect(F).toBe(`${W}\n\n## Fix\nThis command failed:\n$ bun test\n1 fail\n\nReturn your files fixed so it passes.`);
  });
});
