import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyEvent, emptyView, agentStates, type EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { gitOk, initRepo } from '../src/engine/git.ts';
import { runTicket } from '../src/engine/harness.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { REVIEW_JSON_SCHEMA, REVIEW_SYSTEM, reviewMessages } from '../src/engine/prompts.ts';
import { diffOwned, normalizeReview, REVIEW_MAX_TOKENS, reviewAnswer, reviewCapFromEnv, reviewFeedback, reviewFromEnv, reviewThinkingFromEnv, snapshot } from '../src/engine/review.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
import { BackendPool, resolveMachine } from '../src/engine/backends.ts';
import { canMove, type AgentState } from '../src/engine/state.ts';
import { validatePlan, type WorkerOutput } from '../src/tickets/schema.ts';

const t = (id: string, extra: any = {}) => ({ id, title: `Write ${id}`, image: 'node', owns: [`${id}.txt`], acceptance: [`grep -q ok-${id} ${id}.txt`], ...extra });
const ok = (id: string, extra = ''): WorkerOutput => ({ files: { [`${id}.txt`]: `ok-${id}\n${extra}` }, note: `wrote ${id}` });

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), 'mugge-loop-'));
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// these tests follow the loop's states with thinking off (writing / fixing); unset is on, which
// the last test checks
const thinkingWas = process.env.MUGGE_THINKING;
beforeAll(() => {
  process.env.MUGGE_THINKING = 'off';
});
afterAll(() => {
  if (thinkingWas === undefined) delete process.env.MUGGE_THINKING;
  else process.env.MUGGE_THINKING = thinkingWas;
});

/** Each ticket's states in order, from the events. */
function statesOf(events: EngineEvent[]): Map<string, AgentState[]> {
  const out = new Map<string, AgentState[]>();
  for (const e of events) if (e.type === 'state') out.set(e.ticket, [...(out.get(e.ticket) ?? []), e.state]);
  return out;
}

function expectLegal(events: EngineEvent[]) {
  for (const [id, seq] of statesOf(events)) {
    seq.forEach((s, i) => {
      if (!canMove(seq[i - 1], s)) throw new Error(`${id}: ${seq[i - 1]} -> ${s} in ${seq.join(' > ')}`);
    });
  }
}

async function repoWith(name: string) {
  const repo = join(dir, name);
  mkdirSync(repo, { recursive: true });
  writeFileSync(join(repo, 'README.md'), 'scaffold\n');
  return { repo, base: initRepo(repo) };
}

describe('ticket states', () => {
  test('every ticket moves through legal states; agents are never shared', async () => {
    const { repo, base } = await repoWith('states');
    const plan = validatePlan({ tickets: [t('a'), t('b'), t('c', { depends_on: ['a'] }), t('d'), t('x', { budget: { attempts: 2 } })] });
    const fake = startFakeModel({ solutions: { a: ok('a'), b: ok('b'), c: ok('c'), d: ok('d') }, flaky: ['b'], broken: ['x'], latencyMs: 30 });
    const events: EngineEvent[] = [];
    try {
      await runPlan({ repo, base, plan, concurrency: 2, worktreeRoot: join(dir, 'wt-states'), sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), emit: (e) => events.push(e), integrate: false });
    } finally {
      fake.stop();
    }
    expectLegal(events);
    const seq = statesOf(events);
    expect(seq.get('b')!.filter((s) => s !== 'queued' && s !== 'waiting')).toEqual(['writing', 'checking', 'fixing', 'checking', 'reviewing', 'finished']);
    expect(seq.get('c')![0]).toBe('waiting');
    expect(seq.get('x')!.at(-1)).toBe('failed');
    // five tickets, two agents: someone waited for an agent; one machine with two slots: no call queued
    expect(events.some((e) => e.type === 'state' && e.state === 'waiting' && e.reason === 'waiting for a free agent')).toBe(true);
    expect([...seq.values()].some((s) => s.includes('queued'))).toBe(false);
    const fix = events.find((e) => e.type === 'state' && e.ticket === 'b' && e.state === 'fixing');
    expect(fix).toMatchObject({ attempt: 2, of: 4 });
    // replaying the events, an agent never works two tickets at once
    const view = emptyView();
    for (const e of events) {
      applyEvent(view, e);
      const busy = agentStates(view).filter((s) => s.state !== 'idle').map((s) => s.ticket);
      expect(new Set(busy).size).toBe(busy.length);
      for (const s of view.tickets) if (s.agent !== undefined) expect(s.agent).toBeLessThan(2);
    }
    expect(agentStates(view).every((s) => s.state === 'idle')).toBe(true);
  });

  test('a call waiting for a model slot shows queued, then goes back to its call', async () => {
    const { repo, base } = await repoWith('queued');
    const plan = validatePlan({ tickets: [t('a'), t('b'), t('c')] });
    const fake = startFakeModel({ solutions: { a: ok('a'), b: ok('b'), c: ok('c') }, flaky: ['b'], latencyMs: 40 });
    const events: EngineEvent[] = [];
    try {
      const pool = BackendPool.single(new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), 1);
      const report = await runPlan({ repo, base, plan, concurrency: 3, pool, worktreeRoot: join(dir, 'wt-queued'), sandbox: new LocalSandbox(), emit: (e) => events.push(e), integrate: false });
      expect(report.summary.done).toBe(3);
    } finally {
      fake.stop();
    }
    expectLegal(events);
    const seq = statesOf(events);
    const queuedOnes = [...seq.entries()].filter(([, s]) => s.includes('queued'));
    expect(queuedOnes.length).toBeGreaterThan(0);
    for (const [, s] of queuedOnes) {
      const i = s.indexOf('queued');
      expect(['writing', 'fixing', 'reviewing']).toContain(s[i + 1]!);
      expect(s[i + 1]).toBe(s[i - 1]!);
    }
    const q = events.find((e) => e.type === 'state' && e.state === 'queued');
    expect(q).toMatchObject({ agent: expect.any(Number) });
  });

  test('with MUGGE_REVIEW on, the review goes to the machine that serves leads', async () => {
    const { repo, base } = await repoWith('lead');
    const plan = validatePlan({ tickets: [t('a')] });
    const laptop = startFakeModel({ solutions: { a: ok('a') } });
    const mac = startFakeModel({ solutions: {} });
    try {
      const at = (url: string, name: string, roles: Array<'worker' | 'lead' | 'manager'>) =>
        ({ machine: resolveMachine({ name, url, slots: 2, roles }), client: new OpenAICompatible({ baseUrl: url, defaultModel: 'fake' }) });
      const pool = new BackendPool([at(laptop.url, 'laptop', ['worker']), at(mac.url, 'mac', ['manager', 'lead'])]);
      const old = process.env.MUGGE_REVIEW;
      process.env.MUGGE_REVIEW = 'on';
      try {
        const report = await runPlan({ repo, base, plan, concurrency: 1, pool, worktreeRoot: join(dir, 'wt-lead'), sandbox: new LocalSandbox(), integrate: false });
        expect(report.results.get('a')!.review?.verdict).toBe('approve');
      } finally {
        if (old === undefined) delete process.env.MUGGE_REVIEW;
        else process.env.MUGGE_REVIEW = old;
      }
      expect(laptop.calls().map((c) => c.kind)).toEqual(['write']);
      expect(mac.calls().map((c) => c.kind)).toEqual(['review']);
    } finally {
      laptop.stop();
      mac.stop();
    }
  });

  test('a clash the planner settled runs the later ticket after the owner, on its version of the file', async () => {
    const { repo, base } = await repoWith('settled');
    const plan = validatePlan({ tickets: [t('a', { owns: ['shared.txt'], acceptance: ['grep -q ok-a shared.txt'] }), t('b', { owns: ['shared.txt'], acceptance: ['grep -q ok-b shared.txt'] })] });
    const fake = startFakeModel({ solutions: { a: { files: { 'shared.txt': 'ok-a\n' }, note: '' }, b: { files: { 'shared.txt': 'ok-a\nok-b\n' }, note: '' } } });
    const events: EngineEvent[] = [];
    try {
      const report = await runPlan({ repo, base, plan, concurrency: 4, worktreeRoot: join(dir, 'wt-settled'), sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), emit: (e) => events.push(e) });
      expect(report.summary).toMatchObject({ done: 2, blocked: 0, integrated: true });
      // b's write call saw a's file as it is now
      expect(fake.calls().find((c) => c.ticket === 'b')!.prompt).toContain('--- shared.txt\nok-a');
    } finally {
      fake.stop();
    }
    expectLegal(events);
    const view = events.reduce(applyEvent, emptyView());
    expect(view.settled).toEqual([{ ticket: 'b', files: ['shared.txt'], with: ['a'] }]);
    expect(view.clashes ?? []).toEqual([]);
    const waits = events.filter((e) => e.type === 'state' && e.ticket === 'b' && e.state === 'waiting');
    expect(waits[0]).toMatchObject({ reason: 'runs after a: they share shared.txt (settled by the planner)', waitingOn: ['a'] });
  });

  test('a settled clash only orders the tickets: when the owner fails, the later ticket still runs on base', async () => {
    const { repo, base } = await repoWith('settled-fail');
    const plan = validatePlan({ tickets: [t('a', { owns: ['shared.txt'], acceptance: ['grep -q ok-a shared.txt'] }), t('b', { owns: ['shared.txt'], acceptance: ['grep -q ok-b shared.txt'] }), t('c', { depends_on: ['a'] })] });
    const fake = startFakeModel({ solutions: { a: { files: { 'shared.txt': 'ok-a\n' }, note: '' }, b: { files: { 'shared.txt': 'ok-b\n' }, note: '' }, c: ok('c') }, broken: ['a'] });
    const events: EngineEvent[] = [];
    try {
      const report = await runPlan({ repo, base, plan, concurrency: 4, worktreeRoot: join(dir, 'wt-settled-fail'), sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), emit: (e) => events.push(e), integrate: false });
      expect(Object.fromEntries(report.states)).toEqual({ a: 'failed', b: 'done', c: 'skipped' });
      // b started only after a ended, and on base's version of the file
      const order = events.filter((e) => e.type === 'start' || e.type === 'finish').map((e) => `${e.type}:${e.ticket}`);
      expect(order.indexOf('start:b')).toBeGreaterThan(order.indexOf('finish:a'));
      expect(fake.calls().find((c) => c.ticket === 'b')!.prompt).not.toContain('ok-a');
    } finally {
      fake.stop();
    }
    expectLegal(events);
  });

  test('with clashes: block, a file-ownership clash blocks the later ticket for the owner and is recorded', async () => {
    const { repo, base } = await repoWith('clash');
    const plan = validatePlan({ tickets: [t('a', { owns: ['shared.txt'], acceptance: ['grep -q ok-a shared.txt'] }), t('b', { owns: ['shared.txt'] }), t('c', { depends_on: ['b'] }), t('d')] }, { clashes: 'block' });
    const fake = startFakeModel({ solutions: { a: { files: { 'shared.txt': 'ok-a\n' }, note: '' }, d: ok('d') } });
    const events: EngineEvent[] = [];
    try {
      const report = await runPlan({ repo, base, plan, concurrency: 4, worktreeRoot: join(dir, 'wt-clash'), sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), emit: (e) => events.push(e), integrate: false });
      expect(report.states.get('a')).toBe('done');
      expect(report.states.get('b')).toBe('blocked');
      expect(report.states.get('c')).toBe('skipped');
      expect(report.states.get('d')).toBe('done');
      expect(report.summary.blocked).toBe(1);
      expect(report.clashes).toEqual([{ ticket: 'b', files: ['shared.txt'], with: ['a'] }]);
      // b never reached the model
      expect(fake.calls().some((c) => c.ticket === 'b')).toBe(false);
    } finally {
      fake.stop();
    }
    expectLegal(events);
    const view = events.reduce(applyEvent, emptyView());
    expect(view.clashes).toEqual([{ ticket: 'b', files: ['shared.txt'], with: ['a'] }]);
    const b = view.tickets.find((x) => x.id === 'b')!;
    expect(b).toMatchObject({ state: 'blocked', phase: 'blocked', reason: 'two tickets need the same file: shared.txt (also owned by a)' });
    expect(view.tickets.find((x) => x.id === 'c')).toMatchObject({ state: 'skipped', phase: 'blocked', waitingOn: ['b'] });
    expect(view.summary?.blocked).toBe(1);
  });
});

describe('scheduler agents', () => {
  test('a ticket whose worktree cannot be made frees its agent number', async () => {
    const { repo, base } = await repoWith('no-worktree');
    // the repo has mugge/a checked out, so a worktree for ticket a cannot take that branch
    gitOk(repo, ['checkout', '-q', '-b', 'mugge/a']);
    const plan = validatePlan({ tickets: [t('a'), t('b'), t('c')] });
    const fake = startFakeModel({ solutions: { b: ok('b'), c: ok('c') } });
    const events: EngineEvent[] = [];
    try {
      const report = await runPlan({ repo, base, plan, concurrency: 1, worktreeRoot: join(dir, 'wt-no-worktree'), sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), emit: (e) => events.push(e), integrate: false });
      expect(report.summary).toMatchObject({ done: 2, failed: 1 });
    } finally {
      fake.stop();
    }
    const agents = events.filter((e) => e.type === 'state' && e.ticket !== 'a').map((e) => (e as any).agent).filter((x) => x !== undefined);
    expect(agents.length).toBeGreaterThan(0);
    expect(agents.every((x) => x === 0)).toBe(true);
    expectLegal(events);
  });
});

describe('runTicket', () => {
  const plan = validatePlan({ tickets: [t('a')] });
  const ticket = plan.tickets[0]!;
  const folder = (name: string) => {
    const p = join(dir, name);
    mkdirSync(p, { recursive: true });
    return p;
  };

  test('thinking on shows every call as thinking', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') }, flaky: ['a'] });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('think'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', extra: { chat_template_kwargs: { enable_thinking: true } } }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: false });
      expect(r.ok).toBe(true);
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')).toEqual(['thinking', 'checking', 'thinking', 'checking', 'finished']);
    expectLegal(events);
  });

  test('review is on unless switched off', async () => {
    const old = process.env.MUGGE_REVIEW;
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    try {
      delete process.env.MUGGE_REVIEW;
      const on = await runTicket(ticket, folder('default-on'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: () => {} });
      expect(on.ok).toBe(true);
      expect(on.review?.verdict).toBe('approve');
      expect(fake.calls().map((c) => c.kind)).toEqual(['write', 'review']);
      process.env.MUGGE_REVIEW = 'off';
      const off = await runTicket(ticket, folder('off'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: () => {} });
      expect(off.ok).toBe(true);
      expect(off.review).toBeNull();
      expect(fake.calls().map((c) => c.kind)).toEqual(['write', 'review', 'write']);
    } finally {
      if (old === undefined) delete process.env.MUGGE_REVIEW;
      else process.env.MUGGE_REVIEW = old;
      fake.stop();
    }
    expect(reviewFromEnv({})).toBe(true);
    expect(reviewFromEnv({ MUGGE_REVIEW: 'on' })).toBe(true);
    expect(reviewFromEnv({ MUGGE_REVIEW: ' OFF ' })).toBe(false);
    expect(() => reviewFromEnv({ MUGGE_REVIEW: 'yes' })).toThrow(/on or off/);
  });

  test('an approving review finishes the ticket; the review sees the diff and the checks', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('approve'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 1, firstTry: true, reviewUnresolved: false });
      expect(r.review?.verdict).toBe('approve');
      const calls = fake.calls();
      expect(calls.map((c) => c.kind)).toEqual(['write', 'review']);
      expect(calls[1]!.maxTokens).toBe(REVIEW_MAX_TOKENS);
      expect(calls[1]!.prompt).toContain('+++ b/a.txt\n@@ -0,0 +1 @@\n+ok-a');
      expect(calls[1]!.prompt).toContain('## Checks (all passed)\n$ grep -q ok-a a.txt');
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')).toEqual(['writing', 'checking', 'reviewing', 'finished']);
    expect(events.find((e) => e.type === 'review')).toMatchObject({ verdict: 'approve', attempt: 1 });
  });

  test('review problems spend one fix attempt, sent as an ordinary fix call', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['empty input is not handled'], fix: 'handle the empty case' }] },
      fixes: { a: [ok('a', 'empty case\n')] },
    });
    const events: EngineEvent[] = [];
    const wt = folder('problems');
    try {
      const r = await runTicket(ticket, wt, { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 2, firstTry: false, reviewUnresolved: false });
      const calls = fake.calls();
      expect(calls.map((c) => c.kind)).toEqual(['write', 'review', 'fix']); // one review per ticket
      expect(calls[2]!.prompt).toContain('## Fix\nThis command failed:\n$ (lead review)\nThe lead reviewed your passing files and sent them back:\n- empty input is not handled; handle the empty case\n\nReply with your files, fixed so it passes, in the same format.');
      expect(readFileSync(join(wt, 'a.txt'), 'utf8')).toBe('ok-a\nempty case\n');
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')).toEqual(['writing', 'checking', 'reviewing', 'fixing', 'checking', 'finished']);
    expectLegal(events);
  });

  test('the review is part of done: a ticket whose review fix never passes fails', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['wrong'], fix: 'rewrite it' }] },
      fixes: { a: [{ files: { 'a.txt': 'broken\n' }, note: '' }] },
    });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('unresolved'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: false, attempts: 4, reviewUnresolved: true, answerAttempt: null });
      expect(fake.calls().map((c) => c.kind)).toEqual(['write', 'review', 'fix', 'fix', 'fix']); // one review per ticket
    } finally {
      fake.stop();
    }
    expect(events.filter((e) => e.type === 'state').at(-1)).toMatchObject({ state: 'failed', reason: 'lead review problems left unfixed' });
    expectLegal(events);
  });

  test('a review fix whose calls are refused or fail ends failed, through legal states', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['wrong'], fix: 'rewrite it' }] },
      // every fix also writes a file the ticket does not own: refused, so the last state is a call state
      fixes: { a: [{ files: { 'a.txt': 'ok-a\n', 'b.txt': 'x\n' }, note: '' }] },
    });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('refused-fix'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: false, attempts: 4, reviewUnresolved: true });
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')!.slice(-2)).toEqual(['fixing', 'failed']);
    expectLegal(events);
  });

  test('a review fix call that errors is sent again with the review, so the ticket never finishes unreviewed', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['empty input is not handled'], fix: 'handle the empty case' }] },
      fixes: { a: [ok('a', 'empty case\n')] },
    });
    const real = new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' });
    let errors = 0;
    // the first fix call fails in transit (a 500, a dropped connection)
    const inference = {
      complete: real.complete.bind(real),
      text: (model: string, messages: any[], maxTokens: number, signal?: AbortSignal) => {
        if (/\n## Fix\n/.test(messages.at(-1).content) && errors++ === 0) return Promise.reject(new Error('HTTP 500 from the machine'));
        return real.text(model, messages, maxTokens, signal);
      },
    } as unknown as OpenAICompatible;
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('review-fix-500'), { plan, inference, sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 3, reviewUnresolved: false });
      const calls = fake.calls();
      expect(calls.map((c) => c.kind)).toEqual(['write', 'review', 'fix']);
      expect(calls[2]!.prompt).toContain('$ (lead review)\nThe lead reviewed your passing files and sent them back:\n- empty input is not handled');
    } finally {
      fake.stop();
    }
    expectLegal(events);
  });

  test('a run stopped during the lead review ends aborted, not as a failed review', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    const stop = new AbortController();
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('review-stop'), {
        plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), review: true, signal: stop.signal,
        emit: (e) => {
          events.push(e);
          if (e.type === 'state' && e.state === 'reviewing') stop.abort();
        },
      });
      expect(r).toMatchObject({ ok: false, error: 'aborted' });
    } finally {
      fake.stop();
    }
    expect(events.filter((e) => e.type === 'state').at(-1)).toMatchObject({ state: 'failed', reason: 'aborted' });
    expectLegal(events);
  });

  test('review problems with no attempt left escalate like a failing check', async () => {
    const one = validatePlan({ tickets: [t('a', { budget: { attempts: 1 }, escalate_to: 'big' })] });
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['p'], fix: 'f' }] },
      fixes: { a: [ok('a', 'fixed\n')] },
    });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(one.tickets[0]!, folder('escalate'), { plan: one, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 2, model: 'big', reviewUnresolved: false, answerAttempt: 2 });
      expect(fake.calls().map((c) => c.kind)).toEqual(['write', 'review', 'fix']);
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')).toEqual(['writing', 'checking', 'reviewing', 'fixing', 'checking', 'finished']);
    expectLegal(events);
  });

  test('a review answer cut at the cap counts its tokens and shows cut; a review that fails twice fails the ticket', async () => {
    const server = Bun.serve({
      port: 0,
      fetch: async (req) => {
        const body: any = await req.json();
        const review = body.messages[0].content === REVIEW_SYSTEM;
        const content = review ? '{"verdict": "probl' : '```a.txt\nok-a\n```';
        return Response.json({ choices: [{ finish_reason: review ? 'length' : 'stop', message: { content } }], usage: { prompt_tokens: 50, completion_tokens: review ? 300 : 5 } });
      },
    });
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, folder('review-cut'), { plan, inference: new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', maxCallTokens: 300 }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: false, completionTokens: 605 });
      expect(r.error).toStartWith('lead review failed: ');
      expect(r.review).toMatchObject({ verdict: 'error', cut: true, completionTokens: 300 });
    } finally {
      server.stop(true);
    }
    expect(events.filter((e) => e.type === 'review')).toMatchObject([{ verdict: 'error', cut: true, completionTokens: 300 }, { verdict: 'error', cut: true }]);
    expect(events.filter((e) => e.type === 'state').at(-1)).toMatchObject({ state: 'failed', reason: 'lead review failed' });
    expectLegal(events);
  });

  test('an event sink that throws is not a failed model call', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    try {
      const run = runTicket(ticket, folder('sink-throws'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), review: false, emit: (e) => {
        if (e.type === 'model') throw new Error('calls.jsonl: ENOSPC');
      } });
      await expect(run).rejects.toThrow('ENOSPC');
      // one call, not a good answer thrown away and retried as "(model call) failed"
      expect(fake.calls().map((c) => c.kind)).toEqual(['write']);
    } finally {
      fake.stop();
    }
  });

  test('a review on the last attempt cannot fix: the ticket fails', async () => {
    const one = validatePlan({ tickets: [t('a', { budget: { attempts: 1 } })] });
    const fake = startFakeModel({ solutions: { a: ok('a') }, reviews: { a: [{ verdict: 'problems', problems: ['p'], fix: 'f' }] } });
    try {
      const r = await runTicket(one.tickets[0]!, folder('last'), { plan: one, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: () => {}, review: true });
      expect(r).toMatchObject({ ok: false, attempts: 1, reviewUnresolved: true });
    } finally {
      fake.stop();
    }
    expect(normalizeReview({ verdict: 'maybe' })).toMatch(/not approve or send_back/);
    expect(normalizeReview({ verdict: 'problems', problems: [' a ', 3, ''], fix: 7 })).toEqual({ verdict: 'problems', problems: ['a'], fix: '' });
  });

  test('the per-call cap applies to the review call', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    try {
      await runTicket(ticket, folder('cap'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', maxCallTokens: 300 }), sandbox: new LocalSandbox(), emit: () => {}, review: true });
      expect(fake.calls().map((c) => [c.kind, c.maxTokens])).toEqual([['write', 300], ['review', 300]]);
    } finally {
      fake.stop();
    }
  });

  test('the review may write as much as MUGGE_MAX_CALL_TOKENS allows, never more than the ticket has left', async () => {
    const old = process.env.MUGGE_MAX_CALL_TOKENS;
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    const inference = () => new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', maxCallTokens: 8192 });
    try {
      process.env.MUGGE_MAX_CALL_TOKENS = '8192';
      await runTicket(ticket, folder('cap-env'), { plan, inference: inference(), sandbox: new LocalSandbox(), emit: () => {}, review: true });
      expect(fake.calls().at(-1)).toMatchObject({ kind: 'review', maxTokens: 8192 });
      // a ticket with less left than the cap: the review gets what is left
      const small = validatePlan({ tickets: [{ ...ticket, budget: { attempts: 4, max_tokens: 5000 } }] });
      const r = await runTicket(small.tickets[0]!, folder('cap-left'), { plan: small, inference: inference(), sandbox: new LocalSandbox(), emit: () => {}, review: true });
      expect(fake.calls().at(-1)).toMatchObject({ kind: 'review', maxTokens: 5000 - (r.completionTokens - r.review!.completionTokens) });
      // the caller's own cap wins over the environment
      await runTicket(ticket, folder('cap-ctx'), { plan, inference: inference(), sandbox: new LocalSandbox(), emit: () => {}, review: true, reviewMaxTokens: 5000 });
      expect(fake.calls().at(-1)).toMatchObject({ kind: 'review', maxTokens: 5000 });
    } finally {
      if (old === undefined) delete process.env.MUGGE_MAX_CALL_TOKENS;
      else process.env.MUGGE_MAX_CALL_TOKENS = old;
      fake.stop();
    }
    expect(reviewCapFromEnv({})).toBe(REVIEW_MAX_TOKENS);
    expect(reviewCapFromEnv({ MUGGE_MAX_CALL_TOKENS: '16384' })).toBe(16384);
    expect(reviewCapFromEnv({ MUGGE_MAX_CALL_TOKENS: '512' })).toBe(512);
    expect(() => reviewCapFromEnv({ MUGGE_MAX_CALL_TOKENS: 'lots' })).toThrow(/positive integer/);
  });
});

test('diffOwned diffs the owned files from their start, in any folder', () => {
  const wt = join(dir, 'diff');
  mkdirSync(join(wt, 'src'), { recursive: true });
  writeFileSync(join(wt, 'src/m.ts'), 'x\ny\n');
  const before = snapshot(wt, ['src/m.ts', 'new.py', 'gone.txt']);
  expect(diffOwned(wt, before)).toBe('');
  writeFileSync(join(wt, 'src/m.ts'), 'x\nz\n');
  writeFileSync(join(wt, 'new.py'), 'print(1)\n');
  const d = diffOwned(wt, before);
  expect(d).toContain('--- a/src/m.ts\n+++ b/src/m.ts\n@@ -1,2 +1,2 @@\n x\n-y\n+z');
  expect(d).toContain('--- /dev/null\n+++ b/new.py');
});

test('review prompt shape', () => {
  const plan = validatePlan({ tickets: [t('a')] });
  const m = reviewMessages(plan.tickets[0]!, '', [{ command: 'make test', output: 'ok' }]);
  expect(m[0]).toEqual({ role: 'system', content: REVIEW_SYSTEM });
  expect(m[1]!.content).toBe('Ticket: a\nTitle: Write a\nRole: implement\n\n\n\nFiles it owns: a.txt\nDone when these commands pass: grep -q ok-a a.txt\n\n## Diff\n(no changes)\n\n## Checks (all passed)\n$ make test\nok');
  expect(reviewFeedback({ problems: ['x'], fix: '' })).toBe('The lead reviewed your passing files and sent them back:\n- x');
});

test('review answer shape: approve or send_back plus one reason; the older shape still reads', () => {
  expect(normalizeReview({ verdict: 'approve', reason: '' })).toEqual({ verdict: 'approve', problems: [], fix: '' });
  expect(normalizeReview({ verdict: 'send_back', reason: ' format() drops the sign ' })).toEqual({ verdict: 'problems', problems: ['format() drops the sign'], fix: '' });
  // a send-back with no reason still sends it back, with something to act on
  expect(normalizeReview({ verdict: 'send_back', reason: '' })).toMatchObject({ verdict: 'problems', problems: [expect.stringContaining('without a reason')] });
  expect(normalizeReview({ verdict: 'problems', problems: ['p'], fix: 'f' })).toEqual({ verdict: 'problems', problems: ['p'], fix: 'f' });
  expect(reviewAnswer({ verdict: 'problems', problems: ['p'], fix: 'f' })).toEqual({ verdict: 'send_back', reason: 'p; f' });
  expect(REVIEW_JSON_SCHEMA.required).toEqual(['verdict', 'reason']);
});

test('the review call goes out with thinking off unless MUGGE_REVIEW_THINKING=on; worker calls think by default', async () => {
  const was = process.env.MUGGE_THINKING;
  const plan = validatePlan({ tickets: [t('a')] });
  try {
    delete process.env.MUGGE_THINKING;
    for (const [env, want] of [[undefined, false], ['on', true], ['off', false]] as const) {
      if (env === undefined) delete process.env.MUGGE_REVIEW_THINKING;
      else process.env.MUGGE_REVIEW_THINKING = env;
      const fake = startFakeModel({ solutions: { a: ok('a') } });
      const events: EngineEvent[] = [];
      try {
        const wt = join(dir, `rt-${env}`);
        mkdirSync(wt, { recursive: true });
        await runTicket(plan.tickets[0]!, wt, { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
        expect(fake.calls().map((c) => [c.kind, c.thinking])).toEqual([['write', undefined], ['review', want]]);
      } finally {
        fake.stop();
      }
      // MUGGE_THINKING unset: the worker call shows as thinking (Qwen3.8 thinks unless told not to)
      expect(statesOf(events).get('a')![0]).toBe('thinking');
    }
    expect(() => reviewThinkingFromEnv({ MUGGE_REVIEW_THINKING: 'maybe' })).toThrow(/on or off/);
  } finally {
    delete process.env.MUGGE_REVIEW_THINKING;
    process.env.MUGGE_THINKING = was ?? 'off';
  }
});
