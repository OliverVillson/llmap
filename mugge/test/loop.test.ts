import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyEvent, emptyView, agentStates, type EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { gitOk, initRepo } from '../src/engine/git.ts';
import { runTicket } from '../src/engine/harness.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { REVIEW_SYSTEM, reviewMessages } from '../src/engine/prompts.ts';
import { diffOwned, normalizeReview, REVIEW_MAX_TOKENS, reviewFeedback, reviewFromEnv, snapshot } from '../src/engine/review.ts';
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
    expect(seq.get('b')!.filter((s) => s !== 'queued' && s !== 'waiting')).toEqual(['writing', 'checking', 'fixing', 'checking', 'finished']);
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
      expect(['writing', 'fixing']).toContain(s[i + 1]!);
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

  test('a file-ownership clash blocks the later ticket for the owner and is recorded', async () => {
    const { repo, base } = await repoWith('clash');
    const plan = validatePlan({ tickets: [t('a', { owns: ['shared.txt'], acceptance: ['grep -q ok-a shared.txt'] }), t('b', { owns: ['shared.txt'] }), t('c', { depends_on: ['b'] }), t('d')] });
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

  test('review is off unless switched on', async () => {
    const fake = startFakeModel({ solutions: { a: ok('a') } });
    try {
      const old = process.env.MUGGE_REVIEW;
      delete process.env.MUGGE_REVIEW;
      const r = await runTicket(ticket, folder('off'), { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: () => {} });
      if (old !== undefined) process.env.MUGGE_REVIEW = old;
      expect(r.ok).toBe(true);
      expect(r.review).toBeNull();
      expect(fake.calls().map((c) => c.kind)).toEqual(['write']);
    } finally {
      fake.stop();
    }
    expect(reviewFromEnv({ MUGGE_REVIEW: 'on' })).toBe(true);
    expect(reviewFromEnv({})).toBe(false);
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
      expect(calls[2]!.prompt).toContain('## Fix\nThis command failed:\n$ (lead review)\nThe lead reviewed your passing files and found problems:\n- empty input is not handled\n\nSuggested fix: handle the empty case\n\nReturn your files fixed so it passes.');
      expect(readFileSync(join(wt, 'a.txt'), 'utf8')).toBe('ok-a\nempty case\n');
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')).toEqual(['writing', 'checking', 'reviewing', 'fixing', 'checking', 'finished']);
    expectLegal(events);
  });

  test('a review fix that never passes again keeps the version that passed', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['wrong'], fix: 'rewrite it' }] },
      fixes: { a: [{ files: { 'a.txt': 'broken\n' }, note: '' }] },
    });
    const wt = folder('unresolved');
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, wt, { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 4, firstTry: true, reviewUnresolved: true, error: null, answerAttempt: 1 });
      expect(readFileSync(join(wt, 'a.txt'), 'utf8')).toBe('ok-a\n');
    } finally {
      fake.stop();
    }
    const last = events.filter((e) => e.type === 'state').at(-1);
    expect(last).toMatchObject({ state: 'finished', reason: 'lead review problems left unfixed' });
    expectLegal(events);
  });

  test('a review fix whose calls are refused or fail still ends through checking, on the restored files', async () => {
    const fake = startFakeModel({
      solutions: { a: ok('a') },
      reviews: { a: [{ verdict: 'problems', problems: ['wrong'], fix: 'rewrite it' }] },
      // every fix writes a file the ticket does not own: refused, so the last state is a call state
      fixes: { a: [{ files: { 'a.txt': 'ok-a\n', 'b.txt': 'x\n' }, note: '' }] },
    });
    const wt = folder('refused-fix');
    const events: EngineEvent[] = [];
    try {
      const r = await runTicket(ticket, wt, { plan, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: (e) => events.push(e), review: true });
      expect(r).toMatchObject({ ok: true, attempts: 4, reviewUnresolved: true, answerAttempt: 1 });
      expect(readFileSync(join(wt, 'a.txt'), 'utf8')).toBe('ok-a\n');
    } finally {
      fake.stop();
    }
    expect(statesOf(events).get('a')!.slice(-3)).toEqual(['fixing', 'checking', 'finished']);
    // the restored files were checked again
    expect(events.filter((e) => e.type === 'check').length).toBe(2);
    expectLegal(events);
  });

  test('a review answer cut at the cap counts its tokens and shows cut', async () => {
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
      expect(r).toMatchObject({ ok: true, completionTokens: 305 });
      expect(r.review).toMatchObject({ verdict: 'error', cut: true, completionTokens: 300 });
    } finally {
      server.stop(true);
    }
    expect(events.find((e) => e.type === 'review')).toMatchObject({ verdict: 'error', cut: true, completionTokens: 300 });
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

  test('a review on the last attempt cannot fix; a broken review answer counts as approval', async () => {
    const one = validatePlan({ tickets: [t('a', { budget: { attempts: 1 } })] });
    const fake = startFakeModel({ solutions: { a: ok('a') }, reviews: { a: [{ verdict: 'problems', problems: ['p'], fix: 'f' }] } });
    try {
      const r = await runTicket(one.tickets[0]!, folder('last'), { plan: one, inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), emit: () => {}, review: true });
      expect(r).toMatchObject({ ok: true, attempts: 1, reviewUnresolved: true });
    } finally {
      fake.stop();
    }
    expect(normalizeReview({ verdict: 'maybe' })).toMatch(/not approve or problems/);
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
  expect(reviewFeedback({ problems: ['x'], fix: '' })).toBe('The lead reviewed your passing files and found problems:\n- x');
});
