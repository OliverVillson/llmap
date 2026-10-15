import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyEvent, emptyView, slotStates, type EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { initRepo } from '../src/engine/git.ts';
import { runTicket } from '../src/engine/harness.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { REVIEW_SYSTEM, reviewMessages } from '../src/engine/prompts.ts';
import { diffOwned, normalizeReview, REVIEW_MAX_TOKENS, reviewFeedback, reviewFromEnv, snapshot } from '../src/engine/review.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
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
  test('every ticket moves through legal states; slots are never shared', async () => {
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
    // five tickets, two slots: someone had to queue
    expect([...seq.values()].some((s) => s.includes('queued'))).toBe(true);
    const fix = events.find((e) => e.type === 'state' && e.ticket === 'b' && e.state === 'fixing');
    expect(fix).toMatchObject({ attempt: 2, of: 4 });
    // replaying the events, a slot never works two tickets at once
    const view = emptyView();
    for (const e of events) {
      applyEvent(view, e);
      const busy = slotStates(view).filter((s) => s.state !== 'idle').map((s) => s.ticket);
      expect(new Set(busy).size).toBe(busy.length);
      for (const s of view.tickets) if (s.slot !== undefined) expect(s.slot).toBeLessThan(2);
    }
    expect(slotStates(view).every((s) => s.state === 'idle')).toBe(true);
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
      expect(r).toMatchObject({ ok: true, attempts: 4, firstTry: true, reviewUnresolved: true, error: null });
      expect(readFileSync(join(wt, 'a.txt'), 'utf8')).toBe('ok-a\n');
    } finally {
      fake.stop();
    }
    const last = events.filter((e) => e.type === 'state').at(-1);
    expect(last).toMatchObject({ state: 'finished', reason: 'lead review problems left unfixed' });
    expectLegal(events);
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
