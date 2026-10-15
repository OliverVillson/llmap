import { describe, expect, test } from 'bun:test';
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findProjects } from '../bench/load.ts';
import { runBench } from '../bench/run.ts';
import { MESS, runSmoke, SPIKE_DIR } from '../bench/smoke.ts';
import type { EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { parseFiles } from '../src/engine/files.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { classifyAnswer, crewFromEnv, defaultCrew, lostWarning, statusLine, summarizeCallLog, Telemetry, type CallRecord, type CrewStatus } from '../src/engine/telemetry.ts';

const tmp = (p: string) => mkdtempSync(join(tmpdir(), p));
const check = (raw: string, owns: string[], cut = false) => classifyAnswer(raw, owns, parseFiles(raw, owns), cut);

describe('parser drops', () => {
  test('a good answer is not a drop; a partial one says what is missing', () => {
    expect(check('src/a.py\n```python\nx = 1\n```', ['src/a.py']).drop).toBeUndefined();
    const partial = check('src/a.py\n```python\nx = 1\n```', ['src/a.py', 'src/b.py']);
    expect(partial.drop).toBeUndefined();
    expect(partial.missing).toEqual(['src/b.py']);
  });
  test('each way an answer gives no usable file has its reason', () => {
    expect(check('<think>still going and going', ['a.py'], true).drop).toBe('cut');
    expect(check('<think>long</think>', ['a.py']).drop).toBe('empty');
    expect(check('', ['a.py']).drop).toBe('empty');
    expect(check('I would loop over the items and sum them.', ['a.py']).drop).toBe('no_fence');
    expect(check('```\nx\n```\n\n```\ny\n```', ['a.py', 'b.py']).drop).toBe('unlabelled');
    expect(check('a.py\n```\nx\n```\nother/z.py\n```\nz\n```', ['a.py']).drop).toBe('not_owned');
  });
  test('the 20% warning counts drops and cut answers, not call errors', () => {
    const t = new Telemetry({ warn: null });
    const model = (ticket: string, extra: Partial<Extract<EngineEvent, { type: 'model' }>> = {}): EngineEvent => ({ type: 'model', at: 1000, ticket, attempt: 1, kind: 'write', promptTokens: 10, completionTokens: 5, ms: 10, note: '', files: 1, ...extra });
    for (let i = 0; i < 8; i++) t.emit(model(`t${i}`));
    t.emit(model('t8', { drop: 'no_fence', files: 0 }));
    t.emit(model('t9', { cut: true }));
    t.emit({ type: 'call_error', at: 1000, ticket: 't10', attempt: 1, kind: 'write', error: 'inference 502' });
    const c = t.calls();
    expect([c.calls, c.errors, c.dropped, c.cut, c.lost, c.clean]).toEqual([11, 1, 1, 1, 2, 8]);
    expect(lostWarning(c)).toBeNull(); // 2 of 10 answers is 20%: at the line, not over it
    t.emit(model('t11', { drop: 'empty', files: 0 }));
    expect(lostWarning(t.calls())).toContain('3 of 11 answers (27%)');
    expect(t.close().warnings[0]).toContain('empty 1, no_fence 1');
  });
});

describe('crew status', () => {
  test('folds a ticket through write, check, fix and finish, with tok/s over the window', () => {
    let now = 1_000_000;
    const dir = tmp('mugge-tel-');
    const tel = new Telemetry({ dir, now: () => now, warn: null, crew: defaultCrew(2), statusIntervalMs: 0 });
    tel.setTickets([
      { id: 'one', owns: ['a.c'], depends_on: [], maxAttempts: 4 },
      { id: 'two', owns: ['b.c'], depends_on: ['one'], maxAttempts: 4 },
      { id: 'three', owns: ['c.c'], depends_on: [], maxAttempts: 4 },
    ]);
    tel.emit({ type: 'plan', at: now, tickets: [], concurrency: 2 });
    tel.emit({ type: 'start', at: now, ticket: 'one', attempt: 1, worktree: '/w' });
    let s = tel.status();
    const a = s.agents.find((x) => x.ticket === 'one')!;
    expect([a.state, a.attempt, a.owns]).toEqual(['writing', 1, ['a.c']]);
    expect(s.tickets.find((x) => x.id === 'two')!.state).toBe('waiting');
    expect(s.tickets.find((x) => x.id === 'three')!.state).toBe('ready');
    expect(s.crew.state).toBe('working');

    now += 10_000; // a 10 s call writing 1000 tokens: 100 tok/s while it ran
    tel.emit({ type: 'model', at: now, ticket: 'one', attempt: 1, kind: 'write', promptTokens: 3000, completionTokens: 1000, ms: 10_000, note: '', files: 1, model: 'coder-c', maxTokens: 4096 });
    expect(tel.status().agents.find((x) => x.id === a.id)!.state).toBe('checking');
    tel.emit({ type: 'check', at: now, ticket: 'one', attempt: 1, command: 'make', ok: false, ms: 5, tail: 'error' });
    s = tel.status();
    const fixing = s.agents.find((x) => x.id === a.id)!;
    expect([fixing.state, fixing.attempt, fixing.kind]).toEqual(['fixing', 2, 'fix']);
    expect(s.tokS.total).toBe(100); // 1000 tokens over the 10 s since the start
    expect(s.tokS.byTeam[a.team]).toBe(100);
    expect(s.tokS.byMachine.local).toBe(100);
    expect(s.machines[0]!.slotsUsed).toBe(1);
    expect(s.tokS.spark.length).toBe(60);
    expect(s.tokS.spark.at(-1)).toBe(100);

    now += 2000;
    tel.emit({ type: 'model', at: now, ticket: 'one', attempt: 2, kind: 'fix', promptTokens: 3500, completionTokens: 200, ms: 2000, note: 'fixed', files: 1 });
    tel.emit({ type: 'check', at: now, ticket: 'one', attempt: 2, command: 'make', ok: true, ms: 5, tail: '' });
    tel.emit({ type: 'finish', at: now, ticket: 'one', state: 'done', attempts: 2 });
    s = tel.status();
    expect(s.agents.find((x) => x.id === a.id)!.state).toBe('finished');
    expect(s.tickets.find((x) => x.id === 'two')!.state).toBe('ready');

    now += 120_000; // two quiet minutes later the window is empty
    expect(tel.status().tokS.total).toBe(0);
    const end = tel.close();
    expect(end.crew.state).toBe('needs_you'); // two tickets never ran

    const rows: CallRecord[] = readFileSync(join(dir, 'calls.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    expect(rows.map((r) => [r.ticket, r.attempt, r.kind, r.agent, r.clean, r.tokS])).toEqual([['one', 1, 'write', a.id, true, 100], ['one', 2, 'fix', a.id, true, 100]]);
    const onDisk: CrewStatus = JSON.parse(readFileSync(join(dir, 'status.json'), 'utf8'));
    expect(onDisk.schema).toBe('centipede.status/1');
    expect(onDisk.calls.calls).toBe(2);
    expect(summarizeCallLog(join(dir, 'calls.jsonl')).completionTokens).toBe(1200);
    expect(statusLine(onDisk)).toContain('tickets 1/3 done');
    rmSync(dir, { recursive: true, force: true });
  });

  test('tickets spread over the four teams; a scheduler-assigned agent is kept', () => {
    const tel = new Telemetry({ warn: null });
    for (const id of ['p', 'q', 'r', 's', 't']) tel.emit({ type: 'start', at: 1, ticket: id, attempt: 1, worktree: '' });
    tel.emit({ type: 'start', at: 1, ticket: 'u', attempt: 1, worktree: '', agent: 'c7', machine: 'mac' });
    const s = tel.status();
    expect(s.agents.filter((a) => a.state === 'writing').map((a) => a.id)).toEqual(['a1', 'a2', 'b1', 'c1', 'c7', 'd1']);
    expect(s.agents.find((a) => a.id === 'c7')!.machine).toBe('mac');
    expect(s.teams.map((t) => t.working)).toEqual([2, 1, 2, 1]);
  });

  test('slot events queue an agent and say which machine its call runs on; pool snapshots give real slot use', () => {
    const tel = new Telemetry({ warn: null, crew: crewFromEnv({ MUGGE_BACKENDS: JSON.stringify([{ name: '4060', url: 'http://127.0.0.1:8080/v1', roles: ['worker'], slots: 1 }, { name: 'mac', url: 'http://127.0.0.1:8081/v1', roles: ['worker', 'lead', 'manager'], slots: 1 }]) }) });
    tel.setTickets([{ id: 'x', owns: ['x.c'], depends_on: [] }, { id: 'y', owns: ['y.c'], depends_on: [] }]);
    tel.emit({ type: 'start', at: 1, ticket: 'x', attempt: 1, worktree: '' });
    tel.emit({ type: 'start', at: 1, ticket: 'y', attempt: 1, worktree: '' });
    tel.emit({ type: 'slot', at: 2, ticket: 'x', state: 'got', machine: 'mac' });
    tel.emit({ type: 'slot', at: 2, ticket: 'y', state: 'queued' });
    let s = tel.status();
    const ax = s.agents.find((a) => a.ticket === 'x')!, ay = s.agents.find((a) => a.ticket === 'y')!;
    expect([ax.state, ax.machine, ay.state]).toEqual(['writing', 'mac', 'queued']);
    expect(s.tickets.map((t) => t.state)).toEqual(['running', 'queued']);
    expect([s.crew.agents.working, s.crew.agents.queued]).toEqual([2, 1]);
    tel.serving({ at: 3, windowS: 1, agents: [], teams: [], total: { slots: 2, busy: 1, queued: 1, calls: 0, completionTokens: 0, tokPerSec: 0, promptTokPerSec: 0 },
      machines: [{ name: '4060', url: 'http://127.0.0.1:8080/v1', roles: ['worker'], slots: 1, busy: 0, occupancy: 0, queued: 1, calls: 0, promptTokens: 0, completionTokens: 0, tokPerSec: 0, promptTokPerSec: 0 },
        { name: 'mac', url: 'http://127.0.0.1:8081/v1', roles: ['worker'], slots: 1, busy: 1, occupancy: 1, queued: 0, calls: 0, promptTokens: 0, completionTokens: 0, tokPerSec: 0, promptTokPerSec: 250 }] });
    s = tel.status();
    expect(s.machines.map((m) => [m.id, m.slotsUsed, m.queued, m.promptTokS])).toEqual([['4060', 0, 1, 0], ['mac', 1, 0, 250]]);
    tel.emit({ type: 'slot', at: 4, ticket: 'y', state: 'got', machine: '4060' });
    expect(tel.status().agents.find((a) => a.ticket === 'y')!.state).toBe('writing');
  });

  test('MUGGE_CREW sets machines and teams', () => {
    const crew = crewFromEnv({ MUGGE_CREW: JSON.stringify({ machines: [{ id: '4060', slots: 16 }, { id: 'mac', slots: 8 }], teams: [{ id: 'a', machine: '4060', agents: 10, domain: 'backend' }, { id: 'b', machine: 'mac', agents: 5 }] }) });
    const tel = new Telemetry({ crew, warn: null });
    const s = tel.status();
    expect(s.crew.agents.total).toBe(15);
    expect(s.teams[0]!.domain).toBe('backend');
    expect(s.machines.map((m) => [m.id, m.slots, m.agents])).toEqual([['4060', 16, 10], ['mac', 8, 5]]);
    expect(() => crewFromEnv({ MUGGE_CREW: '{"machines":[{"id":"x","slots":1}],"teams":[{"id":"a","machine":"y","agents":1}]}' })).toThrow(/unknown machine/);
  });
});

describe('the inference client reads forced and cut answers', () => {
  test('forced comes from the proxy, cut from finish_reason, max_tokens from the cap', async () => {
    const fake = startFakeModel({ solutions: { t: { files: { 'a.ts': 'x' }, note: '' } }, forced: ['t'], cut: ['u'] });
    try {
      const inf = new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', maxCallTokens: 512 });
      const forced = await inf.text('m', [{ role: 'user', content: 'Ticket: t' }], 4000);
      expect([forced.forced, forced.cut, forced.maxTokens]).toEqual(['length', undefined, 512]);
      const cut = await inf.text('m', [{ role: 'user', content: 'Ticket: u' }], 100);
      expect([cut.forced, cut.cut, cut.maxTokens]).toEqual([undefined, true, 100]);
    } finally {
      fake.stop();
    }
  });
});

describe('spike smoke', () => {
  test('the fake model with a messy first answer on four tickets MEETS, and the log names every bad answer', async () => {
    const out = tmp('mugge-smoke-');
    const r = await runSmoke({ fake: true, fakeMess: true, fakeLatencyMs: 20, out, quiet: true });
    expect(r.verdict).toBe('MEETS');
    expect(r.refs).toEqual({ passed: 15, tickets: 15, failures: [] });
    expect(r.passed).toBe(15);
    expect([r.calls.calls, r.calls.forced, r.calls.cut, r.calls.dropped, r.calls.errors]).toEqual([18, 1, 1, 2, 0]);
    const rows: CallRecord[] = readFileSync(join(out, 'calls.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    const bad = rows.filter((x) => !x.clean).map((x) => `${x.ticket.split('/')[1]} ${x.attempt} ${x.forced ?? ''}${x.cut ? 'cut' : ''}${x.drop ?? ''}`).sort();
    expect(bad).toEqual([`${MESS.cut[0]}#s0 1 cut`, `${MESS.flaky[0]}#s0 1 no_fence`, `${MESS.forced[0]}#s0 1 length`, `${MESS.prose[0]}#s0 1 no_fence`].sort());
    const status: CrewStatus = JSON.parse(readFileSync(join(out, 'status.json'), 'utf8'));
    expect(status.crew.tickets.done).toBe(15);
    expect(existsSync(join(out, 'smoke.json'))).toBe(true);
    rmSync(out, { recursive: true, force: true });
  }, 180_000);

  test('a broken reference answer stops the smoke before any model call', async () => {
    const bench = tmp('mugge-smoke-bench-');
    cpSync(SPIKE_DIR, bench, { recursive: true });
    const f = join(bench, 'solutions', 'py-validate.json');
    const sol = JSON.parse(readFileSync(f, 'utf8'));
    for (const k of Object.keys(sol.files)) sol.files[k] = 'raise SystemExit(1)\n';
    writeFileSync(f, JSON.stringify(sol));
    const out = tmp('mugge-smoke-');
    const r = await runSmoke({ fake: true, out, quiet: true, bench });
    expect(r.verdict).toBe('NO SCORE');
    expect(r.refs!.passed).toBeLessThan(15); // its dependents run on it too
    expect(r.refs!.failures.some((x) => x.includes('/py-validate:'))).toBe(true);
    expect(r.calls.calls).toBe(0);
    rmSync(bench, { recursive: true, force: true });
    rmSync(out, { recursive: true, force: true });
  }, 180_000);

  test('top_k 0 and a missing server are refused up front', async () => {
    const out = tmp('mugge-smoke-');
    expect((await runSmoke({ fake: true, out, quiet: true, topK: 0 })).reasons[0]).toContain('top_k');
    const down = await runSmoke({ url: 'http://127.0.0.1:9/v1', out, quiet: true });
    expect(down.verdict).toBe('NO SCORE');
    expect(down.reasons[0]).toContain('no model server');
    rmSync(out, { recursive: true, force: true });
  });
});

test('bench transcripts mark cut, forced and dropped answers so harvest can skip them', async () => {
  const out = tmp('mugge-bench-tel-');
  const projects = findProjects(SPIKE_DIR).map((p) => ({ ...p, tickets: p.tickets.filter((t) => ['c-hash', 'py-store', 'ts-format'].includes(t.ticket.id)) }));
  const solutions = Object.fromEntries(projects[0]!.tickets.map((t) => [t.ticket.id, { files: JSON.parse(readFileSync(join(SPIKE_DIR, 'solutions', `${t.ticket.id}.json`), 'utf8')).files, note: '' }]));
  const fake = startFakeModel({ solutions, ...MESS });
  try {
    const tel = new Telemetry({ dir: out, warn: null });
    await runBench({ projects, inference: () => new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), sandbox: new LocalSandbox(), seeds: 1, concurrency: 3, out, transcripts: true, telemetry: tel });
    const calls = readFileSync(join(out, 'transcripts.jsonl'), 'utf8').trim().split('\n').flatMap((l) => JSON.parse(l).calls);
    const flags = calls.map((c: any) => `${c.cut ? 'cut' : ''}${c.forced ?? ''}${c.drop ?? ''}` || 'clean').sort();
    expect(flags).toEqual(['clean', 'clean', 'cut', 'length', 'no_fence'].sort());
    const results = readFileSync(join(out, 'results.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    expect(results.reduce((n, r) => n + r.cutAnswers + r.forcedAnswers + r.droppedAnswers, 0)).toBe(3);
    expect(tel.close().crew.agents.finished).toBe(3);
  } finally {
    fake.stop();
    rmSync(out, { recursive: true, force: true });
  }
}, 120_000);
