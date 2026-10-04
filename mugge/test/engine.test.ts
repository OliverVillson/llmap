import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyEvent, emptyView, type EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { git, initRepo } from '../src/engine/git.ts';
import { OpenAICompatible, parseJsonAnswer } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
import { PlanError, validatePlan } from '../src/tickets/schema.ts';

const t = (id: string, extra: any = {}) => ({ id, title: `Write ${id}`, image: 'node', owns: [`${id}.txt`], acceptance: [`grep -q ok-${id} ${id}.txt`], ...extra });

describe('plan validation', () => {
  test('fills defaults', () => {
    const p = validatePlan({ tickets: [t('a')] });
    expect(p.tickets[0].budget.attempts).toBe(4);
    expect(p.tickets[0].role).toBe('implement');
  });
  test('rejects unknown deps, cycles and unsafe paths', () => {
    expect(() => validatePlan({ tickets: [t('a', { depends_on: ['x'] })] })).toThrow(PlanError);
    expect(() => validatePlan({ tickets: [t('a', { depends_on: ['b'] }), t('b', { depends_on: ['a'] })] })).toThrow(/cycle/);
    expect(() => validatePlan({ tickets: [t('a', { owns: ['../etc/passwd'] })] })).toThrow(/owns path/);
    expect(() => validatePlan({ tickets: [t('a', { owns: ['src/*.ts'] })] })).toThrow(/owns path/);
  });
  test('two tickets that can run at once may not own the same file', () => {
    expect(() => validatePlan({ tickets: [t('a', { owns: ['x'] }), t('b', { owns: ['x'] })] })).toThrow(/both own x/);
    expect(() => validatePlan({ tickets: [t('a', { owns: ['x'] }), t('b', { owns: ['x'], depends_on: ['a'] })] })).not.toThrow();
  });
});

test('parseJsonAnswer tolerates fences', () => {
  expect(parseJsonAnswer<any>('```json\n{"a":1}\n```').a).toBe(1);
  expect(parseJsonAnswer<any>('sure: {"a":2} done').a).toBe(2);
});

describe('runPlan with a fake model', () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), 'mugge-engine-'));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test('runs in parallel worktrees, fixes, skips, and integrates', async () => {
    const repo = join(dir, 'repo');
    Bun.spawnSync(['mkdir', '-p', repo]);
    writeFileSync(join(repo, 'README.md'), 'scaffold\n');
    const base = initRepo(repo);
    const plan = validatePlan({
      interfaces: ['README.md'],
      tickets: [
        t('a'), t('b'), t('d'), t('e'),
        t('c', { depends_on: ['a', 'b'], acceptance: ['grep -q ok-a a.txt', 'grep -q ok-b b.txt', 'grep -q ok-c c.txt'] }),
        t('x', { budget: { attempts: 2 } }),
        t('y', { depends_on: ['x'] }),
      ],
    });
    const solutions: any = {};
    for (const id of ['a', 'b', 'c', 'd', 'e', 'x', 'y']) solutions[id] = { files: { [`${id}.txt`]: `ok-${id}\n` }, note: `wrote ${id}` };
    solutions.d.files['README.md'] = 'hijack';
    const fake = startFakeModel({ solutions, flaky: ['b'], broken: ['x'], latencyMs: 50 });
    const events: EngineEvent[] = [];
    try {
      const report = await runPlan({
        repo, base, plan, concurrency: 4, worktreeRoot: join(dir, 'wt'), sandbox: new LocalSandbox(),
        inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }),
        emit: (e) => events.push(e),
      });
      const s = report.summary;
      expect(report.states.get('a')).toBe('done');
      expect(report.states.get('b')).toBe('done');
      expect(report.results.get('b')!.attempts).toBe(2);
      expect(report.states.get('c')).toBe('done');
      expect(report.states.get('d')).toBe('failed'); // keeps writing a file it does not own
      expect(events.some((e) => e.type === 'refused' && e.ticket === 'd')).toBe(true);
      expect(report.states.get('x')).toBe('failed');
      expect(report.results.get('x')!.attempts).toBe(2);
      expect(report.states.get('y')).toBe('skipped');
      expect(s.peakParallel).toBeGreaterThan(1);
      expect(s.firstTry).toBe(3); // a, c and e; b needed a fix
      expect(report.integration?.ok).toBe(true);
      // The integration branch holds every done ticket's file and nothing from failed ones.
      const files = git(repo, ['ls-tree', '--name-only', 'mugge/integration']).out.split('\n');
      expect(files.sort()).toEqual(['README.md', 'a.txt', 'b.txt', 'c.txt', 'e.txt']);
      expect(git(repo, ['show', 'mugge/integration:README.md']).out).toBe('scaffold');
      // Commits carry no trailers.
      expect(git(repo, ['log', '--format=%B', 'mugge/integration']).out).not.toMatch(/Co-Authored|Generated/i);
      const view = events.reduce(applyEvent, emptyView());
      expect(view.summary?.done).toBe(4);
      expect(view.running).toBe(0);
    } finally {
      fake.stop();
    }
  });
});
