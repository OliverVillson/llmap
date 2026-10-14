import { afterAll, describe, expect, test } from 'bun:test';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { checkBench } from '../bench/check.ts';
import { findProjects, loadProject } from '../bench/load.ts';
import { compare, summarize, type Row } from '../bench/report.ts';
import { firstTickets, runBench } from '../bench/run.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import type { WorkerOutput } from '../src/tickets/schema.ts';

const MINI = join(import.meta.dir, 'fixtures', 'bench-mini');
const tmp = mkdtempSync(join(tmpdir(), 'mugge-bench-test-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const ADD = 'def add(a, b):\n    return a + b\n';
const MUL = 'from calc.add import add\n\n\ndef mul(a, b):\n    out = 0\n    for _ in range(a):\n        out = add(out, b)\n    return out\n';

async function bench(solutions: Record<string, WorkerOutput>, out: string, broken: string[] = []) {
  const fake = startFakeModel({ solutions, broken });
  try {
    const rows = await runBench({
      projects: findProjects(MINI),
      inference: (seed) => new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake', extra: { seed } }),
      sandbox: new LocalSandbox(),
      seeds: 2,
      concurrency: 4,
      out: join(tmp, out),
    });
    return { rows, report: JSON.parse(readFileSync(join(tmp, out, 'report.json'), 'utf8')) };
  } finally {
    fake.stop();
  }
}

describe('bench projects', () => {
  test('loads tags, hidden commands and dependency order', () => {
    const p = loadProject(join(MINI, 'calc'));
    expect(p.canary).toBe('MUGGE-BENCH-TEST-FIXTURE');
    expect(p.split).toBe('dev');
    const mul = p.tickets.find((t) => t.ticket.id === 'mul')!;
    expect(mul.key).toBe('calc/mul');
    expect(mul.deps).toEqual(['add']);
    expect(mul.tags).toEqual({ theme: 'ml', language: 'python', role: 'implement', hidden: ['python3 -m unittest tests.test_hidden_mul'] });
  });

  test('the checker admits valid tickets and names what is wrong with bad ones', async () => {
    expect((await checkBench(findProjects(MINI), { sandbox: new LocalSandbox(), repeats: 1 })).every((c) => c.problems.length === 0)).toBe(true);
    const bad = join(tmp, 'bad');
    cpSync(MINI, bad, { recursive: true });
    // add's reference only passes the visible test, and its canary is wrong
    writeFileSync(join(bad, 'calc/solutions/add.json'), JSON.stringify({ files: { 'calc/add.py': 'def add(a, b):\n    return 5\n' }, canary: 'other' }));
    const add = (await checkBench(findProjects(bad), { sandbox: new LocalSandbox(), repeats: 1 })).find((c) => c.key === 'calc/add')!.problems.join('\n');
    expect(add).toContain('hidden tests fail with the reference solution');
    expect(add).toContain('canary');
    // mul's hidden test no longer tests mul
    const weak = join(tmp, 'weak');
    cpSync(MINI, weak, { recursive: true });
    writeFileSync(join(weak, 'calc/hidden/mul/tests/test_hidden_mul.py'), 'import unittest\n\n\nclass T(unittest.TestCase):\n    def test_nothing(self):\n        pass\n');
    const mul = (await checkBench(findProjects(weak), { sandbox: new LocalSandbox(), repeats: 1 })).find((c) => c.key === 'calc/mul')!;
    expect(mul.problems).toEqual(["hidden tests pass without this ticket's solution"]);
  });
});

describe('bench runs', () => {
  test('reference answers solve every ticket on every seed', async () => {
    const { rows, report } = await bench({ add: { files: { 'calc/add.py': ADD }, note: '' }, mul: { files: { 'calc/mul.py': MUL }, note: '' } }, 'ref');
    expect(rows.length).toBe(4);
    expect(rows.every((r) => r.solved && r.hidden && r.firstTry)).toBe(true);
    expect(report.score).toBe(100);
    expect(report.hiddenFailRate).toBe(0);
    expect(report.byTheme).toEqual({ backend: { tickets: 1, score: 100 }, ml: { tickets: 1, score: 100 } });
    expect(report.solvedPerGpuHour).toBeGreaterThan(0);
  });

  test('an answer that only passes the visible test is not solved, and deps come from the reference', async () => {
    const { rows, report } = await bench({ add: { files: { 'calc/add.py': 'def add(a, b):\n    return 5\n' }, note: '' }, mul: { files: { 'calc/mul.py': MUL }, note: '' } }, 'gamed');
    const add = rows.filter((r) => r.ticket === 'add');
    expect(add.every((r) => r.visible && !r.hidden && !r.solved)).toBe(true);
    expect(add[0].error).toContain('hidden');
    // mul runs on the reference add, so add's bad answer does not cascade
    expect(rows.filter((r) => r.ticket === 'mul').every((r) => r.solved)).toBe(true);
    expect(report.score).toBe(50);
    expect(report.visibleScore).toBe(100);
    expect(report.hiddenFailRate).toBe(50);
  });

  test('answers with no files count as format failures and use the whole budget', async () => {
    const { rows, report } = await bench({}, 'empty', ['add', 'mul']);
    expect(rows.every((r) => !r.solved && r.attempts === 2 && r.emptyAnswers === 2)).toBe(true);
    expect(report.score).toBe(0);
    expect(report.emptyAnswerRate).toBe(100);
  });
});

describe('bench report', () => {
  const row = (key: string, seed: number, solved: boolean, theme = 'backend'): Row => ({
    project: key.split('/')[0], ticket: key.split('/')[1], key, seed, split: 'test', theme, language: 'python', role: 'implement',
    visible: solved, hidden: solved, solved, attempts: 1, firstTry: solved, calls: 1, emptyAnswers: 0, refused: 0,
    promptTokens: 100, completionTokens: 50, inferenceMs: 1, checkMs: 1, wallMs: 2, error: null,
  });
  // 200 tickets; the reference solves the first 120, the build a chosen number of those
  const runs = (solvedByBuild: number) => {
    const ref: Row[] = [], build: Row[] = [];
    for (let i = 0; i < 200; i++) {
      const k = `p/t${i}`;
      ref.push(row(k, 0, i < 120));
      build.push(row(k, 0, i < solvedByBuild));
    }
    return { ref, build };
  };

  test('each ticket counts once however many seeds ran', () => {
    const s = summarize([row('p/a', 0, true), row('p/a', 1, false), row('p/a', 2, false), row('p/b', 0, true)]);
    expect(s.tickets).toBe(2);
    expect(s.seeds).toBe(3);
    expect(s.score).toBeCloseTo(66.7, 1);
  });

  test('verdicts: met, missed and too close to call', () => {
    const met = compare(runs(120).build, runs(120).ref);
    expect(met.ratio).toBe(1);
    expect(met.verdict).toBe('met');
    expect(compare(runs(80).build, runs(80).ref).verdict).toBe('missed');
    const close = compare(runs(108).build, runs(108).ref);
    expect(close.ratio).toBe(0.9);
    expect(close.verdict).toBe('too close to call');
    expect(close.diff).toBe(-6);
    expect(close.diffCI[0]).toBeLessThan(-6);
    expect(close.diffCI[1]).toBeGreaterThan(-6);
  });

  test('intervals are the same on every run', () => {
    const a = compare(runs(100).build, runs(100).ref), b = compare(runs(100).build, runs(100).ref);
    expect(a.ratioCI).toEqual(b.ratioCI);
  });
});

test('a quick run takes tickets in turn from each project, in plan order', () => {
  const p = (name: string, ids: string[]) => ({ ...loadProject(join(MINI, 'calc'), name), tickets: ids.map((id) => ({ ...loadProject(join(MINI, 'calc')).tickets[0], key: `${name}/${id}` })) });
  const got = firstTickets([p('a', ['1', '2', '3']), p('b', ['1']), p('c', ['1', '2'])], 5);
  expect(got.flatMap((x) => x.tickets.map((t) => t.key))).toEqual(['a/1', 'a/2', 'a/3', 'b/1', 'c/1', 'c/2'].filter((k) => k !== 'a/3'));
  expect(firstTickets([p('a', ['1'])], 9).flatMap((x) => x.tickets).length).toBe(1);
});
