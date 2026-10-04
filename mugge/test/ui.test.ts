import { describe, expect, test } from 'bun:test';
import type { RunSummary, RunView } from '../src/engine/events.ts';
import { renderDog, renderSprite, dogWidth } from '../src/ui/dog/render.ts';
import { dogFrame } from '../src/ui/dog/line.ts';
import { RUN, SLEEP, TINY_RUN } from '../src/ui/dog/sprites.ts';
import { renderHeader, renderHome, renderRun, renderSplash, stripAnsi, truncateAnsi, visibleWidth, type HomeProject } from '../src/ui/screen.ts';
import { detectColorLevel, type ColorLevel } from '../src/ui/theme.ts';

const NOW = 1_700_000_000_000;
const SUMMARY: RunSummary = {
  tickets: 4, done: 3, failed: 1, skipped: 0, firstTry: 2, calls: 9, promptTokens: 42_000, completionTokens: 12_345,
  peakParallel: 3, wallMs: 130_000, serialMs: 420_000, inferenceMs: 100_000, checkMs: 20_000, integrated: true,
};

function view(over: Partial<RunView> = {}): RunView {
  return {
    tickets: [
      { id: 'T1', title: 'add parser', model: 'qwen2.5-coder-7b', state: 'done', attempt: 1, lastNote: 'wrote src/parser.ts and a very long note that should never overflow the screen width' },
      { id: 'T2-with-a-long-identifier-name', title: 'cli flags', model: 'qwen2.5-coder-7b', state: 'running', attempt: 2, lastNote: 'fixing type error' },
      { id: 'T3', title: 'tests', model: 'deepseek-coder-6.7b-instruct-long-name', state: 'failed', attempt: 3, lastNote: '', error: 'bun test failed\nexpected 1 got 2' },
      { id: 'T4', title: 'docs', model: 'qwen2.5-coder-7b', state: 'waiting', attempt: 0, lastNote: '' },
      { id: 'T5', title: 'skip me', model: 'm', state: 'skipped', attempt: 0, lastNote: '' },
    ],
    concurrency: 4,
    running: 1,
    startedAt: NOW - 65_000,
    summary: null,
    log: Array.from({ length: 30 }, (_, i) => `T${i % 5} #1 write ${i * 10} tok line ${i} ${'x'.repeat(i * 4)}`),
    ...over,
  };
}

const PROJECTS: HomeProject[] = [
  { name: 'demo-app', state: 'open', repo: 'github.com/someone/demo-app-with-a-long-repository-name', lastOpened: NOW - 7_200_000 },
  { name: 'other', state: 'closed' },
  { name: 'booting-project-name-that-is-long', state: 'booting', lastOpened: NOW - 100_000 },
];

const SIZES: Array<[number, number]> = [[20, 8], [40, 12], [60, 20], [80, 24], [132, 40], [33, 50]];
const LEVELS: ColorLevel[] = [0, 1, 2, 3];
const ESC = /\u001b/;

function expectFits(lines: string[], cols: number, rows?: number) {
  for (const l of lines) expect(visibleWidth(l)).toBeLessThanOrEqual(cols);
  if (rows !== undefined) expect(lines.length).toBeLessThanOrEqual(rows);
}

describe('width', () => {
  test('renderRun never exceeds cols or rows', () => {
    for (const [cols, rows] of SIZES)
      for (const level of LEVELS)
        for (const v of [view(), view({ summary: SUMMARY, running: 0 }), view({ tickets: [], log: [] })])
          for (const frame of [0, 3, 17]) {
            const vm = { state: 'ready', gpu: 'NVIDIA L4 24GB', costUsd: 1.2345 };
            expectFits(renderRun(v, { cols, rows, level, frame, project: 'my-project', vm, now: NOW }), cols, rows);
          }
  });

  test('renderHome never exceeds cols and fills rows', () => {
    for (const [cols, rows] of SIZES)
      for (const level of LEVELS)
        for (const ps of [PROJECTS, [], Array.from({ length: 40 }, (_, i) => ({ name: `p${i}`, state: 'closed' as const }))]) {
          const lines = renderHome(ps, { cols, rows, level, frame: 5, selected: 1, now: NOW });
          expectFits(lines, cols, rows);
          expect(lines.length).toBe(rows);
        }
  });

  test('renderSplash is exactly cols x rows', () => {
    for (const [cols, rows] of SIZES)
      for (const level of LEVELS)
        for (const ms of [0, 300, 900, 1500, 2500]) {
          const lines = renderSplash({ cols, rows, level, ms, seed: 42 });
          expect(lines.length).toBe(rows);
          for (const l of lines) expect(visibleWidth(l)).toBe(cols);
        }
  });

  test('renderHeader fits and right-aligns', () => {
    const lines = renderHeader({ cols: 60, level: 3, frame: 1, mode: 'run', title: 'proj', right: '1/4 done' });
    expect(lines.length).toBe(2);
    expectFits(lines, 60);
    expect(visibleWidth(lines[0]!)).toBe(60);
    expect(stripAnsi(lines[0]!).endsWith('1/4 done')).toBe(true);
    expectFits(renderHeader({ cols: 10, level: 3, frame: 1, mode: 'sleep', title: 'a long project title', right: 'x' }), 10);
  });

  test('truncateAnsi keeps escapes and closes them', () => {
    const s = '\u001b[32mhello world\u001b[39m';
    const t = truncateAnsi(s, 5);
    expect(stripAnsi(t)).toBe('hello');
    expect(t.endsWith('\u001b[0m')).toBe(true);
    expect(truncateAnsi(s, 50)).toBe(s);
  });
});

describe('level 0', () => {
  test('no escape codes anywhere', () => {
    const lines = [
      ...renderRun(view({ summary: SUMMARY }), { cols: 80, rows: 30, level: 0, frame: 2, project: 'p', vm: { state: 'booting' }, now: NOW }),
      ...renderHome(PROJECTS, { cols: 80, rows: 24, level: 0, frame: 2, selected: 0, now: NOW }),
      ...renderSplash({ cols: 80, rows: 24, level: 0, ms: 700, seed: 1 }),
      ...renderHeader({ cols: 80, level: 0, frame: 0, mode: 'sleep', title: 't' }),
      ...renderDog(0, { level: 0 }),
      dogFrame(0, { level: 0 }),
    ];
    for (const l of lines) expect(ESC.test(l)).toBe(false);
  });

  test('colour levels do use escapes', () => {
    for (const level of [1, 2, 3] as ColorLevel[])
      expect(renderRun(view(), { cols: 80, rows: 24, level, frame: 0, project: 'p', now: NOW }).some((l) => ESC.test(l))).toBe(true);
  });

  test('NO_COLOR disables colour', () => {
    expect(detectColorLevel({ NO_COLOR: '1' }, true)).toBe(0);
    expect(detectColorLevel({ COLORTERM: 'truecolor' }, true)).toBe(3);
    expect(detectColorLevel({ COLORTERM: 'truecolor' }, false)).toBe(0);
  });
});

describe('dog', () => {
  test('full-size sprites render to 6 rows of half blocks', () => {
    for (let f = 0; f < RUN.length; f++) {
      const lines = renderDog(f, { level: 3 });
      expect(lines.length).toBe(RUN[0]!.length / 2);
      expect(lines.join('')).toMatch(/[▀▄█]/);
      for (const l of lines) expect(visibleWidth(l)).toBe(dogWidth('full', 3, 'run'));
    }
    const sleep = renderDog(0, { mode: 'sleep', level: 2 });
    expect(sleep.length).toBe(SLEEP[0]!.length / 2);
    expect(stripAnsi(sleep.join(''))).toMatch(/[zZ]/);
  });

  test('one-line dog is braille', () => {
    expect(stripAnsi(dogFrame(0, { level: 3 }))).toMatch(/[⠀-⣿]/);
    expect(renderSprite(TINY_RUN[0]!, 1).length).toBe(2);
  });

  test('run header shows a running dog only while tickets run', () => {
    const run = renderRun(view(), { cols: 80, rows: 24, level: 0, frame: 0, project: 'p', now: NOW })[0]!;
    const idle = renderRun(view({ running: 0 }), { cols: 80, rows: 24, level: 0, frame: 0, project: 'p', now: NOW })[0]!;
    expect(run).toContain('(__)>');
    expect(idle).toMatch(/\(__\) ?[zZ]/);
  });

  test('home shows the big dog when there is room', () => {
    const lines = renderHome(PROJECTS, { cols: 80, rows: 30, level: 3, frame: 0, selected: 0, now: NOW });
    expect(lines.some((l) => /[▀▄█]/.test(l))).toBe(true);
    const text = lines.map(stripAnsi).join('\n');
    expect(text).toContain('mugge');
    expect(text).toContain('demo-app');
    expect(text).toContain('enter open');
    expect(text).toContain('q quit');
  });

  test('splash has the wordmark in the middle', () => {
    const text = renderSplash({ cols: 80, rows: 24, level: 3, ms: 500, seed: 9 }).map(stripAnsi).join('\n');
    expect(text).toContain('mugge');
  });
});

describe('run screen content', () => {
  test('summary shows when set', () => {
    const plain = (v: RunView) => renderRun(v, { cols: 100, rows: 40, level: 0, frame: 0, project: 'p', now: NOW }).join('\n');
    const without = plain(view());
    expect(without).not.toContain('speed-up');
    const withS = plain(view({ summary: SUMMARY, running: 0 }));
    expect(withS).toContain('3 done');
    expect(withS).toContain('1 failed');
    expect(withS).toContain('first-try 2/4');
    expect(withS).toContain('peak 3 parallel');
    expect(withS).toContain('3.2x');
    expect(withS).toContain('42k in');
  });

  test('summary survives a short screen', () => {
    const lines = renderRun(view({ summary: SUMMARY, running: 0 }), { cols: 80, rows: 14, level: 0, frame: 0, project: 'p', now: NOW });
    expect(lines.join('\n')).toContain('speed-up');
  });

  test('tickets, vm meter and log tail', () => {
    const lines = renderRun(view(), { cols: 100, rows: 40, level: 0, frame: 0, project: 'proj', vm: { state: 'ready', gpu: 'L4', costUsd: 0.5 } , now: NOW });
    const text = lines.join('\n');
    expect(text).toContain('proj');
    expect(text).toContain('ready');
    expect(text).toContain('L4');
    expect(text).toContain('$0.50');
    expect(text).toContain('1/4');
    expect(text).toContain('bun test failed');
    expect(text).toContain('T4');
    expect(text).toContain('line 29'); // newest log line
    expect(text).not.toContain('line 0 '); // oldest scrolled away
  });

  test('too many tickets collapse into a "more" row', () => {
    const tickets = Array.from({ length: 50 }, (_, i) => ({ id: `T${i}`, title: 't', model: 'm', state: 'waiting' as const, attempt: 0, lastNote: '' }));
    const lines = renderRun(view({ tickets, running: 0 }), { cols: 80, rows: 20, level: 0, frame: 0, project: 'p', now: NOW });
    expect(lines.length).toBeLessThanOrEqual(20);
    expect(lines.join('\n')).toMatch(/\d+ more/);
  });
});
