/**
 * MuggeBench scores. The headline is the Mugge score: the share of tickets solved (acceptance
 * and hidden tests pass within the ticket's budget), each ticket weighted once however many
 * seeds ran. A build is held against a reference run (the best full model) as a paired ratio
 * over the tickets both ran, with a bootstrap 95% interval, and gets a verdict against the bar:
 * met (the whole interval is at or above it), missed (the whole interval is below) or too close
 * to call.
 *
 *   bun bench/report.ts <run dir>                       the run's summary
 *   bun bench/report.ts <run dir> --ref <ref run dir>   plus the verdict against the reference
 *   options: --bar 0.9, --json <file>
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

export interface Row {
  project: string;
  ticket: string;
  key: string;
  seed: number;
  split: string;
  theme: string;
  language: string;
  role: string;
  /** Acceptance commands passed within budget. */
  visible: boolean;
  /** Hidden tests passed (false when the loop already failed). */
  hidden: boolean;
  solved: boolean;
  attempts: number;
  firstTry: boolean;
  calls: number;
  /** Answers with no owned file in them (format failures). */
  emptyAnswers: number;
  /** Answers that wrote files outside `owns`. */
  refused: number;
  promptTokens: number;
  completionTokens: number;
  inferenceMs: number;
  checkMs: number;
  wallMs: number;
  error: string | null;
}

export interface Group {
  tickets: number;
  score: number;
}

export interface Summary {
  name?: string;
  tickets: number;
  seeds: number;
  rows: number;
  /** Mugge score, % of tickets solved (mean over seeds). */
  score: number;
  /** Bootstrap 95% interval of the score over tickets. */
  scoreCI: [number, number];
  /** % solved on acceptance alone, as Mugge itself would count it. */
  visibleScore: number;
  /** % of runs that passed acceptance but failed hidden tests. */
  hiddenFailRate: number;
  firstTryRate: number;
  attemptsPerSolve: number | null;
  completionTokensPerSolve: number | null;
  /** % of model answers with no owned file in them. */
  emptyAnswerRate: number;
  /** % of model answers that wrote outside `owns`. */
  refusedRate: number;
  wallMs?: number;
  concurrency?: number;
  gpus?: number;
  /** Solved runs per GPU-hour at this concurrency. */
  solvedPerGpuHour?: number | null;
  byTheme: Record<string, Group>;
  byLanguage: Record<string, Group>;
  byRole: Record<string, Group>;
  byProject: Record<string, Group>;
}

const pct = (a: number, b: number) => (b ? Math.round((1000 * a) / b) / 10 : 0);
const round1 = (x: number) => Math.round(x * 10) / 10;

/** Solved share per ticket key, averaged over seeds. */
export function perTicket(rows: Row[]): Map<string, number> {
  const acc = new Map<string, [number, number]>();
  for (const r of rows) {
    const a = acc.get(r.key) ?? [0, 0];
    a[0] += r.solved ? 1 : 0;
    a[1]++;
    acc.set(r.key, a);
  }
  return new Map([...acc].map(([k, [s, n]]) => [k, s / n]));
}

/** Seeded PRNG (mulberry32), so intervals are the same on every run. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/** Bootstrap over tickets: stat(indices) for `n` resamples, the 2.5% and 97.5% points. */
export function bootstrap(count: number, stat: (idx: number[]) => number, n = 2000, seed = 1): [number, number] {
  if (!count) return [0, 0];
  const r = rng(seed);
  const out: number[] = [];
  for (let b = 0; b < n; b++) {
    const idx = Array.from({ length: count }, () => Math.floor(r() * count));
    const v = stat(idx);
    if (Number.isFinite(v)) out.push(v);
  }
  out.sort((x, y) => x - y);
  return [quantile(out, 0.025), quantile(out, 0.975)];
}

function groups(rows: Row[], field: 'theme' | 'language' | 'role' | 'project'): Record<string, Group> {
  const by = new Map<string, Row[]>();
  for (const r of rows) by.set(r[field], [...(by.get(r[field]) ?? []), r]);
  const out: Record<string, Group> = {};
  for (const [k, rs] of [...by].sort((a, b) => a[0].localeCompare(b[0]))) {
    const t = [...perTicket(rs).values()];
    out[k] = { tickets: t.length, score: round1((100 * t.reduce((a, b) => a + b, 0)) / Math.max(1, t.length)) };
  }
  return out;
}

export function summarize(rows: Row[], o: { wallMs?: number; gpus?: number; concurrency?: number; name?: string } = {}): Summary {
  const t = [...perTicket(rows).values()];
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  const solved = rows.filter((r) => r.solved);
  const visible = rows.filter((r) => r.visible);
  const calls = rows.reduce((n, r) => n + r.calls, 0);
  const visibleByTicket = new Map<string, number[]>();
  for (const r of rows) visibleByTicket.set(r.key, [...(visibleByTicket.get(r.key) ?? []), r.visible ? 1 : 0]);
  const [lo, hi] = bootstrap(t.length, (idx) => mean(idx.map((i) => t[i])));
  const hours = o.wallMs ? o.wallMs / 3_600_000 : 0;
  return {
    ...(o.name ? { name: o.name } : {}),
    tickets: t.length,
    seeds: Math.max(0, ...rows.map((r) => r.seed + 1)),
    rows: rows.length,
    score: round1(100 * mean(t)),
    scoreCI: [round1(100 * lo), round1(100 * hi)],
    visibleScore: round1(100 * mean([...visibleByTicket.values()].map(mean))),
    hiddenFailRate: pct(visible.filter((r) => !r.hidden).length, visible.length),
    firstTryRate: pct(rows.filter((r) => r.firstTry && r.solved).length, rows.length),
    attemptsPerSolve: solved.length ? Math.round((100 * solved.reduce((n, r) => n + r.attempts, 0)) / solved.length) / 100 : null,
    completionTokensPerSolve: solved.length ? Math.round(rows.reduce((n, r) => n + r.completionTokens, 0) / solved.length) : null,
    emptyAnswerRate: pct(rows.reduce((n, r) => n + r.emptyAnswers, 0), calls),
    refusedRate: pct(rows.reduce((n, r) => n + r.refused, 0), calls),
    ...(o.wallMs !== undefined
      ? { wallMs: o.wallMs, concurrency: o.concurrency, gpus: o.gpus ?? 1, solvedPerGpuHour: hours ? round1(solved.length / (hours * (o.gpus ?? 1))) : null }
      : {}),
    byTheme: groups(rows, 'theme'),
    byLanguage: groups(rows, 'language'),
    byRole: groups(rows, 'role'),
    byProject: groups(rows, 'project'),
  };
}

export interface Comparison {
  tickets: number;
  build: number;
  ref: number;
  /** Build score as a share of the reference's, and its 95% interval. */
  ratio: number;
  ratioCI: [number, number];
  /** Build minus reference, in points, and its 95% interval. */
  diff: number;
  diffCI: [number, number];
  bar: number;
  verdict: 'met' | 'missed' | 'too close to call';
  byTheme: Record<string, { build: number; ref: number; ratio: number | null }>;
}

/** Paired comparison over the tickets both runs scored. */
export function compare(build: Row[], ref: Row[], bar = 0.9): Comparison {
  const b = perTicket(build), r = perTicket(ref);
  const keys = [...b.keys()].filter((k) => r.has(k)).sort();
  const bs = keys.map((k) => b.get(k)!), rs = keys.map((k) => r.get(k)!);
  const sum = (xs: number[], idx: number[]) => idx.reduce((a, i) => a + xs[i], 0);
  const all = keys.map((_, i) => i);
  const ratio = sum(bs, all) / Math.max(1e-9, sum(rs, all));
  const ratioCI = bootstrap(keys.length, (idx) => sum(bs, idx) / sum(rs, idx));
  const diffCI = bootstrap(keys.length, (idx) => (100 * (sum(bs, idx) - sum(rs, idx))) / idx.length);
  const verdict = ratioCI[0] >= bar ? 'met' : ratioCI[1] < bar ? 'missed' : 'too close to call';
  const themeOf = new Map(build.map((x) => [x.key, x.theme]));
  const byTheme: Comparison['byTheme'] = {};
  for (const theme of [...new Set(keys.map((k) => themeOf.get(k)!))].sort()) {
    const idx = all.filter((i) => themeOf.get(keys[i]) === theme);
    const sb = sum(bs, idx), sr = sum(rs, idx);
    byTheme[theme] = { build: round1((100 * sb) / idx.length), ref: round1((100 * sr) / idx.length), ratio: sr ? Math.round((1000 * sb) / sr) / 1000 : null };
  }
  return {
    tickets: keys.length,
    build: round1((100 * sum(bs, all)) / Math.max(1, keys.length)),
    ref: round1((100 * sum(rs, all)) / Math.max(1, keys.length)),
    ratio: Math.round(ratio * 1000) / 1000,
    ratioCI: [Math.round(ratioCI[0] * 1000) / 1000, Math.round(ratioCI[1] * 1000) / 1000],
    diff: round1((100 * (sum(bs, all) - sum(rs, all))) / Math.max(1, keys.length)),
    diffCI: [round1(diffCI[0]), round1(diffCI[1])],
    bar,
    verdict,
    byTheme,
  };
}

export function readRows(dir: string): Row[] {
  return readFileSync(join(dir, 'results.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

function table(title: string, g: Record<string, Group>): string[] {
  const w = Math.max(...Object.keys(g).map((k) => k.length), 4);
  return [title, ...Object.entries(g).map(([k, v]) => `  ${k.padEnd(w)}  ${String(v.score).padStart(5)}%  (${v.tickets} tickets)`)];
}

export function formatSummary(s: Summary): string {
  const rows: Array<[string, string]> = [
    ['Mugge score', `${s.score}%  (95% ${s.scoreCI[0]}–${s.scoreCI[1]}), ${s.tickets} tickets × ${s.seeds} seeds`],
    ['passing acceptance only', `${s.visibleScore}%`],
    ['passed acceptance, failed hidden', `${s.hiddenFailRate}% of passes`],
    ['solved on the first call', `${s.firstTryRate}%`],
    ['attempts per solve', String(s.attemptsPerSolve ?? '-')],
    ['completion tokens per solve', String(s.completionTokensPerSolve ?? '-')],
    ['answers with no owned file', `${s.emptyAnswerRate}%`],
    ['answers writing outside owns', `${s.refusedRate}%`],
  ];
  if (s.wallMs !== undefined) {
    rows.push(['solved per GPU-hour', `${s.solvedPerGpuHour ?? '-'} (concurrency ${s.concurrency}, ${s.gpus} GPU)`]);
    rows.push(['wall clock', `${(s.wallMs / 60000).toFixed(1)} min`]);
  }
  const w = Math.max(...rows.map((r) => r[0].length));
  return [
    ...(s.name ? [s.name] : []),
    ...rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`),
    '',
    ...table('by theme', s.byTheme),
    ...table('by language', s.byLanguage),
    ...table('by role', s.byRole),
  ].join('\n');
}

export function formatComparison(c: Comparison): string {
  return [
    `against the reference on ${c.tickets} shared tickets: ${c.build}% vs ${c.ref}%`,
    `  share of reference ${(100 * c.ratio).toFixed(1)}%  (95% ${(100 * c.ratioCI[0]).toFixed(1)}–${(100 * c.ratioCI[1]).toFixed(1)}%)`,
    `  difference ${c.diff} pts  (95% ${c.diffCI[0]} to ${c.diffCI[1]})`,
    `  bar ${(100 * c.bar).toFixed(0)}% of reference: ${c.verdict.toUpperCase()}`,
    ...Object.entries(c.byTheme).map(([t, v]) => `  ${t}: ${v.build}% vs ${v.ref}%${v.ratio === null ? '' : ` (${(100 * v.ratio).toFixed(0)}%)`}`),
  ].join('\n');
}

if (import.meta.main) {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: { ref: { type: 'string' }, bar: { type: 'string' }, json: { type: 'string' } } });
  if (!positionals[0]) {
    console.error('usage: bun bench/report.ts <run dir> [--ref <ref run dir>] [--bar 0.9] [--json out.json]');
    process.exit(2);
  }
  const rows = readRows(positionals[0]);
  let saved: Partial<Summary> = {};
  try {
    saved = JSON.parse(readFileSync(join(positionals[0], 'report.json'), 'utf8'));
  } catch {}
  const s = summarize(rows, { wallMs: saved.wallMs, gpus: saved.gpus, concurrency: saved.concurrency, name: saved.name });
  console.log(formatSummary(s));
  let c: Comparison | null = null;
  if (values.ref) {
    c = compare(rows, readRows(values.ref), values.bar ? Number(values.bar) : 0.9);
    console.log('\n' + formatComparison(c));
  }
  if (values.json) writeFileSync(values.json, JSON.stringify({ summary: s, comparison: c }, null, 2) + '\n');
}
