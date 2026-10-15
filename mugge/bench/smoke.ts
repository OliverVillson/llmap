/**
 * The spike smoke: the 15 public spike tickets through the engine's own write → check → fix
 * loop (runTicket, the same path MuggeBench and the crew take), in about ten minutes against a
 * local llama-server. Run it before exporting any model: Centipede run 1 shipped an agent that
 * scored 0% from three harness bugs a smoke would have caught in minutes.
 *
 *   1. The reference answers must pass the acceptance and hidden tests in this sandbox, 15/15.
 *      If they do not, the sandbox is broken and no model score means anything (exit 2).
 *   2. Every ticket runs once through the loop, each in a fresh copy of the scaffold with its
 *      dependencies' reference answers in place, and is scored on its hidden tests.
 *   3. It prints the pass count, parser drops, forced and cut answers, wall time and a verdict:
 *      MEETS when at least `--bar` of the tickets pass, at most 20% of answers were lost to the
 *      parser or the cap, at most 20% named a file in a shape the parser had to guess (`prog.c`,
 *      `**a.py**`: not the training format), and no model call failed. MISSES otherwise, with
 *      the reasons.
 *
 *   bun bench/smoke.ts --url http://127.0.0.1:8080/v1 --model BH0.1-1.3 --sandbox local
 *   bun bench/smoke.ts --fake                  the fake model on the reference answers (CI)
 *   bun bench/smoke.ts --fake --fake-mess      … plus a cut, a forced, a prose and an empty first answer
 *   options: --out <dir> (default runs/smoke-<time>), --concurrency 12, --sandbox local|podman|docker,
 *            --thinking off|on (default off: BlackHole workers are trained thinking-off),
 *            --max-tokens <per call> (default MUGGE_MAX_CALL_TOKENS, else 4096 off / 8192 on),
 *            --temperature 0.6, --top-k 20 (must be > 0), --top-p 0.95, --min-p <x> (0 for MiniCPM5),
 *            --bar 0.5, --seeds 1, --skip-refs
 *
 * Writes <out>/calls.jsonl (one line per model call), <out>/status.json (crew status, as the
 * console reads it), <out>/results.jsonl and report.json (MuggeBench's), and <out>/smoke.json.
 * Exit code: 0 MEETS, 1 MISSES, 2 could not score (reference answers fail, no server).
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { inferenceFromEnv } from '../src/config.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { checkTopK, OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox, sandboxFromEnv, tail, type Sandbox } from '../src/engine/sandbox.ts';
import { crewFromEnv, emptyCallSummary, formatCallSummary, LOST_WARN_SHARE, runId, statusLine, Telemetry, type CallSummary } from '../src/engine/telemetry.ts';
import type { WorkerOutput } from '../src/tickets/schema.ts';
import { applyHidden, findProjects, prepare, solution, type BenchProject, type BenchTicket } from './load.ts';
import type { Row } from './report.ts';
import { runBench } from './run.ts';

export const SPIKE_DIR = join(import.meta.dir, '..', 'spike');

export interface SmokeOptions {
  fake?: boolean;
  /** Fake model only: a cut, a forced, a prose and an empty first answer on four tickets. */
  fakeMess?: boolean;
  fakeLatencyMs?: number;
  url?: string;
  model?: string;
  out: string;
  concurrency?: number;
  sandbox?: Sandbox;
  thinking?: 'on' | 'off';
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  /** Share of the tickets that must pass (default 0.5). */
  bar?: number;
  seeds?: number;
  skipRefs?: boolean;
  quiet?: boolean;
  /** Bench folder; the spike unless a test points elsewhere. */
  bench?: string;
}

export interface SmokeResult {
  verdict: 'MEETS' | 'MISSES' | 'NO SCORE';
  reasons: string[];
  refs: { passed: number; tickets: number; failures: string[] } | null;
  passed: number;
  tickets: number;
  calls: CallSummary;
  wallMs: number;
  out: string;
}

/** Fake-mess tickets: first answer cut, prose, forced, empty. */
export const MESS = { cut: ['c-hash'], prose: ['py-store'], forced: ['ts-format'], flaky: ['ts-args'] };

/** Reference answers through acceptance and hidden tests, each ticket once. */
export async function checkReferences(projects: BenchProject[], sandbox: Sandbox, concurrency = 8): Promise<{ passed: number; tickets: number; failures: string[] }> {
  const items = projects.flatMap((p) => p.tickets.map((t) => [p, t] as const));
  const failures: string[] = [];
  const work = mkdtempSync(join(tmpdir(), 'mugge-smoke-refs-'));
  let next = 0;
  const one = async (p: BenchProject, t: BenchTicket) => {
    const dir = join(work, `${p.name}-${t.ticket.id}`);
    prepare(p, dir, [...t.deps, t.ticket.id]);
    applyHidden(p, t.ticket.id, dir);
    for (const command of [...t.ticket.acceptance, ...t.tags.hidden]) {
      const r = await sandbox.run(command, { cwd: dir, image: t.ticket.image });
      if (r.code !== 0) return `${t.key}: ${command}\n${tail(r.out, 600)}`;
    }
    return null;
  };
  try {
    await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, items.length)) }, async () => {
      while (next < items.length) {
        const [p, t] = items[next++]!;
        const f = await one(p, t);
        if (f) failures.push(f);
      }
    }));
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
  return { passed: items.length - failures.length, tickets: items.length, failures };
}

async function reachable(url: string): Promise<string | null> {
  try {
    const r = await fetch(`${url.replace(/\/$/, '')}/models`, { signal: AbortSignal.timeout(10_000) });
    return r.ok ? null : `GET /models answered ${r.status}`;
  } catch (e) {
    return String((e as Error).message ?? e);
  }
}

export async function runSmoke(o: SmokeOptions): Promise<SmokeResult> {
  const say = (s: string) => o.quiet || console.log(s);
  const started = performance.now();
  const projects = findProjects(o.bench ?? SPIKE_DIR);
  const tickets = projects.reduce((n, p) => n + p.tickets.length, 0);
  const seeds = o.seeds ?? 1;
  const thinking = o.thinking ?? 'off';
  const bar = o.bar ?? 0.5;
  const sandbox = o.sandbox ?? (o.fake ? new LocalSandbox() : sandboxFromEnv());
  const concurrency = o.concurrency ?? 12;
  const noScore = (reasons: string[], refs: SmokeResult['refs'] = null): SmokeResult => {
    const r: SmokeResult = { verdict: 'NO SCORE', reasons, refs, passed: 0, tickets, calls: emptyCallSummary(), wallMs: Math.round(performance.now() - started), out: o.out };
    mkdirSync(o.out, { recursive: true });
    writeFileSync(join(o.out, 'smoke.json'), JSON.stringify(r, null, 2) + '\n');
    return r;
  };
  // settings the run cannot start with are no score, before any call (not a MISSES of call errors)
  let cfg: ReturnType<typeof inferenceFromEnv>, crew: ReturnType<typeof crewFromEnv>;
  try {
    if (o.topK !== undefined) checkTopK(o.topK, 'top_k (--top-k)');
    cfg = inferenceFromEnv();
    crew = crewFromEnv(process.env, concurrency);
  } catch (e) {
    const why = String((e as Error).message ?? e);
    return noScore([/top_k/i.test(why) ? `${why}: top_k 0 was one of the three bugs that scored run 1's agent 0%` : why]);
  }
  const maxTokens = o.maxTokens ?? cfg.maxCallTokens ?? (thinking === 'on' ? 8192 : 4096);
  const url = o.url ?? cfg.baseUrl;
  if (!o.fake) {
    const down = await reachable(url);
    if (down) return noScore([`no model server at ${url}: ${down}`]);
  }

  say(`smoke: ${tickets} spike tickets × ${seeds} seed${seeds === 1 ? '' : 's'}, ${o.fake ? 'fake model' : `${o.model ?? cfg.defaultModel} at ${url}`}, thinking ${thinking}, ${maxTokens} tokens per call, ${sandbox.name} sandbox`);
  let refs: SmokeResult['refs'] = null;
  if (!o.skipRefs) {
    refs = await checkReferences(projects, sandbox);
    say(`reference answers: ${refs.passed}/${refs.tickets} pass in the sandbox`);
    if (refs.failures.length) {
      for (const f of refs.failures) say('  ' + f.replace(/\n/g, '\n    '));
      return noScore([`${refs.failures.length} reference answers fail in the ${sandbox.name} sandbox: fix the sandbox before scoring a model`], refs);
    }
  }

  const solutions: Record<string, WorkerOutput> = {};
  if (o.fake) for (const p of projects) for (const t of p.tickets) solutions[t.ticket.id] = { files: solution(p, t.ticket.id), note: 'reference' };
  const fake = o.fake ? startFakeModel({ solutions, latencyMs: o.fakeLatencyMs ?? 50, ...(o.fakeMess ? MESS : {}) }) : null;
  const model = o.model ?? (fake ? 'fake' : cfg.defaultModel);
  const telemetry = new Telemetry({
    dir: o.out,
    crew,
    settings: { thinking, maxCallTokens: maxTokens, model },
    warn: o.quiet ? null : undefined,
  });
  const extra = (seed: number): Record<string, unknown> => ({
    seed,
    top_k: o.topK ?? 20,
    ...(o.topP !== undefined ? { top_p: o.topP } : {}),
    ...(o.minP !== undefined ? { min_p: o.minP } : {}),
    chat_template_kwargs: { enable_thinking: thinking === 'on' },
  });
  const ticker = o.quiet ? null : setInterval(() => say(statusLine(telemetry.status())), 30_000);
  (ticker as any)?.unref?.();
  let rows: Row[];
  try {
    rows = await runBench({
      projects, sandbox, seeds, concurrency, out: o.out, name: 'smoke', telemetry, maxTokens,
      inference: (seed) => new OpenAICompatible({
        ...cfg, baseUrl: fake?.url ?? url, defaultModel: model, models: undefined,
        temperature: o.temperature ?? 0.6, maxCallTokens: maxTokens, extra: extra(seed),
      }),
      progress: (r, done, total) => say(`${String(done).padStart(3)}/${total} ${r.solved ? 'pass  ' : r.visible ? 'HIDDEN' : 'fail  '} ${r.key} (${r.attempts} attempt${r.attempts === 1 ? '' : 's'}${r.droppedAnswers ? `, ${r.droppedAnswers} dropped` : ''}${r.cutAnswers ? `, ${r.cutAnswers} cut` : ''}${r.forcedAnswers ? `, ${r.forcedAnswers} forced` : ''})`),
    });
  } finally {
    if (ticker) clearInterval(ticker);
    fake?.stop();
  }
  const status = telemetry.close();
  const calls = status.calls;
  // a ticket passes when every seed solved it; with one seed, when it was solved
  const byKey = new Map<string, boolean>();
  for (const r of rows) byKey.set(r.key, (byKey.get(r.key) ?? true) && r.solved);
  const passed = [...byKey.values()].filter(Boolean).length;
  const reasons: string[] = [];
  if (passed < Math.ceil(bar * tickets)) reasons.push(`${passed}/${tickets} pass, the bar is ${Math.ceil(bar * tickets)} (${Math.round(bar * 100)}%)`);
  if (calls.lostShare > LOST_WARN_SHARE) reasons.push(`${Math.round(calls.lostShare * 100)}% of answers lost to the parser or the cap (over ${Math.round(LOST_WARN_SHARE * 100)}%)`);
  const answers = calls.calls - calls.errors;
  if (answers && calls.loose / answers > LOST_WARN_SHARE) reasons.push(`${Math.round((100 * calls.loose) / answers)}% of answers named a file in a shape the parser had to guess, not the training format (over ${Math.round(LOST_WARN_SHARE * 100)}%)`);
  if (calls.errors) reasons.push(`${calls.errors} model calls failed (timeouts or server errors)`);
  if (rows.length < tickets * seeds) reasons.push(`only ${rows.length} of ${tickets * seeds} ticket runs finished`);
  const result: SmokeResult = { verdict: reasons.length ? 'MISSES' : 'MEETS', reasons, refs, passed, tickets, calls, wallMs: Math.round(performance.now() - started), out: o.out };
  writeFileSync(join(o.out, 'smoke.json'), JSON.stringify(result, null, 2) + '\n');
  return result;
}

export function formatSmoke(r: SmokeResult): string {
  const c = r.calls;
  const drops = Object.entries(c.dropReasons).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(', ');
  const lines = [
    `reference answers  ${r.refs ? `${r.refs.passed}/${r.refs.tickets}` : 'not checked'}`,
    `tickets passing    ${r.passed}/${r.tickets}`,
    `parser drops       ${c.dropped}${drops ? ` (${drops})` : ''} of ${c.calls} answers`,
    `forced / cut       ${c.forced} / ${c.cut}`,
    `guessed paths      ${c.loose}`,
    `call errors        ${c.errors}`,
    `wall time          ${(r.wallMs / 1000).toFixed(1)} s`,
    '',
    `${r.verdict}${r.reasons.length ? `: ${r.reasons.join('; ')}` : ''}`,
  ];
  return lines.join('\n');
}

if (import.meta.main) {
  const { values } = parseArgs({
    options: {
      fake: { type: 'boolean' }, 'fake-mess': { type: 'boolean' }, url: { type: 'string' }, model: { type: 'string' },
      out: { type: 'string' }, concurrency: { type: 'string' }, sandbox: { type: 'string' }, thinking: { type: 'string' },
      'max-tokens': { type: 'string' }, temperature: { type: 'string' }, 'top-k': { type: 'string' }, 'top-p': { type: 'string' },
      'min-p': { type: 'string' }, bar: { type: 'string' }, seeds: { type: 'string' }, 'skip-refs': { type: 'boolean' },
    },
  });
  const num = (v: string | undefined) => (v === undefined ? undefined : Number(v));
  const thinking = values.thinking ?? (process.env.MUGGE_THINKING?.trim().toLowerCase() === 'on' ? 'on' : 'off');
  if (thinking !== 'on' && thinking !== 'off') {
    console.error('--thinking must be on or off');
    process.exit(2);
  }
  const out = values.out ?? join('runs', `smoke-${runId()}`);
  const r = await runSmoke({
    fake: values.fake, fakeMess: values['fake-mess'], url: values.url, model: values.model, out,
    concurrency: num(values.concurrency), thinking,
    sandbox: values.sandbox ? sandboxFromEnv({ MUGGE_SANDBOX: values.sandbox }) : undefined,
    maxTokens: num(values['max-tokens']), temperature: num(values.temperature),
    topK: num(values['top-k']), topP: num(values['top-p']), minP: num(values['min-p']),
    bar: num(values.bar), seeds: num(values.seeds), skipRefs: values['skip-refs'],
  });
  if (r.calls.calls) console.log('\n' + formatCallSummary(r.calls));
  console.log('\n' + formatSmoke(r) + `\n\nlogs: ${out}/calls.jsonl, ${out}/status.json, ${out}/smoke.json`);
  process.exit(r.verdict === 'MEETS' ? 0 : r.verdict === 'MISSES' ? 1 : 2);
}
