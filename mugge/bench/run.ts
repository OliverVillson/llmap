/**
 * Runs MuggeBench: every ticket of every project, `seeds` times, through the engine's own
 * write → check → fix loop (runTicket), each in a fresh copy of the scaffold with its
 * dependencies' reference solutions applied, so one ticket's failure never touches another.
 * After the loop the ticket's hidden tests are copied in and run; a ticket is solved only when
 * its acceptance commands and its hidden tests all pass.
 *
 *   bun bench/run.ts --bench <dir> --url http://127.0.0.1:8000/v1 --model <served name> --out <dir>
 *   options: --seeds 3, --concurrency 16, --sandbox local|podman|docker, --temperature 0.6,
 *            --top-p 0.95, --top-k 20, --thinking on|off, --gpus 1, --name <label>,
 *            --filter <text in project/ticket>, --split dev|test, --transcripts, --fake,
 *            --max-tokens <most tokens one model call may write>, --tickets <n: a quick run on n tickets>,
 *            --review on|off (else MUGGE_REVIEW when it is set; else off: unlike the engine, the bench
 *            scores the worker loop alone by default, so its numbers stay comparable with earlier runs)
 *
 * Writes <out>/results.jsonl (one row per ticket and seed), <out>/report.json and, with
 * --transcripts, <out>/transcripts.jsonl.
 */
import { appendFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { inferenceFromEnv } from '../src/config.ts';
import type { EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { runTicket, thinkingOn } from '../src/engine/harness.ts';
import { checkTopK, OpenAICompatible, type Inference } from '../src/engine/inference.ts';
import { classifyAnswer, crewFromEnv, formatCallSummary, Telemetry } from '../src/engine/telemetry.ts';
import { parseFiles } from '../src/engine/files.ts';
import { LocalSandbox, sandboxFromEnv, tail, type Sandbox } from '../src/engine/sandbox.ts';
import type { WorkerOutput } from '../src/tickets/schema.ts';
import { applyHidden, findProjects, prepare, solution, type BenchProject, type BenchTicket } from './load.ts';
import { formatSummary, summarize, type Row } from './report.ts';
import { reviewFromEnv } from '../src/engine/review.ts';

export interface BenchOptions {
  projects: BenchProject[];
  /** One inference client per seed. */
  inference: (seed: number) => Inference;
  sandbox: Sandbox;
  seeds: number;
  concurrency: number;
  out: string;
  name?: string;
  gpus?: number;
  checkTimeoutMs?: number;
  /** Most tokens one model call may write; the ticket's budget still caps the total. */
  maxTokens?: number;
  transcripts?: boolean;
  /** Lead review after the checks pass; unset: MUGGE_REVIEW when it is set, else off (see the header). */
  review?: boolean;
  /** Called after each row is written. */
  progress?: (row: Row, done: number, total: number) => void;
  /**
   * Per-call log and crew status (calls.jsonl, status.json). Each ticket × seed shows up as
   * ticket `<project>/<id>#s<seed>`, started and finished like a scheduler ticket.
   */
  telemetry?: Telemetry;
}

/** The telemetry ticket id of one bench run of a ticket. */
export const runKey = (key: string, seed: number) => `${key}#s${seed}`;

/** Runs one hidden command list; the first failure, or null. */
async function runHidden(sandbox: Sandbox, t: BenchTicket, worktree: string, timeoutMs?: number): Promise<string | null> {
  for (const command of t.tags.hidden) {
    const r = await sandbox.run(command, { cwd: worktree, image: t.ticket.image, timeoutMs });
    if (r.code !== 0) return `${command}\n${r.timedOut ? '(timed out)\n' : ''}${tail(r.out, 1500)}`;
  }
  return null;
}

export async function runOne(p: BenchProject, t: BenchTicket, seed: number, o: BenchOptions, work: string): Promise<{ row: Row; transcript: unknown[]; answerAttempt: number | null }> {
  const worktree = mkdtempSync(join(work, `${p.name}-${t.ticket.id}-s${seed}-`));
  const transcript: unknown[] = [];
  let calls = 0, empty = 0, refused = 0;
  const base = o.inference(seed);
  const cap = (n: number) => (o.maxTokens ? Math.min(n, o.maxTokens) : n);
  const inference: Inference = {
    complete: (m, msgs, schema, n) => base.complete(m, msgs, schema, cap(n)),
    text: async (m, msgs, n) => {
      const c = await base.text(m, msgs, cap(n));
      if (o.transcripts) {
        // cut, forced and drop mark answers that must never become training rows
        const seen = classifyAnswer(c.raw, t.ticket.owns, parseFiles(c.raw, t.ticket.owns), c.cut);
        transcript.push({ prompt: msgs[msgs.length - 1]?.content, answer: c.raw, cut: !!c.cut, forced: c.forced ?? null, drop: seen.drop ?? null, loose: seen.loose.length > 0, completionTokens: c.completionTokens, reasoningTokens: c.reasoningTokens ?? null });
      }
      return c;
    },
  };
  const id = runKey(t.key, seed);
  const tel = o.telemetry;
  const emit = (e: EngineEvent) => {
    if (e.type === 'model') {
      // which attempt each transcript call was, so harvest can pick the call that is the answer
      const last = transcript.at(-1) as { attempt?: number } | undefined;
      if (last && last.attempt === undefined) last.attempt = e.attempt;
      calls++;
      if (e.files === 0) empty++;
      if (e.cut) cut++;
      if (e.forced) forced++;
      if (e.drop) dropped++;
    } else if (e.type === 'refused') refused++;
    if (tel && 'ticket' in e) tel.emit({ ...e, ticket: id } as EngineEvent);
  };
  let cut = 0, forced = 0, dropped = 0;
  const started = performance.now();
  let finished = false;
  tel?.emit({ type: 'start', at: Date.now(), ticket: id, attempt: 1, worktree });
  try {
    prepare(p, worktree, t.deps);
    const r = await runTicket(t.ticket, worktree, { plan: p.plan, inference, sandbox: o.sandbox, emit, checkTimeoutMs: o.checkTimeoutMs, thinking: thinkingOn(base), review: o.review ?? (process.env.MUGGE_REVIEW ? reviewFromEnv() : false) });
    let hiddenError: string | null = null;
    if (r.ok && t.tags.hidden.length) {
      applyHidden(p, t.ticket.id, worktree);
      hiddenError = await runHidden(o.sandbox, t, worktree, o.checkTimeoutMs);
    }
    const row: Row = {
      project: p.name, ticket: t.ticket.id, key: t.key, seed, split: p.split,
      theme: t.tags.theme, language: t.tags.language, role: t.tags.role,
      visible: r.ok, hidden: r.ok ? hiddenError === null : false, solved: r.ok && hiddenError === null,
      attempts: r.attempts, firstTry: r.firstTry, calls, emptyAnswers: empty, refused, cutAnswers: cut, forcedAnswers: forced, droppedAnswers: dropped,
      promptTokens: r.promptTokens, completionTokens: r.completionTokens,
      inferenceMs: r.inferenceMs, checkMs: r.checkMs, wallMs: Math.round(performance.now() - started),
      error: r.ok ? hiddenError && `hidden: ${hiddenError}` : r.error && tail(r.error, 1500),
    };
    finished = true;
    tel?.emit({ type: 'finish', at: Date.now(), ticket: id, state: row.solved ? 'done' : 'failed', attempts: r.attempts, error: row.error ?? undefined });
    return { row, transcript, answerAttempt: r.answerAttempt };
  } finally {
    if (!finished) tel?.emit({ type: 'finish', at: Date.now(), ticket: id, state: 'failed', attempts: 0, error: 'bench error' });
    rmSync(worktree, { recursive: true, force: true });
  }
}

export async function runBench(o: BenchOptions): Promise<Row[]> {
  mkdirSync(o.out, { recursive: true });
  const resultsPath = join(o.out, 'results.jsonl');
  const transcriptsPath = join(o.out, 'transcripts.jsonl');
  writeFileSync(resultsPath, '');
  if (o.transcripts) writeFileSync(transcriptsPath, '');
  const work = mkdtempSync(join(tmpdir(), 'mugge-bench-'));
  // Seed-major order: every ticket's first seed is in before any second seed, so a run cut
  // short still scores every ticket once.
  const items: Array<[BenchProject, BenchTicket, number]> = [];
  for (let s = 0; s < o.seeds; s++) for (const p of o.projects) for (const t of p.tickets) items.push([p, t, s]);
  o.telemetry?.setTickets(items.map(([, t, s]) => ({ id: runKey(t.key, s), owns: t.ticket.owns, depends_on: [], maxAttempts: t.ticket.budget.attempts * (t.ticket.escalate_to ? 2 : 1) })));
  o.telemetry?.emit({ type: 'plan', at: Date.now(), tickets: [], concurrency: o.concurrency });
  const rows: Row[] = [];
  const started = performance.now();
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const [p, t, s] = items[next++];
      const { row, transcript, answerAttempt } = await runOne(p, t, s, o, work);
      rows.push(row);
      appendFileSync(resultsPath, JSON.stringify(row) + '\n');
      if (o.transcripts) appendFileSync(transcriptsPath, JSON.stringify({ key: row.key, seed: s, ...(answerAttempt !== null && answerAttempt < row.attempts ? { answerAttempt } : {}), calls: transcript }) + '\n');
      o.progress?.(row, rows.length, items.length);
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.max(1, Math.min(o.concurrency, items.length)) }, worker));
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
  const wallMs = Math.round(performance.now() - started);
  const report = summarize(rows, { wallMs, gpus: o.gpus ?? 1, concurrency: o.concurrency, name: o.name });
  writeFileSync(join(o.out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  return rows;
}

/** Every project's reference answers keyed by ticket id, for the fake model. */
function fakeSolutions(projects: BenchProject[]): Record<string, WorkerOutput> {
  const out: Record<string, WorkerOutput> = {};
  for (const p of projects)
    for (const t of p.tickets) {
      if (out[t.ticket.id]) throw new Error(`--fake needs unique ticket ids across projects; ${t.ticket.id} repeats`);
      out[t.ticket.id] = { files: solution(p, t.ticket.id), note: 'reference' };
    }
  return out;
}

/** n tickets taken in turn from each project, in plan order: a quick run that still spans every project. */
export function firstTickets(projects: BenchProject[], n: number): BenchProject[] {
  const keep = new Set<string>();
  for (let i = 0; keep.size < n && projects.some((p) => i < p.tickets.length); i++)
    for (const p of projects) if (i < p.tickets.length && keep.size < n) keep.add(p.tickets[i].key);
  return projects.map((p) => ({ ...p, tickets: p.tickets.filter((t) => keep.has(t.key)) })).filter((p) => p.tickets.length);
}

if (import.meta.main) {
  const { values } = parseArgs({
    options: {
      bench: { type: 'string' }, url: { type: 'string' }, model: { type: 'string' }, out: { type: 'string' },
      seeds: { type: 'string' }, concurrency: { type: 'string' }, sandbox: { type: 'string' },
      temperature: { type: 'string' }, 'top-p': { type: 'string' }, 'top-k': { type: 'string' },
      thinking: { type: 'string' }, gpus: { type: 'string' }, name: { type: 'string' }, filter: { type: 'string' },
      split: { type: 'string' }, transcripts: { type: 'boolean' }, fake: { type: 'boolean' }, timeout: { type: 'string' },
      'max-tokens': { type: 'string' }, tickets: { type: 'string' }, review: { type: 'string' },
    },
  });
  if (values.review && values.review !== 'on' && values.review !== 'off') {
    console.error('--review must be on or off');
    process.exit(2);
  }
  if (!values.bench || !values.out) {
    console.error('usage: bun bench/run.ts --bench <dir> --out <dir> [--url <v1 url> --model <name>] [options]');
    process.exit(2);
  }
  let projects = findProjects(values.bench);
  if (values.split) projects = projects.filter((p) => p.split === values.split);
  if (values.filter) {
    const f = values.filter;
    projects = projects.map((p) => ({ ...p, tickets: p.tickets.filter((t) => t.key.includes(f)) })).filter((p) => p.tickets.length);
  }
  if (values.tickets) projects = firstTickets(projects, Number(values.tickets));
  // refused before any call: a top_k of 0 would otherwise score every ticket 0% on call errors
  if (values['top-k'] !== undefined) {
    try {
      checkTopK(Number(values['top-k']), '--top-k');
    } catch (e) {
      console.error((e as Error).message);
      process.exit(2);
    }
  }
  const fake = values.fake ? startFakeModel({ solutions: fakeSolutions(projects) }) : null;
  const cfg = inferenceFromEnv();
  // --thinking wins; otherwise MUGGE_THINKING (config.ts); otherwise on, the Centipede setting
  const envThinking = (cfg.extra as any)?.chat_template_kwargs?.enable_thinking;
  const thinking = values.thinking ?? (envThinking === false ? 'off' : 'on');
  const extra = (seed: number): Record<string, unknown> => ({
    seed,
    ...(values['top-p'] ? { top_p: Number(values['top-p']) } : {}),
    ...(values['top-k'] ? { top_k: Number(values['top-k']) } : {}),
    chat_template_kwargs: { enable_thinking: thinking !== 'off' },
  });
  const inference = (seed: number) => new OpenAICompatible({
    ...cfg,
    baseUrl: fake?.url ?? values.url ?? cfg.baseUrl,
    defaultModel: values.model ?? (fake ? 'fake' : cfg.defaultModel),
    models: undefined, // every ticket goes to the model under test
    temperature: values.temperature ? Number(values.temperature) : 0.6,
    extra: extra(seed),
  });
  const sandbox = values.sandbox ? sandboxFromEnv({ MUGGE_SANDBOX: values.sandbox }) : fake ? new LocalSandbox() : sandboxFromEnv();
  const tickets = projects.reduce((n, p) => n + p.tickets.length, 0);
  const seeds = Number(values.seeds ?? 3);
  const concurrency = Number(values.concurrency ?? 16);
  const telemetry = new Telemetry({
    dir: values.out, crew: crewFromEnv(process.env, concurrency),
    settings: { thinking: thinking === 'off' ? 'off' : 'on', maxCallTokens: values['max-tokens'] ? Number(values['max-tokens']) : cfg.maxCallTokens ?? null, model: values.model ?? (fake ? 'fake' : cfg.defaultModel) },
  });
  console.log(`MuggeBench: ${tickets} tickets in ${projects.length} projects × ${seeds} seeds, ${sandbox.name} sandbox`);
  try {
    const rows = await runBench({
      projects, inference, sandbox, seeds,
      concurrency, out: values.out, name: values.name, telemetry,
      gpus: Number(values.gpus ?? 1), transcripts: values.transcripts, review: values.review ? values.review === 'on' : undefined,
      maxTokens: values['max-tokens'] ? Number(values['max-tokens']) : undefined,
      checkTimeoutMs: values.timeout ? Number(values.timeout) * 1000 : undefined,
      progress: (r, done, total) => console.log(`${String(done).padStart(5)}/${total} ${r.solved ? 'solved' : r.visible ? 'HIDDEN' : 'failed'} ${r.key} s${r.seed} (${r.attempts} attempt${r.attempts === 1 ? '' : 's'})`),
    });
    console.log('\n' + formatSummary(await Bun.file(join(values.out, 'report.json')).json()));
    const status = telemetry.close();
    console.log('\n' + formatCallSummary(status.calls) + `\n(per call: ${telemetry.callsPath})`);
    process.exit(rows.length ? 0 : 1);
  } finally {
    fake?.stop();
  }
}
