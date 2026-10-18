#!/usr/bin/env bun
/**
 * The Centipede demo: an orchestrator and a crew of code agents build a project at the same
 * time, and a live view in the browser shows the crew while they work.
 *
 *   bun demo/run.ts --prompt "a CLI that …" [--repo DIR]
 *                                        your task: the orchestrator designs, scaffolds (or reads
 *                                        DIR) and plans it, about one ticket per agent, then the
 *                                        crew builds it (the 64-agent first run, demo/README.md)
 *   bun demo/run.ts                      the built-in task (mdlite, demo/project), for 10 agents
 *   bun demo/run.ts --plan orchestrator  mdlite, with the orchestrator writing the ticket plan
 *   bun demo/run.ts --fake [--prompt …]  the same against a mock model, no GPU (a rehearsal); with
 *                                        --prompt the mock plans one module per agent
 *
 * Options: --port N (7340), --host H (127.0.0.1), --out DIR (demo/runs/<run id>),
 * --concurrency N (MUGGE_CONCURRENCY, else 10), --exit (stop once the run ends; by default the
 * view stays up until Ctrl-C), --checks N (acceptance runs at once, default this machine's CPU
 * count), --fake-speed X (mock calls X times faster, default 1).
 *
 * Everything lands in the run folder: repo/ (the target project, branches mugge/<ticket> and
 * mugge/integration), result/ (the integrated project, its test run and README.html), events.jsonl
 * (every engine event), prompts.jsonl (every model call: the messages as sent and the answer),
 * calls.jsonl and status.json (the engine's per-call log and crew status), steer.jsonl and
 * summary.json. The engine is used as it is: this file only drives it and watches it.
 */
import { appendFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, normalize, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { checkedPoolFromEnv, inferenceFromEnv } from '../src/config.ts';
import type { BackendPool, CallOptions } from '../src/engine/backends.ts';
import type { EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { git, headCommit, initRepo, INTEGRATION_BRANCH } from '../src/engine/git.ts';
import type { CallTweaks, ChatMessage, Completion, Inference } from '../src/engine/inference.ts';
import { REVIEW_SYSTEM } from '../src/engine/prompts.ts';
import { LocalSandbox, sandboxFromEnv, type Sandbox } from '../src/engine/sandbox.ts';
import { availableParallelism } from 'node:os';
import { runPlan, type RunReport } from '../src/engine/scheduler.ts';
import { backendLabel, crewFromEnv, formatCallSummary, runId, Telemetry, type CrewLayout } from '../src/engine/telemetry.ts';
import { architectureMarkdown, design, scaffold, tickets, type PlannedTickets, type PlanOptions } from '../src/plan/architect.ts';
import { fakeProject, type FakeProject } from './fake-prompt.ts';
import { validatePlan, type Plan } from '../src/tickets/schema.ts';
import { DEMO_DIR, loadPlan, PLAN_PATH, PROJECT_DIR, referenceAnswers, REQUEST } from './lib.ts';
import { orchestratorPlan } from './orchestrator.ts';

const { values: opt } = parseArgs({
  options: {
    fake: { type: 'boolean' }, 'fake-speed': { type: 'string' },
    plan: { type: 'string' }, prompt: { type: 'string' }, repo: { type: 'string' },
    port: { type: 'string' }, host: { type: 'string' }, out: { type: 'string' },
    concurrency: { type: 'string' }, checks: { type: 'string' }, exit: { type: 'boolean' }, help: { type: 'boolean', short: 'h' },
  },
});
if (opt.help) {
  console.log(readFileSync(import.meta.path, 'utf8').split('*/')[0]!.replace(/^#!.*\n\/\*\*\n?/, '').replace(/^ \* ?/gm, ''));
  process.exit(0);
}
const die = (msg: string): never => {
  console.error(`demo: ${msg}`);
  process.exit(2);
};
if (opt.plan && !['prepared', 'orchestrator'].includes(opt.plan)) die('--plan is prepared or orchestrator');
if (opt.repo && !opt.prompt) die('--repo needs --prompt: say what the crew should do in that repo');
if (opt.prompt && opt.plan === 'prepared') die('--prompt is planned by the orchestrator; --plan prepared is the built-in task only');

const concurrency = Number(opt.concurrency ?? process.env.MUGGE_CONCURRENCY ?? 10);
if (!Number.isInteger(concurrency) || concurrency < 1) die('--concurrency must be a positive integer');
// every agent's checks run on this machine's CPUs: at most this many at once, the rest wait their turn
const checkSlots = Number(opt.checks ?? availableParallelism());
if (!Number.isInteger(checkSlots) || checkSlots < 1) die('--checks must be a positive integer');
const port = Number(opt.port ?? 7340);
const host = opt.host ?? '127.0.0.1';
const id = runId();
const out = resolve(opt.out ?? join(DEMO_DIR, 'runs', id));
const repo = join(out, 'repo');
const task = opt.prompt ? (opt.repo ? 'your repo' : 'your prompt') : 'mdlite';
const planMode: 'prepared' | 'orchestrator' = opt.prompt ? 'orchestrator' : ((opt.plan as 'prepared' | 'orchestrator' | undefined) ?? 'prepared');
mkdirSync(out, { recursive: true });
if (existsSync(repo)) die(`${repo} exists already; pick another --out`);

// ---------------------------------------------------------------------------------------------
// The model: the mock (a rehearsal with the reference answers) or the crew's machines

let fakeSpeed = 1;
/** How long the mock server holds each call, so the call has its slot for most of its pace. */
const FAKE_HOLD_MS = () => Math.round(20_000 / fakeSpeed);
let stopFake: (() => void) | null = null;
let fakeOwn: FakeProject | null = null;
if (opt.fake) {
  if (opt.repo) die('--fake cannot read your repo: it answers with a project of its own (drop --repo)');
  fakeSpeed = Number(opt['fake-speed'] ?? 1);
  if (!(fakeSpeed > 0)) die('--fake-speed must be a positive number');
  // with --prompt the mock orchestrator plans a project sized to the crew, one module per agent
  fakeOwn = opt.prompt ? fakeProject(concurrency) : null;
  const [flakyId, sentBackId] = fakeOwn ? [fakeOwn.plan.tickets[2]?.id ?? '', fakeOwn.plan.tickets[4]?.id ?? ''] : ['block-list', 'block-quote'];
  const fake = startFakeModel({
    solutions: fakeOwn?.solutions ?? referenceAnswers(loadPlan()),
    // a first answer with no file (the fix loop runs) and one send-back by the lead
    flaky: [flakyId],
    // most of a call's pace is spent inside the server, so the pool shows the slot busy as with vLLM
    latencyMs: FAKE_HOLD_MS(),
    // a review answers a short verdict, so the mock makes it quicker than a write (with the lead's
    // thinking on, as crew.env sets it, a real review can take as long as a write)
    reviewLatencyMs: Math.round(FAKE_HOLD_MS() / 5),
    reviews: { [sentBackId]: fakeOwn ? [{ verdict: 'problems', problems: ['convert() does not round to 6 decimals as the docstring says.'], fix: 'Round the result to 6 decimals.' }, { verdict: 'approve', problems: [], fix: '' }] : [{ verdict: 'problems', problems: ['A quote line with only ">" and spaces must end the paragraph, not join it.'], fix: 'Treat a quote line that is blank after removing ">" as a paragraph break.' }, { verdict: 'approve', problems: [], fix: '' }] },
  });
  stopFake = fake.stop;
  // the crew's shape: one server, separate slots for workers, leads and the orchestrator. With
  // crew.env loaded the rehearsal takes its slot counts (64 workers, 8 leads at CREW_WORKERS=64),
  // else one worker slot per agent and one lead per 8 workers, as deploy/crew.py sizes them.
  const slotsFor = (role: string, otherwise: number) => {
    try {
      const list = JSON.parse(process.env.MUGGE_BACKENDS ?? '') as Array<{ roles?: string[]; slots?: number }>;
      return list.filter((m) => m.roles?.includes(role)).reduce((n, m) => n + (m.slots ?? 0), 0) || otherwise;
    } catch {
      return otherwise;
    }
  };
  const workers = slotsFor('worker', concurrency);
  process.env.MUGGE_BACKENDS = JSON.stringify([
    { name: 'mock-workers', url: fake.url, roles: ['worker'], slots: workers },
    { name: 'mock-leads', url: fake.url, roles: ['lead'], slots: slotsFor('lead', Math.max(2, Math.ceil(workers / 8))) },
    { name: 'mock-orchestrator', url: fake.url, roles: ['manager'], slots: 1 },
  ]);
  process.env.MUGGE_MODEL ||= 'mock';
  process.env.MUGGE_THINKING ||= 'on';
}
// The built-in task is plain Python with the standard library: no container needed.
process.env.MUGGE_SANDBOX ||= 'local';

const say = (line: string) => console.log(line);
const pool0 = await checkedPoolFromEnv(process.env, concurrency, say, planMode === 'orchestrator' ? ['worker', 'manager'] : ['worker']).catch((e) => die(String((e as Error).message ?? e)));
const cfg = inferenceFromEnv();

// ---------------------------------------------------------------------------------------------
// What the view reads: every model call as sent, events, a digest per ticket, the orchestrator

type Role = 'worker' | 'lead' | 'manager';
interface CallLog {
  id: number;
  /** The ticket, or "orchestrator" for planning calls. */
  ticket: string;
  role: Role;
  kind: 'write' | 'fix' | 'review' | 'plan';
  attempt: number;
  sentAt: number;
  endAt: number | null;
  maxTokens: number;
  messages: ChatMessage[];
  answer: string | null;
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number | null;
  reasoningChars: number | null;
  /** The thinking text, when the server returned any. */
  reasoning: string | null;
  ms: number | null;
  cut: boolean;
  thinking: boolean | null;
  error: string | null;
}

interface TicketDigest {
  state: string;
  attempt: number;
  lastCheck: { attempt: number; command: string; ok: boolean; tail: string; at: number } | null;
  lastAnswer: { attempt: number; kind: string; note: string; files: number; completionTokens: number; reasoningTokens: number | null; cut: boolean; drop: string | null; at: number } | null;
  reviews: Array<{ attempt: number; verdict: string; problems: string[]; fix: string; at: number }>;
  error: string | null;
  commit: string | null;
  checks: { passed: number; failed: number };
  /** While queued: 'lead' when the attempt's checks passed and it waits for a lead slot to be reviewed, else 'write'. */
  queuedFor: 'write' | 'lead' | null;
}

const calls: CallLog[] = [];
const events: EngineEvent[] = [];
const digests = new Map<string, TicketDigest>();
const steers: Array<{ ticket: string; note: string; when: 'next' | 'now'; at: number }> = [];
const orchestrator = {
  phase: 'starting' as 'starting' | 'designing' | 'scaffolding' | 'planning' | 'dispatching' | 'integrating' | 'finished' | 'stopped' | 'failed',
  planSource: planMode as 'prepared' | 'orchestrator',
  planNote: '',
  rejected: [] as string[],
  reviews: { approve: 0, problems: 0, error: 0 },
  reviewing: [] as string[],
};
let plan: Plan | null = null;
let result: null | { integrated: boolean | null; merged: string[]; failedChecks: string[]; tests: string | null; testsOk: boolean | null; readme: boolean; summary: RunReport['summary'] } = null;
const listeners = new Set<(chunk: string) => void>();
const broadcast = (type: string, data: unknown) => {
  const chunk = `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const l of listeners) l(chunk);
};

const callKind = (messages: ChatMessage[]): CallLog['kind'] =>
  messages[0]?.content === REVIEW_SYSTEM ? 'review' : /\n## Fix\nThis command failed:/.test(messages.at(-1)?.content ?? '') ? 'fix' : 'write';

/** A call without its messages, thinking and answer (a review's short verdict stays): what lists and the stream carry. */
const brief = (c: CallLog) => ({ ...c, messages: undefined, answer: c.kind === 'review' && (c.answer?.length ?? 0) <= 4000 ? c.answer : undefined, reasoning: undefined, answerChars: c.answer?.length ?? 0 });

/** Records every call an agent's inference makes; in a rehearsal also gives the mock a model's pace. */
function recorded(inf: Inference, o: CallOptions): Inference {
  const role: Role = o.role ?? 'worker';
  const log = async <T,>(messages: ChatMessage[], maxTokens: number, kind: CallLog['kind'] | null, call: () => Promise<Completion<T>>, schema?: object): Promise<Completion<T>> => {
    const ticket = o.agent ?? 'orchestrator';
    const rec: CallLog = {
      id: calls.length + 1, ticket, role, kind: kind ?? callKind(messages), attempt: digests.get(ticket)?.attempt ?? 0,
      sentAt: Date.now(), endAt: null, maxTokens, messages, answer: null, promptTokens: 0, completionTokens: 0,
      reasoningTokens: null, reasoningChars: null, reasoning: null, ms: null, cut: false, thinking: null, error: null,
    };
    calls.push(rec);
    broadcast('call', brief(rec));
    try {
      const c = await call();
      if (opt.fake) {
        // the mock has no orchestrator of its own: it answers the built-in task's plan call with the
        // prepared plan, and --prompt's design, scaffold and ticket calls (and their parts) with the
        // crew-sized project
        if (rec.kind === 'plan') {
          const value = fakeOwn ? fakeOwn.answer(schema, messages) : JSON.parse(readFileSync(PLAN_PATH, 'utf8'));
          (c as Completion<unknown>).value = value;
          c.raw = JSON.stringify(value);
          c.completionTokens = Math.round(c.raw.length / 3.5);
        }
        // a model's pace: about 60 tok/s per call, some thinking first
        const think = role === 'worker' ? 300 + Math.round(Math.random() * 900) : 120;
        const ms = Math.round(((c.completionTokens + think) / 60) * 1000 * (0.7 + Math.random() * 0.6) / fakeSpeed);
        // the server already held the call for FAKE_HOLD_MS (inside c.ms); wait out the rest of its pace
        const rest = Math.max(0, ms - FAKE_HOLD_MS());
        await Bun.sleep(rest);
        c.ms += rest;
        c.reasoningTokens = think;
        c.completionTokens += think;
        if (role === 'worker') c.reasoning = `(mock model: no real thinking; a real run shows the model's own here, about ${think} tokens of it)`;
      }
      Object.assign(rec, { endAt: Date.now(), answer: typeof c.value === 'string' ? c.value : c.raw, reasoning: c.reasoning ?? null, promptTokens: c.promptTokens, completionTokens: c.completionTokens, reasoningTokens: c.reasoningTokens ?? null, reasoningChars: c.reasoningChars ?? null, ms: c.ms, cut: !!c.cut, thinking: c.thinking ?? null });
      return c;
    } catch (e) {
      const got = (e as { completion?: Completion<string> }).completion;
      Object.assign(rec, { endAt: Date.now(), error: String((e as Error).message ?? e).slice(0, 2000), answer: got?.raw ?? null, completionTokens: got?.completionTokens ?? 0, promptTokens: got?.promptTokens ?? 0 });
      throw e;
    } finally {
      appendFileSync(join(out, 'prompts.jsonl'), JSON.stringify(rec) + '\n');
      broadcast('call', brief(rec));
    }
  };
  return {
    text: (model, messages, maxTokens, signal, tweaks) => log(messages, maxTokens, null, () => inf.text(model, messages, maxTokens, signal, tweaks)),
    complete: <T,>(model: string, messages: ChatMessage[], schema: object, maxTokens: number, signal?: AbortSignal, tweaks?: CallTweaks) =>
      log<T>(messages, maxTokens, messages[0]?.content === REVIEW_SYSTEM ? 'review' : 'plan', () => inf.complete<T>(model, messages, schema, maxTokens, signal, tweaks), schema),
  };
}

/** A sandbox that runs at most n commands at once; the rest wait in order (the timeout starts when one runs). */
function limited(sb: Sandbox, n: number): Sandbox {
  let running = 0;
  const waiting: Array<() => void> = [];
  return {
    name: sb.name,
    async run(command, o) {
      if (running >= n) await new Promise<void>((go) => waiting.push(go));
      else running++;
      try {
        return await sb.run(command, o);
      } finally {
        const next = waiting.shift();
        if (next) next();
        else running--;
      }
    },
  };
}

/** The pool as the engine sees it, with every agent's calls recorded. */
const pool = new Proxy(pool0, {
  get(target, prop) {
    if (prop === 'forAgent') return (o: CallOptions = {}) => recorded(target.forAgent(o), o);
    const v = Reflect.get(target, prop, target);
    return typeof v === 'function' ? v.bind(target) : v;
  },
}) as BackendPool;

// ---------------------------------------------------------------------------------------------
// The crew as the status stream shows it: one team of `concurrency` code agents

const machineFor = (role: Role) => pool0.machines.find((m) => m.machine.roles.includes(role))?.machine;
// the engine's label names llama-server for any local address; say which server the crew file names
const serverLabel = (m: { url: string; server: string }) => (opt.fake ? 'local · mock model' : backendLabel(m.url).replace('llama-server', m.server));
const crew: CrewLayout = process.env.MUGGE_CREW?.trim() ? crewFromEnv(process.env, concurrency) : {
  machines: pool0.machines.map((m) => ({ id: m.machine.name, slots: m.machine.slots, backend: serverLabel(m.machine), url: m.machine.url })),
  teams: [{ id: 'a', domain: task, machine: machineFor('worker')!.name, agents: concurrency }],
  manager: machineFor('manager') ? { machine: machineFor('manager')!.name } : undefined,
};
const thinking = (cfg.extra?.chat_template_kwargs as { enable_thinking?: boolean } | undefined)?.enable_thinking;
const telemetry = new Telemetry({
  run: id, dir: out, crew,
  settings: { thinking: thinking === undefined ? 'server' : thinking ? 'on' : 'off', maxCallTokens: cfg.maxCallTokens ?? null, model: machineFor('worker')?.model ?? cfg.defaultModel },
});
pool0.subscribe((s) => telemetry.serving(s));

function digest(t: string): TicketDigest {
  let d = digests.get(t);
  if (!d) digests.set(t, (d = { state: 'waiting', attempt: 0, lastCheck: null, lastAnswer: null, reviews: [], error: null, commit: null, checks: { passed: 0, failed: 0 }, queuedFor: null }));
  return d;
}

/** A line in the terminal (demo.log under nohup), with the time: what someone over ssh follows with tail -f. */
const clock = () => new Date().toTimeString().slice(0, 8);
const tell = (line: string) => console.log(`${clock()} ${line}`);
const firstLine = (s: string | undefined) => (s ?? '').split('\n').find((l) => l.trim())?.trim().slice(0, 200) ?? '';

const emit = (e: EngineEvent) => {
  events.push(e);
  appendFileSync(join(out, 'events.jsonl'), JSON.stringify(e) + '\n');
  try {
    telemetry.emit(e);
  } catch (err) {
    console.error(`telemetry: ${(err as Error).message ?? err}`);
  }
  if ('ticket' in e && typeof e.ticket === 'string') {
    const d = digest(e.ticket);
    switch (e.type) {
      case 'state':
        d.state = e.state;
        if (e.attempt) d.attempt = e.attempt;
        // a slot wait after this attempt's checks passed is the review's call waiting for a lead
        d.queuedFor = e.state !== 'queued' ? null : d.lastCheck?.ok && d.lastCheck.attempt === d.attempt && !d.reviews.some((r) => r.attempt === d.attempt) ? 'lead' : 'write';
        // a review goes reviewing, queued (for a lead slot), reviewing again: it stays on the list until it ends
        if (e.state === 'reviewing') {
          if (!orchestrator.reviewing.includes(e.ticket)) orchestrator.reviewing.push(e.ticket);
        } else if (e.state !== 'queued') orchestrator.reviewing = orchestrator.reviewing.filter((x) => x !== e.ticket);
        break;
      case 'check':
        d.lastCheck = { attempt: e.attempt, command: e.command, ok: e.ok, tail: e.tail, at: e.at };
        d.checks[e.ok ? 'passed' : 'failed']++;
        break;
      case 'model':
        d.lastAnswer = { attempt: e.attempt, kind: e.kind, note: e.note, files: e.files ?? 0, completionTokens: e.completionTokens, reasoningTokens: e.reasoningTokens ?? null, cut: !!e.cut, drop: e.drop ?? null, at: e.at };
        break;
      case 'review':
        d.reviews.push({ attempt: e.attempt, verdict: e.verdict, problems: e.problems, fix: e.fix, at: e.at });
        orchestrator.reviews[e.verdict]++;
        if (e.verdict === 'problems') tell(`lead sends ${e.ticket} back: ${firstLine(e.problems[0] ?? e.fix)}`);
        else if (e.verdict === 'error') tell(`lead review of ${e.ticket} failed: ${firstLine(e.error)}`);
        break;
      case 'call_error':
        d.error = e.error;
        tell(`${e.ticket}: a model call failed: ${firstLine(e.error)}`);
        break;
      case 'finish':
        d.state = e.state === 'done' ? 'finished' : e.state;
        d.error = e.error ?? null;
        d.commit = e.commit ?? null;
        tell(e.state === 'done' ? `${e.ticket} done${e.attempts > 1 ? ` (attempt ${e.attempts})` : ''}` : `${e.ticket} ${e.state}${e.attempts ? ` after ${e.attempts} attempt${e.attempts > 1 ? 's' : ''}` : ''}${e.error ? `: ${firstLine(e.error)}` : ''}`);
        if (plan && [...digests.values()].filter((x) => ['finished', 'failed', 'skipped', 'blocked'].includes(x.state)).length >= plan.tickets.length) orchestrator.phase = 'integrating';
        break;
    }
  }
  if (e.type === 'integrate') tell(`merged ${e.merged.length} tickets into ${e.branch}; the checks after merging ${e.ok ? 'pass' : `FAIL${e.failed.length ? ` for ${e.failed.join(', ')}` : ''}${e.error ? `: ${firstLine(e.error)}` : ''}`}`);
  broadcast('engine', e);
};

const log = (message: string, level: 'info' | 'warn' | 'error' = 'info') => {
  tell(message);
  emit({ type: 'log', at: Date.now(), level, message });
};

// ---------------------------------------------------------------------------------------------
// The live view: GET / (the page), /api/status, /api/stream (SSE), /api/calls?ticket= (the list), /api/call?id= (one call in full), POST /api/steer

const VIEW = join(DEMO_DIR, 'view', 'index.html');
/** Most the integrated project's whole test run may take at the end. */
const RESULT_TESTS_TIMEOUT_MS = 600_000;
/** Frames a view's stream may have unsent before even engine events are skipped for it. */
const BEHIND = 200;

function status() {
  return {
    status: telemetry.status(),
    demo: {
      run: id, task, fake: !!opt.fake, out, now: Date.now(),
      request: opt.prompt ?? REQUEST,
      orchestrator: { ...orchestrator, machine: crew.manager?.machine ?? null, leadMachine: machineFor('lead')?.name ?? null, calls: calls.filter((c) => c.role !== 'worker').length },
      plan: plan && {
        source: orchestrator.planSource,
        tickets: plan.tickets.map((t) => ({ id: t.id, title: t.title, owns: t.owns, depends_on: t.depends_on, acceptance: t.acceptance, attempts: t.budget.attempts })),
        settled: plan.settled ?? [],
      },
      tickets: Object.fromEntries(digests),
      inflight: calls.filter((c) => c.endAt === null).map(brief),
      steers,
      result,
      machines: pool0.machines.map((m) => ({ name: m.machine.name, roles: m.machine.roles, slots: m.machine.slots, model: m.machine.model ?? cfg.defaultModel })),
    },
  };
}

function resultFile(path: string): Response {
  const base = join(out, 'result');
  const p = normalize(join(base, path));
  if (!p.startsWith(base + '/') || !existsSync(p)) return new Response('not found', { status: 404 });
  return new Response(Bun.file(p));
}

// A view still up from an earlier run (it stays up after a run ends) holds the port: say how to free it
function serve(opts: Parameters<typeof Bun.serve>[0]) {
  try {
    return Bun.serve(opts);
  } catch (e) {
    if (!/EADDRINUSE|in use/i.test(String((e as Error)?.message ?? e))) throw e;
    console.error(`demo: port ${port} is in use, probably by an earlier demo run's view. Stop it with: pkill -INT -f 'bun demo/run.ts'  (or pass --port ${port + 1})`);
    process.exit(1);
  }
}

const server = serve({
  port, hostname: host, idleTimeout: 0,
  async fetch(req) {
    const url = new URL(req.url);
    const p = url.pathname;
    if (p === '/' || p === '/index.html') return new Response(Bun.file(VIEW), { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' } });
    if (p === '/api/status') return Response.json(status());
    if (p === '/api/calls') {
      const t = url.searchParams.get('ticket');
      // the list only: a real call carries tens of KB of messages and thinking, fetched one at a time from /api/call
      return Response.json(calls.filter((c) => !t || c.ticket === t || (t === 'orchestrator' && (c.role !== 'worker' || c.ticket.startsWith('orchestrator-')))).map(brief));
    }
    if (p === '/api/call') {
      const c = calls[Number(url.searchParams.get('id')) - 1];
      return c ? Response.json(c) : new Response('not found', { status: 404 });
    }
    if (p === '/api/events') return Response.json(events.slice(-Number(url.searchParams.get('last') ?? 300)));
    if (p === '/api/stream') {
      let send: ((chunk: string) => void) | null = null;
      let timer: ReturnType<typeof setInterval> | null = null;
      const stream = new ReadableStream({
        start(c) {
          const enc = new TextEncoder();
          // a tunnel slower than the stream (a slow link, a Mac asleep with ssh -N still open) would
          // queue every frame in this process for as long as the run lasts: a status frame is
          // skipped while the last ones are still unsent (the next supersedes it), and any frame
          // once far behind
          send = (chunk) => {
            if ((c.desiredSize ?? 1) <= (chunk.startsWith('event: status') ? 0 : -BEHIND)) return;
            try {
              c.enqueue(enc.encode(chunk));
            } catch {
              /* closed */
            }
          };
          send(`event: status\ndata: ${JSON.stringify(status())}\n\n`);
          listeners.add(send);
          timer = setInterval(() => send!(`event: status\ndata: ${JSON.stringify(status())}\n\n`), 1000);
        },
        cancel() {
          if (send) listeners.delete(send);
          if (timer) clearInterval(timer);
        },
      });
      return new Response(stream, { headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' } });
    }
    if (p === '/api/steer' && req.method === 'POST') {
      const body: any = await req.json().catch(() => ({}));
      if (typeof body.ticket !== 'string' || typeof body.note !== 'string' || !body.note.trim()) return Response.json({ error: 'ticket and note are required' }, { status: 400 });
      const s = { ticket: body.ticket, note: body.note.trim().slice(0, 2000), when: body.when === 'now' ? 'now' as const : 'next' as const, at: Date.now() };
      steers.push(s);
      appendFileSync(join(out, 'steer.jsonl'), JSON.stringify(s) + '\n');
      return Response.json({ stored: true, delivered: false, why: 'The engine has no steer path yet: the note is kept in steer.jsonl, but no prompt carries it.' });
    }
    if (p.startsWith('/result/')) return resultFile(decodeURIComponent(p.slice('/result/'.length)));
    return new Response('not found', { status: 404 });
  },
});
const viewUrl = `http://${host === '0.0.0.0' ? '127.0.0.1' : host}:${server.port}/`;

// ---------------------------------------------------------------------------------------------
// The run

const abort = new AbortController();
let stops = 0;
// SIGINT (Ctrl-C, kill -INT) and SIGTERM (plain kill) both stop the crew and write the summary
const stop = () => {
  if (++stops === 1 && orchestrator.phase !== 'finished' && orchestrator.phase !== 'failed') {
    console.log('\nstopping the crew (Ctrl-C again to quit now)');
    orchestrator.phase = 'stopped';
    abort.abort();
    return;
  }
  stopFake?.();
  process.exit(130);
};
process.on('SIGINT', stop);
process.on('SIGTERM', stop);

console.log(`\nlive view: ${viewUrl}  (from your Mac: ssh -N -L ${server.port}:127.0.0.1:${server.port} <this machine>, then open http://127.0.0.1:${server.port}/)`);
console.log(`run folder: ${out}\n`);

// the run's stop ends a planning call in flight too, so one kill stops a run that is still planning
const orchestratorInf = () => pool.forAgent({ role: 'manager', agent: 'orchestrator', team: 'orchestrator', signal: abort.signal });

/**
 * The orchestrator plans for the crew: about one ticket per code agent. Above 16 agents the
 * engine splits the scaffold and the tickets per design component and runs the parts at once,
 * on worker slots (idle while the plan is made), each part shown as orchestrator-<n>.
 */
const planFor: PlanOptions = { crew: concurrency, parts: (i) => pool.forAgent({ role: 'worker', agent: `orchestrator-${i + 1}`, team: 'orchestrator', signal: abort.signal }) };

/** The planned tickets, with what the planner warned about logged and left out of plan.json. */
function planned(p: PlannedTickets): Plan {
  const { warnings, ...plan } = p;
  for (const w of warnings ?? []) log(`orchestrator: ${w}`, 'warn');
  return plan;
}

// while the orchestrator plans, demo.log would stay silent for minutes (a thinking design call
// writes nothing until it ends): one line a minute says it is still at work
const planStart = Date.now();
const planBeat = setInterval(() => {
  if (!['designing', 'scaffolding', 'planning'].includes(orchestrator.phase)) return;
  const s = Math.round((Date.now() - planStart) / 1000);
  const open = calls.filter((c) => c.endAt === null && (c.role !== 'worker' || c.ticket.startsWith('orchestrator')));
  tell(`orchestrator: still ${orchestrator.phase} (${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s in; ${open.length} planning call${open.length === 1 ? '' : 's'} in flight, ${calls.filter((c) => c.endAt !== null && !c.error).length} answered)`);
}, 60_000);

try {
  if (opt.prompt) {
    mkdirSync(repo, { recursive: true });
    const inf = orchestratorInf();
    let files: string[];
    if (opt.repo) {
      const src = resolve(opt.repo);
      if (!existsSync(src)) die(`no such folder ${src}`);
      cpSync(src, repo, { recursive: true, filter: (s) => !s.includes('/.git/') && !s.endsWith('/.git') && !s.includes('/node_modules') });
      files = git(src, ['ls-files']).code === 0 ? git(src, ['ls-files']).out.split('\n').filter(Boolean) : readdirSync(repo, { recursive: true }).map(String);
      orchestrator.phase = 'designing';
      log(`orchestrator: designing "${opt.prompt}" for ${src}`);
      const d = await design(inf, `${opt.prompt}\n\nThe work happens in an existing repository with these files:\n${files.slice(0, 400).map((f) => `- ${f}`).join('\n')}`, [], 'architect', planFor);
      for (const w of d.warnings ?? []) log(`orchestrator: ${w}`, 'warn');
      writeFileSync(join(out, 'ARCHITECTURE.md'), architectureMarkdown(d));
      orchestrator.phase = 'planning';
      log('orchestrator: planning tickets');
      plan = planned(await tickets(inf, d, files, 'planner', planFor));
    } else {
      orchestrator.phase = 'designing';
      log(`orchestrator: designing "${opt.prompt}"`);
      const d = await design(inf, opt.prompt, [], 'architect', planFor);
      for (const w of d.warnings ?? []) log(`orchestrator: ${w}`, 'warn');
      writeFileSync(join(repo, 'ARCHITECTURE.md'), architectureMarkdown(d));
      if (d.questions.length) log(`orchestrator: it would have asked ${d.questions.join(' / ')} (answered by nobody in a demo)`, 'warn');
      orchestrator.phase = 'scaffolding';
      log('orchestrator: writing the scaffold');
      const sc = await scaffold(inf, d, 'planner', planFor);
      for (const w of sc.warnings) log(`orchestrator: ${w}`, 'warn');
      for (const [path, content] of Object.entries(sc.files)) {
        mkdirSync(join(repo, path, '..'), { recursive: true });
        writeFileSync(join(repo, path), content);
      }
      files = Object.keys(sc.files);
      orchestrator.phase = 'planning';
      log(`orchestrator: planning tickets on a ${files.length}-file scaffold`);
      // the scaffold itself, so each component's tickets may own the files its part wrote
      plan = planned(await tickets(inf, d, sc, 'planner', planFor));
    }
    orchestrator.planNote = `${plan.tickets.length} tickets planned by the orchestrator from your prompt`;
    initRepo(repo, undefined, 'scaffold');
  } else {
    cpSync(PROJECT_DIR, repo, { recursive: true });
    initRepo(repo, undefined, 'mdlite scaffold');
    const prepared = loadPlan();
    if (planMode === 'orchestrator') {
      orchestrator.phase = 'planning';
      log('orchestrator: planning the tickets from the scaffold');
      const r = await orchestratorPlan(orchestratorInf(), 'planner', repo, prepared);
      plan = r.plan;
      orchestrator.planSource = r.source;
      orchestrator.rejected = r.rejected ?? [];
      orchestrator.planNote = r.source === 'orchestrator'
        ? `${plan.tickets.length} tickets planned by the orchestrator`
        : `the orchestrator's plan was not used (${orchestrator.rejected.length} problem${orchestrator.rejected.length === 1 ? '' : 's'}); running the prepared plan`;
      log(`orchestrator: ${orchestrator.planNote}`, r.source === 'orchestrator' ? 'info' : 'warn');
      for (const p of orchestrator.rejected.slice(0, 8)) log(`  ${p}`, 'warn');
    } else {
      plan = prepared;
      orchestrator.planNote = `${plan.tickets.length} tickets from the prepared plan`;
    }
  }
  plan = validatePlan(plan);
  writeFileSync(join(out, 'plan.json'), JSON.stringify(plan, null, 2) + '\n');
  telemetry.setPlan(plan);
  for (const t of plan.tickets) digest(t.id);
  for (const c of plan.settled ?? []) log(`orchestrator: ${c.ticket} waits for ${c.with.join(', ')}: both write ${c.files.join(', ')}`);
  clearInterval(planBeat);
  orchestrator.phase = 'dispatching';
  log(`orchestrator: ${plan.tickets.length} tickets, ${plan.tickets.filter((t) => !t.depends_on.length).length} ready at once, for ${concurrency} code agents; at most ${checkSlots} checks run at once (--checks)`);

  // every 30 s while the crew works, one line of where the run stands
  const progress = setInterval(() => {
    const ds = [...digests.values()], st = telemetry.status();
    const n = (...states: string[]) => ds.filter((x) => states.includes(x.state)).length;
    const forLead = ds.filter((x) => x.state === 'queued' && x.queuedFor === 'lead').length;
    tell(`progress: ${n('finished')}/${ds.length} tickets done${n('failed', 'skipped', 'blocked') ? `, ${n('failed', 'skipped', 'blocked')} failed` : ''} · ${n('writing', 'thinking', 'fixing')} writing, ${n('checking')} checking, ${n('reviewing')} in review, ${forLead} waiting for a lead, ${n('queued') - forLead} waiting for a worker slot · ${Math.round(st.tokS.total).toLocaleString('en-US')} tok/s`);
  }, 30_000);
  const report = await runPlan({
    repo, base: headCommit(repo), plan, pool,
    sandbox: limited(sandboxFromEnv(), checkSlots),
    concurrency,
    worktreeRoot: join(out, 'worktrees'),
    emit,
    signal: abort.signal,
  }).finally(() => clearInterval(progress));
  const s = report.summary;
  result = { integrated: s.integrated, merged: report.integration?.merged ?? [], failedChecks: report.integration?.failedChecks ?? [], tests: null, testsOk: null, readme: false, summary: s };

  // the integrated project: its whole test suite, and its own README rendered by the tool the crew built
  const resultDir = join(out, 'result');
  if (git(repo, ['rev-parse', '--verify', '-q', INTEGRATION_BRANCH]).code === 0) {
    rmSync(resultDir, { recursive: true, force: true });
    git(repo, ['worktree', 'add', '-q', '--detach', resultDir, INTEGRATION_BRANCH]);
    // a Python project with a tests/ folder (the built-in task, the mock's, or yours): its whole suite
    const pyTests = existsSync(join(resultDir, 'tests')) && readdirSync(join(resultDir, 'tests')).some((f) => /^test_.*\.py$/.test(f));
    if (pyTests) {
      // tests/ without an __init__.py (each ticket's `python3 -m unittest tests.test_x` runs fine
      // without one) is "not importable" to discover -t . ; in its own process group with a
      // timeout, so a test that never ends cannot hold the run's end (and its summary) forever
      const discover = existsSync(join(resultDir, 'tests', '__init__.py')) ? 'python3 -m unittest discover -s tests -t .' : 'python3 -m unittest discover -s tests';
      const t = await new LocalSandbox().run(discover, { cwd: resultDir, image: 'python', timeoutMs: RESULT_TESTS_TIMEOUT_MS });
      result.tests = `${t.timedOut ? `(timed out after ${RESULT_TESTS_TIMEOUT_MS / 1000} s)\n` : ''}${t.out.split('\n').slice(-4).join('\n')}`.trim();
      result.testsOk = !t.timedOut && t.code === 0 && /\nOK\b|^OK\b/m.test(result.tests);
      if (opt.prompt) log(`whole project: ${result.testsOk ? 'every test passes' : 'tests do not all pass'}`);
    }
    if (!opt.prompt) {
      const r = Bun.spawnSync(['python3', '-m', 'mdlite', 'README.md', '--standalone', '--toc', '-o', 'README.html'], { cwd: resultDir, stderr: 'pipe' });
      result.readme = r.exitCode === 0 && existsSync(join(resultDir, 'README.html'));
      log(`whole project: ${result.testsOk ? 'every test passes' : 'tests do not all pass'}; README.html ${result.readme ? 'rendered by mdlite itself' : 'not rendered'}`);
    }
  }
  orchestrator.phase = abort.signal.aborted ? 'stopped' : 'finished';
  const callSummary = telemetry.close();
  writeFileSync(join(out, 'summary.json'), JSON.stringify({ run: id, task, plan: orchestrator.planSource, summary: s, result, calls: callSummary }, null, 2) + '\n');
  console.log(`\n${s.done}/${s.tickets} tickets done, ${s.failed} failed, ${s.skipped} skipped; ${s.peakParallel} at once at the peak; integration ${s.integrated ? 'green' : 'not green'}`);
  console.log(`${(s.wallMs / 1000).toFixed(0)} s wall, ${(s.serialMs / 1000).toFixed(0)} s of ticket time (${(s.serialMs / Math.max(1, s.wallMs)).toFixed(1)}x)`);
  console.log(`\n${formatCallSummary(callSummary.calls)}`);
  if (existsSync(resultDir)) console.log(`\nthe built project (mugge/integration, every finished ticket merged): ${resultDir}`);
  console.log(`logs in ${out}`);
} catch (e) {
  clearInterval(planBeat);
  orchestrator.phase = 'failed';
  log(stops > 0 ? 'demo stopped while the orchestrator was planning: no ticket ran' : `demo stopped: ${String((e as Error).message ?? e)}`, 'error');
  try {
    telemetry.close();
  } catch {
    /* already closed */
  }
}

// --exit, or a run stopped by a signal (kill, Ctrl-C): the summary is written, so end and free the view's port
if (opt.exit || stops > 0) {
  stopFake?.();
  server.stop(true);
  process.exit(stops > 0 ? 130 : result && result.summary.done === result.summary.tickets && result.integrated ? 0 : 1);
}
console.log(`\nthe view stays up at ${viewUrl}; Ctrl-C to quit`);
