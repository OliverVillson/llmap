/**
 * Telemetry: what every model call cost and what the crew is doing, written where people and
 * screens can read it. It folds the engine's events (events.ts) and keeps three things:
 *
 *   calls.jsonl   one CallRecord per model call: prompt, completion and reasoning tokens, whether
 *                 the answer was cut at the cap or forced after the thinking, and whether the
 *                 fenced-file parser got a usable file out of it (and why not). `clean` is false
 *                 for every call that must never become a training row.
 *   status.json   a CrewStatus snapshot, rewritten in place (write + rename) at most once per
 *                 `statusIntervalMs` and once more at the end: crew and agent states, tickets,
 *                 tok/s over a one-minute window split by team and machine, slots per machine.
 *                 The console, Studio and Train screens read this (or GET /telemetry on the
 *                 engine API, which serves the same object).
 *   a warning     on stderr (and in status.warnings) once more than 20% of answers are lost to
 *                 the parser or the token cap: exp02 lost 670 of 900 teacher fix answers that way
 *                 without anyone noticing.
 *
 * The crew layout (teams a–d, agents a1…d10, the machine each team uses) comes from MUGGE_CREW,
 * else from the backend machines (MUGGE_BACKENDS, backends.ts) with the teams spread over the
 * machines that serve workers. Until the scheduler assigns agents itself (`start.agent`),
 * telemetry binds each started ticket to an idle agent of the least busy team whose machine has
 * a free slot. The backend pool's snapshots (`serving()`) supply each machine's real slot use,
 * and `slot` events say when an agent's call waits for a slot (queued) and where it runs.
 *
 * tok/s counts completion tokens. Calls are not streamed, so a call's tokens are spread evenly
 * over its own duration once it ends: the one-minute figure trails the live rate by up to one
 * call. A serving layer that knows better can feed `tokens()` directly.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { machinesFromEnv } from '../config.ts';
import { settledReason, type Plan, type WorkerOutput } from '../tickets/schema.ts';
import type { Machine, ServingSnapshot } from './backends.ts';
import type { EngineEvent } from './events.ts';
import type { AgentState } from './state.ts';

// ---------------------------------------------------------------------------------------------
// Parser drops

/**
 * Why an answer gave the engine no usable file.
 *   cut         stopped at max_tokens before any owned file was complete
 *   empty       nothing but (closed or open) thinking, or no text at all
 *   no_fence    text without a single fenced block
 *   unlabelled  fenced blocks, but none the parser could tie to an owned path
 *   not_owned   wrote a file outside `owns`; the harness refuses the whole answer
 */
export type DropReason = 'cut' | 'empty' | 'no_fence' | 'unlabelled' | 'not_owned';
export const DROP_REASONS: readonly DropReason[] = ['cut', 'empty', 'no_fence', 'unlabelled', 'not_owned'];

export interface AnswerCheck {
  drop?: DropReason;
  /** Owned paths the answer wrote. */
  written: string[];
  /** Owned paths it left out (a partial answer, not a drop, when `written` is not empty). */
  missing: string[];
  /** Paths it wrote that the ticket does not own. */
  stray: string[];
  /**
   * Owned paths the parser had to guess: not written as the training format writes them (the
   * path alone on its line, then the fence; files.ts renderFiles), e.g. `prog.c` for
   * `libshort/src/hash.c` or `**src/a.py**`. The engine can use such an answer; a training row
   * must not, or the model learns the wrong shape (Centipede run 1's 0%).
   */
  loose: string[];
}

/** What the parser (files.ts parseFiles) made of one answer; read-only, never changes the format. */
export function classifyAnswer(raw: string, owns: string[], out: WorkerOutput | null | undefined, cut?: boolean): AnswerCheck {
  const paths = Object.keys(out?.files ?? {});
  const written = paths.filter((p) => owns.includes(p));
  const stray = paths.filter((p) => !owns.includes(p));
  const missing = owns.filter((p) => !written.includes(p));
  const strict = (p: string) => new RegExp(`^${p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[ \\t]*\\r?\\n\`{3,}`, 'm').test(raw);
  const loose = written.filter((p) => !strict(p));
  let drop: DropReason | undefined;
  if (stray.length) drop = 'not_owned';
  else if (!written.length) {
    const text = raw.replace(/<think>[\s\S]*?(?:<\/think>|$)/g, '').trim();
    if (cut) drop = 'cut';
    else if (!text) drop = 'empty';
    else if (!/^\s*`{3,}/m.test(text)) drop = 'no_fence';
    else drop = 'unlabelled';
  } else if (cut && (raw.replace(/<think>[\s\S]*?(?:<\/think>|$)/g, '').match(/^[ \t]*`{3,}/gm)?.length ?? 0) % 2 === 1) {
    // the cap fell inside a file: the parser kept half of it
    drop = 'cut';
  }
  return { drop, written, missing, stray, loose };
}

/** Share of answers lost to the parser, the cap or a forced answer above which a run warns. */
export const LOST_WARN_SHARE = 0.2;

// ---------------------------------------------------------------------------------------------
// The per-call log

export interface CallRecord {
  v: 1;
  run: string;
  /** When the call ended, ms since the epoch. */
  at: number;
  ticket: string;
  attempt: number;
  /** write and fix are worker calls; review is the lead's review of a passing ticket. */
  kind: 'write' | 'fix' | 'review';
  agent: string | null;
  team: string | null;
  machine: string | null;
  /** The model the ticket routed to (the served name is in the MUGGE_DEBUG_CALLS line). */
  model: string | null;
  promptTokens: number;
  completionTokens: number;
  /** As the server reports it; null when it does not (llama-server). */
  reasoningTokens: number | null;
  /** Thinking characters the server split off the answer, when it does. */
  reasoningChars: number | null;
  /** max_tokens the call was sent with. */
  maxTokens: number | null;
  ms: number;
  /** Completion tokens over the call's own time. */
  tokS: number;
  /** Stopped at max_tokens. */
  cut: boolean;
  /** Forced after the thinking by the server, and why ("length", "stop", "no_reasoning"). */
  forced: string | null;
  /** Files the parser found (owned or not). */
  files: number;
  /** Owned files the answer left out. */
  missing: number;
  /** Owned files whose path the parser had to guess (AnswerCheck.loose). */
  loose: number;
  drop: DropReason | null;
  /** The call threw: no answer at all. */
  error: string | null;
  /** Not cut, not forced, not dropped, no guessed path, no error: the only calls that may ever become training rows. */
  clean: boolean;
}

export interface CallSummary {
  calls: number;
  errors: number;
  clean: number;
  cut: number;
  forced: number;
  /** Answers with a path the parser had to guess. */
  loose: number;
  /** Answers with no usable file (any DropReason). */
  dropped: number;
  dropReasons: Record<DropReason, number>;
  /** Worker answers (write and fix calls that did not error): what lost and loose shares are out of. Review answers are not rows. */
  answers: number;
  /** Worker answers dropped, cut or forced (none can become a row): what the 20% warning counts. */
  lost: number;
  /** lost / answers. */
  lostShare: number;
  write: number;
  fix: number;
  review: number;
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
}

export function emptyCallSummary(): CallSummary {
  return {
    calls: 0, errors: 0, clean: 0, cut: 0, forced: 0, loose: 0, dropped: 0,
    dropReasons: { cut: 0, empty: 0, no_fence: 0, unlabelled: 0, not_owned: 0 },
    answers: 0, lost: 0, lostShare: 0, write: 0, fix: 0, review: 0, promptTokens: 0, completionTokens: 0, reasoningTokens: 0,
  };
}

export function addCall(s: CallSummary, r: CallRecord): CallSummary {
  s.calls++;
  s[r.kind]++;
  if (r.error) s.errors++;
  if (r.clean) s.clean++;
  if (r.cut) s.cut++;
  if (r.forced) s.forced++;
  if (r.loose) s.loose++;
  if (r.drop) {
    s.dropped++;
    s.dropReasons[r.drop]++;
  }
  // only worker answers can become rows: a review answer neither counts as lost nor dilutes the share
  if (r.kind !== 'review' && !r.error) {
    s.answers++;
    // a forced answer can never be a row either, and many of them mean the cap is wrong
    if (r.drop || r.cut || r.forced) s.lost++;
  }
  s.lostShare = s.answers ? s.lost / s.answers : 0;
  s.promptTokens += r.promptTokens;
  s.completionTokens += r.completionTokens;
  s.reasoningTokens += r.reasoningTokens ?? 0;
  return s;
}

/** A summary of a calls.jsonl file, e.g. to check a run before harvesting rows from it. */
export function summarizeCallLog(path: string): CallSummary {
  const s = emptyCallSummary();
  if (!existsSync(path)) return s;
  for (const line of readFileSync(path, 'utf8').split('\n')) if (line.trim()) addCall(s, JSON.parse(line));
  return s;
}

/** The lost-answers warning, or null when the run is under the line. */
export function lostWarning(s: CallSummary, share = LOST_WARN_SHARE): string | null {
  if (!s.answers || s.lostShare <= share) return null;
  const reasons = DROP_REASONS.filter((r) => s.dropReasons[r]).map((r) => `${r} ${s.dropReasons[r]}`).join(', ');
  const cutKept = s.lost - s.dropped;
  return `WARNING: ${s.lost} of ${s.answers} worker answers (${pct(s.lostShare)}) were lost to the parser, the token cap or a forced answer` +
    ` (dropped: ${reasons || 'none'}${cutKept ? `; cut or forced but parsed: ${cutKept}` : ''}). Check the format and MUGGE_MAX_CALL_TOKENS before trusting this run.`;
}

/** Plain lines for the end of a run. */
export function formatCallSummary(s: CallSummary): string {
  const reasons = DROP_REASONS.filter((r) => s.dropReasons[r]).map((r) => `${r} ${s.dropReasons[r]}`).join(', ');
  const rows: Array<[string, string]> = [
    ['model calls (write / fix / review)', `${s.calls} (${s.write} / ${s.fix} / ${s.review})`],
    ['worker answers lost (no row)', `${s.lost}/${s.answers}`],
    ['parser drops', `${s.dropped}${reasons ? ` (${reasons})` : ''}`],
    ['cut at the cap', String(s.cut)],
    ['forced after thinking', String(s.forced)],
    ['paths the parser guessed', String(s.loose)],
    ['call errors', String(s.errors)],
    ['clean answers (may become rows)', `${s.clean}/${s.calls}`],
    ['tokens prompt / completion / reasoning', `${s.promptTokens} / ${s.completionTokens} / ${s.reasoningTokens}`],
  ];
  const w = Math.max(...rows.map((r) => r[0].length));
  return rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`).join('\n');
}

const pct = (x: number) => `${Math.round(100 * x)}%`;

// ---------------------------------------------------------------------------------------------
// Crew layout

export interface MachineSpec {
  id: string;
  /** Shown to people, e.g. "4060 laptop". */
  label?: string;
  /** Calls this machine's server runs at once (llama-server -np). */
  slots: number;
  /** e.g. "local · llama-server". */
  backend?: string;
  url?: string;
  /** For the memory gauge, when known: what the card has, the shared model, one agent's working memory. */
  memoryBytes?: number;
  modelBytes?: number;
  perAgentBytes?: number;
}

export interface TeamSpec {
  id: string;
  /** Editable placeholder, e.g. "backend". */
  domain?: string;
  machine: string;
  agents: number;
}

export interface CrewLayout {
  machines: MachineSpec[];
  teams: TeamSpec[];
  /** Where the manager and leads run, when they do (not yet: BH0.1 is a worker model). */
  manager?: { machine: string };
}

/** One machine with every slot, teams a–d with ten agents each. */
export function defaultCrew(slots = 12, backend = 'local'): CrewLayout {
  return {
    machines: [{ id: 'local', slots, backend }],
    teams: ['a', 'b', 'c', 'd'].map((id) => ({ id, machine: 'local', agents: 10 })),
  };
}

/** Teams a–d over the machines that serve worker calls, in turn. */
export function crewFromMachines(machines: Machine[]): CrewLayout {
  const workers = machines.filter((m) => m.roles.includes('worker'));
  const on = workers.length ? workers : machines;
  return {
    machines: machines.map((m) => ({ id: m.name, slots: m.slots, backend: backendLabel(m.url), url: m.url, memoryBytes: m.memoryBytes, perAgentBytes: m.bytesPerSlot })),
    teams: ['a', 'b', 'c', 'd'].map((id, i) => ({ id, machine: on[i % on.length]!.name, agents: 10 })),
    manager: machines.find((m) => m.roles.includes('manager')) ? { machine: machines.find((m) => m.roles.includes('manager'))!.name } : undefined,
  };
}

/** `local · llama-server` for a URL on this machine or the LAN, `remote` otherwise. */
export function backendLabel(url: string): string {
  try {
    const h = new URL(url).hostname;
    return /^(127\.|localhost$|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1)/.test(h) || h.endsWith('.local') ? 'local · llama-server' : `remote · ${h}`;
  } catch {
    return 'local';
  }
}

/** MUGGE_CREW (a CrewLayout as JSON, inline or in a file), else the backend machines. */
export function crewFromEnv(env: NodeJS.ProcessEnv = process.env, slots = 12): CrewLayout {
  const raw = env.MUGGE_CREW?.trim();
  if (!raw) return crewFromMachines(machinesFromEnv(env, slots));
  let c: CrewLayout;
  try {
    c = JSON.parse(raw.startsWith('{') ? raw : readFileSync(raw, 'utf8'));
  } catch (e) {
    throw new Error(`MUGGE_CREW must be a crew layout as JSON or a path to one: ${(e as Error).message}`);
  }
  if (!Array.isArray(c.machines) || !c.machines.length || !Array.isArray(c.teams) || !c.teams.length) throw new Error('MUGGE_CREW needs machines and teams');
  for (const t of c.teams) if (!c.machines.some((m) => m.id === t.machine)) throw new Error(`MUGGE_CREW team ${t.id} names unknown machine ${t.machine}`);
  return c;
}

// ---------------------------------------------------------------------------------------------
// The status stream

export const STATUS_SCHEMA = 'centipede.status/1';

export type CrewState = 'ready' | 'working' | 'paused' | 'needs_you' | 'finished';

/**
 * An agent's state is the console's (state.ts), taken from the engine's `state` events when it
 * sends them. For an agent, `queued` means its model call waits for a free machine slot (a `slot`
 * event); a ticket that could start but has no agent yet is `ready` on the ticket list instead,
 * and `waiting` and `blocked` before a ticket starts show there too.
 */
export type { AgentState };
/** Has a ticket in progress. */
const WORKING: AgentState[] = ['writing', 'checking', 'fixing', 'thinking', 'reviewing', 'queued'];
/** Is in a model call (holds a machine slot, when the pool does not say otherwise). */
const CALLING: AgentState[] = ['writing', 'fixing', 'thinking'];

/**
 * waiting: a dependency has not finished. ready: could start, every agent slot is taken.
 * blocked: ready, but a running ticket owns one of its files. queued: running, its model call
 * waits for a machine slot (events.ts). running, done, failed, skipped as the engine says.
 */
export type TicketStatusState = 'waiting' | 'ready' | 'blocked' | 'running' | 'queued' | 'done' | 'failed' | 'skipped';

/** idle: nothing written in the window. warming: under a minute in. steady. peak: a new high in the last 5 s. drop: under half the 10-minute rate while agents work. */
export type TokSState = 'idle' | 'warming' | 'steady' | 'peak' | 'drop';

export interface AgentStatus {
  id: string;
  role: 'worker' | 'lead' | 'manager';
  team: string;
  machine: string;
  state: AgentState;
  ticket: string | null;
  owns: string[];
  /** Attempt in progress or last made, and the most the ticket gets. */
  attempt: number;
  maxAttempts: number;
  kind: 'write' | 'fix' | null;
  /** The engine's slot number while it holds one (state events), else null. */
  slot: number | null;
  /** Why it is in this state, when the engine says (blocked, failed). */
  reason: string | null;
  /** Its own completion tok/s over the last minute, and of its last call. */
  tokS: number;
  lastCallTokS: number | null;
  lastNote: string;
  /** When it entered this state, ms since the epoch. */
  since: number;
  calls: number;
}

export interface TicketStatus {
  id: string;
  state: TicketStatusState;
  agent: string | null;
  attempt: number;
  maxAttempts: number;
  owns: string[];
  dependsOn: string[];
  /** Why it is blocked: the running ticket that holds a file it owns. */
  blockedBy?: string;
  /** A clash the planner settled: this ticket runs after those that share its files. */
  settled?: string;
  error?: string;
}

export interface CrewStatus {
  schema: typeof STATUS_SCHEMA;
  run: string;
  updatedAt: number;
  startedAt: number | null;
  elapsedMs: number;
  crew: {
    state: CrewState;
    /** working: has a ticket (queued ones included); queued: waiting for a machine slot. */
    agents: { total: number; working: number; queued: number; idle: number; finished: number; failed: number };
    tickets: Record<TicketStatusState, number> & { total: number };
  };
  settings: { thinking: 'on' | 'off' | 'server'; maxCallTokens: number | null; model: string | null; backend: string };
  tokS: {
    /** Completion tokens per second over the last minute (or since the start, if shorter). */
    total: number;
    state: TokSState;
    peak: number;
    tenMinutes: number;
    session: number;
    /** Per-second completion tokens, oldest first, the last 60 s. */
    spark: number[];
    byTeam: Record<string, number>;
    byMachine: Record<string, number>;
  };
  /** slotsUsed and queued come from the backend pool when it reports (serving()), else from agent states. */
  machines: Array<MachineSpec & { slotsUsed: number; queued: number; agents: number; tokS: number; promptTokS: number | null }>;
  teams: Array<{ id: string; domain: string | null; machine: string; agents: string[]; working: number; tokS: number; tickets: string[] }>;
  agents: AgentStatus[];
  tickets: TicketStatus[];
  calls: CallSummary;
  warnings: string[];
}

export interface TicketInfo {
  id: string;
  owns: string[];
  depends_on: string[];
  /** Model calls the ticket may make in all (budget.attempts, times two with escalation). */
  maxAttempts?: number;
}

export function ticketsOf(plan: Plan): TicketInfo[] {
  return plan.tickets.map((t) => ({ id: t.id, owns: t.owns, depends_on: t.depends_on, maxAttempts: t.budget.attempts * (t.escalate_to ? 2 : 1) }));
}

export interface TelemetryOptions {
  /** Folder for calls.jsonl and status.json; nothing is written without one. */
  dir?: string;
  run?: string;
  crew?: CrewLayout;
  settings?: Partial<CrewStatus['settings']>;
  statusIntervalMs?: number;
  /** Where the lost-answers warning goes (default stderr); null for nowhere. */
  warn?: ((line: string) => void) | null;
  /** Calls before the lost-answers warning may fire mid-run (it always may at close). */
  warnAfter?: number;
  now?: () => number;
}

interface Span { start: number; end: number; tokens: number; agent: string | null; team: string | null; machine: string | null }
interface TicketRow extends TicketStatus { started: boolean }

const WINDOW_MS = 60_000;
const TEN_MIN_MS = 600_000;

export class Telemetry {
  readonly run: string;
  readonly dir: string | null;
  readonly crew: CrewLayout;
  readonly callsPath: string | null;
  readonly statusPath: string | null;
  private readonly now: () => number;
  private readonly warnSink: ((line: string) => void) | null;
  private readonly settings: CrewStatus['settings'];
  private agents: AgentStatus[] = [];
  private tickets = new Map<string, TicketRow>();
  /** Ticket → the owners a settled clash made it wait for: only an order, so an owner that failed still lets it run. */
  private settledWith = new Map<string, Set<string>>();
  private spans: Span[] = [];
  private summary = emptyCallSummary();
  private warnings: string[] = [];
  private warnedLost = false;
  private startedAt: number | null = null;
  private ended = false;
  private peak = 0;
  private peakAt = 0;
  private sessionTokens = 0;
  private lastWrite = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private pool: ServingSnapshot | null = null;
  /** The engine sends `state` events: take agent states from them, not from model and check events. */
  private stated = false;
  /** An agent's call state while its call waits for a slot, to go back to once it gets one. */
  private beforeQueue = new Map<string, AgentState>();
  private writeFailed = new Set<string>();

  constructor(private readonly o: TelemetryOptions = {}) {
    this.now = o.now ?? Date.now;
    this.run = o.run ?? runId(this.now());
    this.crew = o.crew ?? defaultCrew();
    this.warnSink = o.warn === null ? null : o.warn ?? ((l) => console.error(l));
    this.settings = { thinking: 'server', maxCallTokens: null, model: null, backend: this.crew.machines[0]?.backend ?? 'local', ...o.settings };
    for (const t of this.crew.teams)
      for (let i = 1; i <= t.agents; i++)
        this.agents.push({ id: `${t.id}${i}`, role: 'worker', team: t.id, machine: t.machine, state: 'idle', ticket: null, owns: [], attempt: 0, maxAttempts: 0, kind: null, slot: null, reason: null, tokS: 0, lastCallTokS: null, lastNote: '', since: this.now(), calls: 0 });
    this.dir = o.dir ?? null;
    this.callsPath = this.dir ? join(this.dir, 'calls.jsonl') : null;
    this.statusPath = this.dir ? join(this.dir, 'status.json') : null;
    if (this.dir) {
      mkdirSync(this.dir, { recursive: true });
      writeFileSync(this.callsPath!, '');
    }
  }

  /** The tickets the run will see, with their owned files and budgets (the plan event lacks both). */
  setTickets(list: TicketInfo[]): void {
    for (const t of list) {
      const prev = this.tickets.get(t.id);
      this.tickets.set(t.id, { id: t.id, state: prev?.state ?? 'waiting', agent: prev?.agent ?? null, attempt: prev?.attempt ?? 0, maxAttempts: t.maxAttempts ?? 4, owns: t.owns, dependsOn: t.depends_on, started: prev?.started ?? false });
    }
    this.touch();
  }

  setPlan(plan: Plan): void {
    this.setTickets(ticketsOf(plan));
  }

  /** Feeds every engine event through; pass it as (or inside) the scheduler's `emit`. */
  readonly emit = (e: EngineEvent): void => {
    switch (e.type) {
      case 'plan':
        this.startedAt ??= e.at;
        for (const t of e.tickets) if (!this.tickets.has(t.id)) this.ticket(t.id, t.depends_on);
        break;
      case 'slot': {
        const t = this.ticket(e.ticket);
        const a = this.agentFor(t, e.at);
        if (e.machine) a.machine = e.machine;
        if (this.stated) break; // the engine's state events carry queued themselves
        if (e.state === 'queued') {
          t.state = 'running'; // shown as queued below while the agent waits
          if (a.state !== 'queued') this.beforeQueue.set(a.id, a.state);
          this.set(a, 'queued', e.at, {});
        } else if (a.state === 'queued') {
          this.set(a, this.beforeQueue.get(a.id) ?? (a.kind === 'fix' ? 'fixing' : 'writing'), e.at, {});
          this.beforeQueue.delete(a.id);
        }
        break;
      }
      case 'state': {
        this.stated = true;
        this.startedAt ??= e.at;
        const t = this.ticket(e.ticket);
        t.maxAttempts = e.of || t.maxAttempts;
        if (e.attempt) t.attempt = e.attempt;
        if (!t.started && !t.agent && (e.state === 'waiting' || e.state === 'queued' || e.state === 'blocked')) {
          // not started: the ticket list shows it (ready, when only a free agent is missing)
          t.state = e.state === 'blocked' ? 'blocked' : 'waiting';
          if (e.reason) t.error = e.reason;
          break;
        }
        const final = e.state === 'finished' || e.state === 'failed' || e.state === 'blocked';
        if (final) {
          t.state = e.state === 'finished' ? 'done' : e.state === 'blocked' ? 'blocked' : 'failed';
          const a = t.agent ? this.agents.find((x) => x.id === t.agent && x.ticket === t.id) : undefined;
          if (a) {
            this.set(a, e.state, e.at, { attempt: e.attempt || a.attempt });
            a.reason = e.reason ?? null;
            a.slot = null;
          }
          break;
        }
        t.state = 'running';
        t.started = true;
        const a = this.agentFor(t, e.at, e.agent === undefined ? undefined : this.crewAgent(e.agent));
        a.maxAttempts = t.maxAttempts;
        if (e.agent !== undefined) a.slot = e.agent;
        this.set(a, e.state, e.at, { attempt: e.attempt || a.attempt, kind: e.state === 'writing' ? 'write' : e.state === 'fixing' ? 'fix' : undefined });
        break;
      }
      case 'clash': {
        const t = this.ticket(e.ticket);
        if (e.settled) {
          // not a block: the ticket waits on the owners like any dependency
          for (const w of e.with) if (!t.dependsOn.includes(w)) t.dependsOn.push(w);
          for (const w of e.with) (this.settledWith.get(e.ticket) ?? this.settledWith.set(e.ticket, new Set()).get(e.ticket)!).add(w);
          t.settled = settledReason(e);
          break;
        }
        t.state = 'blocked';
        t.blockedBy = e.with[0];
        t.error = `needs ${e.files.join(', ')}, also claimed by ${e.with.join(', ')}`;
        break;
      }
      case 'review': {
        const t = this.ticket(e.ticket);
        const a = this.agentFor(t, e.at);
        this.record({
          v: 1, run: this.run, at: e.at, ticket: e.ticket, attempt: e.attempt, kind: 'review',
          agent: a.id, team: a.team, machine: a.machine, model: null,
          promptTokens: e.promptTokens, completionTokens: e.completionTokens, reasoningTokens: e.reasoningTokens ?? null, reasoningChars: null, maxTokens: null,
          ms: e.ms, tokS: e.ms > 0 ? round1((1000 * e.completionTokens) / e.ms) : 0, cut: !!e.cut, forced: null,
          files: 0, missing: 0, loose: 0, drop: null, error: e.verdict === 'error' ? e.error ?? 'review failed' : null, clean: !e.cut && e.verdict !== 'error',
        });
        this.spans.push({ start: e.at - e.ms, end: e.at, tokens: e.completionTokens, agent: a.id, team: a.team, machine: a.machine });
        this.sessionTokens += e.completionTokens;
        break;
      }
      case 'start': {
        this.startedAt ??= e.at;
        const t = this.ticket(e.ticket);
        t.state = 'running';
        t.started = true;
        t.attempt = e.attempt;
        if (this.stated && e.agent === undefined) break; // its first state event names the agent
        const a = this.bind(t, e.agent, e.machine, e.at);
        this.set(a, 'writing', e.at, { attempt: e.attempt, kind: 'write' });
        break;
      }
      case 'model': {
        const t = this.ticket(e.ticket);
        const a = this.agentFor(t, e.at);
        const tokS = e.ms > 0 ? (1000 * e.completionTokens) / e.ms : 0;
        this.record({
          v: 1, run: this.run, at: e.at, ticket: e.ticket, attempt: e.attempt, kind: e.kind,
          agent: a.id, team: a.team, machine: a.machine, model: e.model ?? null,
          promptTokens: e.promptTokens, completionTokens: e.completionTokens,
          reasoningTokens: e.reasoningTokens ?? null, reasoningChars: e.reasoningChars ?? null, maxTokens: e.maxTokens ?? null,
          ms: e.ms, tokS: Math.round(tokS * 10) / 10, cut: !!e.cut, forced: e.forced ?? null,
          files: e.files ?? 0, missing: e.missing ?? 0, loose: e.loose ?? 0, drop: e.drop ?? null, error: null,
          clean: !e.cut && !e.forced && !e.drop && !e.loose,
        });
        this.spans.push({ start: e.at - e.ms, end: e.at, tokens: e.completionTokens, agent: a.id, team: a.team, machine: a.machine });
        this.sessionTokens += e.completionTokens;
        a.calls++;
        a.lastCallTokS = Math.round(tokS * 10) / 10;
        a.lastNote = e.note;
        t.attempt = e.attempt;
        if (!this.stated) this.set(a, 'checking', e.at, { attempt: e.attempt, kind: e.kind });
        break;
      }
      case 'call_error': {
        const t = this.ticket(e.ticket);
        const a = this.agentFor(t, e.at);
        this.record({
          v: 1, run: this.run, at: e.at, ticket: e.ticket, attempt: e.attempt, kind: e.kind,
          agent: a.id, team: a.team, machine: a.machine, model: e.model ?? null,
          promptTokens: 0, completionTokens: 0, reasoningTokens: null, reasoningChars: null, maxTokens: null,
          ms: 0, tokS: 0, cut: false, forced: null, files: 0, missing: 0, loose: 0, drop: null, error: e.error, clean: false,
        });
        a.calls++;
        this.next(t, a, e.attempt, e.at);
        break;
      }
      case 'refused': {
        const t = this.ticket(e.ticket);
        this.next(t, this.agentFor(t, e.at), e.attempt, e.at);
        break;
      }
      case 'check': {
        if (e.ok) break;
        const t = this.ticket(e.ticket);
        this.next(t, this.agentFor(t, e.at), e.attempt, e.at);
        break;
      }
      case 'finish': {
        const t = this.ticket(e.ticket);
        t.state = e.state;
        t.attempt = e.attempts || t.attempt;
        if (e.error) t.error = e.error.split('\n')[0]!.slice(0, 300);
        const a = t.agent ? this.agents.find((x) => x.id === t.agent) : undefined;
        if (a && a.ticket === t.id) {
          this.set(a, e.state === 'done' ? 'finished' : e.state === 'blocked' ? 'blocked' : 'failed', e.at, { attempt: t.attempt });
          a.slot = null;
        }
        break;
      }
      case 'integrate':
        // integration fix tickets never get a finish event: free their agents
        for (const t of this.tickets.values()) {
          if (t.started || t.state !== 'running') continue;
          t.state = e.ok ? 'done' : 'failed';
          const a = this.agents.find((x) => x.id === t.agent && x.ticket === t.id);
          if (a) this.set(a, e.ok ? 'finished' : 'failed', e.at, {});
        }
        break;
      case 'end':
        this.ended = true;
        break;
    }
    this.touch();
  };

  /** The backend pool's latest numbers (BackendPool.subscribe): real slot use per machine. */
  serving(s: ServingSnapshot): void {
    this.pool = s;
    for (const m of s.machines)
      if (!this.crew.machines.some((x) => x.id === m.name)) this.crew.machines.push({ id: m.name, slots: m.slots, backend: backendLabel(m.url), url: m.url, memoryBytes: m.memoryBytes, perAgentBytes: m.bytesPerSlot });
    this.touch();
  }

  /** Completion tokens a streaming serving layer saw, for a live rate (instead of whole calls). */
  tokens(n: number, startMs: number, endMs: number, agent: string | null = null): void {
    const a = agent ? this.agents.find((x) => x.id === agent) : undefined;
    this.spans.push({ start: startMs, end: endMs, tokens: n, agent, team: a?.team ?? null, machine: a?.machine ?? null });
    this.sessionTokens += n;
  }

  calls(): CallSummary {
    return structuredClone(this.summary);
  }

  status(): CrewStatus {
    const now = this.now();
    this.spans = this.spans.filter((s) => s.end > now - TEN_MIN_MS);
    const elapsed = this.startedAt === null ? 0 : now - this.startedAt;
    const win = Math.max(1000, Math.min(WINDOW_MS, elapsed || WINDOW_MS));
    const rate = (keep: (s: Span) => boolean, from = now - win) => {
      let tok = 0;
      for (const s of this.spans) if (keep(s)) tok += overlap(s, from, now);
      return round1((1000 * tok) / (now - from));
    };
    const total = rate(() => true);
    const ten = rate(() => true, now - Math.max(1000, Math.min(TEN_MIN_MS, elapsed || TEN_MIN_MS)));
    if (total > this.peak) {
      this.peak = total;
      this.peakAt = now;
    }
    const working = this.agents.filter((a) => WORKING.includes(a.state));
    let state: TokSState = 'steady';
    if (this.ended || (!total && !working.length)) state = 'idle';
    else if (elapsed < WINDOW_MS) state = 'warming';
    else if (now - this.peakAt < 5000 && total > 0) state = 'peak';
    else if (working.length && total < ten / 2) state = 'drop';
    const spark: number[] = [];
    for (let i = 59; i >= 0; i--) {
      const to = now - i * 1000, from = to - 1000;
      let tok = 0;
      for (const s of this.spans) tok += overlap(s, from, to);
      spark.push(round1(tok));
    }
    const byTeam: Record<string, number> = {}, byMachine: Record<string, number> = {};
    for (const t of this.crew.teams) byTeam[t.id] = rate((s) => s.team === t.id);
    for (const m of this.crew.machines) byMachine[m.id] = rate((s) => s.machine === m.id);
    for (const a of this.agents) a.tokS = rate((s) => s.agent === a.id);

    const tickets = this.ticketStatuses();
    const tcount = { total: tickets.length, waiting: 0, ready: 0, blocked: 0, running: 0, queued: 0, done: 0, failed: 0, skipped: 0 } as CrewStatus['crew']['tickets'];
    for (const t of tickets) tcount[t.state]++;
    let crewState: CrewState = 'ready';
    if (this.ended) crewState = tcount.done === tcount.total ? 'finished' : 'needs_you';
    else if (this.startedAt !== null) crewState = 'working';

    return {
      schema: STATUS_SCHEMA,
      run: this.run,
      updatedAt: now,
      startedAt: this.startedAt,
      elapsedMs: elapsed,
      crew: {
        state: crewState,
        agents: {
          total: this.agents.length,
          working: working.length,
          queued: this.agents.filter((a) => a.state === 'queued').length,
          idle: this.agents.filter((a) => a.state === 'idle').length,
          finished: this.agents.filter((a) => a.state === 'finished').length,
          failed: this.agents.filter((a) => a.state === 'failed').length,
        },
        tickets: tcount,
      },
      settings: { ...this.settings },
      tokS: { total, state, peak: this.peak, tenMinutes: ten, session: elapsed ? round1((1000 * this.sessionTokens) / elapsed) : 0, spark, byTeam, byMachine },
      machines: this.crew.machines.map((m) => {
        const on = this.agents.filter((a) => a.machine === m.id);
        const pool = this.pool?.machines.find((x) => x.name === m.id);
        return {
          ...m,
          slots: pool?.slots ?? m.slots,
          slotsUsed: pool ? pool.busy : on.filter((a) => CALLING.includes(a.state)).length,
          queued: pool ? pool.queued : on.filter((a) => a.state === 'queued').length,
          agents: on.length,
          tokS: byMachine[m.id] ?? 0,
          promptTokS: pool ? pool.promptTokPerSec : null,
        };
      }),
      teams: this.crew.teams.map((t) => {
        const on = this.agents.filter((a) => a.team === t.id);
        return { id: t.id, domain: t.domain ?? null, machine: t.machine, agents: on.map((a) => a.id), working: on.filter((a) => WORKING.includes(a.state)).length, tokS: byTeam[t.id] ?? 0, tickets: on.filter((a) => a.ticket && WORKING.includes(a.state)).map((a) => a.ticket!) };
      }),
      agents: this.agents.map((a) => ({ ...a, owns: [...a.owns] })),
      tickets,
      calls: this.calls(),
      warnings: [...this.warnings],
    };
  }

  /** Writes the final status and the warning if the whole run is over the line. */
  close(): CrewStatus {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.ended = true;
    this.checkLost(true);
    const s = this.status();
    this.write(s);
    return s;
  }

  // -- internals ------------------------------------------------------------------------------

  private ticket(id: string, dependsOn: string[] = []): TicketRow {
    let t = this.tickets.get(id);
    if (!t) {
      t = { id, state: 'waiting', agent: null, attempt: 0, maxAttempts: 4, owns: [], dependsOn, started: false };
      this.tickets.set(id, t);
    }
    return t;
  }

  /**
   * The crew agent for the engine's agent number n (0 … concurrency-1): the teams in turn, so
   * agents 0–3 are a1 b1 c1 d1, 4–7 are a2 b2 c2 d2, and so on.
   */
  crewAgent(n: number): string {
    const teams = this.crew.teams;
    return `${teams[n % teams.length]!.id}${Math.floor(n / teams.length) + 1}`;
  }

  /**
   * The agent working on t: `want` when the engine names one, else the one bound already, else a
   * new binding (a call for a ticket that never started: integration fixes).
   */
  private agentFor(t: TicketRow, at: number, want?: string): AgentStatus {
    const a = t.agent ? this.agents.find((x) => x.id === t.agent && x.ticket === t.id) : undefined;
    if (a && (!want || a.id === want)) return a;
    if (a) {
      // the engine moved it: free the agent telemetry had picked
      a.ticket = null;
      this.set(a, 'idle', at, {});
    }
    if (t.state === 'waiting') t.state = 'running';
    const b = this.bind(t, want, undefined, at);
    this.set(b, 'writing', at, { attempt: Math.max(1, t.attempt), kind: 'write' });
    return b;
  }

  private bind(t: TicketRow, agentId: string | undefined, machine: string | undefined, at: number): AgentStatus {
    let a = agentId ? this.agents.find((x) => x.id === agentId) : undefined;
    if (agentId && !a) {
      const team = /^[a-z]+/.exec(agentId)?.[0] ?? '?';
      a = { id: agentId, role: 'worker', team, machine: machine ?? this.crew.teams.find((x) => x.id === team)?.machine ?? this.crew.machines[0]!.id, state: 'idle', ticket: null, owns: [], attempt: 0, maxAttempts: 0, kind: null, slot: null, reason: null, tokS: 0, lastCallTokS: null, lastNote: '', since: at, calls: 0 };
      this.agents.push(a);
    }
    if (!a) a = this.pick();
    if (machine) a.machine = machine;
    a.ticket = t.id;
    a.owns = t.owns;
    a.maxAttempts = t.maxAttempts;
    a.lastNote = '';
    a.reason = null;
    t.agent = a.id;
    return a;
  }

  /** An idle agent of the least busy team whose machine has a free slot (any free agent if none). */
  private pick(): AgentStatus {
    const free = this.agents.filter((a) => !WORKING.includes(a.state));
    const busy = (pred: (a: AgentStatus) => boolean) => this.agents.filter((a) => pred(a) && WORKING.includes(a.state)).length;
    const slotFree = (m: string) => this.agents.filter((a) => a.machine === m && CALLING.includes(a.state)).length < (this.crew.machines.find((x) => x.id === m)?.slots ?? Infinity);
    const pool = free.filter((a) => slotFree(a.machine));
    const from = pool.length ? pool : free;
    if (!from.length) {
      // more tickets running than agents: grow the last team
      const t = this.crew.teams[this.crew.teams.length - 1]!;
      const a: AgentStatus = { id: `${t.id}${this.agents.filter((x) => x.team === t.id).length + 1}`, role: 'worker', team: t.id, machine: t.machine, state: 'idle', ticket: null, owns: [], attempt: 0, maxAttempts: 0, kind: null, slot: null, reason: null, tokS: 0, lastCallTokS: null, lastNote: '', since: this.now(), calls: 0 };
      this.agents.push(a);
      return a;
    }
    // prefer never-used agents in the least busy team, so the work spreads over a–d
    const load = (team: string) => busy((a) => a.team === team);
    return [...from].sort((x, y) => load(x.team) - load(y.team) || (x.state === 'idle' ? 0 : 1) - (y.state === 'idle' ? 0 : 1) || this.agents.indexOf(x) - this.agents.indexOf(y))[0]!;
  }

  private set(a: AgentStatus, state: AgentState, at: number, f: { attempt?: number; kind?: 'write' | 'fix' }): void {
    if (a.state !== state) a.since = at;
    a.state = state;
    if (f.attempt !== undefined) a.attempt = f.attempt;
    if (f.kind !== undefined) a.kind = f.kind;
  }

  /** After a failed attempt: the next one is a fix, if any are left. */
  private next(t: TicketRow, a: AgentStatus, attempt: number, at: number): void {
    if (this.stated) return; // the engine's state events say what comes next
    if (attempt < t.maxAttempts) {
      t.attempt = attempt + 1;
      this.set(a, 'fixing', at, { attempt: attempt + 1, kind: 'fix' });
    }
  }

  private ticketStatuses(): TicketStatus[] {
    const rows = [...this.tickets.values()];
    const holder = new Map<string, string>();
    for (const t of rows) if (t.state === 'running') for (const p of t.owns) holder.set(p, t.id);
    return rows.map(({ started: _s, ...t }) => {
      const out: TicketStatus = { ...t, owns: [...t.owns], dependsOn: [...t.dependsOn] };
      if (t.state === 'running' && this.agents.find((a) => a.id === t.agent && a.ticket === t.id)?.state === 'queued') out.state = 'queued';
      const met = (d: string) => {
        const s = this.tickets.get(d)?.state;
        return s === 'done' || (this.settledWith.get(t.id)?.has(d) === true && s !== undefined && s !== 'waiting' && s !== 'running');
      };
      if (t.state === 'waiting' && !this.ended && t.dependsOn.every(met)) {
        const clash = t.owns.map((p) => holder.get(p)).find(Boolean);
        out.state = clash ? 'blocked' : 'ready';
        if (clash) out.blockedBy = clash;
      }
      return out;
    });
  }

  private record(r: CallRecord): void {
    addCall(this.summary, r);
    if (this.callsPath) this.safely('calls.jsonl', () => appendFileSync(this.callsPath!, JSON.stringify(r) + '\n'));
    this.checkLost(false);
  }

  /**
   * A log that cannot be written (disk full, run folder removed) never stops the run: it warns
   * once per file. A calls.jsonl short of lines makes harvest treat the run's calls as unclean.
   */
  private safely(what: string, write: () => void): void {
    try {
      write();
    } catch (e) {
      if (this.writeFailed.has(what)) return;
      this.writeFailed.add(what);
      const w = `could not write ${what}: ${(e as Error).message ?? e}`;
      this.warnings.push(w);
      this.warnSink?.(w);
    }
  }

  private checkLost(final: boolean): void {
    if (this.warnedLost || (!final && this.summary.calls < (this.o.warnAfter ?? 50))) return;
    const w = lostWarning(this.summary);
    if (!w) return;
    this.warnedLost = true;
    this.warnings.push(w);
    this.warnSink?.(w);
  }

  private touch(): void {
    if (!this.statusPath) return;
    const every = this.o.statusIntervalMs ?? 1000;
    const wait = this.lastWrite + every - this.now();
    if (wait <= 0) this.write(this.status());
    else if (!this.timer) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.write(this.status());
      }, wait);
      (this.timer as any).unref?.();
    }
  }

  private write(s: CrewStatus): void {
    this.lastWrite = this.now();
    if (!this.statusPath) return;
    const tmp = `${this.statusPath}.tmp`;
    this.safely('status.json', () => {
      writeFileSync(tmp, JSON.stringify(s) + '\n');
      renameSync(tmp, this.statusPath!);
    });
  }
}

/** A run id from the clock: 20261015-174859. */
export function runId(at = Date.now()): string {
  return new Date(at).toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
}

function overlap(s: Span, from: number, to: number): number {
  const len = Math.max(1, s.end - s.start);
  const o = Math.min(s.end, to) - Math.max(s.start, from);
  if (s.end === s.start) return s.end > from && s.end <= to ? s.tokens : 0;
  return o > 0 ? (s.tokens * o) / len : 0;
}

const round1 = (x: number) => Math.round(x * 10) / 10;

/** The console's status bar from a status: `● working | agents 14/40 · slots 12/12 | 812 tok/s | 3m02s`. */
export function statusLine(s: CrewStatus): string {
  const slots = s.machines.reduce((n, m) => n + m.slots, 0);
  const used = s.machines.reduce((n, m) => n + m.slotsUsed, 0);
  const secs = Math.round(s.elapsedMs / 1000);
  const time = secs >= 60 ? `${Math.floor(secs / 60)}m${String(secs % 60).padStart(2, '0')}s` : `${secs}s`;
  const t = s.crew.tickets;
  const tok = s.tokS.state === 'idle' ? '--' : String(Math.round(s.tokS.total));
  return `${s.crew.state === 'working' ? '●' : '○'} ${s.crew.state.replace('_', ' ')} | agents ${s.crew.agents.working}/${s.crew.agents.total} · slots ${used}/${slots} | ${tok} tok/s | tickets ${t.done}/${t.total} done${t.failed ? `, ${t.failed} failed` : ''} | ${time}`;
}
