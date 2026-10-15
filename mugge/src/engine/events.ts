/**
 * What the engine tells the world while it runs a plan. The scheduler emits these; the run
 * log, the local API's event stream and the TUI all read the same shapes.
 */
import type { OwnershipClash, Ticket } from '../tickets/schema.ts';
import { ACTIVE_STATES, type AgentState } from './state.ts';

/**
 * A ticket's coarse state; `phase` (state.ts) says what it is doing within it. `queued`: running,
 * but its model call is waiting for a free slot (backends.ts). `blocked`: needs the owner.
 */
export type TicketState = 'waiting' | 'running' | 'queued' | 'done' | 'failed' | 'skipped' | 'blocked';

export type EngineEvent =
  | { type: 'plan'; at: number; tickets: Array<Pick<Ticket, 'id' | 'title' | 'depends_on' | 'model' | 'image'>>; concurrency: number; slots?: number }
  /** A ticket's model call waits for a slot (`queued`) or gets one (`got`, on `machine`). */
  | { type: 'slot'; at: number; ticket: string; state: 'queued' | 'got'; machine?: string; waitedMs?: number }
  | { type: 'start'; at: number; ticket: string; attempt: number; worktree: string }
  | { type: 'model'; at: number; ticket: string; attempt: number; kind: 'write' | 'fix'; promptTokens: number; completionTokens: number; reasoningTokens?: number; cut?: boolean; ms: number; note: string; files?: number }
  | { type: 'refused'; at: number; ticket: string; attempt: number; paths: string[] }
  | { type: 'check'; at: number; ticket: string; attempt: number; command: string; ok: boolean; ms: number; tail: string }
  | { type: 'finish'; at: number; ticket: string; state: 'done' | 'failed' | 'skipped' | 'blocked'; attempts: number; error?: string; commit?: string }
  /** Every change of a ticket's state (state.ts). `attempt` of `of`; `agent` while one works it. */
  | { type: 'state'; at: number; ticket: string; state: AgentState; attempt: number; of: number; agent?: number; reason?: string; waitingOn?: string[] }
  /** A ticket blocked because it claims a file an earlier ticket owns (schema.findOwnershipClashes). */
  | { type: 'clash'; at: number; ticket: string; files: string[]; with: string[] }
  /** The lead review of a passing ticket (review.ts); `error` when the call failed or did not parse. */
  | { type: 'review'; at: number; ticket: string; attempt: number; verdict: 'approve' | 'problems' | 'error'; problems: string[]; fix: string; error?: string; promptTokens: number; completionTokens: number; reasoningTokens?: number; cut?: boolean; ms: number }
  | { type: 'integrate'; at: number; ok: boolean; branch: string; merged: string[]; failed: string[]; error?: string }
  | { type: 'log'; at: number; level: 'info' | 'warn' | 'error'; message: string }
  | { type: 'end'; at: number; summary: RunSummary };

export interface RunSummary {
  tickets: number;
  done: number;
  failed: number;
  skipped: number;
  /** Blocked by an ownership clash, waiting for the owner. */
  blocked?: number;
  /** Done on the first model call. */
  firstTry: number;
  /** Model calls made in total. */
  calls: number;
  promptTokens: number;
  completionTokens: number;
  /** Most tickets running at the same moment. */
  peakParallel: number;
  wallMs: number;
  /** Sum of every ticket's own time: what running them one at a time would roughly take. */
  serialMs: number;
  inferenceMs: number;
  checkMs: number;
  integrated: boolean | null;
}

/** A snapshot built by folding events; what the TUI draws. */
export interface RunView {
  tickets: Array<{
    id: string; title: string; model: string; state: TicketState; attempt: number; lastNote: string; error?: string;
    /** What it is doing now (state.ts); `of` is its attempt budget, `agent` the agent working it. */
    phase?: AgentState; of?: number; agent?: number; reason?: string; waitingOn?: string[];
    review?: 'approve' | 'problems' | 'error';
  }>;
  /** Tickets blocked because two tickets need the same file. */
  clashes?: OwnershipClash[];
  concurrency: number;
  running: number;
  startedAt: number | null;
  summary: RunSummary | null;
  log: string[];
}

export type EventSink = (e: EngineEvent) => void;

export function emptyView(): RunView {
  return { tickets: [], clashes: [], concurrency: 0, running: 0, startedAt: null, summary: null, log: [] };
}

/** Folds one event into a view (mutates and returns it). */
export function applyEvent(v: RunView, e: EngineEvent): RunView {
  const t = 'ticket' in e && typeof e.ticket === 'string' ? v.tickets.find((x) => x.id === e.ticket) : undefined;
  const line = (s: string) => {
    v.log.push(s);
    if (v.log.length > 200) v.log.splice(0, v.log.length - 200);
  };
  switch (e.type) {
    case 'plan':
      v.tickets = e.tickets.map((x) => ({ id: x.id, title: x.title, model: x.model, state: 'waiting', attempt: 0, lastNote: '', phase: 'waiting', of: 0 }));
      v.concurrency = e.concurrency;
      v.startedAt = e.at;
      break;
    case 'start':
      if (t) { t.state = 'running'; t.attempt = e.attempt; }
      break;
    case 'model':
      if (t) { t.attempt = e.attempt; t.lastNote = e.note; }
      line(`${e.ticket} #${e.attempt} ${e.kind} ${e.completionTokens} tok ${e.note}`);
      break;
    case 'slot':
      if (t && (t.state === 'running' || t.state === 'queued')) t.state = e.state === 'queued' ? 'queued' : 'running';
      break;
    case 'refused':
      line(`${e.ticket} #${e.attempt} refused ${e.paths.join(', ')}`);
      break;
    case 'check':
      line(`${e.ticket} #${e.attempt} ${e.ok ? 'ok' : 'FAIL'} ${e.command}`);
      break;
    case 'state':
      if (t) {
        t.phase = e.state; t.of = e.of; t.agent = e.agent; t.reason = e.reason; t.waitingOn = e.waitingOn;
        if (e.attempt) t.attempt = e.attempt;
      }
      break;
    case 'clash':
      (v.clashes ??= []).push({ ticket: e.ticket, files: e.files, with: e.with });
      line(`${e.ticket} blocked: two tickets need ${e.files.join(', ')} (also ${e.with.join(', ')})`);
      break;
    case 'review':
      if (t) t.review = e.verdict;
      line(`${e.ticket} #${e.attempt} review ${e.verdict}${e.problems[0] ? `: ${e.problems[0]}` : e.error ? `: ${e.error.split('\n')[0]}` : ''}`);
      break;
    case 'finish':
      if (t) { t.state = e.state; t.attempt = e.attempts; t.error = e.error; }
      line(`${e.ticket} ${e.state}${e.error ? `: ${e.error.split('\n')[0]}` : ''}`);
      break;
    case 'integrate':
      line(`integrate ${e.ok ? 'ok' : 'FAILED'} on ${e.branch}${e.error ? `: ${e.error.split('\n')[0]}` : ''}`);
      break;
    case 'log':
      line(e.message);
      break;
    case 'end':
      v.summary = e.summary;
      break;
  }
  v.running = v.tickets.filter((x) => x.state === 'running' || x.state === 'queued').length;
  return v;
}

/** Each agent (0 … concurrency-1) and what it is doing: its ticket's state, or idle. */
export function agentStates(v: RunView): Array<{ agent: number; state: AgentState; ticket?: string; attempt?: number; of?: number }> {
  return Array.from({ length: v.concurrency }, (_, agent) => {
    const t = v.tickets.find((x) => x.agent === agent && x.phase !== undefined && ACTIVE_STATES.includes(x.phase));
    return t ? { agent, state: t.phase!, ticket: t.id, attempt: t.attempt, of: t.of } : { agent, state: 'idle' as const };
  });
}
