/**
 * What the engine tells the world while it runs a plan. The scheduler emits these; the run
 * log, the local API's event stream and the TUI all read the same shapes.
 */
import type { Ticket } from '../tickets/schema.ts';

export type TicketState = 'waiting' | 'running' | 'done' | 'failed' | 'skipped';

export type EngineEvent =
  | { type: 'plan'; at: number; tickets: Array<Pick<Ticket, 'id' | 'title' | 'depends_on' | 'model' | 'image'>>; concurrency: number }
  | { type: 'start'; at: number; ticket: string; attempt: number; worktree: string }
  | { type: 'model'; at: number; ticket: string; attempt: number; kind: 'write' | 'fix'; promptTokens: number; completionTokens: number; ms: number; note: string }
  | { type: 'refused'; at: number; ticket: string; attempt: number; paths: string[] }
  | { type: 'check'; at: number; ticket: string; attempt: number; command: string; ok: boolean; ms: number; tail: string }
  | { type: 'finish'; at: number; ticket: string; state: 'done' | 'failed' | 'skipped'; attempts: number; error?: string; commit?: string }
  | { type: 'integrate'; at: number; ok: boolean; branch: string; merged: string[]; failed: string[]; error?: string }
  | { type: 'log'; at: number; level: 'info' | 'warn' | 'error'; message: string }
  | { type: 'end'; at: number; summary: RunSummary };

export interface RunSummary {
  tickets: number;
  done: number;
  failed: number;
  skipped: number;
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
  tickets: Array<{ id: string; title: string; model: string; state: TicketState; attempt: number; lastNote: string; error?: string }>;
  concurrency: number;
  running: number;
  startedAt: number | null;
  summary: RunSummary | null;
  log: string[];
}

export type EventSink = (e: EngineEvent) => void;

export function emptyView(): RunView {
  return { tickets: [], concurrency: 0, running: 0, startedAt: null, summary: null, log: [] };
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
      v.tickets = e.tickets.map((x) => ({ id: x.id, title: x.title, model: x.model, state: 'waiting', attempt: 0, lastNote: '' }));
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
    case 'refused':
      line(`${e.ticket} #${e.attempt} refused ${e.paths.join(', ')}`);
      break;
    case 'check':
      line(`${e.ticket} #${e.attempt} ${e.ok ? 'ok' : 'FAIL'} ${e.command}`);
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
  v.running = v.tickets.filter((x) => x.state === 'running').length;
  return v;
}
