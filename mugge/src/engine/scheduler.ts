/**
 * Runs a plan: claims every ticket whose dependencies are done, as long as a slot is free and
 * no running ticket owns one of its files, gives it its own worktree and runs the harness loop.
 * When every ticket has ended it integrates the finished branches. A ticket that claims a file
 * an earlier ticket owns while both could run at once is blocked for the owner, never run.
 * Tickets not yet running are `waiting` (on a dependency, a held file or a free agent); every
 * state event of a running ticket carries its agent number (0 … concurrency-1).
 *
 * `concurrency` is how many tickets (agents) run at once. Every model call a ticket makes takes a
 * slot on a machine of the backend pool (backends.ts) for the length of the call; a ticket whose
 * call is waiting for a slot is "queued". Without a pool the crew is one machine with
 * `concurrency` slots, so a ticket never queues and nothing changes. With more agents than
 * worker slots, some agents run their checks while others use the model.
 */
import { clashReason, findOwnershipClashes, type Plan, type Ticket } from '../tickets/schema.ts';
import type { EngineEvent, EventSink, RunSummary } from './events.ts';
import { branchFor, DEFAULT_AUTHOR, Worktrees, type Author } from './git.ts';
import { runTicket, type TicketResult } from './harness.ts';
import { BackendPool } from './backends.ts';
import type { Inference } from './inference.ts';
import { integrate, type IntegrateResult } from './integrate.ts';
import type { Sandbox } from './sandbox.ts';

export interface RunOptions {
  repo: string;
  /** Commit every ticket without dependencies starts from: the scaffold. */
  base: string;
  plan: Plan;
  /** The model when the crew is one machine; ignored when `pool` is given. */
  inference?: Inference;
  /** The crew's machines and their slots (config.ts poolFromEnv). */
  pool?: BackendPool;
  /** Which team a ticket's agent counts under in the pool's stats (default: its model). */
  teamOf?: (t: Ticket) => string;
  sandbox: Sandbox;
  /** Folder for the worktrees, outside the repo. */
  worktreeRoot: string;
  concurrency: number;
  emit?: EventSink;
  author?: Author;
  memory?: string;
  /** Run the integrate step at the end (default true). */
  integrate?: boolean;
  checkTimeoutMs?: number;
  signal?: AbortSignal;
}

export interface RunReport {
  summary: RunSummary;
  results: Map<string, TicketResult>;
  states: Map<string, 'done' | 'failed' | 'skipped' | 'blocked'>;
  integration: IntegrateResult | null;
  /** Tickets blocked because two tickets need the same file. */
  clashes: ReturnType<typeof findOwnershipClashes>;
}

export async function runPlan(o: RunOptions): Promise<RunReport> {
  const emit: EventSink = o.emit ?? (() => {});
  if (!o.pool && !o.inference) throw new Error('runPlan needs inference or a pool');
  const pool = o.pool ?? BackendPool.single(o.inference!, o.concurrency);
  const teamOf = o.teamOf ?? ((t: Ticket) => t.model.split('@')[0]);
  type SlotHook = (e: { state: 'queued' | 'got' }) => void;
  const agentFor = (t: Ticket, role: 'worker' | 'lead' = 'worker', hook?: SlotHook) => pool.forAgent({
    agent: t.id, team: teamOf(t), role, signal: o.signal,
    onSlot: (e) => {
      emit({ type: 'slot', at: Date.now(), ticket: t.id, state: e.state, machine: e.machine, waitedMs: e.waitedMs });
      hook?.(e);
    },
  });
  // The lead review goes to a machine serving leads when the crew has one; same model either way.
  const reviewRole = pool.slotsFor('lead') > 0 ? 'lead' : 'worker';
  const author = o.author ?? DEFAULT_AUTHOR;
  const wt = new Worktrees(o.repo, o.worktreeRoot, author);
  const started = Date.now();
  const tickets = [...o.plan.tickets].sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id));
  const states = new Map<string, 'waiting' | 'running' | 'done' | 'failed' | 'skipped' | 'blocked'>(tickets.map((t) => [t.id, 'waiting']));
  const agents: Array<string | null> = Array.from({ length: o.concurrency }, () => null);
  const shown = new Map<string, string>(); // ticket → its last state before it runs, so each change is emitted once
  const waitState = (t: Ticket, state: 'waiting' | 'blocked' | 'failed', extra: { reason?: string; waitingOn?: string[] } = {}) => {
    const key = JSON.stringify([state, extra]);
    if (shown.get(t.id) === key) return;
    shown.set(t.id, key);
    emit({ type: 'state', at: Date.now(), ticket: t.id, state, attempt: 0, of: t.budget.attempts * (t.escalate_to ? 2 : 1), ...extra });
  };
  const results = new Map<string, TicketResult>();
  const running = new Map<string, Promise<void>>();
  let peak = 0, serialMs = 0;

  emit({ type: 'plan', at: started, tickets: tickets.map(({ id, title, depends_on, model, image }) => ({ id, title, depends_on, model, image })), concurrency: o.concurrency, slots: pool.slotsFor('worker') });

  const finish = (id: string, state: 'done' | 'failed' | 'skipped' | 'blocked', extra: Partial<Extract<EngineEvent, { type: 'finish' }>> = {}) => {
    states.set(id, state);
    emit({ type: 'finish', at: Date.now(), ticket: id, state, attempts: results.get(id)?.attempts ?? 0, ...extra });
  };

  const ownedByRunning = () => new Set(tickets.filter((t) => states.get(t.id) === 'running').flatMap((t) => t.owns));

  // plan order decides who keeps a file: the earlier ticket
  const clashes = findOwnershipClashes(o.plan.tickets);
  for (const c of clashes) {
    const t = tickets.find((x) => x.id === c.ticket)!;
    emit({ type: 'clash', at: Date.now(), ...c });
    waitState(t, 'blocked', { reason: clashReason(c) });
    finish(t.id, 'blocked', { error: clashReason(c) });
  }

  const start = (t: Ticket) => {
    states.set(t.id, 'running');
    const agent = agents.indexOf(null);
    agents[agent] = t.id;
    // Every state the harness emits carries the agent; while a call waits for a slot it shows queued.
    let last: Extract<EngineEvent, { type: 'state' }> | null = null, queued = false;
    const forAgent: EventSink = (e) => {
      if (e.type === 'state') e = last = { ...e, agent };
      emit(e);
    };
    const onSlot: SlotHook = (e) => {
      if (!last || (e.state === 'got') !== queued) return;
      queued = e.state === 'queued';
      emit({ ...last, at: Date.now(), state: queued ? 'queued' : last.state });
    };
    peak = Math.max(peak, [...states.values()].filter((s) => s === 'running').length);
    const p = (async () => {
      const t0 = Date.now();
      const branch = branchFor(t.id);
      let path: string;
      try {
        path = wt.create(t.id, branch, o.base, t.depends_on.map(branchFor));
      } catch (e) {
        finish(t.id, 'failed', { error: String((e as Error).message) });
        return;
      }
      emit({ type: 'start', at: Date.now(), ticket: t.id, attempt: 1, worktree: path });
      try {
        const r = await runTicket(t, path, { plan: o.plan, inference: agentFor(t, 'worker', onSlot), reviewInference: agentFor(t, reviewRole, onSlot), sandbox: o.sandbox, emit: forAgent, memory: o.memory, checkTimeoutMs: o.checkTimeoutMs, signal: o.signal });
        results.set(t.id, r);
        if (r.ok) {
          const commit = wt.commit(path, t.owns, t.title) ?? undefined;
          finish(t.id, 'done', { commit });
        } else {
          finish(t.id, 'failed', { error: r.error ?? 'failed' });
        }
      } catch (e) {
        finish(t.id, 'failed', { error: String((e as Error).message ?? e) });
      } finally {
        serialMs += Date.now() - t0;
        agents[agent] = null;
        wt.remove(t.id);
        pool.retire(t.id);
      }
    })();
    running.set(t.id, p.finally(() => running.delete(t.id)));
  };

  while (true) {
    // A ticket whose dependency failed or was skipped can never run.
    let changed = true;
    while (changed) {
      changed = false;
      for (const t of tickets) {
        if (states.get(t.id) !== 'waiting') continue;
        const bad = t.depends_on.find((d) => states.get(d) === 'failed' || states.get(d) === 'skipped' || states.get(d) === 'blocked');
        if (bad) {
          waitState(t, 'blocked', { reason: `dependency ${bad} did not finish`, waitingOn: [bad] });
          finish(t.id, 'skipped', { error: `dependency ${bad} did not finish` });
          changed = true;
        }
      }
    }
    if (!o.signal?.aborted) {
      for (const t of tickets) {
        if (states.get(t.id) !== 'waiting') continue;
        const deps = t.depends_on.filter((d) => states.get(d) !== 'done');
        if (deps.length) {
          waitState(t, 'waiting', { waitingOn: deps });
          continue;
        }
        const busy = ownedByRunning();
        if (t.owns.some((p) => busy.has(p))) {
          const holders = tickets.filter((x) => states.get(x.id) === 'running' && x.owns.some((p) => t.owns.includes(p))).map((x) => x.id);
          waitState(t, 'waiting', { reason: 'a running ticket holds one of its files', waitingOn: holders });
          continue;
        }
        if (running.size >= o.concurrency) {
          waitState(t, 'waiting', { reason: 'waiting for a free agent' });
          continue;
        }
        start(t);
      }
    }
    if (running.size === 0) break;
    await Promise.race(running.values());
  }
  for (const t of tickets) {
    if (states.get(t.id) !== 'waiting') continue;
    waitState(t, 'failed', { reason: 'aborted' });
    finish(t.id, 'skipped', { error: 'aborted' });
  }

  const done = tickets.filter((t) => states.get(t.id) === 'done');
  let integration: IntegrateResult | null = null;
  if (o.integrate !== false && done.length && !o.signal?.aborted) {
    integration = await integrate({ repo: o.repo, base: o.base, plan: o.plan, done: done.map((t) => t.id), sandbox: o.sandbox, worktrees: wt, checkTimeoutMs: o.checkTimeoutMs,
      fixer: (fix, path) => runTicket(fix, path, { plan: o.plan, inference: agentFor(fix), sandbox: o.sandbox, emit, memory: o.memory, checkTimeoutMs: o.checkTimeoutMs, signal: o.signal }),
    });
    emit({ type: 'integrate', at: Date.now(), ok: integration.ok, branch: integration.branch, merged: integration.merged, failed: integration.failedChecks, error: integration.error ?? undefined });
  }

  const all = [...results.values()];
  const count = (s: string) => [...states.values()].filter((x) => x === s).length;
  const summary: RunSummary = {
    tickets: tickets.length,
    done: count('done'),
    failed: count('failed'),
    skipped: count('skipped'),
    blocked: count('blocked'),
    firstTry: all.filter((r) => r.ok && r.firstTry).length,
    calls: all.reduce((n, r) => n + r.attempts, 0),
    promptTokens: all.reduce((n, r) => n + r.promptTokens, 0),
    completionTokens: all.reduce((n, r) => n + r.completionTokens, 0),
    peakParallel: peak,
    wallMs: Date.now() - started,
    serialMs,
    inferenceMs: all.reduce((n, r) => n + r.inferenceMs, 0),
    checkMs: all.reduce((n, r) => n + r.checkMs, 0),
    integrated: integration ? integration.ok : null,
  };
  emit({ type: 'end', at: Date.now(), summary });
  return { summary, results, states: states as RunReport['states'], integration, clashes };
}
