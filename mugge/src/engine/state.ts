/**
 * What every ticket, and the slot (agent) working it, is doing right now. The harness and the
 * scheduler emit a `state` event on every change; the run view, the console and the smoke
 * command fold them (events.ts). The states are the Centipede console's:
 *
 *   idle       an agent with no ticket (agents only)
 *   waiting    not started: on a ticket it depends on, a file a running ticket holds, or a free agent
 *   queued     its agent's model call waits for a free slot on a machine (backends.ts)
 *   writing    attempt 1: the write call
 *   thinking   a model call while thinking is on (MUGGE_THINKING=on). Calls do not stream, so
 *              the whole call shows as thinking; the debug line says how much of it was
 *   checking   its acceptance commands are running
 *   fixing     attempt n of N: a fix call with the failing output
 *   reviewing  the lead reviews the passing diff (MUGGE_REVIEW=on only)
 *   blocked    needs the owner: two tickets claim the same file and the plan blocks it (clashes:
 *              block; by default the planner settles a clash and the later ticket waits), or a
 *              dependency failed
 *   finished   its checks pass, and with MUGGE_REVIEW=on the lead approved or its fix passed
 *   failed     every attempt or the token budget used, or the run was stopped
 *
 * Every attempt is one fresh one-shot call; nothing but the attempt counter (and, for a fix,
 * the failing output) passes from one attempt to the next, and nothing from one ticket to another.
 */

export const AGENT_STATES = ['idle', 'queued', 'waiting', 'writing', 'thinking', 'checking', 'fixing', 'reviewing', 'blocked', 'finished', 'failed'] as const;
export type AgentState = (typeof AGENT_STATES)[number];

/** States a ticket never leaves. */
export const FINAL_STATES: readonly AgentState[] = ['blocked', 'finished', 'failed'];

/** States in which an agent works the ticket. */
export const ACTIVE_STATES: readonly AgentState[] = ['queued', 'writing', 'thinking', 'checking', 'fixing', 'reviewing'];

/** Where each state may go next. `idle` is a slot's state and never a ticket's. */
export const NEXT: Record<AgentState, readonly AgentState[]> = {
  idle: [],
  waiting: ['waiting', 'writing', 'thinking', 'blocked', 'failed'],
  // back to the call it was waiting to make, or on when the call fails
  queued: ['writing', 'thinking', 'fixing', 'reviewing', 'failed'],
  writing: ['queued', 'checking', 'fixing', 'thinking', 'failed'],
  thinking: ['queued', 'checking', 'fixing', 'thinking', 'failed'],
  fixing: ['queued', 'checking', 'fixing', 'thinking', 'failed'],
  checking: ['fixing', 'thinking', 'reviewing', 'finished', 'failed'],
  reviewing: ['queued', 'fixing', 'thinking', 'finished', 'failed'],
  blocked: [],
  finished: [],
  failed: [],
};

/** Whether a ticket may move from `from` to `to`; a ticket's first state may be any but idle. */
export function canMove(from: AgentState | undefined, to: AgentState): boolean {
  if (from === undefined) return to !== 'idle';
  return NEXT[from].includes(to);
}

/** The state of a model call: thinking when it is on, else writing for attempt 1 and fixing after. */
export function callState(kind: 'write' | 'fix', thinking: boolean): AgentState {
  return thinking ? 'thinking' : kind === 'write' ? 'writing' : 'fixing';
}
