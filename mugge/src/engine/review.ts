/**
 * The lead review: after a ticket's checks pass, one fresh one-shot call shows the lead the
 * ticket, the diff of its owned files and the check output, and gets back
 * {"verdict": "approve" | "send_back", "reason": "..."} (constrained JSON; the designs' shape,
 * picked for Qwen3.8 on 2026-10-18 over the earlier {verdict, problems, fix}, which let an answer
 * approve and list problems at once). Inside the engine and in `review` events a send-back is
 * verdict "problems" with the reason as its one problem, so the console and telemetry read it
 * as before.
 *
 * The review call goes out with thinking off unless MUGGE_REVIEW_THINKING=on: the answer is a
 * short JSON object, and a thinking lead would need far more than its 2,048-token cap.
 *
 * The review is part of done: a ticket finishes on an approve, or when a fix the review asked
 * for passes the checks. A `problems` verdict with something to fix spends the ticket's next fix
 * attempt, sent as an ordinary fix call whose failing command is `(lead review)`. On by default
 * (Oliver, 2026-10-16); MUGGE_REVIEW=off turns it off for runs that do not need it. BH0.1 is
 * trained for worker calls only, so a BH0.1 crew reviews with an untrained call shape.
 *
 * The prompt lives in prompts.ts (REVIEW_SYSTEM, reviewMessages) next to the worker prompts so
 * the train harness builds review rows from the same code.
 */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import type { Ticket } from '../tickets/schema.ts';
import { git } from './git.ts';
import type { Completion, Inference } from './inference.ts';
import { REVIEW_JSON_SCHEMA, reviewMessages } from './prompts.ts';

export interface LeadReview {
  verdict: 'approve' | 'problems';
  /** One line each. */
  problems: string[];
  /** What the coder should change; empty when there is nothing to fix. */
  fix: string;
}

export interface ReviewOutcome {
  /** `error` when the call failed or the answer did not parse; the harness asks once more, then fails the ticket. */
  verdict: LeadReview['verdict'] | 'error';
  problems: string[];
  fix: string;
  error?: string;
  promptTokens: number;
  completionTokens: number;
  reasoningTokens?: number;
  cut?: boolean;
  ms: number;
}

/** Most tokens the review may write when MUGGE_MAX_CALL_TOKENS is unset; the answer is a short JSON object. */
export const REVIEW_MAX_TOKENS = 2048;
/** Most characters of diff the review sees, so one huge file cannot crowd out the rest. */
const MAX_DIFF_CHARS = 40_000;
/** Most characters of each passing command's output. */
const MAX_CHECK_CHARS = 600;

/** MUGGE_REVIEW: on | off (unset: on). */
export function reviewFromEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = env.MUGGE_REVIEW?.trim().toLowerCase();
  if (v && v !== 'on' && v !== 'off') throw new Error('MUGGE_REVIEW must be on or off');
  return v !== 'off';
}

/** MUGGE_REVIEW_THINKING: on | off (unset: off). */
export function reviewThinkingFromEnv(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = env.MUGGE_REVIEW_THINKING?.trim().toLowerCase();
  if (v && v !== 'on' && v !== 'off') throw new Error('MUGGE_REVIEW_THINKING must be on or off');
  return v === 'on';
}

/**
 * Most tokens the review may write: MUGGE_MAX_CALL_TOKENS when it is set, so a lead thinking
 * before its answer is not cut at 2,048 (Oliver, 2026-10-16); else REVIEW_MAX_TOKENS. The
 * ticket's tokens left still bound it.
 */
export function reviewCapFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const v = env.MUGGE_MAX_CALL_TOKENS?.trim();
  if (!v) return REVIEW_MAX_TOKENS;
  const n = Number(v);
  if (!(Number.isInteger(n) && n > 0)) throw new Error('MUGGE_MAX_CALL_TOKENS must be a positive integer');
  return n;
}

/** The owned files' contents; null for a file that does not exist. */
export type Snapshot = Record<string, string | null>;

export function snapshot(worktree: string, paths: string[]): Snapshot {
  const out: Snapshot = {};
  for (const p of paths) out[p] = existsSync(join(worktree, p)) ? readFileSync(join(worktree, p), 'utf8') : null;
  return out;
}

/**
 * A unified diff (git's format, `a/` and `b/` paths) of the owned files from `before` to their
 * state in the worktree. Works in any folder, git repo or not (the bench's are not).
 */
export function diffOwned(worktree: string, before: Snapshot): string {
  const after = snapshot(worktree, Object.keys(before));
  const changed = Object.keys(before).filter((p) => before[p] !== after[p]);
  if (!changed.length) return '';
  const tmp = mkdtempSync(join(tmpdir(), 'mugge-review-'));
  try {
    for (const [side, snap] of [['a', before], ['b', after]] as const) {
      mkdirSync(join(tmp, side));
      for (const p of changed) {
        if (snap[p] === null) continue;
        mkdirSync(dirname(join(tmp, side, p)), { recursive: true });
        writeFileSync(join(tmp, side, p), snap[p]!);
      }
    }
    const r = git(tmp, ['diff', '--no-index', '--no-color', '--no-prefix', '--', 'a', 'b']);
    // git diff --no-index exits 1 when the trees differ
    if (r.code > 1) throw new Error(r.out);
    return r.out.length > MAX_DIFF_CHARS ? r.out.slice(0, MAX_DIFF_CHARS) + '\n…(truncated)' : r.out;
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

/**
 * The answer as a LeadReview, or a reason it is not one. Takes the wire shape
 * {verdict: approve | send_back, reason}; the older {verdict: approve | problems, problems, fix}
 * still reads, for run logs and canned reviews written before the change.
 */
export function normalizeReview(v: any): LeadReview | string {
  if (!v || typeof v !== 'object') return 'review answer is not an object';
  if (v.verdict === 'send_back' || (v.verdict === 'approve' && 'reason' in v)) {
    const reason = typeof v.reason === 'string' ? v.reason.trim() : '';
    if (v.verdict === 'approve') return { verdict: 'approve', problems: [], fix: '' };
    return { verdict: 'problems', problems: [reason || 'The lead sent it back without a reason; check the ticket again.'], fix: '' };
  }
  if (v.verdict !== 'approve' && v.verdict !== 'problems') return `review verdict ${JSON.stringify(v.verdict)} is not approve or send_back`;
  const problems = Array.isArray(v.problems) ? v.problems.filter((p: unknown) => typeof p === 'string' && p.trim()).map((p: string) => p.trim()) : [];
  const fix = typeof v.fix === 'string' ? v.fix.trim() : '';
  return { verdict: v.verdict, problems, fix };
}

/** A LeadReview as the model writes it (REVIEW_JSON_SCHEMA), for the fake model and tests. */
export function reviewAnswer(r: LeadReview): { verdict: 'approve' | 'send_back'; reason: string } {
  if (r.verdict === 'approve') return { verdict: 'approve', reason: '' };
  return { verdict: 'send_back', reason: [...r.problems, ...(r.fix ? [r.fix] : [])].join('; ') };
}

/** Whether a review asks for a fix attempt: problems, with something to fix. */
export function wantsFix(r: ReviewOutcome): boolean {
  return r.verdict === 'problems' && (r.fix !== '' || r.problems.length > 0);
}

/** The failing output the review's fix call carries (in place of a failing command's output). */
export function reviewFeedback(r: Pick<LeadReview, 'problems' | 'fix'>): string {
  return [
    'The lead reviewed your passing files and sent them back:',
    ...r.problems.map((p) => `- ${p}`),
    ...(r.fix ? ['', `Suggested fix: ${r.fix}`] : []),
  ].join('\n');
}

export interface ReviewInput {
  ticket: Ticket;
  /** diffOwned's output. */
  diff: string;
  /** Every acceptance command that passed, with the tail of its output. */
  checks: Array<{ command: string; output: string }>;
}

/** `maxTokens`: the ticket's tokens left; `cap`: reviewCapFromEnv's answer unless the caller sets one; `thinking`: sent as the call's thinking switch. */
export async function review(inference: Inference, model: string, input: ReviewInput, maxTokens: number, cap: number = REVIEW_MAX_TOKENS, thinking = false): Promise<ReviewOutcome> {
  const checks = input.checks.map((c) => ({ command: c.command, output: c.output.length > MAX_CHECK_CHARS ? '…' + c.output.slice(-MAX_CHECK_CHARS) : c.output }));
  const messages = reviewMessages(input.ticket, input.diff, checks);
  let c: Completion<unknown>;
  try {
    c = await inference.complete<unknown>(model, messages, REVIEW_JSON_SCHEMA, Math.min(maxTokens, cap), undefined, { thinking });
  } catch (e) {
    // an answer that came back but is not JSON (most often cut at the cap) still cost its tokens
    const got = (e as { completion?: Completion<string> }).completion;
    const usage = got ? { promptTokens: got.promptTokens, completionTokens: got.completionTokens, reasoningTokens: got.reasoningTokens, cut: got.cut, ms: got.ms } : { promptTokens: 0, completionTokens: 0, ms: 0 };
    return { verdict: 'error', problems: [], fix: '', error: String((e as Error).message ?? e), ...usage };
  }
  const usage = { promptTokens: c.promptTokens, completionTokens: c.completionTokens, reasoningTokens: c.reasoningTokens, cut: c.cut, ms: c.ms };
  const r = normalizeReview(c.value);
  if (typeof r === 'string') return { verdict: 'error', problems: [], fix: '', error: r, ...usage };
  return { ...r, ...usage };
}
