/**
 * The harness worker: the write → check → fix loop that runs one ticket with a single-turn
 * specialist model.
 *
 *   context pack → model writes the owned files → engine runs every acceptance command in the
 *   sandbox → on failure a fresh fix call gets the trimmed error → repeat until the budget runs
 *   out → escalate to a bigger model once, or end failed with the last error.
 *
 * The lead review is on by default (Oliver, 2026-10-16; MUGGE_REVIEW=off turns it off) and is
 * part of done (Oliver, 2026-10-15): a ticket whose
 * checks pass gets one lead review (review.ts) and finishes only on an approve, or when a fix
 * the review asked for passes the checks. Problems spend the ticket's next fix attempt (and
 * escalate like a failing check); a ticket whose attempts run out first ends failed, as does
 * one whose review call fails twice.
 *
 * Every call is one fresh one-shot message, and every step emits a `state` event (state.ts).
 * The model never decides it is done; the acceptance commands do.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { Plan, Ticket, WorkerOutput } from '../tickets/schema.ts';
import type { EventSink } from './events.ts';
import { parseFiles } from './files.ts';
import type { Completion, Inference } from './inference.ts';
import { buildPack, fixMessages, writeMessages } from './prompts.ts';
import { diffOwned, review as leadReview, reviewCapFromEnv, reviewFeedback, reviewFromEnv, reviewThinkingFromEnv, snapshot, wantsFix, type ReviewOutcome } from './review.ts';
import { tail, type Sandbox } from './sandbox.ts';
import { callState, type AgentState } from './state.ts';
import { classifyAnswer } from './telemetry.ts';

export interface HarnessContext {
  plan: Plan;
  inference: Inference;
  sandbox: Sandbox;
  emit: EventSink;
  memory?: string;
  /** Per acceptance command. */
  checkTimeoutMs?: number;
  signal?: AbortSignal;
  /** Lead review after the checks pass; unset: MUGGE_REVIEW (on unless it is off). */
  review?: boolean;
  /** Most tokens the review may write (the ticket's tokens left still bound it); unset: MUGGE_MAX_CALL_TOKENS, else 2,048. */
  reviewMaxTokens?: number;
  /** Whether the review call thinks; unset: MUGGE_REVIEW_THINKING, else off. */
  reviewThinking?: boolean;
  /** Where the review call goes (the scheduler gives a lead slot when the crew has one); unset: `inference`. */
  reviewInference?: Inference;
  /**
   * Model name for the review. Names the inference config does not map fall back to its default
   * model, which is the plan: one model for every role, the role is in the prompt.
   */
  reviewModel?: string;
  /** Whether calls think (only changes the state shown during a call); unset: the inference config's switch, else MUGGE_THINKING. */
  thinking?: boolean;
  /**
   * What the ticket's later calls do once a thinking call ran out of tokens (cut with no usable
   * file, or forced): "nothink" sends them with thinking off, "same" changes nothing. Unset:
   * MUGGE_CUT_RETRY, else "nothink".
   */
  cutRetry?: CutRetry;
}

export type CutRetry = 'nothink' | 'same';

/** MUGGE_CUT_RETRY: nothink (default) or same. */
export function cutRetryFromEnv(env: Record<string, string | undefined> = process.env): CutRetry {
  const v = env.MUGGE_CUT_RETRY?.trim().toLowerCase();
  if (!v) return 'nothink';
  if (v !== 'nothink' && v !== 'same') throw new Error('MUGGE_CUT_RETRY must be nothink or same');
  return v;
}

export interface TicketResult {
  ok: boolean;
  attempts: number;
  firstTry: boolean;
  error: string | null;
  promptTokens: number;
  completionTokens: number;
  inferenceMs: number;
  checkMs: number;
  /** The model that got it to pass (after escalation, the bigger one). */
  model: string;
  /** The lead review, when one ran. */
  review: ReviewOutcome | null;
  /** The review found problems and no fix got through the checks before the attempts ran out (the ticket failed). */
  reviewUnresolved: boolean;
  /** The attempt whose answer is in the worktree when the ticket finished; null when it failed. */
  answerAttempt: number | null;
}

/**
 * Whether the inference config turns thinking on (MUGGE_THINKING / bench --thinking), else the
 * environment: on unless MUGGE_THINKING=off, since Qwen3.8 thinks when nothing says otherwise.
 */
export function thinkingOn(inference: Inference): boolean {
  const kw = (inference as { cfg?: { extra?: { chat_template_kwargs?: { enable_thinking?: unknown } } } }).cfg?.extra?.chat_template_kwargs;
  if (typeof kw?.enable_thinking === 'boolean') return kw.enable_thinking;
  return process.env.MUGGE_THINKING?.trim().toLowerCase() !== 'off';
}

/**
 * Writes the answer's owned files into the worktree; returns the paths it refused. An answer
 * with refused paths still leaves its owned files written (the next fix call shows them), but
 * the per-call log marks it `drop: not_owned`, unclean, so it never becomes a training row.
 */
export function applyOutput(worktree: string, ticket: Ticket, out: WorkerOutput): string[] {
  const refused: string[] = [];
  for (const [path, content] of Object.entries(out?.files ?? {})) {
    if (!ticket.owns.includes(path) || typeof content !== 'string') {
      refused.push(path);
      continue;
    }
    const abs = join(worktree, path);
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, content);
  }
  return refused;
}

export async function runTicket(ticket: Ticket, worktree: string, ctx: HarnessContext): Promise<TicketResult> {
  const r: TicketResult = { ok: false, attempts: 0, firstTry: false, error: null, promptTokens: 0, completionTokens: 0, inferenceMs: 0, checkMs: 0, model: ticket.model, review: null, reviewUnresolved: false, answerAttempt: null };
  const reviewOn = ctx.review ?? reviewFromEnv();
  const thinking = ctx.thinking ?? thinkingOn(ctx.inference);
  const reviewThinking = ctx.reviewThinking ?? (reviewOn ? reviewThinkingFromEnv() : false);
  const reviewCap = ctx.reviewMaxTokens ?? (reviewOn ? reviewCapFromEnv(process.env, reviewThinking) : 0);
  const cutRetry = ctx.cutRetry ?? cutRetryFromEnv();
  // a thinking call of this ticket ran out of tokens: its later calls go out with thinking off
  let noThink = false;
  const pack = buildPack(worktree, ctx.plan, ticket, ctx.memory);
  const rounds = [ticket.model, ...(ticket.escalate_to ? [ticket.escalate_to] : [])];
  const of = ticket.budget.attempts * rounds.length;
  const state = (s: AgentState, reason?: string) => {
    ctx.emit({ type: 'state', at: Date.now(), ticket: ticket.id, state: s, attempt: r.attempts, of, ...(reason ? { reason } : {}) });
  };
  const before = reviewOn ? snapshot(worktree, ticket.owns) : {};
  let failed: { command: string; output: string } | null = null;

  const finished = (attempt: number) => {
    r.ok = true;
    r.firstTry = attempt === 1;
    r.error = null;
    r.answerAttempt = attempt;
    r.reviewUnresolved = false;
    state('finished');
    return r;
  };
  const fail = (reason: string | undefined) => {
    state('failed', reason);
    return r;
  };

  // Every acceptance command in order; the first that fails, or null when all pass (their output goes to `passing`).
  const check = async (attempt: number, passing: Array<{ command: string; output: string }> = []) => {
    for (const command of ticket.acceptance) {
      const res = await ctx.sandbox.run(command, { cwd: worktree, image: ticket.image, timeoutMs: ctx.checkTimeoutMs });
      r.checkMs += res.ms;
      const ok = res.code === 0;
      ctx.emit({ type: 'check', at: Date.now(), ticket: ticket.id, attempt, command, ok, ms: res.ms, tail: tail(res.out, 800) });
      if (!ok) return { command, output: (res.timedOut ? '(timed out)\n' : '') + tail(res.out) };
      passing.push({ command, output: res.out });
    }
    return null;
  };

  const reviewOnce = async (attempt: number, checks: Array<{ command: string; output: string }>, spent: number) => {
    const rv = await leadReview(ctx.reviewInference ?? ctx.inference, ctx.reviewModel ?? 'lead', { ticket, diff: await diffOwned(worktree, before), checks }, ticket.budget.max_tokens - spent, reviewCap, reviewThinking);
    r.promptTokens += rv.promptTokens;
    r.completionTokens += rv.completionTokens;
    r.inferenceMs += rv.ms;
    ctx.emit({ type: 'review', at: Date.now(), ticket: ticket.id, attempt, verdict: rv.verdict, problems: rv.problems, fix: rv.fix, ...(rv.error ? { error: rv.error } : {}), promptTokens: rv.promptTokens, completionTokens: rv.completionTokens, reasoningTokens: rv.reasoningTokens, cut: rv.cut, ms: rv.ms });
    return rv;
  };

  for (const model of rounds) {
    r.model = model;
    let spent = 0;
    for (let i = 0; i < ticket.budget.attempts && spent < ticket.budget.max_tokens; i++) {
      if (ctx.signal?.aborted) {
        r.error = 'aborted';
        state('failed', 'aborted');
        return r;
      }
      const attempt = ++r.attempts;
      const kind = failed ? 'fix' : 'write';
      const messages = failed ? fixMessages(worktree, pack, failed) : writeMessages(worktree, pack);
      state(callState(kind, thinking && !noThink));
      let c: Completion<string>;
      try {
        c = await ctx.inference.text(model, messages, ticket.budget.max_tokens - spent, ctx.signal, noThink ? { thinking: false } : undefined);
      } catch (e) {
        const msg = String((e as Error).message ?? e);
        // The review's feedback is sent again: a fix that only answered the error would finish
        // the ticket (a pass after the review) without the model ever seeing what the lead asked.
        if (failed?.command !== '(lead review)') failed = { command: '(model call)', output: msg };
        r.error = msg;
        ctx.emit({ type: 'call_error', at: Date.now(), ticket: ticket.id, attempt, kind, model, error: msg.slice(0, 500) });
        ctx.emit({ type: 'log', at: Date.now(), level: 'warn', message: `${ticket.id} #${attempt}: ${msg}` });
        continue;
      }
      // outside the try: a sink that throws is not a failed model call
      spent += c.completionTokens;
      r.promptTokens += c.promptTokens;
      r.completionTokens += c.completionTokens;
      r.inferenceMs += c.ms;
      const out: WorkerOutput = parseFiles(c.value, ticket.owns);
      const seen = classifyAnswer(c.value, ticket.owns, out, c.cut);
      ctx.emit({ type: 'model', at: Date.now(), ticket: ticket.id, attempt, kind, promptTokens: c.promptTokens, completionTokens: c.completionTokens, reasoningTokens: c.reasoningTokens, cut: c.cut, ms: c.ms, note: String(out?.note ?? '').slice(0, 200), files: Object.keys(out?.files ?? {}).length, model, maxTokens: c.maxTokens, forced: c.forced, drop: seen.drop, missing: seen.missing.length || undefined, loose: seen.loose.length || undefined, reasoningChars: c.reasoningChars, thinking: thinking && !noThink });
      if (thinking && !noThink && cutRetry === 'nothink' && (seen.drop === 'cut' || c.forced)) {
        noThink = true;
        ctx.emit({ type: 'log', at: Date.now(), level: 'warn', message: `${ticket.id} #${attempt}: thinking ran out of tokens (${c.forced ? `forced: ${c.forced}` : `cut at ${c.maxTokens ?? '?'}`}); later calls of this ticket go without thinking` });
      }

      const refused = applyOutput(worktree, ticket, out);
      if (refused.length) {
        ctx.emit({ type: 'refused', at: Date.now(), ticket: ticket.id, attempt, paths: refused });
        failed = { command: '(write files)', output: `You wrote files you do not own: ${refused.join(', ')}. Write only: ${ticket.owns.join(', ')}.` };
        r.error = failed.output;
        continue;
      }

      state('checking');
      const checks: Array<{ command: string; output: string }> = [];
      failed = await check(attempt, checks);
      if (failed) {
        r.error = `${failed.command}\n${failed.output}`;
        continue;
      }
      // a pass after the review asked for a fix: the fix it wanted got through the checks
      if (!reviewOn || r.review) return finished(attempt);
      if (spent >= ticket.budget.max_tokens) {
        r.error = 'no tokens left in the budget for the lead review';
        return fail(r.error);
      }

      // One lead review per ticket, on the first version that passes; a call that fails is tried once more.
      state('reviewing');
      let rv = await reviewOnce(attempt, checks, spent);
      spent += rv.completionTokens;
      if (rv.verdict === 'error' && spent < ticket.budget.max_tokens && !ctx.signal?.aborted) {
        rv = await reviewOnce(attempt, checks, spent);
        spent += rv.completionTokens;
      }
      r.review = rv;
      if (ctx.signal?.aborted) {
        // a stopped run, not a review that failed
        r.error = 'aborted';
        return fail('aborted');
      }
      if (rv.verdict === 'error') {
        r.error = `lead review failed: ${rv.error ?? 'no answer'}`;
        return fail('lead review failed');
      }
      if (!wantsFix(rv)) return finished(attempt);
      r.reviewUnresolved = true;
      failed = { command: '(lead review)', output: reviewFeedback(rv) };
      r.error = `${failed.command}\n${failed.output}`;
    }
  }
  return fail(r.reviewUnresolved ? 'lead review problems left unfixed' : r.error?.split('\n')[0]);
}
