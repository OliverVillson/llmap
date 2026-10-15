/**
 * The harness worker: the write → check → fix loop that runs one ticket with a single-turn
 * specialist model.
 *
 *   context pack → model writes the owned files → engine runs every acceptance command in the
 *   sandbox → on failure a fresh fix call gets the trimmed error → repeat until the budget runs
 *   out → escalate to a bigger model once, or end failed with the last error.
 *
 * With MUGGE_REVIEW=on, a ticket whose checks pass gets one lead review (review.ts); problems
 * with a fix spend one more fix attempt. If that attempt and the ones after it cannot get the
 * checks green again, the version that passed is put back and the ticket ends finished: the
 * review never fails a ticket its checks pass.
 *
 * Every call is one fresh one-shot message, and every step emits a `state` event (state.ts).
 * The model never decides it is done; the acceptance commands do.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { Plan, Ticket, WorkerOutput } from '../tickets/schema.ts';
import type { EventSink } from './events.ts';
import { parseFiles } from './files.ts';
import type { Inference } from './inference.ts';
import { buildPack, fixMessages, writeMessages } from './prompts.ts';
import { diffOwned, restore, review as leadReview, reviewFeedback, reviewFromEnv, snapshot, wantsFix, type ReviewOutcome, type Snapshot } from './review.ts';
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
  /** Lead review after the checks pass; unset: MUGGE_REVIEW. */
  review?: boolean;
  /** Where the review call goes (the scheduler gives a lead slot when the crew has one); unset: `inference`. */
  reviewInference?: Inference;
  /**
   * Model name for the review. Names the inference config does not map fall back to its default
   * model, which is the plan: one model for every role, the role is in the prompt.
   */
  reviewModel?: string;
  /** Whether calls think (only changes the state shown during a call); unset: the inference config's switch, else MUGGE_THINKING. */
  thinking?: boolean;
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
  /** The review found problems that its fix attempts did not get through the checks; the passing version was kept. */
  reviewUnresolved: boolean;
}

/** Whether the inference config turns thinking on (MUGGE_THINKING / bench --thinking), else the environment. */
function thinkingOn(inference: Inference): boolean {
  const kw = (inference as { cfg?: { extra?: { chat_template_kwargs?: { enable_thinking?: unknown } } } }).cfg?.extra?.chat_template_kwargs;
  if (typeof kw?.enable_thinking === 'boolean') return kw.enable_thinking;
  return process.env.MUGGE_THINKING?.trim().toLowerCase() === 'on';
}

/** Writes the answer's owned files into the worktree; returns the paths it refused. */
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
  const r: TicketResult = { ok: false, attempts: 0, firstTry: false, error: null, promptTokens: 0, completionTokens: 0, inferenceMs: 0, checkMs: 0, model: ticket.model, review: null, reviewUnresolved: false };
  const reviewOn = ctx.review ?? reviewFromEnv();
  const thinking = ctx.thinking ?? thinkingOn(ctx.inference);
  const pack = buildPack(worktree, ctx.plan, ticket, ctx.memory);
  const rounds = [ticket.model, ...(ticket.escalate_to ? [ticket.escalate_to] : [])];
  const of = ticket.budget.attempts * rounds.length;
  const state = (s: AgentState, reason?: string) =>
    ctx.emit({ type: 'state', at: Date.now(), ticket: ticket.id, state: s, attempt: r.attempts, of, ...(reason ? { reason } : {}) });
  const before = reviewOn ? snapshot(worktree, ticket.owns) : {};
  let failed: { command: string; output: string } | null = null;
  // The last version whose checks passed, while a review's fix is being tried.
  let passed: { files: Snapshot; attempt: number; model: string } | null = null;

  const finished = (attempt: number) => {
    r.ok = true;
    r.firstTry = attempt === 1;
    r.error = null;
    state('finished', r.reviewUnresolved ? 'lead review problems left unfixed' : undefined);
    return r;
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
      state(callState(kind, thinking));
      let out: WorkerOutput;
      try {
        const c = await ctx.inference.text(model, messages, ticket.budget.max_tokens - spent);
        spent += c.completionTokens;
        r.promptTokens += c.promptTokens;
        r.completionTokens += c.completionTokens;
        r.inferenceMs += c.ms;
        out = parseFiles(c.value, ticket.owns);
        const seen = classifyAnswer(c.value, ticket.owns, out, c.cut);
        ctx.emit({ type: 'model', at: Date.now(), ticket: ticket.id, attempt, kind, promptTokens: c.promptTokens, completionTokens: c.completionTokens, reasoningTokens: c.reasoningTokens, cut: c.cut, ms: c.ms, note: String(out?.note ?? '').slice(0, 200), files: Object.keys(out?.files ?? {}).length, model, maxTokens: c.maxTokens, forced: c.forced, drop: seen.drop, missing: seen.missing.length || undefined, loose: seen.loose.length || undefined, reasoningChars: c.reasoningChars });
      } catch (e) {
        failed = { command: '(model call)', output: String((e as Error).message ?? e) };
        r.error = failed.output;
        ctx.emit({ type: 'call_error', at: Date.now(), ticket: ticket.id, attempt, kind, model, error: failed.output.slice(0, 500) });
        ctx.emit({ type: 'log', at: Date.now(), level: 'warn', message: `${ticket.id} #${attempt}: ${failed.output}` });
        continue;
      }

      const refused = applyOutput(worktree, ticket, out);
      if (refused.length) {
        ctx.emit({ type: 'refused', at: Date.now(), ticket: ticket.id, attempt, paths: refused });
        failed = { command: '(write files)', output: `You wrote files you do not own: ${refused.join(', ')}. Write only: ${ticket.owns.join(', ')}.` };
        r.error = failed.output;
        continue;
      }

      state('checking');
      failed = null;
      const checks: Array<{ command: string; output: string }> = [];
      for (const command of ticket.acceptance) {
        const res = await ctx.sandbox.run(command, { cwd: worktree, image: ticket.image, timeoutMs: ctx.checkTimeoutMs });
        r.checkMs += res.ms;
        const ok = res.code === 0;
        ctx.emit({ type: 'check', at: Date.now(), ticket: ticket.id, attempt, command, ok, ms: res.ms, tail: tail(res.out, 800) });
        if (!ok) {
          failed = { command, output: (res.timedOut ? '(timed out)\n' : '') + tail(res.out) };
          break;
        }
        checks.push({ command, output: res.out });
      }
      if (failed) {
        r.error = `${failed.command}\n${failed.output}`;
        continue;
      }
      if (!reviewOn || r.review || spent >= ticket.budget.max_tokens) return finished(attempt);

      // One lead review per ticket, on the first version that passes.
      state('reviewing');
      const rv = await leadReview(ctx.reviewInference ?? ctx.inference, ctx.reviewModel ?? 'lead', { ticket, diff: diffOwned(worktree, before), checks }, ticket.budget.max_tokens - spent);
      spent += rv.completionTokens;
      r.promptTokens += rv.promptTokens;
      r.completionTokens += rv.completionTokens;
      r.inferenceMs += rv.ms;
      r.review = rv;
      ctx.emit({ type: 'review', at: Date.now(), ticket: ticket.id, attempt, verdict: rv.verdict, problems: rv.problems, fix: rv.fix, ...(rv.error ? { error: rv.error } : {}), promptTokens: rv.promptTokens, completionTokens: rv.completionTokens, reasoningTokens: rv.reasoningTokens, cut: rv.cut, ms: rv.ms });
      const room = i + 1 < ticket.budget.attempts && spent < ticket.budget.max_tokens;
      if (!wantsFix(rv) || !room) {
        r.reviewUnresolved = wantsFix(rv);
        return finished(attempt);
      }
      passed = { files: snapshot(worktree, ticket.owns), attempt, model };
      failed = { command: '(lead review)', output: reviewFeedback(rv) };
      r.error = `${failed.command}\n${failed.output}`;
    }
    if (passed) break; // the review's fix ran out of attempts: keep the passing version, do not escalate
  }
  if (passed) {
    restore(worktree, passed.files);
    r.model = passed.model;
    r.reviewUnresolved = true;
    return finished(passed.attempt);
  }
  state('failed', r.error?.split('\n')[0]);
  return r;
}
