/**
 * The harness worker: the write → check → fix loop that runs one ticket with a single-turn
 * specialist model.
 *
 *   context pack → model writes the owned files → engine runs every acceptance command in the
 *   sandbox → on failure a fresh fix call gets the trimmed error → repeat until the budget runs
 *   out → escalate to a bigger model once, or end failed with the last error.
 *
 * The model never decides it is done; the acceptance commands do.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { Plan, Ticket, WorkerOutput } from '../tickets/schema.ts';
import type { EventSink } from './events.ts';
import { parseFiles } from './files.ts';
import type { Inference } from './inference.ts';
import { buildPack, fixMessages, writeMessages } from './prompts.ts';
import { tail, type Sandbox } from './sandbox.ts';

export interface HarnessContext {
  plan: Plan;
  inference: Inference;
  sandbox: Sandbox;
  emit: EventSink;
  memory?: string;
  /** Per acceptance command. */
  checkTimeoutMs?: number;
  signal?: AbortSignal;
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
  const r: TicketResult = { ok: false, attempts: 0, firstTry: false, error: null, promptTokens: 0, completionTokens: 0, inferenceMs: 0, checkMs: 0, model: ticket.model };
  const pack = buildPack(worktree, ctx.plan, ticket, ctx.memory);
  const rounds = [ticket.model, ...(ticket.escalate_to ? [ticket.escalate_to] : [])];
  let failed: { command: string; output: string } | null = null;

  for (const model of rounds) {
    r.model = model;
    let spent = 0;
    for (let i = 0; i < ticket.budget.attempts && spent < ticket.budget.max_tokens; i++) {
      if (ctx.signal?.aborted) {
        r.error = 'aborted';
        return r;
      }
      const attempt = ++r.attempts;
      const kind = failed ? 'fix' : 'write';
      const messages = failed ? fixMessages(worktree, pack, failed) : writeMessages(worktree, pack);
      let out: WorkerOutput;
      try {
        const c = await ctx.inference.text(model, messages, ticket.budget.max_tokens - spent);
        spent += c.completionTokens;
        r.promptTokens += c.promptTokens;
        r.completionTokens += c.completionTokens;
        r.inferenceMs += c.ms;
        out = parseFiles(c.value);
        ctx.emit({ type: 'model', at: Date.now(), ticket: ticket.id, attempt, kind, promptTokens: c.promptTokens, completionTokens: c.completionTokens, ms: c.ms, note: String(out?.note ?? '').slice(0, 200) });
      } catch (e) {
        failed = { command: '(model call)', output: String((e as Error).message ?? e) };
        r.error = failed.output;
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

      failed = null;
      for (const command of ticket.acceptance) {
        const res = await ctx.sandbox.run(command, { cwd: worktree, image: ticket.image, timeoutMs: ctx.checkTimeoutMs });
        r.checkMs += res.ms;
        const ok = res.code === 0;
        ctx.emit({ type: 'check', at: Date.now(), ticket: ticket.id, attempt, command, ok, ms: res.ms, tail: tail(res.out, 800) });
        if (!ok) {
          failed = { command, output: (res.timedOut ? '(timed out)\n' : '') + tail(res.out) };
          break;
        }
      }
      if (!failed) {
        r.ok = true;
        r.firstTry = attempt === 1;
        r.error = null;
        return r;
      }
      r.error = `${failed.command}\n${failed.output}`;
    }
  }
  return r;
}
