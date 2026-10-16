/**
 * A fake OpenAI-compatible model server for tests and CI: it answers each ticket from canned
 * solutions (spike/solutions/<ticket>.json), found by the `Ticket: <id>` line of the prompt.
 * Calls with a JSON schema get JSON back; coder calls get the files as plain fenced files.
 * `flaky` tickets get a useless first answer so the fix loop runs; `latencyMs` stands in for
 * inference time so the parallel speed-up shows. Lead review calls (REVIEW_SYSTEM) get the
 * ticket's canned reviews in turn, else an approval.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { WorkerOutput } from '../tickets/schema.ts';
import { renderFiles } from './files.ts';
import { REVIEW_SYSTEM } from './prompts.ts';
import type { LeadReview } from './review.ts';

export interface FakeOptions {
  solutions: Record<string, WorkerOutput>;
  /** Ticket ids whose first write call returns no files. */
  flaky?: string[];
  /** Ticket ids that never get a correct answer. */
  broken?: string[];
  /** Ticket ids whose first answer is cut at max_tokens mid-file (finish_reason "length"). */
  cut?: string[];
  /** Ticket ids whose first answer is prose with no fenced file (a parser drop). */
  prose?: string[];
  /** Ticket ids whose first answer comes back marked forced, as MuggeBench's proxy marks one. */
  forced?: string[];
  /** Ticket ids whose calls with thinking on (chat_template_kwargs.enable_thinking) think to the cap and write nothing. */
  overthink?: string[];
  latencyMs?: number;
  port?: number;
  /** Ticket id → the answers to its review calls, in order (the last one repeats). */
  reviews?: Record<string, LeadReview[]>;
  /** Ticket id → the answers to its fix calls, in order (the last one repeats), in place of the solution. */
  fixes?: Record<string, WorkerOutput[]>;
}

/** One request the fake answered, for tests. */
export interface FakeCall {
  ticket: string;
  kind: 'review' | 'fix' | 'write' | 'json';
  maxTokens: number;
  prompt: string;
  /** chat_template_kwargs.enable_thinking as sent, when it was. */
  thinking?: boolean;
}

export function loadSolutions(dir: string): Record<string, WorkerOutput> {
  const out: Record<string, WorkerOutput> = {};
  if (!existsSync(dir)) return out;
  for (const f of readdirSync(dir)) {
    if (f.endsWith('.json')) out[f.slice(0, -5)] = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  }
  return out;
}

const approxTokens = (s: string) => Math.ceil(s.length / 4);

export function startFakeModel(o: FakeOptions) {
  const seen = new Map<string, number>(), reviewed = new Map<string, number>(), fixed = new Map<string, number>();
  const calls: FakeCall[] = [];
  let inFlight = 0, peakInFlight = 0, requests = 0;
  const server = Bun.serve({
    port: o.port ?? 0,
    hostname: '127.0.0.1',
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname.endsWith('/models')) return Response.json({ object: 'list', data: [{ id: 'fake', object: 'model' }] });
      if (!url.pathname.endsWith('/chat/completions')) return new Response('not found', { status: 404 });
      requests++;
      inFlight++;
      peakInFlight = Math.max(peakInFlight, inFlight);
      try {
        const body: any = await req.json();
        const prompt: string = body.messages?.map((m: any) => m.content).join('\n') ?? '';
        const id = prompt.match(/^Ticket: (\S+)$/m)?.[1] ?? '';
        const isReview = body.messages?.[0]?.content === REVIEW_SYSTEM;
        const isFix = !isReview && /\n## Fix\n/.test(prompt);
        const thinking = body.chat_template_kwargs?.enable_thinking;
        calls.push({ ticket: id, kind: isReview ? 'review' : body.response_format || body.guided_json ? 'json' : isFix ? 'fix' : 'write', maxTokens: body.max_tokens, prompt, ...(typeof thinking === 'boolean' ? { thinking } : {}) });
        if (o.latencyMs) await Bun.sleep(o.latencyMs);
        if (isReview) {
          const k = reviewed.get(id) ?? 0;
          reviewed.set(id, k + 1);
          const list = o.reviews?.[id] ?? [];
          const rv: LeadReview = list[Math.min(k, list.length - 1)] ?? { verdict: 'approve', problems: [], fix: '' };
          const content = JSON.stringify(rv);
          return Response.json({
            id: `fake-${requests}`, object: 'chat.completion', model: body.model,
            choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content } }],
            usage: { prompt_tokens: approxTokens(prompt), completion_tokens: approxTokens(content), total_tokens: approxTokens(prompt) + approxTokens(content) },
          });
        }
        const n = (seen.get(id) ?? 0) + 1;
        seen.set(id, n);
        let answer: WorkerOutput;
        const fixList = isFix ? o.fixes?.[id] : undefined;
        if (fixList?.length) {
          const k = fixed.get(id) ?? 0;
          fixed.set(id, k + 1);
          answer = fixList[Math.min(k, fixList.length - 1)]!;
        } else if (o.broken?.includes(id)) answer = { files: {}, note: 'no idea' };
        else if (o.flaky?.includes(id) && n === 1) answer = { files: {}, note: 'first try, wrote nothing' };
        else answer = o.solutions[id] ?? { files: {}, note: `no canned answer for ${id || 'this prompt'}` };
        let content = body.response_format || body.guided_json ? JSON.stringify(answer) : renderFiles(answer.files, answer.note);
        let finish = 'stop', forced: string | undefined;
        if (thinking === true && o.overthink?.includes(id)) {
          const content = `<think>\n${'Let me reconsider the edge cases once more. '.repeat(Math.ceil((body.max_tokens ?? 64) / 8))}`;
          return Response.json({
            id: `fake-${requests}`, object: 'chat.completion', model: body.model,
            choices: [{ index: 0, finish_reason: 'length', message: { role: 'assistant', content } }],
            usage: { prompt_tokens: approxTokens(prompt), completion_tokens: body.max_tokens ?? 0, total_tokens: approxTokens(prompt) + (body.max_tokens ?? 0) },
          });
        }
        if (n === 1 && o.cut?.includes(id)) {
          content = content.slice(0, Math.max(1, Math.floor(content.length / 2)));
          finish = 'length';
        } else if (n === 1 && o.prose?.includes(id)) content = 'I would write the function so that it handles every case, then run the tests.';
        else if (n === 1 && o.forced?.includes(id)) forced = 'length';
        return Response.json({
          id: `fake-${requests}`,
          object: 'chat.completion',
          model: body.model,
          choices: [{ index: 0, finish_reason: finish, message: { role: 'assistant', content }, ...(forced ? { forced } : {}) }],
          usage: { prompt_tokens: approxTokens(prompt), completion_tokens: approxTokens(content), total_tokens: approxTokens(prompt) + approxTokens(content) },
        });
      } finally {
        inFlight--;
      }
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}/v1`,
    stats: () => ({ requests, peakInFlight }),
    calls: () => calls,
    stop: () => server.stop(true),
  };
}
