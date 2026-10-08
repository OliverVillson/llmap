/**
 * A fake OpenAI-compatible model server for tests and CI: it answers each ticket from canned
 * solutions (spike/solutions/<ticket>.json), found by the `Ticket: <id>` line of the prompt.
 * Calls with a JSON schema get JSON back; coder calls get the files as plain fenced files.
 * `flaky` tickets get a useless first answer so the fix loop runs; `latencyMs` stands in for
 * inference time so the parallel speed-up shows.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { WorkerOutput } from '../tickets/schema.ts';
import { renderFiles } from './files.ts';

export interface FakeOptions {
  solutions: Record<string, WorkerOutput>;
  /** Ticket ids whose first write call returns no files. */
  flaky?: string[];
  /** Ticket ids that never get a correct answer. */
  broken?: string[];
  latencyMs?: number;
  port?: number;
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
  const seen = new Map<string, number>();
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
        const n = (seen.get(id) ?? 0) + 1;
        seen.set(id, n);
        if (o.latencyMs) await Bun.sleep(o.latencyMs);
        let answer: WorkerOutput;
        if (o.broken?.includes(id)) answer = { files: {}, note: 'no idea' };
        else if (o.flaky?.includes(id) && n === 1) answer = { files: {}, note: 'first try, wrote nothing' };
        else answer = o.solutions[id] ?? { files: {}, note: `no canned answer for ${id || 'this prompt'}` };
        const content = body.response_format || body.guided_json ? JSON.stringify(answer) : renderFiles(answer.files, answer.note);
        return Response.json({
          id: `fake-${requests}`,
          object: 'chat.completion',
          model: body.model,
          choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content } }],
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
    stop: () => server.stop(true),
  };
}
