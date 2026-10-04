/**
 * The engine's local HTTP API, run on the project VM and reached from the laptop through the
 * SSH tunnel. Bound to 127.0.0.1 only.
 *
 *   GET  /health           ok
 *   GET  /status           the current RunView
 *   GET  /events           server-sent events, one EngineEvent per message (replays the run so far)
 *   POST /run              { plan, base?, concurrency? } starts a run on the project repo
 *   POST /stop             aborts the current run
 */
import { join } from 'node:path';
import { applyEvent, emptyView, type EngineEvent, type RunView } from '../engine/events.ts';
import { headCommit } from '../engine/git.ts';
import type { Inference } from '../engine/inference.ts';
import type { Sandbox } from '../engine/sandbox.ts';
import { runPlan } from '../engine/scheduler.ts';
import { validatePlan } from '../tickets/schema.ts';

export interface EngineServerOptions {
  port: number;
  /** The project's code checkout on the VM. */
  repo: string;
  inference: Inference;
  sandbox: Sandbox;
  concurrency: number;
  worktreeRoot?: string;
}

export function startEngineServer(o: EngineServerOptions) {
  let view: RunView = emptyView();
  let events: EngineEvent[] = [];
  let abort: AbortController | null = null;
  const listeners = new Set<(e: EngineEvent) => void>();
  const emit = (e: EngineEvent) => {
    events.push(e);
    applyEvent(view, e);
    for (const l of listeners) l(e);
  };

  const server = Bun.serve({
    port: o.port,
    hostname: '127.0.0.1',
    idleTimeout: 0,
    async fetch(req) {
      const { pathname } = new URL(req.url);
      if (pathname === '/health') return new Response('ok');
      if (pathname === '/status') return Response.json({ running: !!abort, view });
      if (pathname === '/events') {
        let send: ((e: EngineEvent) => void) | null = null;
        const stream = new ReadableStream({
          start(c) {
            const enc = new TextEncoder();
            send = (e) => c.enqueue(enc.encode(`data: ${JSON.stringify(e)}\n\n`));
            for (const e of events) send(e);
            listeners.add(send);
          },
          cancel() {
            if (send) listeners.delete(send);
          },
        });
        return new Response(stream, { headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' } });
      }
      if (pathname === '/run' && req.method === 'POST') {
        if (abort) return Response.json({ error: 'a run is already going' }, { status: 409 });
        const body: any = await req.json();
        let plan;
        try {
          plan = validatePlan(body.plan);
        } catch (e) {
          return Response.json({ error: String((e as Error).message) }, { status: 400 });
        }
        view = emptyView();
        events = [];
        abort = new AbortController();
        const run = runPlan({
          repo: o.repo,
          base: body.base ?? headCommit(o.repo),
          plan,
          inference: o.inference,
          sandbox: o.sandbox,
          concurrency: body.concurrency ?? o.concurrency,
          worktreeRoot: o.worktreeRoot ?? join(o.repo, '..', '.mugge-worktrees'),
          emit,
          signal: abort.signal,
        });
        run
          .catch((e) => emit({ type: 'log', at: Date.now(), level: 'error', message: String(e?.message ?? e) }))
          .finally(() => (abort = null));
        return Response.json({ started: true, tickets: plan.tickets.length });
      }
      if (pathname === '/stop' && req.method === 'POST') {
        abort?.abort();
        return Response.json({ stopping: !!abort });
      }
      return new Response('not found', { status: 404 });
    },
  });
  return { url: `http://127.0.0.1:${server.port}`, port: server.port, stop: () => server.stop(true), view: () => view };
}

/** Reads the engine's event stream from the laptop side (through the tunnel). */
export async function followEvents(url: string, onEvent: (e: EngineEvent) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${url}/events`, { signal });
  if (!res.ok || !res.body) throw new Error(`engine events: ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const line = chunk.split('\n').find((l) => l.startsWith('data: '));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}
