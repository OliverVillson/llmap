/**
 * The engine at crew scale, against the fake model: how much of the slots the engine keeps busy
 * at 64 or 128 agents, and what its own overhead costs. Every ticket owns one file and passes a
 * one-line check; the fake answers each worker call after `--latency` ms, standing in for decode
 * time, and each review after a tenth of that (a short JSON answer, thinking off).
 * A slot is busy from when the engine sends a call until the answer is back, so `busy` is the
 * share of slot time the model would be decoding, and the engine's overhead is the rest.
 *
 *   bun bench/scale.ts --agents 64 --tickets 512 --latency 2000
 *   options: --slots <worker slots> (default agents), --lead <slots> (default worker slots / 8), --review on|off (default on), --flaky <share of
 *            tickets whose first answer is empty, default 0.2>, --tokens <completion tokens a
 *            call reports, default 1500>, --out <run dir> (default a temp folder)
 *
 * Prints wall time, the ideal wall time with no engine overhead, slot occupancy, the largest
 * event-loop stall, model calls a second and the token rate the fake reports, and the sizes of
 * calls.jsonl and status.json. Engine time per call is the CPU the process used divided by calls.
 */
import { mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { BackendPool, resolveMachines } from '../src/engine/backends.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { initRepo } from '../src/engine/git.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
import { defaultCrew, Telemetry } from '../src/engine/telemetry.ts';
import { validatePlan, type WorkerOutput } from '../src/tickets/schema.ts';

export interface ScaleOptions {
  agents: number;
  /** Worker slots on the server (default agents): fewer than agents lets some check while others decode. */
  slots?: number;
  tickets: number;
  latencyMs: number;
  leadSlots?: number;
  review?: boolean;
  flakyShare?: number;
  tokens?: number;
  out?: string;
}

export interface ScaleResult {
  agents: number;
  slots: number;
  tickets: number;
  done: number;
  calls: number;
  wallMs: number;
  /** Wall time if every call ran back to back on its slot with no gap: model time / worker slots. */
  idealMs: number;
  /** Share of worker-slot time a call was out at the model. */
  busy: number;
  peakInFlight: number;
  maxStallMs: number;
  callsPerSec: number;
  tokPerSec: number;
  cpuMsPerCall: number;
  callsLogBytes: number;
  statusBytes: number;
}

export async function runScale(o: ScaleOptions): Promise<ScaleResult> {
  const work = mkdtempSync(join(tmpdir(), 'mugge-scale-'));
  const slots = o.slots ?? o.agents;
  const lead = o.leadSlots ?? Math.max(1, Math.round(slots / 8));
  try {
    const repo = join(work, 'repo');
    mkdirSync(repo, { recursive: true });
    writeFileSync(join(repo, 'README.md'), 'scale\n');
    const base = initRepo(repo, { name: 't', email: 't@t' }, 'Scaffold');
    const ids = Array.from({ length: o.tickets }, (_, i) => `t${i}`);
    const plan = validatePlan({
      tickets: ids.map((id) => ({ id, title: `Write ${id}`, role: 'implement', model: 'coder', owns: [`src/${id}.txt`], acceptance: [`grep -q ok-${id} src/${id}.txt`], image: 'node' })),
    });
    const solutions: Record<string, WorkerOutput> = Object.fromEntries(ids.map((id) => [id, { files: { [`src/${id}.txt`]: `ok-${id}\n` }, note: '' }]));
    const flaky = ids.filter((_, i) => i < Math.round((o.flakyShare ?? 0.2) * ids.length));
    // a review writes a short JSON answer with thinking off: a tenth of a worker call
    const fake = startFakeModel({ solutions, flaky, latencyMs: o.latencyMs, reviewLatencyMs: Math.round(o.latencyMs / 10), completionTokens: o.tokens ?? 1500 });
    const machines = resolveMachines([
      { name: 'b200-workers', url: fake.url, roles: ['worker'], slots },
      { name: 'b200-lead', url: fake.url, roles: ['lead'], slots: lead },
      { name: 'b200-manager', url: fake.url, roles: ['manager'], slots: 1 },
    ]);
    const pool = BackendPool.fromMachines(machines, { baseUrl: fake.url, defaultModel: 'fake', extra: { chat_template_kwargs: { enable_thinking: false } } });
    const out = o.out ?? join(work, 'run');
    const telemetry = new Telemetry({ dir: out, warn: null, crew: defaultCrew(o.agents) });
    telemetry.setPlan(plan);
    pool.subscribe((s) => telemetry.serving(s));

    // the largest gap between two ticks of a 20 ms timer: how long the engine held the event loop
    let maxStall = 0, lastTick = performance.now();
    const ticker = setInterval(() => {
      const now = performance.now();
      maxStall = Math.max(maxStall, now - lastTick - 20);
      lastTick = now;
    }, 20);
    const cpu0 = process.cpuUsage();
    const t0 = performance.now();
    const before = process.env.MUGGE_REVIEW;
    process.env.MUGGE_REVIEW = o.review === false ? 'off' : 'on';
    let report;
    try {
      report = await runPlan({ repo, base, plan, pool, sandbox: new LocalSandbox(), worktreeRoot: join(work, 'wt'), concurrency: o.agents, integrate: false, author: { name: 't', email: 't@t' }, emit: (e) => telemetry.emit(e) });
    } finally {
      if (before === undefined) delete process.env.MUGGE_REVIEW;
      else process.env.MUGGE_REVIEW = before;
      clearInterval(ticker);
      fake.stop();
    }
    const wallMs = performance.now() - t0;
    const cpu = process.cpuUsage(cpu0);
    telemetry.close();
    const calls = fake.calls();
    const workerCalls = calls.filter((c) => c.kind !== 'review').length;
    const size = (p: string) => {
      try {
        return statSync(p).size;
      } catch {
        return 0;
      }
    };
    const { busyMs } = fake.stats();
    // worker slots' model time over the run; review calls go to the lead slots
    const workerBusy = busyMs.worker ?? 0;
    return {
      agents: o.agents, tickets: o.tickets, done: report.summary.done, calls: calls.length, wallMs: Math.round(wallMs),
      slots,
      idealMs: Math.round((workerCalls * o.latencyMs) / Math.min(slots, o.tickets)),
      busy: Math.round((workerBusy / (wallMs * Math.min(slots, o.tickets))) * 1000) / 1000,
      peakInFlight: fake.stats().peakInFlight,
      maxStallMs: Math.round(maxStall),
      callsPerSec: Math.round((calls.length / wallMs) * 10_000) / 10,
      tokPerSec: Math.round((calls.length * (o.tokens ?? 1500)) / (wallMs / 1000)),
      cpuMsPerCall: Math.round(((cpu.user + cpu.system) / 1000 / calls.length) * 10) / 10,
      callsLogBytes: size(join(out, 'calls.jsonl')),
      statusBytes: size(join(out, 'status.json')),
    };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

export function formatScale(r: ScaleResult): string {
  return [
    `${r.agents} agents on ${r.slots} worker slots, ${r.tickets} tickets: ${r.done} done, ${r.calls} model calls`,
    `wall ${(r.wallMs / 1000).toFixed(1)} s (ideal ${(r.idealMs / 1000).toFixed(1)} s), worker slots busy ${(r.busy * 100).toFixed(0)}%, peak ${r.peakInFlight} calls in flight`,
    `${r.callsPerSec} calls/s, ${r.tokPerSec} tok/s at the fake's token count, engine CPU ${r.cpuMsPerCall} ms per call, longest event-loop stall ${r.maxStallMs} ms`,
    `calls.jsonl ${(r.callsLogBytes / 1024).toFixed(0)} KB, status.json ${(r.statusBytes / 1024).toFixed(0)} KB`,
  ].join('\n');
}

if (import.meta.main) {
  const { values } = parseArgs({
    options: {
      agents: { type: 'string' }, slots: { type: 'string' }, tickets: { type: 'string' }, latency: { type: 'string' }, lead: { type: 'string' },
      review: { type: 'string' }, flaky: { type: 'string' }, tokens: { type: 'string' }, out: { type: 'string' },
    },
  });
  const num = (v: string | undefined, d?: number) => (v === undefined ? d : Number(v));
  const agents = num(values.agents, 64)!;
  const r = await runScale({
    agents, slots: num(values.slots), tickets: num(values.tickets, agents * 4)!, latencyMs: num(values.latency, 2000)!, leadSlots: num(values.lead),
    review: values.review !== 'off', flakyShare: num(values.flaky), tokens: num(values.tokens), out: values.out,
  });
  console.log(formatScale(r));
}
