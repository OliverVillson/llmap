/**
 * The engine spike: run the polyglot test project's tickets in parallel worktrees, integrate,
 * and print the metrics from docs/engine-spike.md.
 *
 *   bun run spike/run-spike.ts --fake                 canned answers, local sandbox (CI)
 *   bun run spike/run-spike.ts --url http://127.0.0.1:8000/v1 --model qwen3.6-35b-a3b-fp8 --concurrency 16
 *   options: --serial (also run one at a time and report the speed-up), --sandbox local|podman|docker,
 *            --latency <ms> (fake model think time), --flaky id,id (fake: first answer empty),
 *            --out metrics.json, --keep (keep the work folder)
 */
import { cpSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { inferenceFromEnv } from '../src/config.ts';
import type { EngineEvent, RunSummary } from '../src/engine/events.ts';
import { loadSolutions, startFakeModel } from '../src/engine/fake-model.ts';
import { initRepo } from '../src/engine/git.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox, sandboxFromEnv } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
import { validatePlan } from '../src/tickets/schema.ts';

const SPIKE = import.meta.dir;

export interface SpikeOptions {
  fake?: boolean;
  url?: string;
  model?: string;
  concurrency?: number;
  sandbox?: string;
  latencyMs?: number;
  flaky?: string[];
  quiet?: boolean;
  keep?: boolean;
}

export async function runSpike(o: SpikeOptions): Promise<{ summary: RunSummary; integrated: boolean; workDir: string }> {
  const plan = validatePlan(await Bun.file(join(SPIKE, 'plan.json')).json());
  const work = mkdtempSync(join(tmpdir(), 'mugge-spike-'));
  const repo = join(work, 'repo');
  cpSync(join(SPIKE, 'project'), repo, { recursive: true });
  const base = initRepo(repo, { name: 'mugge spike', email: 'spike@localhost' }, 'Scaffold');

  const fake = o.fake ? startFakeModel({ solutions: loadSolutions(join(SPIKE, 'solutions')), latencyMs: o.latencyMs ?? 0, flaky: o.flaky }) : null;
  const cfg = inferenceFromEnv();
  const inference = new OpenAICompatible({ ...cfg, baseUrl: fake?.url ?? o.url ?? cfg.baseUrl, defaultModel: o.model ?? (fake ? 'fake' : cfg.defaultModel) });
  const sandbox = o.sandbox ? sandboxFromEnv({ MUGGE_SANDBOX: o.sandbox }) : o.fake ? new LocalSandbox() : sandboxFromEnv();
  const log = (e: EngineEvent) => {
    if (o.quiet) return;
    const t = new Date(e.at).toISOString().slice(11, 19);
    if (e.type === 'finish') console.log(`${t} ${e.state.padEnd(7)} ${e.ticket} (${e.attempts} attempt${e.attempts === 1 ? '' : 's'})${e.error ? ': ' + e.error.split('\n')[0] : ''}`);
    else if (e.type === 'start') console.log(`${t} start   ${e.ticket}`);
    else if (e.type === 'integrate') console.log(`${t} integrate ${e.ok ? 'ok' : 'FAILED'} (${e.merged.length} branches)${e.error ? '\n' + e.error : ''}`);
    else if (e.type === 'log') console.log(`${t} ${e.level} ${e.message}`);
  };
  try {
    const report = await runPlan({
      repo, base, plan, inference, sandbox,
      worktreeRoot: join(work, 'worktrees'),
      concurrency: o.concurrency ?? 16,
      emit: log,
      author: { name: 'mugge spike', email: 'spike@localhost' },
    });
    return { summary: report.summary, integrated: !!report.integration?.ok, workDir: work };
  } finally {
    fake?.stop();
    if (!o.keep) rmSync(work, { recursive: true, force: true });
  }
}

export function formatSummary(s: RunSummary, serial?: RunSummary): string {
  const pct = (n: number) => `${Math.round((100 * n) / Math.max(1, s.tickets))}%`;
  const rows: Array<[string, string]> = [
    ['tickets passing, first try', `${s.firstTry}/${s.tickets} (${pct(s.firstTry)})`],
    ['tickets passing, within budget', `${s.done}/${s.tickets} (${pct(s.done)})`],
    ['failed / skipped', `${s.failed} / ${s.skipped}`],
    ['model calls per passing ticket', (s.calls / Math.max(1, s.done)).toFixed(2)],
    ['tokens per passing ticket', String(Math.round((s.promptTokens + s.completionTokens) / Math.max(1, s.done)))],
    ['peak agents at once', String(s.peakParallel)],
    ['wall clock', `${(s.wallMs / 1000).toFixed(1)} s`],
    ['sum of ticket times (serial estimate)', `${(s.serialMs / 1000).toFixed(1)} s`],
    ['time in inference / checks', `${(s.inferenceMs / 1000).toFixed(1)} s / ${(s.checkMs / 1000).toFixed(1)} s`],
    ['integration', s.integrated === null ? 'not run' : s.integrated ? 'clean merge, all checks green' : 'FAILED'],
  ];
  if (serial) rows.push(['measured speed-up vs one at a time', `${(serial.wallMs / Math.max(1, s.wallMs)).toFixed(1)}x`]);
  const w = Math.max(...rows.map((r) => r[0].length));
  return rows.map(([k, v]) => `${k.padEnd(w)}  ${v}`).join('\n');
}

if (import.meta.main) {
  const { values } = parseArgs({
    options: {
      fake: { type: 'boolean' }, url: { type: 'string' }, model: { type: 'string' }, concurrency: { type: 'string' },
      sandbox: { type: 'string' }, latency: { type: 'string' }, flaky: { type: 'string' }, serial: { type: 'boolean' },
      out: { type: 'string' }, keep: { type: 'boolean' },
    },
  });
  const o: SpikeOptions = {
    fake: values.fake, url: values.url, model: values.model, sandbox: values.sandbox, keep: values.keep,
    concurrency: values.concurrency ? Number(values.concurrency) : undefined,
    latencyMs: values.latency ? Number(values.latency) : undefined,
    flaky: values.flaky ? values.flaky.split(',') : undefined,
  };
  const par = await runSpike(o);
  const serial = values.serial ? (await runSpike({ ...o, concurrency: 1, quiet: true })).summary : undefined;
  console.log('\n' + formatSummary(par.summary, serial));
  if (values.keep) console.log(`\nwork folder: ${par.workDir}`);
  if (values.out) writeFileSync(values.out, JSON.stringify({ parallel: par.summary, serial }, null, 2) + '\n');
  process.exit(par.summary.done === par.summary.tickets && par.integrated ? 0 : 1);
}
