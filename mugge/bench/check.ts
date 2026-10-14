/**
 * Admits MuggeBench tickets. spike/check.sh grown for hidden tests: a ticket is valid when, in
 * its sandbox image,
 *   1. its reference solution writes exactly the files it owns, and no hidden file is one of them;
 *   2. acceptance FAILS on the scaffold with only its dependencies' solutions applied;
 *   3. acceptance and hidden tests PASS with its dependencies' solutions plus its own;
 *   4. hidden tests FAIL with only its dependencies applied (they test this ticket's work);
 *   5. step 3 gives the same result `repeats` times, each command under `maxSeconds`;
 * and the project carries its canary in plan.json and every solution file.
 *
 *   bun bench/check.ts <bench dir> [--sandbox local|podman|docker] [--repeats 5] [--max-seconds 60]
 *                                  [--concurrency 8] [--filter text]
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { LocalSandbox, sandboxFromEnv, tail, type Sandbox } from '../src/engine/sandbox.ts';
import { applyHidden, findProjects, hiddenFiles, prepare, ROLES, solution, THEMES, type BenchProject, type BenchTicket } from './load.ts';

export interface CheckOptions {
  sandbox: Sandbox;
  repeats?: number;
  maxSeconds?: number;
  concurrency?: number;
}

export interface TicketCheck {
  key: string;
  problems: string[];
}

interface Ran {
  ok: boolean;
  slowest: number;
  log: string;
}

async function runAll(sandbox: Sandbox, t: BenchTicket, dir: string, commands: string[], maxSeconds: number): Promise<Ran> {
  let slowest = 0;
  for (const command of commands) {
    const r = await sandbox.run(command, { cwd: dir, image: t.ticket.image, timeoutMs: maxSeconds * 2000 });
    slowest = Math.max(slowest, r.ms);
    if (r.code !== 0) return { ok: false, slowest, log: `$ ${command}\n${tail(r.out, 600)}` };
  }
  return { ok: true, slowest, log: '' };
}

export function staticProblems(p: BenchProject, t: BenchTicket): string[] {
  const problems: string[] = [];
  const id = t.ticket.id;
  if (!(THEMES as readonly string[]).includes(t.tags.theme)) problems.push(`theme "${t.tags.theme}" is not one of ${THEMES.join(', ')}`);
  if (!(ROLES as readonly string[]).includes(t.tags.role)) problems.push(`role "${t.tags.role}" is not one of ${ROLES.join(', ')}`);
  if (!t.tags.hidden.length) problems.push('no hidden commands');
  const sol = solution(p, id);
  const extra = Object.keys(sol).filter((f) => !t.ticket.owns.includes(f));
  const missing = t.ticket.owns.filter((f) => !(f in sol));
  if (extra.length) problems.push(`solution writes files it does not own: ${extra.join(', ')}`);
  if (missing.length) problems.push(`solution misses owned files: ${missing.join(', ')}`);
  const clash = hiddenFiles(p, id).filter((f) => t.ticket.owns.includes(f));
  if (clash.length) problems.push(`hidden files would overwrite owned files: ${clash.join(', ')}`);
  try {
    const canary = JSON.parse(readFileSync(join(p.dir, 'solutions', `${id}.json`), 'utf8')).canary;
    if (!p.canary.startsWith('MUGGE-BENCH-')) problems.push('plan.json has no bench.canary starting with MUGGE-BENCH- (stages/data.py drops rows holding one)');
    else if (canary !== p.canary) problems.push('solution file lacks the project canary');
  } catch {
    problems.push('no solution file');
  }
  return problems;
}

export async function checkTicket(p: BenchProject, t: BenchTicket, o: CheckOptions): Promise<TicketCheck> {
  const problems = staticProblems(p, t);
  if (problems.some((x) => x.startsWith('no solution'))) return { key: t.key, problems };
  const maxSeconds = o.maxSeconds ?? 60;
  const work = mkdtempSync(join(tmpdir(), 'mugge-bench-check-'));
  const fresh = (name: string, ids: string[], hidden: boolean) => {
    const dir = join(work, name);
    prepare(p, dir, ids);
    if (hidden) applyHidden(p, t.ticket.id, dir);
    return dir;
  };
  try {
    const depsOnly = await runAll(o.sandbox, t, fresh('deps', t.deps, false), t.ticket.acceptance, maxSeconds);
    if (depsOnly.ok) problems.push('acceptance passes without this ticket\'s solution');
    if (t.tags.hidden.length) {
      const hiddenDeps = await runAll(o.sandbox, t, fresh('deps-hidden', t.deps, true), t.tags.hidden, maxSeconds);
      if (hiddenDeps.ok) problems.push('hidden tests pass without this ticket\'s solution');
    }
    const results: boolean[] = [];
    for (let i = 0; i < (o.repeats ?? 3); i++) {
      const dir = fresh(`ref-${i}`, [...t.deps, t.ticket.id], true);
      const acc = await runAll(o.sandbox, t, dir, t.ticket.acceptance, maxSeconds);
      const hid = acc.ok ? await runAll(o.sandbox, t, dir, t.tags.hidden, maxSeconds) : acc;
      results.push(acc.ok && hid.ok);
      if (!acc.ok && i === 0) problems.push(`acceptance fails with the reference solution:\n${acc.log}`);
      else if (!hid.ok && i === 0) problems.push(`hidden tests fail with the reference solution:\n${hid.log}`);
      const slow = Math.max(acc.slowest, hid.slowest);
      if (slow > maxSeconds * 1000) problems.push(`a command took ${(slow / 1000).toFixed(0)} s (limit ${maxSeconds} s)`);
    }
    if (new Set(results).size > 1) problems.push(`flaky: reference passed ${results.filter(Boolean).length} of ${results.length} runs`);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
  return { key: t.key, problems: [...new Set(problems)] };
}

export async function checkBench(projects: BenchProject[], o: CheckOptions, progress?: (c: TicketCheck) => void): Promise<TicketCheck[]> {
  const items = projects.flatMap((p) => p.tickets.map((t) => [p, t] as const));
  const out: TicketCheck[] = [];
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const [p, t] = items[next++];
      const c = await checkTicket(p, t, o);
      out.push(c);
      progress?.(c);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(o.concurrency ?? 8, items.length)) }, worker));
  return out.sort((a, b) => a.key.localeCompare(b.key));
}

if (import.meta.main) {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: { sandbox: { type: 'string' }, repeats: { type: 'string' }, 'max-seconds': { type: 'string' }, concurrency: { type: 'string' }, filter: { type: 'string' } },
  });
  if (!positionals[0]) {
    console.error('usage: bun bench/check.ts <bench dir> [--sandbox local|podman|docker] [--repeats 5] [--max-seconds 60]');
    process.exit(2);
  }
  let projects = findProjects(positionals[0]);
  if (values.filter) projects = projects.map((p) => ({ ...p, tickets: p.tickets.filter((t) => t.key.includes(values.filter!)) }));
  const sandbox = values.sandbox ? sandboxFromEnv({ MUGGE_SANDBOX: values.sandbox }) : new LocalSandbox();
  const checks = await checkBench(projects, {
    sandbox,
    repeats: values.repeats ? Number(values.repeats) : 5,
    maxSeconds: values['max-seconds'] ? Number(values['max-seconds']) : 60,
    concurrency: values.concurrency ? Number(values.concurrency) : 8,
  }, (c) => console.log(`${c.problems.length ? 'FAIL' : 'ok  '} ${c.key}${c.problems.map((x) => '\n     ' + x.replace(/\n/g, '\n     ')).join('')}`));
  const bad = checks.filter((c) => c.problems.length);
  console.log(`\n${checks.length - bad.length}/${checks.length} tickets admitted`);
  process.exit(bad.length ? 1 : 0);
}
