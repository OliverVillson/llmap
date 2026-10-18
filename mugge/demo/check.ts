#!/usr/bin/env bun
/**
 * Proves the demo task is fair before any model sees it, the way spike/check.sh does for the spike:
 *   (a) every ticket's acceptance fails on the bare scaffold,
 *   (b) ...and still fails with only the tickets it waits on applied (its test needs its own work),
 *   (c) it passes with those tickets and its own reference answer,
 *   (d) with every answer applied, the whole test suite passes and the command line renders README.md.
 *
 *   bun demo/check.ts
 */
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { ancestors, loadPlan, PROJECT_DIR, referenceAnswers } from './lib.ts';

const plan = loadPlan();
const answers = referenceAnswers(plan);
const work = mkdtempSync(join(tmpdir(), 'mdlite-check-'));
let failures = 0;

/** A copy of the scaffold with these tickets' answers written in, in plan order (later answers win). */
function tree(name: string, ids: Iterable<string>): string {
  const dir = join(work, name);
  cpSync(PROJECT_DIR, dir, { recursive: true });
  const want = new Set(ids);
  for (const t of plan.tickets) {
    if (!want.has(t.id)) continue;
    for (const [p, content] of Object.entries(answers[t.id]!.files)) {
      mkdirSync(dirname(join(dir, p)), { recursive: true });
      writeFileSync(join(dir, p), content);
    }
  }
  return dir;
}

function passes(dir: string, commands: string[]): { ok: boolean; out: string } {
  for (const c of commands) {
    const r = Bun.spawnSync(['bash', '-c', c], { cwd: dir, stdout: 'pipe', stderr: 'pipe' });
    if (r.exitCode !== 0) return { ok: false, out: (r.stdout.toString() + r.stderr.toString()).trim() };
  }
  return { ok: true, out: '' };
}

function bad(what: string, out = '') {
  failures++;
  console.log(`  FAIL ${what}`);
  if (out) console.log(out.split('\n').slice(-12).map((l) => `    | ${l}`).join('\n'));
}

try {
  for (const t of plan.tickets) {
    const before = ancestors(plan, t.id);
    if (passes(tree(`${t.id}-scaffold`, []), t.acceptance).ok) bad(`${t.id}: passes on the bare scaffold`);
    if (before.size && passes(tree(`${t.id}-deps`, before), t.acceptance).ok) bad(`${t.id}: passes with only ${[...before].join(', ')}`);
    const r = passes(tree(`${t.id}-own`, [...before, t.id]), t.acceptance);
    if (!r.ok) bad(`${t.id}: fails with its answer and ${before.size ? [...before].join(', ') : 'nothing else'}`, r.out);
    else console.log(`  ok ${t.id}${before.size ? ` (after ${[...before].join(', ')})` : ''}`);
  }
  const all = tree('all', plan.tickets.map((t) => t.id));
  const suite = passes(all, ['python3 -m unittest discover -s tests -t .', 'python3 -m mdlite README.md --standalone --toc -o README.html']);
  if (!suite.ok) bad('the whole project with every answer', suite.out);
  else console.log('  ok whole project: every test passes and README.md renders');
  const settled = plan.settled ?? [];
  console.log(`\n${plan.tickets.length} tickets, ${plan.tickets.filter((t) => !t.depends_on.length).length} ready at the start, ${settled.length} clash settled (${settled.map((c) => `${c.ticket} after ${c.with.join(', ')} for ${c.files.join(', ')}`).join('; ')})`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
console.log(failures ? `\n${failures} problem(s)` : '\nthe demo task is fair');
process.exit(failures ? 1 : 0);
