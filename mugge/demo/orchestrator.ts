/**
 * The orchestrator's planning call, for `run.ts --plan orchestrator`: one constrained-JSON call on
 * the manager slot that turns the request and the scaffold into the ticket plan. Its plan is
 * checked against the scaffold before anything runs (every stub filled by exactly one ticket,
 * every test some ticket's check, dependencies that cover what each test imports); a plan that
 * fails the check is not used and the prepared plan runs instead, so the demo never stalls on a
 * bad plan. Clashes in either plan are settled by validatePlan, as in every engine run.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Inference } from '../src/engine/inference.ts';
import { PLAN_JSON_SCHEMA, validatePlan, type Plan } from '../src/tickets/schema.ts';
import { REQUEST } from './lib.ts';

export const PLANNER_SYSTEM = [
  'You are the orchestrator of a crew of coding agents. You split a scaffolded project into tickets that the agents fill in at the same time.',
  'Each ticket owns the files it writes (exact paths), lists the files it should read, and names the acceptance commands that prove it done.',
  'Tickets that can run at the same time should not own the same file; if two must, list the one that should go first earlier in the plan.',
  'A ticket depends on the tickets whose code its tests run. Prefer many small tickets with few dependencies, so most can run at once.',
  'Reply with one JSON object: {"interfaces": [...], "tickets": [...]}.',
].join('\n');

const MODULE_IMPORT = /^\s*from (mdlite(?:\.\w+)?) import ([\w, ]+)|^\s*import (mdlite\.\w+)/gm;

/** The modules a Python file imports from the package (mdlite.x, or mdlite's own names for `from mdlite import a, b`). */
export function importsOf(text: string): string[] {
  const out = new Set<string>();
  for (const m of text.matchAll(MODULE_IMPORT)) {
    if (m[3]) out.add(m[3]);
    else if (m[1] === 'mdlite') for (const n of m[2]!.split(',')) out.add(`mdlite.${n.trim()}`);
    else out.add(m[1]!);
  }
  return [...out].filter((x) => x !== 'mdlite.types');
}

const modPath = (m: string) => (m === 'mdlite.__main__' ? 'mdlite/__main__.py' : `${m.replace('.', '/')}.py`);

/** The user message: the request, the architecture, every stub with what it imports, every test with what it runs. */
export function plannerMessage(repo: string): string {
  const stubs = readdirSync(join(repo, 'mdlite')).filter((f) => f.endsWith('.py') && f !== '__init__.py' && f !== 'types.py').sort();
  const tests = readdirSync(join(repo, 'tests')).filter((f) => f.startsWith('test_')).sort();
  return [
    REQUEST,
    '',
    readFileSync(join(repo, 'ARCHITECTURE.md'), 'utf8'),
    '## Files to fill in (stubs that raise NotImplementedError)',
    ...stubs.map((f) => {
      const text = readFileSync(join(repo, 'mdlite', f), 'utf8');
      const imp = importsOf(text);
      const shared = /also owns (mdlite\/[\w.]+)/.exec(text.replace(/\s+/g, ' '))?.[1];
      return `- mdlite/${f}${imp.length ? ` (imports ${imp.join(', ')})` : ''}${shared ? `; its docstring says the ticket also owns ${shared}` : ''}`;
    }),
    '',
    '## Tests (run each with: python3 -m unittest -v tests.<name>)',
    ...tests.map((f) => `- tests/${f}: imports ${importsOf(readFileSync(join(repo, 'tests', f), 'utf8')).join(', ') || 'nothing from mdlite'}`),
    '',
    '## Rules for this project',
    '- One ticket per stub file; give each ticket a short lowercase id with dashes (e.g. block-table).',
    '- owns: the stub file, plus any shared file its docstring says it owns. reads: ARCHITECTURE.md, its test file and the modules it imports.',
    '- acceptance: ["python3 -m unittest -v tests.test_<module>"]; image "python"; model "coder"; role "implement".',
    '- depends_on: the tickets that fill in the modules its test needs at run time (what the stub and the test import, and what those import).',
    '- interfaces: ["ARCHITECTURE.md", "mdlite/types.py"]. budget: {"attempts": 4, "max_tokens": 96000}. escalate_to: null.',
  ].join('\n');
}

/**
 * What is wrong with a plan for this scaffold, or nothing: every stub owned by exactly one
 * ticket (shared files aside), every test some ticket's acceptance, and each ticket's
 * dependencies covering every module its test needs at run time.
 */
export function checkPlan(plan: Plan, repo: string): string[] {
  const problems: string[] = [];
  const stubs = readdirSync(join(repo, 'mdlite')).filter((f) => f.endsWith('.py') && f !== '__init__.py' && f !== 'types.py').map((f) => `mdlite/${f}`);
  const owner = new Map<string, string>();
  for (const t of plan.tickets) {
    for (const p of t.owns) {
      if (!existsSync(join(repo, p))) problems.push(`${t.id} owns ${p}, which is not in the scaffold`);
      if (stubs.includes(p)) {
        if (owner.has(p)) problems.push(`${p} is owned by both ${owner.get(p)} and ${t.id}`);
        owner.set(p, t.id);
      }
    }
  }
  for (const s of stubs) if (!owner.has(s)) problems.push(`no ticket fills in ${s}`);
  for (const f of readdirSync(join(repo, 'tests')).filter((x) => x.startsWith('test_'))) {
    const mod = `tests.${f.replace(/\.py$/, '')}`;
    if (!plan.tickets.some((t) => t.acceptance.some((c) => c.includes(mod)))) problems.push(`no ticket runs ${mod}`);
  }
  // the modules a ticket's test needs, through what each module imports
  const needs = (start: string[]): Set<string> => {
    const seen = new Set<string>(), todo = [...start];
    while (todo.length) {
      const m = todo.pop()!;
      if (seen.has(m)) continue;
      seen.add(m);
      const path = join(repo, modPath(m));
      if (existsSync(path)) todo.push(...importsOf(readFileSync(path, 'utf8')));
    }
    return seen;
  };
  const anc = (id: string, seen = new Set<string>()): Set<string> => {
    for (const d of plan.tickets.find((t) => t.id === id)?.depends_on ?? []) if (!seen.has(d)) { seen.add(d); anc(d, seen); }
    return seen;
  };
  for (const t of plan.tickets) {
    const before = anc(t.id);
    for (const c of t.acceptance) {
      const test = /tests\.(test_\w+)/.exec(c)?.[1];
      if (!test || !existsSync(join(repo, 'tests', `${test}.py`))) continue;
      for (const m of needs(importsOf(readFileSync(join(repo, 'tests', `${test}.py`), 'utf8')))) {
        const o = owner.get(modPath(m));
        if (o && o !== t.id && !before.has(o)) problems.push(`${t.id}: its test needs ${m} (${o}) but does not wait for it`);
      }
    }
  }
  return problems;
}

export interface PlanOutcome {
  plan: Plan;
  /** Who made the plan that runs. */
  source: 'orchestrator' | 'prepared';
  /** Why the orchestrator's plan was not used, when it was not. */
  rejected?: string[];
  promptTokens: number;
  completionTokens: number;
  ms: number;
}

export async function orchestratorPlan(inf: Inference, model: string, repo: string, prepared: Plan, maxTokens = 32_000): Promise<PlanOutcome> {
  const messages = [{ role: 'system' as const, content: PLANNER_SYSTEM }, { role: 'user' as const, content: plannerMessage(repo) }];
  let c;
  try {
    c = await inf.complete<Plan>(model, messages, PLAN_JSON_SCHEMA, maxTokens);
  } catch (e) {
    return { plan: prepared, source: 'prepared', rejected: [`the planning call failed: ${String((e as Error).message ?? e).slice(0, 300)}`], promptTokens: 0, completionTokens: 0, ms: 0 };
  }
  const usage = { promptTokens: c.promptTokens, completionTokens: c.completionTokens, ms: c.ms };
  let plan: Plan;
  try {
    plan = validatePlan(c.value);
  } catch (e) {
    return { plan: prepared, source: 'prepared', rejected: [String((e as Error).message)], ...usage };
  }
  const problems = checkPlan(plan, repo);
  if (problems.length) return { plan: prepared, source: 'prepared', rejected: problems, ...usage };
  return { plan, source: 'orchestrator', ...usage };
}
