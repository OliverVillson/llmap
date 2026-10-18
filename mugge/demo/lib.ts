/**
 * The demo's fixed parts: the target project (project/, the scaffold every ticket starts
 * from), the plan the orchestrator works from (plan.json) and the reference answers
 * (reference/) the mock model hands back, so the whole run can be rehearsed without a GPU.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { validatePlan, type Plan, type WorkerOutput } from '../src/tickets/schema.ts';

export const DEMO_DIR = import.meta.dir;
export const PROJECT_DIR = join(DEMO_DIR, 'project');
export const PLAN_PATH = join(DEMO_DIR, 'plan.json');
export const REFERENCE_DIR = join(DEMO_DIR, 'reference');

/** What the crew is asked to build, as the orchestrator is told it. */
export const REQUEST = 'Build mdlite, a small Markdown to HTML converter in plain Python 3 with a command line (python3 -m mdlite), from the scaffold: every module is stubbed and has its tests.';

/** The prepared plan, validated: the planner settles the one file clash (block-code waits on block-table). */
export function loadPlan(path = PLAN_PATH): Plan {
  return validatePlan(JSON.parse(readFileSync(path, 'utf8')));
}

/**
 * The reference answer for each ticket: its owned files as the finished project has them.
 * A ticket can carry its own version of a shared file (reference/_by-ticket/<id>/...): the
 * table ticket writes the stylesheet before the code ticket adds to it.
 */
export function referenceAnswers(plan: Plan): Record<string, WorkerOutput> {
  const out: Record<string, WorkerOutput> = {};
  for (const t of plan.tickets) {
    const files: Record<string, string> = {};
    for (const p of t.owns) {
      const own = join(REFERENCE_DIR, '_by-ticket', t.id, p);
      files[p] = readFileSync(existsSync(own) ? own : join(REFERENCE_DIR, p), 'utf8');
    }
    out[t.id] = { files, note: `${t.title.toLowerCase()} done` };
  }
  return out;
}

/** Every ticket a ticket needs first, its own dependencies' dependencies included. */
export function ancestors(plan: Plan, id: string, seen = new Set<string>()): Set<string> {
  for (const d of plan.tickets.find((t) => t.id === id)?.depends_on ?? []) {
    if (!seen.has(d)) {
      seen.add(d);
      ancestors(plan, d, seen);
    }
  }
  return seen;
}
