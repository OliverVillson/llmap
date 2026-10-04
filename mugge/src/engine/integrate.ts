/**
 * The integrate step: merge every finished ticket branch onto `mugge/integration` in
 * dependency order, then run every acceptance command again on the merged tree. Because no two
 * concurrent tickets own the same file, the merges are expected to be clean; a conflict or a
 * failing check becomes a `fix` ticket that owns the files involved.
 */
import type { Plan, Ticket } from '../tickets/schema.ts';
import { DEFAULT_BUDGET, topoOrder } from '../tickets/schema.ts';
import { branchFor, git, gitOk, INTEGRATION_BRANCH, type Worktrees } from './git.ts';
import type { TicketResult } from './harness.ts';
import { tail, type Sandbox } from './sandbox.ts';

export interface IntegrateOptions {
  repo: string;
  base: string;
  plan: Plan;
  /** Ids of the tickets that ended done. */
  done: string[];
  sandbox: Sandbox;
  worktrees: Worktrees;
  checkTimeoutMs?: number;
  /** Runs a fix ticket inside the integration worktree; one round, when given. */
  fixer?: (ticket: Ticket, worktree: string) => Promise<TicketResult>;
}

export interface IntegrateResult {
  ok: boolean;
  branch: string;
  commit: string | null;
  merged: string[];
  /** Tickets whose acceptance failed on the merged tree. */
  failedChecks: string[];
  error: string | null;
  /** The fix ticket made from the failure, whether or not a fixer ran it. */
  fix: Ticket | null;
}

async function checkAll(o: IntegrateOptions, path: string, tickets: Ticket[]): Promise<{ failed: string[]; log: string }> {
  const failed: string[] = [];
  let log = '';
  for (const t of tickets) {
    for (const command of t.acceptance) {
      const r = await o.sandbox.run(command, { cwd: path, image: t.image, timeoutMs: o.checkTimeoutMs });
      if (r.code !== 0) {
        failed.push(t.id);
        log += `$ ${command}\n${tail(r.out, 1500)}\n`;
        break;
      }
    }
  }
  return { failed, log };
}

export function fixTicketFor(plan: Plan, ids: string[], error: string): Ticket {
  const involved = plan.tickets.filter((t) => ids.includes(t.id));
  return {
    id: 'integrate-fix',
    title: `Fix integration of ${ids.join(', ')}`,
    role: 'fix',
    model: involved[0]?.model ?? 'coder',
    priority: 1,
    depends_on: [],
    owns: [...new Set(involved.flatMap((t) => t.owns))],
    reads: [...new Set(involved.flatMap((t) => t.reads))],
    context: `The merged tree fails these tickets' checks. Fix the owned files so every check passes.\n\n${error}`,
    acceptance: involved.flatMap((t) => t.acceptance),
    image: involved[0]?.image ?? 'node',
    budget: DEFAULT_BUDGET,
    escalate_to: null,
  };
}

export async function integrate(o: IntegrateOptions): Promise<IntegrateResult> {
  const order = topoOrder(o.plan.tickets).filter((t) => o.done.includes(t.id));
  const path = o.worktrees.create('integration', INTEGRATION_BRANCH, o.base);
  const res: IntegrateResult = { ok: false, branch: INTEGRATION_BRANCH, commit: null, merged: [], failedChecks: [], error: null, fix: null };
  try {
    for (const t of order) {
      const r = git(path, ['merge', '-q', '--no-edit', '-m', `Merge ${t.title}`, branchFor(t.id)], o.worktrees.author);
      if (r.code !== 0) {
        const conflicted = gitOk(path, ['diff', '--name-only', '--diff-filter=U']).split('\n').filter(Boolean);
        git(path, ['merge', '--abort']);
        res.error = `merge of ${t.id} conflicts on ${conflicted.join(', ') || '?'}`;
        res.failedChecks = [t.id];
        res.fix = fixTicketFor(o.plan, [t.id], res.error);
        return res;
      }
      res.merged.push(t.id);
    }
    let { failed, log } = await checkAll(o, path, order);
    if (failed.length) {
      res.fix = fixTicketFor(o.plan, failed, log);
      if (o.fixer) {
        const fr = await o.fixer(res.fix, path);
        if (fr.ok) {
          o.worktrees.commit(path, res.fix.owns, res.fix.title);
          ({ failed, log } = await checkAll(o, path, order));
        }
      }
    }
    res.failedChecks = failed;
    res.ok = failed.length === 0;
    res.error = res.ok ? null : log;
    res.commit = gitOk(path, ['rev-parse', 'HEAD']);
    return res;
  } finally {
    o.worktrees.remove('integration');
  }
}
