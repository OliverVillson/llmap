/**
 * MuggeBench projects on disk. A project is a folder shaped like spike/:
 *
 *   plan.json              Mugge's plan; each ticket carries a `bench` block the engine ignores:
 *                          { theme, language, role, hidden: ["<command>", ...] }
 *                          and the plan a top-level `bench` block: { canary, version, split }
 *   project/               the scaffold: interfaces written, owned files stubbed
 *   solutions/<id>.json    reference answers, { files: { path: content }, canary }
 *   hidden/<id>/<path>     hidden test files, copied over the worktree after the loop ends
 *
 * The model never sees hidden/ or solutions/. See plan/mugge-bench.md in the project files.
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { ancestors, topoOrder, validatePlan, type Plan, type Ticket } from '../src/tickets/schema.ts';

export const THEMES = ['backend', 'frontend', 'ml', 'systems', 'dotnet'] as const;
export const ROLES = ['implement', 'fix', 'test', 'refactor', 'integrate'] as const;

export interface BenchTags {
  theme: string;
  language: string;
  role: string;
  /** Commands run after the loop, with hidden/<id>/ copied in; all must exit 0. */
  hidden: string[];
}

export interface BenchTicket {
  /** `<project>/<ticket id>`, unique across the bench. */
  key: string;
  ticket: Ticket;
  tags: BenchTags;
  /** Ticket ids whose reference solutions go into the worktree first, in dependency order. */
  deps: string[];
}

export interface BenchProject {
  name: string;
  dir: string;
  plan: Plan;
  canary: string;
  split: string;
  tickets: BenchTicket[];
}

export function loadProject(dir: string, name = dir.split('/').filter(Boolean).pop()!): BenchProject {
  const raw = JSON.parse(readFileSync(join(dir, 'plan.json'), 'utf8'));
  const plan = validatePlan(raw);
  const rawById = new Map<string, any>(raw.tickets.map((t: any) => [t.id, t]));
  const anc = ancestors(plan.tickets);
  const order = topoOrder(plan.tickets).map((t) => t.id);
  const tickets = plan.tickets.map((ticket) => {
    const b = rawById.get(ticket.id)?.bench ?? {};
    const tags: BenchTags = {
      theme: String(b.theme ?? 'unknown'),
      language: String(b.language ?? ticket.image),
      role: String(b.role ?? ticket.role),
      hidden: Array.isArray(b.hidden) ? b.hidden.map(String) : [],
    };
    const deps = order.filter((id) => anc.get(ticket.id)?.has(id));
    return { key: `${name}/${ticket.id}`, ticket, tags, deps };
  });
  return { name, dir, plan, canary: String(raw.bench?.canary ?? ''), split: String(raw.bench?.split ?? 'test'), tickets };
}

/** Every project under root: root itself if it holds a plan.json, else each subfolder that does. */
export function findProjects(root: string): BenchProject[] {
  if (existsSync(join(root, 'plan.json'))) return [loadProject(root)];
  return readdirSync(root)
    .filter((n) => existsSync(join(root, n, 'plan.json')))
    .sort()
    .map((n) => loadProject(join(root, n), n));
}

export function solution(p: BenchProject, id: string): Record<string, string> {
  const f = join(p.dir, 'solutions', `${id}.json`);
  return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')).files ?? {} : {};
}

export function writeFiles(root: string, files: Record<string, string>): void {
  for (const [path, content] of Object.entries(files)) {
    const abs = join(root, path);
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, content);
  }
}

/** A fresh copy of the scaffold in dest with these tickets' reference solutions applied, in order. */
export function prepare(p: BenchProject, dest: string, ids: string[]): void {
  mkdirSync(dest, { recursive: true });
  cpSync(join(p.dir, 'project'), dest, { recursive: true });
  for (const id of ids) writeFiles(dest, solution(p, id));
}

function walk(dir: string, out: string[] = [], root = dir): string[] {
  if (!existsSync(dir)) return out;
  for (const n of readdirSync(dir)) {
    const abs = join(dir, n);
    if (statSync(abs).isDirectory()) walk(abs, out, root);
    else out.push(relative(root, abs));
  }
  return out;
}

/** Repo-relative paths of a ticket's hidden files. */
export function hiddenFiles(p: BenchProject, id: string): string[] {
  return walk(join(p.dir, 'hidden', id)).sort();
}

/** Copies a ticket's hidden files over the worktree. */
export function applyHidden(p: BenchProject, id: string, worktree: string): void {
  const src = join(p.dir, 'hidden', id);
  if (existsSync(src)) cpSync(src, worktree, { recursive: true });
}
