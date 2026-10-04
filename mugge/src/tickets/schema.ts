/**
 * The ticket: one unit of work the planner writes and the engine runs. `owns` is the conflict
 * guard (tickets that can run at once never own the same file) and `acceptance` is the
 * definition of done, checked by the engine rather than claimed by the model.
 *
 * Plain TypeScript validation, no schema library: the same rules are exported as JSON Schema
 * (PLAN_JSON_SCHEMA) so the planner can be served with constrained decoding and always parse.
 */

export const ROLES = ['architect', 'implement', 'test', 'fix', 'refactor', 'docs', 'review', 'integrate'] as const;
export type Role = (typeof ROLES)[number];

/** Toolchain image a ticket's acceptance commands run in. See sandbox/images/. */
export const IMAGES = ['c', 'node', 'python', 'web'] as const;
export type Image = (typeof IMAGES)[number];

export interface Budget {
  /** Model calls before the ticket ends failed (the first write plus fixes). */
  attempts: number;
  /** Total completion tokens across attempts. */
  max_tokens: number;
}

export interface Ticket {
  id: string;
  title: string;
  role: Role;
  /** Specialist (base + adapter) to route to, e.g. `coder-ts@v3`; the router may override. */
  model: string;
  /** 1 (highest) to 5. */
  priority: number;
  depends_on: string[];
  /** Files this ticket may write. Exact paths, relative to the repo root. */
  owns: string[];
  /** Files packed as read-only context. Globs with `*` are allowed. */
  reads: string[];
  /** What the model is told about the job. */
  context: string;
  /** Shell commands run in the ticket's worktree; all must exit 0. */
  acceptance: string[];
  /** Which toolchain image runs `acceptance`. */
  image: Image;
  budget: Budget;
  /** Model to hand the ticket to when the budget runs out; null = end failed. */
  escalate_to: string | null;
}

export interface Plan {
  /** Repo-relative paths of the scaffold's interface files, packed into every ticket's context. */
  interfaces: string[];
  tickets: Ticket[];
}

export const DEFAULT_BUDGET: Budget = { attempts: 4, max_tokens: 60000 };

const ID = /^[a-z0-9][a-z0-9-]{0,62}$/;

export class PlanError extends Error {
  constructor(public problems: string[]) {
    super(`invalid plan:\n  ${problems.join('\n  ')}`);
  }
}

function strings(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((s) => typeof s === 'string');
}

/** Fills defaults and checks one ticket; returns the problems found (empty = valid). */
export function normalizeTicket(raw: any, problems: string[] = []): Ticket {
  const where = `ticket ${typeof raw?.id === 'string' ? raw.id : '?'}`;
  const t: Ticket = {
    id: raw?.id,
    title: raw?.title ?? raw?.id,
    role: raw?.role ?? 'implement',
    model: raw?.model ?? 'coder',
    priority: raw?.priority ?? 3,
    depends_on: raw?.depends_on ?? [],
    owns: raw?.owns ?? [],
    reads: raw?.reads ?? [],
    context: raw?.context ?? '',
    acceptance: raw?.acceptance ?? [],
    image: raw?.image ?? 'node',
    budget: { ...DEFAULT_BUDGET, ...(raw?.budget ?? {}) },
    escalate_to: raw?.escalate_to ?? null,
  };
  if (typeof t.id !== 'string' || !ID.test(t.id)) problems.push(`${where}: id must match ${ID}`);
  if (typeof t.title !== 'string') problems.push(`${where}: title must be a string`);
  if (!ROLES.includes(t.role)) problems.push(`${where}: role must be one of ${ROLES.join(', ')}`);
  if (!IMAGES.includes(t.image)) problems.push(`${where}: image must be one of ${IMAGES.join(', ')}`);
  if (!Number.isInteger(t.priority) || t.priority < 1 || t.priority > 5) problems.push(`${where}: priority must be 1..5`);
  for (const k of ['depends_on', 'owns', 'reads', 'acceptance'] as const) {
    if (!strings(t[k])) problems.push(`${where}: ${k} must be a list of strings`);
  }
  if (strings(t.owns)) {
    if (t.role !== 'review' && t.role !== 'architect' && t.owns.length === 0) problems.push(`${where}: owns is empty`);
    for (const p of t.owns) if (!safePath(p)) problems.push(`${where}: owns path ${JSON.stringify(p)} must be relative, inside the repo, no globs`);
  }
  if (!Number.isInteger(t.budget.attempts) || t.budget.attempts < 1) problems.push(`${where}: budget.attempts must be >= 1`);
  if (!Number.isInteger(t.budget.max_tokens) || t.budget.max_tokens < 1) problems.push(`${where}: budget.max_tokens must be >= 1`);
  return t;
}

/** True for a plain repo-relative path: no leading slash, no `..`, no globs, not inside .git. */
export function safePath(p: string): boolean {
  if (typeof p !== 'string' || p === '' || p.startsWith('/') || p.includes('\\') || /[*?[\]]/.test(p)) return false;
  const parts = p.split('/');
  return !parts.some((s) => s === '' || s === '.' || s === '..') && parts[0] !== '.git';
}

/**
 * Validates a whole plan: every ticket, unique ids, known dependencies, no cycles, and no two
 * tickets owning the same file unless one depends (transitively) on the other, because only
 * then can they never run at the same time. Throws PlanError listing every problem.
 */
export function validatePlan(raw: any): Plan {
  const problems: string[] = [];
  if (!raw || !Array.isArray(raw.tickets)) throw new PlanError(['plan.tickets must be a list']);
  const interfaces = raw.interfaces ?? [];
  if (!strings(interfaces)) problems.push('plan.interfaces must be a list of strings');
  const tickets = raw.tickets.map((t: any) => normalizeTicket(t, problems));
  const byId = new Map<string, Ticket>();
  for (const t of tickets) {
    if (byId.has(t.id)) problems.push(`duplicate ticket id ${t.id}`);
    byId.set(t.id, t);
  }
  for (const t of tickets) {
    for (const d of t.depends_on ?? []) {
      if (!byId.has(d)) problems.push(`ticket ${t.id}: depends on unknown ticket ${d}`);
      if (d === t.id) problems.push(`ticket ${t.id}: depends on itself`);
    }
  }
  if (problems.length) throw new PlanError(problems);
  const cycle = findCycle(tickets);
  if (cycle) throw new PlanError([`dependency cycle: ${cycle.join(' -> ')}`]);
  const anc = ancestors(tickets);
  for (let i = 0; i < tickets.length; i++) {
    for (let j = i + 1; j < tickets.length; j++) {
      const a = tickets[i], b = tickets[j];
      const shared = a.owns.filter((p: string) => b.owns.includes(p));
      if (shared.length && !anc.get(a.id)!.has(b.id) && !anc.get(b.id)!.has(a.id)) {
        problems.push(`tickets ${a.id} and ${b.id} can run at once and both own ${shared.join(', ')}`);
      }
    }
  }
  if (problems.length) throw new PlanError(problems);
  return { interfaces, tickets };
}

function findCycle(tickets: Ticket[]): string[] | null {
  const byId = new Map(tickets.map((t) => [t.id, t]));
  const state = new Map<string, 1 | 2>();
  const stack: string[] = [];
  const visit = (id: string): string[] | null => {
    if (state.get(id) === 2) return null;
    if (state.get(id) === 1) return [...stack.slice(stack.indexOf(id)), id];
    state.set(id, 1);
    stack.push(id);
    for (const d of byId.get(id)!.depends_on) {
      const c = visit(d);
      if (c) return c;
    }
    stack.pop();
    state.set(id, 2);
    return null;
  };
  for (const t of tickets) {
    const c = visit(t.id);
    if (c) return c;
  }
  return null;
}

/** id → every ticket it depends on, directly or not. */
export function ancestors(tickets: Ticket[]): Map<string, Set<string>> {
  const byId = new Map(tickets.map((t) => [t.id, t]));
  const memo = new Map<string, Set<string>>();
  const get = (id: string): Set<string> => {
    let s = memo.get(id);
    if (s) return s;
    s = new Set();
    memo.set(id, s);
    for (const d of byId.get(id)!.depends_on) {
      s.add(d);
      for (const x of get(d)) s.add(x);
    }
    return s;
  };
  for (const t of tickets) get(t.id);
  return memo;
}

/** Tickets in an order where every ticket comes after its dependencies (priority breaks ties). */
export function topoOrder(tickets: Ticket[]): Ticket[] {
  const done = new Set<string>();
  const out: Ticket[] = [];
  const left = [...tickets].sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id));
  while (left.length) {
    const i = left.findIndex((t) => t.depends_on.every((d) => done.has(d)));
    if (i < 0) throw new PlanError(['dependency cycle']);
    const [t] = left.splice(i, 1);
    done.add(t.id);
    out.push(t);
  }
  return out;
}

/** JSON Schema of one ticket, for constrained decoding of the planner's output. */
export const TICKET_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'title', 'role', 'model', 'priority', 'depends_on', 'owns', 'reads', 'context', 'acceptance', 'image'],
  properties: {
    id: { type: 'string', pattern: ID.source },
    title: { type: 'string' },
    role: { enum: [...ROLES] },
    model: { type: 'string' },
    priority: { type: 'integer', minimum: 1, maximum: 5 },
    depends_on: { type: 'array', items: { type: 'string' } },
    owns: { type: 'array', items: { type: 'string' } },
    reads: { type: 'array', items: { type: 'string' } },
    context: { type: 'string' },
    acceptance: { type: 'array', items: { type: 'string' } },
    image: { enum: [...IMAGES] },
    budget: {
      type: 'object',
      additionalProperties: false,
      properties: { attempts: { type: 'integer', minimum: 1 }, max_tokens: { type: 'integer', minimum: 1 } },
    },
    escalate_to: { type: ['string', 'null'] },
  },
} as const;

export const PLAN_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['interfaces', 'tickets'],
  properties: {
    interfaces: { type: 'array', items: { type: 'string' } },
    tickets: { type: 'array', items: TICKET_JSON_SCHEMA },
  },
} as const;

/** What a coder model returns for one call: full new contents of the files it writes. */
export interface WorkerOutput {
  files: Record<string, string>;
  note: string;
}

export const WORKER_OUTPUT_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['files', 'note'],
  properties: {
    files: { type: 'object', additionalProperties: { type: 'string' } },
    note: { type: 'string' },
  },
} as const;
