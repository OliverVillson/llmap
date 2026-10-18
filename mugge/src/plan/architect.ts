/**
 * Architect and planner: the two calls that turn a request into work.
 *
 *   architect: request → design (components, stacks, interfaces, decisions) → ARCHITECTURE.md
 *   planner, step 1 (scaffold): design → real files: folder tree, interfaces, stubs, one failing
 *     test per feature, committed as the base every ticket starts from
 *   planner, step 2 (tickets): design + scaffold file list → a validated Plan
 *
 * Every call is constrained JSON, so every answer parses. They run on the bigger model
 * (`architect` / `planner` names in the inference config): few calls, high leverage.
 * With thinking on (the default; Qwen3.8 thinks first) each call gets PLANNER_THINK_TOKENS on top
 * of its answer budget, so the thinking does not eat the JSON. Planning calls are not capped by
 * MUGGE_MAX_CALL_TOKENS (`planning` tweak): their answers grow with the crew.
 *
 * The crew size (`crew`) goes into every prompt: about one ticket per code agent, most of them
 * free to start at once. A crew above SPLIT_AT with `parts` given splits the scaffold and the
 * tickets per design component, run in parallel on the parts' slots: one call writes the shared
 * files (build files and every interface), then one call per component writes its stubs and
 * tests, then one call per component plans its tickets. A 64-ticket plan is about 13–16K tokens
 * of JSON and a 64-module scaffold about 55K, more than one call should write.
 */
import { thinkingOn } from '../engine/harness.ts';
import type { CallTweaks, ChatMessage, Completion, Inference } from '../engine/inference.ts';
import { IMAGES, PLAN_JSON_SCHEMA, safePath, settleClashes, validatePlan, WORKER_OUTPUT_JSON_SCHEMA, type Plan, type Ticket, type WorkerOutput } from '../tickets/schema.ts';

export interface Design {
  summary: string;
  components: Array<{
    name: string;
    language: string;
    image: (typeof IMAGES)[number];
    path: string;
    responsibility: string;
    interfaces: Array<{ path: string; description: string }>;
  }>;
  data_flow: string;
  decisions: Array<{ decision: string; why: string }>;
  /** Up to three questions for the user; empty when the request is clear enough. */
  questions: string[];
  /** What the planner settled while designing (a call asked again); not part of the model's answer. */
  warnings?: string[];
}

export const DESIGN_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary', 'components', 'data_flow', 'decisions', 'questions'],
  properties: {
    summary: { type: 'string' },
    components: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'language', 'image', 'path', 'responsibility', 'interfaces'],
        properties: {
          name: { type: 'string' },
          language: { type: 'string' },
          image: { enum: [...IMAGES] },
          path: { type: 'string' },
          responsibility: { type: 'string' },
          interfaces: {
            type: 'array',
            items: { type: 'object', additionalProperties: false, required: ['path', 'description'], properties: { path: { type: 'string' }, description: { type: 'string' } } },
          },
        },
      },
    },
    data_flow: { type: 'string' },
    decisions: {
      type: 'array',
      items: { type: 'object', additionalProperties: false, required: ['decision', 'why'], properties: { decision: { type: 'string' }, why: { type: 'string' } } },
    },
    questions: { type: 'array', items: { type: 'string' }, maxItems: 3 },
  },
} as const;

const ARCHITECT_SYSTEM = [
  'You are the software architect for a project that many small coding models will build in parallel.',
  'Pick a stack per component (C, TypeScript/JavaScript, Python, or web frontend), module boundaries, and the public interfaces between modules.',
  'Interfaces must be concrete files (headers, type modules, Python protocols) so coders can work behind them without talking to each other.',
  'Ask at most three questions, only when the answer changes the design. Reply with one JSON object.',
].join('\n');

const SCAFFOLD_SYSTEM = [
  'You write the scaffold of a project from its design: the folder tree, every interface file in full,',
  'a stub for each implementation file (compiles or imports, but does not work yet), build files, and one failing test per feature.',
  'Reply with {"files": {path: content}, "note": one line}.',
].join('\n');

const SCAFFOLD_SHARED_SYSTEM = [
  'You write the shared part of a project scaffold from its design: the build files, the test setup, and every interface file in full.',
  'Write no implementation stubs and no tests: other calls write those, one component at a time, behind the interfaces you write.',
  'Reply with {"files": {path: content}, "note": one line}.',
].join('\n');

const SCAFFOLD_PART_SYSTEM = [
  'You write one component of a project scaffold: a stub for each of its implementation files (compiles or imports, but does not work yet)',
  'and one failing test per feature. The build files and every interface file already exist, as shown; do not write them again.',
  'Write only files of your component: under its folder, and tests under its folder or in a test file named after it.',
  'Reply with {"files": {path: content}, "note": one line}.',
].join('\n');

const TICKETS_SYSTEM = [
  'You split a scaffolded project into tickets for parallel coding agents.',
  'Each ticket owns the implementation files it fills in; tickets that can run at the same time never own the same file.',
  'Each ticket lists acceptance commands (its tests, plus build/type checks) that pass only when it is done, and the toolchain image they run in.',
  'Prefer many small tickets with few dependencies, so most can run at once. Reply with {"interfaces": [...], "tickets": [...]}.',
].join('\n');

const TICKETS_PART_SYSTEM = [
  'You split one component of a scaffolded project into tickets for parallel coding agents. Other calls plan the other components.',
  'Each ticket owns implementation files of your component only; no two tickets own the same file.',
  'Each ticket lists acceptance commands (its tests, plus build/type checks) that pass only when it is done, and the toolchain image they run in.',
  'Work behind the interfaces: depend only on tickets of your own component, and only when a test needs their code. Reply with {"interfaces": [...], "tickets": [...]}.',
].join('\n');

/** Tokens a planning call may think before its answer, on top of the answer's own budget. */
export const PLANNER_THINK_TOKENS = 32_768;
/** Crews above this split the scaffold and the tickets per component, when parts are given. */
export const SPLIT_AT = 16;
/** Answer tokens per ticket of a plan (a ticket is about 200 tokens of JSON) and per scaffolded module (a stub and its test). */
export const TOKENS_PER_TICKET = 400;
export const TOKENS_PER_MODULE = 1200;
/** Every planned ticket's budget: room for 4 attempts at a 16K call cap plus the review. */
export const PLANNED_BUDGET = { attempts: 4, max_tokens: 96_000 } as const;
/** A plan with fewer tickets than this share of the crew is asked for again, once. */
export const MIN_CREW_SHARE = 0.75;
/** A plan (or a part's) where fewer than this share of its tickets can start at once is asked for again, once. */
export const MIN_START_SHARE = 0.5;
/** Implementation files per component the design aims for, so each part's scaffold and tickets fit one call. */
export const FILES_PER_COMPONENT = 16;

export interface PlanOptions {
  /** Code agents in the crew: the plan aims for about this many tickets, most of them free to start at once. */
  crew?: number;
  /** The client for parallel part `i` (one per component); unset, or a crew of SPLIT_AT or fewer, plans in one call each. */
  parts?: (i: number) => Inference;
}

const PLANNING: CallTweaks = { planning: true };
const budget = (inf: Inference, answer: number) => answer + (thinkingOn(inf) ? PLANNER_THINK_TOKENS : 0);
const splits = (d: Design, o: PlanOptions) => !!o.parts && (o.crew ?? 0) > SPLIT_AT && d.components.length >= 2;

const crewLine = (crew: number | undefined, what: string) =>
  crew ? `\n\n## Crew\n${crew} coding agents work at the same time. ${what}` : '';

/** Each component's share of the crew, by how many interfaces it has (at least one each). */
export function crewShares(d: Design, crew: number): number[] {
  const w = d.components.map((c) => c.interfaces.length + 1);
  const total = w.reduce((a, b) => a + b, 0);
  const raw = w.map((x) => (x / total) * crew);
  const out = raw.map((x) => Math.max(1, Math.floor(x)));
  // hand out what flooring left, biggest remainders first
  let left = crew - out.reduce((a, b) => a + b, 0);
  for (const i of raw.map((x, i) => [x - Math.floor(x), i] as const).sort((a, b) => b[0] - a[0]).map(([, i]) => i)) {
    if (left <= 0) break;
    out[i]!++;
    left--;
  }
  return out;
}

/** Components a crew's design should have at least: about FILES_PER_COMPONENT files each. */
const componentsFor = (crew: number) => Math.max(2, Math.ceil(crew / FILES_PER_COMPONENT));

export async function design(inf: Inference, request: string, answers: string[] = [], model = 'architect', o: PlanOptions = {}): Promise<Design> {
  const asked = answers.length ? `${request}\n\nAnswers to your questions:\n${answers.join('\n')}` : request;
  const split = !!o.parts && (o.crew ?? 0) > SPLIT_AT;
  const many = split ? ` Use at least ${componentsFor(o.crew!)} components, each with its own folder and about ${FILES_PER_COMPONENT} files at most.` : '';
  const user = asked + crewLine(o.crew, `Size the design for them: about ${o.crew} implementation files of similar size, one per agent, spread over components that each own a folder.${many}`);
  const messages = [{ role: 'system' as const, content: ARCHITECT_SYSTEM }, { role: 'user' as const, content: user }];
  const max = budget(inf, Math.max(8000, (o.crew ?? 0) * 150));
  const warnings: string[] = [];
  const withWarnings = (d: Design): Design => (warnings.length ? { ...d, warnings } : d);
  const first = await planCall<Design>(inf, model, messages, DESIGN_JSON_SCHEMA, max, 'the design', warnings);
  if (!Array.isArray(first?.components) || !first.components.length) throw new Error('the design came back with no components');
  // a big crew on one component would scaffold and plan in one call each, which cannot fit: ask once more
  if (!split || first.components.length >= 2) return withWarnings(first);
  const again = [...messages, { role: 'assistant' as const, content: JSON.stringify(first) }, { role: 'user' as const, content: `That design has ${first.components.length} component${first.components.length === 1 ? '' : 's'}, too few for ${o.crew} agents. Split it into at least ${componentsFor(o.crew!)} components, each owning its own folder with about ${FILES_PER_COMPONENT} files at most, and reply with the whole design again.` }];
  let second: Design | undefined;
  try {
    second = await planCall<Design>(inf, model, again, DESIGN_JSON_SCHEMA, max, 'the design (asked again)', warnings);
  } catch (e) {
    warnings.push(`the design asked again failed (${why(e)}); the first answer is kept`);
  }
  return withWarnings(Array.isArray(second?.components) && second.components.length > first.components.length ? second : first);
}

export function architectureMarkdown(d: Design): string {
  return [
    '# Architecture',
    '',
    d.summary,
    '',
    '## Components',
    '',
    '| Component | Language | Path | Responsibility |',
    '|---|---|---|---|',
    ...d.components.map((c) => `| ${c.name} | ${c.language} | \`${c.path}\` | ${c.responsibility} |`),
    '',
    '## Interfaces',
    '',
    ...d.components.flatMap((c) => c.interfaces.map((i) => `- \`${i.path}\` (${c.name}): ${i.description}`)),
    '',
    '## Data flow',
    '',
    d.data_flow,
    '',
    '## Decisions',
    '',
    ...d.decisions.map((x) => `- **${x.decision}.** ${x.why}`),
    '',
  ].join('\n');
}

export interface Scaffold extends WorkerOutput {
  /** What the split left out or settled (a file two components both wrote, a file outside its component, a part that failed). */
  warnings: string[];
  /** The component (index in the design) whose part wrote each file; files the shared call wrote are not listed. */
  writtenBy?: Record<string, number>;
  /** Components (indexes) whose part failed: they have no stubs and get no tickets. */
  failed?: number[];
}

/**
 * A model-written path as a plain repo-relative one: leading `./` and `/` dropped, doubled slashes
 * merged. Undefined when it is still not safe (`..`, globs, inside .git, a folder).
 */
export function normPath(p: string): string | undefined {
  if (typeof p !== 'string') return undefined;
  const q = p.trim().replace(/\\/g, '/').replace(/\/{2,}/g, '/').replace(/^(\.\/|\/)+/, '');
  return safePath(q) ? q : undefined;
}

/**
 * The complete part of a JSON answer cut at max_tokens: everything up to the last comma between
 * two items of the top object or of a list or object directly inside it (a whole file of a
 * scaffold, a whole ticket of a plan), with the open brackets closed. Undefined when nothing is.
 */
export function salvageJson(raw: string): unknown {
  const start = raw.indexOf('{');
  if (start < 0) return undefined;
  const stack: string[] = [];
  const cuts: Array<[number, string]> = [];
  let inString = false, escaped = false;
  for (let i = start; i < raw.length; i++) {
    const ch = raw[i]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{' || ch === '[') stack.push(ch === '{' ? '}' : ']');
    else if (ch === '}' || ch === ']') {
      stack.pop();
      if (!stack.length) break;
    } else if (ch === ',' && stack.length <= 2) cuts.push([i, [...stack].reverse().join('')]);
  }
  for (let k = cuts.length - 1; k >= 0 && k >= cuts.length - 4; k--) {
    try {
      return JSON.parse(raw.slice(start, cuts[k]![0]) + cuts[k]![1]);
    } catch {
      /* the next earlier cut */
    }
  }
  return undefined;
}

/** Pause before a planning call that failed with a server error is sent once more. */
export let PLAN_RETRY_MS = 5_000;
export const setPlanRetryMs = (ms: number) => { PLAN_RETRY_MS = ms; };
/** Errors worth one more try: the server's 5xx, a dropped or refused connection (Bun: "Unable to connect"), a timeout. */
const TRANSIENT = /inference 5\d\d|fetch failed|ECONNRESET|ECONNREFUSED|Unable to connect|socket|connection|timed? ?out|aborted due to timeout/i;

/**
 * One planning call. An answer cut at max_tokens keeps its complete part (salvageJson) and says
 * so in `warnings`; a server error is tried once more; anything else that fails throws as before.
 */
async function planCall<T>(client: Inference, model: string, messages: ChatMessage[], schema: object, max: number, what: string, warnings: string[], retryMs = PLAN_RETRY_MS, tweaks: CallTweaks = PLANNING): Promise<T> {
  try {
    return (await client.complete<T>(model, messages, schema, max, undefined, tweaks)).value;
  } catch (e) {
    // a server error or a dropped connection with nothing back: once more after a pause, so one
    // 500 does not stop a run before its first ticket
    if (!(e as { completion?: unknown }).completion && retryMs >= 0 && TRANSIENT.test(String((e as Error)?.message ?? e))) {
      warnings.push(`${what} failed (${why(e)}); asked once more`);
      await Bun.sleep(retryMs);
      return planCall<T>(client, model, messages, schema, max, what, warnings, -1, tweaks);
    }
    const c = (e as { completion?: Completion<string> }).completion;
    const v = c?.raw ? salvageJson(c.raw) : undefined;
    // thought until max_tokens with no complete answer to keep: once more, thinking less
    if ((v === undefined || typeof v !== 'object') && c?.cut && c.thinking !== false && !tweaks.reasoningEffort && thinkingOn(client)) {
      warnings.push(`${what} ran out of tokens (cut at ${c.completionTokens}${c.reasoningChars ? `, ${c.reasoningChars} characters of thinking` : ''}) with no answer to keep; asked once more, thinking low`);
      return planCall<T>(client, model, messages, schema, max, what, warnings, retryMs, { ...PLANNING, thinking: true, reasoningEffort: 'low' });
    }
    if (v === undefined || typeof v !== 'object') throw e;
    warnings.push(`${what} was ${c!.cut ? `cut at ${c!.completionTokens} tokens` : 'not valid JSON'}; its complete part is kept`);
    return v as T;
  }
}

const why = (e: unknown) => String((e as Error)?.message ?? e).split('\n')[0]!.slice(0, 200);

/** A scaffold call's files with their paths made plain; unsafe ones are left out and said. */
function plainFiles(files: Record<string, string> | undefined, who: string, warnings: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [p, text] of Object.entries(files ?? {})) {
    const q = normPath(p);
    if (!q || typeof text !== 'string') warnings.push(`${who} wrote ${JSON.stringify(p)}, not a file path inside the repo; left out`);
    else out[q] = text;
  }
  return out;
}

export async function scaffold(inf: Inference, d: Design, model = 'planner', o: PlanOptions = {}): Promise<Scaffold> {
  const warnings: string[] = [];
  if (!splits(d, o)) {
    const user = architectureMarkdown(d) + crewLine(o.crew, `Write about ${o.crew} implementation stubs, one per agent, each with its own failing test.`);
    const v = await planCall<WorkerOutput>(inf, model, [{ role: 'system', content: SCAFFOLD_SYSTEM }, { role: 'user', content: user }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(32000, (o.crew ?? 0) * TOKENS_PER_MODULE)), 'the scaffold', warnings);
    return { files: plainFiles(v.files, 'the scaffold', warnings), note: v.note ?? '', warnings };
  }
  const interfaces = d.components.flatMap((c) => c.interfaces.map((i) => normPath(i.path) ?? i.path));
  const shared = await planCall<WorkerOutput>(inf, model, [{ role: 'system', content: SCAFFOLD_SHARED_SYSTEM }, { role: 'user', content: architectureMarkdown(d) }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(16000, interfaces.length * 600)), 'the shared scaffold', warnings);
  const files = plainFiles(shared.files, 'the shared scaffold', warnings);
  for (const p of interfaces) if (!(p in files)) warnings.push(`the shared scaffold left out interface ${p}`);
  const written = Object.entries(files).map(([p, text]) => `### ${p}\n\`\`\`\n${text}\n\`\`\``).join('\n\n');
  const shares = crewShares(d, o.crew!);
  // one part that fails (cut while thinking, a server error) costs its component, not the run
  const parts = await Promise.all(d.components.map(async (comp, i) => {
    const user = [
      architectureMarkdown(d),
      `## Files already written\n\n${written}`,
      `## Your component: ${comp.name} (\`${comp.path}\`)\n${comp.responsibility}`,
      crewLine(o.crew, `Your component gets ${shares[i]} of them: write about ${shares[i]} implementation stubs under \`${comp.path}\`, each with its own failing test.`).trim(),
    ].join('\n\n');
    try {
      return await planCall<WorkerOutput>(o.parts!(i), model, [{ role: 'system', content: SCAFFOLD_PART_SYSTEM }, { role: 'user', content: user }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(8000, shares[i]! * TOKENS_PER_MODULE)), `${comp.name}'s scaffold`, warnings);
    } catch (e) {
      warnings.push(`${comp.name}'s scaffold failed (${why(e)}); the component has no stubs and gets no tickets`);
      return undefined;
    }
  }));
  if (parts.every((p) => !p)) throw new Error(`every component's scaffold failed: ${warnings.filter((w) => w.includes('scaffold failed')).join('; ')}`);
  const writtenBy: Record<string, number> = {};
  parts.forEach((c, i) => {
    if (!c) return;
    const comp = d.components[i]!;
    for (const [p, text] of Object.entries(plainFiles(c.files, comp.name, warnings))) {
      if (p in files) {
        warnings.push(p in writtenBy ? `${comp.name} also wrote ${p}; ${d.components[writtenBy[p]!]!.name}'s copy is kept` : `${comp.name} rewrote shared file ${p}; the shared copy is kept`);
        continue;
      }
      files[p] = text;
      writtenBy[p] = i;
    }
  });
  const note = [shared.note, ...parts.map((c, i) => (c ? `${d.components[i]!.name}: ${c.note}` : ''))].filter(Boolean).join(' / ');
  const failed = parts.flatMap((c, i) => (c ? [] : [i]));
  return { files, note, warnings, writtenBy, ...(failed.length ? { failed } : {}) };
}

export interface PlanShape {
  tickets: number;
  /** Tickets with no dependencies: the agents that can start at once. */
  startable: number;
  warnings: string[];
}

/** How well a plan fits a crew of `crew` agents: about one ticket each, most of them free to start at once. */
export function planShape(plan: Plan, crew: number): PlanShape {
  const tickets = plan.tickets.length;
  const startable = plan.tickets.filter((t) => !t.depends_on.length).length;
  const warnings: string[] = [];
  if (tickets < Math.ceil(MIN_CREW_SHARE * crew)) warnings.push(`${tickets} tickets for ${crew} agents: ${crew - tickets} agents have nothing to do`);
  if (startable < Math.ceil(MIN_START_SHARE * Math.min(crew, tickets))) warnings.push(`only ${startable} of ${tickets} tickets can start at once; the rest wait on dependencies`);
  return { tickets, startable, warnings };
}

const dirOf = (c: Design['components'][number]) => (normPath(c.path.replace(/\/+$/, '')) ?? '').replace(/\/?$/, '/');

/** The component a file belongs to: the one whose folder is the longest prefix of its path, or -1. */
export function componentOf(d: Design, path: string): number {
  let best = -1, len = -1;
  d.components.forEach((c, i) => {
    const dir = dirOf(c);
    if (dir === '/') return;
    if ((path.startsWith(dir) || `${path}/` === dir) && dir.length > len) {
      best = i;
      len = dir.length;
    }
  });
  return best;
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 24).replace(/-$/, '') || 'part';

/** A ticket id the plan accepts: lowercase letters, digits and dashes, at most 63 characters, `-<n>` added to repeats. */
function fitId(id: string, n = 1): string {
  let base = String(id ?? '').toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+/, '').replace(/-+$/, '') || 'ticket';
  const tail = n > 1 ? `-${n}` : '';
  const room = 63 - tail.length;
  // too long: keep the start (the component) and the end (what tells sibling tickets apart)
  if (base.length > room) base = `${base.slice(0, Math.floor(room / 2)).replace(/-+$/, '')}-${base.slice(-(room - Math.floor(room / 2) - 1)).replace(/^-+/, '')}`;
  return base + tail;
}

/**
 * Drop what would make a merged plan invalid: unsafe or empty owned paths, ids the plan cannot
 * take or that repeat, unknown or self dependencies, and dependency cycles (the edge that
 * closes each one). Each is said in `warnings`.
 */
function tidy(ts: Ticket[], warnings: string[]): Ticket[] {
  const seen = new Set<string>();
  // an id as the plan wrote it -> the id it has now (the first ticket that used it)
  const renamed = new Map<string, string>();
  const out: Ticket[] = [];
  for (const t of ts) {
    const owns = (Array.isArray(t.owns) ? t.owns : []).map((p) => normPath(p));
    if (owns.some((p) => !p)) warnings.push(`${t.id}: dropped owned paths outside the repo (${t.owns.filter((p) => !normPath(p)).map((p) => JSON.stringify(p)).join(', ')})`);
    const kept = [...new Set(owns.filter((p): p is string => !!p))];
    if (!kept.length && t.role !== 'review' && t.role !== 'architect') {
      warnings.push(`${t.id} owns no file and was left out`);
      continue;
    }
    let id = fitId(t.id);
    for (let k = 2; seen.has(id); k++) id = fitId(t.id, k);
    if (id !== t.id) warnings.push(seen.has(fitId(t.id)) ? `ticket id ${t.id} was used twice; the second is ${id}` : `ticket id ${t.id} is now ${id}`);
    seen.add(id);
    if (!renamed.has(t.id)) renamed.set(t.id, id);
    out.push({ ...t, id, owns: kept, depends_on: Array.isArray(t.depends_on) ? t.depends_on : [] });
  }
  const tidied = out.map((t) => {
    const deps = [...new Set(t.depends_on.map((x) => renamed.get(x) ?? x).filter((x) => x !== t.id && seen.has(x)))];
    const lost = t.depends_on.filter((x) => !seen.has(renamed.get(x) ?? x) || (renamed.get(x) ?? x) === t.id);
    if (lost.length) warnings.push(`${t.id}: dropped dependencies on ${lost.join(', ')}`);
    return { ...t, depends_on: deps };
  });
  return breakCycles(tidied, warnings);
}

/** The plan with every dependency cycle opened at the edge that closes it, each said in `warnings`. */
function breakCycles(ts: Ticket[], warnings: string[]): Ticket[] {
  const byId = new Map(ts.map((t) => [t.id, { ...t, depends_on: [...t.depends_on] }]));
  const state = new Map<string, 1 | 2>();
  const visit = (id: string, path: string[]) => {
    state.set(id, 1);
    const t = byId.get(id)!;
    for (const d of [...t.depends_on]) {
      if (state.get(d) === 1) {
        t.depends_on = t.depends_on.filter((x) => x !== d);
        warnings.push(`dependency cycle ${[...path.slice(path.indexOf(d)), id, d].join(' -> ')}: ${id} no longer waits for ${d}`);
      } else if (!state.has(d)) visit(d, [...path, id]);
    }
    state.set(id, 2);
  };
  for (const t of ts) if (!state.has(t.id)) visit(t.id, []);
  return ts.map((t) => byId.get(t.id)!);
}

/** Files owned by more than one ticket that cannot run at once, which settling turns into a queue. */
function crowdedFiles(ts: Ticket[]): Array<[string, string[]]> {
  const owners = new Map<string, string[]>();
  for (const t of ts) for (const p of t.owns ?? []) owners.set(p, [...(owners.get(p) ?? []), t.id]);
  return [...owners].filter(([, ids]) => ids.length > 1).sort((a, b) => b[1].length - a[1].length);
}

/** Tickets of a (part's) plan that can start at once once its clashes are settled. */
function startableAfterSettle(ts: Ticket[]): number {
  try {
    const w: string[] = [];
    return settleClashes(tidy(ts, w)).tickets.filter((t) => !t.depends_on.length).length;
  } catch {
    return 0;
  }
}

export interface PlannedTickets extends Plan {
  /** planShape's warnings, and what merging the parts dropped. */
  warnings?: string[];
}

/**
 * The tickets for a scaffold: `files` is its file list, or the Scaffold itself, whose `writtenBy`
 * lets each component's tickets own the files its part wrote even outside the component's folder.
 */
export async function tickets(inf: Inference, d: Design, files: string[] | Scaffold, model = 'planner', o: PlanOptions = {}): Promise<PlannedTickets> {
  const crew = o.crew;
  const writtenBy = Array.isArray(files) ? {} : (files.writtenBy ?? {});
  const list = Array.isArray(files) ? files : Object.keys(files.files);
  const warnings: string[] = [];
  const ask = async (client: Inference, system: string, user: string, want: number | undefined, answer: number, what: string): Promise<Plan> => {
    const messages: ChatMessage[] = [{ role: 'system', content: system }, { role: 'user', content: user }];
    const first = await planCall<Plan>(client, model, messages, PLAN_JSON_SCHEMA, budget(inf, answer), what, warnings);
    const ts = Array.isArray(first?.tickets) ? first.tickets : [];
    // too few tickets for the agents, or most of them queued behind a file they all own: ask once more, saying so
    const few = !!want && ts.length < Math.ceil(MIN_CREW_SHARE * want);
    const start = startableAfterSettle(ts);
    const queued = ts.length > 1 && start < Math.ceil(MIN_START_SHARE * ts.length) ? crowdedFiles(ts).slice(0, 3) : [];
    if (!few && !queued.length) return { interfaces: first?.interfaces ?? [], tickets: ts };
    const asks = [
      ...(few ? [`That is ${ts.length} tickets for ${want} agents. Split the work into about ${want} smaller tickets, most with no dependencies.`] : []),
      ...(queued.length ? [`Tickets that own the same file run one after another, so only ${start} of ${ts.length} can start at once: ${queued.map(([p, ids]) => `${p} is owned by ${ids.length} tickets`).join('; ')}. Give each file to exactly one ticket; the others read it.`] : []),
    ];
    const again: ChatMessage[] = [...messages, { role: 'assistant', content: JSON.stringify(first) }, { role: 'user', content: `${asks.join(' ')} Reply with the whole plan again.` }];
    let second: Plan;
    try {
      second = await planCall<Plan>(client, model, again, PLAN_JSON_SCHEMA, budget(inf, answer), `${what} (asked again)`, warnings);
    } catch (e) {
      warnings.push(`${what} asked again failed (${why(e)}); the first answer is kept`);
      return { interfaces: first?.interfaces ?? [], tickets: ts };
    }
    const ts2 = Array.isArray(second?.tickets) ? second.tickets : [];
    // the answer that lets more agents start at once, then the one with more tickets
    const better = startableAfterSettle(ts2) > start || (startableAfterSettle(ts2) === start && ts2.length > ts.length);
    return better ? { interfaces: second.interfaces ?? [], tickets: ts2 } : { interfaces: first?.interfaces ?? [], tickets: ts };
  };
  let raw: Plan;
  if (!splits(d, o)) {
    const user = `${architectureMarkdown(d)}\n\n## Scaffold files\n${list.map((f) => `- ${f}`).join('\n')}` + crewLine(crew, `Aim for about ${crew} tickets, one per agent, most with no dependencies so they all start at once.`);
    raw = await ask(inf, TICKETS_SYSTEM, user, crew, Math.max(16000, (crew ?? 0) * TOKENS_PER_TICKET), 'the ticket plan');
    raw = { ...raw, tickets: tidy(raw.tickets, warnings) };
  } else {
    const shares = crewShares(d, crew!);
    // whose a file is: the part that wrote it, else the component whose folder holds it (and any
    // other component with that same folder)
    const owner = (f: string) => writtenBy[f] ?? componentOf(d, f);
    const mayOwn = (i: number, f: string) => {
      const c = owner(f);
      return c === i || (c >= 0 && dirOf(d.components[c]!) === dirOf(d.components[i]!)) || (c === -1 && !list.includes(f));
    };
    const mine = d.components.map((_, i) => list.filter((f) => mayOwn(i, f)));
    const common = list.filter((f) => owner(f) === -1);
    // ticket id prefixes, one per component even when two have the same name
    const prefixes: string[] = [];
    for (const comp of d.components) {
      let p = slug(comp.name);
      for (let k = 2; prefixes.includes(p); k++) p = `${slug(comp.name)}${k}`;
      prefixes.push(p);
    }
    const parts = await Promise.all(d.components.map(async (comp, i) => {
      const prefix = prefixes[i]!;
      // its scaffold part failed (said then): no stubs or tests to fill in
      if (!Array.isArray(files) && files.failed?.includes(i)) return undefined;
      const user = [
        architectureMarkdown(d),
        `## Shared scaffold files (read them, own none)\n${common.map((f) => `- ${f}`).join('\n') || '- none'}`,
        `## Your component: ${comp.name} (\`${comp.path}\`)\n${comp.responsibility}\n\nIts files:\n${mine[i]!.map((f) => `- ${f}`).join('\n') || '- none yet'}`,
        crewLine(crew, `Your component gets ${shares[i]} of them: aim for about ${shares[i]} tickets, one per agent, most with no dependencies. Start every ticket id with "${prefix}-".`).trim(),
      ].join('\n\n');
      try {
        return await ask(o.parts!(i), TICKETS_PART_SYSTEM, user, shares[i], Math.max(4000, shares[i]! * TOKENS_PER_TICKET), `${comp.name}'s tickets`);
      } catch (e) {
        warnings.push(`${comp.name}'s tickets failed (${why(e)}); the component gets no tickets`);
        return undefined;
      }
    }));
    if (parts.every((p) => !p)) throw new Error(`no component's ticket plan came back: ${warnings.filter((w) => w.includes('tickets failed')).join('; ')}`);
    const merged: Ticket[] = [];
    parts.forEach((p, i) => {
      if (!p) return;
      const comp = d.components[i]!;
      const prefix = prefixes[i]!;
      // ids made unique per component, dependencies kept within it
      const ids = new Map(p.tickets.map((t) => [t.id, fitId(String(t.id ?? '').startsWith(`${prefix}-`) ? t.id : `${prefix}-${t.id}`)]));
      for (const t of p.tickets) {
        const paths = (Array.isArray(t.owns) ? t.owns : []).map((f) => normPath(f) ?? f);
        const owns = paths.filter((f) => mayOwn(i, f));
        if (owns.length !== paths.length) warnings.push(`${ids.get(t.id)}: dropped owned files of other components or shared files (${paths.filter((f) => !owns.includes(f)).join(', ')})`);
        if (!owns.length) {
          warnings.push(`${ids.get(t.id)} owned nothing in ${comp.name} and was left out`);
          continue;
        }
        merged.push({ ...t, id: ids.get(t.id)!, owns, depends_on: (Array.isArray(t.depends_on) ? t.depends_on : []).filter((x) => ids.has(x)).map((x) => ids.get(x)!) });
      }
    });
    const left = new Set(merged.map((t) => t.id));
    raw = { interfaces: [...new Set(parts.flatMap((p) => p?.interfaces ?? []))], tickets: tidy(merged.map((t) => ({ ...t, depends_on: t.depends_on.filter((x) => left.has(x)) })), warnings) };
  }
  // the budget is the engine's, not the model's: a guessed max_tokens of a few thousand would cut
  // every thinking call of the ticket and fail it on its first answer
  const guessed = raw.tickets.filter((t) => t.budget && (t.budget.attempts !== PLANNED_BUDGET.attempts || t.budget.max_tokens !== PLANNED_BUDGET.max_tokens));
  if (guessed.length) warnings.push(`${guessed.length} tickets came with a budget of their own (smallest ${Math.min(...guessed.map((t) => Number(t.budget.max_tokens) || 0))} tokens); every ticket gets ${PLANNED_BUDGET.attempts} attempts and ${PLANNED_BUDGET.max_tokens} tokens`);
  raw = { ...raw, tickets: raw.tickets.map((t) => ({ ...t, budget: { ...PLANNED_BUDGET } })) };
  const plan = validatePlan(raw);
  for (const [p, ids] of crowdedFiles(plan.tickets).slice(0, 3)) if (ids.length > 2) warnings.push(`${p} is owned by ${ids.length} tickets, so they run one after another`);
  if (crew) warnings.push(...planShape(plan, crew).warnings);
  return warnings.length ? { ...plan, warnings } : plan;
}
