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
import type { CallTweaks, Inference } from '../engine/inference.ts';
import { IMAGES, PLAN_JSON_SCHEMA, validatePlan, WORKER_OUTPUT_JSON_SCHEMA, type Plan, type Ticket, type WorkerOutput } from '../tickets/schema.ts';

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
/** A plan with fewer tickets than this share of the crew is asked for again, once. */
export const MIN_CREW_SHARE = 0.75;

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

export async function design(inf: Inference, request: string, answers: string[] = [], model = 'architect', o: PlanOptions = {}): Promise<Design> {
  const asked = answers.length ? `${request}\n\nAnswers to your questions:\n${answers.join('\n')}` : request;
  const user = asked + crewLine(o.crew, `Size the design for them: about ${o.crew} implementation files of similar size, one per agent, spread over components that each own a folder.`);
  const c = await inf.complete<Design>(model, [{ role: 'system', content: ARCHITECT_SYSTEM }, { role: 'user', content: user }], DESIGN_JSON_SCHEMA, budget(inf, Math.max(8000, (o.crew ?? 0) * 150)), undefined, PLANNING);
  return c.value;
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
  /** What the split left out or settled (a file two components both wrote, a file outside its component). */
  warnings: string[];
}

export async function scaffold(inf: Inference, d: Design, model = 'planner', o: PlanOptions = {}): Promise<Scaffold> {
  if (!splits(d, o)) {
    const user = architectureMarkdown(d) + crewLine(o.crew, `Write about ${o.crew} implementation stubs, one per agent, each with its own failing test.`);
    const c = await inf.complete<WorkerOutput>(model, [{ role: 'system', content: SCAFFOLD_SYSTEM }, { role: 'user', content: user }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(32000, (o.crew ?? 0) * TOKENS_PER_MODULE)), undefined, PLANNING);
    return { ...c.value, warnings: [] };
  }
  const interfaces = d.components.flatMap((c) => c.interfaces.map((i) => i.path));
  const shared = await inf.complete<WorkerOutput>(model, [{ role: 'system', content: SCAFFOLD_SHARED_SYSTEM }, { role: 'user', content: architectureMarkdown(d) }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(16000, interfaces.length * 600)), undefined, PLANNING);
  const files: Record<string, string> = { ...shared.value.files };
  const warnings: string[] = [];
  for (const p of interfaces) if (!(p in files)) warnings.push(`the shared scaffold left out interface ${p}`);
  const written = Object.entries(files).map(([p, text]) => `### ${p}\n\`\`\`\n${text}\n\`\`\``).join('\n\n');
  const shares = crewShares(d, o.crew!);
  const parts = await Promise.all(d.components.map((comp, i) => {
    const user = [
      architectureMarkdown(d),
      `## Files already written\n\n${written}`,
      `## Your component: ${comp.name} (\`${comp.path}\`)\n${comp.responsibility}`,
      crewLine(o.crew, `Your component gets ${shares[i]} of them: write about ${shares[i]} implementation stubs under \`${comp.path}\`, each with its own failing test.`).trim(),
    ].join('\n\n');
    return o.parts!(i).complete<WorkerOutput>(model, [{ role: 'system', content: SCAFFOLD_PART_SYSTEM }, { role: 'user', content: user }], WORKER_OUTPUT_JSON_SCHEMA, budget(inf, Math.max(8000, shares[i]! * TOKENS_PER_MODULE)), undefined, PLANNING);
  }));
  const from = new Map<string, string>();
  parts.forEach((c, i) => {
    const comp = d.components[i]!;
    for (const [p, text] of Object.entries(c.value.files)) {
      if (p in files) {
        warnings.push(from.has(p) ? `${comp.name} also wrote ${p}; ${from.get(p)}'s copy is kept` : `${comp.name} rewrote shared file ${p}; the shared copy is kept`);
        continue;
      }
      files[p] = text;
      from.set(p, comp.name);
    }
  });
  const note = [shared.value.note, ...parts.map((c, i) => `${d.components[i]!.name}: ${c.value.note}`)].filter(Boolean).join(' / ');
  return { files, note, warnings };
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
  if (startable < Math.ceil(0.5 * Math.min(crew, tickets))) warnings.push(`only ${startable} of ${tickets} tickets can start at once; the rest wait on dependencies`);
  return { tickets, startable, warnings };
}

/** The component a file belongs to: the one whose folder is the longest prefix of its path, or -1. */
export function componentOf(d: Design, path: string): number {
  let best = -1, len = -1;
  d.components.forEach((c, i) => {
    const dir = c.path.replace(/^\.\//, '').replace(/\/?$/, '/');
    if ((path.startsWith(dir) || path === c.path) && dir.length > len) {
      best = i;
      len = dir.length;
    }
  });
  return best;
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'part';

/** Drop what would make a merged plan invalid: unknown or self dependencies, repeated ids. */
function tidy(ts: Ticket[], warnings: string[]): Ticket[] {
  const seen = new Set<string>();
  const out: Ticket[] = [];
  for (const t of ts) {
    let id = t.id;
    for (let k = 2; seen.has(id); k++) id = `${t.id}-${k}`;
    if (id !== t.id) warnings.push(`ticket id ${t.id} was used twice; the second is ${id}`);
    seen.add(id);
    out.push({ ...t, id });
  }
  return out.map((t) => {
    const deps = (t.depends_on ?? []).filter((x) => x !== t.id && seen.has(x));
    if (deps.length !== (t.depends_on ?? []).length) warnings.push(`${t.id}: dropped dependencies on ${(t.depends_on ?? []).filter((x) => !deps.includes(x)).join(', ')}`);
    return { ...t, depends_on: deps };
  });
}

export interface PlannedTickets extends Plan {
  /** planShape's warnings, and what merging the parts dropped. */
  warnings?: string[];
}

export async function tickets(inf: Inference, d: Design, files: string[], model = 'planner', o: PlanOptions = {}): Promise<PlannedTickets> {
  const crew = o.crew;
  const ask = async (client: Inference, system: string, user: string, want: number | undefined, answer: number): Promise<Plan> => {
    const messages = [{ role: 'system' as const, content: system }, { role: 'user' as const, content: user }];
    const c = await client.complete<Plan>(model, messages, PLAN_JSON_SCHEMA, budget(inf, answer), undefined, PLANNING);
    // too few tickets for the agents it is meant for: ask once more, saying so
    if (!want || c.value.tickets.length >= Math.ceil(MIN_CREW_SHARE * want)) return c.value;
    const again = [...messages, { role: 'assistant' as const, content: JSON.stringify(c.value) }, { role: 'user' as const, content: `That is ${c.value.tickets.length} tickets for ${want} agents. Split the work into about ${want} smaller tickets, most with no dependencies, and reply with the whole plan again.` }];
    const c2 = await client.complete<Plan>(model, again, PLAN_JSON_SCHEMA, budget(inf, answer), undefined, PLANNING);
    return c2.value.tickets.length > c.value.tickets.length ? c2.value : c.value;
  };
  const warnings: string[] = [];
  let raw: Plan;
  if (!splits(d, o)) {
    const user = `${architectureMarkdown(d)}\n\n## Scaffold files\n${files.map((f) => `- ${f}`).join('\n')}` + crewLine(crew, `Aim for about ${crew} tickets, one per agent, most with no dependencies so they all start at once.`);
    raw = await ask(inf, TICKETS_SYSTEM, user, crew, Math.max(16000, (crew ?? 0) * TOKENS_PER_TICKET));
    raw = { ...raw, tickets: tidy(raw.tickets, warnings) };
  } else {
    const shares = crewShares(d, crew!);
    const mine = d.components.map((_, i) => files.filter((f) => componentOf(d, f) === i));
    const common = files.filter((f) => componentOf(d, f) === -1);
    const parts = await Promise.all(d.components.map((comp, i) => {
      const prefix = slug(comp.name);
      const user = [
        architectureMarkdown(d),
        `## Shared scaffold files (read them, own none)\n${common.map((f) => `- ${f}`).join('\n') || '- none'}`,
        `## Your component: ${comp.name} (\`${comp.path}\`)\n${comp.responsibility}\n\nIts files:\n${mine[i]!.map((f) => `- ${f}`).join('\n') || '- none yet'}`,
        crewLine(crew, `Your component gets ${shares[i]} of them: aim for about ${shares[i]} tickets, one per agent, most with no dependencies. Start every ticket id with "${prefix}-".`).trim(),
      ].join('\n\n');
      return ask(o.parts!(i), TICKETS_PART_SYSTEM, user, shares[i], Math.max(4000, shares[i]! * TOKENS_PER_TICKET));
    }));
    const merged: Ticket[] = [];
    parts.forEach((p, i) => {
      const comp = d.components[i]!;
      const prefix = slug(comp.name);
      // ids made unique per component, dependencies kept within it
      const ids = new Map(p.tickets.map((t) => [t.id, t.id.startsWith(`${prefix}-`) ? t.id : `${prefix}-${t.id}`]));
      for (const t of p.tickets) {
        const owns = t.owns.filter((f) => { const c = componentOf(d, f); return c === i || c === -1 && !files.includes(f); });
        if (owns.length !== t.owns.length) warnings.push(`${ids.get(t.id)}: dropped owned files of other components or shared files (${t.owns.filter((f) => !owns.includes(f)).join(', ')})`);
        if (!owns.length) {
          warnings.push(`${ids.get(t.id)} owned nothing in ${comp.name} and was left out`);
          continue;
        }
        merged.push({ ...t, id: ids.get(t.id)!, owns, depends_on: (t.depends_on ?? []).filter((x) => ids.has(x)).map((x) => ids.get(x)!) });
      }
    });
    const left = new Set(merged.map((t) => t.id));
    raw = { interfaces: [...new Set(parts.flatMap((p) => p.interfaces ?? []))], tickets: tidy(merged.map((t) => ({ ...t, depends_on: t.depends_on.filter((x) => left.has(x)) })), warnings) };
  }
  const plan = validatePlan(raw);
  if (crew) warnings.push(...planShape(plan, crew).warnings);
  return warnings.length ? { ...plan, warnings } : plan;
}
