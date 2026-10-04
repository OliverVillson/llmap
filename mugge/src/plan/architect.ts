/**
 * Architect and planner: the two calls that turn a request into work.
 *
 *   architect: request → design (components, stacks, interfaces, decisions) → ARCHITECTURE.md
 *   planner, step 1 (scaffold): design → real files: folder tree, interfaces, stubs, one failing
 *     test per feature, committed as the base every ticket starts from
 *   planner, step 2 (tickets): design + scaffold file list → a validated Plan
 *
 * All three are single constrained-JSON calls, so every answer parses. They run on the bigger
 * model (`architect` / `planner` names in the inference config): few calls, high leverage.
 */
import type { Inference } from '../engine/inference.ts';
import { IMAGES, PLAN_JSON_SCHEMA, validatePlan, WORKER_OUTPUT_JSON_SCHEMA, type Plan, type WorkerOutput } from '../tickets/schema.ts';

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

const TICKETS_SYSTEM = [
  'You split a scaffolded project into tickets for parallel coding agents.',
  'Each ticket owns the implementation files it fills in; tickets that can run at the same time never own the same file.',
  'Each ticket lists acceptance commands (its tests, plus build/type checks) that pass only when it is done, and the toolchain image they run in.',
  'Prefer many small tickets with few dependencies, so most can run at once. Reply with {"interfaces": [...], "tickets": [...]}.',
].join('\n');

export async function design(inf: Inference, request: string, answers: string[] = [], model = 'architect'): Promise<Design> {
  const user = answers.length ? `${request}\n\nAnswers to your questions:\n${answers.join('\n')}` : request;
  const c = await inf.complete<Design>(model, [{ role: 'system', content: ARCHITECT_SYSTEM }, { role: 'user', content: user }], DESIGN_JSON_SCHEMA, 8000);
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

export async function scaffold(inf: Inference, d: Design, model = 'planner'): Promise<WorkerOutput> {
  const c = await inf.complete<WorkerOutput>(model, [{ role: 'system', content: SCAFFOLD_SYSTEM }, { role: 'user', content: architectureMarkdown(d) }], WORKER_OUTPUT_JSON_SCHEMA, 32000);
  return c.value;
}

export async function tickets(inf: Inference, d: Design, files: string[], model = 'planner'): Promise<Plan> {
  const user = `${architectureMarkdown(d)}\n\n## Scaffold files\n${files.map((f) => `- ${f}`).join('\n')}`;
  const c = await inf.complete<Plan>(model, [{ role: 'system', content: TICKETS_SYSTEM }, { role: 'user', content: user }], PLAN_JSON_SCHEMA, 16000);
  return validatePlan(c.value);
}
