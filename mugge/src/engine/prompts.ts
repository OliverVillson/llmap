/**
 * Harness-shaped prompts: one input, one output. A write call gets the context pack; a fix call
 * gets the same pack plus the failing command's trimmed output. Both answer with the owned files
 * as plain fenced files (files.ts). Tuned for Qwen3.8-27B (2026-10-18): the system prompt shows
 * the exact answer shape once, and says no text around the files and no elided code. llmap's
 * stages/harness.py still builds the earlier BlackHole bytes; BlackHole is parked, so they are
 * no longer kept in sync (test/wire-contract.test.ts pins these bytes instead).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { Plan, Ticket } from '../tickets/schema.ts';
import type { ChatMessage } from './inference.ts';

export const SYSTEM = [
  'You are a coding agent working on one ticket of a larger project; other agents work on the other tickets at the same time.',
  'Reply with every file you own, each in full, in exactly this format:',
  '',
  'src/example.py',
  '```python',
  '<the whole file>',
  '```',
  '',
  'The path goes alone on the line before its fence, exactly as listed under "Files you own". Write each file completely: never leave code out with "..." or "rest unchanged", and never write a diff.',
  'After the files you may add one line "Note: <what you changed>". Write nothing else before, between or after the files.',
  'Write only the files the ticket owns. Do not change interfaces you only read.',
  'No comments that mention AI, tickets or this process; write code the way the repo already does.',
].join('\n');

/** Most context a single file may take in the pack, so one huge file cannot crowd out the rest. */
const MAX_FILE_CHARS = 20_000;

function globToRegExp(glob: string): RegExp {
  const esc = glob.replace(/[.+^${}()|\\]/g, '\\$&').replace(/\*\*/g, '\u0000').replace(/\*/g, '[^/]*').replace(/\?/g, '[^/]');
  return new RegExp(`^${esc.replace(/\u0000/g, '.*')}$`);
}

function walk(root: string, dir = root, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === '.git' || name === 'node_modules' || name === 'build' || name === '__pycache__') continue;
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(root, p, out);
    else out.push(relative(root, p));
  }
  return out;
}

/** Expands `reads` (paths or globs) against the worktree, in a stable order. */
export function expandReads(worktree: string, patterns: string[]): string[] {
  const all = patterns.some((p) => /[*?]/.test(p)) ? walk(worktree).sort() : [];
  const out: string[] = [];
  for (const p of patterns) {
    if (/[*?]/.test(p)) {
      const re = globToRegExp(p);
      for (const f of all) if (re.test(f) && !out.includes(f)) out.push(f);
    } else if (!out.includes(p) && existsSync(join(worktree, p))) out.push(p);
  }
  return out;
}

function fileBlock(worktree: string, path: string): string {
  let text = existsSync(join(worktree, path)) ? readFileSync(join(worktree, path), 'utf8') : '';
  if (text.length > MAX_FILE_CHARS) text = text.slice(0, MAX_FILE_CHARS) + '\n…(truncated)';
  return `--- ${path}\n${text}`;
}

export interface ContextPack {
  ticket: Ticket;
  owned: string[];
  reads: string[];
  interfaces: string[];
  memory: string;
}

/** Everything the model sees about the ticket, read from its worktree. */
export function buildPack(worktree: string, plan: Plan, ticket: Ticket, memory = ''): ContextPack {
  const interfaces = plan.interfaces.filter((p) => !ticket.owns.includes(p));
  const reads = expandReads(worktree, ticket.reads).filter((p) => !ticket.owns.includes(p) && !interfaces.includes(p));
  return { ticket, owned: ticket.owns, reads, interfaces, memory };
}

function packText(worktree: string, pack: ContextPack): string {
  const t = pack.ticket;
  return [
    `Ticket: ${t.id}`,
    `Title: ${t.title}`,
    `Role: ${t.role}`,
    '',
    t.context,
    '',
    `Files you own (write each in full): ${t.owns.join(', ')}`,
    `Done when these commands pass: ${t.acceptance.join(' ; ')}`,
    pack.memory ? `\nProject notes:\n${pack.memory}` : '',
    '',
    '## Interfaces',
    ...pack.interfaces.map((p) => fileBlock(worktree, p)),
    '',
    '## Files to read',
    ...pack.reads.map((p) => fileBlock(worktree, p)),
    '',
    '## Your files as they are now',
    ...pack.owned.map((p) => fileBlock(worktree, p)),
  ].join('\n');
}

export function writeMessages(worktree: string, pack: ContextPack): ChatMessage[] {
  return [
    { role: 'system', content: SYSTEM },
    { role: 'user', content: packText(worktree, pack) },
  ];
}

export function fixMessages(worktree: string, pack: ContextPack, failed: { command: string; output: string }): ChatMessage[] {
  return [
    { role: 'system', content: SYSTEM },
    {
      role: 'user',
      content: `${packText(worktree, pack)}\n\n## Fix\nThis command failed:\n$ ${failed.command}\n${failed.output}\n\nReply with your files, fixed so it passes, in the same format.`,
    },
  ];
}

/**
 * The lead review (review.ts): a new call type, on unless MUGGE_REVIEW=off. One fresh message:
 * the ticket, the diff of its owned files since the ticket started, and the output of the
 * acceptance commands that passed. The answer is constrained to REVIEW_JSON_SCHEMA: approve or
 * send back, plus one reason, the shape the console designs show.
 */
export const REVIEW_SYSTEM = [
  'You are the lead reviewing one finished ticket of a larger project. Its acceptance commands already pass.',
  'Read the ticket, the diff of the files it owns and the check output. Look for what the checks miss: parts of the ticket not done, stubs left in, interfaces changed or misused, wrong edge cases.',
  'Reply with one JSON object: {"verdict": "approve" or "send_back", "reason": "..."}. For send_back the reason says in one or two sentences what is wrong and what the coder should change; for approve it may be empty.',
  'Approve unless a problem makes the ticket wrong or incomplete; style alone is no reason to send it back.',
].join('\n');

export const REVIEW_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'reason'],
  properties: {
    verdict: { enum: ['approve', 'send_back'] },
    reason: { type: 'string' },
  },
} as const;

export function reviewMessages(ticket: Ticket, diff: string, checks: Array<{ command: string; output: string }>): ChatMessage[] {
  return [
    { role: 'system', content: REVIEW_SYSTEM },
    {
      role: 'user',
      content: [
        `Ticket: ${ticket.id}`,
        `Title: ${ticket.title}`,
        `Role: ${ticket.role}`,
        '',
        ticket.context,
        '',
        `Files it owns: ${ticket.owns.join(', ')}`,
        `Done when these commands pass: ${ticket.acceptance.join(' ; ')}`,
        '',
        '## Diff',
        diff || '(no changes)',
        '',
        '## Checks (all passed)',
        ...checks.map((c) => `$ ${c.command}\n${c.output}`),
      ].join('\n'),
    },
  ];
}
