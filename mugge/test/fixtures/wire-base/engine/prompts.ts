/**
 * Harness-shaped prompts: one input, one output, the way llmap's specialists are trained
 * (llmap's stages/harness.py builds the same prompts for training; keep them in sync).
 * A write call gets the context pack; a fix call gets the same pack plus the failing command's
 * trimmed output. Both answer with the owned files as plain fenced files (files.ts).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { Plan, Ticket } from '../tickets/schema.ts';
import type { ChatMessage } from './inference.ts';

export const SYSTEM = [
  'You are a coding specialist working on one ticket of a larger project.',
  'Reply with each file you own in full: its path on one line, then its content in one fenced code block. You may end with one line "Note: <what you changed>".',
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
      content: `${packText(worktree, pack)}\n\n## Fix\nThis command failed:\n$ ${failed.command}\n${failed.output}\n\nReturn your files fixed so it passes.`,
    },
  ];
}
