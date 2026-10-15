/**
 * The engine's own worker calls, rendered on request, for training data. Every training row
 * the Centipede train harness writes (llmap scripts/bh/) comes from here, so a row is the
 * exact message the runtime harness sends: prompts.ts builds the pack and the write or fix
 * message, files.ts parses and renders the answer. Nothing here copies the wire format; if
 * prompts.ts or files.ts change, the rows change with them.
 *
 *   bun bench/render.ts < requests.jsonl > replies.jsonl      (also: mugge render-call)
 *
 * One JSON request per line, one reply per line in the same order. Every request may carry
 * an `id`, echoed back. A request that fails gets {id, error} and the rest go on.
 *
 *   {op: "tickets", bench}                      every ticket under a bench dir (bench/load.ts):
 *       → {tickets: [{key, project, dir, id, owns, acceptance, hidden, image, deps, split, canary,
 *                     theme, language, role}]}
 *   {op: "prepare", project, ticket, dest, own?, hidden?}
 *       the worktree the bench starts the ticket in (scaffold + its dependencies' reference
 *       answers, bench/run.ts runOne); own: also its own reference; hidden: also its hidden files
 *       → {dest}
 *   {op: "write", project, ticket, worktree?, current?, memory?, answer?}
 *   {op: "fix",   project, ticket, worktree?, current?, memory?, failed: {command, output, timedOut?, raw?}, answer?}
 *       the messages the engine sends for this ticket (harness.ts runTicket). Without
 *       worktree the ticket's start worktree is prepared in a temp dir and `current` (the owned
 *       files as the failing attempt left them) is written over it; with worktree that dir is
 *       read as it is (the checks' leftovers included, as the engine sees them). failed.raw:
 *       output is a check's whole output, trimmed here the way harness.ts trims it; otherwise
 *       it is used as given (already trimmed, e.g. "(write files)").
 *       answer (the model's raw text) adds the canonical assistant message: the files parsed
 *       with the ticket's owns (parseFiles) and rendered back (renderFiles) in owns order.
 *       → {messages, parsed?: {files, note, missing, extra}}
 *   {op: "hidden", project, ticket, dest}       copies the ticket's hidden files over dest (after
 *       its acceptance ran, as bench/run.ts does) → {dest}
 *   {op: "canonical", answer, owns}             → {content, parsed}   (as `answer` on write/fix)
 *   {op: "parse", answer, owns}                 → {files, note}
 *   {op: "answer", files, note?}                → {content}   (renderFiles)
 *
 * project is a bench project dir (plan.json, project/, solutions/, hidden/).
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseFiles, renderFiles } from '../src/engine/files.ts';
import type { ChatMessage } from '../src/engine/inference.ts';
import { buildPack, fixMessages, writeMessages } from '../src/engine/prompts.ts';
import { tail } from '../src/engine/sandbox.ts';
import { applyHidden, findProjects, loadProject, prepare, writeFiles, type BenchProject, type BenchTicket } from './load.ts';

export interface Failed {
  command: string;
  output: string;
  timedOut?: boolean;
  /** output is the check's whole output; trim it the way harness.ts does. */
  raw?: boolean;
}

/** The failing command as harness.ts hands it to fixMessages after a failed acceptance check. */
export function failedFromCheck(command: string, out: string, timedOut = false): { command: string; output: string } {
  // sandbox.ts gives (stdout + stderr).trim(); harness.ts adds the timeout mark and keeps the tail
  return { command, output: (timedOut ? '(timed out)\n' : '') + tail(out.trim()) };
}

export interface Parsed {
  files: Record<string, string>;
  note: string;
  /** Owned files the answer did not write. */
  missing: string[];
  /** Files the answer wrote that the ticket does not own. */
  extra: string[];
}

/** The answer as training wants it: parsed as the engine parses it, rendered back in owns order. */
export function canonicalAnswer(raw: string, owns: string[]): { content: string; parsed: Parsed } {
  const out = parseFiles(raw, owns);
  const files: Record<string, string> = {};
  for (const p of owns) if (p in out.files) files[p] = out.files[p]!;
  const parsed: Parsed = {
    files,
    note: String(out.note ?? ''),
    missing: owns.filter((p) => !(p in out.files)),
    extra: Object.keys(out.files).filter((p) => !owns.includes(p)),
  };
  return { content: renderFiles(files, parsed.note), parsed };
}

const projects = new Map<string, BenchProject>();

function project(dir: string): BenchProject {
  const abs = resolve(dir);
  let p = projects.get(abs);
  if (!p) projects.set(abs, (p = loadProject(abs)));
  return p;
}

function ticket(p: BenchProject, id: string): BenchTicket {
  const t = p.tickets.find((x) => x.ticket.id === id);
  if (!t) throw new Error(`no ticket ${id} in ${p.dir}`);
  return t;
}

export interface CallRequest {
  project: string;
  ticket: string;
  worktree?: string;
  current?: Record<string, string>;
  memory?: string;
  failed?: Failed;
  answer?: string;
}

/** The engine's messages for one write (no failed) or fix call, plus the canonical answer when given. */
export function renderCall(r: CallRequest): { messages: ChatMessage[]; parsed?: Parsed } {
  const p = project(r.project);
  const t = ticket(p, r.ticket);
  let dir = r.worktree, tmp = '';
  if (!dir) {
    dir = tmp = mkdtempSync(join(tmpdir(), 'mugge-render-'));
    prepare(p, dir, t.deps);
    if (r.current) writeFiles(dir, r.current);
  }
  try {
    const pack = buildPack(dir, p.plan, t.ticket, r.memory ?? '');
    const failed = r.failed && (r.failed.raw ? failedFromCheck(r.failed.command, r.failed.output, r.failed.timedOut) : { command: r.failed.command, output: r.failed.output });
    const messages = failed ? fixMessages(dir, pack, failed) : writeMessages(dir, pack);
    if (r.answer === undefined) return { messages };
    const { content, parsed } = canonicalAnswer(r.answer, t.ticket.owns);
    return { messages: [...messages, { role: 'assistant', content }], parsed };
  } finally {
    if (tmp) rmSync(tmp, { recursive: true, force: true });
  }
}

export function listTickets(bench: string) {
  return findProjects(resolve(bench)).flatMap((p) => p.tickets.map((t) => ({
    key: t.key, project: p.name, dir: p.dir, id: t.ticket.id, owns: t.ticket.owns, acceptance: t.ticket.acceptance,
    hidden: t.tags.hidden, image: t.ticket.image, deps: t.deps, split: p.split, canary: p.canary,
    theme: t.tags.theme, language: t.tags.language, role: t.tags.role,
  })));
}

export function handle(req: any): Record<string, unknown> {
  switch (req.op) {
    case 'tickets':
      return { tickets: listTickets(req.bench) };
    case 'prepare': {
      const p = project(req.project);
      const t = ticket(p, req.ticket);
      prepare(p, req.dest, req.own ? [...t.deps, t.ticket.id] : t.deps);
      if (req.hidden) applyHidden(p, t.ticket.id, req.dest);
      return { dest: req.dest };
    }
    case 'hidden': {
      applyHidden(project(req.project), req.ticket, req.dest);
      return { dest: req.dest };
    }
    case 'canonical':
      return canonicalAnswer(String(req.answer ?? ''), req.owns ?? []) as unknown as Record<string, unknown>;
    case 'write':
      if (req.failed) throw new Error('a write call has no failed command; use op "fix"');
      return renderCall(req) as unknown as Record<string, unknown>;
    case 'fix':
      if (!req.failed?.command) throw new Error('a fix call needs failed: {command, output}');
      return renderCall(req) as unknown as Record<string, unknown>;
    case 'parse': {
      const out = parseFiles(String(req.answer ?? ''), req.owns ?? []);
      return { files: out.files, note: out.note };
    }
    case 'answer':
      return { content: renderFiles(req.files ?? {}, req.note ?? '') };
    default:
      throw new Error(`unknown op ${JSON.stringify(req.op)}`);
  }
}

if (import.meta.main) {
  const text = await new Response(Bun.stdin.stream()).text();
  const out: string[] = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    let id: unknown;
    try {
      const req = JSON.parse(line);
      id = req.id;
      out.push(JSON.stringify({ id, ...handle(req) }));
    } catch (e) {
      out.push(JSON.stringify({ id, error: String((e as Error).message ?? e) }));
    }
  }
  await Bun.write(Bun.stdout, out.length ? out.join('\n') + '\n' : '');
}
