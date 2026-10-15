/**
 * The train harness's rows must be the engine's own calls, byte for byte: for every spike
 * ticket, run the engine's loop (runTicket) with a model that first answers wrong and then
 * with the reference, record what the engine sent, and check bench/render.ts renders the
 * same write and fix messages from the same inputs (spike tickets are smoke data only).
 */
import { afterAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findProjects, prepare, solution } from '../bench/load.ts';
import { canonicalAnswer, handle, renderCall } from '../bench/render.ts';
import { renderFiles } from '../src/engine/files.ts';
import { runTicket } from '../src/engine/harness.ts';
import type { ChatMessage, Inference } from '../src/engine/inference.ts';
import { LocalSandbox, type RunResult, type Sandbox } from '../src/engine/sandbox.ts';

const SPIKE = join(import.meta.dir, '..', 'spike');
const tmp = mkdtempSync(join(tmpdir(), 'mugge-render-test-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** A model that answers from a list, recording every call's messages. */
function scripted(answers: string[]) {
  const calls: ChatMessage[][] = [];
  const inference: Inference = {
    complete: () => Promise.reject(new Error('no JSON calls here')),
    text: async (_m, messages) => {
      calls.push(messages.map((m) => ({ ...m })));
      const value = answers[Math.min(calls.length - 1, answers.length - 1)]!;
      return { value, raw: value, promptTokens: 0, completionTokens: 10, ms: 0 };
    },
  };
  return { inference, calls };
}

/** LocalSandbox that keeps each run's whole result. */
function recording(): { sandbox: Sandbox; runs: Array<{ command: string } & RunResult> } {
  const local = new LocalSandbox();
  const runs: Array<{ command: string } & RunResult> = [];
  return {
    runs,
    sandbox: { name: 'local', run: async (command, o) => { const r = await local.run(command, o); runs.push({ command, ...r }); return r; } },
  };
}

const [spike] = findProjects(SPIKE);

describe('render-call', () => {
  for (const t of spike!.tickets) {
    test(`renders the engine's write and fix calls for ${t.key}`, async () => {
      const ref = solution(spike!, t.ticket.id);
      const broken = Object.fromEntries(t.ticket.owns.map((p) => [p, 'this does not build\n']));
      const { inference, calls } = scripted([renderFiles(broken, 'first try'), renderFiles(ref, 'reference')]);
      const { sandbox, runs } = recording();
      const wt = join(tmp, t.ticket.id);
      prepare(spike!, wt, t.deps);
      const r = await runTicket(t.ticket, wt, { plan: spike!.plan, inference, sandbox, emit: () => {} });
      expect(r.ok).toBe(true);
      expect(calls.length).toBe(2);
      const fail = runs.find((x) => x.code !== 0)!;

      const write = renderCall({ project: spike!.dir, ticket: t.ticket.id });
      expect(write.messages).toEqual(calls[0]!);
      const fix = renderCall({
        project: spike!.dir, ticket: t.ticket.id, current: broken,
        failed: { command: fail.command, output: fail.out, timedOut: fail.timedOut, raw: true },
      });
      expect(fix.messages).toEqual(calls[1]!);
      // the fix call keeps the write call's pack as its prefix up to the owned files (prefix caching)
      const head = calls[0]![1]!.content.split('## Your files as they are now')[0]!;
      expect(calls[1]![1]!.content.startsWith(head)).toBe(true);
    }, 60_000);
  }

  test('an answer comes back in the canonical form, with missing and extra files named', () => {
    const t = spike!.tickets.find((x) => x.ticket.owns.length === 1)!;
    const path = t.ticket.owns[0]!;
    const raw = `Here you go.\n\n**${path}**\n\`\`\`\nint x;\n\`\`\`\n\nother.c\n\`\`\`c\nint y;\n\`\`\`\nNote: done`;
    const { content, parsed } = canonicalAnswer(raw, t.ticket.owns);
    expect(parsed.missing).toEqual([]);
    expect(parsed.extra).toEqual(['other.c']);
    expect(content).toBe(renderFiles({ [path]: 'int x;\n' }, 'done'));
    // canonical answers are a fixed point
    expect(canonicalAnswer(content, t.ticket.owns).content).toBe(content);
    expect(canonicalAnswer('no files at all', t.ticket.owns).parsed.missing).toEqual([path]);
  });

  test('the JSONL ops: tickets, parse, answer, and errors name the problem', () => {
    const list = handle({ op: 'tickets', bench: SPIKE }).tickets as any[];
    expect(list.length).toBe(15);
    expect(list[0]).toHaveProperty('hidden');
    expect(handle({ op: 'answer', files: { 'a.py': 'x = 1\n' }, note: 'n' }).content).toBe('a.py\n```python\nx = 1\n```\n\nNote: n');
    expect(handle({ op: 'parse', answer: 'a.py\n```python\nx = 1\n```', owns: ['a.py'] }).files).toEqual({ 'a.py': 'x = 1\n' });
    expect(() => handle({ op: 'fix', project: spike!.dir, ticket: spike!.tickets[0]!.ticket.id })).toThrow('failed');
    expect(() => handle({ op: 'nope' })).toThrow('unknown op');
  });
});
