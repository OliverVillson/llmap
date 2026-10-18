import { afterAll, describe, expect, test } from 'bun:test';
import { componentOf, crewShares, design, planShape, scaffold, tickets, type Design } from '../src/plan/architect.ts';
import { contextFit, OpenAICompatible, type CallTweaks, type ChatMessage, type Completion, type Inference } from '../src/engine/inference.ts';
import { validatePlan } from '../src/tickets/schema.ts';

interface Seen { client: string; system: string; user: string; maxTokens: number; tweaks?: CallTweaks; turns: number }

/** A planner that answers from the prompt: `answer(system, user, turns)` gives the JSON. */
function scripted(name: string, seen: Seen[], answer: (system: string, user: string, turns: number) => unknown): Inference {
  const complete = async <T,>(_m: string, messages: ChatMessage[], _s: object, maxTokens: number, _sig?: AbortSignal, tweaks?: CallTweaks): Promise<Completion<T>> => {
    const system = messages[0]!.content, user = messages[1]!.content;
    seen.push({ client: name, system, user, maxTokens, tweaks, turns: messages.length });
    await Bun.sleep(5);
    return { value: answer(system, user, messages.length) as T, raw: '', promptTokens: 0, completionTokens: 0, ms: 5 };
  };
  return { complete, text: () => { throw new Error('no text calls'); } } as Inference;
}

const DESIGN: Design = {
  summary: 'A toolkit.',
  components: ['Parser', 'Render', 'Cli', 'Store'].map((name, i) => ({
    name, language: 'Python', image: 'python' as const, path: `src/${name.toLowerCase()}`, responsibility: `the ${name.toLowerCase()}`,
    interfaces: [{ path: `src/${name.toLowerCase()}/api.py`, description: 'its API' }, ...(i === 0 ? [{ path: 'src/parser/types.py', description: 'types' }] : [])],
  })),
  data_flow: 'parser → render', decisions: [], questions: [],
};

const ticket = (id: string, owns: string[], depends_on: string[] = []) => ({
  id, title: id, role: 'implement', model: 'coder', priority: 2, depends_on, owns, reads: [], context: id,
  acceptance: [`python3 -m unittest tests.${id}`], image: 'python', budget: { attempts: 4, max_tokens: 96000 }, escalate_to: null,
});

describe('planning one prompt into a crew of tickets', () => {
  test('the shares add up to the crew, more interfaces getting more', () => {
    const s = crewShares(DESIGN, 64);
    expect(s.reduce((a, b) => a + b, 0)).toBe(64);
    expect(s[0]).toBeGreaterThan(s[1]!);
    expect(crewShares(DESIGN, 3).every((x) => x >= 1)).toBe(true);
  });

  test('a file belongs to the component whose folder holds it', () => {
    expect(componentOf(DESIGN, 'src/render/html.py')).toBe(1);
    expect(componentOf(DESIGN, 'pyproject.toml')).toBe(-1);
  });

  test('the design is told the crew size, and no planning call is held to the per-call cap', async () => {
    const seen: Seen[] = [];
    await design(scripted('m', seen, () => DESIGN), 'build a toolkit', [], 'architect', { crew: 64 });
    expect(seen[0]!.user).toContain('64 coding agents work at the same time');
    expect(seen[0]!.tweaks?.planning).toBe(true);
  });

  test('64 agents: shared files first, then per-component scaffolds and tickets in parallel, merged into 64 tickets', async () => {
    const seen: Seen[] = [];
    const shares = crewShares(DESIGN, 64);
    const manager = scripted('manager', seen, (system) => {
      if (system.includes('shared part')) return { files: { 'pyproject.toml': '[project]\n', ...Object.fromEntries(DESIGN.components.flatMap((c) => c.interfaces.map((i) => [i.path, '# api\n']))) }, note: 'shared' };
      throw new Error(`the manager got ${system.slice(0, 60)}`);
    });
    const parts = (i: number) => scripted(`part-${i}`, seen, (system) => {
      const c = DESIGN.components[i]!, dir = c.path;
      if (system.includes('one component of a project scaffold')) {
        const files: Record<string, string> = Object.fromEntries(Array.from({ length: shares[i]! }, (_, k) => [[`${dir}/m${k}.py`, 'raise NotImplementedError\n'], [`tests/test_${c.name.toLowerCase()}_${k}.py`, 'import unittest\n']]).flat());
        // one part writes a shared file again: the shared copy is kept
        if (i === 1) files['pyproject.toml'] = '[project]\nname = "x"\n';
        return { files, note: c.name };
      }
      // tickets: one per stub, ids without the component prefix (so they clash across parts), one
      // depending on the first, and one reaching into another component's file
      const ts = Array.from({ length: shares[i]! }, (_, k) => ticket(`m${k}`, [`${dir}/m${k}.py`, ...(i === 2 && k === 0 ? ['src/parser/m0.py'] : [])], k === 1 ? ['m0'] : []));
      return { interfaces: [`${dir}/api.py`], tickets: ts };
    });
    const sc = await scaffold(manager, DESIGN, 'planner', { crew: 64, parts });
    expect(sc.files['pyproject.toml']).toBe('[project]\n');
    expect(sc.warnings.some((w) => w.includes('rewrote shared file pyproject.toml'))).toBe(true);
    expect(Object.keys(sc.files).filter((f) => /m\d+\.py$/.test(f))).toHaveLength(64);

    const plan = await tickets(manager, DESIGN, Object.keys(sc.files), 'planner', { crew: 64, parts });
    expect(plan.tickets).toHaveLength(64);
    expect(new Set(plan.tickets.map((t) => t.id)).size).toBe(64);
    expect(plan.tickets.find((t) => t.id === 'render-m1')!.depends_on).toEqual(['render-m0']);
    expect(plan.tickets.find((t) => t.id === 'cli-m0')!.owns).toEqual(['src/cli/m0.py']);
    expect(plan.warnings!.some((w) => w.includes('cli-m0: dropped owned files'))).toBe(true);
    expect(planShape(plan, 64)).toMatchObject({ tickets: 64, startable: 60, warnings: [] });
    expect(() => validatePlan(plan)).not.toThrow();

    // 1 shared + 4 scaffold parts + 4 ticket parts; every part on its own client, each told its share
    expect(seen.map((s) => s.client).sort()).toEqual(['manager', ...[0, 1, 2, 3].flatMap((i) => [`part-${i}`, `part-${i}`])].sort());
    expect(seen.every((s) => s.tweaks?.planning)).toBe(true);
    const parserTickets = seen.find((s) => s.client === 'part-0' && s.system.includes('tickets'))!;
    expect(parserTickets.user).toContain(`aim for about ${shares[0]} tickets`);
    expect(parserTickets.user).toContain('Start every ticket id with "parser-"');
  });

  test('too few tickets for the crew are asked for again once', async () => {
    const seen: Seen[] = [];
    const files = Array.from({ length: 20 }, (_, k) => `src/parser/m${k}.py`);
    const inf = scripted('m', seen, (_s, _u, turns) => ({ interfaces: [], tickets: files.slice(0, turns === 2 ? 10 : 20).map((f, k) => ticket(`t${k}`, [f])) }));
    const plan = await tickets(inf, DESIGN, files, 'planner', { crew: 20 });
    expect(plan.tickets).toHaveLength(20);
    expect(seen).toHaveLength(2);
    expect(seen[0]!.user).toContain('Aim for about 20 tickets');
    expect(plan.warnings).toBeUndefined();
  });

  test('a plan still short after the second ask keeps the larger answer and says so', async () => {
    const seen: Seen[] = [];
    const inf = scripted('m', seen, (_s, _u, turns) => ({ interfaces: [], tickets: Array.from({ length: turns === 2 ? 6 : 4 }, (_, k) => ticket(`t${k}`, [`f${k}.py`], k ? ['t0', 'ghost'] : [])) }));
    const plan = await tickets(inf, DESIGN, [], 'planner', { crew: 20 });
    expect(plan.tickets).toHaveLength(6);
    expect(plan.warnings).toContain('6 tickets for 20 agents: 14 agents have nothing to do');
    expect(plan.warnings).toContain('only 1 of 6 tickets can start at once; the rest wait on dependencies');
    expect(plan.warnings!.some((w) => w.includes('dropped dependencies on ghost'))).toBe(true);
  });

  test('a small crew, or no parts, plans in one call as before', async () => {
    const seen: Seen[] = [];
    const inf = scripted('m', seen, (system) => (system.startsWith('You write the scaffold') ? { files: { 'a.py': '' }, note: '' } : { interfaces: [], tickets: [ticket('a', ['a.py'])] }));
    await scaffold(inf, DESIGN, 'planner', { crew: 64 });
    await tickets(inf, DESIGN, ['a.py']);
    expect(seen).toHaveLength(2);
    expect(seen[1]!.user).not.toContain('## Crew');
  });
});

describe('fitting a planning call to the context', () => {
  test('reads what fits from vLLM and SGLang refusals', () => {
    expect(contextFit("This model's maximum context length is 65536 tokens. However, you requested 70000 tokens (5000 in the messages, 65000 in the completion).")).toBe(60520);
    expect(contextFit("'max_tokens' or 'max_completion_tokens' is too large: 65000. This model's maximum context length is 65536 tokens and your request has 5000 input tokens (65000 > 65536 - 5000).")).toBe(60520);
    expect(contextFit("Requested token count exceeds the model's maximum context length of 65536 tokens. You requested a total of 70000 tokens: 5000 tokens from the input messages and 65000 tokens for the completion.")).toBe(60520);
    expect(contextFit('model not found')).toBeUndefined();
    expect(contextFit("maximum context length is 65536 tokens and your request has 65000 input tokens")).toBeUndefined();
  });

  const asked: number[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const body = (await req.json()) as { max_tokens: number };
      asked.push(body.max_tokens);
      if (body.max_tokens > 61_000) return new Response(JSON.stringify({ message: `This model's maximum context length is 65536 tokens. However, you requested ${body.max_tokens + 5000} tokens (5000 in the messages, ${body.max_tokens} in the completion).` }), { status: 400 });
      return Response.json({ choices: [{ message: { content: '{"ok": true}' }, finish_reason: 'stop' }], usage: { prompt_tokens: 5000, completion_tokens: 5 } });
    },
  });
  afterAll(() => server.stop(true));

  test('a call over the context is asked once more with what fits; planning ignores the per-call cap', async () => {
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', maxCallTokens: 16_384 });
    const c = await inf.complete<{ ok: boolean }>('m', [{ role: 'user', content: 'plan' }], {}, 65_000, undefined, { planning: true });
    expect(c.value.ok).toBe(true);
    expect(asked).toEqual([65_000, 60_520]);
    expect(c.maxTokens).toBe(60_520);
    asked.length = 0;
    await inf.complete('m', [{ role: 'user', content: 'code' }], {}, 65_000);
    expect(asked).toEqual([16_384]);
    asked.length = 0;
    const capped = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', maxCallTokens: 16_384, planMaxCallTokens: 40_000 });
    await capped.complete('m', [{ role: 'user', content: 'plan' }], {}, 65_000, undefined, { planning: true });
    expect(asked).toEqual([40_000]);
  });
});
