import { afterAll, describe, expect, test } from 'bun:test';
import { componentOf, crewShares, design, normPath, planShape, salvageJson, scaffold, tickets, type Design } from '../src/plan/architect.ts';
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

describe('a 64-agent plan survives what a real model gets wrong (debug round 2026-10-18)', () => {
  const parts5 = (fn: (i: number, system: string, user: string, turns: number) => unknown) => (i: number) => scripted(`part-${i}`, [], (system, user, turns) => fn(i, system, user, turns));
  const stubs = (i: number) => Object.fromEntries(Array.from({ length: 16 }, (_, k) => [`${DESIGN.components[i]!.path}/m${k}.py`, '']));
  const sharedFiles = { 'pyproject.toml': '', ...Object.fromEntries(DESIGN.components.flatMap((c) => c.interfaces.map((x) => [x.path, '']))) };
  const manager = scripted('manager', [], () => ({ files: sharedFiles, note: '' }));
  const fail = (message: string, completion?: Partial<Completion<string>>) => Object.assign(new Error(message), completion ? { completion } : {});

  test('paths are made plain, and unsafe ones refused', () => {
    expect(normPath('./src/a.py')).toBe('src/a.py');
    expect(normPath('/src//a.py')).toBe('src/a.py');
    expect(normPath('../a.py')).toBeUndefined();
    expect(normPath('src/')).toBeUndefined();
    expect(normPath('.git/hooks/x')).toBeUndefined();
  });

  test('a JSON answer cut at max_tokens keeps its whole files and whole tickets', () => {
    expect(salvageJson('{"files": {"a.py": "x = 1\\n", "b.py": "y = {\\"k\\": [1,')).toEqual({ files: { 'a.py': 'x = 1\n' } });
    const t = JSON.stringify(ticket('a', ['a.py']));
    expect(salvageJson(`{"interfaces": ["i.py"], "tickets": [${t}, ${t.replace('"a"', '"b"')}, {"id": "c", "ow`)).toEqual({ interfaces: ['i.py'], tickets: [ticket('a', ['a.py']), { ...ticket('a', ['a.py']), id: 'b' }] });
    expect(salvageJson('{"files": {"a.py": "unfinished')).toBeUndefined();
    expect(salvageJson('')).toBeUndefined();
  });

  test('a cut scaffold part keeps its complete files; a failed one costs its component, not the plan', async () => {
    const parts = (i: number) => ({
      complete: async () => {
        if (i === 1) throw fail('Unexpected end of JSON input', { raw: '{"files": {"src/render/m0.py": "", "src/render/m1.py": "", "src/render/m2', cut: true, completionTokens: 30000 });
        if (i === 2) throw fail('inference 500: engine dead');
        return { value: { files: stubs(i), note: '' }, raw: '', promptTokens: 0, completionTokens: 0, ms: 1 };
      },
      text: async () => { throw new Error('no'); },
    }) as unknown as Inference;
    const sc = await scaffold(manager, DESIGN, 'planner', { crew: 64, parts });
    expect(Object.keys(sc.files).filter((f) => f.startsWith('src/render/m'))).toEqual(['src/render/m0.py', 'src/render/m1.py']);
    expect(sc.warnings).toContain("Render's scaffold was cut at 30000 tokens; its complete part is kept");
    expect(sc.warnings.some((w) => w.startsWith("Cli's scaffold failed (inference 500: engine dead)"))).toBe(true);
    expect(sc.failed).toEqual([2]);
    // the failed component gets no ticket call; the others plan as usual
    const asked: number[] = [];
    const plan = await tickets(manager, DESIGN, sc, 'planner', { crew: 64, parts: parts5((i, _s, user) => { asked.push(i); return { interfaces: [], tickets: Object.keys(sc.files).filter((f) => user.includes(`- ${f}`) && /m\d+\.py$/.test(f)).map((f, k) => ticket(`t${k}`, [f])) }; }) });
    expect([...new Set(asked)].sort()).toEqual([0, 1, 3]);
    expect(plan.tickets.some((t) => t.id.startsWith('cli-'))).toBe(false);
    const dead = (() => ({ complete: async () => { throw fail('inference 500'); }, text: async () => '' })) as unknown as (i: number) => Inference;
    expect(scaffold(manager, DESIGN, 'planner', { crew: 64, parts: dead })).rejects.toThrow(/every component's scaffold failed/);
  });

  test('a ticket part that fails costs its component; cycles, ./ paths, long ids and empty tickets are mended, not fatal', async () => {
    const files = DESIGN.components.flatMap((_, i) => Object.keys(stubs(i)));
    const plan = await tickets(manager, DESIGN, files, 'planner', {
      crew: 64,
      parts: parts5((i) => {
        if (i === 3) throw new Error('inference 400: bad request');
        const dir = DESIGN.components[i]!.path;
        return { interfaces: [], tickets: [
          ticket('a', [`./${dir}/m0.py`], ['b']), ticket('b', [`/${dir}/m1.py`], ['a']),
          ticket(`${'very-long-ticket-name-'.repeat(3)}x`, [`${dir}/m2.py`]), ticket(`${'very-long-ticket-name-'.repeat(3)}y`, [`${dir}/m3.py`]),
          ticket('nothing', []), ticket('escape', ['../etc/passwd']),
        ] };
      }),
    });
    expect(() => validatePlan(plan)).not.toThrow();
    expect(plan.tickets.find((t) => t.id === 'parser-a')!.owns).toEqual(['src/parser/m0.py']);
    expect(plan.tickets.find((t) => t.id === 'parser-b')!.owns).toEqual(['src/parser/m1.py']);
    expect(plan.warnings!.some((w) => w.startsWith('dependency cycle parser-a -> parser-b -> parser-a'))).toBe(true);
    const long = plan.tickets.filter((t) => t.id.startsWith('parser-very'));
    expect(long).toHaveLength(2);
    expect(long.every((t) => t.id.length <= 63)).toBe(true);
    expect(long.map((t) => t.id.at(-1)).sort()).toEqual(['x', 'y']);
    expect(plan.tickets.some((t) => t.id.endsWith('nothing') || t.id.endsWith('escape'))).toBe(false);
    expect(plan.warnings!.some((w) => w.startsWith("Store's tickets failed (inference 400: bad request)"))).toBe(true);
    expect(plan.tickets.filter((t) => t.id.startsWith('store-'))).toHaveLength(0);
  });

  test("files a part wrote outside its folder are its tickets' to own; components sharing a folder share its files", async () => {
    const sc = await scaffold(manager, DESIGN, 'planner', { crew: 64, parts: parts5((i) => ({ files: { [`lib/${DESIGN.components[i]!.name.toLowerCase()}_impl.py`]: '' }, note: '' })) });
    expect(sc.writtenBy!['lib/render_impl.py']).toBe(1);
    const plan = await tickets(manager, DESIGN, sc, 'planner', { crew: 64, parts: parts5((i, _s, user) => {
      expect(user).toContain(`- lib/${DESIGN.components[i]!.name.toLowerCase()}_impl.py`);
      return { interfaces: [], tickets: [ticket('impl', [`lib/${DESIGN.components[i]!.name.toLowerCase()}_impl.py`])] };
    }) });
    expect(plan.tickets.map((t) => t.id).sort()).toEqual(['cli-impl', 'parser-impl', 'render-impl', 'store-impl']);
    const oneDir: Design = { ...DESIGN, components: DESIGN.components.map((c) => ({ ...c, path: 'tk/' })) };
    const p2 = await tickets(manager, oneDir, ['tk/a.py', 'tk/b.py', 'tk/c.py', 'tk/d.py'], 'planner', { crew: 64, parts: parts5((i) => ({ interfaces: [], tickets: [ticket('x', [`tk/${'abcd'[i]}.py`])] })) });
    expect(p2.tickets).toHaveLength(4);
  });

  test('a part whose tickets all queue behind one file is asked again, naming the file', async () => {
    const users: string[] = [];
    const plan = await tickets(manager, DESIGN, DESIGN.components.flatMap((_, i) => Object.keys(stubs(i))), 'planner', { crew: 64, parts: parts5((i, _s, _u, turns) => {
      const dir = DESIGN.components[i]!.path;
      if (turns > 2) users.push(_u);
      const n = crewShares(DESIGN, 64)[i]!;
      return { interfaces: [], tickets: Array.from({ length: n }, (_, k) => ticket(`m${k}`, [`${dir}/m${k}.py`, ...(turns === 2 ? [`${dir}/__init__.py`] : [])])) };
    }) });
    expect(planShape(plan, 64).startable).toBe(64);
    expect(plan.warnings).toBeUndefined();
    const parser = await (async () => {
      const seen: Seen[] = [];
      await tickets(manager, DESIGN, [], 'planner', { crew: 64, parts: (i) => scripted(`p${i}`, seen, (_s, _u, turns) => ({ interfaces: [], tickets: Array.from({ length: 4 }, (_, k) => ticket(`m${k}`, [`src/x${i}/m${k}.py`.replace(`src/x${i}`, DESIGN.components[i]!.path), `${DESIGN.components[i]!.path}/__init__.py`])) })) });
      return seen;
    })();
    const reask = parser.find((s) => s.turns > 2)!;
    expect(reask).toBeDefined();
  });

  test('one component for a big crew: the design is asked again for more', async () => {
    const seen: Seen[] = [];
    const one = { ...DESIGN, components: DESIGN.components.slice(0, 1) };
    const d = await design(scripted('m', seen, (_s, _u, turns) => (turns === 2 ? one : DESIGN)), 'build it', [], 'architect', { crew: 64, parts: () => scripted('p', [], () => ({})) });
    expect(d.components).toHaveLength(4);
    expect(seen).toHaveLength(2);
    expect(seen[0]!.user).toContain('Use at least 4 components');
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
