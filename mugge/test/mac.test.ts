/**
 * The Mac demo (deploy/mac.sh, llama-server with Qwen3.5): the answers llama-server really sends
 * (test/fixtures/llama-server, captured from a llama.cpp build of 2026-10-19 serving a
 * random-weight model with Qwen3.5's tokenizer and chat template), its context refusal, the
 * thinking budget, and deploy/mac_crew.py's crew agreeing with the engine's presets.
 */
import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { inferenceFromEnv, machinesFromEnv } from '../src/config.ts';
import { BackendPool, kvBytesPerToken, llamaServerArgs, MODEL_PRESETS } from '../src/engine/backends.ts';
import { contextFit, contextRefusal, OpenAICompatible, thinkingBudgetFromEnv } from '../src/engine/inference.ts';

const FIX = join(import.meta.dir, 'fixtures', 'llama-server');
const fixture = (name: string) => JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')) as { status: number; request: any; response: any };

/** A stand-in llama-server that answers each call with the next captured response, keeping the bodies sent. */
function replay(names: string[]) {
  const bodies: any[] = [];
  let i = 0;
  const server = Bun.serve({
    port: 0,
    fetch: async (req) => {
      bodies.push(await req.json());
      const f = fixture(names[i++]!);
      return Response.json(f.response, { status: f.status });
    },
  });
  return { url: `${server.url}v1`, bodies, stop: () => server.stop(true) };
}

const msgs = [{ role: 'user' as const, content: 'Write a Python function add(a, b).' }];

describe('llama-server answers', () => {
  test('thinking off: the answer in content, no thinking; thinking on and cut: reasoning_content only, an empty answer', async () => {
    const s = replay(['thinking-off', 'thinking-on-cut']);
    try {
      const c = new OpenAICompatible(inferenceFromEnv({ MUGGE_INFERENCE_URL: s.url, MUGGE_MODEL: 'qwen3.5-9b' }));
      const off = await c.text('w', msgs, 12, undefined, { thinking: false });
      expect(off.value.length).toBeGreaterThan(0);
      expect(off.reasoning).toBeUndefined();
      expect(off.cut).toBe(true);
      expect(off.thinking).toBe(false);
      const on = await c.text('w', msgs, 12, undefined, { thinking: true });
      expect(on.value).toBe('');
      expect(on.reasoning!.length).toBeGreaterThan(0);
      expect(on.reasoningChars).toBe(on.reasoning!.length);
      expect(on.cut).toBe(true);
      // llama-server does not count thinking tokens apart
      expect(on.reasoningTokens).toBeUndefined();
      expect(s.bodies[0].chat_template_kwargs).toEqual({ enable_thinking: false });
      expect(s.bodies[1].chat_template_kwargs).toEqual({ enable_thinking: true });
    } finally {
      s.stop();
    }
  });

  test('a thinking budget: the server ends the thinking, the answer follows, and a schema answer after it parses', async () => {
    const s = replay(['thinking-budget', 'schema-thinking-budget']);
    try {
      const env = { MUGGE_INFERENCE_URL: s.url, MUGGE_THINKING_BUDGET: 'low=6,xhigh=8192' };
      const c = new OpenAICompatible(inferenceFromEnv(env));
      const t = await c.text('w', msgs, 24, undefined, { thinking: true, reasoningEffort: 'low' });
      expect(t.reasoning!.length).toBeGreaterThan(0);
      expect(t.value.length).toBeGreaterThan(0);
      const schema = fixture('schema-thinking-budget').request.response_format.json_schema.schema;
      const r = await c.complete<{ verdict: string; reason: string }>('lead', msgs, schema, 200, undefined, { thinking: true, reasoningEffort: 'low' });
      expect(['approve', 'send_back']).toContain(r.value.verdict);
      expect(r.reasoning!.length).toBeGreaterThan(0);
      expect(s.bodies.map((b) => b.thinking_budget_tokens)).toEqual([6, 6]);
      expect(s.bodies[1].response_format.type).toBe('json_schema');
    } finally {
      s.stop();
    }
  });

  test('the context refusal: read, counted, and not asked again when the prompt alone does not fit', async () => {
    const f = fixture('over-context');
    const text = JSON.stringify(f.response);
    expect(f.status).toBe(400);
    expect(contextRefusal(text)).toEqual({ limit: 2048, input: 2112, atLeast: false });
    expect(contextFit(text)).toBeUndefined();
    // older builds said only "the request exceeds the available context size" beside the counts
    expect(contextRefusal('{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error","n_prompt_tokens":5000,"n_ctx":16384}}'))
      .toEqual({ limit: 16384, input: 5000, atLeast: false });
    expect(contextFit('request (5000 tokens) exceeds the available context size (16384 tokens), try increasing it')).toBe(16384 - 5000 - 16);
    const s = replay(['over-context']);
    try {
      const c = new OpenAICompatible(inferenceFromEnv({ MUGGE_INFERENCE_URL: s.url }));
      await expect(c.text('w', msgs, 10)).rejects.toThrow(/inference 400: .*exceeds the available context size/);
      expect(s.bodies.length).toBe(1);
    } finally {
      s.stop();
    }
  });
});

describe('thinking budget', () => {
  test('MUGGE_THINKING_BUDGET: one number, numbers per level, or both', () => {
    expect(thinkingBudgetFromEnv({})).toBeUndefined();
    expect(thinkingBudgetFromEnv({ MUGGE_THINKING_BUDGET: '2048' })).toEqual({ all: 2048, levels: {} });
    expect(thinkingBudgetFromEnv({ MUGGE_THINKING_BUDGET: 'low=1024, Medium=4096,xhigh=8192' })).toEqual({ levels: { low: 1024, medium: 4096, xhigh: 8192 } });
    expect(thinkingBudgetFromEnv({ MUGGE_THINKING_BUDGET: '2048,xhigh=0' })).toEqual({ all: 2048, levels: { xhigh: 0 } });
    expect(() => thinkingBudgetFromEnv({ MUGGE_THINKING_BUDGET: 'worker=10' })).toThrow(/xhigh, medium or low/);
    expect(() => thinkingBudgetFromEnv({ MUGGE_THINKING_BUDGET: 'low=-1' })).toThrow(/whole number/);
  });

  test('each role\'s level picks its budget; thinking-off calls and unset budgets send none', async () => {
    const bodies: any[] = [];
    const server = Bun.serve({ port: 0, fetch: async (req) => (bodies.push(await req.json()), Response.json({ choices: [{ message: { content: '{"verdict":"approve","reason":""}' }, finish_reason: 'stop' }], usage: {} })) });
    const url = `${server.url}v1`;
    const env = {
      MUGGE_REASONING_EFFORT: 'worker=low,lead=off,manager=medium', MUGGE_THINKING_BUDGET: 'low=1024,medium=4096',
      MUGGE_BACKENDS: JSON.stringify([{ name: 'w', url, roles: ['worker'], slots: 2 }, { name: 'l', url, roles: ['lead'], slots: 1 }, { name: 'm', url, roles: ['manager'], slots: 1 }]),
    };
    try {
      const pool = BackendPool.fromMachines(machinesFromEnv(env), inferenceFromEnv({ ...env, MUGGE_INFERENCE_URL: url }), env);
      await pool.forAgent({ role: 'worker' }).text('coder', msgs, 10);
      await pool.forAgent({ role: 'worker' }).text('coder', msgs, 10, undefined, { thinking: false });
      await pool.forAgent({ role: 'lead' }).complete('lead', msgs, {}, 10, undefined, { thinking: false, thinkingIsDefault: true });
      await pool.forAgent({ role: 'manager' }).complete('architect', msgs, {}, 10, undefined, { planning: true });
      expect(bodies.map((b) => b.thinking_budget_tokens)).toEqual([1024, undefined, undefined, 4096]);
      // no budget set: nothing extra on the wire (vLLM and the B200 crew see the same bodies as before)
      bodies.length = 0;
      const plain = new OpenAICompatible(inferenceFromEnv({ MUGGE_INFERENCE_URL: url, MUGGE_REASONING_EFFORT: 'medium' }));
      await plain.text('coder', msgs, 10);
      expect('thinking_budget_tokens' in bodies[0]).toBe(false);
    } finally {
      server.stop(true);
    }
  });
});

describe('the Mac crew (deploy/mac_crew.py)', () => {
  const crew = (...args: string[]) => {
    const r = Bun.spawnSync(['python3', join(import.meta.dir, '..', 'deploy', 'mac_crew.py'), ...args], { env: { PATH: process.env.PATH ?? '' } });
    if (r.exitCode !== 0) throw new Error(r.stderr.toString());
    return r.stdout.toString();
  };

  test('16 GB with the macOS default limit: Qwen3.5-9B Q4_K_M, 8 workers + 2 leads + 1 orchestrator, every call at its longest', () => {
    const pl = JSON.parse(crew('--mem-gb', '16', '--wired-mb', '0', '--json'));
    expect(pl).toMatchObject({ file: 'Qwen3.5-9B-Q4_K_M.gguf', workers: 8, leads: 2, orchestrator: 1, slots: 11, agents: 10, ctx: 16384, fits: true });
    expect(pl.need_bytes).toBeLessThan(pl.wired_bytes);
    // a smaller limit gives a smaller crew rather than an overcommitted one
    const small = JSON.parse(crew('--mem-gb', '16', '--wired-mb', '9000', '--json'));
    expect(small.workers).toBeLessThan(8);
    expect(small.fits).toBe(true);
  });

  test('its KV and state figures are the engine presets\' (one source of truth for memory)', () => {
    for (const [model, preset] of [['9b', 'qwen3.5-9b'], ['4b', 'qwen3.5-4b']] as const) {
      const pl = JSON.parse(crew('--mem-gb', '16', '--wired-mb', '0', '--model', model, '--json'));
      const p = MODEL_PRESETS[preset]!;
      expect(pl.preset).toBe(preset);
      expect(pl.kv_bytes_per_token).toBeCloseTo(kvBytesPerToken(p, 'q8_0') , 6);
      expect(pl.state_bytes).toBe(p.stateBytesPerSlot);
    }
    const reap = JSON.parse(crew('--mem-gb', '16', '--wired-mb', '12288', '--model', '/nowhere/qwen36-r50w95s.gguf', '--preset', 'qwen3.6-35b-a3b', '--json'));
    expect(reap.state_bytes).toBe(MODEL_PRESETS['qwen3.6-35b-a3b']!.stateBytesPerSlot);
  });

  test('its machine list loads, and the llama-server line it implies has no shared pool', () => {
    const list = crew('--mem-gb', '16', '--wired-mb', '0', '--backends');
    const machines = machinesFromEnv({ MUGGE_BACKENDS: list });
    expect(machines.map((m) => [m.name, m.roles[0], m.slots])).toEqual([['mac-workers', 'worker', 8], ['mac-leads', 'lead', 2], ['mac-orchestrator', 'manager', 1]]);
    expect(machines.every((m) => m.server === 'llama-server' && m.model === 'qwen3.5-9b' && m.ctxPerCall === 16384)).toBe(true);
    expect(llamaServerArgs(machines[0]!).join(' ')).toContain('-np 8 -c 131072 -fa on');
    expect(crew('--mem-gb', '16', '--wired-mb', '0', '--sizes').trim()).toBe('8 2 1 11 10');
  });
});
