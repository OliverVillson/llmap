/**
 * What a real run on vLLM 0.31 can hit that the mocks did not: its own wording for a call over
 * the context, a server that restarts or blips mid-run, and a check whose child process never ends.
 */
import { afterAll, afterEach, describe, expect, test } from 'bun:test';
import { contextFit, OpenAICompatible, setServerRetryMs, SERVER_RETRY_MS } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';

const ok = (content = '{"ok": true}') => Response.json({ choices: [{ message: { role: 'assistant', content, reasoning: 'thought' }, finish_reason: 'stop' }], usage: { prompt_tokens: 5000, completion_tokens: 5 } });
/** vLLM 0.31's error body (entrypoints/serve/exception_handling: ErrorResponse, VLLMValidationError's str). */
const vllmError = (message: string, param: string, value: number) => Response.json({ error: { message: `${message} (parameter=${param}, value=${value})`, type: 'Bad Request', param, code: 400 } }, { status: 400 });

describe('vLLM 0.31 context refusals', () => {
  // the two refusals as vllm/renderers/params.py words them in 0.31.0
  const tooBig = (max: number) => `max_tokens=${max} cannot be greater than max_model_len=65536. Please request fewer output tokens.`;
  const overContext = (max: number, prompt: string) => `This model's maximum context length is 65536 tokens. However, you requested ${max} output tokens and your prompt contains ${prompt} input tokens, for a total of ${prompt.replace('at least ', '') === prompt ? Number(prompt) + max : `at least ${Number(prompt.replace('at least ', '')) + max}`} tokens. Please reduce the length of the input prompt or the number of requested output tokens.`;

  test('reads what fits from both', () => {
    expect(contextFit(overContext(65_000, '5000'))).toBe(60_520);
    expect(contextFit(`{"error":{"message":"${overContext(51_968, '15000')} (parameter=input_tokens, value=15000)"}}`)).toBe(65536 - 15000 - 16);
    expect(contextFit(overContext(16_384, 'at least 49153'))).toBe(65536 - 49153 - 16);
    // the whole context is refused before the prompt is counted: ask for one less, and the next refusal says the prompt's size
    expect(contextFit(tooBig(80_000))).toBe(65_535);
    expect(contextFit(overContext(16_384, '65000'))).toBeUndefined();
  });

  const asked: number[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const body = (await req.json()) as { max_tokens: number };
      asked.push(body.max_tokens);
      if (body.max_tokens > 65536) return vllmError(tooBig(body.max_tokens), 'max_tokens', body.max_tokens);
      if (5000 + body.max_tokens > 65536) return vllmError(overContext(body.max_tokens, '5000'), 'input_tokens', 5000);
      return ok();
    },
  });
  afterAll(() => server.stop(true));

  test('a planning call above the whole context is fitted in two steps', async () => {
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', maxCallTokens: 16_384 });
    const c = await inf.complete<{ ok: boolean }>('m', [{ role: 'user', content: 'plan' }], {}, 80_000, undefined, { planning: true });
    expect(c.value.ok).toBe(true);
    expect(asked).toEqual([80_000, 65_535, 60_520]);
    expect(c.reasoning).toBe('thought');
  });
});

describe('a server that goes away mid-run', () => {
  const saved = [...SERVER_RETRY_MS];
  afterEach(() => setServerRetryMs(saved));

  test('a restart (refused, then 503 while loading) is waited out, not spent as an attempt', async () => {
    setServerRetryMs([50]);
    const probe = Bun.serve({ port: 0, fetch: () => new Response('') });
    const port = probe.port;
    probe.stop(true);
    let calls = 0;
    let up: ReturnType<typeof Bun.serve> | null = null;
    // nothing listens for 300 ms, then the server answers 503 twice before it is ready
    setTimeout(() => {
      up = Bun.serve({ port, fetch: () => (++calls <= 2 ? new Response('loading', { status: 503 }) : ok('done')) });
    }, 300);
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${port}/v1`, defaultModel: 'm' });
    const c = await inf.text('m', [{ role: 'user', content: 'x' }], 100);
    expect(c.value).toBe('done');
    expect(calls).toBe(3);
    up!.stop(true);
  });

  test('a 500 is sent twice more, then fails; a 400 is not sent again', async () => {
    setServerRetryMs([10]);
    let n = 0;
    const s = Bun.serve({ port: 0, fetch: () => (++n, new Response('{"error":{"message":"EngineCore encountered an issue"}}', { status: 500 })) });
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${s.port}/v1`, defaultModel: 'm' });
    await expect(inf.text('m', [{ role: 'user', content: 'x' }], 100)).rejects.toThrow(/inference 500/);
    expect(n).toBe(3);
    n = 0;
    const bad = Bun.serve({ port: 0, fetch: () => (++n, new Response('{"error":{"message":"bad"}}', { status: 400 })) });
    await expect(new OpenAICompatible({ baseUrl: `http://127.0.0.1:${bad.port}/v1`, defaultModel: 'm' }).text('m', [{ role: 'user', content: 'x' }], 100)).rejects.toThrow(/inference 400/);
    expect(n).toBe(1);
    s.stop(true);
    bad.stop(true);
  });

  test('a server that stays away fails the call after the wait; the run stop ends the wait at once', async () => {
    setServerRetryMs([50]);
    const probe = Bun.serve({ port: 0, fetch: () => new Response('') });
    const port = probe.port;
    probe.stop(true);
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${port}/v1`, defaultModel: 'm', serverWaitMs: 300 });
    await expect(inf.text('m', [{ role: 'user', content: 'x' }], 100)).rejects.toThrow(/not answering after \d+ tries/);
    const stop = new AbortController();
    const t0 = Date.now();
    setTimeout(() => stop.abort(), 100);
    await expect(new OpenAICompatible({ baseUrl: `http://127.0.0.1:${port}/v1`, defaultModel: 'm' }).text('m', [{ role: 'user', content: 'x' }], 100, stop.signal)).rejects.toThrow();
    expect(Date.now() - t0).toBeLessThan(1000);
  });
});

describe('a check whose child process never ends', () => {
  test('times out on time, and the child is gone', async () => {
    const tag = `mugge-hang-${process.pid}`;
    const t0 = Date.now();
    // the test runner itself is killed at the timeout; before, its child kept the output open forever
    const r = await new LocalSandbox().run(`python3 -c "import subprocess; subprocess.run(['bash', '-c', 'exec -a ${tag} sleep 600'])"`, { cwd: '/tmp', image: 'python', timeoutMs: 1000 });
    expect(r.timedOut).toBe(true);
    expect(r.code).toBe(124);
    expect(Date.now() - t0).toBeLessThan(4000);
    await Bun.sleep(200);
    const left = Bun.spawnSync(['pgrep', '-f', tag]).stdout.toString().trim();
    expect(left).toBe('');
  });
});

describe('a planning call that thinks until max_tokens', () => {
  const sent: Array<{ effort?: string; thinking?: boolean }> = [];
  const design = { summary: 's', components: [{ name: 'core', language: 'Python', image: 'python', path: 'tk/', responsibility: 'r', interfaces: [] }], data_flow: 'd', decisions: [], questions: [] };
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const body = (await req.json()) as { max_tokens: number; chat_template_kwargs?: { reasoning_effort?: string; enable_thinking?: boolean } };
      sent.push({ effort: body.chat_template_kwargs?.reasoning_effort, thinking: body.chat_template_kwargs?.enable_thinking });
      // vLLM's qwen3 parser: cut while thinking leaves content null and everything in reasoning
      if (body.chat_template_kwargs?.reasoning_effort !== 'low') return Response.json({ choices: [{ message: { role: 'assistant', content: null, reasoning: 'hmm '.repeat(500) }, finish_reason: 'length' }], usage: { prompt_tokens: 100, completion_tokens: body.max_tokens } });
      return ok(JSON.stringify(design));
    },
  });
  afterAll(() => server.stop(true));

  test('is asked once more with thinking low, and says so', async () => {
    const { design: plan } = await import('../src/plan/architect.ts');
    const inf = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', extra: { chat_template_kwargs: { enable_thinking: true, reasoning_effort: 'xhigh' } } });
    const d = await plan(inf, 'a toolbox');
    expect(d.components).toHaveLength(1);
    expect(sent.map((s) => s.effort)).toEqual(['xhigh', 'low']);
    expect(d.warnings?.[0]).toMatch(/the design ran out of tokens .*asked once more, thinking low/);
  });
});

describe('a planned ticket budget', () => {
  test('is the engine\'s, whatever the model wrote', async () => {
    const { tickets } = await import('../src/plan/architect.ts');
    const d = { summary: 's', components: [{ name: 'core', language: 'Python', image: 'python' as const, path: 'tk/', responsibility: 'r', interfaces: [] }], data_flow: 'd', decisions: [], questions: [] };
    const t = (id: string, budget?: object) => ({ id, title: id, role: 'implement', model: 'coder', priority: 2, depends_on: [], owns: [`tk/${id}.py`], reads: [], context: id, acceptance: [`python3 -m unittest tests.test_${id}`], image: 'python', ...(budget ? { budget } : {}) });
    const inf = { complete: async () => ({ value: { interfaces: [], tickets: [t('a', { attempts: 2, max_tokens: 3000 }), t('b')] }, raw: '', promptTokens: 0, completionTokens: 0, ms: 1 }), text: async () => { throw new Error('no'); } } as any;
    const p = await tickets(inf, d, ['tk/a.py', 'tk/b.py'], 'planner', { crew: 2 });
    expect(p.tickets.map((x) => x.budget)).toEqual([{ attempts: 4, max_tokens: 96_000 }, { attempts: 4, max_tokens: 96_000 }]);
    expect(p.warnings?.join(' ')).toContain('1 tickets came with a budget of their own (smallest 3000 tokens)');
  });
});
