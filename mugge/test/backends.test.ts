import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { machinesFromEnv, poolFromEnv, samplingFromEnv } from '../src/config.ts';
import { BackendPool, describeProbe, MODEL_PRESETS, kvBytesPerToken, llamaServerArgs, MAX_DERIVED_SLOTS, probeMachines, resolveMachine, resolveMachines, usableMachines } from '../src/engine/backends.ts';
import { applyEvent, emptyView, type EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { initRepo } from '../src/engine/git.ts';
import { OpenAICompatible, samplingFields, type Completion, type Inference } from '../src/engine/inference.ts';
import { buildPack, fixMessages, writeMessages } from '../src/engine/prompts.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { runPlan } from '../src/engine/scheduler.ts';
import { validatePlan } from '../src/tickets/schema.ts';

/** A model that answers after `ms`, or when released by hand. */
function slowModel(ms = 0) {
  let inFlight = 0, peak = 0;
  const gates: Array<() => void> = [];
  const answer = async (): Promise<Completion<string>> => {
    inFlight++;
    peak = Math.max(peak, inFlight);
    if (ms) await Bun.sleep(ms);
    else await new Promise<void>((r) => gates.push(r));
    inFlight--;
    return { value: 'ok', raw: 'ok', promptTokens: 1000, completionTokens: 100, ms: 500 };
  };
  const inf: Inference = { text: answer, complete: answer as any };
  return { inf, peak: () => peak, open: () => gates.splice(0).forEach((g) => g()) };
}

describe('KV budget', () => {
  test('presets match the BlackHole figures (KiB per token at f16, GB per 16K agent at q8_0)', () => {
    expect(kvBytesPerToken(MODEL_PRESETS['qwen3.5-2b'], 'f16')).toBe(12 * 1024);
    expect(kvBytesPerToken(MODEL_PRESETS['minicpm5-2b'], 'f16')).toBe(42 * 1024);
    const gb = (p: string) => kvBytesPerToken(MODEL_PRESETS[p], 'q8_0') * 16384 / 2 ** 30;
    expect(gb('qwen3.5-2b')).toBeCloseTo(0.1, 1);
    expect(gb('minicpm5-2b')).toBeCloseTo(0.35, 1);
    expect(gb('granite-4.2-3b')).toBeCloseTo(0.66, 1);
  });

  test('slots come from memory, capped at MAX_DERIVED_SLOTS; explicit slots win and warn when they overrun', () => {
    const mini8k = resolveMachine({ name: 'laptop', url: 'http://x/v1', preset: 'minicpm5-2b', memory_gb: 5.3, ctx: 8192 });
    expect(mini8k.slotSource).toBe('memory');
    expect(mini8k.slots).toBe(28);
    expect(resolveMachine({ name: 'laptop', url: 'http://x/v1', preset: 'minicpm5-2b', memory_gb: 5.3, ctx: 16384 }).slots).toBe(14);
    const qwen = resolveMachine({ name: 'laptop', url: 'http://x/v1', preset: 'qwen3.5-2b', memory_gb: 5.3, ctx: 8192 });
    expect(qwen.slotsFromMemory).toBeGreaterThan(40);
    expect(qwen.slots).toBe(qwen.slotsFromMemory!);
    const huge = resolveMachine({ name: 'big', url: 'http://x/v1', preset: 'qwen3.5-2b', memory_gb: 500, ctx: 8192 });
    expect(huge.slots).toBe(MAX_DERIVED_SLOTS);
    // a B200's 120 GB pool with Qwen3.8-27B at 16K and an fp8 cache: about a hundred calls
    const b200 = resolveMachine({ name: 'b200', url: 'http://x/v1', preset: 'qwen3.8-27b', cache_type: 'fp8', memory_gb: 120, ctx: 16384 });
    expect(b200.slots).toBeGreaterThan(90);
    expect(b200.slots).toBeLessThan(MAX_DERIVED_SLOTS);
    const over = resolveMachine({ name: 'laptop', url: 'http://x/v1', preset: 'minicpm5-2b', memory_gb: 1, ctx: 16384, slots: 12 });
    expect(over.slots).toBe(12);
    expect(over.warnings[0]).toContain('fits 2');
    expect(resolveMachine({ name: 'a', url: 'http://x/v1' })).toMatchObject({ slots: 12, slotSource: 'default' });
    // a fixed recurrent state per sequence counts too
    const withState = resolveMachine({ name: 'a', url: 'http://x/v1', kv_bytes_per_token: 1000, state_bytes_per_slot: 1e6, ctx: 1000, memory_gb: 0.02 });
    expect(withState.slots).toBe(10);
  });

  test('bad machines are refused by name', () => {
    expect(() => resolveMachine({ name: 'a', url: 'x' })).toThrow(/url/);
    expect(() => resolveMachine({ name: 'a', url: 'http://x/v1', roles: ['boss' as any] })).toThrow(/roles/);
    expect(() => resolveMachine({ name: 'a', url: 'http://x/v1', preset: 'llama-70b' })).toThrow(/preset/);
    expect(() => resolveMachine({ name: 'a', url: 'http://x/v1', sampling: { top_k: 0 } })).toThrow(/top_k/);
    expect(() => resolveMachine({ name: 'a', url: 'http://x/v1', preset: 'granite-4.2-3b', memory_gb: 0.1, ctx: 16384 })).toThrow(/does not fit/);
    expect(() => resolveMachines([{ name: 'a', url: 'http://x/v1' }, { name: 'a', url: 'http://y/v1' }])).toThrow(/twice/);
    expect(() => resolveMachines([{ name: 'mac', url: 'http://x/v1', roles: ['manager'] }])).toThrow(/worker/);
  });

  test('llama-server flags: one slot per call, total context for all of them, finite top_k, the preset min_p', () => {
    const m = resolveMachine({ name: 'laptop', url: 'http://x/v1', preset: 'minicpm5-2b', slots: 15, ctx: 16384 });
    const args = llamaServerArgs(m).join(' ');
    expect(args).toContain('-np 15 -c 245760 --kv-unified');
    expect(args).toContain('-ctk q8_0 -ctv q8_0');
    expect(args).toContain('--top-k 20');
    expect(args).toContain('--min-p 0');
  });
});

describe('sampling', () => {
  test('defaults in one place, preset needs on top, env on top of that; top_k always finite', () => {
    // Qwen3.8's sets: thinking unless the call says it does not
    expect(samplingFields({})).toEqual({ temperature: 1, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0 });
    expect(samplingFields({}, true)).toEqual(samplingFields({}));
    expect(samplingFields({}, false)).toEqual({ temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 0 });
    expect(samplingFromEnv({ MUGGE_PRESENCE_PENALTY: '1.5' })).toEqual({ presence_penalty: 1.5 });
    expect(() => samplingFromEnv({ MUGGE_PRESENCE_PENALTY: '3' })).toThrow();
    expect(samplingFromEnv({ MUGGE_PRESET: 'minicpm5-2b' })).toEqual({ min_p: 0 });
    expect(samplingFromEnv({ MUGGE_PRESET: 'minicpm5-2b', MUGGE_MIN_P: '0.02', MUGGE_TOP_K: '20' })).toEqual({ min_p: 0.02, top_k: 20 });
    expect(samplingFromEnv({})).toBeUndefined();
    expect(() => samplingFromEnv({ MUGGE_TOP_K: '0' })).toThrow(/whole vocabulary/);
    expect(() => samplingFromEnv({ MUGGE_TOP_K: '-1' })).toThrow();
    expect(() => samplingFromEnv({ MUGGE_PRESET: 'nope' })).toThrow();
    expect(samplingFields({ sampling: { temperature: 0.7 }, temperature: 0.3 }, false)).toEqual({ temperature: 0.3, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 0 });
  });

  test('every request carries a finite top_k, and a top_k of 0 is never sent', async () => {
    const bodies: any[] = [];
    const server = Bun.serve({ port: 0, fetch: async (req) => (bodies.push(await req.json()), Response.json({ choices: [{ message: { content: 'ok' } }], usage: {} })) });
    try {
      await new OpenAICompatible({ baseUrl: `${server.url}v1`, defaultModel: 'm' }).text('coder', [{ role: 'user', content: 'hi' }], 10);
      expect(bodies[0]).toMatchObject({ top_k: 20, temperature: 1, top_p: 0.95, min_p: 0 });
      // a call sent with thinking off (the cut retry) gets the non-thinking set
      await new OpenAICompatible({ baseUrl: `${server.url}v1`, defaultModel: 'm', sampling: { min_p: 0.01 } }).text('coder', [{ role: 'user', content: 'hi' }], 10, undefined, { thinking: false });
      expect(bodies[1]).toMatchObject({ top_k: 20, temperature: 0.7, top_p: 0.8, min_p: 0.01, chat_template_kwargs: { enable_thinking: false } });
      const bad = new OpenAICompatible({ baseUrl: `${server.url}v1`, defaultModel: 'm', extra: { top_k: 0 } });
      await expect(bad.text('coder', [{ role: 'user', content: 'hi' }], 10)).rejects.toThrow(/top_k/);
      expect(bodies.length).toBe(2);
    } finally {
      server.stop(true);
    }
  });
});

describe('machines from the environment', () => {
  test('without MUGGE_BACKENDS: one machine with one slot per ticket, as before', () => {
    const [m] = machinesFromEnv({ MUGGE_INFERENCE_URL: 'http://gpu:8080/v1' }, 12);
    expect(m).toMatchObject({ name: 'local', url: 'http://gpu:8080/v1', slots: 12, roles: ['worker', 'lead', 'manager'] });
    expect(machinesFromEnv({ MUGGE_SLOTS: '30' }, 12)[0].slots).toBe(30);
    expect(machinesFromEnv({ MUGGE_PRESET: 'minicpm5-2b', MUGGE_MEMORY_GB: '5.3', MUGGE_CTX: '16384' }, 12)[0].slots).toBe(14);
  });

  test('MUGGE_BACKENDS as JSON or @file, each machine its own client and sampling', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mugge-backends-'));
    try {
      const list = [
        { name: 'laptop', url: 'http://10.0.0.2:8080/v1', roles: ['worker'], preset: 'minicpm5-2b', memory_gb: 5.3, ctx: 16384 },
        { name: 'mac', url: 'http://10.0.0.3:8080/v1', roles: ['manager', 'lead'], slots: 5, ctx: 32768 },
      ];
      writeFileSync(join(dir, 'b.json'), JSON.stringify(list));
      for (const v of [JSON.stringify(list), `@${join(dir, 'b.json')}`]) {
        const ms = machinesFromEnv({ MUGGE_BACKENDS: v });
        expect(ms.map((m) => [m.name, m.slots])).toEqual([['laptop', 14], ['mac', 5]]);
      }
      const pool = poolFromEnv({ MUGGE_BACKENDS: JSON.stringify(list), MUGGE_THINKING: 'off' });
      expect(pool.slotsFor('worker')).toBe(14);
      expect(pool.slotsFor('lead')).toBe(5);
      const laptop = pool.machines[0].client as OpenAICompatible;
      expect(laptop.cfg.baseUrl).toBe('http://10.0.0.2:8080/v1');
      expect(laptop.cfg.sampling).toEqual({ min_p: 0 });
      expect(laptop.cfg.extra).toEqual({ chat_template_kwargs: { enable_thinking: false } });
      expect(() => machinesFromEnv({ MUGGE_BACKENDS: 'not json' })).toThrow(/MUGGE_BACKENDS/);
      // crew-wide settings sit between a preset's needs and a machine's own
      const withEnv = machinesFromEnv({ MUGGE_BACKENDS: JSON.stringify([{ ...list[0], sampling: { top_k: 30 } }, list[1]]), MUGGE_TOP_K: '20', MUGGE_MIN_P: '0.01' });
      expect(withEnv.map((m) => m.sampling)).toEqual([{ min_p: 0.01, top_k: 30 }, { min_p: 0.01, top_k: 20 }]);
      expect(() => machinesFromEnv({ MUGGE_TOP_K: '0' })).toThrow(/MUGGE_TOP_K/);
      expect(llamaServerArgs(machinesFromEnv({ MUGGE_TOP_K: '20' })[0]).join(' ')).toContain('--top-k 20');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('the pool', () => {
  const machine = (name: string, slots: number, roles: any = ['worker']) => resolveMachine({ name, url: `http://${name}/v1`, slots, roles });

  test('never sends more calls than a machine has slots; the rest queue, in order', async () => {
    const m = slowModel();
    const pool = new BackendPool([{ machine: machine('gpu', 2), client: m.inf }]);
    const events: string[] = [];
    const calls = ['a', 'b', 'c', 'd'].map((id) =>
      pool.forAgent({ agent: id, onSlot: (e) => events.push(`${id}:${e.state}`) }).text('coder', [], 10));
    await Bun.sleep(5);
    expect(m.peak()).toBe(2);
    let s = pool.snapshot();
    expect(s.total).toMatchObject({ slots: 2, busy: 2, queued: 2 });
    expect(s.agents.filter((a) => a.state === 'queued').map((a) => a.agent)).toEqual(['c', 'd']);
    expect(s.machines[0].occupancy).toBe(1);
    while (events.filter((e) => e.endsWith('got')).length < 4) {
      m.open();
      await Bun.sleep(1);
    }
    m.open();
    await Promise.all(calls);
    expect(events.filter((e) => e.endsWith('got'))).toEqual(['a:got', 'b:got', 'c:got', 'd:got']);
    s = pool.snapshot();
    expect(s.total).toMatchObject({ busy: 0, queued: 0, calls: 4, completionTokens: 400 });
    expect(s.agents.every((a) => a.state === 'idle' && a.lastCallTokPerSec === 200)).toBe(true);
  });

  test('roles pick machines; an agent keeps to the machine of its last call while it has room', async () => {
    const used: string[] = [];
    const client = (name: string): Inference => {
      const answer = async () => (used.push(name), { value: 'ok', raw: 'ok', promptTokens: 1, completionTokens: 1, ms: 1 });
      return { text: answer, complete: answer as any };
    };
    const pool = new BackendPool([
      { machine: machine('laptop', 4), client: client('laptop') },
      { machine: machine('laptop2', 4), client: client('laptop2') },
      { machine: machine('mac', 2, ['manager', 'lead']), client: client('mac') },
    ]);
    await pool.forAgent({ role: 'manager' }).text('m', [], 1);
    await pool.forAgent({ role: 'lead' }).text('m', [], 1);
    expect(used).toEqual(['mac', 'mac']);
    const t = pool.forAgent({ agent: 't1' });
    await t.text('m', [], 1);
    const first = used.at(-1)!;
    for (let i = 0; i < 3; i++) await t.text('m', [], 1);
    expect(used.slice(-3)).toEqual([first, first, first]);
    expect(() => new BackendPool([{ machine: machine('gpu', 1), client: client('gpu') }]).forAgent({ role: 'lead' })).toThrow(/no backend serves lead/);
  });

  test('an abort while queued gives the slot to nobody and leaves the agent idle', async () => {
    const m = slowModel();
    const pool = new BackendPool([{ machine: machine('gpu', 1), client: m.inf }]);
    const ac = new AbortController();
    const a = pool.forAgent({ agent: 'a' }).text('c', [], 1);
    const b = pool.forAgent({ agent: 'b', signal: ac.signal }).text('c', [], 1);
    await Bun.sleep(2);
    ac.abort();
    await expect(b).rejects.toThrow('aborted');
    expect(pool.snapshot().agents.find((x) => x.agent === 'b')!.state).toBe('idle');
    m.open();
    await a;
    expect(pool.snapshot().total).toMatchObject({ busy: 0, queued: 0, calls: 1 });
  });

  test('a listener or slot hook that throws never keeps the slot', async () => {
    const m = slowModel(1);
    const pool = new BackendPool([{ machine: machine('gpu', 1), client: m.inf }]);
    pool.subscribe(() => { throw new Error('status.json: ENOSPC'); });
    const err = console.error;
    console.error = () => {};
    try {
      await expect(pool.forAgent({ agent: 'a', onSlot: (e) => { if (e.state === 'got') throw new Error('hook'); } }).text('c', [], 1)).rejects.toThrow('hook');
      // the one slot is free again: the next call gets it instead of queueing forever
      await pool.forAgent({ agent: 'b' }).text('c', [], 1);
    } finally {
      console.error = err;
    }
    expect(pool.snapshot().total).toMatchObject({ busy: 0, queued: 0, calls: 1 });
  });

  test('an abort ends a call already sent, and its slot is free at once', async () => {
    // a server that never answers
    const server = Bun.serve({ port: 0, fetch: () => new Promise<Response>(() => {}) });
    try {
      const client = new OpenAICompatible({ baseUrl: `http://127.0.0.1:${server.port}/v1`, defaultModel: 'm', timeoutMs: 60_000 });
      const pool = new BackendPool([{ machine: machine('gpu', 1), client }]);
      const ac = new AbortController();
      const call = pool.forAgent({ agent: 'a', signal: ac.signal }).text('c', [], 10);
      await Bun.sleep(50);
      expect(pool.snapshot().total.busy).toBe(1);
      const t0 = Date.now();
      ac.abort();
      await expect(call).rejects.toThrow();
      expect(Date.now() - t0).toBeLessThan(1000);
      expect(pool.snapshot().total).toMatchObject({ busy: 0, queued: 0 });
    } finally {
      server.stop(true);
    }
  });

  test('tok/s per agent, team and machine over the last minute', async () => {
    const m = slowModel(1);
    const pool = new BackendPool([{ machine: machine('gpu', 4), client: m.inf }]);
    await Promise.all([
      pool.forAgent({ agent: 'a1', team: 'a' }).text('c', [], 1),
      pool.forAgent({ agent: 'a2', team: 'a' }).text('c', [], 1),
      pool.forAgent({ agent: 'b1', team: 'b' }).text('c', [], 1),
    ]);
    // a minute after the start the window is a full minute
    const s = pool.snapshot(Date.now() + 59_990);
    expect(s.windowS).toBeCloseTo(60, 0);
    expect(s.total.tokPerSec).toBe(5);
    expect(s.total.promptTokPerSec).toBe(50);
    expect(s.teams.map((t) => [t.team, t.agents, t.completionTokens])).toEqual([['a', 2, 200], ['b', 1, 100]]);
    expect(s.machines[0]).toMatchObject({ name: 'gpu', calls: 3, completionTokens: 300 });
    pool.retire('a1');
    const later = pool.snapshot(Date.now() + 120_000);
    expect(later.total.tokPerSec).toBe(0);
    expect(later.agents.map((a) => a.agent)).toEqual(['a2', 'b1']);
    expect(later.teams.find((t) => t.team === 'a')!.completionTokens).toBe(200);
  });
});

describe('runPlan over a pool', () => {
  let dir: string;
  beforeAll(() => (dir = mkdtempSync(join(tmpdir(), 'mugge-pool-run-'))));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test('more agents than slots: calls queue, every ticket still finishes, slots never overrun', async () => {
    const repo = join(dir, 'repo');
    mkdirSync(repo, { recursive: true });
    writeFileSync(join(repo, 'README.md'), 'scaffold\n');
    const base = initRepo(repo);
    const ids = ['a', 'b', 'c', 'd', 'e', 'f'];
    const plan = validatePlan({ tickets: ids.map((id) => ({ id, title: id, image: 'node', owns: [`${id}.txt`], acceptance: [`grep -q ok-${id} ${id}.txt`] })) });
    const fake = startFakeModel({ solutions: Object.fromEntries(ids.map((id) => [id, { files: { [`${id}.txt`]: `ok-${id}\n` }, note: id }])), latencyMs: 60 });
    try {
      const pool = new BackendPool([{ machine: resolveMachine({ name: 'gpu', url: fake.url, slots: 2 }), client: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }) }]);
      const events: EngineEvent[] = [];
      let peakBusy = 0;
      pool.subscribe((s) => (peakBusy = Math.max(peakBusy, s.total.busy)));
      const view = emptyView();
      const report = await runPlan({
        repo, base, plan, pool, sandbox: new LocalSandbox(), concurrency: 6, worktreeRoot: join(dir, 'wt'), integrate: false,
        emit: (e) => (events.push(e), applyEvent(view, e)),
      });
      expect(report.summary.done).toBe(6);
      expect(fake.stats().peakInFlight).toBe(2);
      expect(peakBusy).toBe(2);
      expect(events.find((e) => e.type === 'plan')).toMatchObject({ concurrency: 6, slots: 2 });
      expect(events.filter((e) => e.type === 'slot' && e.state === 'queued').length).toBeGreaterThan(0);
      expect(events.filter((e) => e.type === 'slot' && e.state === 'got').every((e: any) => e.machine === 'gpu')).toBe(true);
      expect(pool.snapshot().agents).toEqual([]);
      expect(pool.snapshot().teams.map((t) => t.team)).toEqual(['coder']);
    } finally {
      fake.stop();
    }
  });

  test('a queued slot event shows as the queued state, a got one as running again', () => {
    const v = emptyView();
    applyEvent(v, { type: 'plan', at: 0, concurrency: 2, tickets: [{ id: 'a', title: 'a', depends_on: [], model: 'coder', image: 'node' }] });
    applyEvent(v, { type: 'start', at: 1, ticket: 'a', attempt: 1, worktree: '/x' });
    applyEvent(v, { type: 'slot', at: 2, ticket: 'a', state: 'queued' });
    expect(v.tickets[0].state).toBe('queued');
    expect(v.running).toBe(1);
    applyEvent(v, { type: 'slot', at: 3, ticket: 'a', state: 'got', machine: 'gpu' });
    expect(v.tickets[0].state).toBe('running');
  });
});

test('a fix call repeats the write call byte for byte up to the owned files, so the server can reuse its prompt cache', () => {
  const dir = mkdtempSync(join(tmpdir(), 'mugge-prefix-'));
  try {
    mkdirSync(join(dir, 'src'));
    writeFileSync(join(dir, 'api.h'), 'int add(int, int);\n');
    writeFileSync(join(dir, 'src', 'util.c'), 'int one(void) { return 1; }\n');
    const plan = validatePlan({ interfaces: ['api.h'], tickets: [{ id: 't', title: 'add', image: 'c', owns: ['add.c'], reads: ['src/*.c'], acceptance: ['make test'], context: 'Write add.' }] });
    const pack = buildPack(dir, plan, plan.tickets[0], 'notes');
    const write = writeMessages(dir, pack);
    writeFileSync(join(dir, 'add.c'), 'int add(int a, int b) { return a - b; }\n');
    const fix = fixMessages(dir, pack, { command: 'make test', output: 'expected 3, got -1' });
    expect(fix[0]).toEqual(write[0]);
    const owned = write[1].content.indexOf('## Your files as they are now');
    expect(owned).toBeGreaterThan(0);
    const stable = write[1].content.slice(0, owned);
    expect(stable).toContain('int one(void)');
    expect(fix[1].content.startsWith(stable)).toBe(true);
    // what changes (the owned files, then the failure) comes after the stable part
    expect(fix[1].content.indexOf('return a - b')).toBeGreaterThan(owned);
    expect(fix[1].content.indexOf('expected 3, got -1')).toBeGreaterThan(fix[1].content.indexOf('return a - b'));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe('remote and cloud machines', () => {
  test('a key per machine goes out as a bearer header; a timeout per machine reaches its client', async () => {
    const seen: Array<string | null> = [];
    const server = Bun.serve({
      port: 0,
      fetch: (req) => {
        seen.push(req.headers.get('authorization'));
        if (req.headers.get('authorization') !== 'Bearer s3cret') return new Response('no', { status: 401 });
        return new URL(req.url).pathname.endsWith('/models')
          ? Response.json({ data: [] })
          : Response.json({ choices: [{ finish_reason: 'stop', message: { content: 'ok' } }], usage: { prompt_tokens: 1, completion_tokens: 1 } });
      },
    });
    try {
      const url = `http://127.0.0.1:${server.port}/v1`;
      const machines = resolveMachines([{ name: 'b200', url, api_key_env: 'EVROC_KEY', timeout_s: 3600, roles: ['worker'] }]);
      const env = { EVROC_KEY: 's3cret' };
      expect(await probeMachines(machines, env)).toMatchObject([{ name: 'b200', ok: true, detail: 'reachable' }]);
      expect((await probeMachines(machines, { EVROC_KEY: 'wrong' }))[0]).toMatchObject({ ok: false, detail: 'the API key was refused (401)' });
      expect((await probeMachines(machines, {}))[0]).toMatchObject({ ok: false, kind: 'config', detail: 'EVROC_KEY is not set' });
      // a key that is missing or refused stops the run even on an optional machine: it is a mistake, not a laptop asleep
      const opt = resolveMachines([{ name: 'b200', url, api_key_env: 'EVROC_KEY', optional: true }, { name: 'w', url, api_key_env: 'EVROC_KEY' }]);
      const wrongKey = [(await probeMachines(opt, { EVROC_KEY: 'wrong' }))[0]!, { name: 'w', url, optional: false, ok: true, detail: 'reachable', ms: 1 }];
      expect(describeProbe(wrongKey[0]!)).not.toContain('left out');
      expect(() => usableMachines(opt, wrongKey)).toThrow(/b200 \(the API key was refused \(401\)\) is set up wrong/);
      // the pool refuses a key variable that is not set rather than sending no key
      expect(() => BackendPool.fromMachines(machines, { baseUrl: url, defaultModel: 'm' }, { EVROC_KEY: ' ' })).toThrow(/b200: EVROC_KEY is not set/);
      const pool = BackendPool.fromMachines(machines, { baseUrl: url, defaultModel: 'm' }, env);
      expect((pool.machines[0]!.client as OpenAICompatible).cfg.timeoutMs).toBe(3_600_000);
      await pool.forAgent({}).text('m', [], 1);
      expect(seen.at(-1)).toBe('Bearer s3cret');
    } finally {
      server.stop(true);
    }
  });

  test('an unreachable optional machine is left out; an unreachable required one stops the run', async () => {
    const server = Bun.serve({ port: 0, fetch: () => Response.json({ data: [] }) });
    // a port nothing listens on
    const dead = Bun.serve({ port: 0, fetch: () => new Response('') });
    const deadUrl = `http://127.0.0.1:${dead.port}/v1`;
    dead.stop(true);
    try {
      const up = `http://127.0.0.1:${server.port}/v1`;
      const machines = resolveMachines([
        { name: 'laptop', url: up, roles: ['worker'] },
        { name: 'wifi-mac', url: deadUrl, roles: ['worker', 'lead'], optional: true },
      ]);
      const probes = await probeMachines(machines, {}, 2000);
      expect(probes.map((p) => [p.name, p.ok, p.kind])).toEqual([['laptop', true, undefined], ['wifi-mac', false, 'unreachable']]);
      expect(describeProbe(probes[1]!)).toContain('(optional: left out)');
      expect(usableMachines(machines, probes).map((m) => m.name)).toEqual(['laptop']);
      const required = resolveMachines([{ name: 'laptop', url: up }, { name: 'aws-l40s', url: deadUrl }]);
      expect(() => usableMachines(required, [probes[0]!, { ...probes[1]!, name: 'aws-l40s', optional: false }])).toThrow(/aws-l40s .*cannot be reached/);
      const onlyOptional = resolveMachines([{ name: 'wifi-mac', url: deadUrl, optional: true }]);
      expect(() => usableMachines(onlyOptional, [probes[1]!])).toThrow(/no reachable backend serves worker calls/);
      expect(() => resolveMachine({ name: 'x', url: up, optional: 'yes' as any })).toThrow(/optional must be true or false/);
      // the roles a command needs: mugge plan needs a manager
      const withMgr = resolveMachines([{ name: 'laptop', url: up, roles: ['worker'] }, { name: 'wifi-mac', url: deadUrl, roles: ['manager'], optional: true }]);
      expect(() => usableMachines(withMgr, [probes[0]!, { ...probes[1]! }], ['manager'])).toThrow(/no reachable backend serves manager calls/);
    } finally {
      server.stop(true);
    }
  });

  test('a url with no API behind it (404) and a gateway with no GPU behind it (5xx) are not "up"', async () => {
    const server = Bun.serve({ port: 0, fetch: (req) => new URL(req.url).pathname.startsWith('/gw') ? new Response('bad gateway', { status: 502 }) : new Response('not here', { status: 404 }) });
    try {
      const root = `http://127.0.0.1:${server.port}`;
      const machines = resolveMachines([{ name: 'b200', url: root }, { name: 'l40s', url: `${root}/gw/v1`, optional: true }, { name: 'auth', url: `http://user:pw@127.0.0.1:${server.port}/v1` }]);
      const probes = await probeMachines(machines, {});
      expect(probes.map((p) => [p.name, p.ok, p.kind])).toEqual([['b200', false, 'config'], ['l40s', false, 'unreachable'], ['auth', false, 'config']]);
      expect(probes[0]!.detail).toMatch(/404: the url must be the API root, ending in \/v1/);
      expect(describeProbe(probes[1]!)).toContain('answered 502');
      expect(describeProbe(probes[2]!)).not.toContain('pw');
      expect(() => usableMachines(machines, probes)).toThrow(/b200 .*set up wrong/);
    } finally {
      server.stop(true);
    }
  });

  test('a machine whose call fails is passed over by retries while another can take them', async () => {
    const used: string[] = [];
    const ok = async () => ({ value: 'ok', raw: 'ok', promptTokens: 1, completionTokens: 1, ms: 1 });
    const dead: Inference = { text: async () => { used.push('dead'); throw new Error('Unable to connect'); }, complete: (async () => { used.push('dead'); throw new Error('Unable to connect'); }) as any };
    const live: Inference = { text: async () => (used.push('live'), ok()), complete: (async () => (used.push('live'), ok())) as any };
    const m = (name: string) => resolveMachine({ name, url: `http://${name}/v1`, slots: 4 });
    // the asleep laptop is listed first, so with equal load it is picked first
    const pool = new BackendPool([{ machine: m('laptop'), client: dead }, { machine: m('gpu'), client: live }]);
    for (const id of ['t0', 't1', 't2']) {
      const a = pool.forAgent({ agent: id });
      for (let i = 0; i < 3; i++) await a.text('m', [], 1).catch(() => {});
    }
    // one failure, then every later call goes to the machine that answers
    expect(used).toEqual(['dead', 'live', 'live', 'live', 'live', 'live', 'live', 'live', 'live']);
  });
});
