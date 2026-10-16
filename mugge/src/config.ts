/**
 * Engine settings from the environment, so the same binary runs against the fake model in CI,
 * vLLM on the project VM, or a hosted endpoint later.
 *
 *   MUGGE_INFERENCE_URL   OpenAI-compatible base URL (default http://127.0.0.1:8000/v1, vLLM)
 *   MUGGE_MODEL           served model for names not in MUGGE_MODELS (default: mugge-small)
 *   MUGGE_MODELS          JSON map ticket model → served name, e.g. {"coder-ts":"coder-ts-lora"}
 *   MUGGE_GUIDED          response_format | guided_json | none
 *   MUGGE_API_KEY         bearer token, for hosted endpoints
 *   MUGGE_TIMEOUT_S       most seconds one model call may take once sent (default 1800; waiting
 *                         for a slot does not count, and Bun's own 300 s fetch cut is off)
 *   MUGGE_CONCURRENCY     tickets at once, i.e. agents working (default 12)
 *   MUGGE_TEMPERATURE, MUGGE_TOP_K, MUGGE_TOP_P, MUGGE_MIN_P
 *                         sampling for every call (defaults: engine/inference.ts DEFAULT_SAMPLING;
 *                         top_k must be a positive integer, never 0)
 *   MUGGE_PRESET          model preset (engine/backends.ts MODEL_PRESETS, e.g. minicpm5-2b): its
 *                         sampling needs (min_p 0) and KV cost, for the single-URL machine
 *
 * Machines (engine/backends.ts). Without MUGGE_BACKENDS the crew is one machine at
 * MUGGE_INFERENCE_URL serving every role, with MUGGE_SLOTS slots, else slots from
 * MUGGE_MEMORY_GB and the preset's KV cost at MUGGE_CTX tokens per call, else MUGGE_CONCURRENCY:
 * the engine behaves as it always has.
 *   MUGGE_BACKENDS        JSON array of machines, or @path/to/backends.json, e.g.
 *                         [{"name":"laptop","url":"http://10.0.0.2:8080/v1","roles":["worker"],
 *                           "preset":"qwen3.5-2b","memory_gb":5.3,"ctx":12288},
 *                          {"name":"mac","url":"http://10.0.0.3:8080/v1","roles":["manager","lead"],"slots":5,"optional":true},
 *                          {"name":"evroc-b200","url":"https://b200.example/v1","roles":["worker"],"slots":40,
 *                           "api_key_env":"EVROC_KEY","timeout_s":3600}]
 *                         api_key_env names the variable holding that machine's key (sent as a bearer
 *                         header); timeout_s overrides MUGGE_TIMEOUT_S for that machine; an optional
 *                         machine that does not answer at start is left out (`mugge run` and
 *                         `mugge engine serve` check every machine first; `mugge backends` lists them)
 *   MUGGE_SLOTS, MUGGE_MEMORY_GB, MUGGE_CTX
 *                         the single-URL machine's slots, working-memory budget and context per call
 *   MUGGE_SANDBOX         local | podman | docker (see engine/sandbox.ts)
 *   MUGGE_THINKING        on | off: sent as chat_template_kwargs.enable_thinking on every call
 *                         (unset: the server's default). The bench's --thinking flag is the same switch.
 *   MUGGE_MAX_CALL_TOKENS most tokens one model call may write, whatever the ticket's budget leaves
 *                         (unset: the budget's remainder, 60K by default, which a thinking model can
 *                         spend in one call)
 *   MUGGE_CUT_RETRY       nothink | same (default nothink): once a thinking call of a ticket runs out
 *                         of tokens (cut with no usable file, or forced), that ticket's later calls go
 *                         out with thinking off; same keeps thinking on. Read by the harness itself.
 *   MUGGE_DEBUG_CALLS     1: one line per model call on stderr: model, prompt, completion and
 *                         reasoning tokens, time, and whether the answer was cut at max_tokens
 *   MUGGE_REVIEW          on | off (default on): one lead review of each ticket whose checks pass
 *                         (engine/review.ts); read by the harness itself
 */
import { readFileSync } from 'node:fs';
import { BackendPool, describeProbe, MODEL_PRESETS, PROBE_TIMEOUT_MS, probeMachines, resolveMachine, resolveMachines, usableMachines, type CallRole, type Machine, type MachineInput } from './engine/backends.ts';
import { checkTopK, type InferenceConfig, type Sampling } from './engine/inference.ts';

function num(env: NodeJS.ProcessEnv, key: string, check: (n: number) => boolean, what: string): number | undefined {
  const v = env[key]?.trim();
  if (!v) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || !check(n)) throw new Error(`${key} must be ${what}`);
  return n;
}

/** MUGGE_TEMPERATURE / TOP_K / TOP_P / MIN_P, on top of the MUGGE_PRESET's needs. */
export function samplingFromEnv(env: NodeJS.ProcessEnv = process.env): Sampling | undefined {
  const preset = env.MUGGE_PRESET?.trim().toLowerCase();
  if (preset && !MODEL_PRESETS[preset]) throw new Error(`MUGGE_PRESET must be one of ${Object.keys(MODEL_PRESETS).join(', ')}`);
  const topK = num(env, 'MUGGE_TOP_K', () => true, 'a positive integer');
  if (topK !== undefined) checkTopK(topK, 'MUGGE_TOP_K');
  const s: Sampling = {
    ...(preset ? MODEL_PRESETS[preset].sampling : {}),
    ...defined({
      temperature: num(env, 'MUGGE_TEMPERATURE', (n) => n >= 0, 'a number ≥ 0'),
      top_k: topK,
      top_p: num(env, 'MUGGE_TOP_P', (n) => n > 0 && n <= 1, 'a number in (0, 1]'),
      min_p: num(env, 'MUGGE_MIN_P', (n) => n >= 0 && n < 1, 'a number in [0, 1)'),
    }),
  };
  return Object.keys(s).length ? s : undefined;
}

function defined<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

export function inferenceFromEnv(env: NodeJS.ProcessEnv = process.env): InferenceConfig {
  let models: Record<string, string> | undefined;
  if (env.MUGGE_MODELS) {
    try {
      models = JSON.parse(env.MUGGE_MODELS);
    } catch {
      throw new Error('MUGGE_MODELS must be a JSON object');
    }
  }
  const guided = (env.MUGGE_GUIDED as InferenceConfig['guided']) || 'response_format';
  if (!['response_format', 'guided_json', 'none'].includes(guided)) throw new Error('MUGGE_GUIDED must be response_format, guided_json or none');
  const thinking = env.MUGGE_THINKING?.trim().toLowerCase();
  if (thinking && !['on', 'off'].includes(thinking)) throw new Error('MUGGE_THINKING must be on or off');
  const maxCall = env.MUGGE_MAX_CALL_TOKENS ? Number(env.MUGGE_MAX_CALL_TOKENS) : undefined;
  if (maxCall !== undefined && !(Number.isInteger(maxCall) && maxCall > 0)) throw new Error('MUGGE_MAX_CALL_TOKENS must be a positive integer');
  return {
    baseUrl: env.MUGGE_INFERENCE_URL || 'http://127.0.0.1:8000/v1',
    defaultModel: env.MUGGE_MODEL || 'mugge-small',
    models,
    guided,
    apiKey: env.MUGGE_API_KEY || undefined,
    timeoutMs: Number(env.MUGGE_TIMEOUT_S) > 0 ? Number(env.MUGGE_TIMEOUT_S) * 1000 : undefined,
    sampling: samplingFromEnv(env),
    extra: thinking ? { chat_template_kwargs: { enable_thinking: thinking === 'on' } } : undefined,
    maxCallTokens: maxCall,
    debug: env.MUGGE_DEBUG_CALLS === '1',
  };
}

export function concurrencyFromEnv(env: NodeJS.ProcessEnv = process.env, fallback = 12): number {
  const n = Number(env.MUGGE_CONCURRENCY);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}

/** The crew's machines: MUGGE_BACKENDS, or the single MUGGE_INFERENCE_URL machine with `concurrency` slots unless told otherwise. */
export function machinesFromEnv(env: NodeJS.ProcessEnv = process.env, concurrency = concurrencyFromEnv(env)): Machine[] {
  const raw = env.MUGGE_BACKENDS?.trim();
  const sampling = samplingFromEnv(env) ?? {};
  if (raw) {
    let list: MachineInput[];
    try {
      list = JSON.parse(raw.startsWith('@') ? readFileSync(raw.slice(1), 'utf8') : raw);
    } catch (e) {
      throw new Error(`MUGGE_BACKENDS must be a JSON array or @file: ${(e as Error).message}`);
    }
    return resolveMachines(list, sampling);
  }
  const slots = num(env, 'MUGGE_SLOTS', (n) => Number.isInteger(n) && n > 0, 'a positive integer');
  const memory = num(env, 'MUGGE_MEMORY_GB', (n) => n > 0, 'a positive number');
  const ctx = num(env, 'MUGGE_CTX', (n) => Number.isInteger(n) && n > 0, 'a positive integer');
  const preset = env.MUGGE_PRESET?.trim().toLowerCase() || undefined;
  return [resolveMachine({
    name: 'local', url: env.MUGGE_INFERENCE_URL || 'http://127.0.0.1:8000/v1', preset, ctx, memory_gb: memory,
    // nothing about memory said: one slot per ticket, as before
    slots: slots ?? (memory === undefined ? concurrency : undefined),
  }, sampling)];
}

/** The machines with a client each, sharing MUGGE_MODELS, thinking and the call cap; sampling is per machine (preset, then MUGGE_*, then its own). */
export function poolFromEnv(env: NodeJS.ProcessEnv = process.env, concurrency = concurrencyFromEnv(env)): BackendPool {
  return BackendPool.fromMachines(machinesFromEnv(env, concurrency), inferenceFromEnv(env), env);
}

/**
 * poolFromEnv after a start check: says which machines answer (one line each to `say`), leaves
 * out unreachable optional ones and throws if a required one cannot be reached.
 */
export async function checkedPoolFromEnv(env: NodeJS.ProcessEnv = process.env, concurrency = concurrencyFromEnv(env), say: (line: string) => void = (l) => console.error(l), needed: CallRole[] = ['worker']): Promise<BackendPool> {
  const machines = machinesFromEnv(env, concurrency);
  const base = inferenceFromEnv(env);
  const probes = await probeMachines(machines, env, PROBE_TIMEOUT_MS, base.apiKey);
  for (const p of probes) say(describeProbe(p));
  return BackendPool.fromMachines(usableMachines(machines, probes, needed), base, env);
}
