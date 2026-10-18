/**
 * The machines a crew runs on. Each machine is one OpenAI-compatible URL (llama-server with
 * `-np N --kv-unified`, vLLM, SGLang, a hosted service later) serving one shared copy of the
 * model to many one-shot calls at once, so the 8 GB laptop can serve workers while the Mac
 * serves the manager and leads.
 *
 * A slot is one call in flight on a machine. Its slot count comes from, in order: an explicit
 * `slots`; else the memory the machine has for working memory divided by what one call needs
 * (KV bytes per token × context per call + any fixed recurrent state), at most MAX_DERIVED_SLOTS;
 * else DEFAULT_SLOTS. Calls past a machine's slots wait in the pool ("queued") rather than in
 * the server, so a call's timeout only starts once it is sent.
 *
 * The pool also keeps per-agent, per-team and per-machine tok/s and slot occupancy as plain
 * data (`snapshot()`, `subscribe()`), for the run log and status stream to read.
 */
import { DEFAULT_SAMPLING, OpenAICompatible, checkEffort, checkTopK, effortsFromEnv, effortTweaks, type Effort, type ChatMessage, type Completion, type CallTweaks, type Inference, type InferenceConfig, type Sampling } from './inference.ts';

/** Which kind of call a machine serves. Worker = a ticket's write and fix calls. */
export const CALL_ROLES = ['worker', 'lead', 'manager'] as const;
export type CallRole = (typeof CALL_ROLES)[number];

export const DEFAULT_SLOTS = 12;
/** Slots a memory budget may give one machine: a B200 serving Qwen3.8-27B holds about 100 calls at 16K. */
export const MAX_DERIVED_SLOTS = 256;
/** Least ms between two serving snapshots sent to subscribers (the first after a quiet spell goes at once). */
export const NOTIFY_MS = 100;
/** Tokens one worker call needs with thinking off: a pack of up to ~10K plus a 1–2K answer. */
export const DEFAULT_CTX_PER_CALL = 12_288;

/** Bytes per element of llama.cpp's KV cache types (`-ctk` / `-ctv`), and vLLM's fp8 (`--kv-cache-dtype fp8`). */
export const CACHE_TYPE_BYTES = { f32: 4, f16: 2, bf16: 2, fp8: 1, q8_0: 34 / 32, q5_1: 24 / 32, q5_0: 22 / 32, q4_1: 20 / 32, q4_0: 18 / 32, iq4_nl: 18 / 32 } as const;
export type CacheType = keyof typeof CACHE_TYPE_BYTES;

export interface ModelPreset {
  /** Layers that keep a KV cache (full attention only, for hybrid models). */
  kvLayers: number;
  kvHeads: number;
  headDim: number;
  /** Sampling this model needs whatever the engine default is. */
  sampling?: Sampling;
  /** Fixed bytes per sequence beside the KV cache (a hybrid model's recurrent state); a machine's state_bytes_per_slot wins. */
  stateBytesPerSlot?: number;
  note: string;
}

/** The demo model (Qwen3.8-27B), the BlackHole base candidates and the Centipede run 1 reference, from their configs. */
export const MODEL_PRESETS: Record<string, ModelPreset> = {
  // 64 layers, 1 in 4 full attention (4 KV heads × 256); the other 48 are Gated DeltaNet with a
  // 48 × 128 × 128 state each at fp32, plus conv state: 153,944,064 B a sequence, an estimate
  // until a real vLLM run reports its numbers
  'qwen3.8-27b': { kvLayers: 16, kvHeads: 4, headDim: 256, stateBytesPerSlot: 153_944_064, note: 'Qwen/Qwen3.8-27B-FP8, the demo model: hybrid, 16 of 64 layers keep a KV cache (64 KiB a token at bf16, 32 KiB at fp8); recurrent state per sequence estimated' },
  'qwen3.5-2b': { kvLayers: 6, kvHeads: 2, headDim: 256, note: 'hybrid: 6 of 24 layers keep a KV cache; the rest hold a fixed recurrent state per sequence (set state_bytes_per_slot from footprint.sh)' },
  'minicpm5-2b': { kvLayers: 42, kvHeads: 2, headDim: 128, sampling: { min_p: 0 }, note: 'plain attention; min_p 0 (llama.cpp default 0.05 makes it repeat)' },
  'granite-4.2-3b': { kvLayers: 40, kvHeads: 8, headDim: 64, note: 'every layer keeps a KV cache; Centipede run 1 base' },
};

/** KV bytes one token costs: K and V, every KV layer, at the cache type. */
export function kvBytesPerToken(p: Pick<ModelPreset, 'kvLayers' | 'kvHeads' | 'headDim'>, cache: CacheType = 'q8_0'): number {
  return 2 * p.kvLayers * p.kvHeads * p.headDim * CACHE_TYPE_BYTES[cache];
}

/** One machine as written in MUGGE_BACKENDS (snake_case, like plan JSON). */
export interface MachineInput {
  name: string;
  /** Base URL up to and including `/v1`. */
  url: string;
  /** Kinds of call it serves (default all). */
  roles?: CallRole[];
  /** Explicit slot count; wins over the memory budget. */
  slots?: number;
  /** Memory left for working memory (KV cache + recurrent state) after weights and compute buffers. */
  memory_gb?: number;
  /** A MODEL_PRESETS key: gives KV bytes per token and any sampling the model needs. */
  preset?: string;
  /** KV bytes per token, when the model has no preset (overrides the preset's). */
  kv_bytes_per_token?: number;
  /** Fixed bytes per sequence beside the KV cache (hybrid models' recurrent state). */
  state_bytes_per_slot?: number;
  /** KV cache type the server runs with (default q8_0). */
  cache_type?: CacheType;
  /** Context one call needs (default DEFAULT_CTX_PER_CALL). */
  ctx?: number;
  /** Served model name for calls not mapped by MUGGE_MODELS (default MUGGE_MODEL). */
  model?: string;
  /** Name of the environment variable holding this machine's API key, sent as `Authorization: Bearer` (keys stay out of the JSON). */
  api_key_env?: string;
  sampling?: Sampling;
  /** Most seconds one call may take on this machine once sent (a far cloud GPU may need more than the crew default). */
  timeout_s?: number;
  /** How hard calls on this machine think: xhigh | medium | low | off; wins over MUGGE_REASONING_EFFORT for them. */
  reasoning_effort?: string;
  /** What serves it: llama-server (default) gets a matching llama-server line in `mugge backends`; vllm and sglang do not. */
  server?: 'llama-server' | 'vllm' | 'sglang';
  /** A machine the crew can do without (a laptop on wifi that may be asleep): unreachable at start, it is left out instead of failing the run. */
  optional?: boolean;
}

export interface Machine {
  name: string;
  url: string;
  roles: CallRole[];
  slots: number;
  /** Where `slots` came from. */
  slotSource: 'explicit' | 'memory' | 'default';
  ctxPerCall: number;
  cacheType: CacheType;
  preset?: string;
  kvBytesPerToken?: number;
  stateBytesPerSlot: number;
  /** Working memory one call costs: KV for ctxPerCall tokens plus the fixed state; unknown without a KV figure. */
  bytesPerSlot?: number;
  memoryBytes?: number;
  /** Slots the memory budget allows (before the MAX_DERIVED_SLOTS cap). */
  slotsFromMemory?: number;
  model?: string;
  apiKeyEnv?: string;
  sampling: Sampling;
  timeoutMs?: number;
  optional: boolean;
  server: 'llama-server' | 'vllm' | 'sglang';
  reasoningEffort?: Effort;
  /** Things worth a look, e.g. explicit slots that overrun the memory budget. */
  warnings: string[];
}

const NAME = /^[a-z0-9][a-z0-9._-]{0,31}$/i;

function positive(v: unknown, what: string, integer = false): number | undefined {
  if (v === undefined) return undefined;
  if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0 || (integer && !Number.isInteger(v))) throw new Error(`${what} must be a positive ${integer ? 'integer' : 'number'}`);
  return v;
}

/**
 * Validates one machine and works out its slots. Sampling: the preset's needs, then `base`
 * (the crew-wide MUGGE_* settings), then the machine's own.
 */
export function resolveMachine(raw: MachineInput, base: Sampling = {}): Machine {
  const where = `backend ${typeof raw?.name === 'string' ? raw.name : '?'}`;
  if (typeof raw?.name !== 'string' || !NAME.test(raw.name)) throw new Error(`${where}: name must be 1–32 letters, digits, . _ -`);
  if (typeof raw.url !== 'string' || !/^https?:\/\//.test(raw.url)) throw new Error(`${where}: url must be an http(s) URL ending in /v1`);
  const roles = raw.roles ?? [...CALL_ROLES];
  if (!Array.isArray(roles) || !roles.length || roles.some((r) => !CALL_ROLES.includes(r))) throw new Error(`${where}: roles must be some of ${CALL_ROLES.join(', ')}`);
  const preset = raw.preset === undefined ? undefined : MODEL_PRESETS[raw.preset.toLowerCase()];
  if (raw.preset !== undefined && !preset) throw new Error(`${where}: unknown preset ${raw.preset} (known: ${Object.keys(MODEL_PRESETS).join(', ')})`);
  const cacheType = raw.cache_type ?? 'q8_0';
  if (!(cacheType in CACHE_TYPE_BYTES)) throw new Error(`${where}: cache_type must be one of ${Object.keys(CACHE_TYPE_BYTES).join(', ')}`);
  const slots = positive(raw.slots, `${where}: slots`, true);
  const memoryGb = positive(raw.memory_gb, `${where}: memory_gb`);
  const ctxPerCall = positive(raw.ctx, `${where}: ctx`, true) ?? DEFAULT_CTX_PER_CALL;
  const kv = positive(raw.kv_bytes_per_token, `${where}: kv_bytes_per_token`) ?? (preset ? kvBytesPerToken(preset, cacheType) : undefined);
  const state = raw.state_bytes_per_slot === undefined ? (preset?.stateBytesPerSlot ?? 0) : positive(raw.state_bytes_per_slot, `${where}: state_bytes_per_slot`)!;
  const timeoutS = positive(raw.timeout_s, `${where}: timeout_s`);
  if (raw.optional !== undefined && typeof raw.optional !== 'boolean') throw new Error(`${where}: optional must be true or false`);
  if (raw.server !== undefined && !['llama-server', 'vllm', 'sglang'].includes(raw.server)) throw new Error(`${where}: server must be llama-server, vllm or sglang`);
  if (raw.api_key_env !== undefined && (typeof raw.api_key_env !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(raw.api_key_env))) throw new Error(`${where}: api_key_env must name an environment variable`);
  const sampling: Sampling = { ...preset?.sampling, ...base, ...raw.sampling };
  if (sampling.top_k !== undefined) checkTopK(sampling.top_k, `${where}: sampling.top_k`);

  const warnings: string[] = [];
  const bytesPerSlot = kv !== undefined ? Math.ceil(kv * ctxPerCall + state) : undefined;
  const memoryBytes = memoryGb !== undefined ? Math.floor(memoryGb * 1e9) : undefined;
  const slotsFromMemory = bytesPerSlot && memoryBytes !== undefined ? Math.floor(memoryBytes / bytesPerSlot) : undefined;
  if (memoryBytes !== undefined && bytesPerSlot === undefined) warnings.push('memory_gb is set but the KV cost per token is unknown: give a preset or kv_bytes_per_token');
  if (slotsFromMemory === 0) throw new Error(`${where}: ${memoryGb} GB does not fit one ${ctxPerCall}-token call (${fmtBytes(bytesPerSlot!)} each)`);
  let n: number, slotSource: Machine['slotSource'];
  if (slots !== undefined) {
    n = slots;
    slotSource = 'explicit';
    if (slotsFromMemory !== undefined && slots > slotsFromMemory) warnings.push(`${slots} slots need ${fmtBytes(slots * bytesPerSlot!)} but the budget is ${fmtBytes(memoryBytes!)} (fits ${slotsFromMemory})`);
  } else if (slotsFromMemory !== undefined) {
    n = Math.min(slotsFromMemory, MAX_DERIVED_SLOTS);
    slotSource = 'memory';
  } else {
    n = DEFAULT_SLOTS;
    slotSource = 'default';
  }
  return {
    name: raw.name, url: raw.url, roles, slots: n, slotSource, ctxPerCall, cacheType,
    preset: raw.preset?.toLowerCase(), kvBytesPerToken: kv, stateBytesPerSlot: state, bytesPerSlot, memoryBytes, slotsFromMemory,
    model: raw.model, apiKeyEnv: raw.api_key_env, sampling, timeoutMs: timeoutS !== undefined ? timeoutS * 1000 : undefined, optional: raw.optional ?? false, server: raw.server ?? 'llama-server',
    reasoningEffort: raw.reasoning_effort === undefined ? undefined : checkEffort(raw.reasoning_effort, `${where}: reasoning_effort`), warnings,
  };
}

/** Whether a machine answered at start, and if not, why. */
export interface MachineProbe {
  name: string;
  url: string;
  optional: boolean;
  ok: boolean;
  /**
   * Why a machine is down: `unreachable` (no answer, or a gateway with nothing behind it) may be
   * a laptop asleep, so an optional machine is left out; `config` (a key that is not set or is
   * refused, a URL with no API behind it) is a mistake that stops the run even for an optional one.
   */
  kind?: 'unreachable' | 'config';
  /** "reachable", or what went wrong (no answer, a refused key, a key variable that is not set). */
  detail: string;
  ms: number;
}

/** Most milliseconds the start check waits for one machine. */
export const PROBE_TIMEOUT_MS = 5000;

/**
 * Asks every machine for GET /models, with its API key when it has one, all at once. A 2xx/3xx
 * answer counts as reachable. No answer or a 5xx (a cloud gateway in front of a stopped GPU, a
 * server still loading) is unreachable; a refused key, an api_key_env that is not set, or a 404
 * (the url is not the API root ending in /v1, so every call would 404) is a config mistake.
 */
export async function probeMachines(machines: Machine[], env: NodeJS.ProcessEnv = process.env, timeoutMs = PROBE_TIMEOUT_MS, crewKey?: string): Promise<MachineProbe[]> {
  return Promise.all(machines.map(async (m): Promise<MachineProbe> => {
    const started = Date.now();
    const done = (ok: boolean, detail: string, kind?: MachineProbe['kind']): MachineProbe => ({ name: m.name, url: m.url, optional: m.optional, ok, ...(kind ? { kind } : {}), detail, ms: Date.now() - started });
    // the crew-wide key (MUGGE_API_KEY) for a machine without its own, as fromMachines sends it
    const key = m.apiKeyEnv ? env[m.apiKeyEnv]?.trim() : crewKey;
    if (m.apiKeyEnv && !key) return done(false, `${m.apiKeyEnv} is not set`, 'config');
    try {
      const r = await fetch(`${m.url.replace(/\/$/, '')}/models`, { headers: key ? { authorization: `Bearer ${key}` } : {}, signal: AbortSignal.timeout(timeoutMs) });
      await r.body?.cancel().catch(() => {});
      if (r.status === 401 || r.status === 403) return done(false, `the API key was refused (${r.status})`, 'config');
      if (r.status === 404) return done(false, 'GET /models answered 404: the url must be the API root, ending in /v1', 'config');
      if (r.status >= 500) return done(false, `answered ${r.status}${r.status === 503 ? ' (still loading the model?)' : ' (a gateway with no model server behind it?)'}`, 'unreachable');
      if (r.status >= 400) return done(false, `GET /models answered ${r.status}`, 'config');
      return done(true, 'reachable');
    } catch (e) {
      return done(false, `no answer: ${(e as Error).message ?? e}`, 'unreachable');
    }
  }));
}

/** One line per probe, for the start of a run and `mugge backends`. */
export function describeProbe(p: MachineProbe): string {
  return `${p.ok ? 'up  ' : 'DOWN'}  ${p.name}  ${hideUserinfo(p.url)}  ${p.detail}${p.ok ? ` (${p.ms} ms)` : ''}${!p.ok && p.optional && p.kind !== 'config' ? ' (optional: left out)' : ''}`;
}

/** A URL as it may be printed: no user:password@. */
export function hideUserinfo(url: string): string {
  return url.replace(/^(https?:\/\/)[^/@]*@/, '$1***@');
}

/**
 * The machines a run starts with: every reachable one. An unreachable optional machine is left
 * out; an unreachable required one, a misconfigured one (optional or not), or no reachable
 * machine serving one of the `needed` roles stops the run.
 */
export function usableMachines(machines: Machine[], probes: MachineProbe[], needed: CallRole[] = ['worker']): Machine[] {
  const down = new Set(probes.filter((p) => !p.ok).map((p) => p.name));
  const wrong = probes.filter((p) => !p.ok && p.kind === 'config');
  if (wrong.length) throw new Error(`backend ${wrong.map((p) => `${p.name} (${p.detail})`).join(', ')} is set up wrong; fix its entry in MUGGE_BACKENDS`);
  const required = probes.filter((p) => !p.ok && !p.optional);
  if (required.length) throw new Error(`backend ${required.map((p) => `${p.name} (${p.detail})`).join(', ')} cannot be reached; mark a machine "optional": true to run without it`);
  const up = machines.filter((m) => !down.has(m.name));
  for (const role of needed) if (!up.some((m) => m.roles.includes(role))) throw new Error(`no reachable backend serves ${role} calls`);
  return up;
}

/** Validates a crew's machines: unique names, and every kind of call served somewhere. */
export function resolveMachines(raw: MachineInput[], base: Sampling = {}): Machine[] {
  if (!Array.isArray(raw) || !raw.length) throw new Error('MUGGE_BACKENDS must be a non-empty JSON array of machines');
  const machines = raw.map((m) => resolveMachine(m, base));
  const names = new Set<string>();
  for (const m of machines) {
    if (names.has(m.name)) throw new Error(`backend ${m.name} is listed twice`);
    names.add(m.name);
  }
  if (!machines.some((m) => m.roles.includes('worker'))) throw new Error('no backend serves worker calls');
  return machines;
}

/** The llama-server flags that match a machine: one slot per call, one shared unified KV cache. */
export function llamaServerArgs(m: Machine): string[] {
  const top_k = m.sampling.top_k ?? DEFAULT_SAMPLING.top_k;
  return ['-np', String(m.slots), '-c', String(m.slots * m.ctxPerCall), '--kv-unified', '-fa', 'on', '-ctk', m.cacheType, '-ctv', m.cacheType, '--top-k', String(top_k),
    ...(m.sampling.min_p !== undefined ? ['--min-p', String(m.sampling.min_p)] : [])];
}

export function fmtBytes(n: number): string {
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`;
}

/** One line per machine, for `mugge backends`. */
export function describeMachine(m: Machine): string {
  const mem = m.bytesPerSlot !== undefined ? `, ${fmtBytes(m.bytesPerSlot)}/slot at ${m.ctxPerCall} ctx ${m.cacheType}` : '';
  const budget = m.memoryBytes !== undefined ? ` of ${fmtBytes(m.memoryBytes)} (fits ${m.slotsFromMemory})` : '';
  return `${m.name}  ${hideUserinfo(m.url)}  ${m.roles.join('+')}  ${m.slots} slots (${m.slotSource})${mem}${budget}${m.optional ? ', optional' : ''}${m.apiKeyEnv ? `, key from ${m.apiKeyEnv}` : ''}${m.timeoutMs ? `, ${m.timeoutMs / 1000} s timeout` : ''}`;
}

// ---- the pool --------------------------------------------------------------------------------

/** What one agent (a ticket in flight) is doing with the model. */
export type AgentSlotState = 'idle' | 'queued' | 'calling';

export interface MachineStats {
  name: string;
  url: string;
  roles: CallRole[];
  slots: number;
  busy: number;
  /** busy / slots. */
  occupancy: number;
  /** Calls waiting that this machine could take. */
  queued: number;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  /** Completion tokens per second over the last window. */
  tokPerSec: number;
  /** Prompt tokens per second over the last window (prefill). */
  promptTokPerSec: number;
  bytesPerSlot?: number;
  memoryBytes?: number;
}

export interface AgentStats {
  /** The ticket id (or the role, for manager and lead calls without a ticket). */
  agent: string;
  team: string;
  role: CallRole;
  state: AgentSlotState;
  machine?: string;
  /** Since the current wait began, when queued. */
  queuedMs?: number;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  tokPerSec: number;
  /** completion tokens / seconds of the last call (prefill included). */
  lastCallTokPerSec?: number;
}

export interface TeamStats {
  team: string;
  agents: number;
  calling: number;
  queued: number;
  completionTokens: number;
  tokPerSec: number;
}

export interface ServingSnapshot {
  at: number;
  /** Seconds the tok/s figures are averaged over (shorter just after start). */
  windowS: number;
  machines: MachineStats[];
  agents: AgentStats[];
  teams: TeamStats[];
  total: { slots: number; busy: number; queued: number; calls: number; completionTokens: number; tokPerSec: number; promptTokPerSec: number };
}

export interface CallOptions {
  /** Ticket id; also the agent's name in the stats. */
  agent?: string;
  team?: string;
  role?: CallRole;
  signal?: AbortSignal;
  /** Told when the agent starts waiting for a slot and when it gets one. */
  onSlot?: (e: { state: 'queued' | 'got'; machine?: string; waitedMs?: number }) => void;
}

interface Slotted {
  machine: Machine;
  client: Inference;
  busy: number;
  /** When its last call failed (cleared by the next one that answers). */
  failedAt?: number;
}

/** How long a machine whose call failed is passed over while another can take the call. */
export const FAILED_MS = 30_000;

interface Waiter {
  role: CallRole;
  resolve: (m: Slotted) => void;
  reject: (e: Error) => void;
}

interface CallRecord {
  end: number;
  machine: string;
  agent: string;
  team: string;
  promptTokens: number;
  completionTokens: number;
}

interface AgentRecord {
  agent: string;
  team: string;
  role: CallRole;
  state: AgentSlotState;
  machine?: string;
  since: number;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  lastCallTokPerSec?: number;
}

export const STATS_WINDOW_MS = 60_000;

export class BackendPool {
  readonly machines: Slotted[];
  private waiters: Waiter[] = [];
  private records: CallRecord[] = [];
  private agents = new Map<string, AgentRecord>();
  private totals = new Map<string, { calls: number; promptTokens: number; completionTokens: number }>();
  private teamTokens = new Map<string, number>();
  private listeners = new Set<(s: ServingSnapshot) => void>();
  private readonly started = Date.now();
  /** MUGGE_REASONING_EFFORT's per-role levels; a machine's own reasoning_effort wins for its calls. */
  roleEffort: Partial<Record<CallRole, Effort>> = {};

  constructor(machines: Array<{ machine: Machine; client: Inference }>) {
    if (!machines.length) throw new Error('a pool needs at least one machine');
    this.machines = machines.map((m) => ({ ...m, busy: 0 }));
    for (const m of machines) this.totals.set(m.machine.name, { calls: 0, promptTokens: 0, completionTokens: 0 });
  }

  /** One OpenAI-compatible client per machine; `base` gives the shared settings (models, thinking, cap). */
  static fromMachines(machines: Machine[], base: InferenceConfig, env: NodeJS.ProcessEnv = process.env): BackendPool {
    // a key variable that is not set would send no key at all and come back as a bare 401
    for (const m of machines) if (m.apiKeyEnv && !env[m.apiKeyEnv]?.trim()) throw new Error(`backend ${m.name}: ${m.apiKeyEnv} is not set`);
    const pool = new BackendPool(machines.map((m) => ({
      machine: m,
      client: new OpenAICompatible({
        ...base,
        baseUrl: m.url,
        defaultModel: m.model ?? base.defaultModel,
        apiKey: m.apiKeyEnv ? env[m.apiKeyEnv]!.trim() : base.apiKey,
        sampling: m.sampling,
        timeoutMs: m.timeoutMs ?? base.timeoutMs,
      }),
    })));
    pool.roleEffort = effortsFromEnv(env).roles;
    return pool;
  }

  /** The effort a call of `role` goes out with on machine `m`: the machine's, else the role's; undefined leaves the config alone. */
  effortFor(role: CallRole, m?: Machine): Effort | undefined {
    return m?.reasoningEffort ?? this.roleEffort[role];
  }

  /**
   * Whether calls of `role` think, for the state agents show: off when every machine serving it
   * (or the role's level) says off, else the client's switch (MUGGE_THINKING, on by default).
   */
  thinksFor(role: CallRole, clientThinks: (c: Inference) => boolean): boolean {
    const serving = this.machines.filter((m) => m.machine.roles.includes(role));
    if (!serving.length) return false;
    if (serving.every((m) => this.effortFor(role, m.machine) === 'off')) return false;
    if (serving.some((m) => { const e = this.effortFor(role, m.machine); return e !== undefined && e !== 'off'; })) return true;
    return clientThinks(serving[0]!.client);
  }

  /** The single-URL crew: one machine serving every role with `slots` slots (`url` is only shown). */
  static single(client: Inference, slots: number, url = ''): BackendPool {
    const machine = resolveMachine({ name: 'local', url: 'http://local/v1', slots });
    return new BackendPool([{ machine: { ...machine, url }, client }]);
  }

  get slots(): number {
    return this.machines.reduce((n, m) => n + m.machine.slots, 0);
  }

  /** Slots on machines that serve `role`. */
  slotsFor(role: CallRole): number {
    return this.machines.filter((m) => m.machine.roles.includes(role)).reduce((n, m) => n + m.machine.slots, 0);
  }

  /** An Inference for one agent: every call takes a slot on a machine serving its role, preferring the machine of its last call (its pack may still be in that server's prompt cache). */
  forAgent(o: CallOptions = {}): Inference {
    const role = o.role ?? 'worker';
    if (!this.machines.some((m) => m.machine.roles.includes(role))) throw new Error(`no backend serves ${role} calls`);
    const agent = o.agent ?? role;
    const team = o.team ?? (role === 'worker' ? 'crew' : role);
    let last: string | undefined;
    // the role's effort (or its machine's) under each call's own tweaks: a cut retry's thinking off wins
    const tweaked = (m: Slotted, t?: CallTweaks): CallTweaks | undefined => {
      const e = effortTweaks(this.effortFor(role, m.machine));
      if (!Object.keys(e).length) return t;
      return t?.thinkingIsDefault ? { ...t, ...e } : { ...e, ...t };
    };
    const call = async <T>(run: (c: Inference, m: Slotted) => Promise<Completion<T>>): Promise<Completion<T>> => {
      const rec = this.agent(agent, team, role);
      const waitStart = Date.now();
      let m = this.tryTake(role, last);
      if (!m) {
        rec.state = 'queued';
        rec.since = waitStart;
        o.onSlot?.({ state: 'queued' });
        this.changed();
        try {
          m = await this.wait(role, o.signal);
        } catch (e) {
          rec.state = 'idle';
          this.changed();
          throw e;
        }
      }
      rec.state = 'calling';
      rec.machine = m.machine.name;
      rec.since = Date.now();
      // from here the slot is ours: whatever throws, even a listener, must give it back
      try {
        o.onSlot?.({ state: 'got', machine: m.machine.name, waitedMs: Date.now() - waitStart });
        this.changed();
        const c = await run(m.client, m);
        this.record(m.machine.name, rec, c);
        // the next call prefers this machine (prompt cache), but only once it has answered
        last = m.machine.name;
        m.failedAt = undefined;
        return c;
      } catch (e) {
        // a machine that fails a call (asleep, gone, 500s) frees its slot at once and would look
        // like the least busy one: retries go elsewhere for a while
        if (!o.signal?.aborted) m.failedAt = Date.now();
        if (last === m.machine.name) last = undefined;
        throw e;
      } finally {
        rec.state = 'idle';
        rec.since = Date.now();
        this.release(m);
      }
    };
    return {
      complete: <T>(model: string, messages: ChatMessage[], schema: object, maxTokens: number, signal?: AbortSignal, tweaks?: CallTweaks) => call<T>((c, m) => c.complete<T>(model, messages, schema, maxTokens, signal ?? o.signal, tweaked(m, tweaks))),
      text: (model: string, messages: ChatMessage[], maxTokens: number, signal?: AbortSignal, tweaks?: CallTweaks) => call<string>((c, m) => c.text(model, messages, maxTokens, signal ?? o.signal, tweaked(m, tweaks))),
    };
  }

  /** Forgets an agent's row once its ticket has ended (its tokens stay in the machine and team totals). */
  retire(agent: string): void {
    if (this.agents.delete(agent)) this.changed();
  }

  snapshot(now = Date.now()): ServingSnapshot {
    this.prune(now);
    const windowS = Math.max(1, Math.min(STATS_WINDOW_MS, now - this.started) / 1000);
    const sum = (rs: CallRecord[], k: 'promptTokens' | 'completionTokens') => rs.reduce((n, r) => n + r[k], 0);
    const rate = (rs: CallRecord[], k: 'promptTokens' | 'completionTokens' = 'completionTokens') => round1(sum(rs, k) / windowS);
    const waitingFor = (m: Slotted) => this.waiters.filter((w) => m.machine.roles.includes(w.role)).length;
    // one pass over the window's records, grouped: a snapshot goes out on every slot change, and
    // filtering the records once per agent cost O(agents × records) each time at crew scale
    const by = (key: 'machine' | 'agent' | 'team') => {
      const g = new Map<string, CallRecord[]>();
      for (const r of this.records) {
        const k = r[key] as string;
        (g.get(k) ?? g.set(k, []).get(k)!).push(r);
      }
      return (k: string) => g.get(k) ?? [];
    };
    const ofMachine = by('machine'), ofAgent = by('agent'), ofTeam = by('team');
    const machines: MachineStats[] = this.machines.map((m) => {
      const rs = ofMachine(m.machine.name);
      const t = this.totals.get(m.machine.name)!;
      return {
        name: m.machine.name, url: m.machine.url, roles: m.machine.roles, slots: m.machine.slots, busy: m.busy, occupancy: round1(m.busy / m.machine.slots * 100) / 100,
        queued: waitingFor(m), ...t, tokPerSec: rate(rs), promptTokPerSec: rate(rs, 'promptTokens'),
        bytesPerSlot: m.machine.bytesPerSlot, memoryBytes: m.machine.memoryBytes,
      };
    });
    const agents: AgentStats[] = [...this.agents.values()].map((a) => ({
      agent: a.agent, team: a.team, role: a.role, state: a.state, machine: a.machine,
      ...(a.state === 'queued' ? { queuedMs: now - a.since } : {}),
      calls: a.calls, promptTokens: a.promptTokens, completionTokens: a.completionTokens,
      tokPerSec: rate(ofAgent(a.agent)), lastCallTokPerSec: a.lastCallTokPerSec,
    }));
    const teamNames = [...new Set([...agents.map((a) => a.team), ...this.teamTokens.keys()])].sort();
    const teams: TeamStats[] = teamNames.map((team) => {
      const as = agents.filter((a) => a.team === team);
      return {
        team, agents: as.length, calling: as.filter((a) => a.state === 'calling').length, queued: as.filter((a) => a.state === 'queued').length,
        completionTokens: this.teamTokens.get(team) ?? 0, tokPerSec: rate(ofTeam(team)),
      };
    });
    return {
      at: now, windowS, machines, agents, teams,
      total: {
        slots: this.slots, busy: this.machines.reduce((n, m) => n + m.busy, 0), queued: this.waiters.length,
        calls: machines.reduce((n, m) => n + m.calls, 0), completionTokens: machines.reduce((n, m) => n + m.completionTokens, 0),
        tokPerSec: rate(this.records), promptTokPerSec: rate(this.records, 'promptTokens'),
      },
    };
  }

  /** Called with a fresh snapshot whenever a slot is taken or freed or an agent queues; returns an unsubscribe. */
  subscribe(fn: (s: ServingSnapshot) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private agent(agent: string, team: string, role: CallRole): AgentRecord {
    let a = this.agents.get(agent);
    if (!a) this.agents.set(agent, (a = { agent, team, role, state: 'idle', since: Date.now(), calls: 0, promptTokens: 0, completionTokens: 0 }));
    return a;
  }

  private record(machine: string, a: AgentRecord, c: Completion<unknown>) {
    const t = this.totals.get(machine)!;
    t.calls++;
    t.promptTokens += c.promptTokens;
    t.completionTokens += c.completionTokens;
    a.calls++;
    a.promptTokens += c.promptTokens;
    a.completionTokens += c.completionTokens;
    this.teamTokens.set(a.team, (this.teamTokens.get(a.team) ?? 0) + c.completionTokens);
    if (c.ms > 0) a.lastCallTokPerSec = round1(c.completionTokens / (c.ms / 1000));
    this.records.push({ end: Date.now(), machine, agent: a.agent, team: a.team, promptTokens: c.promptTokens, completionTokens: c.completionTokens });
  }

  /**
   * A free slot on a machine serving `role`: the preferred machine if it has one, else the least
   * occupied, leaving out machines whose last call failed in the past FAILED_MS while another is free.
   */
  private tryTake(role: CallRole, prefer?: string): Slotted | null {
    const all = this.machines.filter((m) => m.machine.roles.includes(role) && m.busy < m.machine.slots);
    const now = Date.now();
    const healthy = all.filter((m) => m.failedAt === undefined || now - m.failedAt > FAILED_MS);
    const free = healthy.length ? healthy : all;
    if (!free.length) return null;
    const m = free.find((x) => x.machine.name === prefer) ?? free.reduce((a, b) => (b.busy / b.machine.slots < a.busy / a.machine.slots ? b : a));
    m.busy++;
    return m;
  }

  private wait(role: CallRole, signal?: AbortSignal): Promise<Slotted> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(new Error('aborted'));
      const w: Waiter = { role, resolve, reject };
      const onAbort = () => {
        const i = this.waiters.indexOf(w);
        if (i >= 0) this.waiters.splice(i, 1);
        reject(new Error('aborted'));
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      w.resolve = (m) => {
        signal?.removeEventListener('abort', onAbort);
        resolve(m);
      };
      this.waiters.push(w);
    });
  }

  /** Frees a slot and hands it to the first waiter (in arrival order) that the machine serves. */
  private release(m: Slotted) {
    m.busy--;
    // a slot freed by a machine that just failed is not handed on while a healthy machine serves
    // the waiter: that machine frees a slot when its call ends
    const failing = (x: Slotted) => x.failedAt !== undefined && Date.now() - x.failedAt <= FAILED_MS;
    const i = this.waiters.findIndex((w) => m.machine.roles.includes(w.role)
      && !(failing(m) && this.machines.some((x) => x !== m && x.machine.roles.includes(w.role) && !failing(x))));
    if (i >= 0) {
      const [w] = this.waiters.splice(i, 1);
      m.busy++;
      w.resolve(m);
    }
    this.changed();
  }

  private prune(now: number) {
    const cut = now - STATS_WINDOW_MS;
    let i = 0;
    while (i < this.records.length && this.records[i].end < cut) i++;
    if (i) this.records.splice(0, i);
  }

  /**
   * Tells the listeners; one that throws (a status file that cannot be written) never reaches a
   * call or a slot. The first change after a quiet spell goes out at once; changes within
   * NOTIFY_MS of it go out as one trailing snapshot, so a crew of 128 agents taking and freeing
   * slots builds a few snapshots a second, not hundreds.
   */
  private lastNotify = 0;
  private notifyTimer: ReturnType<typeof setTimeout> | null = null;
  private changed() {
    if (!this.listeners.size || this.notifyTimer) return;
    const wait = this.lastNotify + NOTIFY_MS - Date.now();
    if (wait <= 0) return this.notify();
    this.notifyTimer = setTimeout(() => {
      this.notifyTimer = null;
      this.notify();
    }, wait);
    (this.notifyTimer as { unref?: () => void }).unref?.();
  }

  private notify() {
    this.lastNotify = Date.now();
    const s = this.snapshot();
    for (const fn of this.listeners) {
      try {
        fn(s);
      } catch (e) {
        console.error(`serving listener failed: ${(e as Error).message ?? e}`);
      }
    }
  }
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}
