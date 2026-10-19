/**
 * Client for any OpenAI-compatible chat endpoint: vLLM or SGLang on the project VM, or a
 * hosted service later. Planner calls are constrained to a JSON schema so every answer parses;
 * coder calls are plain text (files.ts), since code escaped inside JSON comes out worse.
 *
 * Specialists are LoRA adapters served under their own model names by vLLM
 * (`--lora-modules coder-ts=/models/adapters/coder-ts`), so routing is only a name lookup.
 */
import { inlineThinkingChars, stripThinking } from './files.ts';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface Completion<T> {
  value: T;
  raw: string;
  promptTokens: number;
  completionTokens: number;
  /** Tokens the model spent thinking, when the server reports them (vLLM, llama-server). */
  reasoningTokens?: number;
  /** The answer stopped at max_tokens, so it is probably incomplete. */
  cut?: boolean;
  /**
   * The answer was forced: the server (MuggeBench's proxy in stages/eval.py) closed a thinking
   * that ran out of tokens or stopped inside it and asked for the answer on top. The value says
   * why ("length", "stop", "no_reasoning"). A forced answer must never become a training row.
   */
  forced?: string;
  /** max_tokens the call was sent with, after MUGGE_MAX_CALL_TOKENS. */
  maxTokens?: number;
  /** Characters of thinking: what the server returned beside the answer (reasoning_content), else what came inside content. */
  reasoningChars?: number;
  /** The thinking text itself, from reasoning_content or from inside content, when there was any. */
  reasoning?: string;
  /** chat_template_kwargs.enable_thinking the call was sent with, when it was sent. */
  thinking?: boolean;
  ms: number;
}

/** What one call may change from the config. */
export interface CallTweaks {
  /** Sent as chat_template_kwargs.enable_thinking for this call only (the harness turns it off after a thinking call ran out of tokens). */
  thinking?: boolean;
  /** Sent as chat_template_kwargs.reasoning_effort for this call only (a role's effort, backends.ts). */
  reasoningEffort?: ReasoningEffort;
  /** `thinking` is only the caller's default (the lead review's off): a role's or machine's effort wins over it. */
  thinkingIsDefault?: boolean;
  /** A planning call (design, scaffold, tickets): capped by planMaxCallTokens, not maxCallTokens. */
  planning?: boolean;
}

/** MUGGE_THINKING_BUDGET: thinking tokens per reasoning_effort level, and for any level (`all`). */
export interface ThinkingBudget {
  all?: number;
  levels: Partial<Record<ReasoningEffort, number>>;
}

/**
 * MUGGE_THINKING_BUDGET: one number for every thinking call (`2048`), numbers per level
 * (`low=1024,medium=4096,xhigh=8192`), or both (`2048,xhigh=8192`).
 */
export function thinkingBudgetFromEnv(env: Record<string, string | undefined> = process.env): ThinkingBudget | undefined {
  const raw = env.MUGGE_THINKING_BUDGET?.trim();
  if (!raw) return undefined;
  const out: ThinkingBudget = { levels: {} };
  const count = (v: string, where: string) => {
    const n = Number(v);
    if (!Number.isInteger(n) || n < 0) throw new Error(`${where} must be a whole number of tokens, 0 or more (got ${JSON.stringify(v)})`);
    return n;
  };
  for (const part of raw.split(',').map((x) => x.trim()).filter(Boolean)) {
    const [k, v] = part.includes('=') ? part.split('=', 2).map((x) => x.trim().toLowerCase()) : [undefined, part];
    if (k === undefined) out.all = count(v!, 'MUGGE_THINKING_BUDGET');
    else if ((REASONING_EFFORTS as readonly string[]).includes(k)) out.levels[k as ReasoningEffort] = count(v!, `MUGGE_THINKING_BUDGET ${k}`);
    else throw new Error(`MUGGE_THINKING_BUDGET: ${k} is not xhigh, medium or low`);
  }
  return out;
}

/** Qwen3.8's reasoning_effort levels. */
export const REASONING_EFFORTS = ['xhigh', 'medium', 'low'] as const;
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];
/** How hard one kind of call thinks: a reasoning_effort level, or off (no thinking at all). */
export type Effort = ReasoningEffort | 'off';
export type CallRoleName = 'worker' | 'lead' | 'manager';

/** The call settings an effort means. */
export function effortTweaks(e: Effort | undefined): CallTweaks {
  if (!e) return {};
  return e === 'off' ? { thinking: false } : { thinking: true, reasoningEffort: e };
}

export function checkEffort(v: unknown, where: string): Effort {
  if (typeof v === 'string' && ([...REASONING_EFFORTS, 'off'] as string[]).includes(v.trim().toLowerCase())) return v.trim().toLowerCase() as Effort;
  throw new Error(`${where} must be xhigh, medium, low or off (got ${JSON.stringify(v)})`);
}

/**
 * MUGGE_REASONING_EFFORT: one level for every call (`medium`), or levels per kind of call
 * (`worker=medium,lead=xhigh,manager=xhigh`), or both (`low,manager=xhigh`). Levels are xhigh,
 * medium, low, or off (that kind of call does not think).
 */
export function effortsFromEnv(env: Record<string, string | undefined> = process.env): { all?: Effort; roles: Partial<Record<CallRoleName, Effort>> } {
  const raw = env.MUGGE_REASONING_EFFORT?.trim();
  const out: { all?: Effort; roles: Partial<Record<CallRoleName, Effort>> } = { roles: {} };
  if (!raw) return out;
  for (const part of raw.split(',').map((x) => x.trim()).filter(Boolean)) {
    const [k, v] = part.includes('=') ? part.split('=', 2).map((x) => x.trim().toLowerCase()) : [undefined, part];
    if (k === undefined) out.all = checkEffort(v, 'MUGGE_REASONING_EFFORT');
    else if (k === 'worker' || k === 'lead' || k === 'manager') out.roles[k] = checkEffort(v, `MUGGE_REASONING_EFFORT ${k}`);
    else throw new Error(`MUGGE_REASONING_EFFORT: ${k} is not worker, lead or manager`);
  }
  return out;
}

/**
 * Sampling fields sent on every call. Every engine call goes out with a finite top_k: a top_k
 * of 0 (or none, which vLLM and SGLang read as the whole vocabulary) makes llama.cpp sort the
 * full vocabulary on the CPU for every token of every slot, and it was one of the three bugs
 * that scored Centipede run 1's agent 0%. Fields left undefined are not sent, so the server's
 * own default applies (llama-server's min_p is 0.05; MiniCPM5-2B needs 0, see backends.ts).
 */
export interface Sampling {
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  presence_penalty?: number;
}

/**
 * The engine's sampling defaults, the one place they live: Qwen3.8's own recommendation for
 * each mode (model card of Qwen/Qwen3.8-27B-FP8), picked per call by whether that call thinks.
 * One change from the card: presence_penalty stays 0 with thinking off (the card says 1.5 for
 * chat), because a worker rewrites whole files and code repeats its own tokens on purpose.
 */
export const THINKING_SAMPLING: Readonly<Required<Sampling>> = { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0, presence_penalty: 0 };
export const NOTHINK_SAMPLING: Readonly<Required<Sampling>> = { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0, presence_penalty: 0 };
/** The set a call gets when nothing says whether it thinks (Qwen3.8 thinks unless told not to). */
export const DEFAULT_SAMPLING = THINKING_SAMPLING;

/** The default set for a call: THINKING_SAMPLING unless the call goes out with thinking off. */
export function defaultSampling(thinking: boolean | undefined): Readonly<Required<Sampling>> {
  return thinking === false ? NOTHINK_SAMPLING : THINKING_SAMPLING;
}

/** Default for InferenceConfig.timeoutMs: long enough for a 16K prefill on a busy laptop GPU. */
export const DEFAULT_TIMEOUT_MS = 1_800_000;

/** Default for InferenceConfig.serverWaitMs: a vLLM restart (load, compile cache warm, cudagraphs) fits well inside it. */
export const DEFAULT_SERVER_WAIT_MS = 1_200_000;
/** Pauses between sends to a server that is not answering, the last one repeating. */
export let SERVER_RETRY_MS = [2_000, 5_000, 10_000, 20_000];
export const setServerRetryMs = (ms: number[]) => { SERVER_RETRY_MS = ms; };
/** A 500 is tried this many more times (it can be one request's own fault, unlike a server that is down). */
export const MAX_500_RETRIES = 2;

/** Throws unless top_k is a positive integer: finite, so llama.cpp never sorts the whole vocabulary. */
export function checkTopK(topK: unknown, where = 'top_k'): number {
  if (typeof topK !== 'number' || !Number.isInteger(topK) || topK <= 0) {
    throw new Error(`${where} must be a positive integer (got ${JSON.stringify(topK)}): 0 or none means the whole vocabulary`);
  }
  return topK;
}

/** The sampling fields of a request: the mode's defaults, then `cfg.sampling`, then `cfg.temperature`. */
export function samplingFields(cfg: Pick<InferenceConfig, 'sampling' | 'temperature'>, thinking?: boolean): Record<string, number> {
  const s: Sampling = { ...defaultSampling(thinking), ...dropUndefined(cfg.sampling ?? {}), ...(cfg.temperature !== undefined ? { temperature: cfg.temperature } : {}) };
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(s)) if (v !== undefined) out[k] = v;
  return out;
}

function dropUndefined<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

export interface InferenceConfig {
  /** Base URL up to and including `/v1`, e.g. http://127.0.0.1:8000/v1. */
  baseUrl: string;
  apiKey?: string;
  /** Ticket model name → served model name. Names not listed fall back to `defaultModel`. */
  models?: Record<string, string>;
  defaultModel: string;
  /**
   * How to ask for constrained JSON. `response_format` is the OpenAI shape that vLLM, SGLang
   * and most hosts accept; `guided_json` is vLLM's older extra field; `none` relies on the prompt.
   */
  guided?: 'response_format' | 'guided_json' | 'none';
  /** Overrides `sampling.temperature` (kept for callers that set only this). */
  temperature?: number;
  /** temperature, top_k, top_p, min_p, presence_penalty; unset fields take the mode's defaults (THINKING_SAMPLING / NOTHINK_SAMPLING). */
  sampling?: Sampling;
  /** Extra request fields sent as they are, e.g. seed, chat_template_kwargs; a top_k here must still be finite. */
  extra?: Record<string, unknown>;
  /**
   * Most milliseconds one call may take once it is sent (default DEFAULT_TIMEOUT_MS). Time spent
   * waiting for a slot (backends.ts) does not count. Bun's own 300 s fetch cut is always off.
   */
  timeoutMs?: number;
  /** Most tokens one call may write, however much the caller offers (MUGGE_MAX_CALL_TOKENS). */
  maxCallTokens?: number;
  /** The same for planning calls, whose answers grow with the crew (MUGGE_PLAN_MAX_CALL_TOKENS; unset: no cap). */
  planMaxCallTokens?: number;
  /**
   * Most tokens a thinking call may think, by its reasoning_effort level (`all` for any level),
   * sent as thinking_budget_tokens: llama-server then ends the thinking itself and the answer
   * follows within max_tokens (MUGGE_THINKING_BUDGET). For models whose chat template ignores
   * reasoning_effort (Qwen3.5), this is what makes the levels differ.
   */
  thinkingBudget?: ThinkingBudget;
  /** One line per call on stderr (MUGGE_DEBUG_CALLS=1). */
  debug?: boolean;
  /**
   * Most milliseconds one call waits for a server that is down, restarting or answering 502/503/504
   * before it fails (MUGGE_SERVER_WAIT_S; default DEFAULT_SERVER_WAIT_MS). Until then the call is
   * sent again with a growing pause, so a restart does not spend the tickets' attempts.
   */
  serverWaitMs?: number;
}

export interface Inference {
  /** An answer constrained to `schema`, parsed. `signal` aborts the call, even once it is sent. */
  complete<T>(model: string, messages: ChatMessage[], schema: object, maxTokens: number, signal?: AbortSignal, opts?: CallTweaks): Promise<Completion<T>>;
  /** A free-text answer. */
  text(model: string, messages: ChatMessage[], maxTokens: number, signal?: AbortSignal, opts?: CallTweaks): Promise<Completion<string>>;
}

export class OpenAICompatible implements Inference {
  constructor(readonly cfg: InferenceConfig) {}

  served(model: string): string {
    return this.cfg.models?.[model] ?? this.cfg.models?.[model.split('@')[0]] ?? this.cfg.defaultModel;
  }

  async complete<T>(model: string, messages: ChatMessage[], schema: object, maxTokens: number, signal?: AbortSignal, opts?: CallTweaks): Promise<Completion<T>> {
    const c = await this.send(model, messages, maxTokens, signal, schema, opts);
    try {
      return { ...c, value: parseJsonAnswer<T>(c.raw) };
    } catch (e) {
      // the caller can still count what the answer cost and see whether it was cut
      throw Object.assign(e as Error, { completion: c });
    }
  }

  text(model: string, messages: ChatMessage[], maxTokens: number, signal?: AbortSignal, opts?: CallTweaks): Promise<Completion<string>> {
    return this.send(model, messages, maxTokens, signal, undefined, opts);
  }

  private async send(model: string, messages: ChatMessage[], maxTokens: number, signal?: AbortSignal, schema?: object, opts?: CallTweaks, refits = 2): Promise<Completion<string>> {
    const started = performance.now();
    const cap = opts?.planning ? this.cfg.planMaxCallTokens : this.cfg.maxCallTokens;
    const kwargs = {
      ...(this.cfg.extra?.chat_template_kwargs as Record<string, unknown> | undefined),
      ...(typeof opts?.thinking === 'boolean' ? { enable_thinking: opts.thinking } : {}),
      ...(opts?.reasoningEffort ? { reasoning_effort: opts.reasoningEffort } : {}),
    };
    // the sampling set follows whether this call thinks: a cut retry with thinking off gets the non-thinking set
    const thinks = typeof kwargs.enable_thinking === 'boolean' ? kwargs.enable_thinking : undefined;
    const body: Record<string, unknown> = {
      model: this.served(model),
      messages,
      max_tokens: cap && cap < maxTokens ? cap : maxTokens,
      ...samplingFields(this.cfg, thinks),
      ...this.cfg.extra,
    };
    if (Object.keys(kwargs).length) body.chat_template_kwargs = kwargs;
    const level = kwargs.reasoning_effort as ReasoningEffort | undefined;
    const budget = thinks === false || !this.cfg.thinkingBudget ? undefined : (level !== undefined ? this.cfg.thinkingBudget.levels[level] : undefined) ?? this.cfg.thinkingBudget.all;
    if (budget !== undefined) body.thinking_budget_tokens = budget;
    checkTopK(body.top_k);
    const guided = this.cfg.guided ?? 'response_format';
    if (schema && guided === 'response_format') body.response_format = { type: 'json_schema', json_schema: { name: 'output', schema, strict: true } };
    else if (schema && guided === 'guided_json') body.guided_json = schema;
    const res = await this.post(body, signal);
    if (!res.ok) {
      // prompt + max_tokens over the server's context: ask again for what fits. vLLM 0.31 does not
      // say the prompt's size: it refuses a max_tokens above the whole context before counting,
      // and otherwise counts a prompt cut at what would fit plus one ("at least N"), so the prompt
      // is counted with the server's own /tokenize (a guess from its length if that fails)
      const refused = res.status === 400 && refits > 0 ? contextRefusal(res.text) : undefined;
      if (refused) {
        const prompt = refused.input !== undefined && !refused.atLeast ? refused.input : Math.max(refused.input ?? 0, await this.promptTokens(body, signal));
        const fit = refused.limit - prompt - 16;
        if (fit >= MIN_REFIT_TOKENS && fit < (body.max_tokens as number)) return this.send(model, messages, fit, signal, schema, opts, refits - 1);
      }
      throw new Error(`inference ${res.status}: ${res.text.slice(0, 500)}`);
    }
    const data: any = res.data;
    const choice = data?.choices?.[0];
    const raw: string = choice?.message?.content ?? '';
    // without vLLM's --reasoning-parser the thinking comes back inside content; Qwen's template
    // opens <think> in the prompt, so content holds only its closing tag
    const inlineThought = inlineThinkingChars(raw);
    const reasoning = data?.usage?.completion_tokens_details?.reasoning_tokens;
    const forced = choice?.forced ?? data?.forced;
    const thought = choice?.message?.reasoning_content ?? choice?.message?.reasoning;
    const sentThinking = (body.chat_template_kwargs as { enable_thinking?: unknown } | undefined)?.enable_thinking;
    const thoughtChars = typeof thought === 'string' && thought ? thought.length : inlineThought || undefined;
    const cut = choice?.finish_reason === 'length';
    // thinking on, no parser on the server, cut before </think>: all of content is thinking
    const cutThinking = sentThinking === true && cut && typeof thought !== 'string' && !raw.includes('</think>');
    const close = raw.lastIndexOf('</think>');
    const inlineText = cutThinking ? raw : close >= 0 ? raw.slice(0, close).replace(/^\s*<think>\n?/, '') : '';
    const reasoningText = typeof thought === 'string' && thought ? thought : inlineText || undefined;
    const out: Completion<string> = {
      value: cutThinking ? '' : stripThinking(raw),
      raw,
      promptTokens: data?.usage?.prompt_tokens ?? 0,
      completionTokens: data?.usage?.completion_tokens ?? 0,
      reasoningTokens: typeof reasoning === 'number' ? reasoning : undefined,
      cut: cut || undefined,
      forced: typeof forced === 'string' && forced ? forced : forced === true ? 'yes' : undefined,
      maxTokens: body.max_tokens as number,
      reasoningChars: cutThinking ? raw.length : thoughtChars,
      reasoning: reasoningText,
      thinking: typeof sentThinking === 'boolean' ? sentThinking : undefined,
      ms: Math.round(performance.now() - started),
    };
    if (this.cfg.debug) console.error(describeCall(body.model as string, body.max_tokens as number, out));
    return out;
  }

  /**
   * The prompt's tokens as the server counts them (vLLM's POST /tokenize, beside /v1, with the
   * same chat template switches), else a guess on the high side from its length.
   */
  private async promptTokens(body: Record<string, unknown>, signal?: AbortSignal): Promise<number> {
    const guess = Math.ceil(JSON.stringify(body.messages).length / PROMPT_CHARS_PER_TOKEN);
    try {
      const res = await fetch(`${this.cfg.baseUrl.replace(/\/$/, '').replace(/\/v1$/, '')}/tokenize`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(this.cfg.apiKey ? { authorization: `Bearer ${this.cfg.apiKey}` } : {}) },
        body: JSON.stringify({ model: body.model, messages: body.messages, add_generation_prompt: true, ...(body.chat_template_kwargs ? { chat_template_kwargs: body.chat_template_kwargs } : {}) }),
        signal: signal ? AbortSignal.any([AbortSignal.timeout(30_000), signal]) : AbortSignal.timeout(30_000),
      });
      const count = res.ok ? ((await res.json()) as { count?: unknown }).count : undefined;
      return typeof count === 'number' && count > 0 ? count : guess;
    } catch {
      return guess;
    }
  }

  /**
   * One POST, sent again while the server is down or restarting (refused or dropped connection,
   * 502/503/504) for up to serverWaitMs, and a 500 up to MAX_500_RETRIES times, with growing
   * pauses. The run's stop and the call's own timeout are never retried.
   */
  private async post(body: Record<string, unknown>, signal?: AbortSignal): Promise<{ ok: true; data: any } | { ok: false; status: number; text: string }> {
    const url = `${this.cfg.baseUrl.replace(/\/$/, '')}/chat/completions`;
    // the wait starts when the server first fails this call, not when it was sent: a call that ran
    // (or sat in vLLM's queue) for 20 minutes before a restart dropped it still waits the restart out
    let until: number | undefined;
    let tries = 0, fivehundreds = 0;
    for (;;) {
      let problem: string;
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'content-type': 'application/json', ...(this.cfg.apiKey ? { authorization: `Bearer ${this.cfg.apiKey}` } : {}) },
          body: JSON.stringify(body),
          // the run's stop (POST /stop) ends a call in flight too, so it does not hold its slot for up to timeoutMs
          signal: signal ? AbortSignal.any([AbortSignal.timeout(this.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS), signal]) : AbortSignal.timeout(this.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS),
          // Bun also cuts every fetch at 300 s unless told not to; timeoutMs is the only limit
          timeout: false,
        } as RequestInit);
        if (res.ok) return { ok: true, data: await res.json() };
        const text = await res.text();
        const down = res.status === 502 || res.status === 503 || res.status === 504;
        if (!down && !(res.status === 500 && fivehundreds++ < MAX_500_RETRIES)) return { ok: false, status: res.status, text };
        problem = `inference ${res.status}: ${text.slice(0, 200)}`;
      } catch (e) {
        // a stop or the call's own timeout ends the call; anything else (refused, reset, a body
        // cut off) is the server going away
        if (signal?.aborted || isAbort(e)) throw e;
        problem = String((e as Error)?.message ?? e);
      }
      const pause = SERVER_RETRY_MS[Math.min(tries++, SERVER_RETRY_MS.length - 1)]!;
      until ??= Date.now() + (this.cfg.serverWaitMs ?? DEFAULT_SERVER_WAIT_MS);
      if (Date.now() + pause > until) throw new Error(`inference server ${this.cfg.baseUrl} not answering after ${tries} tries: ${problem}`);
      serverNote(this.cfg.baseUrl, problem);
      await sleepUnless(pause, signal);
    }
  }
}

const isAbort = (e: unknown) => ['AbortError', 'TimeoutError'].includes((e as Error)?.name);

/** One line on stderr per server and half minute while calls wait for it (80 agents would print 80). */
const noted = new Map<string, number>();
function serverNote(url: string, problem: string) {
  const now = Date.now();
  if (now - (noted.get(url) ?? 0) < 30_000) return;
  noted.set(url, now);
  console.error(`${new Date().toTimeString().slice(0, 8)} inference server ${url} not answering (${problem.split('\n')[0]!.slice(0, 160)}); calls wait for it and are sent again`);
}

function sleepUnless(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new Error('aborted'));
    const t = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(signal!.reason ?? new Error('aborted'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Fewest tokens worth asking for again after a context-length refusal. */
export const MIN_REFIT_TOKENS = 1024;

/**
 * The max_tokens that fits, from a server's context-length refusal (vLLM's wordings up to 0.31,
 * SGLang's and llama-server's): the context less the prompt, less a little room. A refusal that does not count
 * the prompt (`max_tokens=N cannot be greater than max_model_len=L`) gives L - 1; an "at least N"
 * count is taken as it is. send() counts the prompt itself for both (contextRefusal).
 * Undefined when the error is something else or too little would fit.
 */
export function contextFit(error: string): number | undefined {
  const r = contextRefusal(error);
  if (!r) return undefined;
  if (r.input === undefined) return r.limit - 1;
  const fit = r.limit - r.input - 16;
  return fit >= MIN_REFIT_TOKENS ? fit : undefined;
}

/** Characters per token for the prompt guess when the server cannot count it: low, so the guess errs high. */
export const PROMPT_CHARS_PER_TOKEN = 2.5;

/**
 * A context-length refusal read: the context, and the prompt's tokens when the error says them
 * (`atLeast`: vLLM 0.31 counted a prompt it had cut, so the real one may be far longer). No input
 * for the refusals that do not count the prompt (a max_tokens above the whole context, or 0.31's
 * "contains N characters" when it was too long to tokenize). Undefined for any other error.
 */
export function contextRefusal(error: string): { limit: number; input?: number; atLeast: boolean } | undefined {
  // llama-server: the prompt alone fills the slot ("request (N tokens) exceeds the available context
  // size (M tokens)", type exceed_context_size_error, with n_prompt_tokens and n_ctx beside it)
  const llama = /request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)/.exec(error);
  if (llama) return { limit: Number(llama[2]), input: Number(llama[1]), atLeast: false };
  if (error.includes('exceed_context_size_error')) {
    const n = /"n_prompt_tokens"\s*:\s*(\d+)/.exec(error)?.[1], c = /"n_ctx"\s*:\s*(\d+)/.exec(error)?.[1];
    if (n && c) return { limit: Number(c), input: Number(n), atLeast: false };
  }
  const whole = /max(?:_completion)?_tokens=\d+ cannot be greater than max_model_len=(\d+)/.exec(error)?.[1];
  if (whole) return { limit: Number(whole), atLeast: true };
  const limit = /maximum context length (?:is |of )?(\d+)/.exec(error)?.[1];
  if (!limit) return undefined;
  if (/your prompt contains \d+ characters/.test(error)) return { limit: Number(limit), atLeast: true };
  const m = /(\d+) in the messages|has (\d+) input tokens|(\d+) tokens from the input|contains (at least )?(\d+) input tokens/.exec(error);
  const input = m?.[1] ?? m?.[2] ?? m?.[3] ?? m?.[5];
  if (!input) return undefined;
  return { limit: Number(limit), input: Number(input), atLeast: !!m?.[4] };
}

/** The MUGGE_DEBUG_CALLS line: what one call cost and whether it finished. */
export function describeCall(model: string, maxTokens: number, c: Completion<string>): string {
  const reasoning = c.reasoningTokens === undefined ? '' : ` reasoning ${c.reasoningTokens}`;
  return `call ${model}: prompt ${c.promptTokens} completion ${c.completionTokens}${reasoning} of ${maxTokens} max, ${c.ms} ms, ${c.cut ? 'CUT at max_tokens' : 'finished'}${c.forced ? `, FORCED (${c.forced})` : ''}, ${c.raw.length} chars`;
}

/** Parses a JSON answer, tolerating a ```json fence or text around one object (for `guided: none`). */
export function parseJsonAnswer<T>(raw: string): T {
  const s = stripThinking(raw).trim();
  try {
    return JSON.parse(s);
  } catch {
    const fenced = s.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (fenced) return JSON.parse(fenced[1]);
    const a = s.indexOf('{'), b = s.lastIndexOf('}');
    if (a >= 0 && b > a) return JSON.parse(s.slice(a, b + 1));
    throw new Error(`model answer is not JSON: ${s.slice(0, 200)}`);
  }
}

