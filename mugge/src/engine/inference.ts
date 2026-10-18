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
  /** One line per call on stderr (MUGGE_DEBUG_CALLS=1). */
  debug?: boolean;
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

  private async send(model: string, messages: ChatMessage[], maxTokens: number, signal?: AbortSignal, schema?: object, opts?: CallTweaks): Promise<Completion<string>> {
    const started = performance.now();
    const cap = this.cfg.maxCallTokens;
    const kwargs = { ...(this.cfg.extra?.chat_template_kwargs as Record<string, unknown> | undefined), ...(typeof opts?.thinking === 'boolean' ? { enable_thinking: opts.thinking } : {}) };
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
    checkTopK(body.top_k);
    const guided = this.cfg.guided ?? 'response_format';
    if (schema && guided === 'response_format') body.response_format = { type: 'json_schema', json_schema: { name: 'output', schema, strict: true } };
    else if (schema && guided === 'guided_json') body.guided_json = schema;
    const res = await fetch(`${this.cfg.baseUrl.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(this.cfg.apiKey ? { authorization: `Bearer ${this.cfg.apiKey}` } : {}) },
      body: JSON.stringify(body),
      // the run's stop (POST /stop) ends a call in flight too, so it does not hold its slot for up to timeoutMs
      signal: signal ? AbortSignal.any([AbortSignal.timeout(this.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS), signal]) : AbortSignal.timeout(this.cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS),
      // Bun also cuts every fetch at 300 s unless told not to; timeoutMs is the only limit
      timeout: false,
    } as RequestInit);
    if (!res.ok) throw new Error(`inference ${res.status}: ${(await res.text()).slice(0, 500)}`);
    const data: any = await res.json();
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

