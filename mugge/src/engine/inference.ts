/**
 * Client for any OpenAI-compatible chat endpoint: vLLM or SGLang on the project VM, or a
 * hosted service later. Output is constrained to a JSON schema so every answer parses.
 *
 * Specialists are LoRA adapters served under their own model names by vLLM
 * (`--lora-modules coder-ts=/models/adapters/coder-ts`), so routing is only a name lookup.
 */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface Completion<T> {
  value: T;
  raw: string;
  promptTokens: number;
  completionTokens: number;
  ms: number;
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
  temperature?: number;
  timeoutMs?: number;
}

export interface Inference {
  complete<T>(model: string, messages: ChatMessage[], schema: object, maxTokens: number): Promise<Completion<T>>;
}

export class OpenAICompatible implements Inference {
  constructor(readonly cfg: InferenceConfig) {}

  served(model: string): string {
    return this.cfg.models?.[model] ?? this.cfg.models?.[model.split('@')[0]] ?? this.cfg.defaultModel;
  }

  async complete<T>(model: string, messages: ChatMessage[], schema: object, maxTokens: number): Promise<Completion<T>> {
    const started = performance.now();
    const body: Record<string, unknown> = {
      model: this.served(model),
      messages,
      max_tokens: maxTokens,
      temperature: this.cfg.temperature ?? 0.2,
    };
    const guided = this.cfg.guided ?? 'response_format';
    if (guided === 'response_format') body.response_format = { type: 'json_schema', json_schema: { name: 'output', schema, strict: true } };
    else if (guided === 'guided_json') body.guided_json = schema;
    const res = await fetch(`${this.cfg.baseUrl.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(this.cfg.apiKey ? { authorization: `Bearer ${this.cfg.apiKey}` } : {}) },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.cfg.timeoutMs ?? 600_000),
    });
    if (!res.ok) throw new Error(`inference ${res.status}: ${(await res.text()).slice(0, 500)}`);
    const data: any = await res.json();
    const raw: string = data?.choices?.[0]?.message?.content ?? '';
    return {
      value: parseJsonAnswer<T>(raw),
      raw,
      promptTokens: data?.usage?.prompt_tokens ?? 0,
      completionTokens: data?.usage?.completion_tokens ?? 0,
      ms: Math.round(performance.now() - started),
    };
  }
}

/** Parses a JSON answer, tolerating a ```json fence or text around one object (for `guided: none`). */
export function parseJsonAnswer<T>(raw: string): T {
  const s = raw.trim();
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
