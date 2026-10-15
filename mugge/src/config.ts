/**
 * Engine settings from the environment, so the same binary runs against the fake model in CI,
 * vLLM on the project VM, or a hosted endpoint later.
 *
 *   MUGGE_INFERENCE_URL   OpenAI-compatible base URL (default http://127.0.0.1:8000/v1, vLLM)
 *   MUGGE_MODEL           served model for names not in MUGGE_MODELS (default: mugge-small)
 *   MUGGE_MODELS          JSON map ticket model → served name, e.g. {"coder-ts":"coder-ts-lora"}
 *   MUGGE_GUIDED          response_format | guided_json | none
 *   MUGGE_API_KEY         bearer token, for hosted endpoints
 *   MUGGE_TIMEOUT_S       most seconds one model call may take (default 600)
 *   MUGGE_CONCURRENCY     tickets at once (default 12)
 *   MUGGE_SANDBOX         local | podman | docker (see engine/sandbox.ts)
 *   MUGGE_THINKING        on | off: sent as chat_template_kwargs.enable_thinking on every call
 *                         (unset: the server's default). The bench's --thinking flag is the same switch.
 *   MUGGE_MAX_CALL_TOKENS most tokens one model call may write, whatever the ticket's budget leaves
 *                         (unset: the budget's remainder, 60K by default, which a thinking model can
 *                         spend in one call)
 *   MUGGE_DEBUG_CALLS     1: one line per model call on stderr: model, prompt, completion and
 *                         reasoning tokens, time, and whether the answer was cut at max_tokens
 *   MUGGE_REVIEW          on | off (default off): one lead review of each ticket whose checks pass
 *                         (engine/review.ts); read by the harness itself
 */
import type { InferenceConfig } from './engine/inference.ts';

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
    extra: thinking ? { chat_template_kwargs: { enable_thinking: thinking === 'on' } } : undefined,
    maxCallTokens: maxCall,
    debug: env.MUGGE_DEBUG_CALLS === '1',
  };
}

export function concurrencyFromEnv(env: NodeJS.ProcessEnv = process.env, fallback = 12): number {
  const n = Number(env.MUGGE_CONCURRENCY);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}
