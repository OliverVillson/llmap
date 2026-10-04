/**
 * Engine settings from the environment, so the same binary runs against the fake model in CI,
 * vLLM on the project VM, or a hosted endpoint later.
 *
 *   MUGGE_INFERENCE_URL   OpenAI-compatible base URL (default http://127.0.0.1:8000/v1, vLLM)
 *   MUGGE_MODEL           served model for names not in MUGGE_MODELS (default: mugge-small)
 *   MUGGE_MODELS          JSON map ticket model → served name, e.g. {"coder-ts":"coder-ts-lora"}
 *   MUGGE_GUIDED          response_format | guided_json | none
 *   MUGGE_API_KEY         bearer token, for hosted endpoints
 *   MUGGE_CONCURRENCY     tickets at once (default 12)
 *   MUGGE_SANDBOX         local | podman | docker (see engine/sandbox.ts)
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
  return {
    baseUrl: env.MUGGE_INFERENCE_URL || 'http://127.0.0.1:8000/v1',
    defaultModel: env.MUGGE_MODEL || 'mugge-small',
    models,
    guided,
    apiKey: env.MUGGE_API_KEY || undefined,
  };
}

export function concurrencyFromEnv(env: NodeJS.ProcessEnv = process.env, fallback = 12): number {
  const n = Number(env.MUGGE_CONCURRENCY);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}
