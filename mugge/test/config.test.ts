import { describe, expect, test } from 'bun:test';
import { inferenceFromEnv } from '../src/config.ts';
import { OpenAICompatible, describeCall } from '../src/engine/inference.ts';

describe('thinking switch, per-call cap and debug line', () => {
  test('MUGGE_THINKING becomes chat_template_kwargs.enable_thinking; unset leaves the server default', () => {
    expect(inferenceFromEnv({}).extra).toBeUndefined();
    expect(inferenceFromEnv({ MUGGE_THINKING: 'off' }).extra).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(inferenceFromEnv({ MUGGE_THINKING: 'ON' }).extra).toEqual({ chat_template_kwargs: { enable_thinking: true } });
    expect(() => inferenceFromEnv({ MUGGE_THINKING: 'maybe' })).toThrow();
  });

  test('MUGGE_MAX_CALL_TOKENS must be a positive integer', () => {
    expect(inferenceFromEnv({}).maxCallTokens).toBeUndefined();
    expect(inferenceFromEnv({ MUGGE_MAX_CALL_TOKENS: '4096' }).maxCallTokens).toBe(4096);
    expect(() => inferenceFromEnv({ MUGGE_MAX_CALL_TOKENS: '0' })).toThrow();
    expect(() => inferenceFromEnv({ MUGGE_MAX_CALL_TOKENS: 'lots' })).toThrow();
    expect(inferenceFromEnv({ MUGGE_DEBUG_CALLS: '1' }).debug).toBe(true);
  });

  test('a call sends the switch, caps max_tokens at the budget or the cap, and reads reasoning tokens and the cut', async () => {
    const bodies: any[] = [];
    const server = Bun.serve({
      port: 0,
      fetch: async (req) => {
        bodies.push(await req.json());
        return Response.json({
          choices: [{ message: { content: 'x.py\n```python\nprint(1)\n```' }, finish_reason: 'length' }],
          usage: { prompt_tokens: 10, completion_tokens: 4096, completion_tokens_details: { reasoning_tokens: 4000 } },
        });
      },
    });
    try {
      const inf = new OpenAICompatible({ ...inferenceFromEnv({ MUGGE_THINKING: 'off', MUGGE_MAX_CALL_TOKENS: '4096' }), baseUrl: `${server.url}v1` });
      const c = await inf.text('coder', [{ role: 'user', content: 'hi' }], 60000);
      expect(bodies[0].max_tokens).toBe(4096);
      expect(bodies[0].chat_template_kwargs).toEqual({ enable_thinking: false });
      expect(c.reasoningTokens).toBe(4000);
      expect(c.cut).toBe(true);
      // the budget's remainder wins when it is smaller than the cap
      await inf.text('coder', [{ role: 'user', content: 'hi' }], 100);
      expect(bodies[1].max_tokens).toBe(100);
      expect(describeCall('coder', 4096, c)).toContain('reasoning 4000');
      expect(describeCall('coder', 4096, c)).toContain('CUT at max_tokens');
    } finally {
      server.stop(true);
    }
  });

  test('without the cap or a reported reasoning count, nothing is invented', async () => {
    const server = Bun.serve({
      port: 0,
      fetch: async () => Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    });
    try {
      const inf = new OpenAICompatible({ ...inferenceFromEnv({}), baseUrl: `${server.url}v1` });
      const c = await inf.text('coder', [{ role: 'user', content: 'hi' }], 60000);
      expect(c.reasoningTokens).toBeUndefined();
      expect(c.cut).toBeUndefined();
      expect(describeCall('coder', 60000, c)).toContain('finished');
    } finally {
      server.stop(true);
    }
  });
});
