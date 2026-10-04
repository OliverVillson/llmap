import { expect, test } from 'bun:test';
import { runSpike } from '../spike/run-spike.ts';

// The whole engine spike against the fake model: 15 tickets in parallel worktrees, three of them
// needing a fix round, then a clean integration with every check green.
test('engine spike runs and integrates the polyglot project', async () => {
  const { summary, integrated } = await runSpike({ fake: true, quiet: true, latencyMs: 50, flaky: ['c-hash', 'py-store', 'ts-format'], concurrency: 16 });
  expect(summary.done).toBe(15);
  expect(summary.firstTry).toBe(12);
  expect(summary.peakParallel).toBeGreaterThanOrEqual(6);
  expect(integrated).toBe(true);
}, 180_000);
