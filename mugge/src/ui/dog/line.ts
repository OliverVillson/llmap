// Adapted from salu (github.com/OliverVillson/salu) src/tui/dog/line.ts
/**
 * Stateless single-line API for hosts that already own an animation tick (the TUI header):
 * pass any increasing `frame`, get one ANSI string. No timers here.
 */
import type { ColorLevel } from '../theme.ts';
import { detectColorLevel } from '../theme.ts';
import { renderBraille, renderDog } from './render.ts';
import { ASCII_LINE_RUN, ASCII_LINE_SLEEP, ASCII_LINE_WIDTH, TINY_RUN, TINY_SLEEP, TINY_WIDTH } from './sprites.ts';

/** Cells occupied by `dogFrame` in colour (braille, 2 pixels per cell). ASCII is ASCII_LINE_WIDTH wide. */
export const DOG_WIDTH = TINY_WIDTH / 2;
export { ASCII_LINE_WIDTH };

const defaultLevel = (): ColorLevel => detectColorLevel(process.env, true);

/** ONE line: the small dog in braille dots, bright green (plain ASCII at level 0). */
export function dogFrame(frame: number, opts: { level?: ColorLevel } = {}): string {
  const level = opts.level ?? defaultLevel();
  const n = TINY_RUN.length;
  const i = ((frame % n) + n) % n;
  return level === 0 ? ASCII_LINE_RUN[i]! : renderBraille(TINY_RUN[i]!, level);
}

/** ONE line: the small dog asleep (braille, 7 cells with its z; ASCII at level 0). For tight spots. */
export function sleepFrame(opts: { level?: ColorLevel } = {}): string {
  const level = opts.level ?? defaultLevel();
  return level === 0 ? ASCII_LINE_SLEEP[0]! : renderBraille(TINY_SLEEP[0]!, level);
}

/** Multi-line (6 rows, 24 cells) variant. */
export function dogLines(frame: number, opts: { level?: ColorLevel } = {}): string[] {
  return renderDog(frame, { level: opts.level ?? defaultLevel() });
}
