// Adapted from salu (github.com/OliverVillson/salu) src/tui/dog/render.ts
/**
 * Pure frame renderer: pixel sprites in, terminal lines out. Two pixels per cell using the
 * half blocks ▀ ▄ █, or 2 x 4 braille dots per cell for the one-line dog, painted with the mugge
 * palette only (greens). No I/O, no timers.
 */
import { PALETTE, type ColorLevel, type Role } from '../theme.ts';
import {
  ASCII_LINE_RUN, ASCII_LINE_SLEEP, ASCII_LINE_WIDTH, ASCII_RUN, ASCII_SLEEP, ASCII_WIDTH, RUN, SLEEP, SLEEP_WIDTH,
  TINY_RUN, TINY_SLEEP, TINY_SLEEP_WIDTH, TINY_WIDTH, WIDTH,
} from './sprites.ts';

export type DogSize = 'full' | 'mini';
export type DogMode = 'run' | 'sleep';

/** Pixel letter to palette role. X is the one-colour pixel of the small dog. */
const ROLE: Record<string, Role> = { A: 'accent', T: 'text', M: 'ok', D: 'chrome', X: 'accent' };
/** Letters drawn as themselves (the snores), with their colour. */
const LETTER: Record<string, Role> = { z: 'chrome', Z: 'text' };
const ESC = '\u001b[';

function fgCode(role: Role, level: ColorLevel): string {
  const s = PALETTE[role];
  return level === 3 ? `38;2;${s.rgb.join(';')}` : level === 2 ? `38;5;${s.c256}` : String(s.c16);
}
function bgCode(role: Role, level: ColorLevel): string {
  const s = PALETTE[role];
  return level === 3 ? `48;2;${s.rgb.join(';')}` : level === 2 ? `48;5;${s.c256}` : String(s.c16 + 10);
}

/** Joins (char, sgr) cells into one line, opening an escape only when the colour changes. */
function paintCells(cells: Array<[string, string]>): string {
  let out = '';
  let cur = ''; // SGR currently open
  for (const [ch, sgr] of cells) {
    if (sgr !== cur) {
      if (cur) out += `${ESC}39;49m`;
      if (sgr) out += `${ESC}${sgr}m`;
      cur = sgr;
    }
    out += ch;
  }
  if (cur) out += `${ESC}39;49m`;
  return out;
}

/** One terminal row from two pixel rows. */
export function cellRow(top: string, bottom: string, level: ColorLevel): string {
  const cells: Array<[string, string]> = [];
  const w = Math.max(top.length, bottom.length);
  for (let i = 0; i < w; i++) {
    const letter = LETTER[top[i] ?? '.'] ?? LETTER[bottom[i] ?? '.'];
    if (letter) {
      cells.push([(top[i] !== '.' ? top[i] : bottom[i])!, fgCode(letter, level)]);
      continue;
    }
    const t = ROLE[top[i] ?? '.'];
    const b = ROLE[bottom[i] ?? '.'];
    if (!t && !b) cells.push([' ', '']);
    else if (t && !b) cells.push(['▀', fgCode(t, level)]);
    else if (!t && b) cells.push(['▄', fgCode(b, level)]);
    else if (t === b) cells.push(['█', fgCode(t!, level)]);
    else cells.push(['▀', `${fgCode(t!, level)};${bgCode(b!, level)}`]);
  }
  return paintCells(cells);
}

/** Sprite rows to terminal lines. */
export function renderSprite(px: readonly string[], level: ColorLevel): string[] {
  const lines: string[] = [];
  for (let r = 0; r < px.length; r += 2) lines.push(cellRow(px[r]!, px[r + 1] ?? '', level));
  return lines;
}

/** Braille dot bits for (column, row) inside a 2 x 4 cell. */
const DOT = [
  [0x01, 0x02, 0x04, 0x40],
  [0x08, 0x10, 0x20, 0x80],
];

/**
 * Four pixel rows to ONE terminal line of braille (U+2800 block: single width, in every common
 * terminal font or its fallback). Any non-'.' pixel is a dot; a cell holding a z/Z letter is
 * drawn as that letter. Dots take the accent colour.
 */
export function renderBraille(px: readonly string[], level: ColorLevel): string {
  const w = Math.max(...px.map((r) => r.length));
  const cells: Array<[string, string]> = [];
  for (let x = 0; x < w; x += 2) {
    let bits = 0;
    let letter = '';
    for (let dx = 0; dx < 2; dx++)
      for (let y = 0; y < 4; y++) {
        const c = px[y]?.[x + dx] ?? '.';
        if (LETTER[c]) letter = c;
        else if (c !== '.') bits |= DOT[dx]![y]!;
      }
    if (letter) cells.push([letter, fgCode(LETTER[letter]!, level)]);
    else if (bits) cells.push([String.fromCharCode(0x2800 + bits), fgCode('accent', level)]);
    else cells.push([' ', '']);
  }
  return paintCells(cells);
}

function spriteSet(mode: DogMode, size: DogSize): string[][] {
  if (size === 'mini') return mode === 'run' ? TINY_RUN : TINY_SLEEP;
  return mode === 'run' ? RUN : SLEEP;
}

/** Frame counts per mode and size (the ticker cycles through these). */
export function frameCount(mode: DogMode, size: DogSize = 'full'): number {
  return spriteSet(mode, size).length;
}

/** Dog width in terminal cells for a size and mode at a colour level. */
export function dogWidth(size: DogSize = 'full', level: ColorLevel = 3, mode: DogMode = 'run'): number {
  if (level === 0) return size === 'mini' ? ASCII_LINE_WIDTH : ASCII_WIDTH;
  if (size === 'mini') return mode === 'run' ? TINY_WIDTH : TINY_SLEEP_WIDTH;
  return mode === 'run' ? WIDTH : SLEEP_WIDTH;
}

/**
 * Lines for one frame. `tick` is any increasing counter; the frame index wraps. Level 0 (NO_COLOR,
 * dumb terminals) returns plain ASCII with no escapes. Sleeping frames advance 8x slower so the
 * z's drift gently rather than flicker.
 */
export function renderDog(tick: number, opts: { mode?: DogMode; size?: DogSize; level: ColorLevel }): string[] {
  const { mode = 'run', size = 'full', level } = opts;
  const t = mode === 'sleep' ? Math.floor(tick / 8) : tick;
  const n = frameCount(mode, size);
  const i = ((t % n) + n) % n;
  if (level === 0) {
    if (size === 'mini') return [(mode === 'run' ? ASCII_LINE_RUN : ASCII_LINE_SLEEP)[i % (mode === 'run' ? ASCII_LINE_RUN.length : ASCII_LINE_SLEEP.length)]!];
    return [...(mode === 'run' ? ASCII_RUN : ASCII_SLEEP)[i]!];
  }
  return renderSprite(spriteSet(mode, size)[i]!, level);
}

/**
 * The dog running along a track `width` cells wide; `progress` (0..1) places it. Each line is
 * padded on the left so the lines stay the same visible width as `width`.
 */
export function renderTrack(tick: number, width: number, progress: number, opts: { size?: DogSize; level: ColorLevel }): string[] {
  const dw = dogWidth(opts.size, opts.level);
  const room = Math.max(0, width - dw);
  const x = Math.round(Math.min(1, Math.max(0, progress)) * room);
  return renderDog(tick, { ...opts, mode: 'run' }).map((l) => ' '.repeat(x) + l);
}

/** Visible width of a rendered line (escapes stripped). */
export const visibleWidth = (s: string) => s.replace(/\u001b\[[0-9;]*m/g, '').length;
