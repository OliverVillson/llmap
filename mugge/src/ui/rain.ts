// Adapted from salu (github.com/OliverVillson/salu) src/tui/rain.ts
/**
 * Digital rain for the splash screen: one drop per column falls from the top with a bright
 * head and a fading trail, starting a little apart so it washes down the screen like a curtain,
 * and every drop has left the screen by RAIN_MS. Pure: a seed and a size in, lines out, so a
 * frame is the same for the same (seed, ms) and the timer lives with the caller.
 */
import { painter, type ColorLevel } from './theme.ts';

/** How long the rain lasts; the screen is empty again by then. */
export const RAIN_MS = 2000;

/** Half-width katakana and digits (single width, like the film); plain ASCII when unicode is off. */
const KANA = 'ｦｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789';
const ASCII = '0123456789abcdefghijkmnopqrstuvwxyzZ:=*+-<>|';

interface Drop {
  /** ms before the head enters the top row */
  delay: number;
  /** rows per ms */
  speed: number;
  /** trail length in rows */
  trail: number;
}

export interface Rain {
  cols: number;
  rows: number;
  seed: number;
  chars: string[];
  drops: Drop[];
}

/** A small deterministic integer hash (so frames repeat for the same seed). */
function hash(a: number, b = 0, c = 0): number {
  let h = (a * 0x27d4eb2d) ^ (b * 0x165667b1) ^ (c * 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}
const unit = (a: number, b = 0, c = 0) => hash(a, b, c) / 0x100000000;

export function makeRain(cols: number, rows: number, seed: number, unicode = true): Rain {
  const drops: Drop[] = [];
  for (let c = 0; c < cols; c++) {
    const delay = unit(seed, c, 1) * 450;
    const trail = Math.max(4, Math.round(rows * (0.35 + unit(seed, c, 2) * 0.5)));
    // Gone (head and trail below the last row) somewhere between 1.5 s and RAIN_MS - 80 ms.
    const end = RAIN_MS - 80 - unit(seed, c, 3) * 420;
    drops.push({ delay, trail, speed: (rows + trail) / (end - delay) });
  }
  return { cols, rows, seed, chars: [...(unicode ? KANA : ASCII)], drops };
}

type Shade = 'head' | 'bright' | 'mid' | 'tail';

/** What sits at (column, row) after `ms`: a character and its shade, or null for empty. */
export function rainCell(rain: Rain, c: number, r: number, ms: number): { ch: string; shade: Shade } | null {
  const d = rain.drops[c];
  if (!d || ms < d.delay) return null;
  const head = Math.floor((ms - d.delay) * d.speed);
  const dist = head - r;
  if (dist < 0 || dist >= d.trail) return null;
  const shade: Shade = dist === 0 ? 'head' : dist <= 2 ? 'bright' : dist < d.trail * 0.55 ? 'mid' : 'tail';
  // Trail characters flicker now and then; the head changes every step.
  const step = dist === 0 ? head : Math.floor((ms + c * 53) / 140);
  const ch = rain.chars[hash(rain.seed, c * 4099 + r, step) % rain.chars.length]!;
  return { ch, shade };
}

/** Whether every drop has left the screen at `ms`. */
export function rainDone(rain: Rain, ms: number): boolean {
  return rain.drops.every((d) => ms >= d.delay && Math.floor((ms - d.delay) * d.speed) - d.trail >= rain.rows - 1);
}

/** Row `r` of the rain between columns c0 (inclusive) and c1 (exclusive), painted at `level`. */
export function rainSpan(rain: Rain, r: number, ms: number, level: ColorLevel, c0 = 0, c1 = rain.cols): string {
  const text = painter('text', level);
  const bold = (s: string) => (level && s ? `\u001b[1m${s}\u001b[22m` : s);
  const paint: Record<Shade, (s: string) => string> = {
    head: (s) => bold(text(s)),
    bright: painter('accent', level),
    mid: painter('ok', level),
    tail: painter('chrome', level),
  };
  let line = '';
  let run = '';
  let cur: Shade | null = null;
  const flush = () => {
    if (run) line += cur ? paint[cur](run) : run;
    run = '';
  };
  for (let c = Math.max(0, c0); c < Math.min(c1, rain.cols); c++) {
    const cell = rainCell(rain, c, r, ms);
    const shade = cell?.shade ?? null;
    if (shade !== cur) {
      flush();
      cur = shade;
    }
    run += cell?.ch ?? ' ';
  }
  flush();
  return line;
}

/** The whole screen at `ms`: `rain.rows` lines, each exactly `rain.cols` cells. */
export function rainLines(rain: Rain, ms: number, level: ColorLevel): string[] {
  const out: string[] = [];
  for (let r = 0; r < rain.rows; r++) out.push(rainSpan(rain, r, ms, level));
  return out;
}
