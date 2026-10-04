// Adapted from salu (github.com/OliverVillson/salu) src/ui/theme.ts
/**
 * The mugge palette: matrix green on black ("hackermode"). One module feeds the TUI and plain
 * CLI output, so every surface agrees.
 *
 * Roles, not colours, are the vocabulary. Each role has a truecolor RGB, a 256-colour index and
 * a 16-colour SGR code, picked by the terminal's colour level.
 */
import { GLYPHS } from './glyphs.ts';

export type ColorLevel = 0 | 1 | 2 | 3; // none | 16 | 256 | truecolor

export type Role = 'accent' | 'text' | 'ok' | 'chrome' | 'warn' | 'error' | 'paused';

interface Swatch {
  rgb: [number, number, number];
  c256: number;
  /** SGR foreground for 16-colour terminals */
  c16: number;
}

export const PALETTE: Record<Role, Swatch> = {
  /** focus: cursor, running, wordmark. Classic matrix bright green. */
  accent: { rgb: [0, 255, 65], c256: 46, c16: 92 },
  /** body text that should read as green rather than white */
  text: { rgb: [143, 255, 170], c256: 121, c16: 92 },
  /** success, done */
  ok: { rgb: [0, 200, 83], c256: 41, c16: 32 },
  /** chrome: borders, hints, secondary text */
  chrome: { rgb: [30, 143, 60], c256: 29, c16: 32 },
  /** warnings, blocked: lime green (never amber: the TUI is green only), told apart by symbol and label */
  warn: { rgb: [190, 255, 60], c256: 154, c16: 92 },
  /** errors, failed */
  error: { rgb: [255, 85, 85], c256: 203, c16: 91 },
  /** paused by the usage window: cool teal, distinct from every green */
  paused: { rgb: [64, 224, 208], c256: 80, c16: 36 },
};

export const hex = (r: Role) => '#' + PALETTE[r].rgb.map((n) => n.toString(16).padStart(2, '0')).join('');

/**
 * Terminal colour support. NO_COLOR and TERM=dumb disable it; FORCE_COLOR (0-3) overrides;
 * otherwise COLORTERM/TERM decide. `isTTY` false (piped) means none unless forced.
 */
export function detectColorLevel(env: NodeJS.ProcessEnv = process.env, isTTY = !!process.stdout.isTTY): ColorLevel {
  if (env.NO_COLOR !== undefined && env.NO_COLOR !== '') return 0;
  const force = env.FORCE_COLOR;
  if (force !== undefined) {
    if (force === '0' || force === 'false') return 0;
    if (force === '1') return 1;
    if (force === '2') return 2;
    if (force === '3') return 3;
    // bare FORCE_COLOR / "true": fall through to detection, but as if it were a TTY
    isTTY = true;
  }
  if (!isTTY) return 0;
  if (env.TERM === 'dumb') return 0;
  if (/truecolor|24bit/i.test(env.COLORTERM ?? '')) return 3;
  if (env.WT_SESSION || env.TERM_PROGRAM === 'iTerm.app' || env.TERM_PROGRAM === 'vscode') return 3;
  if (/256/.test(env.TERM ?? '') || env.TERM_PROGRAM === 'Apple_Terminal') return 2;
  return 1;
}

/** SGR parameters that set a role's foreground at `level`; null when colour is off. */
export function sgr(role: Role, level: ColorLevel): string | null {
  if (level === 0) return null;
  const s = PALETTE[role];
  if (level === 3) return `38;2;${s.rgb.join(';')}`;
  if (level === 2) return `38;5;${s.c256}`;
  return String(s.c16);
}

export type Painter = (s: string) => string;

/**
 * A painter for `role` at `level`; identity when colour is off. `restore` is the escape written
 * after the span instead of "default foreground", so plain text following a painted span stays
 * green rather than falling back to the terminal's own colour.
 */
export function painter(role: Role, level: ColorLevel, restore = '\u001b[39m'): Painter {
  const open = sgr(role, level);
  if (!open) return (s) => s;
  const pre = `\u001b[${open}m`;
  return (s) => (s ? pre + s + restore : s);
}

/** Escape that sets the base text colour at `level` ('' when colour is off). */
export function baseOpen(level: ColorLevel): string {
  const open = sgr('text', level);
  return open ? `\u001b[${open}m` : '';
}

export const WORDMARK = 'mugge';
/** Block cursor in front of the wordmark: the prompt you are about to type into. */
export const MARK = GLYPHS.mark;
