// Adapted from salu (github.com/OliverVillson/salu) src/ui/glyphs.ts
/**
 * Every symbol mugge prints, in one table, so the TUI and the plain CLI agree and each glyph is
 * chosen once. The unicode set only uses characters that are (a) in Menlo / SF Mono / DejaVu
 * Sans Mono (Menlo's base) or drawn by the terminal itself, (b) not emoji, so no terminal swaps
 * in a colour emoji (⏺ ▶ ✳ ✔ ⏸ are avoided for that reason), and (c) single width. Terminals
 * that cannot show unicode (a non-UTF-8 locale, or MUGGE_ASCII=1) get the ASCII set.
 */
export interface Glyphs {
  /** ticket statuses */
  backlog: string;
  todo: string;
  running: string;
  paused: string;
  blocked: string;
  failed: string;
  done: string;
  interrupted: string;
  /** orchestrator on / off */
  on: string;
  off: string;
  /** a worker started or dispatch resumed */
  start: string;
  /** assistant text and tool calls in a transcript, and the result under a tool call */
  say: string;
  toolResult: string;
  /** selected row */
  cursor: string;
  /** end of the selected row: pressing → goes deeper (opens a menu, a project, the properties) */
  deeper: string;
  /** block cursor in front of the wordmark */
  mark: string;
  /** separator in status lines and breadcrumbs */
  dot: string;
  crumb: string;
  ellipsis: string;
  /** usage meter: filled and empty cell */
  barFull: string;
  barEmpty: string;
  /** the "thinking" spinner for running tickets */
  spinner: string[];
}

export const UNICODE_GLYPHS: Glyphs = {
  backlog: '◌',
  todo: '○',
  running: '●',
  paused: '‖',
  blocked: '?',
  failed: '✗',
  done: '✓',
  interrupted: '■',
  on: '●',
  off: '○',
  start: '▸',
  say: '●',
  toolResult: '└',
  cursor: '❯',
  deeper: '▸',
  mark: '▌',
  dot: '·',
  crumb: '›',
  ellipsis: '…',
  barFull: '▰',
  barEmpty: '▱',
  spinner: ['·', '✢', '✶', '✻', '✽', '✻', '✶', '✢'],
};

export const ASCII_GLYPHS: Glyphs = {
  backlog: '.',
  todo: 'o',
  running: '*',
  paused: '=',
  blocked: '?',
  failed: 'x',
  done: '+',
  interrupted: '#',
  on: '*',
  off: 'o',
  start: '>',
  say: '*',
  toolResult: '`',
  cursor: '>',
  deeper: '>',
  mark: '|',
  dot: '-',
  crumb: '>',
  ellipsis: '...',
  barFull: '#',
  barEmpty: '-',
  spinner: ['-', '\\', '|', '/'],
};

/**
 * Whether the terminal can show the unicode set: not when the locale names a charset other than
 * UTF-8, or is plain C / POSIX. No locale at all (common in GUI terminals on macOS) counts as
 * unicode. MUGGE_ASCII=1 forces ASCII, MUGGE_ASCII=0
 * forces unicode.
 */
export function supportsUnicode(env: NodeJS.ProcessEnv = process.env): boolean {
  if (env.MUGGE_ASCII === '0') return true;
  if (env.MUGGE_ASCII) return false;
  const locale = env.LC_ALL || env.LC_CTYPE || env.LANG || '';
  if (/\./.test(locale)) return /utf-?8/i.test(locale);
  return locale !== 'C' && locale !== 'POSIX';
}

export const glyphsFor = (env: NodeJS.ProcessEnv = process.env): Glyphs => (supportsUnicode(env) ? UNICODE_GLYPHS : ASCII_GLYPHS);

/** The set for this process. */
export const GLYPHS: Glyphs = glyphsFor();
