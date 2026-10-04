/**
 * Pure screen renderers for the mugge terminal client: state in, an array of lines out. Every
 * line is at most `cols` visible cells wide, and at level 0 there are no escape codes at all.
 * No I/O and no timers here; tui.ts owns the terminal and the clock.
 */
import type { RunView, TicketState } from '../engine/events.ts';
import { renderBraille, renderDog, dogWidth, type DogMode } from './dog/render.ts';
import { ASCII_LINE_RUN, ASCII_LINE_SLEEP, TINY_RUN, TINY_SLEEP } from './dog/sprites.ts';
import { ASCII_GLYPHS, GLYPHS } from './glyphs.ts';
import { makeRain, rainSpan } from './rain.ts';
import { MARK, WORDMARK, painter, type ColorLevel, type Role } from './theme.ts';

// ---------------------------------------------------------------------------------------------
// width helpers

const ANSI_RE = /\u001b\[[0-9;?]*[A-Za-z]/g;

/** `s` without escape codes. */
export const stripAnsi = (s: string): string => s.replace(ANSI_RE, '');

/** Visible width in cells. Every glyph mugge prints is single width, so code points = cells. */
export const visibleWidth = (s: string): number => [...stripAnsi(s)].length;

/** Cuts a painted line to `w` visible cells, keeping its escapes; resets colour if it cut one. */
export function truncateAnsi(s: string, w: number): string {
  if (w <= 0) return '';
  if (visibleWidth(s) <= w) return s;
  let out = '';
  let n = 0;
  let painted = false;
  let i = 0;
  while (i < s.length && n < w) {
    if (s[i] === '\u001b') {
      const m = /^\u001b\[[0-9;?]*[A-Za-z]/.exec(s.slice(i));
      if (m) { out += m[0]; painted = true; i += m[0].length; continue; }
    }
    const cp = s.codePointAt(i)!;
    const ch = String.fromCodePoint(cp);
    out += ch;
    i += ch.length;
    n++;
  }
  return painted ? out + '\u001b[0m' : out;
}

/** Plain text clipped to `w` cells with an ellipsis. */
export function clip(s: string, w: number): string {
  if (w <= 0) return '';
  const cps = [...s];
  if (cps.length <= w) return s;
  const e = GLYPHS.ellipsis;
  if (w <= e.length) return cps.slice(0, w).join('');
  return cps.slice(0, w - e.length).join('') + e;
}

/** Plain text padded (or clipped) to exactly `w` cells. */
const padR = (s: string, w: number) => {
  const c = clip(s, w);
  return c + ' '.repeat(Math.max(0, w - [...c].length));
};
/** Painted text padded on the right to `w` cells. */
const padAnsi = (s: string, w: number) => s + ' '.repeat(Math.max(0, w - visibleWidth(s)));
const center = (s: string, w: number) => ' '.repeat(Math.max(0, Math.floor((w - visibleWidth(s)) / 2))) + s;
const fit = (lines: string[], cols: number) => lines.map((l) => truncateAnsi(l, cols));

const ascii = () => GLYPHS === ASCII_GLYPHS;
const ruleChar = () => (ascii() ? '-' : '─');

interface Paint {
  level: ColorLevel;
  accent: (s: string) => string;
  text: (s: string) => string;
  ok: (s: string) => string;
  chrome: (s: string) => string;
  warn: (s: string) => string;
  error: (s: string) => string;
  paused: (s: string) => string;
  bold: (s: string) => string;
  role: (r: Role, s: string) => string;
}

function paints(level: ColorLevel): Paint {
  const p = (r: Role) => painter(r, level);
  const cache: Partial<Record<Role, (s: string) => string>> = {};
  const role = (r: Role, s: string) => (cache[r] ??= p(r))(s);
  return {
    level,
    accent: p('accent'), text: p('text'), ok: p('ok'), chrome: p('chrome'), warn: p('warn'), error: p('error'), paused: p('paused'),
    bold: (s) => (level && s ? `\u001b[1m${s}\u001b[22m` : s),
    role,
  };
}

// ---------------------------------------------------------------------------------------------
// formatting

/** 850ms, 12.3s, 2m10s, 1h05m */
export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '-';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const s = Math.round(ms / 1000);
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
}

/** 950, 12.3k, 1.2M */
export function fmtTok(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

function fmtAgo(then: number, now: number): string {
  const d = Math.max(0, now - then) / 1000;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

/** Joins painted pieces with `sep`, wrapping onto new lines so no line passes `cols`. */
function wrapPieces(pieces: string[], cols: number, sep: string, indent = ''): string[] {
  const out: string[] = [];
  let cur = '';
  for (const piece of pieces) {
    if (!cur) { cur = indent + piece; continue; }
    if (visibleWidth(cur) + visibleWidth(sep) + visibleWidth(piece) <= cols) cur += sep + piece;
    else { out.push(cur); cur = indent + piece; }
  }
  if (cur) out.push(cur);
  return out;
}

// ---------------------------------------------------------------------------------------------
// header

/** The one-line braille dog: running, or curled up with its z's (which drift 8x slower). */
function lineDog(frame: number, mode: DogMode, level: ColorLevel): string {
  const f = mode === 'sleep' ? Math.floor(frame / 8) : frame;
  const set = level === 0 ? (mode === 'run' ? ASCII_LINE_RUN : ASCII_LINE_SLEEP) : (mode === 'run' ? TINY_RUN : TINY_SLEEP);
  const i = ((f % set.length) + set.length) % set.length;
  return level === 0 ? (set as string[])[i]! : renderBraille((set as string[][])[i]!, level);
}

export interface HeaderOpts {
  cols: number;
  level: ColorLevel;
  frame: number;
  mode: 'run' | 'sleep';
  title: string;
  right?: string;
}

/** Two lines: dog, wordmark, title and right-aligned text; then a rule. */
export function renderHeader(o: HeaderOpts): string[] {
  const P = paints(o.level);
  const dog = lineDog(o.frame, o.mode, o.level);
  const mark = P.accent(MARK) + P.bold(P.accent(WORDMARK));
  let left = dog + '  ' + mark;
  const room = o.cols - visibleWidth(left) - 2;
  const right = o.right ?? '';
  const rw = [...right].length;
  let title = o.title;
  if (right && room - rw - 2 >= Math.min(8, [...title].length)) {
    title = clip(title, room - rw - 2);
    left += '  ' + P.text(title);
    const gap = o.cols - visibleWidth(left) - rw;
    left += ' '.repeat(Math.max(1, gap)) + P.chrome(right);
  } else if (title && room > 0) {
    left += '  ' + P.text(clip(title, room));
  }
  return fit([left, P.chrome(ruleChar().repeat(Math.max(0, o.cols)))], o.cols);
}

// ---------------------------------------------------------------------------------------------
// run screen

export interface VmInfo {
  state: string;
  gpu?: string;
  costUsd?: number;
}

export interface RunOpts {
  cols: number;
  rows: number;
  level: ColorLevel;
  frame: number;
  project: string;
  vm?: VmInfo;
  /** clock for the elapsed time; defaults to Date.now() */
  now?: number;
}

const STATE_ROLE: Record<TicketState, Role> = { waiting: 'chrome', running: 'accent', done: 'ok', failed: 'error', skipped: 'chrome' };

function stateIcon(s: TicketState, frame: number): string {
  switch (s) {
    case 'waiting': return GLYPHS.todo;
    case 'running': return GLYPHS.spinner[frame % GLYPHS.spinner.length]!;
    case 'done': return GLYPHS.done;
    case 'failed': return GLYPHS.failed;
    case 'skipped': return GLYPHS.backlog;
  }
}

function vmRole(state: string): Role {
  const s = state.toLowerCase();
  if (/error|fail|dead/.test(s)) return 'error';
  if (/boot|start|provision|resum|pending/.test(s)) return 'warn';
  if (/stop|off|sleep|suspend|closed/.test(s)) return 'paused';
  return 'accent';
}

function meter(used: number, total: number, P: Paint): string {
  const cells = Math.max(1, Math.min(16, total || 1));
  const full = total > 0 ? Math.round((Math.min(used, total) / total) * cells) : 0;
  return P.accent(GLYPHS.barFull.repeat(full)) + P.chrome(GLYPHS.barEmpty.repeat(cells - full));
}

function logRole(line: string): Role {
  if (/\bFAIL|failed|refused|error|FAILED/i.test(line)) return 'error';
  if (/\bwarn/i.test(line)) return 'warn';
  if (/ done\b| ok\b|integrate ok/.test(line)) return 'ok';
  return 'chrome';
}

function sectionRule(label: string, cols: number, P: Paint): string {
  const r = ruleChar();
  const head = `${r}${r} ${label} `;
  return P.chrome(head + r.repeat(Math.max(0, cols - [...head].length)));
}

/** The run screen: header, VM meter, ticket table, log tail, and the summary once there is one. */
export function renderRun(view: RunView, o: RunOpts): string[] {
  const P = paints(o.level);
  const { cols, rows } = o;
  const sep = P.chrome(` ${GLYPHS.dot} `);
  const n = view.tickets.length;
  const count = (s: TicketState) => view.tickets.filter((t) => t.state === s).length;
  const done = count('done');
  const now = o.now ?? Date.now();

  // header
  const right: string[] = [];
  if (n) right.push(`${done}/${n} done`);
  if (view.summary) right.push(fmtMs(view.summary.wallMs));
  else if (view.startedAt) right.push(fmtMs(now - view.startedAt));
  const out = renderHeader({ cols, level: o.level, frame: o.frame, mode: view.running > 0 ? 'run' : 'sleep', title: o.project, right: right.join(` ${GLYPHS.dot} `) });

  // VM meter
  const vmParts: string[] = [];
  if (o.vm) {
    vmParts.push(P.chrome('vm ') + P.role(vmRole(o.vm.state), `${GLYPHS.on} ${o.vm.state}`));
    if (o.vm.gpu) vmParts.push(P.chrome('gpu ') + P.text(o.vm.gpu));
  } else vmParts.push(P.chrome('vm ') + P.text('local'));
  vmParts.push(P.chrome('agents ') + meter(view.running, view.concurrency, P) + ' ' + P.text(`${view.running}/${view.concurrency || '-'}`));
  if (o.vm?.costUsd !== undefined) vmParts.push(P.chrome('cost ') + P.text(`$${o.vm.costUsd.toFixed(2)}`));
  out.push(...wrapPieces(vmParts, cols, sep).slice(0, 1), '');

  // summary (built first: it has priority over the log for rows)
  const summary: string[] = [];
  const s = view.summary;
  if (s) {
    const counts = [
      P.ok(`${GLYPHS.done} ${s.done} done`),
      (s.failed ? P.error : P.chrome)(`${GLYPHS.failed} ${s.failed} failed`),
    ];
    if (s.skipped) counts.push(P.chrome(`${GLYPHS.backlog} ${s.skipped} skipped`));
    const metrics = [
      P.chrome('first-try ') + P.text(`${s.firstTry}/${s.tickets}`),
      P.chrome('tokens ') + P.text(`${fmtTok(s.promptTokens)} in / ${fmtTok(s.completionTokens)} out`),
      P.chrome('calls ') + P.text(String(s.calls)),
      P.chrome('peak ') + P.text(`${s.peakParallel} parallel`),
      P.chrome('speed-up ') + P.accent(s.wallMs > 0 ? `${(s.serialMs / s.wallMs).toFixed(1)}x` : '-') + P.chrome(` (${fmtMs(s.serialMs)} serial / ${fmtMs(s.wallMs)} wall)`),
    ];
    if (s.integrated !== null) metrics.push(s.integrated ? P.ok('integrated') : P.error('integration failed'));
    summary.push('', sectionRule('summary', cols, P), ...wrapPieces(counts, cols, '  '), ...wrapPieces(metrics, cols, sep));
  }

  // ticket table
  let avail = rows - out.length - summary.length;
  if (!n) {
    out.push(P.chrome(`${GLYPHS.todo} waiting for a plan${GLYPHS.ellipsis}`));
    avail--;
  } else if (avail > 1) {
    const idW = Math.max(2, Math.min(20, Math.max(...view.tickets.map((t) => [...t.id].length))));
    const showModel = cols >= 50;
    const modelW = showModel ? Math.max(5, Math.min(18, Math.max(...view.tickets.map((t) => [...t.model].length)))) : 0;
    const fixedW = 2 + idW + 2 + (showModel ? modelW + 2 : 0) + 3 + 2;
    const noteW = cols - fixedW;
    const head = '  ' + padR('id', idW) + '  ' + (showModel ? padR('model', modelW) + '  ' : '') + 'try' + (noteW >= 4 ? '  note' : '');
    out.push(P.chrome(head));
    avail--;
    const logWant = view.log.length ? Math.min(4, view.log.length + 1) : 0;
    let shown = Math.min(n, Math.max(1, avail - logWant));
    const hidden = n - shown;
    if (hidden > 0) shown = Math.max(0, shown - 1);
    // keep running and failed tickets in view first, then the rest in plan order
    const rank: Record<TicketState, number> = { running: 0, failed: 1, waiting: 2, done: 3, skipped: 4 };
    const pick = hidden > 0
      ? new Set([...view.tickets].sort((a, b) => rank[a.state] - rank[b.state]).slice(0, shown))
      : new Set(view.tickets);
    for (const t of view.tickets) {
      if (!pick.has(t)) continue;
      const role = STATE_ROLE[t.state];
      const note = t.state === 'failed' && t.error ? t.error.split('\n')[0]! : t.lastNote || t.title;
      let line = P.role(role, stateIcon(t.state, o.frame)) + ' ' + P.role(role, padR(t.id, idW)) + '  ';
      if (showModel) line += P.chrome(padR(t.model, modelW)) + '  ';
      line += P.text(padR(t.attempt ? `#${t.attempt}` : '', 3));
      if (noteW >= 4) line += '  ' + (t.state === 'failed' ? P.error(clip(note, noteW)) : P.chrome(clip(note, noteW)));
      out.push(line);
      avail--;
    }
    if (hidden > 0) {
      const more = n - shown;
      out.push(P.chrome(`  ${GLYPHS.ellipsis} ${more} more`));
      avail--;
    }
  }

  // log tail
  if (view.log.length && avail >= 2) {
    out.push(sectionRule('log', cols, P));
    avail--;
    for (const l of view.log.slice(-avail)) out.push(P.role(logRole(l), clip(l.replace(/[\r\n\t]+/g, ' '), cols)));
  }

  out.push(...summary);
  return fit(out.slice(0, Math.max(0, rows)), cols);
}

// ---------------------------------------------------------------------------------------------
// home screen

export interface HomeProject {
  name: string;
  state: 'open' | 'closed' | 'booting';
  repo?: string;
  lastOpened?: number;
}

export interface HomeOpts {
  cols: number;
  rows: number;
  level: ColorLevel;
  frame: number;
  selected: number;
  now?: number;
}

function bigDog(frame: number, mode: DogMode, level: ColorLevel): { lines: string[]; width: number } {
  const lines = renderDog(frame, { mode, size: 'full', level });
  return { lines, width: Math.max(dogWidth('full', level, mode), ...lines.map(visibleWidth)) };
}

/** Key hints like "enter open · n new": keys bright, labels dim. */
function hints(pairs: Array<[string, string]>, P: Paint): string {
  return pairs.map(([k, l]) => P.accent(k) + ' ' + P.chrome(l)).join(P.chrome(`  ${GLYPHS.dot} `));
}

/** The home screen: the big dog, the project list and the key hints, filling `rows`. */
export function renderHome(projects: HomeProject[], o: HomeOpts): string[] {
  const P = paints(o.level);
  const { cols, rows } = o;
  const now = o.now ?? Date.now();
  const out: string[] = [];
  const footer = [
    P.chrome(ruleChar().repeat(Math.max(0, cols))),
    hints([[ascii() ? 'up/down' : '↑↓', 'move'], ['enter', 'open'], ['n', 'new'], ['c', 'close'], ['q', 'quit']], P),
  ];
  const listWant = Math.max(1, projects.length);
  const mode: DogMode = projects.some((p) => p.state === 'booting' || p.state === 'open') ? 'run' : 'sleep';
  const dog = bigDog(o.frame, mode, o.level);
  const word = P.accent(MARK) + P.bold(P.accent(WORDMARK));
  const tag = P.chrome('parallel coding with small models');

  // top: dog (when it fits), wordmark, tagline
  const dogRows = dog.lines.length + 1;
  const topBase = 3; // wordmark, tagline, blank
  const withDog = cols >= dog.width + 2 && rows - footer.length - topBase - dogRows >= Math.min(listWant, 3) + 1;
  out.push('');
  if (withDog) {
    const x = Math.max(0, Math.floor((cols - dog.width) / 2));
    for (const l of dog.lines) out.push(' '.repeat(x) + l);
  }
  out.push(center(word, cols));
  if (rows - footer.length - out.length >= 3) out.push(center(tag, cols));
  out.push('');

  // project list
  const listRows = Math.max(1, rows - footer.length - out.length);
  if (!projects.length) {
    out.push(center(P.chrome('no projects yet ') + P.accent('n') + P.chrome(' creates one'), cols));
  } else {
    const sel = Math.max(0, Math.min(projects.length - 1, o.selected));
    const start = Math.max(0, Math.min(projects.length - listRows, sel - Math.floor(listRows / 2)));
    const visible = projects.slice(start, start + listRows);
    const nameW = Math.max(4, Math.min(24, Math.max(...projects.map((p) => [...p.name].length))));
    const width = Math.min(cols, Math.max(48, nameW + 40));
    const x = Math.max(0, Math.floor((cols - width) / 2));
    visible.forEach((p, k) => {
      const i = start + k;
      const on = i === sel;
      const icon = p.state === 'open' ? P.accent(GLYPHS.on) : p.state === 'booting' ? P.warn(GLYPHS.spinner[o.frame % GLYPHS.spinner.length]!) : P.chrome(GLYPHS.off);
      const label = p.state === 'open' ? P.accent(padR('open', 8)) : p.state === 'booting' ? P.warn(padR('booting', 8)) : P.chrome(padR('closed', 8));
      let line = ' '.repeat(x) + (on ? P.accent(GLYPHS.cursor) : ' ') + ' ' + icon + ' ' + (on ? P.bold(P.accent(padR(p.name, nameW))) : P.text(padR(p.name, nameW))) + '  ' + label;
      const ago = p.lastOpened ? fmtAgo(p.lastOpened, now) : '';
      const restW = cols - visibleWidth(line) - 2;
      if (restW > 4) {
        const repoW = restW - (ago ? ago.length + 2 : 0);
        let tail = p.repo && repoW > 4 ? clip(p.repo, repoW) : '';
        if (ago) tail = padR(tail, Math.max(0, restW - ago.length - 2)) + '  ' + ago;
        if (tail) line += '  ' + P.chrome(tail);
      }
      out.push(line);
    });
  }

  while (out.length < rows - footer.length) out.push('');
  out.push(...footer);
  return fit(out.slice(Math.max(0, out.length - rows)), cols);
}

// ---------------------------------------------------------------------------------------------
// splash

export interface SplashOpts {
  cols: number;
  rows: number;
  level: ColorLevel;
  ms: number;
  seed: number;
}

/** One frame of matrix rain with the big dog and the wordmark cut out in the middle. */
export function renderSplash(o: SplashOpts): string[] {
  const P = paints(o.level);
  const cols = Math.max(0, o.cols);
  const rows = Math.max(0, o.rows);
  const rain = makeRain(cols, rows, o.seed, !ascii());
  const dog = bigDog(Math.floor(o.ms / 100), 'run', o.level);
  const word = P.accent(MARK) + P.bold(P.accent(WORDMARK));
  let content: string[] = [...dog.lines, '', center(word, dog.width)];
  let boxW = dog.width + 4;
  if (boxW > cols || content.length + 2 > rows) {
    content = [word];
    boxW = Math.min(cols, 1 + WORDMARK.length + 4);
  }
  const boxH = Math.min(rows, content.length + 2);
  const y0 = Math.max(0, Math.floor((rows - boxH) / 2));
  const x0 = Math.max(0, Math.floor((cols - boxW) / 2));
  const out: string[] = [];
  for (let r = 0; r < rows; r++) {
    const k = r - y0;
    if (k < 0 || k >= boxH) { out.push(rainSpan(rain, r, o.ms, o.level)); continue; }
    const inner = k >= 1 && k <= content.length ? content[k - 1]! : '';
    const cell = padAnsi('  ' + inner, boxW);
    out.push(rainSpan(rain, r, o.ms, o.level, 0, x0) + truncateAnsi(cell, boxW) + rainSpan(rain, r, o.ms, o.level, x0 + boxW, cols));
  }
  return fit(out, cols);
}
