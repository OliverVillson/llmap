/**
 * The interactive terminal loops: the run screen, the home screen and the dog demo.
 *
 * Drawing: the alternate screen is NOT used, so whatever the run printed stays in scrollback
 * after you quit (the last frame is left on screen). Each frame moves the cursor back up over
 * the lines the previous frame drew, rewrites them (erase-line before each) and clears anything
 * below. A frame is at most `rows - 1` lines so the region never scrolls out of reach of the
 * cursor-up; on resize the visible screen is cleared once and drawing restarts at the top.
 * When stdout is not a TTY nothing animates: one plain render is printed and the call returns.
 */
import { applyEvent, emptyView, type RunView } from '../engine/events.ts';
import { DOG_FPS } from './dog/ticker.ts';
import { dogWidth, renderDog, renderTrack } from './dog/render.ts';
import { renderHome, renderRun, renderSplash, type HomeProject, type VmInfo } from './screen.ts';
import { detectColorLevel, type ColorLevel } from './theme.ts';

export { applyEvent, emptyView };

const ESC = '\u001b[';
const SPLASH_MS = 1500;

/** In-place redrawing region on the normal screen. */
class Region {
  private drawn = 0;
  private out = process.stdout;
  private onResize = () => {
    this.out.write(`${ESC}2J${ESC}H`);
    this.drawn = 0;
  };
  constructor() {
    this.out.write(`${ESC}?25l`);
    this.out.on('resize', this.onResize);
  }
  get cols(): number { return Math.max(20, this.out.columns || 80); }
  get rows(): number { return Math.max(6, (this.out.rows || 24) - 1); }
  draw(lines: string[]): void {
    let s = this.drawn ? `\r${ESC}${this.drawn}A` : '\r';
    for (const l of lines) s += `${ESC}2K${l}\n`;
    s += `${ESC}J`;
    this.out.write(s);
    this.drawn = lines.length;
  }
  /** Erases what this region drew (for handing the terminal to a prompt). */
  clear(): void {
    if (this.drawn) this.out.write(`\r${ESC}${this.drawn}A${ESC}J`);
    this.drawn = 0;
  }
  close(): void {
    this.out.off('resize', this.onResize);
    this.out.write(`${ESC}0m${ESC}?25h`);
  }
}

type KeyHandler = (key: string) => void;

/** Raw-mode keyboard; returns a restore function. Safe to call when stdin is not a TTY. */
function rawKeys(onKey: KeyHandler): () => void {
  const stdin = process.stdin;
  const tty = !!stdin.isTTY;
  if (tty) stdin.setRawMode(true);
  stdin.resume();
  stdin.setEncoding('utf8');
  const h = (d: string | Buffer) => onKey(String(d));
  stdin.on('data', h);
  return () => {
    stdin.off('data', h);
    if (tty) stdin.setRawMode(false);
    stdin.pause();
  };
}

const KEY = { up: ['\u001b[A', '\u001bOA', 'k'], down: ['\u001b[B', '\u001bOB', 'j'], enter: ['\r', '\n'], quit: ['q', '\u0003', '\u001b'] };

/** Sets up terminal teardown on signals and exit; returns the function that undoes it. */
function guard(restore: () => void): () => void {
  let done = false;
  const once = () => { if (!done) { done = true; restore(); } };
  const sig = () => { once(); process.exit(130); };
  process.on('SIGINT', sig);
  process.on('SIGTERM', sig);
  process.on('exit', once);
  return () => {
    process.off('SIGINT', sig);
    process.off('SIGTERM', sig);
    process.off('exit', once);
    once();
  };
}

/** Plays the splash in `region` until SPLASH_MS passes or a key is pressed. */
function playSplash(region: Region, level: ColorLevel, keys: { skip: (() => void) | null }): Promise<void> {
  return new Promise((resolve) => {
    const seed = (Math.random() * 0x7fffffff) | 0;
    const t0 = Date.now();
    let timer: ReturnType<typeof setInterval>;
    const end = () => { clearInterval(timer); keys.skip = null; resolve(); };
    keys.skip = end;
    const frame = () => {
      const ms = Date.now() - t0;
      if (ms >= SPLASH_MS) return end();
      region.draw(renderSplash({ cols: region.cols, rows: region.rows, level, ms, seed }));
    };
    timer = setInterval(frame, 50);
    frame();
  });
}

export interface TuiOptions {
  /** push-style updates; returns an unsubscribe function */
  subscribe?: (fn: (v: RunView) => void) => () => void;
  /** pull-style updates, called about twice a second */
  poll?: () => Promise<RunView>;
  project: string;
  vm?: () => VmInfo;
  onQuit?: () => void;
  /** skip the splash (default: play it) */
  splash?: boolean;
}

/** Waits (briefly) for one view from whichever source there is. */
async function oneView(o: TuiOptions): Promise<RunView> {
  if (o.poll) return o.poll().catch(() => emptyView());
  if (!o.subscribe) return emptyView();
  const sub = o.subscribe;
  return new Promise<RunView>((resolve) => {
    let settled = false;
    let unsub: (() => void) | null = null;
    const finish = (v: RunView) => {
      if (settled) return;
      settled = true;
      clearTimeout(t);
      queueMicrotask(() => unsub?.());
      resolve(v);
    };
    const t = setTimeout(() => finish(emptyView()), 1000);
    unsub = sub(finish);
    if (settled) unsub();
  });
}

/**
 * The live run screen. Resolves when the user quits (q, Esc or Ctrl-C); `p` is reserved.
 * Not a TTY: prints one plain render of the current view and resolves.
 */
export async function openTui(o: TuiOptions): Promise<void> {
  const level = detectColorLevel(process.env, !!process.stdout.isTTY);
  if (!process.stdout.isTTY) {
    const v = await oneView(o);
    const lines = renderRun(v, { cols: process.stdout.columns || 100, rows: 1000, level, frame: 0, project: o.project, vm: o.vm?.() });
    process.stdout.write(lines.join('\n') + '\n');
    o.onQuit?.();
    return;
  }

  const region = new Region();
  let view: RunView = emptyView();
  let frame = 0;
  let quitting = false;
  const keys: { skip: (() => void) | null } = { skip: null };
  let resolveQuit!: () => void;
  const quitP = new Promise<void>((r) => (resolveQuit = r));

  const restoreKeys = rawKeys((k) => {
    if (keys.skip) { keys.skip(); if (!KEY.quit.includes(k)) return; }
    if (KEY.quit.includes(k)) { quitting = true; resolveQuit(); }
    // 'p' is reserved
  });
  const unguard = guard(() => { restoreKeys(); region.close(); });

  const unsub = o.subscribe?.((v) => { view = v; });
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  if (o.poll) {
    const poll = o.poll;
    const tick = async () => {
      try { view = await poll(); } catch { /* keep the last view */ }
      if (!quitting) pollTimer = setTimeout(tick, 500);
    };
    void tick();
  }

  if (o.splash !== false) await Promise.race([playSplash(region, level, keys), quitP]);

  const draw = () => {
    let vm: VmInfo | undefined;
    try { vm = o.vm?.(); } catch { vm = undefined; }
    region.draw(renderRun(view, { cols: region.cols, rows: region.rows, level, frame, project: o.project, vm }));
  };
  const timer = quitting ? null : setInterval(() => { frame++; draw(); }, Math.round(1000 / DOG_FPS));
  if (!quitting) draw();

  await quitP;
  if (timer) clearInterval(timer);
  if (pollTimer) clearTimeout(pollTimer);
  unsub?.();
  draw(); // leave the last frame in scrollback
  unguard();
  o.onQuit?.();
}

export interface HomeOptions {
  /** the list, or a function that (re)loads it; reloaded after every action and every 2 s */
  projects: HomeProject[] | (() => HomeProject[] | Promise<HomeProject[]>);
  selected?: number;
  /**
   * `n` (new) and `c` (close). Runs with the terminal handed back (raw mode off, screen region
   * cleared) so it may prompt; the home screen comes back when it resolves. Without it those
   * keys do nothing.
   */
  onAction?: (action: 'new' | 'close', project: string | null) => void | Promise<void>;
  /** skip the splash (default: play it) */
  splash?: boolean;
}

/**
 * The home screen: resolves with the chosen project's name on enter, or null on q / Esc /
 * Ctrl-C. Not a TTY: prints the list once and resolves null.
 */
export async function openHome(o: HomeOptions): Promise<string | null> {
  const load = async (): Promise<HomeProject[]> => (typeof o.projects === 'function' ? await o.projects() : o.projects);
  let projects = await load().catch(() => [] as HomeProject[]);
  const level = detectColorLevel(process.env, !!process.stdout.isTTY);
  if (!process.stdout.isTTY) {
    const lines = renderHome(projects, { cols: process.stdout.columns || 80, rows: Math.max(12, projects.length + 6), level, frame: 0, selected: o.selected ?? 0 });
    process.stdout.write(lines.filter((l, i, a) => l.trim() || (i > 0 && a[i - 1]!.trim())).join('\n') + '\n');
    return null;
  }

  let selected = Math.max(0, o.selected ?? 0);
  let frame = 0;
  let busy = false;
  const region = new Region();
  const keys: { skip: (() => void) | null } = { skip: null };
  let finish!: (v: string | null) => void;
  const result = new Promise<string | null>((r) => (finish = r));
  let done = false;

  const draw = () => {
    if (busy) return;
    selected = Math.min(selected, Math.max(0, projects.length - 1));
    region.draw(renderHome(projects, { cols: region.cols, rows: region.rows, level, frame, selected }));
  };

  let restoreKeys: () => void = () => {};
  const onKey = (k: string) => {
    if (keys.skip) { keys.skip(); if (!KEY.quit.includes(k)) return; }
    if (busy || done) return;
    if (KEY.quit.includes(k)) { done = true; finish(null); return; }
    if (KEY.up.includes(k)) { selected = Math.max(0, selected - 1); draw(); return; }
    if (KEY.down.includes(k)) { selected = Math.min(Math.max(0, projects.length - 1), selected + 1); draw(); return; }
    if (KEY.enter.includes(k)) { const p = projects[selected]; if (p) { done = true; finish(p.name); } return; }
    if ((k === 'n' || k === 'c') && o.onAction) {
      const action = k === 'n' ? 'new' : 'close';
      const name = projects[selected]?.name ?? null;
      if (action === 'close' && !name) return;
      busy = true;
      restoreKeys();
      region.clear();
      process.stdout.write(`${ESC}?25h`);
      void (async () => {
        try { await o.onAction!(action, name); } catch (e) { process.stderr.write(`${(e as Error).message}\n`); }
        projects = await load().catch(() => projects);
        process.stdout.write(`${ESC}?25l`);
        restoreKeys = rawKeys(onKey);
        busy = false;
        draw();
      })();
    }
  };
  restoreKeys = rawKeys(onKey);
  const unguard = guard(() => { restoreKeys(); region.close(); });

  if (o.splash !== false) await Promise.race([playSplash(region, level, keys), result]);

  const timer = setInterval(() => { frame++; draw(); }, Math.round(1000 / DOG_FPS));
  const reload = typeof o.projects === 'function'
    ? setInterval(() => { if (!busy) void load().then((p) => { projects = p; }).catch(() => {}); }, 2000)
    : null;
  draw();

  const choice = await result;
  clearInterval(timer);
  if (reload) clearInterval(reload);
  region.clear();
  unguard();
  return choice;
}

/** `mugge dog`: the full-size dog runs across the terminal, then curls up. Any key stops it. */
export async function dogDemo(o: { ms?: number } = {}): Promise<void> {
  const level = detectColorLevel(process.env, !!process.stdout.isTTY);
  if (!process.stdout.isTTY) {
    process.stdout.write(renderDog(0, { size: 'full', level }).join('\n') + '\n');
    return;
  }
  const total = o.ms ?? 5000;
  const region = new Region();
  let stop!: () => void;
  const stopped = new Promise<void>((r) => (stop = r));
  const restoreKeys = rawKeys(() => stop());
  const unguard = guard(() => { restoreKeys(); region.close(); });
  const width = Math.min(region.cols, 72);
  const lapMs = Math.max(1000, total * 0.7);
  const t0 = Date.now();
  let tick = 0;
  const timer = setInterval(() => {
    const ms = Date.now() - t0;
    if (ms >= total) return stop();
    tick++;
    const lines = ms < lapMs
      ? renderTrack(tick, width, ms / lapMs, { size: 'full', level })
      : renderDog(tick, { mode: 'sleep', size: 'full', level }).map((l) => ' '.repeat(Math.max(0, width - dogWidth('full', level, 'sleep'))) + l);
    region.draw(lines);
  }, Math.round(1000 / DOG_FPS));
  await stopped;
  clearInterval(timer);
  unguard();
}
