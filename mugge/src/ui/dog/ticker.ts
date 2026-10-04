// Adapted from salu (github.com/OliverVillson/salu) src/tui/dog/ticker.ts
/**
 * One shared animation timer for every dog on screen. It only runs while something listens and
 * the `active` flag is set, so an idle TUI has no timer at all and keypress latency is untouched.
 * Subscribers get the tick number; each redraws only its own component.
 */
export const DOG_FPS = 10;

type Listener = (tick: number) => void;

export class Ticker {
  private listeners = new Set<Listener>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private n = 0;
  constructor(private fps = DOG_FPS) {}

  get tick(): number { return this.n; }
  get running(): boolean { return this.timer !== null; }
  get size(): number { return this.listeners.size; }

  /** Subscribe; starts the timer on the first listener. Returns an unsubscribe function. */
  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    if (!this.timer) {
      this.timer = setInterval(() => {
        this.n++;
        for (const l of this.listeners) l(this.n);
      }, Math.round(1000 / this.fps));
      this.timer.unref?.();
    }
    return () => {
      this.listeners.delete(fn);
      if (this.listeners.size === 0 && this.timer) { clearInterval(this.timer); this.timer = null; }
    };
  }
}

export const sharedTicker = new Ticker();
