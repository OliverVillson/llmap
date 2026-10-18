/**
 * Where acceptance commands run. On the project VM that is a throwaway container from the
 * ticket's toolchain image (sandbox/images/<image>), with the ticket's worktree mounted at
 * /work, no network, and CPU/memory caps. Sandboxes need CPU, not GPU, and never see weights.
 *
 * `LocalSandbox` runs the command straight on the host. It is for tests and CI, where the
 * toolchains are installed on the machine and the code is our own.
 */
import type { Image } from '../tickets/schema.ts';

export interface RunResult {
  code: number;
  /** stdout and stderr interleaved as far as the runtime allows. */
  out: string;
  ms: number;
  timedOut: boolean;
}

export interface RunOptions {
  cwd: string;
  image: Image;
  timeoutMs?: number;
}

export interface Sandbox {
  readonly name: string;
  run(command: string, o: RunOptions): Promise<RunResult>;
}

export const DEFAULT_TIMEOUT_MS = 120_000;

/** How long the output may still take to end once a timed-out command was killed (a child that left the group). */
const DRAIN_MS = 5_000;

/** Process groups of commands still running: killed when this process exits, since detached ones outlive it. */
const groups = new Set<number>();
let exitHook = false;
const killGroup = (pid: number) => {
  try {
    process.kill(-pid, 'SIGKILL');
    return true;
  } catch {
    return false;
  }
};

async function spawnWithTimeout(argv: string[], cwd: string | undefined, timeoutMs: number): Promise<RunResult> {
  const started = performance.now();
  // its own process group, so a timeout kills what it started too: a test that runs the CLI in a
  // subprocess which never ends would otherwise hold the output pipes open, and the check forever
  const p = Bun.spawn(argv, { cwd, stdout: 'pipe', stderr: 'pipe', stdin: 'ignore', env: process.env, detached: true } as Parameters<typeof Bun.spawn>[1]);
  groups.add(p.pid);
  if (!exitHook) {
    exitHook = true;
    process.on('exit', () => groups.forEach(killGroup));
  }
  let timedOut = false;
  let drained: () => void = () => {};
  const cutOff = new Promise<void>((resolve) => (drained = resolve));
  const timer = setTimeout(() => {
    timedOut = true;
    if (!killGroup(p.pid)) p.kill('SIGKILL');
    setTimeout(drained, DRAIN_MS);
  }, timeoutMs);
  const text = (s: ReadableStream<Uint8Array>) => {
    const r = s.getReader();
    const chunks: Uint8Array[] = [];
    const pump = (): Promise<string> => r.read().then(({ done, value }) => (done ? Buffer.concat(chunks).toString() : (chunks.push(value), pump())));
    return Promise.race([pump(), cutOff.then(() => (r.cancel().catch(() => {}), Buffer.concat(chunks).toString()))]);
  };
  const [out, err] = await Promise.all([text(p.stdout as ReadableStream<Uint8Array>), text(p.stderr as ReadableStream<Uint8Array>)]);
  const code = await p.exited;
  clearTimeout(timer);
  groups.delete(p.pid);
  return { code: timedOut ? 124 : code, out: (out + err).trim(), ms: Math.round(performance.now() - started), timedOut };
}

export class LocalSandbox implements Sandbox {
  readonly name = 'local';
  run(command: string, o: RunOptions): Promise<RunResult> {
    return spawnWithTimeout(['bash', '-c', command], o.cwd, o.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  }
}

export interface ContainerOptions {
  /** `podman` (rootless, the VM default) or `docker`. */
  engine?: 'podman' | 'docker';
  /** Image name prefix; images are `<prefix>-<image>`, built by sandbox/build.sh. */
  prefix?: string;
  /** OCI runtime, e.g. `runsc` for gVisor. */
  runtime?: string;
  memory?: string;
  cpus?: number;
}

export class ContainerSandbox implements Sandbox {
  readonly name: string;
  constructor(private o: ContainerOptions = {}) {
    this.name = o.engine ?? 'podman';
  }

  argv(command: string, opts: RunOptions): string[] {
    const o = this.o;
    return [
      o.engine ?? 'podman', 'run', '--rm', '--network', 'none',
      ...(o.runtime ? ['--runtime', o.runtime] : []),
      '--memory', o.memory ?? '2g', '--cpus', String(o.cpus ?? 2), '--pids-limit', '512',
      '--security-opt', 'no-new-privileges', '--cap-drop', 'ALL',
      // Run as the worktree's owner so files the checks write stay editable on the host.
      ...((o.engine ?? 'podman') === 'podman' ? ['--userns', 'keep-id'] : ['--user', `${process.getuid?.() ?? 1000}:${process.getgid?.() ?? 1000}`]),
      '-e', 'HOME=/tmp',
      '-v', `${opts.cwd}:/work`, '-w', '/work',
      `${o.prefix ?? 'mugge'}-${opts.image}`, 'bash', '-c', command,
    ];
  }

  run(command: string, o: RunOptions): Promise<RunResult> {
    return spawnWithTimeout(this.argv(command, o), undefined, o.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  }
}

/** `MUGGE_SANDBOX=local|podman|docker` picks the sandbox; podman when unset. */
export function sandboxFromEnv(env: NodeJS.ProcessEnv = process.env): Sandbox {
  const kind = env.MUGGE_SANDBOX ?? 'podman';
  if (kind === 'local') return new LocalSandbox();
  if (kind !== 'podman' && kind !== 'docker') throw new Error(`MUGGE_SANDBOX must be local, podman or docker, got ${kind}`);
  return new ContainerSandbox({ engine: kind, runtime: env.MUGGE_OCI_RUNTIME || undefined });
}

/** Keeps the end of a long log, where compilers and test runners put the part that matters. */
export function tail(text: string, maxChars = 3000): string {
  if (text.length <= maxChars) return text;
  return '…' + text.slice(text.length - maxChars);
}
