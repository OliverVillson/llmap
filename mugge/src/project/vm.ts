/**
 * The project VM, seen from the laptop. `mugge open` boots it (or finds it running), pushes the
 * private state store up, makes sure the engine API is running there and opens an SSH tunnel
 * to it. `mugge close` pulls state back down and stops the VM, so billing stops.
 *
 * Same pattern lobbot uses to reach its VM. Everything goes through the system `ssh`/`rsync`,
 * so the user's keys and ssh config apply. Every command is built by a pure function first,
 * which is what the tests check; `run` is injectable.
 */
import type { Project, VmConfig } from './project.ts';
import { stateDir } from './project.ts';

export const DEFAULT_ENGINE_PORT = 7341;
export const DEFAULT_REMOTE_DIR = '~/mugge';

export type Runner = (argv: string[], opts?: { input?: string }) => { code: number; out: string };

export const defaultRunner: Runner = (argv) => {
  const r = Bun.spawnSync(argv, { stdout: 'pipe', stderr: 'pipe' });
  return { code: r.exitCode ?? 1, out: (r.stdout.toString() + r.stderr.toString()).trim() };
};

function sshBase(vm: VmConfig): string[] {
  return [
    'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
    ...(vm.port ? ['-p', String(vm.port)] : []),
    ...(vm.identity ? ['-i', vm.identity] : []),
    `${vm.user}@${vm.host}`,
  ];
}

export function sshCommand(vm: VmConfig, remote: string): string[] {
  return [...sshBase(vm), remote];
}

export function tunnelCommand(vm: VmConfig, localPort: number): string[] {
  const base = sshBase(vm);
  const target = base.pop()!;
  return [...base, '-N', '-o', 'ExitOnForwardFailure=yes', '-L', `127.0.0.1:${localPort}:127.0.0.1:${vm.enginePort ?? DEFAULT_ENGINE_PORT}`, target];
}

function rsyncSsh(vm: VmConfig): string {
  return ['ssh', '-o', 'BatchMode=yes', ...(vm.port ? ['-p', String(vm.port)] : []), ...(vm.identity ? ['-i', vm.identity] : [])].join(' ');
}

/** rsync the private state store up (`push`) or down (`pull`). */
export function syncStateCommand(p: Project, direction: 'push' | 'pull'): string[] {
  const vm = p.vm!;
  const remote = `${vm.user}@${vm.host}:${vm.remoteDir ?? DEFAULT_REMOTE_DIR}/${p.name}/state/`;
  const local = `${stateDir(p.name)}/`;
  return ['rsync', '-az', '--delete', '-e', rsyncSsh(vm), ...(direction === 'push' ? [local, remote] : [remote, local])];
}

/** Starts the engine API on the VM if it is not answering yet. */
export function ensureEngineRemote(p: Project): string {
  const vm = p.vm!;
  const port = vm.enginePort ?? DEFAULT_ENGINE_PORT;
  const dir = `${vm.remoteDir ?? DEFAULT_REMOTE_DIR}/${p.name}`;
  return [
    `mkdir -p ${dir}/state`,
    `if ! curl -fsS http://127.0.0.1:${port}/health >/dev/null 2>&1; then`,
    `  nohup mugge engine serve --port ${port} --project ${dir} >${dir}/engine.log 2>&1 &`,
    `fi`,
  ].join('\n');
}

const fill = (cmd: string, p: Project) => cmd.replaceAll('{name}', p.name);

export interface VmSession {
  localPort: number;
  url: string;
  close(): void;
}

export class Vm {
  constructor(private p: Project, private run: Runner = defaultRunner) {
    if (!p.vm) throw new Error(`project ${p.name} has no VM configured (mugge vm ${p.name} --host ... --user ...)`);
  }

  private get vm(): VmConfig {
    return this.p.vm!;
  }

  private must(argv: string[], what: string): string {
    const r = this.run(argv);
    if (r.code !== 0) throw new Error(`${what} failed:\n${r.out}`);
    return r.out;
  }

  /** Boots the VM (command driver) and waits for ssh to answer. */
  boot(waitMs = 300_000): void {
    if (this.vm.driver === 'command' && this.vm.start) this.must(['bash', '-c', fill(this.vm.start, this.p)], 'starting the VM');
    const until = Date.now() + waitMs;
    while (this.run(sshCommand(this.vm, 'true')).code !== 0) {
      if (Date.now() > until) throw new Error(`the VM at ${this.vm.host} did not answer ssh`);
      Bun.sleepSync(3000);
    }
  }

  pushState(): void {
    this.must(sshCommand(this.vm, `mkdir -p ${this.vm.remoteDir ?? DEFAULT_REMOTE_DIR}/${this.p.name}/state`), 'making the remote folder');
    this.must(syncStateCommand(this.p, 'push'), 'pushing project state');
  }

  pullState(): void {
    this.must(syncStateCommand(this.p, 'pull'), 'pulling project state');
  }

  startEngine(): void {
    this.must(sshCommand(this.vm, ensureEngineRemote(this.p)), 'starting the engine');
  }

  /** Opens the tunnel to the engine API and returns its local URL. */
  tunnel(localPort = 0): VmSession {
    const port = localPort || 20000 + Math.floor(Math.random() * 20000);
    const proc = Bun.spawn(tunnelCommand(this.vm, port), { stdout: 'ignore', stderr: 'pipe', stdin: 'ignore' });
    return { localPort: port, url: `http://127.0.0.1:${port}`, close: () => proc.kill() };
  }

  /** Stops the VM (command driver); a plain ssh host is left running. */
  stop(): void {
    if (this.vm.driver === 'command' && this.vm.stop) this.must(['bash', '-c', fill(this.vm.stop, this.p)], 'stopping the VM');
  }
}
