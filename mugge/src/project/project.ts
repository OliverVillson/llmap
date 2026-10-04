/**
 * Projects on the user's machine. Mugge is project-centric: a project has a code repo (the
 * user's, clean), a private state store (plans, ARCHITECTURE.md, run logs, memory: never in the
 * code repo), and one VM that runs only while the project is open.
 *
 * Layout: $MUGGE_HOME (default ~/.mugge)/projects/<name>/{project.json, state/}
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

export interface VmConfig {
  /** `ssh`: a host that is always there. `command`: start/stop through your cloud's CLI (evroc). */
  driver: 'ssh' | 'command';
  host: string;
  user: string;
  port?: number;
  identity?: string;
  /** Shell commands for the `command` driver, run locally; `{name}` is replaced by the project name. */
  start?: string;
  stop?: string;
  /** Port of the engine API on the VM (bound to 127.0.0.1 there, reached through the tunnel). */
  enginePort?: number;
  /** Folder on the VM holding the project's checkout and state. */
  remoteDir?: string;
}

export interface Project {
  name: string;
  /** Where shipped code goes: a git remote URL or a local folder. */
  repo: string | null;
  request: string | null;
  vm: VmConfig | null;
  state: 'open' | 'closed';
  createdAt: number;
  lastOpened: number | null;
  /** Author for commits Mugge makes: the user, so nothing carries a stamp. */
  author: { name: string; email: string } | null;
}

export const muggeHome = (env: NodeJS.ProcessEnv = process.env) => env.MUGGE_HOME || join(homedir(), '.mugge');
export const projectsDir = () => join(muggeHome(), 'projects');
export const projectDir = (name: string) => join(projectsDir(), name);
export const stateDir = (name: string) => join(projectDir(name), 'state');

const NAME = /^[a-z0-9][a-z0-9._-]{0,62}$/;

export function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'project';
}

export function listProjects(): Project[] {
  if (!existsSync(projectsDir())) return [];
  return readdirSync(projectsDir())
    .filter((n) => existsSync(join(projectDir(n), 'project.json')))
    .map(loadProject)
    .sort((a, b) => (b.lastOpened ?? b.createdAt) - (a.lastOpened ?? a.createdAt));
}

export function loadProject(name: string): Project {
  const p = join(projectDir(name), 'project.json');
  if (!existsSync(p)) throw new Error(`no project named ${name} (mugge new to make one)`);
  return JSON.parse(readFileSync(p, 'utf8'));
}

export function saveProject(p: Project): void {
  mkdirSync(stateDir(p.name), { recursive: true });
  writeFileSync(join(projectDir(p.name), 'project.json'), JSON.stringify(p, null, 2) + '\n');
}

export function createProject(o: { name: string; request?: string; repo?: string; vm?: VmConfig | null; author?: Project['author'] }): Project {
  if (!NAME.test(o.name)) throw new Error(`project name must match ${NAME}`);
  if (existsSync(join(projectDir(o.name), 'project.json'))) throw new Error(`project ${o.name} already exists`);
  const p: Project = {
    name: o.name,
    repo: o.repo ?? null,
    request: o.request ?? null,
    vm: o.vm ?? null,
    state: 'closed',
    createdAt: Date.now(),
    lastOpened: null,
    author: o.author ?? null,
  };
  saveProject(p);
  writeFileSync(join(stateDir(p.name), 'memory.md'), `# ${p.name}\n\n${p.request ?? ''}\n`);
  return p;
}
