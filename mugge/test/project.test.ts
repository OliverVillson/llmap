import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { followEvents, startEngineServer } from '../src/api/server.ts';
import type { EngineEvent } from '../src/engine/events.ts';
import { startFakeModel } from '../src/engine/fake-model.ts';
import { git, gitOk, initRepo } from '../src/engine/git.ts';
import type { Inference } from '../src/engine/inference.ts';
import { OpenAICompatible } from '../src/engine/inference.ts';
import { LocalSandbox } from '../src/engine/sandbox.ts';
import { architectureMarkdown, design, tickets, type Design } from '../src/plan/architect.ts';
import { createProject, listProjects, loadProject, saveProject, type Project } from '../src/project/project.ts';
import { shipToFolder } from '../src/project/ship.ts';
import { ensureEngineRemote, sshCommand, syncStateCommand, tunnelCommand, Vm, type Runner } from '../src/project/vm.ts';

let dir: string;
beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), 'mugge-project-'));
  process.env.MUGGE_HOME = join(dir, 'home');
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe('projects', () => {
  test('create, list, save', () => {
    const p = createProject({ name: 'chat', request: 'a chat app' });
    expect(readFileSync(join(dir, 'home/projects/chat/state/memory.md'), 'utf8')).toContain('a chat app');
    expect(() => createProject({ name: 'chat' })).toThrow(/exists/);
    expect(() => createProject({ name: 'Bad Name' })).toThrow();
    saveProject({ ...p, state: 'open' });
    expect(loadProject('chat').state).toBe('open');
    expect(listProjects().map((x) => x.name)).toEqual(['chat']);
  });
});

describe('vm commands', () => {
  const p: Project = {
    name: 'chat', repo: null, request: null, state: 'closed', createdAt: 0, lastOpened: null, author: null,
    vm: { driver: 'command', host: 'gpu1.example', user: 'ubuntu', port: 2222, identity: '/k', start: 'evroc vm start {name}', stop: 'evroc vm stop {name}', enginePort: 9000 },
  };
  test('ssh, tunnel and rsync are built from the config', () => {
    expect(sshCommand(p.vm!, 'true')).toEqual(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-p', '2222', '-i', '/k', 'ubuntu@gpu1.example', 'true']);
    const t = tunnelCommand(p.vm!, 5555);
    expect(t).toContain('127.0.0.1:5555:127.0.0.1:9000');
    expect(t.at(-1)).toBe('ubuntu@gpu1.example');
    const push = syncStateCommand(p, 'push');
    expect(push.at(-1)).toBe('ubuntu@gpu1.example:~/mugge/chat/state/');
    expect(syncStateCommand(p, 'pull').at(-2)).toBe('ubuntu@gpu1.example:~/mugge/chat/state/');
    expect(ensureEngineRemote(p)).toContain('mugge engine serve --port 9000');
  });
  test('open and close run start, ssh, rsync and stop in order', () => {
    const calls: string[] = [];
    const run: Runner = (argv) => {
      calls.push(argv[0] === 'bash' ? argv[2] : argv[0]);
      return { code: 0, out: '' };
    };
    const vm = new Vm(p, run);
    vm.boot();
    vm.pushState();
    vm.startEngine();
    vm.pullState();
    vm.stop();
    expect(calls).toEqual(['evroc vm start chat', 'ssh', 'ssh', 'rsync', 'ssh', 'rsync', 'evroc vm stop chat']);
  });
});

describe('architect and planner', () => {
  const d: Design = {
    summary: 'A URL shortener.',
    components: [{ name: 'lib', language: 'C', image: 'c', path: 'lib/', responsibility: 'encoding', interfaces: [{ path: 'lib/short.h', description: 'encode/decode' }] }],
    data_flow: 'cli → api → lib',
    decisions: [{ decision: 'C for the core', why: 'speed' }],
    questions: [],
  };
  const canned = (value: unknown): Inference => ({
    complete: async () => ({ value: value as any, raw: '', promptTokens: 1, completionTokens: 1, ms: 1 }),
    text: async () => ({ value: '', raw: '', promptTokens: 1, completionTokens: 1, ms: 1 }),
  });
  test('design renders ARCHITECTURE.md', async () => {
    const got = await design(canned(d), 'shortener');
    const md = architectureMarkdown(got);
    expect(md).toContain('| lib | C | `lib/` | encoding |');
    expect(md).toContain('**C for the core.** speed');
  });
  test('tickets are validated', async () => {
    const good = { interfaces: ['lib/short.h'], tickets: [{ id: 'enc', title: 'Encode', image: 'c', owns: ['lib/enc.c'], acceptance: ['make test'] }] };
    expect((await tickets(canned(good), d, ['lib/short.h'])).tickets[0].budget.attempts).toBe(4);
    const bad = { interfaces: [], tickets: [{ id: 'a', owns: ['x'] }, { id: 'b', owns: ['x'] }] };
    expect(tickets(canned(bad), d, [])).rejects.toThrow(/both own/);
  });
});

test('ship to a folder commits as the user with no trailers', () => {
  const repo = join(dir, 'ship-src');
  mkdirSync(repo);
  writeFileSync(join(repo, 'a.c'), 'int main(){return 0;}\n');
  initRepo(repo);
  gitOk(repo, ['branch', 'mugge/integration']);
  const dest = join(dir, 'ship-dest');
  mkdirSync(dest);
  gitOk(dest, ['init', '-q']);
  const c = shipToFolder({ repo, author: { name: 'Oliver', email: 'o@example.com' }, message: 'Add the server', dir: dest });
  expect(c).toBeTruthy();
  expect(git(dest, ['log', '-1', '--format=%an <%ae>|%B']).out).toBe('Oliver <o@example.com>|Add the server');
  expect(readFileSync(join(dest, 'a.c'), 'utf8')).toContain('main');
});

test('engine API runs a plan and streams events', async () => {
  const repo = join(dir, 'api-repo');
  mkdirSync(repo);
  writeFileSync(join(repo, 'README.md'), 'x\n');
  initRepo(repo);
  const fake = startFakeModel({ solutions: { a: { files: { 'a.txt': 'ok\n' }, note: 'a' } } });
  const server = startEngineServer({ port: 0, repo, concurrency: 2, sandbox: new LocalSandbox(), inference: new OpenAICompatible({ baseUrl: fake.url, defaultModel: 'fake' }), worktreeRoot: join(dir, 'api-wt') });
  try {
    expect(await (await fetch(`${server.url}/health`)).text()).toBe('ok');
    const bad = await fetch(`${server.url}/run`, { method: 'POST', body: JSON.stringify({ plan: { tickets: [{ id: 'A!' }] } }) });
    expect(bad.status).toBe(400);
    const plan = { tickets: [{ id: 'a', title: 'A', owns: ['a.txt'], acceptance: ['grep -q ok a.txt'] }] };
    const res = await fetch(`${server.url}/run`, { method: 'POST', body: JSON.stringify({ plan }) });
    expect((await res.json()).started).toBe(true);
    const seen: EngineEvent[] = [];
    const ac = new AbortController();
    await new Promise<void>((resolve, reject) => {
      followEvents(server.url, (e) => {
        seen.push(e);
        if (e.type === 'end') resolve();
      }, ac.signal).catch((e) => (ac.signal.aborted ? resolve() : reject(e)));
    });
    ac.abort();
    expect(seen.find((e) => e.type === 'finish')).toMatchObject({ ticket: 'a', state: 'done' });
    expect(server.view().summary?.integrated).toBe(true);
  } finally {
    server.stop();
    fake.stop();
  }
});
