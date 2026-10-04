#!/usr/bin/env bun
/**
 * mugge: the terminal client. Thin on the laptop; the engine runs on the project's VM and is
 * reached through an SSH tunnel.
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { startEngineServer, followEvents } from './api/server.ts';
import { concurrencyFromEnv, inferenceFromEnv } from './config.ts';
import { applyEvent, emptyView, type EngineEvent, type RunView } from './engine/events.ts';
import { headCommit, initRepo } from './engine/git.ts';
import { OpenAICompatible } from './engine/inference.ts';
import { sandboxFromEnv } from './engine/sandbox.ts';
import { runPlan } from './engine/scheduler.ts';
import { architectureMarkdown, design, scaffold, tickets } from './plan/architect.ts';
import { createProject, listProjects, loadProject, saveProject, slugify, stateDir, type VmConfig } from './project/project.ts';
import { shipToFolder, shipToRemote } from './project/ship.ts';
import { Vm } from './project/vm.ts';
import { validatePlan } from './tickets/schema.ts';

const VERSION = '0.1.0';

const HELP = `mugge ${VERSION}: parallel coding with small specialist models

  mugge                          projects, with the dog
  mugge new "<what to build>"    make a project (--name, --repo <git url or folder>)
  mugge vm <project> --host H --user U [--driver ssh|command --start CMD --stop CMD --port N --identity KEY]
  mugge open <project>           boot the VM, restore state, follow the engine live
  mugge plan <project>           architect + scaffold + tickets (needs a model endpoint)
  mugge run <plan.json> --repo <dir>   run a plan here (MUGGE_SANDBOX=local for no containers)
  mugge ship --repo <dir> (--to <folder> | --github <remote> --branch <b>) -m "message"
  mugge close <project>          save state, stop the VM, billing stops
  mugge list                     projects and their state
  mugge spike [--fake]           the engine spike on the polyglot test project
  mugge engine serve --port N --project <dir>    (on the VM) the engine API
  mugge dog                      the dog

Model endpoint: MUGGE_INFERENCE_URL, MUGGE_MODEL, MUGGE_MODELS (see src/config.ts).`;

function die(msg: string): never {
  console.error(`mugge: ${msg}`);
  process.exit(1);
}

async function ui() {
  return import('./ui/tui.ts');
}

async function follow(project: string, subscribe: (fn: (v: RunView) => void) => () => void, vm?: () => { state: string; gpu?: string; costUsd?: number }) {
  const { openTui } = await ui();
  await openTui({ project, subscribe, vm });
}

function viewFeed() {
  const view = emptyView();
  const subs = new Set<(v: RunView) => void>();
  return {
    emit: (e: EngineEvent) => {
      applyEvent(view, e);
      for (const s of subs) s(view);
    },
    subscribe: (fn: (v: RunView) => void) => {
      subs.add(fn);
      fn(view);
      return () => subs.delete(fn);
    },
    view,
  };
}

function plainEvent(e: EngineEvent) {
  const v = applyEvent(emptyView(), e).log.at(-1);
  if (v) console.log(v);
}

async function main(argv: string[]) {
  const [cmd, ...rest] = argv;
  switch (cmd) {
    case undefined: {
      const projects = listProjects();
      const { openHome } = await ui();
      const picked = await openHome({ projects: projects.map((p) => ({ name: p.name, state: p.state, repo: p.repo ?? undefined, lastOpened: p.lastOpened ?? undefined })) });
      if (picked) return main(['open', picked]);
      return;
    }
    case 'help': case '--help': case '-h':
      console.log(HELP);
      return;
    case 'version': case '--version':
      console.log(VERSION);
      return;
    case 'dog': {
      const { dogDemo } = await ui();
      await dogDemo();
      return;
    }
    case 'list': {
      const ps = listProjects();
      if (!ps.length) return console.log('no projects yet: mugge new "what to build"');
      for (const p of ps) console.log(`${p.state === 'open' ? '●' : '○'} ${p.name.padEnd(24)} ${p.vm ? p.vm.host : '(no VM)'}  ${p.repo ?? ''}`);
      return;
    }
    case 'new': {
      const { values, positionals } = parseArgs({ args: rest, allowPositionals: true, options: { name: { type: 'string' }, repo: { type: 'string' } } });
      const request = positionals.join(' ').trim();
      if (!request) die('say what to build: mugge new "chat app with a C server and a TS web client"');
      const p = createProject({ name: values.name ?? slugify(request), request, repo: values.repo });
      console.log(`made ${p.name}. Next: mugge vm ${p.name} --host <vm> --user <user>, then mugge open ${p.name}`);
      return;
    }
    case 'vm': {
      const { values, positionals } = parseArgs({
        args: rest, allowPositionals: true,
        options: { host: { type: 'string' }, user: { type: 'string' }, driver: { type: 'string' }, start: { type: 'string' }, stop: { type: 'string' }, port: { type: 'string' }, identity: { type: 'string' }, 'engine-port': { type: 'string' }, dir: { type: 'string' } },
      });
      const p = loadProject(positionals[0] ?? die('which project?'));
      if (!values.host || !values.user) die('--host and --user are required');
      const vm: VmConfig = {
        driver: values.driver === 'command' ? 'command' : 'ssh',
        host: values.host, user: values.user,
        port: values.port ? Number(values.port) : undefined,
        identity: values.identity, start: values.start, stop: values.stop,
        enginePort: values['engine-port'] ? Number(values['engine-port']) : undefined,
        remoteDir: values.dir,
      };
      if (vm.driver === 'command' && (!vm.start || !vm.stop)) die('the command driver needs --start and --stop');
      saveProject({ ...p, vm });
      console.log(`${p.name} will use ${vm.user}@${vm.host}`);
      return;
    }
    case 'open': {
      const p = loadProject(rest[0] ?? die('which project?'));
      const vm = new Vm(p);
      console.log(`booting ${p.vm!.host} ...`);
      vm.boot();
      vm.pushState();
      vm.startEngine();
      const session = vm.tunnel();
      saveProject({ ...p, state: 'open', lastOpened: Date.now() });
      const feed = viewFeed();
      const abort = new AbortController();
      const connect = async () => {
        for (let i = 0; i < 20 && !abort.signal.aborted; i++) {
          try {
            await followEvents(session.url, feed.emit, abort.signal);
            return;
          } catch {
            await Bun.sleep(1000);
          }
        }
      };
      void connect();
      try {
        await follow(p.name, feed.subscribe, () => ({ state: 'open' }));
      } finally {
        abort.abort();
        session.close();
      }
      console.log(`${p.name} stays open (the VM keeps running): mugge close ${p.name} to stop billing`);
      return;
    }
    case 'close': {
      const p = loadProject(rest[0] ?? die('which project?'));
      if (p.vm) {
        const vm = new Vm(p);
        try {
          vm.pullState();
        } catch (e) {
          console.error(`state not pulled: ${(e as Error).message}`);
        }
        vm.stop();
      }
      saveProject({ ...p, state: 'closed' });
      console.log(`${p.name} closed${p.vm?.driver === 'command' ? ', VM stopped' : ''}`);
      return;
    }
    case 'plan': {
      const p = loadProject(rest[0] ?? die('which project?'));
      if (!p.request) die(`${p.name} has no request`);
      const inf = new OpenAICompatible(inferenceFromEnv());
      const d = await design(inf, p.request);
      if (d.questions.length) {
        console.log('The architect asks:\n' + d.questions.map((q) => `  - ${q}`).join('\n'));
      }
      const dir = stateDir(p.name);
      writeFileSync(join(dir, 'ARCHITECTURE.md'), architectureMarkdown(d));
      const sc = await scaffold(inf, d);
      const scDir = join(dir, 'scaffold');
      for (const [path, content] of Object.entries(sc.files)) {
        mkdirSync(join(scDir, path, '..'), { recursive: true });
        writeFileSync(join(scDir, path), content);
      }
      const plan = await tickets(inf, d, Object.keys(sc.files));
      writeFileSync(join(dir, 'plan.json'), JSON.stringify(plan, null, 2) + '\n');
      console.log(`${plan.tickets.length} tickets on a ${Object.keys(sc.files).length}-file scaffold, saved in ${dir}`);
      return;
    }
    case 'run': {
      const { values, positionals } = parseArgs({ args: rest, allowPositionals: true, options: { repo: { type: 'string' }, concurrency: { type: 'string' }, plain: { type: 'boolean' } } });
      const plan = validatePlan(await Bun.file(positionals[0] ?? die('which plan.json?')).json());
      const repo = resolve(values.repo ?? die('--repo <dir> is required'));
      if (!existsSync(join(repo, '.git'))) initRepo(repo);
      const feed = viewFeed();
      const run = runPlan({
        repo, base: headCommit(repo), plan,
        inference: new OpenAICompatible(inferenceFromEnv()),
        sandbox: sandboxFromEnv(),
        concurrency: values.concurrency ? Number(values.concurrency) : concurrencyFromEnv(),
        worktreeRoot: join(repo, '..', '.mugge-worktrees'),
        emit: values.plain || !process.stdout.isTTY ? plainEvent : feed.emit,
      });
      if (!values.plain && process.stdout.isTTY) void follow(repo.split('/').pop()!, feed.subscribe);
      const report = await run;
      const s = report.summary;
      console.log(`\n${s.done}/${s.tickets} done, ${s.failed} failed, ${s.skipped} skipped; integration ${s.integrated ? 'green on mugge/integration' : 'not green'}`);
      process.exit(s.done === s.tickets && s.integrated ? 0 : 1);
    }
    case 'ship': {
      const { values } = parseArgs({ args: rest, options: { repo: { type: 'string' }, to: { type: 'string' }, github: { type: 'string' }, branch: { type: 'string' }, message: { type: 'string', short: 'm' }, name: { type: 'string' }, email: { type: 'string' } } });
      const repo = resolve(values.repo ?? die('--repo is required'));
      const name = values.name ?? Bun.spawnSync(['git', 'config', 'user.name']).stdout.toString().trim();
      const email = values.email ?? Bun.spawnSync(['git', 'config', 'user.email']).stdout.toString().trim();
      if (!name || !email) die('set git user.name and user.email (or pass --name and --email): shipped commits are yours');
      const message = values.message ?? die('-m "message" is required');
      if (values.to) {
        const c = shipToFolder({ repo, author: { name, email }, message, dir: resolve(values.to) });
        console.log(c ? `committed ${c.slice(0, 10)} in ${values.to}` : `copied into ${values.to}`);
      } else if (values.github) {
        const c = shipToRemote({ repo, author: { name, email }, message, remote: values.github, branch: values.branch ?? 'mugge' });
        console.log(`pushed ${c.slice(0, 10)} to ${values.github} ${values.branch ?? 'mugge'}`);
      } else die('--to <folder> or --github <remote>');
      return;
    }
    case 'spike': {
      const { runSpike, formatSummary } = await import('../spike/run-spike.ts');
      const fake = rest.includes('--fake');
      const r = await runSpike({ fake, latencyMs: fake ? 300 : undefined });
      console.log('\n' + formatSummary(r.summary));
      process.exit(r.summary.done === r.summary.tickets && r.integrated ? 0 : 1);
    }
    case 'engine': {
      const { values, positionals } = parseArgs({ args: rest, allowPositionals: true, options: { port: { type: 'string' }, project: { type: 'string' }, concurrency: { type: 'string' } } });
      if (positionals[0] !== 'serve') die('mugge engine serve --port N --project <dir>');
      const dir = resolve(values.project ?? '.');
      const repo = join(dir, 'repo');
      mkdirSync(repo, { recursive: true });
      if (!existsSync(join(repo, '.git'))) initRepo(repo);
      const s = startEngineServer({
        port: Number(values.port ?? 7341), repo,
        inference: new OpenAICompatible(inferenceFromEnv()),
        sandbox: sandboxFromEnv(),
        concurrency: values.concurrency ? Number(values.concurrency) : concurrencyFromEnv(),
        worktreeRoot: join(dir, 'worktrees'),
      });
      console.log(`engine on ${s.url}, repo ${repo}`);
      return;
    }
    default:
      die(`unknown command ${cmd}\n\n${HELP}`);
  }
}

await main(process.argv.slice(2));
