#!/usr/bin/env bun
/**
 * mugge: the terminal client. Thin on the laptop; the engine runs on the project's VM and is
 * reached through an SSH tunnel.
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { startEngineServer, followEvents } from './api/server.ts';
import { checkedPoolFromEnv, concurrencyFromEnv, inferenceFromEnv, machinesFromEnv } from './config.ts';
import { describeMachine, describeProbe, llamaServerArgs, PROBE_TIMEOUT_MS, probeMachines, usableMachines } from './engine/backends.ts';
import { applyEvent, emptyView, type EngineEvent, type RunView } from './engine/events.ts';
import { headCommit, initRepo } from './engine/git.ts';
import { OpenAICompatible } from './engine/inference.ts';
import { sandboxFromEnv } from './engine/sandbox.ts';
import { runPlan } from './engine/scheduler.ts';
import { crewFromEnv, formatCallSummary, runId, statusLine, Telemetry, type CrewStatus } from './engine/telemetry.ts';
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
  mugge run <plan.json> --repo <dir>   run a plan here (MUGGE_SANDBOX=local for no containers);
                                 per-call log and crew status in --out (default <repo>/../.mugge-runs/<run>)
  mugge status <run dir> [--watch]     the crew status a run is writing (status.json)
  mugge smoke [--fake] [--url U --model M]   the 15 spike tickets through the loop: MEETS / MISSES
  mugge ship --repo <dir> (--to <folder> | --github <remote> --branch <b>) -m "message"
  mugge close <project>          save state, stop the VM, billing stops
  mugge list                     projects and their state
  mugge spike [--fake]           the engine spike on the polyglot test project
  mugge bench run|check|report … MuggeBench (see bench/run.ts, check.ts, report.ts)
  mugge engine serve --port N --project <dir>    (on the VM) the engine API
  mugge backends                 the crew's machines, their slots and matching llama-server flags
  mugge dog                      the dog

Model endpoint: MUGGE_INFERENCE_URL, MUGGE_MODEL, MUGGE_MODELS, or MUGGE_BACKENDS for several machines (see src/config.ts).`;

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

function thinkingOf(extra: Record<string, unknown> | undefined): 'on' | 'off' | 'server' {
  const v = (extra?.chat_template_kwargs as { enable_thinking?: boolean } | undefined)?.enable_thinking;
  return v === undefined ? 'server' : v ? 'on' : 'off';
}

/** A crew status as a few plain lines: the status bar, teams, working agents, warnings. */
function describeStatus(s: CrewStatus): string {
  const lines = [statusLine(s)];
  for (const t of s.teams) lines.push(`team ${t.id}${t.domain ? ` (${t.domain})` : ''} on ${t.machine}: ${t.working} working, ${Math.round(t.tokS)} tok/s${t.tickets.length ? `, ${t.tickets.join(' ')}` : ''}`);
  for (const m of s.machines) lines.push(`machine ${m.id}: slots ${m.slotsUsed}/${m.slots}, ${Math.round(m.tokS)} tok/s`);
  for (const a of s.agents.filter((x) => x.ticket && !['idle', 'finished', 'failed'].includes(x.state)))
    lines.push(`  ${a.id.padEnd(4)} ${a.state.padEnd(8)} ${a.ticket} ${a.attempt}/${a.maxAttempts} ${Math.round(a.tokS)} tok/s ${a.owns.join(' ')}`);
  const c = s.calls;
  lines.push(`calls ${c.calls}: ${c.dropped} dropped, ${c.cut} cut, ${c.forced} forced, ${c.errors} errors, ${c.clean} clean`);
  lines.push(...s.warnings);
  return lines.join('\n');
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
      // the same start check as run: an optional manager machine that is down is left out
      const crew = concurrencyFromEnv();
      const pool = await checkedPoolFromEnv(process.env, crew, undefined, ['manager']).catch((e) => die(String((e as Error).message ?? e)));
      const inf = pool.forAgent({ role: 'manager' });
      // a big crew's scaffold and tickets are planned per component, in parallel on the worker slots
      const serves = (role: 'worker') => { try { pool.forAgent({ role }); return true; } catch { return false; } };
      const o = { crew, parts: serves('worker') ? (i: number) => pool.forAgent({ role: 'worker', agent: `planner-${i + 1}`, team: 'orchestrator' }) : () => inf };
      const d = await design(inf, p.request, [], 'architect', o);
      if (d.questions.length) {
        console.log('The architect asks:\n' + d.questions.map((q) => `  - ${q}`).join('\n'));
      }
      const dir = stateDir(p.name);
      writeFileSync(join(dir, 'ARCHITECTURE.md'), architectureMarkdown(d));
      const sc = await scaffold(inf, d, 'planner', o);
      for (const w of sc.warnings) console.warn(`scaffold: ${w}`);
      const scDir = join(dir, 'scaffold');
      for (const [path, content] of Object.entries(sc.files)) {
        mkdirSync(join(scDir, path, '..'), { recursive: true });
        writeFileSync(join(scDir, path), content);
      }
      const { warnings, ...plan } = await tickets(inf, d, Object.keys(sc.files), 'planner', o);
      for (const w of warnings ?? []) console.warn(`plan: ${w}`);
      writeFileSync(join(dir, 'plan.json'), JSON.stringify(plan, null, 2) + '\n');
      console.log(`${plan.tickets.length} tickets on a ${Object.keys(sc.files).length}-file scaffold, saved in ${dir}`);
      return;
    }
    case 'run': {
      const { values, positionals } = parseArgs({ args: rest, allowPositionals: true, options: { repo: { type: 'string' }, concurrency: { type: 'string' }, plain: { type: 'boolean' }, out: { type: 'string' } } });
      const plan = validatePlan(await Bun.file(positionals[0] ?? die('which plan.json?')).json());
      const repo = resolve(values.repo ?? die('--repo <dir> is required'));
      if (!existsSync(join(repo, '.git'))) initRepo(repo);
      const feed = viewFeed();
      const cfg = inferenceFromEnv();
      const concurrency = values.concurrency ? Number(values.concurrency) : concurrencyFromEnv();
      const pool = await checkedPoolFromEnv(process.env, concurrency).catch((e) => die(String((e as Error).message ?? e)));
      const id = runId();
      const telemetry = new Telemetry({
        run: id, dir: resolve(values.out ?? join(repo, '..', '.mugge-runs', id)), crew: crewFromEnv(process.env, concurrency),
        settings: { thinking: thinkingOf(cfg.extra), maxCallTokens: cfg.maxCallTokens ?? null, model: cfg.defaultModel },
      });
      telemetry.setPlan(plan);
      pool.subscribe((s) => telemetry.serving(s));
      const show = values.plain || !process.stdout.isTTY ? plainEvent : feed.emit;
      const run = runPlan({
        repo, base: headCommit(repo), plan, pool,
        sandbox: sandboxFromEnv(),
        concurrency,
        worktreeRoot: join(repo, '..', '.mugge-worktrees'),
        emit: (e) => {
          telemetry.emit(e);
          show(e);
        },
      });
      if (!values.plain && process.stdout.isTTY) void follow(repo.split('/').pop()!, feed.subscribe);
      const report = await run;
      const s = report.summary;
      const status = telemetry.close();
      console.log(`\n${s.done}/${s.tickets} done, ${s.failed} failed, ${s.skipped} skipped${s.blocked ? `, ${s.blocked} blocked (two tickets need the same file)` : ''}; integration ${s.integrated ? 'green on mugge/integration' : 'not green'}`);
      console.log(`\n${formatCallSummary(status.calls)}\n(per call: ${telemetry.callsPath}; crew status: ${telemetry.statusPath})`);
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
    case 'status': {
      const { values, positionals } = parseArgs({ args: rest, allowPositionals: true, options: { watch: { type: 'boolean' }, json: { type: 'boolean' } } });
      const where = resolve(positionals[0] ?? die('which run? mugge status <run dir or status.json>'));
      const file = where.endsWith('.json') ? where : join(where, 'status.json');
      do {
        if (!existsSync(file)) die(`no status at ${file}`);
        const st: CrewStatus = await Bun.file(file).json();
        if (values.watch && process.stdout.isTTY) process.stdout.write('\x1b[2J\x1b[H');
        console.log(values.json ? JSON.stringify(st, null, 2) : describeStatus(st));
        if (values.watch) await Bun.sleep(2000);
      } while (values.watch);
      return;
    }
    case 'smoke': {
      const p = Bun.spawnSync(['bun', join(import.meta.dir, '..', 'bench', 'smoke.ts'), ...rest], { stdout: 'inherit', stderr: 'inherit', stdin: 'inherit' });
      process.exit(p.exitCode ?? 2);
    }
    case 'spike': {
      const { runSpike, formatSummary } = await import('../spike/run-spike.ts');
      const fake = rest.includes('--fake');
      const r = await runSpike({ fake, latencyMs: fake ? 300 : undefined });
      console.log('\n' + formatSummary(r.summary));
      process.exit(r.summary.done === r.summary.tickets && r.integrated ? 0 : 1);
    }
    case 'bench': {
      const sub = rest[0];
      if (!['run', 'check', 'report'].includes(sub)) die('mugge bench run|check|report …');
      const p = Bun.spawnSync(['bun', join(import.meta.dir, '..', 'bench', `${sub}.ts`), ...rest.slice(1)], { stdout: 'inherit', stderr: 'inherit', stdin: 'inherit' });
      process.exit(p.exitCode ?? 1);
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
        // a crew of machines (laptops on wifi, cloud GPUs) when MUGGE_BACKENDS lists them
        pool: process.env.MUGGE_BACKENDS?.trim() ? await checkedPoolFromEnv().catch((e) => die(String((e as Error).message ?? e))) : undefined,
        sandbox: sandboxFromEnv(),
        concurrency: values.concurrency ? Number(values.concurrency) : concurrencyFromEnv(),
        worktreeRoot: join(dir, 'worktrees'),
        runsDir: join(dir, 'runs'),
      });
      console.log(`engine on ${s.url}, repo ${repo}`);
      return;
    }
    case 'backends': {
      const machines = machinesFromEnv();
      const probes = await probeMachines(machines, process.env, PROBE_TIMEOUT_MS, inferenceFromEnv().apiKey);
      for (const [i, m] of machines.entries()) {
        console.log(describeMachine(m));
        console.log(`  ${describeProbe(probes[i]!)}`);
        if (m.server === 'llama-server') console.log(`  llama-server ${llamaServerArgs(m).join(' ')}`);
        for (const w of m.warnings) console.log(`  warning: ${w}`);
      }
      // the verdict mugge run would reach, and a non-zero exit when it would not start
      try {
        usableMachines(machines, probes);
      } catch (e) {
        die(String((e as Error).message ?? e));
      }
      return;
    }
    default:
      die(`unknown command ${cmd}\n\n${HELP}`);
  }
}

await main(process.argv.slice(2));
