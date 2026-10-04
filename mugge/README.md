# mugge

Parallel coding with small specialist models. A planner splits the work into tickets, many
single-turn coder models fill them in at once on one GPU VM, each in its own git worktree and
sandbox, and the engine (not the model) decides when a ticket is done by running its checks.
You drive it from the terminal, with the dog.

The models come from llmap (the rest of this repo). This folder is the engine and the client.
Bun and TypeScript, no runtime dependencies; the dog, palette and matrix rain come from
[salu](https://github.com/OliverVillson/salu).

```bash
bun install
bun test                                  # unit tests + the whole spike against a fake model
bun run spike/run-spike.ts --fake --serial  # the engine spike with metrics
bun src/cli.ts                            # the TUI (projects, the dog)
```

## How a run works

```
plan.json ─► scheduler ─► ticket ready? (deps done, a free slot, nobody running owns its files)
                │
                ├─► git worktree on mugge/<ticket>, from the scaffold or a merge of its deps
                ├─► harness: context pack → model writes owned files → acceptance in the sandbox
                │            → failed? fresh fix call with the trimmed error → … → budget → escalate
                └─► commit the owned files (as the user, no trailers)
all ended ─► integrate: merge done branches onto mugge/integration in dependency order,
             re-run every check; a failure becomes one fix ticket
```

| Piece | File |
|---|---|
| Ticket and plan schema, validation, JSON Schema for constrained decoding | `src/tickets/schema.ts` |
| Write → check → fix loop (the `WorkerRunner` the plan asks for) | `src/engine/harness.ts`, `src/engine/prompts.ts` |
| Scheduler: dependencies, `owns` guard, concurrency, metrics | `src/engine/scheduler.ts` |
| Worktree per ticket, branches, merges | `src/engine/git.ts` |
| Integrate step and the fix ticket | `src/engine/integrate.ts` |
| OpenAI-compatible client (vLLM, SGLang, hosted), LoRA name routing | `src/engine/inference.ts`, `src/config.ts` |
| Sandboxes: podman/docker (+ gVisor) or local | `src/engine/sandbox.ts`, `sandbox/images/{c,node,python,web}` |
| Architect, scaffold and ticket calls | `src/plan/architect.ts` |
| Projects, VM open/close over SSH, private state store | `src/project/` |
| Shipping with no stamp (one commit authored by you) | `src/project/ship.ts` |
| Engine API on the VM (status, SSE events, run, stop) | `src/api/server.ts` |
| TUI: dog, matrix splash, run view, project list | `src/ui/` |
| VM setup and vLLM with every adapter | `vm/setup.sh`, `vm/serve-vllm.sh` |

## The spike

`spike/project` is a small polyglot URL shortener (C library, Python API over ctypes,
TypeScript CLI) with interfaces written and every implementation file stubbed. `spike/plan.json`
holds 15 tickets, 9 without dependencies. `spike/solutions/` are canned answers for the fake
model; `spike/check.sh` proves each ticket fails on the scaffold, passes with its solution, and
needs nothing beyond its dependencies.

On a real VM, against vLLM:

```bash
vm/setup.sh                                          # once, then snapshot the disk
MODEL=Qwen/Qwen3.6-35B-A3B-FP8 vm/serve-vllm.sh &
bun run spike/run-spike.ts --url http://127.0.0.1:8000/v1 --model mugge-small \
  --concurrency 16 --sandbox podman --serial --out metrics.json
```

The pass bar from the spec: 12+ agents at once without preemption, 70%+ of tickets within 4
attempts, a clean integration, 3x faster than one at a time.

## Commands

```
mugge                          projects, with the dog
mugge new "<what to build>"    make a project
mugge vm <project> --host H --user U [--driver command --start "evroc ..." --stop "evroc ..."]
mugge open <project>           boot the VM, push state, start the engine, follow it live
mugge plan <project>           architect → ARCHITECTURE.md, scaffold, tickets
mugge run <plan.json> --repo <dir>
mugge ship --repo <dir> (--to <folder> | --github <remote> --branch <b>) -m "message"
mugge close <project>          pull state, stop the VM
mugge dog
```

## Not done yet

- `mugge open` starts the engine and follows it, but nothing yet sends the planned scaffold and
  tickets from `mugge plan` to the VM's engine; today that is `mugge run` on the VM.
- The scheduler counts slots, not KV cache; vLLM metrics are not read yet.
- `web` image acceptance (Playwright, screenshots in the TUI), architecture and plan-graph screens,
  the VM cost meter, idle auto-close and VM snapshots.
- The evroc driver is a pair of shell commands you supply; there is no evroc API client.
