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

On a real machine, against vLLM serving Qwen3.8-27B (the demo model, `Qwen/Qwen3.8-27B-FP8`; the
server scripts and the crew file live in `deploy/`, from the serving branch, claude/qwen-serving):

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b --reasoning-parser qwen3 &   # or deploy/'s script
bun run spike/run-spike.ts --url http://127.0.0.1:8000/v1 --model qwen3.8-27b \
  --concurrency 10 --sandbox podman --serial --out metrics.json
```

The pass bar from the spec: 12+ agents at once without preemption, 70%+ of tickets within 4
attempts, a clean integration, 3x faster than one at a time.

## Running on Qwen3.8-27B

The engine's defaults are Qwen3.8's (2026-10-18): thinking on, sent on every call as
`chat_template_kwargs.enable_thinking`; the model card's sampling per mode (thinking: temperature
1.0, top_p 0.95, top_k 20, min_p 0; thinking off: 0.7, 0.8, 20, 0; presence_penalty 0 in both,
because workers rewrite whole files); the lead review sent with thinking off and answered as
`{"verdict": "approve" | "send_back", "reason": "..."}`. The server must run vLLM's
`--reasoning-parser qwen3` so the review's JSON constraint applies after the thinking; the engine
also copes with thinking left inside the answer (a bare `</think>`), but never relies on it.

From a fresh clone on the GPU box, with the server up on port 8000:

```bash
cd llmap/mugge && bun install
export MUGGE_MAX_CALL_TOKENS=16384            # per call: thinking plus the files
# 1. one endpoint: the 15 spike tickets through the write → check → fix loop, MEETS or MISSES
bun bench/smoke.ts --url http://127.0.0.1:8000/v1 --model qwen3.8-27b --sandbox local   # needs gcc, make, python3, node
# 2. the crew: every machine in MUGGE_BACKENDS answers, then the spike plan with 10 agents
export MUGGE_BACKENDS=@deploy/crew-b200.json MUGGE_CONCURRENCY=10
bun src/cli.ts backends
cp -r spike/project /tmp/spike-repo && bun src/cli.ts run spike/plan.json --repo /tmp/spike-repo --plain
```

The smoke writes `runs/smoke-<time>/calls.jsonl` (one line per call: tokens, cut, thinking),
`status.json` and `smoke.json`; `bun src/cli.ts status <run dir> --watch` follows a crew run.
Add `MUGGE_REVIEW=on` to the smoke to score with the lead review, `MUGGE_DEBUG_CALLS=1` for one
line per call on stderr, `--thinking off` for the fastest run.

Speed levers, first to try first:

- `MUGGE_REASONING_EFFORT=worker=medium,lead=xhigh,manager=xhigh` sets how hard each kind of call
  thinks (`off` for no thinking). Use one level for everything (`medium`), or give a machine in
  `MUGGE_BACKENDS` its own `"reasoning_effort"`. A lead with a level reviews with thinking on, and
  a thinking review's cap is `MUGGE_MAX_CALL_TOKENS`, else 16,384.
- Run more agents than worker slots, e.g. `MUGGE_CONCURRENCY=80` on 64 slots. While some agents
  run checks, wait for a review or set up a worktree, others keep the slots decoding.
- Give the lead about one slot for every eight worker slots. Reviews are short, but a ticket waits
  for its review while it holds its agent.

`bun bench/scale.ts --agents 80 --slots 64 --tickets 768 --latency 5000` measures the engine at
crew size against the fake model: how busy the slots stay, event-loop stalls, CPU per call and
log sizes. Without a GPU, `bun bench/smoke.ts --fake --fake-qwen` runs the same path
against the fake model answering the way Qwen does.

## Commands

```
mugge                          projects, with the dog
mugge new "<what to build>"    make a project
mugge vm <project> --host H --user U [--driver command --start "evroc ..." --stop "evroc ..."]
mugge open <project>           boot the VM, push state, start the engine, follow it live
mugge plan <project>           architect → ARCHITECTURE.md, scaffold, tickets
mugge run <plan.json> --repo <dir> [--out <run dir>]   per-call log and crew status in the run dir
mugge status <run dir> [--watch]   the crew status a run is writing
mugge smoke [--fake] [--url U --model M]   the 15 spike tickets through the loop: MEETS / MISSES
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
