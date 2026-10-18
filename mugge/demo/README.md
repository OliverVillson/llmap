# The Centipede demo

One prompt in, a crew of code agents building it at once on one Qwen3.8-27B-FP8 server, and a
live view in the browser while they work. The orchestrator designs the project, writes its
scaffold and splits it into tickets, about one per code agent, so most of the crew starts at
once. Each ticket's agent writes its files until its checks pass, a lead reviews every finished
ticket, and the run ends by merging every branch and running the whole project's tests.

## First run: 64 agents from one prompt (B200)

On the B200. A fresh VM (or one that kept `/data` but lost its root disk) gets everything from
one command, `deploy/fresh.sh`: apt packages, the data disk at `/data`, the checkout at
`/data/llmap` with `~/llmap` pointing at it (so `demo/runs/` lands on the big disk), uv and bun on
every shell's PATH, vLLM, the weights and a preflight (`deploy/README.md`, "On the B200, from a
fresh VM"). It is safe to run again; if `/data` was never formatted it stops and prints the
`--format /dev/vdc` line to run.

On the Mac first, only when the VM was re-created (a new VM has a new host key):

```bash
ssh-keygen -R 194.14.80.240
```

Then:

```bash
ssh evroc
curl -fsSLo /tmp/fresh.sh https://raw.githubusercontent.com/OliverVillson/llmap/claude/centipede-demo/mugge/deploy/fresh.sh
bash /tmp/fresh.sh                            # ends in "preflight: PASS" and "ready"
cd ~/llmap/mugge
CREW_WORKERS=64 deploy/qwen.sh start          # runs the preflight again, then starts; the crew size sticks
deploy/qwen.sh wait                           # warns about memory for 64 on a 180 GB B200: expected
deploy/qwen.sh check                          # must end in PASS
set -a; . deploy/crew.env; set +a             # reads 64 back (again in every new shell)
nohup bun demo/run.ts --prompt "A Python 3 command-line toolbox, tk, with many small subcommands (text, numbers, dates, encodings, validation), each in its own module with its own unittest file, standard library only." > demo.log 2>&1 &
tail -f demo.log
```

Then on the Mac: `ssh -o ExitOnForwardFailure=yes -N -L 7340:127.0.0.1:7340 evroc` and open
http://127.0.0.1:7340/. If it says the port is in use, an old tunnel holds it:
`pkill -f 'L 7340:'` and run it again.

Stop a run with `kill -INT <pid>` or plain `kill`: both write the summary and then end the demo,
which frees the view's port; a second one quits at once. A run that finishes on its own keeps
the view up; before the next run, `pkill -INT -f 'bun demo/run.ts'` (or pass `--port 7341`), else
the new run says the port is in use.

`crew.env` runs 80 code agents on the 64 worker slots (the extra 16 keep the slots busy while
other agents run their checks), so the orchestrator plans for 80: **about 80 tickets for 64
slots**. The demo needs `python3` (3.8 or newer), for `crew.env` and for Python projects' tests.
It prints the view's address and the run folder, and keeps the view up after the run ends until
Ctrl-C. To keep it running after you log out, start it with
`nohup bun demo/run.ts --prompt "…" > demo.log 2>&1 &` (stop it with `kill`), or inside tmux.

What to expect at 64:

- **Planning comes first.** The orchestrator designs the project on its own slot, then (above 16
  agents) splits the work by design component: one call writes the shared and interface files,
  then one call per component writes its stubs and tests and one call per component plans its
  tickets, the components at the same time on worker slots that are idle until the plan is
  ready. The view shows how long the planning calls have run. Planning calls are not capped by
  `MUGGE_MAX_CALL_TOKENS`; they can run to the server's 65,536-token limit. If the plan comes out
  well short of 80 tickets, or with few that can start at once, the log says so.
- **Memory.** `wait` warns that the worst case (every slot at its longest at once) is a little
  over the B200's pool; that is expected (`deploy/README.md`, "64 agents and memory"). Watch
  `deploy/qwen.sh status`: if preemptions keep climbing, restart with an fp8 KV cache:
  `deploy/qwen.sh stop && KV_DTYPE=fp8 deploy/qwen.sh start && deploy/qwen.sh wait`.
- **Checks share the CPUs.** Each ticket's tests run on this machine, at most one per CPU at
  once (`--checks N` to change it). An agent whose check is waiting shows "checking".
- **Leads are fewer than agents.** 8 leads review 80 agents' tickets, so finished tickets can
  queue for a review: "queued for a lead to review" in the view, "waiting for a lead" in the
  log's progress line.

Rehearse the same run with no GPU: load `crew.env` the same way (with no server started, give the
size yourself: `CREW_WORKERS=64; set -a; . deploy/crew.env; set +a`), then
`bun demo/run.ts --fake --prompt "anything"`. The mock
orchestrator answers the same planning calls with unit converters in 5 groups, about one
module per agent and a table per group that waits on its converters (80 tickets, 75 ready at
once), and the mock workers answer each ticket. It takes the crew's slot counts from
`crew.env`.

## The built-in task (10 agents)

`bun demo/run.ts` with no prompt runs **mdlite**, a Markdown to HTML converter in plain Python 3.
`project/` is its scaffold: 17 stub modules that raise `NotImplementedError` and 118 unit tests.
`plan.json` cuts it into 17 tickets, of which 12 can start at once, so it suits a crew of 10.
Two tickets (`block-table` and `block-code`) both add rules to `mdlite/theme.css`, and the
orchestrator settles that clash by running `block-code` after `block-table`. At the end the
crew's own tool renders `project/README.md`.

## From the Mac

```bash
ssh -N -L 7340:127.0.0.1:7340 evroc
```

then open http://127.0.0.1:7340/. The view only listens on 127.0.0.1; do not open its port to
the internet (`--host 0.0.0.0` exists but has no password).

## Other runs

| Command | What it does |
|---|---|
| `bun demo/run.ts --prompt "…"` | your task, planned into about one ticket per agent (the first run above) |
| `bun demo/run.ts` | the built-in task, mdlite |
| `bun demo/run.ts --plan orchestrator` | mdlite, with the orchestrator writing the 17-ticket plan itself (one JSON call); a plan that does not cover the scaffold is not used, the prepared plan runs instead, and the view says why |
| `bun demo/run.ts --prompt "…" --repo DIR` | your own task on a copy of your repo (DIR itself is not touched) |
| `bun demo/run.ts --fake [--prompt "…" --concurrency 64]` | a rehearsal against a mock model, no GPU (`--fake-speed 5` for a faster one) |
| `bun demo/check.ts` | proves the built-in task is fair: each ticket fails on the scaffold and passes with its reference answer |

Options: `--port N` (7340), `--out DIR`, `--concurrency N` (`MUGGE_CONCURRENCY`, which
`crew.env` sets to the crew size; else 10), `--checks N` (checks at once, default the CPU count),
`--exit` (stop when the run ends; the exit code says whether it succeeded).

## The view

- **Orchestrator**: what it is doing now (planning, dispatching, integrating), the plan's source,
  and what the lead is reviewing.
- **Code agents**: one dot per agent with its state (writing, thinking, checking, fixing, in
  review, done, failed). Press a dot, or a ticket, for its panel:
  - **Now**: the ticket, its attempt, what it owns, the last check output, the lead's verdict.
  - **Prompt history**: every call for that ticket, with the exact messages sent, the thinking
    and the answer.
  - **Steer** is display-only: a note is saved to `steer.jsonl` but does not reach the agent.
    The engine has no way yet to put a note into the next prompt or stop a running attempt.
- **Tickets**: Waiting, Working, In review, Done, Failed, and the clash the orchestrator settled.
- **Throughput**: tokens per second across the crew, peak, and slots per machine.

Calls do not stream, so an agent's panel shows how long its current call has run, not the text
as it is written; the answer appears when the call returns.

## The run folder

`demo/runs/<run id>/` holds:

- `repo/`: the project, with branches `mugge/<ticket>` and `mugge/integration`.
- `result/`: the integrated project, with its whole test run (and, for mdlite, `README.html`).
- `prompts.jsonl`: every model call, with its messages and answer.
- `events.jsonl`: every engine event.
- `calls.jsonl` and `status.json`: the engine's per-call log and crew status.
- `steer.jsonl` and `summary.json`.

## Files

- `run.ts`: the one command. It drives the engine as it is (`runPlan`) and serves the view.
- `orchestrator.ts`: the planning call and the check an orchestrator's plan must pass.
- `view/index.html`: the live view.
- `fake-prompt.ts`: the mock orchestrator for `--fake --prompt`, about one module per agent.
- `project/`, `reference/`, `plan.json`: mdlite's scaffold, an answer for every ticket (used by
  `check.ts` and the mock model), and its prepared plan.
