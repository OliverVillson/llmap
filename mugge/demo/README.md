# The Centipede demo

One prompt in, a crew of code agents building it at once on one Qwen3.8-27B-FP8 server, and a
live view in the browser while they work. The orchestrator designs the project, writes its
scaffold and splits it into tickets, about one per code agent, so most of the crew starts at
once. Each ticket's agent writes its files until its checks pass, a lead reviews every finished
ticket, and the run ends by merging every branch and running the whole project's tests.

## First run: 64 agents from one prompt (B200)

From the mugge folder:

```bash
git fetch origin && git checkout claude/centipede-demo   # until it is merged; it includes deploy/
bun install
CREW_WORKERS=64 deploy/qwen.sh start          # once; the server keeps the crew size
deploy/qwen.sh wait && deploy/qwen.sh check   # check must end in PASS
set -a; . deploy/crew.env; set +a             # reads 64 back: 64 workers, 4 leads, 1 orchestrator
bun demo/run.ts --prompt "A Python 3 command-line toolbox, tk, with 64 small subcommands (text, numbers, dates, encodings, validation), each in its own module with its own unittest file, standard library only."
```

`deploy/README.md` covers the server itself. The demo needs `python3` (3.8 or newer), for
`crew.env` and for Python projects' tests. It prints the view's address and the run folder, and
keeps the view up after the run ends until Ctrl-C. To keep it running after you log out, start it
with `nohup bun demo/run.ts --prompt "…" > demo.log 2>&1 &` (stop it with `kill`), or inside tmux.

What to expect at 64:

- **Planning comes first and takes a while.** The orchestrator makes three calls on its own
  slot: design, scaffold and tickets. With 64 modules the scaffold is the long one (a stub and a
  test per module), and the view shows how long the call in flight has run. A planning call is
  capped at `MUGGE_MAX_CALL_TOKENS` (16,384), thinking included. If the scaffold or the plan is
  cut there, the run stops with that error rather than starting a half plan; see "Not verified"
  in the PR.
- **Checks share the CPUs.** Each ticket's tests run on this machine, at most one per CPU at
  once (`--checks N` to change it). An agent whose check is waiting shows "checking".
- **Leads are fewer than agents.** 4 leads review 64 agents' tickets, so finished tickets can
  queue for a review ("queued for a slot").

Rehearse the same run with no GPU: `bun demo/run.ts --fake --concurrency 64 --prompt "anything"`.
The mock orchestrator plans one module per agent, 64 at once, plus a registry ticket that waits
on all of them, and the mock workers answer each ticket.

## The built-in task (10 agents)

`bun demo/run.ts` with no prompt runs **mdlite**, a Markdown to HTML converter in plain Python 3.
`project/` is its scaffold: 17 stub modules that raise `NotImplementedError` and 118 unit tests.
`plan.json` cuts it into 17 tickets, of which 12 can start at once, so it suits a crew of 10.
Two tickets (`block-table` and `block-code`) both add rules to `mdlite/theme.css`, and the
orchestrator settles that clash by running `block-code` after `block-table`. At the end the
crew's own tool renders `project/README.md`.

## From the Mac

```bash
ssh -N -L 7340:127.0.0.1:7340 b200
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
- `fake-prompt.ts`: the mock orchestrator for `--fake --prompt`, one module per agent.
- `project/`, `reference/`, `plan.json`: mdlite's scaffold, an answer for every ticket (used by
  `check.ts` and the mock model), and its prepared plan.
