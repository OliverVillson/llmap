# The Centipede demo

One orchestrator and ten code agents build a small real project at the same time on one
Qwen3.8-27B-FP8 server, and a live view in the browser shows the crew while they work: the
plan, ten agents in parallel, a file clash the orchestrator settles, the lead's review of every
finished ticket, and the integrated result with its tests green.

The built-in task is **mdlite**, a Markdown to HTML converter in plain Python 3 with a command
line. `project/` is its scaffold: 17 stub modules that raise `NotImplementedError`, each with its
contract in the docstring, and 118 unit tests. `plan.json` cuts it into 17 tickets, one per
module; 12 can start at once, so all ten agents have work from the first second. Two tickets
(`block-table` and `block-code`) both add rules to `mdlite/theme.css`; the orchestrator settles
that clash by running `block-code` after `block-table`. When every ticket is merged, the run
runs the whole test suite and renders `project/README.md` with the crew's own tool.

## On the B200

Once `deploy/qwen.sh check` ends in PASS (deploy/README.md), from the mugge folder:

```bash
git fetch origin && git checkout claude/centipede-demo   # until it is merged; it includes deploy/
bun install
set -a; . deploy/crew.env; set +a
bun demo/run.ts
```

It needs `python3` (3.8 or newer), for `crew.env` and for the project's tests. The crew has 10
code agents by default; set `CREW_WORKERS=24` (or 32, 64) before loading `crew.env`, and start
the server with the same value. The built-in task never has more than 12 tickets ready at once,
so agents beyond 12 stay idle there; a bigger crew pays off on a bigger `--prompt` task. The run prints the view's address and
the run folder, and keeps the view up after the run ends until Ctrl-C. To keep it running after
you log out, start it with `nohup bun demo/run.ts > demo.log 2>&1 &` (stop it with `kill`), or
inside tmux.

## From the Mac

```bash
ssh -N -L 7340:127.0.0.1:7340 b200
```

then open http://127.0.0.1:7340/. The view only listens on 127.0.0.1; do not open its port to
the internet (`--host 0.0.0.0` exists but has no password).

## Other runs

| Command | What it does |
|---|---|
| `bun demo/run.ts --plan orchestrator` | the orchestrator writes the 17-ticket plan itself (one JSON call); a plan that does not cover the scaffold is not used, the prepared plan runs instead, and the view says why |
| `bun demo/run.ts --prompt "a CLI that …"` | your own task: the orchestrator designs it, scaffolds it and plans the tickets, then the crew builds it |
| `bun demo/run.ts --prompt "…" --repo DIR` | your own task on a copy of your repo (DIR itself is not touched) |
| `bun demo/run.ts --fake` | a rehearsal against a mock model, no GPU (`--fake-speed 5` for a faster one) |
| `bun demo/check.ts` | proves the built-in task is fair: each ticket fails on the scaffold and passes with its reference answer |

Options: `--port N` (7340), `--out DIR`, `--concurrency N` (`MUGGE_CONCURRENCY`, else 10), `--exit`
(stop when the run ends; the exit code says whether it succeeded).

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
- `result/`: the integrated project, its test output and `README.html`.
- `prompts.jsonl`: every model call, with its messages and answer.
- `events.jsonl`: every engine event.
- `calls.jsonl` and `status.json`: the engine's per-call log and crew status.
- `steer.jsonl` and `summary.json`.

## Files

- `run.ts`: the one command. It drives the engine as it is (`runPlan`) and serves the view.
- `orchestrator.ts`: the planning call and the check an orchestrator's plan must pass.
- `view/index.html`: the live view.
- `project/`: the scaffold the crew starts from.
- `reference/`: an answer for every ticket, used by `check.ts` and by the mock model.
- `plan.json`: the prepared plan.
