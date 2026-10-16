# MuggeBench

Scores a model on what Mugge actually asks of it: Mugge tickets, run through the engine's own
write → check → fix loop (`runTicket`), graded on hidden tests. Design and reasoning:
`plan/mugge-bench.md` in the project files.

The tasks live in the private `llmap-bench` repo, never here: this repo is public, and public
tasks end up in training data. Only the runner, the checker and two examples live here
(`spike/`, the engine's own example project, and `test/fixtures/bench-mini/`).

## A bench project

```
<project>/
  plan.json            Mugge's plan. Top level: "bench": {"canary": "MUGGE-BENCH-…", "version", "split": "dev"|"test"}
                       Each ticket: "bench": {"theme", "language", "role", "hidden": ["<command>", …]}
  project/             the scaffold: interfaces written, owned files stubbed
  solutions/<id>.json  reference answer: {"files": {path: content}, "canary": "MUGGE-BENCH-…"}
  hidden/<id>/<path>   hidden test files, copied over the worktree after the loop
```

Themes: `backend`, `frontend`, `ml`, `systems`, `dotnet`. Roles: `implement`, `fix`, `test`,
`refactor`, `integrate`. A hidden command can be anything that exits 0 on success, so a
test-role ticket's hidden command runs the model's tests against mutants, and a refactor's
checks the structure.

## Commands

```bash
bun bench/check.ts <bench dir> --sandbox podman          # admit tickets (see check.ts for the five checks)
bun bench/run.ts --bench <bench dir> --out runs/build \
  --url http://127.0.0.1:8000/v1 --model build \
  --seeds 3 --concurrency 16 --thinking on --temperature 0.6 --top-p 0.95 --top-k 20
bun bench/report.ts runs/build --ref runs/full            # paired verdict against the full model
bun bench/run.ts --bench test/fixtures/bench-mini --out /tmp/mini --fake --seeds 1   # no model needed
```

`mugge bench run|check|report …` runs the same three. In llmap, set `mugge_bench_dir` in a
job's config and the eval stage runs the bench on every served candidate (`stages/eval.py`).

## What a run reports

- **Mugge score**: % of tickets solved (acceptance and hidden tests pass within the ticket's
  budget), each ticket once, averaged over seeds, with a bootstrap 95% interval.
- Against a reference run: the build's share of the reference score over shared tickets, its
  interval, and **met / missed / too close to call** against the bar (default 90%).
- Solved runs per GPU-hour at the run's concurrency, completion tokens and attempts per solve,
  first-call solves, answers with no owned file, writes outside `owns`, and the share of
  acceptance passes that failed hidden tests.
- All of it by theme, language, role and project.

## The spike smoke

One command, about ten minutes against a local llama-server: the 15 public spike tickets through
the engine's own loop. BlackHole runs it before exporting anything.

```bash
bun bench/smoke.ts --url http://127.0.0.1:8080/v1 --model BH0.1-1.3 --sandbox local   # or: mugge smoke …
bun bench/smoke.ts --fake --fake-mess        # no model: the fake server, with four messy first answers (CI)
```

1. The reference answers must pass acceptance and hidden tests in the sandbox, 15/15; otherwise
   it stops with NO SCORE (exit 2), because the sandbox is broken, not the model.
2. Every ticket runs once through `runTicket` (write → check → fix, 4 attempts), thinking off by
   default, `top_k` 20 (0 is refused), 4096 tokens per call (`--max-tokens`, or
   `MUGGE_MAX_CALL_TOKENS`). `--min-p 0` for MiniCPM5-2B.
3. It prints tickets passing, parser drops (by reason), forced and cut answers, call errors, wall
   time and **MEETS** (exit 0) when at least `--bar` (default 0.5) of the tickets pass, at most
   20% of answers were lost to the parser, the cap or a forced answer, at most 20% wrote a file path the parser
   had to guess (`prog.c`, `**a.py**`: not the training format), and no call failed; **MISSES**
   (exit 1) otherwise, with the reasons.
4. A `per-call cap` line says how many answers ran out of tokens and how long the finished ones
   were (median, 90th percentile, longest), with a suggested `--max-tokens` when more than 5%
   ran out or when every answer used under half the cap. With `--thinking on`, a ticket whose
   thinking call ran out of tokens sends its later calls with thinking off (`MUGGE_CUT_RETRY`,
   default `nothink`; `same` keeps thinking on); `calls.jsonl` marks each call's `thinking`.

Writes `calls.jsonl`, `status.json`, `smoke.json` and the bench's `results.jsonl` / `report.json`
in `--out` (default `runs/smoke-<time>`).

## Per-call log and crew status

Every run (`mugge run`, `bench/run.ts`, the smoke, the engine API) writes, through
`src/engine/telemetry.ts`:

- `calls.jsonl`: one line per model call with ticket, attempt, write/fix, agent/team/machine,
  prompt, completion and reasoning tokens, max_tokens, ms, tok/s, `cut` (stopped at max_tokens),
  `forced` (the bench proxy closed the thinking and forced the answer: why, or null), `drop` (the
  parser got no usable file: `cut`, `empty`, `no_fence`, `unlabelled`, `not_owned`), `loose`
  (owned files whose path the parser had to guess, not the training format), `error`, and
  `clean`: false for every call that must never become a training row. A warning goes to stderr
  when more than 20% of answers are dropped or cut. `bench/run.ts --transcripts` carries the same
  `cut`, `forced`, `drop` and `loose` on every transcript call.
- `status.json` (schema `centipede.status/1`, also `GET /telemetry` on the engine API): crew state,
  every agent (a1–d10: state, ticket, owned files, attempt n/4, own tok/s), tickets, total tok/s
  over one minute with a 60-point sparkline and its split by team and machine, slots used and
  queued per machine. The types are `CrewStatus` in telemetry.ts.

`MUGGE_DEBUG_CALLS=1` still prints the one human line per call.
