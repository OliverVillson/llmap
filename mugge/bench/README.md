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
