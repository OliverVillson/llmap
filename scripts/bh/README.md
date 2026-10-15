# BlackHole train harness

`scripts/bh_run.py` trains BH<version>-<GB>, the small worker model for the Centipede crew. A large teacher works kit tickets through the engine's own loop. Its passing calls become the student's training rows, and the student is exported, measured and scored.

This page is the whole run sheet. Everything it names lives in this repository, and every command runs from a fresh clone.

## The commands

```
python3 scripts/bh_run.py PROFILE [STAGES] [options]

  PROFILE   g0 | s1 | laptop | eval | s2
  STAGES    none: the profile's whole plan, resuming (finished stages are skipped)
            one or more stages, or a span of the plan: sft..smoke, data.., ..sft
            round2 = round2-sample round2-repair round2-rft
            status: the plan with what is done, and the results so far (one shot)
  start [STAGES]   the same, detached: it keeps going when you log out (one log per run)
  watch            the live view of this profile's run: stage, clock, gates, tok/s, log tail
  stop             end the run cleanly (as Ctrl-C: uploads, then exits; start again resumes)
  --set key=value  override a config field (repeatable; scripts/bh/config.py)
  --dry-run DIR    on the CPU with mocks: DIR/<profile> is the machine, DIR/bucket the bucket
  --force          export even when the smoke said BROKEN
  --shutdown       power the VM off once everything is uploaded
```

That is the whole interface. `scripts/bh/data.py` is the data path that bh_run.py calls. You only run it yourself for its GPU-free check on the public spike: `python3 scripts/bh/data.py smoke --work /tmp/bh`.

## Profiles: where each runs and what it does

| Profile | Machine | Plan | Budget |
|---|---|---|---|
| `g0` | any Linux GPU box (the 4060, WSL2 too) | every stage; Qwen3.5-0.8B taught by Qwen3.5-2B, 20 kits, LoRA, tiny caps; its G2 only reports | an hour or two |
| `s1` | B200 session 1 | setup, data, teacher-serve, teacher-answers, sft, smoke | 200 min, billed from boot |
| `laptop` | the 4060 laptop between sessions | export, footprint, round2-sample | none |
| `eval` | the second laptop (or the same one afterwards) | score; waits up to 4 h for the shipped GGUF to reach the bucket | none |
| `s2` | B200 session 2 | setup, data, teacher-serve, round2-sample (if the laptops didn't), round2-repair, round2-rft | 150 min |

Run them in order: `s1`, then `laptop` and `eval` (at the same time on two laptops, or one after the other on one), then `s2`, then `laptop --set student=round2` and `eval --set student=round2` to ship and score the round-2 model. The machines hand over through the bucket. Each one fetches what an earlier machine made, so a fresh machine needs only the clone and the bucket login.

## Before a run

Check the plumbing first. It needs only `python3` (3.10 or newer), `bash` and `tar`, and no GPU, models, bun or bucket:

```
git clone -b claude/centipede-train-pipeline https://github.com/OliverVillson/llmap && cd llmap
python3 scripts/bh_run.py g0 --dry-run /tmp/bh-dry
for p in s1 laptop eval s2; do python3 scripts/bh_run.py $p --dry-run /tmp/bh-dry2; done   # the hand-over
```

Then, once per machine:

- **The bucket (every machine):** `evroc login --oidc-port 8765`. Every machine of a run must use the same bucket. On one machine you can use a local folder instead: `export BH_BUCKET=/some/dir`.
- **The B200:** nothing. The `setup` stage runs `scripts/setup_vm.sh` with exact pins (vLLM 0.31.0, transformers 5.x, the llama.cpp commit, UV_EXCLUDE_NEWER), builds MuggeBench's sandbox images and boot-checks the teacher ladder. It refuses to start a paid session when a pin is loose. Run the profile inside `tmux`, so a dropped ssh does not stop it.
- **Each laptop (the 4060 and the second one; Ubuntu or WSL2 with an NVIDIA GPU):** clone into the Linux home (`~/llmap`, not `/mnt/c/...`), then run these once:
  ```
  NVME=~/bh NVME_LOCAL=1 SKIP_EXTRAS=1 TEACHER=Qwen/Qwen3.5-2B STUDENT=Qwen/Qwen3.5-0.8B bash scripts/setup_vm.sh
  curl -fsSL https://bun.sh/install | bash
  cd mugge && ENGINE=docker bash sandbox/build.sh && cd ..   # the bench's sandbox images (needs docker)
  ```
  The first line builds llama.cpp and the two venvs in `~/bh` (the harness's default `NVME` on a laptop) and downloads the G0 models. Do not run `scripts/setup_muggebench.sh` on a laptop: it is for the VMs and resets Docker's storage.
- **Data (the `data` stage, on the B200 or for G0):** bhdata, BlackHole's kit library. Set `BHDATA` to its package directory (the folder holding `llmap.py`), unless it is at `/mnt/project-files/blackhole/data/bhdata`, `~/blackhole/data/bhdata` or `~/bhdata`.
- **Scoring on the Mac:** set `MUGGEBENCH_SH` to MuggeBench's `muggebench.sh`, and run `python3 scripts/bh_run.py eval`. Without it, `score` runs `mugge/bench/run.ts` against llama-server, on the spike or on the dev set given by `--set score_bench=DIR`.

## Stages

| Stage | What it does | Writes (under W = `$NVME/jobs/bh/<run>`) |
|---|---|---|
| setup | the VM, pins, sandbox images, teacher boot dry-run | `W/setup.json`, `W/teacher/boot.json` |
| data | `data.py export` (bhdata kits → train-bench, heldout-bench, free rows), then `check` (every reference answer in the sandbox; MuggeBench decontam) | `W/data/export/`, `W/data/stats/check-*.json` |
| teacher-serve | vLLM with the teacher, thinking off; gate G2 | `W/teacher/teacher.json` |
| teacher-answers | the teacher works train-bench through `bench/run.ts` (2-4 seeds, transcripts, its top-20 log-probs), then `data.py teacher`, `extra` and `build --round 1`, then the row guards | `W/rows/train.jsonl`, `rows.json`, `topk.jsonl.gz` |
| sft | full FT (LoRA on G0) with masked prompts and top-20 logit KL on the Qwen path | `W/sft/sft/model/` |
| smoke | the bf16 student on the 15 public spike tickets (llama-server BF16 GGUF when vLLM can't load it) | `W/smoke/sft/smoke.json` |
| export | vocab cut, BF16 GGUF, imatrix on training-shaped text, the quant ladder | `W/export/<student>/` |
| footprint | each quant's llama.cpp footprint (one 16K q8 slot after a 12K prompt); the best that fits 2 GB ships | `W/ship/<student>/BH<v>-<GB>.gguf`, `W/footprint/` |
| score | MuggeBench through `muggebench.sh`, else `bench/run.ts`: mean per-ticket solve rate with a 95% CI | `W/score/<student>/score.json` |
| round2-sample | the student answers held-out tickets (laptop GGUF or B200 bf16), transcripts kept | `W/round2/samples/` |
| round2-repair | `data.py repair-requests`, then `answer` (the teacher, through the proxy), `repair`, `rft` and `build --round 2` | `W/rows/train-round2.jsonl` |
| round2-rft | SFT from the base on teacher, extra, own-repair and the student's own passes, then its smoke | `W/sft/round2/model/` |

## Running unattended

- **Resume:**
  - A stage whose output exists is skipped, so rerunning the same command continues where it stopped, also after a crash or a kill.
  - Teacher answers resume per shard, the SFT from its checkpoint, the export per file.
  - Inputs another machine made are fetched from the bucket.
  - To redo a finished stage, delete its folder under W (the table above). `score` reruns by itself when a new GGUF ships.
- **Exit code:** 0 when every stage finished or was skipped, and non-zero when a required stage failed. Optional steps (vocab cut, tokenizer checks, bhdata's free rows, the smoke) report and carry on. There are no prompts.
- **Uploads:** after every step, and when the run ends however it ends (a failure, Ctrl-C, a kill or a dropped ssh), `results-<profile>.tar` (everything small, plus the logs) and the files later machines need go to `bucket://mugge-library/blackhole/<run>/`. A hand-over file that fails to upload is retried, in order, after every later step and by the next run; `status` lists any that are still waiting.
- **Shutdown:** `--shutdown` powers a VM off only after a clean finish whose last upload went through.
- **Gates that decide on their own:**
  - **G2:** the first teacher of the ladder that solves at least 55% of 200 kit tickets at 250 answers/min or more within 15 minutes is used. Otherwise the next teacher is tried, and the run stops when none passes. The ladder is Qwen3.8-27B-NVFP4, then Qwen3.6-35B-A3B-NVFP4, then Qwen3.6-35B-A3B-FP8.
  - **Smoke:** a student that writes no files is BROKEN, and export refuses it unless you pass `--force`.
  - **Session clock:** on a B200, each step gets what the session can spare after keeping time for the steps that follow. A bench run is stopped at its share and what finished counts; a step with no time left is skipped.
  - **GPU watchdog:** a long GPU step whose GPU sits idle for its first 5 minutes is stopped (for the SFT, the 5 minutes start when it loads the model, after reading the rows on the CPU).
  - **Drops:** when more than 20% of answers drop for length or parsing, the run warns and prints the counts. Cut, forced and thinking answers never become rows.
- **Read the results:** `python3 scripts/bh_run.py PROFILE status` (the plan, what is done, the results, how the last run ended and its warnings), or `results-<profile>.tar` in the bucket.

## Checking in over SSH

Start a run detached on the machine, log out, and check on it from the Mac whenever you like. Each command takes the same profile and options the run was started with.

```
ssh laptop1                                          # or the B200, or WSL2 (below)
cd llmap
python3 scripts/bh_run.py g0 start                   # detached; prints its log, pid and these hints
python3 scripts/bh_run.py g0 watch                   # live: Ctrl-C leaves the view, not the run
python3 scripts/bh_run.py g0 status                  # one shot: the plan, what is done, the results
python3 scripts/bh_run.py g0 stop                    # clean stop; `g0 start` again resumes
```

Or in one line from the Mac: `ssh -t laptop1 'cd llmap && python3 scripts/bh_run.py g0 watch'`.

- **What watch shows**, redrawn every 3 seconds:
  - the run's state, pid and uptime (and the minutes left on a B200 session);
  - every stage as done ✓, running ▶, waiting ·, skipped –, stopped ■ or failed ✗;
  - the current stage's time against its guess, and its progress when it reports one (the SFT's update count);
  - tok/s over the last minute from the recording proxy, with how many answers were cut;
  - the gate readings so far (G2, smoke, footprint, score), the run's warnings, and the last lines of its log.
- **One run per profile.** A second `start` of the same profile is refused while one is going, and so is a plain run in the foreground.
- **Stop.** `stop` sends Ctrl-C: the step in flight stops, the servers stop, the results upload, and the run exits 130. A second `stop` while it is still uploading interrupts that too. A dropped ssh or a kill ends a foreground run the same way.
- **Files**, in `~/bh-logs/<run>/` (BH_LOGS):
  - `<profile>-current.log` is the newest run's whole output: `tail -f ~/bh-logs/*/g0-current.log`.
  - `NN-<stage>.log` holds each step's own output.
  - `<profile>-run.json` is what watch reads: the state, the stage and when it started, each stage's state, and the exit code.
- **Crew and smoke runs.** With the engine's telemetry (PR #14), each bench run also writes the engine's own `status.json` (agents working and queued, tickets done, tok/s) next to its results under W. watch shows it as a `crew` line. To see it whole, run `cd mugge && bun src/cli.ts status <that folder> --watch`. A smoke started by hand (`cd mugge && bun bench/smoke.ts`) writes it to `mugge/runs/smoke-<time>/`.
- **Keep the machine awake.** A detached run outlives the ssh session, but not a laptop that sleeps.
  - **Linux:** `sudo systemctl mask sleep.target suspend.target`, or set the lid to do nothing.
  - **macOS:** `caffeinate -dims python3 scripts/bh_run.py g0 watch` keeps it awake while you watch; for a whole run, use `sudo pmset -a sleep 0`.
  - **Windows:** turn sleep off in Windows (Power settings). WSL2 also stops when its last terminal closes. Either keep one WSL window open, or set `vmIdleTimeout=-1` under `[wsl2]` in `%UserProfile%\.wslconfig`.
  - **SSH into WSL2:** run `sshd` inside WSL, then port-forward from Windows: `netsh interface portproxy add v4tov4 listenport=2222 connectport=22 connectaddress=<WSL IP>`. Then `ssh -p 2222 you@laptop1`.

## Two laptops

Two laptops on one network split the work **by role**. They cannot pool their GPUs into one model, so neither can serve the teacher: the 27B NVFP4 teacher (and its fallbacks) needs the B200, and it stays there. The laptops share the work between sessions.

**By profile (the default).** Run both at once with the same bucket:

```
laptop1 (the 4060):   python3 scripts/bh_run.py laptop   # export, footprint, then round-2 samples
laptop2:              python3 scripts/bh_run.py eval     # waits for the shipped GGUF, then MuggeBench
```

Each laptop serves its own llama-server. laptop2 waits (up to 4 hours) until laptop1's footprint has uploaded the ship, then scores it while laptop1 samples the held-out tickets. Start them in either order.

**By machine (a role served elsewhere).** Every stage that talks to a model has a role:
- `teacher`: teacher-serve, teacher-answers and round2-repair.
- `student`: smoke and round2-sample.
- `eval`: score.

`BH_BACKENDS` (or `--set backends=...`) names the machines that serve roles. It uses the same JSON shape as the engine's `MUGGE_BACKENDS`: `name`, `url` up to `/v1`, `roles`, and an optional served `model`. A listed role sends its calls to that URL, through this machine's recording proxy (so the sampling and logs stay the same), and starts no server here. The bench's sandboxes still run on the machine that runs the stage.

```
# laptop1 drives round-2 sampling (the sandboxes run here) while laptop2's GPU serves the student
export BH_BACKENDS='[{"name": "laptop2", "url": "http://laptop2:8080/v1", "roles": ["student"]}]'
python3 scripts/bh_run.py laptop round2-sample

# on laptop2: the shipped GGUF as the harness serves it (12 slots of 16K, q8 KV; the proxy
# sends the sampling with every call)
llama-server -m BH0.1-1.7.gguf -a student -ngl 999 --host 0.0.0.0 --port 8080 -c 196608 -np 12 \
  --kv-unified --jinja -fa on -ctk q8_0 -ctv q8_0
```

A role served elsewhere must already be up: the stage checks `/v1/models` and stops with the machine's name if it does not answer. The file form works too: `BH_BACKENDS=@backends.json`.

## Config

All settings are in `scripts/bh/config.py` (`Config` and `PROFILE_SETTINGS`). The ones most often changed:

```
--set base=minicpm5-2b          # the other base: no logit KL (another tokenizer), 4-bit ladder
--set teachers=qwen3.6-35b-a3b-fp8
--set samples=4                 # teacher seeds per ticket
--set run=bh02 --set version=0.2
--set student=round2            # laptop and eval: ship and score the round-2 model
--set score_bench=/path/to/dev  # score without muggebench.sh
```

Environment variables, all optional:

| Variable | What it is |
|---|---|
| `NVME` | the fast disk: `/mnt/nvme` on the VMs, `~/bh` on a laptop |
| `BH_BUCKET` | a local folder instead of the evroc bucket |
| `BH_BACKENDS` | machines that serve a role (above) |
| `BHDATA` | bhdata's package directory |
| `MUGGEBENCH_SH` | BlackHole's `muggebench.sh`, for `score` |
| `FOOTPRINT_SH` | BlackHole's `footprint.sh`, the reference footprint measurement |

## The contract with the model (BlackHole)

These are fixed by the harness. Change them only together with BlackHole.

- **Rows:** every row is one real engine call, `{"messages": [system, user, assistant], "meta": {...}}`, rendered by the engine itself (`mugge/bench/render.ts`). The system prompt is `prompts.ts` SYSTEM. BH0.1 trains write, fix and write-from-fix rows. bhdata's review rows are set aside for BH0.2 (`W/data/rows/extra-review.jsonl`).
- **Sampling:** every call goes through the recording proxy, which forces `scripts/bh/sampling.py`: temperature 0.7, top_p 0.8, top_k 20 and min_p 0. Thinking is off (`enable_thinking=false`), each call is capped at 4096 tokens, and slots are 16K.
- **Logit KL:** the teacher's top-20 log-probs are recorded at generation time, so its distribution is truncated to 20 and renormalised.
  - KL applies where a row's (user, answer) matches a logged call.
  - write-from-fix rows and re-laid-out answers train on CE alone. `rows.json` reports the coverage.
- **Round 2:** an own-repair row's prompt is the fix call the engine would make next on the student's own failed files, with the real failing check. Round-2 RFT trains from the base.
- **Name:** `BH<v>-<GB>`, no hyphen after BH. GB is the measured footprint, rounded up to one decimal. Round 2's model of the same run ships as `BH<v>.1-<GB>` (for example `BH0.1-1.7`, then `BH0.1.1-1.7`); `--set round2_suffix=...` renames it. Eval scores the exact GGUF that run's footprint shipped for `--set student=sft` or `round2`, never the newest file under a name.
- **Never trained on:** MuggeBench tickets, the spike, or anything carrying the `MUGGE-BENCH-` canary. data.py's decontam and the row guards both check.

## Files

`scripts/bh_run.py` holds the stages and the command line. The other files are in `scripts/bh/`:

- **Configuration and sampling:** `config.py`, `sampling.py`.
- **Servers and the proxy:** `servers.py` (vLLM and llama-server) and `proxy.py` (the recording proxy).
- **Running the bench:** `bench.py` (`bench/run.ts` runs and their reading).
- **Training:** `rows.py` (the last guards and tokenization) and `sft.py` (the trainer).
- **Export and scoring:** `export.py`, `footprint.py`, `vllm_check.py`.
- **Dry-run stand-ins:** `exec.py`, `mock.py`.
- **The data path:** `data.py`, `harvest.py`, `check.py`, `decontam.py`, `render.py`, `bhdata_io.py`.

Tests: `tests/test_bh_run.py` and `tests/test_bh_data.py`.
