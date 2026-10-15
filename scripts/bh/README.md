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
            status: the plan with what is done, and the results so far
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

- **Every machine:**
  - Python 3.12 or newer, and bun (the engine and its bench live in `mugge/`).
  - Network access to the bucket: `evroc login --oidc-port 8765`. To use a local folder instead, set `BH_BUCKET=/some/dir`.
- **GPU boxes (B200, the 4060):**
  - The `setup` stage runs `scripts/setup_vm.sh` with exact pins: vLLM 0.31.0, transformers 5.x, the llama.cpp commit and UV_EXCLUDE_NEWER.
  - It also builds MuggeBench's sandbox images and does a boot dry-run of the teacher ladder.
  - A B200 refuses to start a paid session when a pin is loose.
  - On the 4060, run `NVME=<dir> SKIP_POLYGLOT=1 bash scripts/setup_vm.sh` once yourself.
- **Data:** bhdata, BlackHole's kit library.
  - Set `BHDATA` to its package directory (the folder holding `llmap.py`), unless it sits at `/mnt/project-files/blackhole/data/bhdata`, `~/blackhole/data/bhdata` or `~/bhdata`.
- **Score on the Mac:** set `MUGGEBENCH_SH` to MuggeBench's `muggebench.sh`. Without it, score runs `mugge/bench/run.ts` on the dev set given by `--set score_bench=DIR`.

To check the plumbing first, with no GPU, models or bucket: `python3 scripts/bh_run.py g0 --dry-run /tmp/bh-dry`. Running `s1`, `laptop`, `eval` and `s2` in turn with the same DIR exercises the real hand-over.

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
| footprint | each quant's llama.cpp footprint (one 16K q8 slot after a 12K prompt); the best that fits 2 GB ships | `W/ship/BH<v>-<GB>.gguf`, `W/footprint/` |
| score | MuggeBench through `muggebench.sh`, else `bench/run.ts`: mean per-ticket solve rate with a 95% CI | `W/score/<student>/score.json` |
| round2-sample | the student answers held-out tickets (laptop GGUF or B200 bf16), transcripts kept | `W/round2/samples/` |
| round2-repair | `data.py repair-requests`, then `answer` (the teacher, through the proxy), `repair`, `rft` and `build --round 2` | `W/rows/train-round2.jsonl` |
| round2-rft | SFT from the base on teacher, extra, own-repair and the student's own passes, then its smoke | `W/sft/round2/model/` |

## Running unattended

- **Resume:**
  - A stage whose output exists is skipped, so rerunning the same command continues where it stopped.
  - Teacher answers resume per shard, and the SFT resumes from its checkpoint.
  - Inputs another machine made are fetched from the bucket.
- **Exit code:** 0 when every stage finished or was skipped, and non-zero when a required stage failed. Optional steps (vocab cut, tokenizer check, smoke bench) report and carry on. There are no prompts.
- **Uploads:** after every step, and when the run ends however it ends, `results-<profile>.tar` (everything small, plus the logs) and the files later machines need go to `bucket://mugge-library/blackhole/<run>/`.
- **Shutdown:** `--shutdown` powers a VM off only after a clean finish with everything uploaded.
- **Gates that decide on their own:**
  - **G2:** the first teacher of the ladder that solves at least 55% of 200 kit tickets at 250 answers/min or more within 15 minutes is used. Otherwise the next teacher is tried, and the run stops when none passes. The ladder is Qwen3.8-27B-NVFP4, then Qwen3.6-35B-A3B-NVFP4, then Qwen3.6-35B-A3B-FP8.
  - **Smoke:** a student that writes no files is BROKEN, and export refuses it unless you pass `--force`.
  - **Session clock:** on a B200, a step that cannot finish in the session's remaining minutes is shortened or skipped, keeping time for the steps after it.
  - **GPU watchdog:** a long step whose GPU sits idle for its first 5 minutes is stopped.
  - **Drops:** when more than 20% of answers drop for length or parsing, the run warns and prints the counts. Cut, forced and thinking answers never become rows.
- **Read the results:** `python3 scripts/bh_run.py PROFILE status`, or `results-<profile>.tar` in the bucket.

## Two laptops

Two laptops on one network split the work **by role**. They cannot pool their GPUs into one model, so neither can serve the teacher: the 27B NVFP4 teacher (and its fallbacks) needs the B200, and it stays there. The laptops share the work between sessions.

**By profile (the default).** Run both at once with the same bucket:

```
laptop1 (the 4060):   python3 scripts/bh_run.py laptop   # export, footprint, then round-2 samples
laptop2:              python3 scripts/bh_run.py eval     # waits for the shipped GGUF, then MuggeBench
```

Each laptop serves its own llama-server. laptop2's score starts when laptop1's footprint uploads the ship, and it runs while laptop1 samples the held-out tickets.

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

Environment variables: `NVME` (the fast disk), `BH_WORK`, `BH_LOGS`, `BH_BUCKET`, `BH_BACKENDS`, `BHDATA`, `MUGGEBENCH_SH`, `FOOTPRINT_SH` and `CODEBENCH`.

## The contract with the model (BlackHole)

These are fixed by the harness. Change them only together with BlackHole.

- **Rows:** every row is one real engine call, `{"messages": [system, user, assistant], "meta": {...}}`, rendered by the engine itself (`mugge/bench/render.ts`). The system prompt is `prompts.ts` SYSTEM. BH0.1 trains write, fix and write-from-fix rows. bhdata's review rows are set aside for BH0.2 (`W/data/rows/extra-review.jsonl`).
- **Sampling:** every call goes through the recording proxy, which forces `scripts/bh/sampling.py`: temperature 0.7, top_p 0.8, top_k 20 and min_p 0. Thinking is off (`enable_thinking=false`), each call is capped at 4096 tokens, and slots are 16K.
- **Logit KL:** the teacher's top-20 log-probs are recorded at generation time, so its distribution is truncated to 20 and renormalised.
  - KL applies where a row's (user, answer) matches a logged call.
  - write-from-fix rows and re-laid-out answers train on CE alone. `rows.json` reports the coverage.
- **Round 2:** an own-repair row's prompt is the fix call the engine would make next on the student's own failed files, with the real failing check. Round-2 RFT trains from the base.
- **Name:** `BH<v>-<GB>`. GB is the measured footprint, rounded up to one decimal.
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
