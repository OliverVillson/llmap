# MuggeBench on the laptops

The target is a Mugge score of 50% or more with a footprint of 2 GB or less. The footprint
is the model file plus its working memory for one agent at a 16k context.
[muggebench.sh](muggebench.sh) measures both for one GGUF with one command. It works on the
RTX 4060 laptop under WSL2 and on Apple Silicon Macs.

It reuses llmap's own scoring path rather than reimplementing it, so laptop scores are
comparable with the Centipede run on the B200:

- llmap's eval stage serves the GGUF with llama-server.
- The stage's forced-answer proxy sits in front of the server. When a thinking call reaches its token cap, the proxy closes the thinking and makes the model answer.
- `mugge/bench/run.ts` runs every ticket through the Mugge engine's write, check and fix loop, and grades it on hidden tests.

Sampling is BlackHole's one set (`SAMPLING=blackhole`, the default):
- Thinking on: temperature 0.6, top_p 0.95, top_k 20, with 8,192 tokens per call.
- Thinking off: temperature 0.7, top_p 0.8, top_k 20.
- min_p 0 and no repetition penalty, both ways.

`SAMPLING=centipede` uses the Centipede run's set instead (`scripts/centipede.py` EVAL: temperature 1.0, top_p 0.95, no top_k), to compare with the VM's numbers. Its results are kept apart, under `<name>-centipede`.

Other settings:
- 3 seeds.
- 8-bit KV cache, which fits more agents into 8 GB of VRAM.
- When a thinking call runs out of tokens with no answer, it is asked again at once, with the same prompt and thinking off, and the ticket's later calls go without thinking. That is what Centipede's harness does (llmap branch `claude/thinking-cap`), so a laptop score matches what the model would do in Centipede. `CUT_RETRY=force` brings back the old forced answer on top of the thinking.

## Where MuggeBench is

- **Runner:** llmap branch `claude/project-thread-3t4k6m`, pinned in the script. It is the MuggeBench branch plus Centipede's thinking-cap rule (`claude/thinking-cap`), the retry without thinking in the eval's proxy, and `scripts/mugge_rollouts.py`. The code is in `mugge/bench/` plus `stages/eval.py`. None of it has reached llmap's main branch yet.
- **Tickets:**
  - The public example is `mugge/spike`: 15 tickets in one project, across C, Python and TypeScript.
  - The real score comes from the private dev set, `muggebench-dev.tar.gz`. That file has been on Oliver's Mac (copied to the VM from `~/Downloads`) and in the evroc bucket at `bench/muggebench-dev.tar.gz`. The `llmap-bench` repo it comes from is not reachable from these threads.
  - Copy the tarball to the laptop's home folder, or set `BENCH_TAR`, and the script unpacks it next to the spike.

## The 4060 laptop (WSL2)

One-time setup, in a WSL2 Ubuntu shell:

1. Install the NVIDIA driver on Windows. WSL2 gets the GPU through it.
2. Inside WSL, run `sudo apt-get install -y nvidia-cuda-toolkit cmake build-essential git curl`. The script builds llama.cpp with CUDA on its first run, at the Centipede run's llama.cpp commit.
3. Install Docker Desktop for Windows and turn on WSL integration for your distro. The dev set needs it: its `ml` and `dotnet` tickets run in the Mugge sandbox images, and fix prompts show `/work` paths as in training. The script builds those images once, and with the dev set present it stops if neither Docker nor Podman is running (`SANDBOX=local` scores anyway, marked as not a deciding score). Without Docker, the spike runs in WSL with its own toolchains.
4. Keep the models and `~/muggebench` on the WSL filesystem, not `/mnt/c`. Reading from `/mnt/c` is very slow.

Then score a model:

```bash
GGUF=~/models/qwen3-1.7b-Q4_K_M.gguf bash muggebench.sh                 # the spike only, until the dev set is here
GGUF=~/models/qwen3-1.7b-Q4_K_M.gguf SEEDS=1 TICKETS=20 bash muggebench.sh  # a quick look on the dev set
GGUF=~/models/BH-0.1-1.7B-Q4_K_M.gguf bash muggebench.sh                # the full score, 3 seeds
```

## What the script does

1. Installs uv, Bun 1.3.14, and llama-server if they are missing.
2. Checks out llmap at the bench commit into `~/muggebench/llmap`.
3. Builds the sandbox images the bench needs.
4. Runs a preflight: every ticket once with its reference answer. This must score 100%, which proves the sandbox works on this machine.
5. Scores the model. Re-running skips steps that are done, and a model already scored is reported from its saved results. To score it again, delete `~/muggebench/jobs/<name>`.

Before scoring, it measures the footprint the way the training plan defines it:
- It serves the GGUF with one slot of 16,384 tokens (`FOOTPRINT_CTX`), the bench's KV cache type (`q8_0`) and flash attention.
- It sends a prompt of about 12,000 tokens (`FOOTPRINT_PROMPT`), cut from llmap's own source code, so the memory is the memory of a real long ticket.
- The footprint is the GGUF file's size plus llama-server's working memory: its KV cache, recurrent state, compute and output buffers. The weights count as the file, not as llama-server's model buffers, because on Metal those are two mapped views of the same file (`CPU_Mapped` and `MTL0_Mapped`), and summing both counted the weights twice. The raw sum of every buffer is still printed on a second line.
- Newer llama-server builds (brew's 0.5.0) log buffer sizes only at verbosity 4, so the script sets `LLAMA_ARG_LOG_VERBOSITY=4` for these runs. The result also prints the llama-server build, because on a Mac the script uses whatever `llama-server` brew installed, not the pinned commit.
- It does this twice: at llama.cpp's default micro-batch, and at `-ub 128` (`FOOTPRINT_UBATCH`). The compute buffer grows with the micro-batch, and Qwen3.5's 248K-token vocabulary can make it a few hundred MB at the default. The micro-batch changes prompt speed, not answers, so the smaller of the two totals is the footprint. Both are printed and saved in `footprint.json`.

Where `nvidia-smi` is available, it also prints the peak GPU memory during each run. That figure includes the CUDA context, so it is shown for reference only.

It prints:
- the Mugge score with its 95% interval
- how many calls ran out of thinking and were asked again without thinking
- results by theme
- the footprint, with the GGUF file size next to it
- one of three verdicts against the target: **MEETS**, **MISSES** or **TOO CLOSE TO CALL**. MEETS needs the interval's low end at or above 50% and a footprint of at most 2.0e9 bytes.

**The footprint is tighter than it looks.** For Qwen3-1.7B, a 16k KV cache at 8 bits is about 1.0 GB (28 layers × 8 KV heads × 128 dims × K and V). That is arithmetic, not a measurement. With a 1.1 GB Q4_K_M file and a few hundred MB of compute buffer, the total comes to about 2.4 to 2.6 GB, which misses the target. Ways to get under 2 GB:
- a smaller or more aggressively quantized file
- a 4-bit KV cache (`KV=q4_0`)
- an architecture with a small KV cache: fewer KV heads or layers, sliding-window attention, or a hybrid with recurrent state such as Granite 4.

## Is the thinking looping?

Every run also writes `~/muggebench/jobs/<name>/rollouts/<name>.jsonl`, one row per thinking call, in the rollout format `scripts/thinking_report.py` reads: thinking tokens, how the call ended, whether it ran out and was asked again, whether that answer solved the ticket, and the end of the thinking. The report tells thinking that loops from thinking that is long but finishes:

```bash
python3 ~/muggebench/llmap/scripts/thinking_report.py ~/muggebench/jobs/BH0.1-1.3/rollouts/BH0.1-1.3.jsonl
python3 ~/muggebench/llmap/scripts/thinking_report.py ~/muggebench/jobs/*/rollouts/*.jsonl --ref qwen3-1.7b
```

Calls sent with thinking off, which are a ticket's calls after one ran out, have no thinking and are left out. To convert another bench run: `python3 scripts/mugge_rollouts.py <bench out dir> --out <name>.jsonl`.

## 16K mode

By default every model call gets about 28.7k tokens of context: up to 16,384 of prompt, 8,192 of thinking and a 4,096-token forced answer (`limits()` in `stages/eval.py`). That matches the Centipede run, but it is more than the 16k the footprint is measured at.

`CTX16=1` scores the model the way it would run inside the 2 GB budget. Each call gets a 16,384-token context (`LOBBOT_EVAL_CTX`) and 4,096 tokens of thinking (`TOKENS`) before the answer is forced. That leaves about 8k tokens for the ticket prompt; a longer prompt runs out of context and that call fails, which is part of what this mode measures. Results go under `<name>-ctx16k`, apart from the default run's.

```bash
GGUF=~/models/BH0.1-1.3.gguf CTX16=1 bash muggebench.sh
```

## Memory on 8 GB

The slot count is sized from free VRAM, after the weights and a 2 GB reserve. Each slot needs about a 28.7k context by default (16k with `CTX16=1`). With an 8-bit KV cache, that is about 2 GB per slot for a 1.7B model. So a 2 GB model on the 4060 gets one or two slots, and the bench runs that many tickets at once. A full 3-seed dev-set run is therefore an overnight job there. Use `SEEDS=1 TICKETS=20` to compare candidates quickly, and run the full score only for the ones that look close.

If the laptop has 8 GB of system RAM (not just VRAM), WSL2 gets about half of it by default. Raise that in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=6GB`) so Docker, llama-server and the tests fit.

## Knobs

All are environment variables:

| Variable | Default | What it sets |
|---|---|---|
| `GGUF` | none, required | the model file |
| `NAME` | the file name | label for the results |
| `BENCH_TAR` | `~/muggebench-dev.tar.gz` | where the dev set is |
| `SEEDS` | 3 | seeds per ticket |
| `TICKETS` | all | score only this many tickets |
| `SLOTS` | sized from VRAM (2 on a Mac) | tickets at once |
| `SAMPLING` | blackhole | the sampling set: `blackhole` or `centipede` |
| `TEMP`, `TOP_P`, `TOP_K` | from `SAMPLING` | thinking-on sampling, one value at a time |
| `NOTHINK_TEMP`, `NOTHINK_TOP_P`, `NOTHINK_TOP_K` | from `SAMPLING` | thinking-off sampling |
| `THINKING`, `TOKENS`, `KV` | 1, 8192, q8_0 | thinking, tokens per call, KV cache |
| `SANDBOX` | Docker, else Podman | where tickets run; `local` only for the spike |
| `TARGET_SCORE`, `TARGET_GB` | 50, 2.0 | the target |
| `CTX16` | 0 | 1 scores with a 16k context per call and 4k of thinking |
| `CUT_RETRY` | nothink | what a thinking call that runs out gets: asked again without thinking (Centipede), or `force` |
| `FOOTPRINT_CTX` | 16384 | the context of the one agent the footprint is measured for |
| `FOOTPRINT_PROMPT` | 12000 | the prompt length the footprint is measured after |
| `FOOTPRINT_UBATCH` | 128 | the second micro-batch size measured (empty for the default only) |

Change sampling settings only for every model you compare. Two results are comparable only when they used the same settings and the same tickets.

## What has been checked

The whole script ran end to end on Linux, standing in for WSL2:
- A stand-in llama-server answered every spike ticket with its reference solution and sometimes "thought" past its cap, which exercised the forced-answer path.
- The run produced 15 of 15 solved, the right server flags (`-ctk q8_0 -ctv q8_0 -fa on`, 32k context per slot) and the verdict output.
- Both sampling sets reached the server as set: with `blackhole`, 0.6 / 0.95 / 20 on thinking calls and 0.7 / 0.8 / 20 on the calls asked again without thinking; with `centipede`, the old 1.0 / 0.95 / 100.
- Calls that "thought" past their cap were asked again without thinking (11 of 30 calls in a 2-seed run), the count was printed, and the rollouts file went through `thinking_report.py`, which flagged those looping thinkings.
- The footprint step parsed llama.cpp-style buffer log lines and gave the verdict from their sum. The buffer sizes in that test were made up, not measured.

Not checked yet:
- a real model
- a real CUDA build in WSL2
- Docker-sandboxed runs. Docker is not available where this was built.
