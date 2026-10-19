# The Centipede demo on a Mac

The same demo as the B200 run (deploy/README.md, demo/README.md), on one Apple-silicon Mac:
Qwen3.5-9B behind llama.cpp's `llama-server` on the Mac's GPU (Metal), with a crew sized to its
memory. On a 16 GB MacBook Air that is **8 worker slots + 2 lead slots + 1 orchestrator slot,
10 agents** (tickets in flight at once). Nothing here touches the B200 path (`deploy/qwen.sh`).

## Run it

From a terminal on the Mac, one line at a time (zsh or bash):

```sh
git clone --depth 1 -b claude/mac-demo https://github.com/OliverVillson/llmap.git ~/llmap-demo || git -C ~/llmap-demo pull --ff-only
cd ~/llmap-demo/mugge && deploy/mac.sh setup && deploy/mac.sh download && deploy/mac.sh wired && deploy/mac.sh start && deploy/mac.sh wait && deploy/mac.sh check
```

- `setup` installs llama.cpp with Homebrew, or upgrades it if it is too old for the flags used here.
- `download` fetches the 5.7 GB model. It resumes if cut off.
- `wired` lets the GPU use 12 GB until the next restart. macOS asks for your password.
- `check` ends with `PASS`.

Then the demo:

```sh
cd ~/llmap-demo/mugge && set -a && . deploy/mac.env && set +a && nohup bun demo/run.ts --plan orchestrator > demo.log 2>&1 &
until curl -fs -o /dev/null http://127.0.0.1:7340/; do sleep 2; done; open http://127.0.0.1:7340/
tail -f ~/llmap-demo/mugge/demo.log
```

The third line opens the live view in your browser as soon as the demo serves it. It stays up while
the crew works, and after the run ends. No tunnel is needed, because it runs on this Mac.
`--plan orchestrator` runs the built-in task (mdlite, 17 tickets) with the orchestrator writing the
ticket plan. If that plan fails, it falls back to the prepared one.

For your own task, use `bun demo/run.ts --prompt "…"`. It plans about one ticket per agent (10 here).

Quit LM Studio, Ollama and other heavy apps first. Plug the Mac in. `mac.sh start` keeps it awake
while the server runs.

## What it runs, and why

**Server: llama.cpp `llama-server`, not mlx-lm.**
- llama-server gives:
  - parallel slots with continuous batching on Metal
  - quantized KV cache (`-ctk/-ctv q8_0|q4_0`) that keeps batching on
  - JSON-schema answers applied after the thinking
  - thinking returned apart in `reasoning_content`
  - a per-request thinking budget (`thinking_budget_tokens`)
  - Qwen3.5's hybrid architecture (`qwen35`)
- mlx-lm's server is faster for one stream, but it falls short here:
  - it turns batching off when the KV cache is quantized (`--kv-bits`)
  - it has no `response_format` JSON schema
  - it returns thinking as `reasoning`
- Ollama runs one request at a time by default and sets the KV type globally.
- LM Studio batches only on its llama.cpp engine.

**Model: Qwen3.5-9B, Q4_K_M GGUF (unsloth, 5.68 GB).**
- Qwen3.5 is the only current Qwen family with small dense models (0.8B, 2B, 4B, 9B).
- Qwen3.6 and 3.8 start at 27B or at 35B-A3B (about 20 GB at 4 bits), which do not fit 16 GB with a crew.
- Same family as the B200 model, and the same thinking switch (`enable_thinking`).
- LiveCodeBench v6: 65.6 for 9B, 55.8 for 4B (Qwen's model card).
- `MODEL=4b` runs Qwen3.5-4B Q6_K (3.5 GB): faster and lighter, but it codes worse.

**KV cache.** Qwen3.5 is hybrid:
- Only 8 of its 32 layers keep a KV cache: 4 KV heads × 256 dims, so 32 KiB a token at f16 and 17 KiB at q8_0.
- The other 24 layers each keep a fixed Gated DeltaNet state, about 50 MiB per sequence.

The settings, and why each was picked:

| Setting | Value | Why |
|---|---|---|
| KV type | q8_0 for K and V | half of f16, near-lossless; `KV=q4_0` or `KV_V=q4_0` if memory is short |
| Context per slot | 16,384 tokens | a pack of up to ~10K plus a call of up to 6,144 tokens (`MUGGE_MAX_CALL_TOKENS`) |
| Slots | one per call: `-np 11 -c 180224` | each slot has its own 16K (below) |
| Prompt reuse | `--ctx-checkpoints 2` | a ticket's fix call reuses its write call's pack: measured, only ~500 of ~4,400 prompt tokens processed again. Each checkpoint is ~50 MB of RAM |
| Host prompt cache | `--cache-ram 0` | its default is 8 GB, half of this Mac |

**Why no shared KV pool (`--kv-unified`).** Measured on llama.cpp master (2026-10-19): when a
shared pool runs out, llama-server fails every call running on it at once with HTTP 500 "Context
size has been exceeded". It does not pause any of them the way vLLM preempts. With a slot of its
own, a call that outgrows its context is the only one cut, with `finish_reason: length`. So the crew
must fit with every call at its longest, and `mac_crew.py` sizes it that way.

**Memory on a 16 GB Mac** (`deploy/mac.sh crew` prints this for the Mac it runs on):

```
the GPU may use       11.45 GB   (macOS default, ~2/3 of 16 GiB; 12.88 GB after deploy/mac.sh wired)
weights                5.68 GB   Qwen3.5-9B-Q4_K_M
Metal compute          1.00 GB   (estimate)
margin                 0.50 GB
one slot             322 MiB    16,384 tokens x 17.0 KiB (q8_0) + 50.2 MiB recurrent state
11 slots               3.72 GB   8 workers + 2 leads + 1 orchestrator
GPU total             10.90 GB   fits under the default limit; memory alone would allow 12 slots
left for macOS etc.  ~6.3 GB     engine, checks, browser
```

Workers are capped at 8 (`MAX_WORKERS`). Past that the GPU is already busy, and more slots only
make every call slower. With `wired`, `MAX_WORKERS=10` fits too: 10 + 2 + 1 slots, 11.4 GB.

| Mac | GPU limit | Model | Crew (w+l+o) | Agents | GPU total |
|---|---|---|---|---|---|
| 16 GB | 11.5 GB | 9B Q4_K_M | 8+2+1 | 10 | 10.9 GB |
| 16 GB | 11.5 GB | 4B Q6_K | 8+2+1 | 10 | 8.4 GB |
| 24 GB | 17.2 GB | 9B Q6_K | 8+2+1 | 10 | 12.7 GB |
| 32 GB | 22.9 GB | 9B Q8_0 | 16+2+1 | 20 | 17.4 GB |

**Thinking.** Qwen3.5's chat template ignores `reasoning_effort`. `deploy/mac.env` turns each level
into a thinking budget instead (`MUGGE_THINKING_BUDGET=low=1024,medium=4096,xhigh=8192`, sent as
`thinking_budget_tokens`). llama-server ends the thinking at that point and the answer follows.
- Workers think at low: up to 1,024 tokens.
- Leads review with thinking off.
- The orchestrator plans at medium: up to 4,096 tokens, inside a 12,288-token planning call.

## Commands

`deploy/mac.sh setup | wired | download | crew | cmd | preflight | start | wait | status | check | bench | logs | stop`.

Knobs given to `start` are remembered in `~/qwen-mac`. Later commands and `deploy/mac.env` read
them back.

| Knob | Values | What it does |
|---|---|---|
| `MODEL` | `9b`, `4b`, or a `.gguf` path (with `PRESET=`) | which model to serve |
| `QUANT` | `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` | the GGUF's quantization |
| `KV` | `q8_0`, `q4_0`, `f16` | KV type for both K and V; `KV_K` / `KV_V` set them apart |
| `CTX` | default 16384 | tokens of context per slot |
| `CREW_WORKERS` | a number | this many workers, whatever memory says (refused if it does not fit) |
| `MAX_WORKERS` | default 8 | the cap on workers when memory decides |
| `PORT` | default 8080 | the server's port |
| `EXTRA_ARGS` | llama-server flags | appended to the command |

A GGUF you already have, for example the expert-pruned Qwen3.6-35B-A3B:

```sh
deploy/mac.sh stop; MODEL=~/llmap-mac/models/qwen36-r50w95s.gguf PRESET=qwen3.6-35b-a3b CREW_WORKERS=4 deploy/mac.sh start && deploy/mac.sh wait
```

It measured ~28 tok/s per call at 2 calls on this Mac. Its 9.3 GB leaves room for only a small
crew, and only after `deploy/mac.sh wired`.

## If something fails

| Symptom | Fix |
|---|---|
| `start` refuses | each FAIL line from preflight says the fix |
| server died at start | `wait` prints the log's errors and a likely fix (old llama.cpp: `brew upgrade llama.cpp`; out of memory: quit apps, then `KV=q4_0` or `CREW_WORKERS=6`) |
| swap climbs during a run (`deploy/mac.sh status`) | the crew is too big for what else is open. Stop, then `CREW_WORKERS=6 deploy/mac.sh start` |
| calls cut (`finish_reason: length`) more than rarely | lower `MUGGE_MAX_CALL_TOKENS`, or raise `CTX` |

Paste failures in the project chat.

## Not verified until it runs on the Mac

- Metal speed: measured on Oliver's M5 Air (16 GB, llama.cpp brew 0.7.0, 2026-10-19). With calls that read 500 tokens and write 128,
  1 call made 13 tok/s and 8 at once made 22 tok/s together, with the GPU 99% busy. The 9B is limited by the GPU's compute, not
  memory, so more slots add little, and the 17-ticket demo takes about an hour. llama.cpp 0.5.0 (build 11146) made 8 calls at once
  no faster than 1; `mac.sh` now refuses builds before 11669.
- The real Metal compute-buffer size (the plan assumes 1 GB).
- Whether 11 slots at full context stay out of swap with a browser open.
- How well the 9B plans and codes the demo task.

Tested without a Mac (2026-10-19):
- the scripts against `deploy/mock_llama.py` (CI, deploy/test.sh)
- the engine and `mac.sh check` against a real llama.cpp master `llama-server` on CPU, serving a random-weight model with Qwen3.5's hybrid layout, tokenizer and chat template. This covered:
  - thinking on/off
  - the thinking budget
  - JSON with thinking
  - the context refusal
  - 11 slots at once
  - a full `demo/run.ts --plan orchestrator` run (`harness/mac-demo.md` in the project files)
