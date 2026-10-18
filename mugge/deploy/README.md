# Qwen on one GPU for a Centipede crew

One vLLM server with **Qwen/Qwen3.8-27B-FP8** (the official FP8 checkpoint, 31 GB) serves the
whole crew on one GPU: the first run is 64 worker slots, 8 lead reviewer slots and 1 orchestrator
slot, with 80 code agents taking turns on the worker slots (`CREW_WORKERS=64`; without it the
crew is 10 worker slots, 13 agents, 2 leads and 1 orchestrator). The evroc B200
is the target; H100 and L40S notes are at the end.

| File | What it is |
|---|---|
| `qwen.sh` | setup, download, start (detached), wait, status, check, bench, sweep, compare, logs, stop, cmd |
| `check.py` | the server check (plain, thinking and JSON calls, then the crew at once) and the speed bench (stdlib only) |
| `crew.py`, `crew-b200.json` | the crew's machine list (`MUGGE_BACKENDS`) for `CREW_WORKERS`; the JSON is the default crew of 10 |
| `crew.env` | the engine settings for that crew |
| `mock_vllm.py`, `test.sh` | a no-GPU stand-in for vLLM and the dry run CI runs |

## On the B200, from a fresh clone

Needs an NVIDIA driver (`nvidia-smi` works; 570 or newer for a B200) and about 50 GB free disk.

```bash
git clone https://github.com/OliverVillson/llmap && cd llmap/mugge
git checkout claude/qwen-serving        # until it is merged

deploy/qwen.sh setup                     # vLLM 0.31.0 in ~/venvs/qwen-vllm, matched to the driver's CUDA
deploy/qwen.sh download                  # the weights into ~/.cache/huggingface
CREW_WORKERS=64 deploy/qwen.sh start     # detached; survives logging out; the crew size sticks
deploy/qwen.sh wait                      # first start takes several minutes (load, compile, CUDA graphs)
deploy/qwen.sh check                     # must end in PASS
nproc; free -g                           # the agents' checks run on these CPUs (see "CPUs" below)
```

`CREW_WORKERS=64` is given once, at `start`: the server then admits 73 calls at once (64 workers,
8 leads, 1 orchestrator), and `wait`, `check`, `bench`, `status` and `deploy/crew.env` read the
size back, over ssh too. Starting again without it keeps 64; `CREW_WORKERS=10 deploy/qwen.sh start`
changes it.

`wait` prints vLLM's own pool and checks the crew against it, for example
`crew: 64 workers + 8 leads + 1 orchestrator need at most 137 GB of KV cache and state; the pool holds 1xx GB`.
On a 180 GB B200 that is a little over 100% and `wait` warns; that is expected, see "64 agents
and memory" below.

`check` ends in `PASS` when: the model answers; thinking comes back apart from the answer
(so file blocks never contain it); a JSON-schema call parses (the planner's calls); and the whole
crew's calls at once (73) all answer, ran together and none was preempted. It prints tokens per
second together and per call.

## Checking in over ssh from the Mac

```bash
ssh b200 'cd llmap/mugge && deploy/qwen.sh status'     # running?, requests running/waiting, KV use, preemptions
ssh b200 'tail -n 50 ~/qwen/serve.log'                 # the server log
ssh b200 'cd llmap/mugge && deploy/qwen.sh stop'
```

The server listens on 127.0.0.1 only. To use it from the Mac (an engine run there, or
`check.py`), open a tunnel and talk to your own port 8000:

```bash
ssh -N -L 8000:127.0.0.1:8000 b200 &
python3 deploy/check.py --url http://127.0.0.1:8000 --status
```

Do not open port 8000 to the internet: vLLM's `--api-key` does not cover every endpoint.
`qwen.sh` refuses a public `HOST` unless `ALLOW_PUBLIC=1` and `VLLM_API_KEY` are set.

## The crew

```bash
cd llmap/mugge
curl -fsSL https://bun.sh/install | bash && export PATH="$HOME/.bun/bin:$PATH" && bun install
set -a; . deploy/crew.env; set +a       # from the mugge folder; picks up the 64 from start
bun src/cli.ts backends                 # three entries (64, 8 and 1 slots), all "up"
```

The machine list (`deploy/crew.py`, loaded by `crew.env`) names the one server three times, so
each kind of call has slots of its own and a burst of reviews never takes a worker's slot:

| Entry | Role | Slots | Who |
|---|---|---|---|
| b200-workers | worker | `CREW_WORKERS` (64 for the first run) | the code agents: each ticket's write and fix calls |
| b200-leads | lead | one per 8 workers, at least 2 (8 for 64) | lead review of each ticket whose checks pass |
| b200-orchestrator | manager | 1 | the architect, scaffold and ticket plan (`mugge plan`) |

### Crew size: one setting

`CREW_WORKERS` sets the crew everywhere at once: the worker and lead slots, `MUGGE_CONCURRENCY`
(tickets at once: 1.25 per worker slot, 80 for 64, so a slot freed while an agent runs its checks
or waits for review goes straight to another agent), and the server's `--max-num-seqs` (raised to
cover every call the crew can have in flight: workers + leads + 1; the extra agents' calls wait
in the engine, not on the GPU). The ratios come from the engine's crew-scale measurements. Give it to `start`; everything after reads it back from `~/qwen/serve.crew`:

```bash
CREW_WORKERS=24 deploy/qwen.sh start && deploy/qwen.sh wait    # wait says whether 24 fit in memory
set -a; . deploy/crew.env; set +a                              # the engine gets 24 too
```

How to pick it: run `deploy/qwen.sh bench` and read the thinking table. "Together tok/s" is how
fast the whole crew writes; "per call tok/s" is how fast one agent does. Raise the crew while
together tok/s still grows clearly (more than about 1.3x from 13 to 32 at once) and per call
tok/s stays fast enough that one attempt (4,096 tokens) finishes in an acceptable time. Past the
point where together tok/s flattens, more agents only make each one slower. `wait` also prints
how many workers fit by memory if every call reached 32K tokens at once; that is the ceiling, and
on a B200 it is far above where speed flattens.

Each entry uses the engine's `qwen3.8-27b` preset (16 KV layers × 4 × 256, plus the DeltaNet
state) and `"server": "vllm"`. `crew.env` turns thinking on (`MUGGE_THINKING=on`) with an effort
per role, `MUGGE_REASONING_EFFORT=worker=medium,lead=xhigh,manager=xhigh`: workers write most of
the tokens and think less, the few review and plan calls think fully (`off` for a role sends its
calls without thinking). A call is capped at 16,384 tokens. It sets no
sampling: the engine sends Qwen's set per call from that call's thinking switch, and the server
pins none.

## Memory and slots

Qwen3.8-27B is hybrid: of its 64 layers only 16 keep a KV cache (4 KV heads × 256), the other
48 are Gated DeltaNet layers with a fixed state per sequence.

| Per call | bf16 KV (B200, H100) | fp8 KV (L40S) |
|---|---|---|
| KV per token: 2 × 16 layers × 4 heads × 256 × bytes | 64 KiB | 32 KiB |
| fixed state: 48 layers × (48 × 128 × 128 fp32 + conv 3 × 10,240 bf16) | 154 MB | 154 MB |
| one call at 16K tokens | 1.23 GB | 0.69 GB |
| one call at 32K tokens | 2.30 GB | 1.23 GB |
| one call at 64K tokens | 4.45 GB | 2.30 GB |

vLLM takes KV as a call grows, not up front, so these are ceilings. Estimates for the cache
pool (the server prints the real figure at start):

| GPU | Memory × utilisation | Weights | Runtime | Left for KV | Profile | The default crew of 13 calls needs |
|---|---|---|---|---|---|---|
| B200 180 GB | 169 GB (0.94) | ~30 GB | ~10 GB | **~129 GB** | 64K max, bf16, 64+ seqs | 24 GB (64 workers: 137 GB, below) |
| H100 80 GB | 72 GB (0.90) | ~30 GB | ~6 GB | ~36 GB | 32K max, bf16, 16 seqs | 30 GB at 32K each |
| L40S 48 GB | 44 GB (0.92) | ~30 GB | ~4 GB | ~10 GB | 16K max, fp8 KV, 13 seqs | 9 GB at 16K each |

### 64 agents and memory

The most each call can cost is its longest context plus the DeltaNet state (`deploy/crew.py`):

| Call | Longest context | bf16 KV | fp8 KV |
|---|---|---|---|
| worker | 26,624 tokens (pack ~10K + `MUGGE_MAX_CALL_TOKENS` 16,384) | 1.90 GB | 1.03 GB |
| lead | 24,576 tokens (review prompt and diff, thinking at xhigh up to 16,384) | 1.76 GB | 0.96 GB |
| orchestrator | 24,576 tokens (a plan call with thinking) | 1.76 GB | 0.96 GB |
| **64 + 8 + 1, all at their longest at once** | | **137 GB** | **74 GB** |

The B200 profile gives vLLM 94% of the GPU (`GPU_UTIL=0.94`); after ~30 GB of weights and the
runtime that leaves an estimated ~129 GB for the pool (the B200's 180 GB; if `nvidia-smi` shows
more, there is more). So the worst case, every slot at its longest at once, is about 8 GB over.
The 80 agents add nothing to this: the engine never has more calls in flight than the 73 slots.
Planning is not part of this sum either: plan calls are not capped by `MUGGE_MAX_CALL_TOKENS` and
can run to the server's 65,536-token limit, and above 16 agents the plan's parts are written in
parallel on the worker slots, but all of that happens before any ticket starts, while the worker
and lead slots are otherwise empty, so it uses a fraction of the same pool.

We keep bf16 KV as the default anyway, because the worst case does not happen in practice: a
worker at medium effort thinks far less than the 16,384 cap, an agent holds KV only while its
call runs (not while its checks run), calls start at their pack size and grow, and the 8 lead
slots are full only when 8 tickets pass their checks at the same moment. vLLM handles a short
overflow by pausing a call and resuming it. So `wait` will warn for 64 on a 180 GB B200; that is
expected. Then watch `deploy/qwen.sh status` during the run: preemptions there mean vLLM
paused calls for memory (they resume, slower). If they keep climbing, restart with an fp8 KV
cache, which halves the KV memory at a small quality cost:

```bash
deploy/qwen.sh stop && KV_DTYPE=fp8 deploy/qwen.sh start && deploy/qwen.sh wait
```

`max-model-len` (65,536) does not reserve memory, so it needs no change for 64.

### CPUs

Each agent runs its ticket's checks in a sandbox on the same VM, so 64 agents can mean 64 test
runs at once. Check `nproc` and `free -g` on the box: with far fewer cores than agents, the run is
bound by the checks rather than the GPU, and a smaller crew finishes as fast.

## Server flags, and why

| Flag | Why |
|---|---|
| `--served-model-name qwen3.8-27b Qwen/Qwen3.8-27B-FP8` | the short name the crew file uses, and the full id |
| `--reasoning-parser qwen3` | thinking goes to `reasoning_content`, never into the answer the engine parses for files |
| `--enable-prefix-caching` | every worker call shares the engine's system prompt and pack layout |
| `--language-model-only` | the engine sends text only; skips the vision encoder |
| `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization`, `--kv-cache-dtype` | per GPU, above |
| `MTP=3` (off by default) | `--speculative-config {"method":"mtp","num_speculative_tokens":3}`: the model's own draft head, faster decoding; turn it on once `check` passes without it, and run `check` again |

Overrides: any of `MAX_MODEL_LEN`, `KV_DTYPE`, `MAX_NUM_SEQS`, `GPU_UTIL`, `PORT`, `TP`, or
`EXTRA_ARGS="..."`. `deploy/qwen.sh cmd` prints the command without running anything.

## NVFP4 instead of FP8 (B200 only)

`QUANT=nvfp4` serves [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
(NVIDIA ModelOpt, September 2026): NVFP4 in the MLPs and `lm_head`, FP8 in the attention and
DeltaNet projections, and an fp8 KV cache, as in NVIDIA's own serve command. The crew file does
not change: the server answers to `qwen3.8-27b` either way. FP8 stays the default until the two
are measured on the box:

```bash
deploy/qwen.sh bench                                   # FP8 is running: saves ~/qwen/bench-fp8.json
deploy/qwen.sh stop
QUANT=nvfp4 deploy/qwen.sh download
QUANT=nvfp4 deploy/qwen.sh start && deploy/qwen.sh wait && deploy/qwen.sh check
deploy/qwen.sh bench                                   # saves ~/qwen/bench-nvfp4.json
deploy/qwen.sh compare                                 # NVFP4 against FP8 at 1, 13, 32 and 64 calls at once
```

Speed is half the question; the other half is whether the files still come out exact. Run the
engine smoke against each (`bun bench/smoke.ts --url http://127.0.0.1:8000/v1 --model qwen3.8-27b`,
15 tickets) and keep NVFP4 only if its pass count and lost answers match FP8's.

What is known so far: NVIDIA's card shows NVFP4 within about 1 point of BF16 (GPQA 88.0 vs 88.9,
Terminal-Bench 74.0 vs 75.6, IFBench 78.9 vs 80.1). A community comparison on a DGX Spark found
the same on code (HumanEval 94.5 vs FP8's 95.1, MBPP 70.4 vs 69.3) with single-stream decode
1.9x FP8's, but a token distribution further from BF16 than FP8's (KL 0.076 against 0.012), and
thinking loops with every NVFP4 build it tried. On a B200 with many calls at once the gain will be
smaller than on that box, since the B200 has the memory bandwidth FP8 needs; `compare` says how
much. MTP with this checkpoint is not verified.

## Making it faster

Measure before changing anything, on the box, with the crew idle:

```bash
setsid nohup deploy/qwen.sh sweep > ~/qwen/sweep.log 2>&1 &     # about 30 minutes: FP8 with MTP off, then on
tail -f ~/qwen/sweep.log                                          # ends with the comparison
```

`sweep` restarts the server once per variant (`SWEEP="fp8 fp8+mtp3"` by default; add `nvfp4` and
`nvfp4+mtp3` after `QUANT=nvfp4 deploy/qwen.sh download`), runs `bench` on each, then `compare`.
`bench` measures 1, 13, 32 and 64 calls at once, each reading its own ~8K-token pack, for two
answer lengths: `thinking` (4,096 tokens, one thinking attempt) and `short` (1,024, an answer
with thinking off). The server is left stopped; start the winner. Then, in this order:

1. **MTP** (`MTP=3`). Speculative decoding with the model's own draft head: the answers are the
   same tokens, only faster, so there is no quality cost. Keep it if `compare` shows it ahead at
   your crew size (its gain shrinks as more calls share the GPU).
2. **Fewer tokens per ticket.** Thinking is most of what a worker writes. `crew.env` already
   runs workers at `medium`; `worker=low` or `worker=off` in `MUGGE_REASONING_EFFORT` goes further (compare the bench's thinking and
   short tables for the time an attempt takes). Both can cost passed tickets: run the engine
   smoke (`bun bench/smoke.ts --url http://127.0.0.1:8000/v1 --model qwen3.8-27b`) before and after.
3. **Crew size.** Above: the most workers before together tok/s flattens.
4. **Prefix caching** is already on (`status` says `prefix caching on` and the hit rate). Every call
   shares the engine's system prompt, and a fix call shares its write call's pack, so nothing more
   to do; a hit rate near 0 would mean the packs changed order between calls.
5. **NVFP4** (above): only with the smoke matching FP8's.
6. **More GPUs.** A second server (another B200, or a second GPU on the same box with its own
   `PORT`) is another entry in the machine list with its own worker slots; the engine spreads
   calls across both. One 27B model does not need tensor parallelism.

## H100 or L40S instead

The same commands; the profile comes from `nvidia-smi` (or `GPU=h100`, `GPU=l40s`).

- **H100 80 GB**: 32K tokens per call, bf16 KV, room for about 15 calls at 32K each. The crew
  file works as it is; set `ctx` to 32768 (it already is). Keep `MUGGE_MAX_CALL_TOKENS` plus the
  largest pack under 32K.
- **L40S 48 GB**: 16K tokens per call with an fp8 KV cache, about 13 calls at 16K each: the crew
  only just fits, with no thinking room. Leave MTP off. If `check` reports preemptions, use 8
  workers. vLLM's block-FP8 path on this GPU (Ada) is not verified.
