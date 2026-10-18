# Qwen on one GPU for a Centipede crew

One vLLM server with **Qwen/Qwen3.8-27B-FP8** (the official FP8 checkpoint, 31 GB) serves the
whole crew: 10 code agents, 2 lead reviewers and 1 orchestrator, all on one GPU. The evroc
B200 is the target; H100 and L40S notes are at the end.

| File | What it is |
|---|---|
| `qwen.sh` | setup, download, start (detached), wait, status, check, bench, compare, logs, stop, cmd |
| `check.py` | the server check: plain, thinking and JSON calls, then 13 calls at once; and the speed bench (stdlib only) |
| `crew-b200.json` | the crew's machine list (`MUGGE_BACKENDS`) |
| `crew.env` | the engine settings for that crew |
| `mock_vllm.py`, `test.sh` | a no-GPU stand-in for vLLM and the dry run CI runs |

## On the B200, from a fresh clone

Needs an NVIDIA driver (`nvidia-smi` works; 570 or newer for a B200) and about 50 GB free disk.

```bash
git clone https://github.com/OliverVillson/llmap && cd llmap/mugge
git checkout claude/qwen-serving        # until it is merged

deploy/qwen.sh setup      # vLLM 0.31.0 in ~/venvs/qwen-vllm, matched to the driver's CUDA
deploy/qwen.sh download   # the weights into ~/.cache/huggingface
deploy/qwen.sh start      # detached; survives logging out
deploy/qwen.sh wait       # first start takes several minutes (load, compile, CUDA graphs)
deploy/qwen.sh check      # must end in PASS
```

`wait` prints vLLM's own memory figure, for example
`KV cache size: 1,9xx,xxx tokens, Maximum concurrency for 65,536 tokens per request: 30x`.
The "maximum concurrency" there must be at least 13 (it counts every call at the full
65,536 tokens, so it is the worst case).

`check` ends in `PASS` when: the model answers; thinking comes back apart from the answer
(so file blocks never contain it); a JSON-schema call parses (the planner's calls); and 13 calls
at once all answer, ran together and none was preempted. It prints tokens per second together
and per call.

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
set -a; . deploy/crew.env; set +a       # from the mugge folder: the machine list path is relative
bun src/cli.ts backends                 # three entries, all "up"
```

`crew-b200.json` lists the one server three times, so each kind of call has slots of its own
and a burst of reviews never takes a worker's slot:

| Entry | Role | Slots | Who |
|---|---|---|---|
| b200-workers | worker | 10 | the 10 code agents: each ticket's write and fix calls |
| b200-leads | lead | 2 | lead review of each ticket whose checks pass |
| b200-orchestrator | manager | 1 | the architect, scaffold and ticket plan (`mugge plan`) |

`MUGGE_CONCURRENCY=10` lets 10 tickets run at once. At most 13 calls are in flight; the server
admits 32 at once (`MAX_NUM_SEQS`), so the crew can grow to about 25 workers by raising the
worker slots and `MUGGE_CONCURRENCY`, with no server change.

Each entry uses the engine's `qwen3.8-27b` preset (16 KV layers × 4 × 256, plus the DeltaNet
state) and `"server": "vllm"`. `crew.env` turns thinking on (`MUGGE_THINKING=on`, sent on every
call; the review call goes out with thinking off) and caps a call at 16,384 tokens. It sets no
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

| GPU | Memory × utilisation | Weights | Runtime | Left for KV | Profile | 13 calls need (at the profile's max) |
|---|---|---|---|---|---|---|
| B200 180 GB | 162 GB (0.90) | ~30 GB | ~10 GB | **~120 GB** | 64K max, bf16, 32 seqs | 58 GB at 64K each; 30 GB at 32K |
| H100 80 GB | 72 GB (0.90) | ~30 GB | ~6 GB | ~36 GB | 32K max, bf16, 16 seqs | 30 GB at 32K each |
| L40S 48 GB | 44 GB (0.92) | ~30 GB | ~4 GB | ~10 GB | 16K max, fp8 KV, 13 seqs | 9 GB at 16K each |

On the B200 the crew uses at most half the pool even if every call reaches 64K tokens.

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
deploy/qwen.sh compare                                 # tok/s alone and with 13 at once, NVFP4 against FP8
```

Speed is half the question; the other half is whether the files still come out exact. Run the
engine smoke against each (`bun bench/smoke.ts --url http://127.0.0.1:8000/v1 --model qwen3.8-27b`,
15 tickets) and keep NVFP4 only if its pass count and lost answers match FP8's.

What is known so far: NVIDIA's card shows NVFP4 within about 1 point of BF16 (GPQA 88.0 vs 88.9,
Terminal-Bench 74.0 vs 75.6, IFBench 78.9 vs 80.1). A community comparison on a DGX Spark found
the same on code (HumanEval 94.5 vs FP8's 95.1, MBPP 70.4 vs 69.3) with single-stream decode
1.9x FP8's, but a token distribution further from BF16 than FP8's (KL 0.076 against 0.012), and
thinking loops with every NVFP4 build it tried. On a B200 with 13 calls at once the gain will be
smaller than on that box, since the B200 has the memory bandwidth FP8 needs; `compare` says how
much. MTP with this checkpoint is not verified.

## H100 or L40S instead

The same commands; the profile comes from `nvidia-smi` (or `GPU=h100`, `GPU=l40s`).

- **H100 80 GB**: 32K tokens per call, bf16 KV, room for about 15 calls at 32K each. The crew
  file works as it is; set `ctx` to 32768 (it already is). Keep `MUGGE_MAX_CALL_TOKENS` plus the
  largest pack under 32K.
- **L40S 48 GB**: 16K tokens per call with an fp8 KV cache, about 13 calls at 16K each: the crew
  only just fits, with no thinking room. Leave MTP off. If `check` reports preemptions, use 8
  workers. vLLM's block-FP8 path on this GPU (Ada) is not verified.
