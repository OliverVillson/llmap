#!/usr/bin/env python3
"""Event Horizon QAT: Granite 4.2 3B at 40 LiveCodeBench points per GB of its shipped
GGUF, on our own harness readings, on one B200 or in three sessions (plan:
/mnt/project-files/plan/event-horizon-qat.md, sections 3 and 5; runbook:
/mnt/project-files/plan/event-horizon.md).

    python3 scripts/event_horizon.py --session all       # one B200: measure, train and read back to back
    python3 scripts/event_horizon.py --session all --shutdown  # ... then power off once the results are uploaded
    python3 scripts/event_horizon.py --session measure   # L40S/B200: R0, training text, trim, PTQ, Block-AP, Gate 1
    python3 scripts/event_horizon.py --session train     # B200: end-to-end KD, on-policy rounds, export, choice.json
    python3 scripts/event_horizon.py --session read      # L40S/B200: Gate 2, the pre-registered record, the verdict
    python3 scripts/event_horizon.py --list [--session S]  # the steps and their time guesses (no GPU needed)
    python3 scripts/event_horizon.py summary             # every reading, the gates and the verdict

Session all is the three sessions' steps on one VM: the setup, the restore, the download
and the measure smoke tests run once (train's own smoke and rates step still runs), with
one upload step at the end. Re-run the same command after a stop: finished steps are
skipped, and a fresh VM gets back from the bucket whatever matches its hashes.

Goal: points/GB = our harness pass@1 (0-100) / (shipped GGUF bytes / 1e9), at least
GOAL_PTS_PER_GB. There is no adapter: the file is the whole artifact. The largest file
that can reach 40 is R0 x kept / 40 GB, so the byte budget follows the original's
reading R0: max(1.172, min(1.25, 0.85 x R0 / 40)) GB, and the PTQ map must pass a
preflight (its file within the budget, and 40 x its size <= R0) before Block-AP.

Gates, coded as rules (gate0(), preflight(), screen_gate(), round_rule(), parity()) and
written before any reading:
  Gate 0  after the original's reading: R0 >= 52, else stop
  Gate 1  after the Block-AP dev screen: R0 x (dev retention) / GB, reported (no stop)
  rounds  an on-policy round is kept only if its held-out on-policy KL drops >= 5%,
          loop% and cap% rise by at most 2 SE, and paired held-out pass@1 falls by at
          most 2 SE; rounds stop after 2 in a row not kept, or after 4 (qat onpolicy
          decides as it trains, one session per step so each round is uploaded as it
          ends; audit_rounds() checks each decision against the rule, and a
          disagreement stops the run before the export)
  parity  the final export: bit-exact pack, every type as mapped, size within 0.5% of
          the estimate, llama.cpp KLD <= 0.01 nats/token and top-1 >= 98% against a
          BF16 GGUF of the same weights, llama-tokenize == HF
  Gate 2  after the final dev screen: predicted R0 x retention / GB, reported (no stop)
A failed gate (or preflight, or parity) skips every step after it but the upload, which
still runs: the run ends cleanly (exit 0), the summary says where and why it stopped
(stop_line()), the curve is published, and --shutdown powers off under the same
fingerprint rule as after a full run.

Pre-registration: the reference reading writes the K_i file (answers per problem in the
record: 2 / 8 / 4 on problems the original went 0/4 / mixed / 4/4); the train steps
write choice.json (GGUF sha256 and bytes, record seeds 1000.., the K_i file's sha256,
the verdict rule, the Gate 2 threshold) before any reading of the final model. The
record then reads exactly that file once. Its verdict (summary()):
  reached              (R - 1.645 SE) / GB >= 40, with SE = 100 sqrt(sum_i p_i(1-p_i)/(K_i-1)) / N
  point estimate only  R / GB >= 40, the lower bound below
  not reached          otherwise
  blocked              a check failed: errored answers, a problem without its K_i
                       answers, settings or provenance that differ from the original's
                       reading, a parity gate, the GGUF not choice.json's, or the
                       llama-server reading more than 2 SE of the paired difference below
Each reading's harness settings are read back from its config.effective.json and
provenance.json and compared with the original's; nothing is assumed "the same".

Every LiveCodeBench reading runs on one GPU, an L40S or a B200: the one the original's
reference reading ran on, as its provenance.json records. That GPU defines the harness:
a later reading on another GPU name blocks the verdict (provenance_diff, for the dev
screens and the llama-server check too), and is refused before it starts
(gpu_mismatch()). So comparability holds within a run, all of whose readings are on one
GPU. Each reading keeps RECORD_SEQS of that GPU in flight (Config.vllm_seqs and
LOBBOT_VLLM_SEQS): the reference reading is given its GPU's value, and every later one
reads it back from the reference's config.json (record_seqs()). The sandbox runs on
physical cores - 2. Each reading keeps its thinking for the 8k re-scoring
(stages/budget.py) and gets a thinking report (scripts/thinking_report.py). Every GPU
job first waits for the last one's memory and fails, listing the processes holding it,
if it is not freed.

Artifacts: each step's outputs live on the NVMe in dirs named by hashes of what they
were made from (map, args, imatrix sha, traces sha and held-out split, vocab sha, eval
settings), so a changed input never reuses an old result. After every step the runner's
after_step hook uploads them to <bucket>/horizon/<run-id>/<step>/<hash>/ (HORIZON_RUN
names the run), and once more however the run ends; the first step after setup restores
whatever matches. W/.uploaded on the NVMe holds the fingerprint of what was uploaded, so
it is wiped with the results; --shutdown powers off only after a run that exited 0 and
whose results still match it. The evroc login is checked at the start (--shutdown
refuses to start without one) and the last step waits a bounded time for one.

The stages run through their CLIs (python -m stages.X), in the training venv that
.env.vm puts first on PATH; this script itself is stdlib only, and its `smoke` and
`rescore` helpers are the venv-side parts (python scripts/event_horizon.py smoke ...).
Logs are in ~/event-horizon-logs/<session> (HORIZON_LOGS; session all's in .../all), and
in the bucket.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import thinking_report as tr
from blackhole import share_kept
from runner import REPO, Done, Skip, Step, above, evroc_ok, run_plan, sh

SESSIONS = ("measure", "train", "read")  # the three-session mode: L40S (or B200), B200, L40S (or B200)
ALL = "all"  # the three sessions' steps on one B200, back to back
READS = ("measure", "read", ALL)  # the sessions that read LiveCodeBench
GPUS = ("L40S", "B200")  # the GPUs the run knows: readings on either, training on the B200
MACHINE = {"measure": "L40S", "train": "B200", "read": "L40S", ALL: "B200"}  # each session's GPU when none is found
NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
J = Path(os.environ.get("LOBBOT_JOBS", NVME / "jobs"))
W = J / "event-horizon"  # everything this run writes, on the NVMe
JOBS = W / "jobs"  # the eval jobs, one per reading
LOGS = Path(os.environ.get("HORIZON_LOGS", Path.home() / "event-horizon-logs"))
RUN_ID = os.environ.get("HORIZON_RUN", "eh1")
BUCKET = "bucket://mugge-library"
MODEL = "ibm-granite/granite-4.2-3b"
# The revision `hf download` reads. TODO: session 1 records the commit "main" resolved to
# (the summary's "Granite 4.2 3B at ..." line, each reading's provenance.json env,
# choice.json); put that commit here before session 2, so every session trains and reads
# the same weights.
MODEL_REVISION = "main"
HF_DIR = NVME / "models" / "granite-4.2-3b"
BF16 = NVME / "models" / "gguf" / "granite-4.2-3b-BF16.gguf"
LLAMA_CPP = NVME / "llama.cpp"
LLAMA_CPP_SHA = os.environ.get("LLAMA_CPP_SHA", "af50f2f248b475d9760f8e67503d01c0f194f5f4")
# What setup builds: the tools, and the library llama-perplexity is built on (stages/horizon.py
# runs the tool through it so that its log, the parity KLD in it, is flushed before it exits).
LLAMA_BINS = ("llama-server", "llama-quantize", "llama-imatrix", "llama-perplexity", "llama-tokenize",
              "libllama-perplexity-impl.so")
CODEBENCH = NVME / "codebench"
SUITES = ("livecodebench", "codecontests", "apps")
TASKSPEC = REPO / "examples" / "python-utils.code.taskspec.json"
CHOICE = W / "choice.json"
# Both venvs resolve as PyPI stood at this moment (uv's exclude-newer, after vllm 0.31.0's
# upload), so every session's VM installs the same versions; from the second setup on they
# also install with the first setup's freeze as constraints (pinned_venvs). The record's
# verdict compares the readings' venv versions exactly (PROVENANCE_BLOCKING).
UV_EXCLUDE_NEWER = "2026-10-12T00:00:00Z"
VENVS = ("venv-vllm", "venv-train")

# ---------- the goal, the budget and the gates ----------

GOAL_PTS_PER_GB = 40
P1_GB, BUDGET_MAX_GB, BUDGET_KEPT = 1.172, 1.25, 0.85  # budget = max(P1, min(MAX, KEPT x R0 / 40))
GATE0_MIN_R0 = 52.0
# Gates 1 and 2 report their numbers but never stop the run (Oliver 2026-10-14: a lower
# result per GB is fine; the run goes to the end). The verdict still holds it against 40.
GATE1_MIN_PTS_PER_GB = 0.0
GATE2_MIN_PTS_PER_GB = 0.0
Z_ONE_SIDED = 1.645  # one-sided 95%
K_BY_OUTCOME = {"none": 2, "mixed": 8, "all": 4}  # record answers by the original's 0/4, mixed, 4/4
KEEP_KL_DROP, KEEP_SE = 0.05, 2.0
ROUNDS_MAX, ROUNDS_STOP_AFTER = 4, 2
ONPOLICY_SESSIONS = ROUNDS_MAX + 1  # qat onpolicy sessions: each trains a round; the last only judges
PARITY = {"size_rel": 0.005, "kld": 0.01, "top1": 0.98}
LLAMA_CHECK_SE = 2.0
BOOT_DRAWS = 4000

# ---------- the record harness (section 5), frozen after S1.2 ----------

LCB_TOKENS, BUDGETS, LCB_MAX_CASES = 16384, [8192], 50
# Answers in flight in every reading (vLLM's --max-num-seqs: Config.vllm_seqs, and
# LOBBOT_VLLM_SEQS), per GPU. All of a run's readings use one value: the reference
# reading is given its GPU's, and every later reading reads it back from the reference's
# config.json (record_seqs()), so the original's reading and the record agree by
# construction; settings_diff checks it with the rest of the harness (vllm_seqs).
#   L40S  96: past what its KV cache holds on average at 16k thinking; the rest only wait
#         inside vLLM.
#   B200 192, from its KV budget. Granite 4.2 3B's KV cache is 40 layers x 8 KV heads x
#         64 x 2 (K, V) x 2 bytes = 80 KiB a token in bf16. vLLM gets READ_GPU_UTIL of the
#         B200's 179 GiB (LOBBOT_VLLM_GPU_UTIL) = 161 GiB; less the bf16 weights (6.8 GiB) and
#         ~4 GiB of activations and CUDA graphs, ~150 GiB of KV cache: ~1.97M tokens. An
#         answer holds up to 24,576 (eval's --max-model-len: 16,384 thinking + 4,096 forced
#         answer + 4,096 for the prompt), so 80 fit at their longest; but a live answer
#         holds ~8k on average (run 2's answers averaged ~9.6k tokens, and the long ones
#         hold their slots longest: prompt + E[L^2] / 2E[L]), so ~245 fit on average. 192
#         fill ~78% of the cache on average (1.54M tokens): room for long answers to bunch
#         up, and for a quantized model whose loops hit the cap more (~10k a live answer:
#         1.92M), without heavy preemption; 256 would fill it on average and preempt all
#         along. And 192 keep the B200 busy: each decode step reads every live answer's KV
#         (~0.6 GiB at 8k) and the weights once, so past ~12 answers the steps are bound by
#         reading the KV cache, which 192 answers keep streaming.
RECORD_SEQS = {"L40S": 96, "B200": 192}
READ_GPU_UTIL = 0.9  # vLLM's --gpu-memory-utilization in every reading
# The record harness, less vllm_seqs (per GPU: harness_settings() adds the run's). The
# readings' keys hash EVAL, so the reference reading's key does not move with the GPU a
# later session runs on.
EVAL = {
    "teacher": MODEL, "code_eval_suites": ["livecodebench"], "code_eval_ref": "",
    "code_eval_thinking": True, "code_eval_thinking_temperature": 1.0, "thinking_top_k": 0, "eval_min_p": 0.0,
    "code_eval_max_tokens": LCB_TOKENS, "code_eval_budgets": BUDGETS, "lcb_max_cases": LCB_MAX_CASES,
    "eval_kv_type": "", "eval_vllm_args": [], "eval_reasoning_parser": "qwen3",
}
# Keys of config.effective.json that differ between readings by design.
PER_READING = {"eval_candidates", "code_eval_ids", "code_eval_samples", "code_eval_samples_by_id",
               "code_eval_seed_offset", "size_gb_override", "eval_model_fingerprint"}
# The record harness in config.effective.json: EVAL's keys and every key in these
# namespaces (new eval knobs follow them), less PER_READING. A difference there, or a key
# one side lacks, blocks the verdict; any other Config field (a pipeline knob added
# between sessions, say) is reported only.
HARNESS_PREFIXES = ("code_eval_", "eval_", "lcb_", "vllm_", "thinking_")
# provenance.json fields that must match the original's reading; the rest is reported.
# Both venvs are frozen across sessions (UV_EXCLUDE_NEWER, the first setup's freeze), so
# a version difference here is real drift; code_suites is the suites' fetch records.
PROVENANCE_BLOCKING = ("gpu", "vllm_venv", "eval_venv", "llama_cpp", "sandbox_workers", "nproc", "code_suites")
DEV_PROBLEMS = 128
SMOKE_PROBLEMS, SMOKE_TOKENS = 8, 4096

# ---------- what each step makes ----------

GEN = {"measure": {"tokens": 8e6}, "train": {"tokens": 24e6}}
GEN_SEQS = {"L40S": 160, "B200": 384}  # ternary.SEQS: gen's --max-num-seqs on the GPU it runs on (fp8 KV)
GEN_SAMPLES, GEN_PORT, SMOKE_PORT = 2, 8093, 8094
IMATRIX_TOKENS = 1_500_000
BLOCK = {"windows": 2048, "length": 2048, "epochs": 2, "lr": 1e-4}
E2E = {"tokens": 40e6, "length": 16384, "max_minutes": 60}
ROLLOUT_TOKENS = 8000  # mean tokens of an on-policy answer, for the time guess
# 2 held-out answers, 16k, 2 epochs: qat's. Each session is capped: its training at
# max_minutes (the LR anneals by time, and the codes freeze before the cap), its fresh
# rollouts at rollout_tokens_cap answer tokens (the guess's mean; a looping model would
# otherwise double them). The held-out set is never cut: the keep rules pair it.
ONPOLICY = {"rounds": ROUNDS_MAX, "fresh": 250, "rollouts": 4, "held": 128, "max_minutes": 25,
            "rollout_tokens_cap": 250 * 4 * ROLLOUT_TOKENS}
KLD = {"ctx": 8192, "chunks": 16}  # the parity KLD windows
# S2.2: a few problems on vLLM, 20 QAT steps at 16k with the codes never frozen (the e2e
# step's own cost), and one tiny on-policy session (its serve dir on vLLM, rollouts,
# labels, the judging and a training round in the trainer's own process).
RATES = {"qat_tokens": 20 * 16384, "vllm_problems": 5,
         "onpolicy": {"rounds": 1, "fresh": 2, "rollouts": 2, "held": 2, "max_minutes": 5}}

# ---------- time guesses (minutes) ----------

# Measured: run 2 read 728 LiveCodeBench answers (16k thinking, vLLM, bf16) on the L40S
# in 3h54m, at ~500 answer tokens/s. Estimate: a B200 reads B200_READ_GAIN times as fast
# (decode is bound by reading the KV cache: ~8 TB/s against the L40S's 864 GB/s, taken
# at well under its 9x; never measured on the B200). Once the original's reference
# reading has ended, every later reading's guess is re-based on the answers a minute it
# measured on the run's one reading GPU (reference_rate()), plus READ_BOOT_MIN for the
# server's start and the sandbox after the answers; everything else is an estimate, and
# the training steps re-base theirs on what they measure (minutes()).
L40S_ANSWERS_PER_MIN, RUN2_TOK_S = 728 / 234, 500
B200_READ_GAIN = 4.0
ANSWERS_PER_MIN = {"L40S": L40S_ANSWERS_PER_MIN, "B200": B200_READ_GAIN * L40S_ANSWERS_PER_MIN}
# llama-server on the L40S, ~100 answers an hour in run 2; elsewhere, and once re-based,
# in that proportion to vLLM's rate
LLAMA_ANSWERS_PER_MIN = 100 / 60
READ_BOOT_MIN = 5
# gen-measure: 8M tokens at run 2's ~500 tok/s (bf16 KV) times FP8_KV_GAIN for gen's fp8
# KV cache, plus GEN_BOOT_MIN for the server; re-based on the original's reading (minutes()).
FP8_KV_GAIN, GEN_BOOT_MIN = 2.0, 5
GUESS = {"setup": 45, "restore": 10, "download": 5, "gguf": 5, "smoke": 30, "gen-measure": 130, "vocab": 30,
         "imatrix": 12, "ptq": 48, "blockap": 60, "upload": 10, "budgets": 12, "rates": 30, "gen-train": 36,
         "e2e": 50, "onpolicy": 4 * 45, "onpolicy-judge": 15, "final": 24, "choice": 2, "rebuild": 10,
         "rescore": 18}
# Estimates for the measure steps when they run on a B200 (session all; none of them ever
# ran there): the GPU work at about the readings' B200_READ_GAIN, the CPU work as on the
# L40S. gen: 8M tokens at run 2's rate x the gain x the fp8 KV gain, plus the server's
# start; smoke: mostly vLLM starts and llama-quantize; vocab: the BPE rewrite (CPU) and
# the gates' held-out forward passes; imatrix: 1.5M tokens through llama-imatrix and the
# trimmed BF16 GGUF; PTQ: three llama-quantize runs and the dequantization cache (CPU),
# then ~320 delta-KL swaps; Block-AP: 2 epochs of 2048 x 2048-token windows, then the
# export; budgets: the 8k re-scoring's forced answers, mostly the server's start.
GUESS_B200 = {"smoke": 25, "gen-measure": GEN["measure"]["tokens"] / (RUN2_TOK_S * B200_READ_GAIN * FP8_KV_GAIN) / 60
              + GEN_BOOT_MIN, "vocab": 20, "imatrix": 5, "ptq": 30, "blockap": 20, "budgets": 6}


def minutes(kind: str = "L40S") -> dict[str, float]:
    """GUESS on kind's GPU (GUESS_B200's estimates on a B200), re-based on what the run
    measured: the training text's time from the original's reference reading (its answer
    tokens/s, on the GPU gen runs on too, times the fp8 KV gain); the training steps' from
    QAT tokens/s in S2.2's 20 steps and the B200's answer tokens/s in the training text's
    gen, with each on-policy session's caps. The readings' guesses: reading_minutes()."""
    m = {**GUESS, **(GUESS_B200 if kind == "B200" else {})}
    ref = reading_dir(READINGS["original-ref"])
    cand = ((load(ref / "out" / "eval.json") if ref else None) or {}).get("candidates") or [{}]
    tok_s = (cand[0].get("code") or {}).get("tok_s_total")
    if tok_s:
        m["gen-measure"] = GEN["measure"]["tokens"] / (tok_s * FP8_KV_GAIN) / 60 + GEN_BOOT_MIN
    r = load(rates_dir() / "rates.json") if rates_dir() else None
    c = load(traces_dir("train") / "traces.count") if traces_dir("train") else None
    qat, roll = (r or {}).get("qat_tok_s"), (c or {}).get("tok_s")
    if qat:
        m["e2e"] = min(E2E["tokens"] / qat / 60, E2E["max_minutes"]) + 5
    if qat and roll:
        a = ONPOLICY
        # qat holds the fresh rollouts to the cap within about one problem (it asks lazily,
        # each problem in flight reserved at the mean answer so far), so a model that loops
        # costs the cap's tokens, fewer problems, not more tokens
        fresh = min(a["fresh"] * a["rollouts"] * ROLLOUT_TOKENS,
                    (a["rollout_tokens_cap"] or math.inf) + a["rollouts"] * LCB_TOKENS)
        held = a["held"] * 2 * ROLLOUT_TOKENS
        train = min(2 * fresh / qat / 60, a["max_minutes"])
        m["onpolicy"] = a["rounds"] * ((fresh + held) / roll / 60 + train + 10)
        m["onpolicy-judge"] = held / roll / 60 + 10
    return m


def reference_rate() -> float | None:
    """Answers a minute the original's reference reading wrote on the run's reading GPU:
    its answer tokens/s (eval.json's tok_s_total, forced answers in) over its mean tokens
    an answer, the same way. None until it has ended."""
    r = READINGS["original-ref"]
    d = reading_dir(r)
    if not (d and (d / ".done" / "eval").exists()):
        return None
    try:
        cand = ((load(d / "out" / "eval.json") or {}).get("candidates") or [{}])[0]
        tok_s = (cand.get("code") or {}).get("tok_s_total")
        rows = rows_of(r)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    per = statistics.fmean((x.get("tokens") or 0) + (x.get("forced_tokens") or 0) for x in rows) if rows else 0
    return 60 * tok_s / per if tok_s and per else None


def read_rate(kind: str) -> tuple[float, bool]:
    """(vLLM answers a minute, measured): the reference reading's once it has ended (every
    later reading runs on its GPU), else kind's ANSWERS_PER_MIN (the B200's an estimate)."""
    got = reference_rate()
    return (got, True) if got else (ANSWERS_PER_MIN[kind], False)


def reading_minutes(r: Reading, kind: str) -> dict[str, float]:
    """A reading's guess: its answers at read_rate (llama-server's in run 2's proportion to
    vLLM's), plus READ_BOOT_MIN once that rate is measured, and the 8k re-scoring."""
    rate, measured_rate = read_rate(kind)
    if r.model == "final-gguf":
        rate *= LLAMA_ANSWERS_PER_MIN / L40S_ANSWERS_PER_MIN
    return {"eval": expected_answers(r) / rate + (READ_BOOT_MIN if measured_rate else 0),
            **({"budgets": minutes(kind)["budgets"]} if r.budgets else {})}


def measured(kind: str) -> str:
    """Where the time guesses on kind's GPU come from."""
    r = load(rates_dir() / "rates.json") if rates_dir() else None
    rate = reference_rate()
    if rate:
        reads = (f"LiveCodeBench readings from the original's reference reading, measured on the "
                 f"{harness_gpu() or 'reading GPU'}: {rate:.1f} answers a minute on vLLM (llama-server in run 2's "
                 "proportion)")
    elif kind == "B200":
        reads = (f"LiveCodeBench readings at {B200_READ_GAIN:g}x the L40S's measured 728 answers in 3h54m, an "
                 "ESTIMATE (never read on a B200) until the original's reference reading measures the rate")
    else:
        reads = ("LiveCodeBench readings from the L40S's measured 728 answers in 3h54m on vLLM (~100 an hour on "
                 "llama-server)")
    return (f"Times on the {kind}: {reads}"
            + ("; gen, vocab trim, imatrix, PTQ and Block-AP are B200 estimates" if kind == "B200" else "")
            + ("; training from the rates S2.2 measured" if r else "; training times are estimates until S2.2 "
               "measures them")
            + "; the rest are estimates.")


# ---------- small helpers ----------

def h(*parts) -> str:
    """A short hash of what an artifact was made from."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def load(p: Path | None) -> dict | list | None:
    try:
        return json.loads(p.read_text()) if p else None
    except (OSError, ValueError):
        return None


def dump(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)


def text(p: Path | None) -> str | None:
    try:
        return p.read_text().strip() if p else None
    except OSError:
        return None


def make_job(d: Path, cfg: dict) -> Path:
    """A pipeline job dir with this config.json; a different config clears its eval."""
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if load(d / "config.json") != cfg:
        for stage in ("eval", "budgets"):
            (d / ".done" / stage).unlink(missing_ok=True)
        dump(d / "config.json", cfg)
    if not (d / "taskspec.json").exists():
        spec = json.loads(TASKSPEC.read_text())
        spec["target"] = {**spec.get("target", {}), "max_size_gb": 10.0}
        dump(d / "taskspec.json", spec)
    return d


# ---------- GPU and CPU ----------

def gpu_free(tries: int = 120, every: int = 5, limit_mib: int = 2000) -> str:
    """Shell that waits until the GPU's memory is back (a stopped vLLM's engine lets go a
    little after the server), and fails, naming the processes that hold it, if it is
    not back after tries x every seconds."""
    used = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1"
    return (f"for i in $(seq {tries}); do m=$({used}); [ -n \"$m\" ] && [ \"$m\" -lt {limit_mib} ] && break; "
            f"sleep {every}; done\n"
            f"m=$({used}); if [ -z \"$m\" ] || [ \"$m\" -ge {limit_mib} ]; then "
            f"echo \"GPU not free after {tries * every}s: ${{m:-no reading of}} MiB in use. Holders:\"; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; fi")


GPU_FREE = gpu_free()


def gpu(cmd: str) -> str:
    return f"set -e\n{GPU_FREE}\n{cmd}"


def gpu_info() -> tuple[str, float] | None:
    """(name, memory GiB) of the first GPU, or None without nvidia-smi or a GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60, check=False).stdout
        name, mib = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mib) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_kind(name: str | None) -> str | None:
    """Which of GPUS a GPU name is, or None."""
    return next((k for k in GPUS if name and k in name), None)


@functools.cache
def this_gpu() -> str | None:
    """This machine's GPU, as one of GPUS (None without a GPU, or with another)."""
    g = gpu_info()
    return gpu_kind(g[0]) if g else None


def machine(session: str) -> str:
    """The GPU a session runs on, for its time guesses, gen's and the readings' answers in
    flight: this machine's, else (--list without a GPU, say) the one it is planned for."""
    return this_gpu() or MACHINE[session]


def machine_problem(session: str) -> str | None:
    """Why this machine cannot run the session, or None. Readings run on an L40S or a B200,
    the GPUs RECORD_SEQS has a value for (HORIZON_ANY_GPU=1 allows another, at its
    session's planned GPU's value); which of them is the one the original's reference
    reading ran on, and run_restore checks that once the reading is back from the bucket
    (gpu_mismatch()). Full-weight QAT needs a B200's memory."""
    g = gpu_info()
    if g is None:
        return ("nvidia-smi found no GPU. On a new VM the driver may still be installing (wait for cloud-init "
                "and its reboot). If none is coming, run `bash scripts/setup_vm.sh`: it installs NVIDIA's "
                "nvidia-open driver (Blackwell needs the open modules), then reboot and re-run.")
    name, gib = g
    if session in ("train", ALL) and gib < 150:
        return (f"session {session} trains every weight and needs a B200 (~180 GB); this GPU is {name} with "
                f"{gib:.0f} GiB")
    if session in READS and gpu_kind(name) is None and os.environ.get("HORIZON_ANY_GPU") != "1":
        return (f"session {session} reads LiveCodeBench on an L40S or a B200 (RECORD_SEQS has their answers in "
                f"flight); this GPU is {name}. HORIZON_ANY_GPU=1 runs it anyway, at the {MACHINE[session]}'s value.")
    return None


def gpu_mismatch() -> str | None:
    """Why this machine must not read: the original's reference reading, which defines the
    harness, ran on another GPU (a reading there would block the verdict). None when the
    names match, when either is unknown, or with HORIZON_ANY_GPU=1."""
    want, here = harness_gpu(), (gpu_info() or (None, 0))[0]
    if not want or not here or want == here or os.environ.get("HORIZON_ANY_GPU") == "1":
        return None
    return (f"the original's reference reading, which defines the record harness, ran on {want}; this GPU is {here}. "
            f"Every reading of a run must be on one GPU (provenance blocks the verdict otherwise): use a {want}, or "
            "start a new run (HORIZON_RUN). HORIZON_ANY_GPU=1 reads anyway.")


def physical_cores() -> int:
    try:
        out = subprocess.run(["lscpu", "-p=core,socket"], capture_output=True, text=True, timeout=30,
                             check=False).stdout
        cores = {l for l in out.splitlines() if l and not l.startswith("#")}
        if cores:
            return len(cores)
    except (OSError, subprocess.SubprocessError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def nvme_env() -> dict:
    """Temp files and the vLLM/torch compile caches on the NVMe, not the 14 GB root disk:
    the sandbox writes each answer's cases.json under TMPDIR, and one problem's cases can
    run to hundreds of MB. .env.vm sets the same for every stage; this covers a VM whose
    .env.vm predates it."""
    out = {"TMPDIR": NVME / "tmp", "XDG_CACHE_HOME": NVME / "cache", "VLLM_CACHE_ROOT": NVME / "cache" / "vllm",
           "TRITON_CACHE_DIR": NVME / "cache" / "triton", "TORCHINDUCTOR_CACHE_DIR": NVME / "cache" / "torchinductor"}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in out.items()}


def read_env(seqs: int) -> dict:
    """Every LiveCodeBench reading: seqs answers in flight (the job's vllm_seqs, its GPU's
    RECORD_SEQS), vLLM on READ_GPU_UTIL of the GPU, and the sandbox on physical cores - 2.
    The base's revision and downloaded commit go into each reading's provenance.json (eval
    keeps the LOBBOT_EVAL_* environment there)."""
    return {**nvme_env(), "LOBBOT_VLLM_SEQS": str(seqs),
            "LOBBOT_SANDBOX_WORKERS": str(max(1, physical_cores() - 2)), "LOBBOT_VLLM_GPU_UTIL": str(READ_GPU_UTIL),
            "LOBBOT_EVAL_BASE_REVISION": MODEL_REVISION, "LOBBOT_EVAL_BASE_COMMIT": download_commit() or ""}


def vllm_script(model: Path, port: int, log: Path, then: str, seqs: int = 32) -> str:
    """Shell that serves model on vLLM in its own process group, waits up to 20 minutes for
    it to be healthy, runs `then`, and stops the whole group however that ends. Started as
    gen's server is (ternary.gen_command: fp8 KV, the vLLM venv's bin first on PATH, no
    FlashInfer JIT, see eval.VLLM_JIT_FREE), so the smoke test tries gen's settings."""
    return "\n".join([
        'PATH="$(dirname "$LOBBOT_VLLM_PY"):$PATH" VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} '
        f"setsid $LOBBOT_VLLM_PY -m vllm.entrypoints.openai.api_server --model {model} --served-model-name student "
        f"--host 127.0.0.1 --port {port} --max-model-len 22528 --max-num-seqs {seqs} --kv-cache-dtype fp8 "
        "${LOBBOT_VLLM_ATTN_BACKEND:+--attention-backend $LOBBOT_VLLM_ATTN_BACKEND} "
        f"--gpu-memory-utilization 0.9 > {log} 2>&1 & srv=$!",
        # set +e: under the caller's set -e, the KILL of a group already gone would end the
        # trap, and so the script, with exit 1 after `then` succeeded.
        "trap 'set +e; kill -TERM -- -$srv 2>/dev/null; for i in $(seq 30); do kill -0 $srv 2>/dev/null || break; "
        "sleep 1; done; kill -KILL -- -$srv 2>/dev/null; true' EXIT",
        f"for i in $(seq 240); do curl -sf http://127.0.0.1:{port}/health >/dev/null && break; "
        f"kill -0 $srv 2>/dev/null || {{ tail -50 {log}; exit 1; }}; sleep 5; done",
        f"curl -sf http://127.0.0.1:{port}/health >/dev/null || {{ echo 'vLLM not healthy after 20 min'; "
        f"tail -50 {log}; exit 1; }}",
        then])


# ---------- the problem sets ----------

def lcb_meta() -> dict | None:
    return load(CODEBENCH / "livecodebench.meta.json")


def window(name: str) -> dict | None:
    return ((lcb_meta() or {}).get("windows") or {}).get(name)


def dev_ids() -> list[str] | None:
    """The dev subset: DEV_PROBLEMS of LiveCodeBench 2024-08-01..2024-12-31, fixed by crc32(id)."""
    w = window("dev")
    return sorted(w["ids"], key=lambda i: (zlib.crc32(i.encode()), i))[:DEV_PROBLEMS] if w else None


def training_sources() -> list | None:
    """What the training text's problems come from: the CodeContests and APPS fetches."""
    cc = load(CODEBENCH / "codecontests.meta.json")
    apps = load(CODEBENCH / "apps.meta.json") or {}
    return [cc.get("sha"), cc.get("ids_sha256"), apps.get("sha"), apps.get("ids_sha256")] if cc else None


# ---------- the hash-keyed artifacts, upstream first ----------

def traces_dir(session: str = "measure") -> Path | None:
    """The original's traces: 8M answer tokens in session measure, extended to 24M (a copy
    of those, resumed) in session train."""
    src = training_sources()
    if src is None:
        return None
    if session == "measure":
        return W / f"traces-{h(MODEL, src, GEN['measure']['tokens'], GEN_SAMPLES, LCB_TOKENS)}"
    sha = traces_sha()
    return W / f"traces24-{h(sha, GEN['train']['tokens'])}" if sha else None


def traces_file(session: str = "measure") -> Path | None:
    d = traces_dir(session)
    return d / "traces.jsonl" if d else None


def traces_sha(session: str = "measure") -> str | None:
    d = traces_dir(session)
    return text(d / "traces.sha256") if d else None


def traces_progress(session: str) -> float | None:
    """Answer tokens written over the target, from gen's traces.count sidecar."""
    d = traces_dir(session)
    c = load(d / "traces.count") if d else None
    return min(0.99, c["tokens"] / c["target"]) if c and c.get("target") else None


@functools.cache
def split_rule() -> dict:
    """The held-out split of the training text (stages/traces.py's HELD_FRAC and
    SPLIT_RULE, read from its source: this script is stdlib only). Every key below the
    traces carries it: vocab's gates, the imatrix's text, horizon's sweep and check
    halves, the Block-AP windows and everything trained after them depend on the split,
    so an artifact made under another split is never reused. The traces themselves do
    not (gen writes every problem's rows)."""
    src = REPO / "stages" / "traces.py"
    got = {t.id: ast.literal_eval(n.value) for n in ast.parse(src.read_text()).body if isinstance(n, ast.Assign)
           for t in n.targets if isinstance(t, ast.Name) and t.id in ("HELD_FRAC", "SPLIT_RULE")}
    if set(got) != {"HELD_FRAC", "SPLIT_RULE"}:
        raise RuntimeError(f"{src} does not set HELD_FRAC and SPLIT_RULE as constants (found {sorted(got)})")
    return {"held_frac": got["HELD_FRAC"], "rule": got["SPLIT_RULE"]}


def vocab_dir() -> Path | None:
    sha = traces_sha()
    return W / f"vocab-{h(sha, split_rule(), MODEL, 'vocab run: the measured set, 65536, then none')}" if sha else None


def vocab() -> dict | None:
    """{"rows", "hf", "index", "tried"}: the checkpoint the trim gates accepted (rows None: no trim)."""
    d = vocab_dir()
    return load(d / "vocab.json") if d else None


def vocab_sha() -> str | None:
    d = vocab_dir()
    return text(d / "vocab.sha256") if d else None


def vocab_hf() -> Path | None:
    v = vocab()
    return Path(v["hf"]) if v else None


def vocab_flag() -> str:
    v = vocab()
    return f" --vocab-index {v['index']}" if v and v.get("index") else ""


def imatrix_dir() -> Path | None:
    v, t = vocab_sha(), traces_sha()
    return W / f"imatrix-{h(v, t, split_rule(), IMATRIX_TOKENS, 'ctx 8192, output head')}" if v and t else None


def imatrix_sha() -> str | None:
    d = imatrix_dir()
    return text(d / "work" / "imatrix.sha256") if d else None


def ptq_dir() -> Path | None:
    v, i, g = vocab_sha(), imatrix_sha(), gate("gate0")
    return W / f"ptq-{h(v, i, g and g.get('budget_gb'), split_rule(), 'horizon run')}" if v and i and g else None


def type_map() -> dict | None:
    """horizon's map.json: {"types": {GGUF tensor: type}, "bytes" (the file, exact), "sha256", ...}."""
    d = ptq_dir()
    return load(d / "map.json") if d else None


def blockap_dir() -> Path | None:
    m, i, t, v = type_map(), imatrix_sha(), traces_sha(), vocab_sha()
    return W / f"blockap-{h(m and m.get('sha256'), i, t, v, split_rule(), BLOCK)}" if m and i and t and v else None


def rates_dir() -> Path | None:
    b = blockap_dir()
    return W / f"rates-{h(b.name, RATES)}" if b else None


def e2e_dir() -> Path | None:
    b, t = blockap_dir(), traces_sha("train")
    return W / f"e2e-{h(b.name, t, E2E)}" if b and t else None


def onpolicy_dir() -> Path | None:
    """qat onpolicy's work dir: its state.json, a dir per round, the kept model's state."""
    e = e2e_dir()
    return W / f"onpolicy-{h(e.name, ONPOLICY, KEEP_KL_DROP, KEEP_SE, ROUNDS_STOP_AFTER)}" if e else None


def final_state() -> Path | None:
    """The state the shipped GGUF is exported from: the last kept model of the rounds."""
    o = onpolicy_dir()
    f = (load(o / "onpolicy.json") or {}).get("final") if o else None
    return Path(f) if f else None


def final_dir() -> Path | None:
    o, f = onpolicy_dir(), final_state()
    return W / f"final-{h(o.name, f.relative_to(o).as_posix() if is_under(f, o) else str(f), PARITY, KLD)}" \
        if o and f else None


def choice() -> dict | None:
    return load(CHOICE)


def final_gguf() -> Path | None:
    c = choice()
    return NVME / c["gguf"] if c else None


def final_hf() -> Path | None:
    """The vLLM checkpoint session read writes from the shipped GGUF's gguf-py dequantization."""
    c = choice()
    return W / f"final-hf-{c['gguf_sha256'][:12]}" if c else None


# ---------- readings ----------

@dataclass(frozen=True)
class Reading:
    name: str
    title: str  # the step's name
    role: str
    window: str  # "test" (2025-01..04, 182 problems) or "dev" (the 128-problem subset)
    model: str  # original, blockap, final (its vLLM checkpoint), final-gguf (llama-server)
    seed: int  # answer i is seeded seed + i
    samples: int  # answers per problem; 0: the K_i file
    ref: str  # the original's reading it is paired with
    budgets: bool = True  # keep the thinking and re-score it at 8k
    gate: str | None = None  # the gate decided right after it


READINGS = {r.name: r for r in [
    Reading("original-ref", "original reference reading, 182 x 4 (Gate 0)", "R0, budget, K_i allocation",
            "test", "original", 0, 4, "original-ref", gate="gate0"),
    Reading("original-dev", "original on the dev subset, 128 x 2", "baseline for the screens",
            "dev", "original", 0, 2, "original-dev"),
    Reading("blockap-dev", "Block-AP dev screen, 128 x 2 (Gate 1)", "Gate 1",
            "dev", "blockap", 50, 2, "original-dev", gate="gate1"),
    Reading("final-dev", "final dev screen, 128 x 2 (Gate 2)", "Gate 2",
            "dev", "final", 50, 2, "original-dev", gate="gate2"),
    Reading("final-record", "record reading, 182 x K_i", "the verdict",
            "test", "final", 1000, 0, "original-ref"),
    Reading("final-llama", "shipped-file check on llama-server, 182 x 1", "gross-breakage check of the GGUF",
            "test", "final-gguf", 1000, 1, "original-ref", budgets=False),
]}


def original_gb() -> float:
    if BF16.exists():
        return BF16.stat().st_size / 1e9
    files = list(HF_DIR.glob("*.safetensors")) if HF_DIR.exists() else []
    return sum(f.stat().st_size for f in files) / 1e9 if files else 7.32


def model_of(kind: str) -> tuple[Path, str, float] | None:
    """(what is served, its fingerprint, shipped GB) of a reading's model, or None until it exists."""
    if kind == "original":
        return HF_DIR, MODEL, original_gb()
    if kind == "blockap":
        b = blockap_dir()
        ex = load(b / "export" / "export.json") if b else None
        sha = text(b / "export" / "model.gguf.sha256") if b else None
        return (b / "export" / "vllm", sha, ex["bytes"] / 1e9) if ex and sha else None
    c = choice()
    if not c:
        return None
    path = final_gguf() if kind == "final-gguf" else final_hf()
    return path, c["gguf_sha256"], c["gguf_bytes"] / 1e9


def k_alloc() -> dict[str, int] | None:
    d = reading_dir(READINGS["original-ref"])
    return load(d / "out" / "k_alloc.json") if d else None


def reading_key(r: Reading) -> str | None:
    """The hash a reading's job dir is named by: the harness (EVAL: its answers in flight,
    the reference reading's by construction, are left out, so a later session finds the
    reference reading whatever GPU it runs on), the problems, seeds, answers and model."""
    meta, win, m = lcb_meta(), window(r.window), model_of(r.model)
    if not (meta and win and m and m[1]):
        return None
    k = None
    if r.samples == 0:
        k = k_alloc()
        if k is None:
            return None
    return h(EVAL, meta.get("sha"), win["ids_sha256"], r.window, r.seed, r.samples, r.budgets, k, m[1], r.model)


def reading_dir(r: Reading) -> Path | None:
    """The reading's job dir; its candidate (and so its jsonl) has the same name."""
    key = reading_key(r)
    return JOBS / f"{r.name}-{key}" if key else None


def record_seqs(kind: str | None = None) -> int | None:
    """Answers in flight in every reading of the run: the value the original's reference
    reading was given, read back from its config.json, so that every later reading has
    it by construction; before that reading exists, kind's RECORD_SEQS (the GPU it will
    be read on), or None without a kind."""
    d = reading_dir(READINGS["original-ref"])
    n = (load(d / "config.json") or {}).get("vllm_seqs") if d else None
    return n or (RECORD_SEQS[kind] if kind else None)


def harness_settings() -> dict:
    """The record harness as the run reads it: EVAL and its answers in flight."""
    return {**EVAL, "vllm_seqs": record_seqs()}


def gpu_names(prov: dict | None) -> list[str]:
    """The GPU names in a provenance.json ("NVIDIA B200, 183359 MiB, 580.95" -> its name)."""
    return [g.split(",")[0].strip() for g in ((prov or {}).get("gpu") or [])]


def reading_gpu(r: Reading) -> str | None:
    """The GPU a reading ran on, from its provenance.json (eval writes it as it starts)."""
    d = reading_dir(r)
    names = gpu_names(load(d / "work" / "provenance.json")) if d else []
    return names[0] if names else None


def harness_gpu() -> str | None:
    """The GPU the record harness is defined on: the one the original's reference reading
    ran on. Every later reading must run on the same GPU name."""
    return reading_gpu(READINGS["original-ref"])


def reading_config(r: Reading, kind: str) -> dict:
    win, (path, fp, gb) = window(r.window), model_of(r.model)
    name = reading_dir(r).name
    # eval keys resumed answers by this content fingerprint, not the files' mtime: a new VM
    # downloads, re-exports or rebuilds the same weights with new mtimes.
    cfg = {**EVAL, "vllm_seqs": record_seqs(kind), "lcb_since": win["since"], "lcb_until": win["until"],
           "code_eval_seed_offset": r.seed,
           "eval_candidates": {name: str(path)},
           "eval_model_fingerprint": original_fingerprint() if r.model == "original" else fp}
    if r.model != "original":
        cfg["size_gb_override"] = {name: round(gb, 6)}  # the shipped GGUF's bytes, not the bf16 checkpoint's
    if r.window == "dev":
        cfg["code_eval_ids"] = dev_ids()
    if r.samples:
        cfg["code_eval_samples"] = r.samples
    else:
        cfg["code_eval_samples"], cfg["code_eval_samples_by_id"] = 4, k_alloc()
    if not r.budgets:
        cfg["code_eval_budgets"] = []
    return cfg


def expected_answers(r: Reading) -> int:
    if r.samples == 0:
        k = k_alloc()
        return sum(k.values()) if k else 880
    return r.samples * (DEV_PROBLEMS if r.window == "dev" else 182)


def reading_done(r: Reading) -> bool:
    d = reading_dir(r)
    return bool(d and (d / ".done" / "eval").exists() and (not r.budgets or (d / ".done" / "budgets").exists())
                and (d / "out" / "thinking.json").exists() and (not r.gate or (d / "out" / f"{r.gate}.json").exists()))


def run_eval(step: Step, d: Path, budgets: bool = True) -> None:
    """The eval stage on d (served on vLLM or llama-server by eval.py), then the 8k
    re-scoring, with the job's own answers in flight (its config's vllm_seqs)."""
    env = read_env(load(d / "config.json")["vllm_seqs"])
    if not (d / ".done" / "eval").exists():
        sh(step, gpu(f"python pipeline.py --job {d} --only eval"), env)
    if budgets and not (d / ".done" / "budgets").exists():
        sh(step, gpu(f"python -m stages.budget --job {d}"), env)


def run_reading(step: Step, r: Reading, kind: str) -> None:
    """A reading on this machine's GPU (kind), which must be the original's reference
    reading's once that has one (gpu_mismatch())."""
    d = reading_dir(r)
    if d is None:
        raise RuntimeError(f"{r.name}: its inputs are not there yet (problem set, model or K_i file)")
    why = gpu_mismatch()
    if why:
        raise RuntimeError(f"{r.name} is not read here: {why}")
    make_job(d, reading_config(r, kind))
    run_eval(step, d, r.budgets)
    ref = reading_dir(READINGS[r.ref])
    srcs = f"{d}" + (f" {ref}" if ref and ref != d and ref.exists() else "")
    sh(step, f"python3 scripts/thinking_report.py {srcs} --ref {ref.name if ref else r.ref} --json "
             f"--out {d}/out/thinking.json > /dev/null")
    if r.gate:
        GATE_WRITERS[r.gate]()


# ---------- what a reading says ----------

def rows_of(r: Reading, suffix: str = "") -> list[dict]:
    """The graded answer rows of a reading (suffix "-allcases": the CPU re-score on every case)."""
    d = reading_dir(r)
    p = d / "work" / "code_eval" / f"{d.name}{suffix}.jsonl" if d else None
    if not p or not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [x for x in rows if x.get("suite", "livecodebench") == "livecodebench"]


def outcomes(rows: list[dict]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for x in rows:
        out.setdefault(x["id"], []).append(bool(x.get("passed")))
    return out


def score(outs: dict[str, list[bool]]) -> float | None:
    """R = 100 x the mean over problems of each problem's pass rate."""
    return 100 * statistics.fmean(sum(v) / len(v) for v in outs.values()) if outs else None


def fixed_se(outs: dict[str, list[bool]]) -> float | None:
    """The reading's SE on its fixed problem set, from the answers' spread within each
    problem: 100 sqrt(sum_i p_i(1-p_i)/(K_i-1)) / N. None when a problem has one answer."""
    if not outs or any(len(v) < 2 for v in outs.values()):
        return None
    var = sum((sum(v) / len(v)) * (1 - sum(v) / len(v)) / (len(v) - 1) for v in outs.values())
    return 100 * math.sqrt(var) / len(outs)


def bootstrap(outs: dict[str, list[bool]], draws: int = BOOT_DRAWS, seed: int = 0) -> tuple[float, float] | None:
    """95% interval of R over problems drawn again with replacement (new problems, not
    just new answers; reported only)."""
    if not outs:
        return None
    rates = [sum(v) / len(v) for v in outs.values()]
    rng = random.Random(seed)
    rs = sorted(100 * statistics.fmean(rng.choices(rates, k=len(rates))) for _ in range(draws))
    return rs[int(0.025 * draws)], rs[int(0.975 * draws) - 1]


def paired(model: dict[str, list[bool]], ref: dict[str, list[bool]]) -> tuple[float, float | None, int] | None:
    """(mean difference in points, its SE, problems) on the problems both answered."""
    ids = sorted(set(model) & set(ref))
    if not ids:
        return None
    d = [sum(model[i]) / len(model[i]) - sum(ref[i]) / len(ref[i]) for i in ids]
    se = 100 * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None
    return 100 * statistics.fmean(d), se, len(ids)


def lower_bound(R: float, se: float | None) -> float | None:
    return R - Z_ONE_SIDED * se if se is not None else None


def shares(rows: list[dict]) -> dict[str, float]:
    """Shares of the answers that errored, timed out on a test case, hit the cap, needed a
    forced answer, looped, or were cut after their thinking closed."""
    if not rows:
        return {}
    n = len(rows)
    tests = {"error": tr.errored, "case_timeout": lambda x: x.get("reason") == "case_timeout",
             "cap": lambda x: x.get("finish") == "length", "forced": lambda x: bool(x.get("forced")),
             "loop": tr.looped, "answer_cut": tr.answer_cut}
    return {k: sum(map(f, rows)) / n for k, f in tests.items()}


def budget_score(r: Reading, b: int = BUDGETS[0]) -> float | None:
    """The reading re-scored at b thinking tokens (stages/budget.py; first answer per problem).
    b = LCB_TOKENS is the long run itself on those same answers (budget.long_run), the
    figure an 8k re-score is paired with; R is over every answer."""
    d = reading_dir(r)
    rep = load(d / "out" / "eval.json") if d else None
    for c in (rep or {}).get("candidates", []):
        s = ((c.get("code") or {}).get("budgets") or {}).get(str(b)) or {}
        if s.get("mean_pass@1") is not None:
            return 100 * s["mean_pass@1"]
    return None


def reading_stats(r: Reading) -> dict | None:
    rows = rows_of(r)
    m = model_of(r.model)
    if not rows or not m:
        return None
    outs = outcomes(rows)
    R, se = score(outs), fixed_se(outs)
    allc = outcomes(rows_of(r, "-allcases"))
    return {"R": R, "se": se, "lb": lower_bound(R, se), "gb": m[2], "problems": len(outs), "answers": len(rows),
            "outs": outs, "boot": bootstrap(outs), "shares": shares(rows), "budget8k": budget_score(r),
            "budget16k": budget_score(r, LCB_TOKENS), "allcases": score(allc) if allc else None}


# ---------- the gates, as rules ----------

def budget_gb(R0: float) -> float:
    return max(P1_GB, min(BUDGET_MAX_GB, BUDGET_KEPT * R0 / GOAL_PTS_PER_GB))


def gate0(R0: float) -> dict:
    return {"gate": "gate0", "R0": round(R0, 4), "pass": R0 >= GATE0_MIN_R0, "budget_gb": round(budget_gb(R0), 6),
            "rule": f"R0 >= {GATE0_MIN_R0}; budget = max({P1_GB}, min({BUDGET_MAX_GB}, {BUDGET_KEPT} x R0 / "
                    f"{GOAL_PTS_PER_GB})) GB"}


def k_allocation(outs: dict[str, list[bool]], answers: int = 4) -> dict[str, int]:
    """Record answers per problem from the original's reference reading: K_BY_OUTCOME by
    whether it went 0/4, mixed or 4/4."""
    bad = [i for i, v in outs.items() if len(v) != answers]
    if bad:
        raise ValueError(f"{len(bad)} problems do not have {answers} answers in the reference reading, e.g. {bad[0]}")
    return {i: K_BY_OUTCOME["none" if not any(v) else "all" if all(v) else "mixed"] for i, v in sorted(outs.items())}


def preflight(gguf_bytes: int, budget: float, R0: float, types_ok: bool = True) -> dict:
    gb = gguf_bytes / 1e9
    checks = {"within the budget": gb <= budget, "40 x size <= R0": GOAL_PTS_PER_GB * gb <= R0,
              "every tensor's type as mapped": bool(types_ok)}
    return {"gate": "preflight", "gb": round(gb, 6), "budget_gb": budget, "R0": R0, "checks": checks,
            "pass": all(checks.values()), "why": ", ".join(k for k, ok in checks.items() if not ok) or None}


def screen_gate(name: str, R0: float, model: dict, orig: dict, gb: float, threshold: float) -> dict:
    """Gate 1 or 2: the reading predicted from a dev screen, R0 x (screen R / original R on
    the same problems), per GB of the screened file, against the threshold."""
    ids = sorted(set(model) & set(orig))
    sub = lambda o: {i: o[i] for i in ids}
    Rm, Ro = score(sub(model)), score(sub(orig))
    retention = Rm / Ro if Ro else 0.0
    pts = R0 * retention / gb
    return {"gate": name, "R0": R0, "screen_R": Rm, "original_R": Ro, "retention": round(retention, 4),
            "gb": round(gb, 6), "predicted_pts_per_gb": round(pts, 3), "threshold": threshold, "problems": len(ids),
            "pass": pts >= threshold, "rule": f"R0 x retention / GB >= {threshold}"}


def round_rule(rules: dict) -> dict[str, bool]:
    """S2.5's keep rule, from the numbers qat onpolicy recorded for a round (against the
    last kept model, paired by held-out problem): on-policy KL drops >= KEEP_KL_DROP,
    loop% and cap% rise by at most KEEP_SE SE, pass@1 falls by at most KEEP_SE SE."""
    kl = rules.get("kl") or {}
    drop = 1 - kl["new"] / kl["kept"] if kl.get("kept") else 0.0
    ok = {"kl": drop >= KEEP_KL_DROP - 1e-9}
    for m in ("loop", "cap"):
        r = rules.get(m) or {}
        ok[m] = r.get("rise") is not None and r["rise"] <= KEEP_SE * (r.get("se") or 0.0) + 1e-9
    r = rules.get("pass") or {}
    ok["pass"] = r.get("diff") is not None and r["diff"] >= -KEEP_SE * (r.get("se") or 0.0) - 1e-9
    ok["problems"] = bool((rules.get("problems") or {}).get("n"))
    return ok


def rounds_over(kept: list[bool]) -> str | None:
    """Why no further round may run: ROUNDS_MAX done, or ROUNDS_STOP_AFTER in a row not kept."""
    if len(kept) >= ROUNDS_MAX:
        return f"{ROUNDS_MAX} rounds done"
    if len(kept) >= ROUNDS_STOP_AFTER and not any(kept[-ROUNDS_STOP_AFTER:]):
        return f"{ROUNDS_STOP_AFTER} rounds in a row not kept"
    return None


def audit_rounds(st: dict, rounds: int = ROUNDS_MAX) -> list[str]:
    """Where the trainer's rounds (qat onpolicy's state.json) break the plan's rules: a
    round kept or dropped against round_rule, or a round after the stop rule fired."""
    bad, kept = [], []
    for r in st.get("rounds") or []:
        why = rounds_over(kept)
        if why:
            bad.append(f"round {r.get('round')} ran after the stop rule ({why})")
        ok = round_rule(r.get("rules") or {})
        if bool(r.get("kept")) != all(ok.values()):
            bad.append(f"round {r.get('round')} {'kept' if r.get('kept') else 'dropped'}, but the rule says "
                       f"{'keep' if all(ok.values()) else 'drop'} ({', '.join(k for k, v in ok.items() if not v) or 'all pass'})")
        kept.append(bool(r.get("kept")))
    if st.get("done") and not (rounds_over(kept) or len(kept) >= rounds):
        bad.append(f"the rounds stopped after {len(kept)} with no stop rule met")
    return bad


def parity(export: dict, map_info: dict, kld: dict, tok: dict) -> dict:
    """S2.6: what is scored is what ships. export: qat export's export.json; map_info:
    horizon's map.json; kld: the shipped file's row of horizon ladder's output; tok:
    vocab tokcheck's report on the shipped file."""
    est = map_info["bytes"]
    size_rel = abs(export["bytes"] - est) / est
    want = {}
    for t in map_info["types"].values():
        if t in ("q2_k", "q3_k", "q4_k"):
            want[t] = want.get(t, 0) + 1
    exact = export.get("exact") or {}
    checks = {"pack bit-exact": bool(exact) and all(exact.values()),
              "every type as mapped": export.get("types") == want,
              f"size within {PARITY['size_rel']:.1%} of the estimate": size_rel <= PARITY["size_rel"],
              f"KLD <= {PARITY['kld']}": kld.get("kld") is not None and kld["kld"] <= PARITY["kld"],
              f"top-1 >= {PARITY['top1']:.0%}": kld.get("top1") is not None and kld["top1"] >= PARITY["top1"],
              "llama-tokenize == HF": tok.get("pass") is True}
    return {"gate": "parity", "checks": checks, "size_rel": size_rel, "kld_mean": kld.get("kld"),
            "top1": kld.get("top1"), "pass": all(checks.values()),
            "why": ", ".join(k for k, ok in checks.items() if not ok) or None}


def gate_path(name: str) -> Path | None:
    if name == "preflight":
        d = ptq_dir()
        return d / "preflight.json" if d else None
    if name == "parity":
        d = final_dir()
        return d / "parity.json" if d else None
    r = next(x for x in READINGS.values() if x.gate == name)
    d = reading_dir(r)
    return d / "out" / f"{name}.json" if d else None


def gate(name: str) -> dict | None:
    return load(gate_path(name))


def stopped(names: tuple[str, ...]) -> str | None:
    """Why the steps behind these gates do not run: the first that was decided and failed."""
    for n in names:
        g = gate(n)
        if g is not None and not g.get("pass"):
            return f"{n} did not pass ({g.get('why') or g.get('rule') or 'see summary'}); the run stops before this step"
    return None


# Every gate a run can stop at, in the order the steps decide them: (name, what, where).
STOPS = (("gate0", "Gate 0", "after the original's reference reading"), ("preflight", "the preflight", "after PTQ"),
         ("gate1", "Gate 1", "after the Block-AP dev screen"), ("parity", "the parity gates", "after the final export"),
         ("gate2", "Gate 2", "after the final dev screen"))


def stop_line() -> str | None:
    """Where and why the run stopped, when a gate failed: the first one, with its numbers."""
    for name, what, where in STOPS:
        g = gate(name)
        if g is None or g.get("pass"):
            continue
        if name == "gate0":
            why = f"R0 {g['R0']:.1f} < {GATE0_MIN_R0}"
        elif name in ("gate1", "gate2"):
            why = (f"R0 {g['R0']:.1f} x retention {g['retention']:.1%} / {g['gb']:.3f} GB = "
                   f"{g['predicted_pts_per_gb']:.1f}/GB < {g['threshold']}")
        else:
            why = g.get("why") or "see below"
        return f"STOPPED at {what}, {where}: {why}. Every step after it is skipped; the upload still runs."
    return None


def write_gate0() -> None:
    r = READINGS["original-ref"]
    outs = outcomes(rows_of(r))
    d = reading_dir(r)
    dump(d / "out" / "k_alloc.json", k_allocation(outs, r.samples))
    dump(d / "out" / "gate0.json", gate0(score(outs)))


def write_screen_gate(name: str, screen: str, threshold: float) -> None:
    g0, m = gate("gate0"), model_of(READINGS[screen].model)
    orig = outcomes(rows_of(READINGS["original-dev"]))
    if not g0 or not orig:
        raise RuntimeError(f"{name} needs Gate 0's R0 and the original's dev reading")
    dump(gate_path(name), screen_gate(name, g0["R0"], outcomes(rows_of(READINGS[screen])), orig, m[2], threshold))


GATE_WRITERS: dict[str, Callable[[], None]] = {
    "gate0": write_gate0,
    "gate1": lambda: write_screen_gate("gate1", "blockap-dev", GATE1_MIN_PTS_PER_GB),
    "gate2": lambda: write_screen_gate("gate2", "final-dev", GATE2_MIN_PTS_PER_GB),
}


# ---------- the bucket ----------

BIG_BYTES = 64 << 20  # files above this go up one by one; the rest in one tar per step


def bucket(*parts: str) -> str:
    return "/".join([BUCKET, "horizon", RUN_ID, *parts])


def evroc_copy(src: str, dst: str) -> bool:
    """One `evroc storage bucket copy` (either way), after refreshing the S3 credentials,
    which last about an hour."""
    if not shutil.which("evroc"):
        print("evroc: CLI not on PATH", file=sys.stderr)
        return False

    def run(*a) -> bool:
        # stdout is dropped (get-s3-credentials may print secrets); stderr's tail is shown.
        r = subprocess.run(["evroc", "storage", "bucket", *a], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, check=False)
        if r.returncode:
            print(f"evroc storage bucket {a[0]} {' '.join(a[1:])} -> exit {r.returncode}: "
                  f"{r.stderr.strip()[-600:]}", file=sys.stderr)
        return r.returncode == 0

    # A read from the bucket doubles as an existence probe, so only uploads are retried.
    tries = 1 if src.startswith("bucket://") else 3
    for k in range(tries):
        if run("get-s3-credentials") and run("copy", "--from", src, "--to", dst):
            return True
        if k + 1 < tries:
            time.sleep(10 * (k + 1))
    return False


def put(src: Path, key: str) -> bool:
    return evroc_copy(str(src), key)


def get(key: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if not evroc_copy(key, str(tmp)) or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dst)
    return True


GET_RETRY_S = 10  # get_tries waits this, then twice this, between its tries


def get_tries(key: str, dst: Path, tries: int = 3) -> bool:
    """get, tried again on a failure: for reads whose answer decides what the run uses
    (evroc answers some reads of a present key with a 500, as it does a missing key)."""
    for k in range(tries):
        if get(key, dst):
            return True
        if k + 1 < tries:
            time.sleep(GET_RETRY_S * (k + 1))
    return False


@dataclass
class Art:
    """A step's outputs: uploaded to horizon/<run>/<slug>/<key>/ and restored from there.
    A session in `sessions` gets all of it back, and so does session all (every step is
    its own); the others only the `light` files (results and hash sidecars, for the
    summary and the keys downstream)."""
    slug: str
    key: Callable[[], str | None]
    paths: Callable[[], list[Path]]
    sessions: tuple[str, ...]
    light: Callable[[], list[Path]] = list
    exclude: tuple[str, ...] = ()  # glob patterns of rebuildable files that are never uploaded
    # The run's pinned copy: uploaded once, complete, with each file's sha256; a disk that
    # holds other files under its key is refused, not marked in sync (restore_art then
    # replaces them with the bucket's).
    once: bool = False
    # A once art's reasons not to pin what is on the disk (a partial fetch, say): while it
    # lists any, upload_art uploads nothing, so a later setup fetches again.
    unfit: Callable[[], list[str]] = list

    def files(self) -> list[Path]:
        out: list[Path] = []
        for p in self.paths():
            if p.is_file():
                out.append(p)
            elif p.is_dir():
                out += sorted(f for f in p.rglob("*") if f.is_file())
        skip = ("*.part", "*.tmp", *self.exclude)
        return [f for f in out if is_under(f, NVME) and not any(f.match(x) or any(q.match(x) for q in f.parents)
                                                                 for x in skip)]

    def is_light(self, f: Path) -> bool:
        return any(f == p or p in f.parents for p in self.light())


def is_under(f: Path, root: Path) -> bool:
    try:
        f.relative_to(root)
        return True
    except ValueError:
        return False


def fingerprint(files: list[Path]) -> str:
    return h([(f.relative_to(NVME).as_posix(), f.stat().st_size, f.stat().st_mtime_ns) for f in files])


def marker(art: Art) -> Path:
    return W / ".uploads" / f"{art.slug}.json"


def in_sync(art: Art) -> bool:
    """The art's files on disk are the ones last uploaded (or restored) under its key."""
    m, key = load(marker(art)), art.key()
    files = art.files()
    return bool(m and key and files and m.get("key") == key and m.get("fingerprint") == fingerprint(files)
                and m.get("complete", True))


def upload_art(art: Art, complete: bool = True) -> bool:
    """Uploads what the art has on disk: big files one by one, the rest as small.tar, then
    the manifest last (its presence means the rest is there). False when there is nothing,
    and for a once art that is not complete or is unfit (a partial pin would be restored
    as the pin).
    A once art the bucket already holds is marked in sync only when every file's sha256
    is the manifest's; other files under its key raise."""
    key, files = art.key(), art.files()
    if key is None or not files or (art.once and (not complete or art.unfit())):
        return False
    fp = fingerprint(files)
    m = load(marker(art))
    if m and m.get("key") == key and m.get("fingerprint") == fp and (m.get("complete", True) or not complete):
        return True
    base = bucket(art.slug, key)
    tmp = W / ".uploads" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if art.once and get(f"{base}/manifest.json", tmp / f"{art.slug}.manifest.json"):
        man = json.loads((tmp / f"{art.slug}.manifest.json").read_text())
        pinned = {f["path"]: f.get("sha256") for f in man["files"]}
        here = {f.relative_to(NVME).as_posix(): sha256_file(f) for f in files}
        if here != pinned:
            differ = sorted(p for p in set(here) | set(pinned) if here.get(p) != pinned.get(p))
            raise RuntimeError(f"{art.slug}: the files on this disk are not the ones {base} pins ({', '.join(differ)}); "
                               "the restore step gets the bucket's copy")
        dump(marker(art), {"key": key, "fingerprint": fp, "complete": True, "note": "the bucket's copy"})
        return True
    recs, small = [], []
    for f in files:
        rel, size = f.relative_to(NVME).as_posix(), f.stat().st_size
        recs.append({"path": rel, "size": size, "big": size >= BIG_BYTES, "light": art.is_light(f),
                     **({"sha256": sha256_file(f)} if art.once else {})})
        if size < BIG_BYTES:
            small.append((f, rel))
        elif not put(f, f"{base}/f/{rel}"):
            raise RuntimeError(f"upload of {rel} to {base} failed")
    if small:
        tar = tmp / f"{art.slug}.tar"
        with tarfile.open(tar, "w") as t:
            for f, rel in small:
                t.add(f, arcname=rel)
        ok = put(tar, f"{base}/small.tar")
        tar.unlink(missing_ok=True)
        if not ok:
            raise RuntimeError(f"upload of {art.slug}'s small files to {base} failed")
    man = tmp / f"{art.slug}.manifest.json"
    dump(man, {"slug": art.slug, "key": key, "run": RUN_ID, "complete": complete, "files": recs,
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not put(man, f"{base}/manifest.json"):
        raise RuntimeError(f"upload of {art.slug}'s manifest to {base} failed")
    dump(marker(art), {"key": key, "fingerprint": fp, "complete": complete})
    return True


def restore_art(art: Art, session: str) -> str | None:
    """Gets the art back from the bucket under its current key: all of it in its own
    sessions (session all's are every art), its light files elsewhere. None when there
    is nothing to get."""
    key = art.key()
    if key is None or in_sync(art):
        return None
    base, tmp = bucket(art.slug, key), W / ".uploads" / "tmp"
    man_path = tmp / f"{art.slug}.manifest.json"
    if not get(f"{base}/manifest.json", man_path):
        return None
    man = json.loads(man_path.read_text())
    full = session in art.sessions or session == ALL
    want = [f for f in man["files"] if full or f.get("light")]
    for f in (f for f in want if f["big"]):  # the big files first, so the small ones (results, metas) land last
        dst = NVME / f["path"]
        same = dst.exists() and dst.stat().st_size == f["size"] and (not f.get("sha256") or sha256_file(dst) == f["sha256"])
        if not same and not get(f"{base}/f/{f['path']}", dst):
            raise RuntimeError(f"restore of {f['path']} from {base} failed")
    small = {f["path"] for f in want if not f["big"]}
    if small:
        tar = tmp / f"{art.slug}.tar"
        if not get(f"{base}/small.tar", tar):
            raise RuntimeError(f"restore of {art.slug}'s small files from {base} failed")
        with tarfile.open(tar) as t:
            members = [m for m in t.getmembers() if m.name in small]
            try:
                t.extractall(NVME, members=members, filter="data")
            except TypeError:  # Python before 3.12
                t.extractall(NVME, members=members)
        tar.unlink(missing_ok=True)
    if full and man.get("complete", True) and art.files():
        dump(marker(art), {"key": key, "fingerprint": fingerprint(art.files()), "complete": True})
    return "all" if full else "results"


def wait_for_login(minutes: float) -> bool:
    """Polls for an evroc login for up to `minutes`, saying how to log in."""
    deadline, said = time.time() + minutes * 60, 0.0
    while not evroc_ok():
        if time.time() >= deadline:
            return False
        if time.time() - said >= 600:
            above(f"    waiting for an evroc login to upload (up to {minutes:.0f} min): evroc login --oidc-port 8765")
            said = time.time()
        time.sleep(60)
    return True


# ---------- the steps ----------

STEP_ARTS: dict[tuple[str, str], Art | None] = {}  # (session, step name): what its step makes
SHUTDOWN = False
LOGIN_WAIT_MINUTES = float(os.environ.get("HORIZON_LOGIN_WAIT_MIN", "180"))  # with --shutdown


def step(session: str, name: str, mins: dict[str, float] | Callable[[], dict[str, float]],
         run: Callable[[Step], None], done: Callable[[], bool], art: Art | None = None, gates: tuple[str, ...] = (),
         progress: Callable[[], float | None] | None = None) -> Step:
    """A runner Step that re-checks done() (a restore can finish it: runner's Done, so its
    ~0 s leave the speed alone), skips when a gate before it failed, and whose outputs
    `art` the upload hook sends to the bucket. mins as a callable is a guess the run
    re-bases (minutes()): the runner reads it again when a step starts."""
    def safe_done() -> bool:
        try:
            return bool(done())
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def go(s: Step) -> None:
        if safe_done():
            raise Done("done before it started: a restore or an earlier step finished it")
        why = stopped(gates)
        if why:
            raise Skip(why)
        run(s)

    def guess() -> dict[str, float]:
        return {k: round(v, 1) for k, v in (mins() if callable(mins) else mins).items()}

    STEP_ARTS[(session, name)] = art
    return Step(name, guess(), go, safe_done, progress or (lambda: None), refresh=guess if callable(mins) else None)


def run_setup(step: Step) -> None:
    """S1.1/S2.1/S3.1: setup_vm.sh. The run's pinned inputs come from the bucket when it
    has them: the code suites (session 1's fetch; setup_vm.sh then skips its own, and the
    bucket's copy replaces anything on the disk once the NVMe is mounted) and the first
    setup's freeze of both venvs (installed as constraints). Only the first setup of a run
    fetches the suites and freezes the venvs, and uploads them (the setup step's art, and
    pin_venvs); a first fetch that is not whole fails the step instead, and is never
    pinned. Their bucket copies are looked up into a temp dir on the root disk: on a new
    VM the NVMe is mounted only by setup_vm.sh."""
    pins = Path(tempfile.mkdtemp(prefix="event-horizon-setup-"))
    try:
        cons = pins / "venvs"
        have_venvs = pinned_venvs(cons)
        have_bench = codebench_pinned(pins)
        sh(step, f"SKIP_EXTRAS=1 NVME={NVME} TEACHER={MODEL} TEACHER_REVISION={MODEL_REVISION} STUDENT= "
                 f"LLAMA_CPP_SHA={LLAMA_CPP_SHA} UV_EXCLUDE_NEWER={UV_EXCLUDE_NEWER} "
                 f"SKIP_SUITES={'1' if have_bench else ''} CONSTRAINTS={cons if have_venvs else ''} "
                 "bash scripts/setup_vm.sh")
    finally:
        shutil.rmtree(pins, ignore_errors=True)
    if not have_venvs:
        pin_venvs()
    if have_bench:
        restore_codebench()
        return
    # The first fetch is the one pinned, so it must be whole: setup_vm.sh only warns when a
    # suite fails to download, and a pin without one could never be repaired (every later
    # setup restores the pin and skips the fetch). The step fails instead, nothing is
    # pinned (upload_art refuses an unfit pin), and a re-run fetches the missing suites.
    bad = codebench_problems()
    if bad:
        raise RuntimeError("the code suites this setup fetched are not whole, so nothing is pinned: "
                           + "; ".join(bad) + ". Re-run: setup_vm.sh fetches every suite whose meta is missing "
                           f"(for a JSONL that does not match its meta, delete both from {CODEBENCH} first).")


def codebench_pinned(tmp: Path) -> bool:
    """The bucket holds the run's code suites, uploaded complete."""
    man = tmp / "codebench.manifest.json"
    return bool(evroc_ok() and get_tries(bucket("codebench", "pinned", "manifest.json"), man)
                and (load(man) or {}).get("complete", True))


def restore_codebench() -> None:
    """The bucket's code suites onto the NVMe (they replace whatever is there), checked
    file by file against the manifest's sha256 and each JSONL against its meta's."""
    art = codebench_art()
    restore_art(art, SESSIONS[0])
    bad = ([] if in_sync(art) else ["the files on the disk are not the bucket's"]) + codebench_problems()
    if bad:
        raise RuntimeError("the code suites restored from the bucket do not check out: " + "; ".join(bad))


def codebench_problems() -> list[str]:
    """Why the code suites on the disk are not whole: a suite of SUITES (setup_vm.sh's own
    list with SKIP_EXTRAS: the record's problems, the training text's CodeContests and
    APPS) without its meta or its JSONL, or a JSONL whose sha256 is not the one its meta
    recorded when it was fetched."""
    out = []
    for s in SUITES:
        meta, f = load(CODEBENCH / f"{s}.meta.json"), CODEBENCH / f"{s}.jsonl"
        if meta is None:
            out.append(f"no {s}.meta.json")
        elif not f.exists():
            out.append(f"{s}.jsonl missing")
        elif meta.get("file_sha256") and sha256_file(f) != meta["file_sha256"]:
            out.append(f"{s}.jsonl is not the file its meta records (sha256)")
    return out


def venvs_dir() -> Path:
    """setup_vm.sh's freeze of each venv: <venv>.txt, one name==version per line."""
    return NVME / "venvs"


def pinned_venvs(dst: Path) -> bool:
    """The run's frozen venvs (the first setup's freeze) from the bucket into dst."""
    if not evroc_ok():
        return False
    return all([get(bucket("venvs", f"{v}.txt"), dst / f"{v}.txt") for v in VENVS])


def pin_venvs() -> bool:
    """Uploads this VM's freeze of both venvs as the run's pins, unless the bucket has one
    (the first setup's wins). VENVS[0] goes last, so its presence means all are there."""
    files = [venvs_dir() / f"{v}.txt" for v in VENVS]
    if not evroc_ok() or not all(f.exists() for f in files):
        return False
    if get(bucket("venvs", files[0].name), W / ".uploads" / "tmp" / f"pinned-{files[0].name}"):
        return False
    for f in reversed(files):
        if not put(f, bucket("venvs", f.name)):
            raise RuntimeError(f"upload of {f} to {bucket('venvs')} failed")
    above(f"    pinned this VM's venvs for the later sessions: {bucket('venvs')}")
    return True


def llama_cpp_head() -> str | None:
    return text(LLAMA_CPP / ".git" / "HEAD")


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and llama_cpp_head() == LLAMA_CPP_SHA
            and all((LLAMA_CPP / "build" / "bin" / b).exists() for b in LLAMA_BINS)
            and all((CODEBENCH / f"{s}.meta.json").exists() for s in SUITES))


def run_restore(step: Step, session: str) -> None:
    """Whatever the bucket holds under the current hashes, for this session and those
    before (session all: everything), then what the session needs checked."""
    if not evroc_ok():
        if session in ("measure", ALL):  # the first session's steps: there is nothing before them
            above("    no evroc login: nothing restored. Log in (evroc login) before the uploads.")
            return
        raise RuntimeError(f"no evroc login: session {session} needs the earlier sessions' results from the bucket. "
                           "Log in (evroc login --oidc-port 8765) and re-run.")
    got = []
    for art in arts_through(session):
        how = restore_art(art, session)
        if how:
            got.append(f"{art.slug} ({how})")
    above("    restored: " + (", ".join(got) if got else "nothing"))
    # Every reading must score the run's one copy of the suites: when the bucket pins one
    # and the disk does not hold it (a flaky read in run_setup or above kept this VM's own
    # fetch, 2026-10-14), it is restored again, or the step fails.
    if not in_sync(codebench_art()) and codebench_pinned(W / ".uploads" / "tmp"):
        restore_codebench()
        above("    restored: the run's pinned code suites over this VM's own fetch")
    bad = codebench_problems()
    if bad:
        raise RuntimeError("the code suites on the NVMe are not whole: " + "; ".join(bad))
    missing = session_inputs(session)
    if missing:
        raise RuntimeError(f"session {session} cannot start: " + "; ".join(missing))
    register(step)


def session_inputs(session: str) -> list[str]:
    """What a session needs from the ones before it, or why it must not run: a session
    that reads must run on the GPU the original's reference reading ran on, when that
    reading is back (gpu_mismatch())."""
    out = []
    why = gpu_mismatch() if session in READS else None
    if why:
        out.append(why)
    if session == "train":
        for g in ("gate0", "preflight", "gate1"):
            s = gate(g)
            if s is None:
                out.append(f"{g} has not been decided (run session measure)")
            elif not s.get("pass"):
                out.append(f"{g} did not pass, so the B200 is not used")
        b, v = blockap_dir(), vocab_hf()
        if not (b and (b / "state").exists()):
            out.append("the Block-AP state")
        if not (v and v.exists()):
            out.append("the trimmed checkpoint")
        t = traces_file()
        if not (t and t.exists()):
            out.append("the training text")
    if session == "read":
        c, g = choice(), final_gguf()
        if not c:
            out.append("choice.json (session train writes it)")
        elif not (g and g.exists()):
            out.append(f"the pre-registered GGUF {c['gguf']}")
        elif not g.with_name("model-hf").exists():
            out.append(f"the checkpoint the GGUF was built from ({g.with_name('model-hf')})")
        if not reading_done(READINGS["original-ref"]):
            out.append("the original's reference reading")
    return out


DOWNLOAD_RESTARTS = 3  # a download that stops without DOWNLOADS_DONE is started again this often


def downloaded() -> bool:
    log = NVME / "download.log"
    return log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace") and (HF_DIR / "config.json").exists()


def download_done() -> bool:
    return downloaded() and not base_problems()


def download_command() -> str:
    """setup_vm.sh's download of the base, in the background, appending to download.log
    (sh() sources .env.vm, so HF_HOME and UV_CACHE_DIR are the NVMe's; uv lives in
    ~/.local/bin, which a shell opened before the setup may not have on PATH)."""
    return (f"export PATH=$HOME/.local/bin:$PATH; setsid nohup bash -c 'uv tool run --from huggingface_hub hf download "
            f"{MODEL} --revision {MODEL_REVISION} --local-dir {HF_DIR} && echo DOWNLOADS_DONE' "
            f">> {NVME / 'download.log'} 2>&1 < /dev/null &")


def run_download(step: Step) -> None:
    """Waits for setup_vm.sh's background download, starting it again (up to
    DOWNLOAD_RESTARTS times) when it stopped without finishing: the setup step that
    started it is done, so a re-run alone would never restart it. Then the base must be
    the architecture every stage assumes (base_problems), before any GPU step."""
    restarts = 0
    while not downloaded():
        if subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL, check=False).returncode != 0:
            time.sleep(10)
            if downloaded():
                break
            if restarts >= DOWNLOAD_RESTARTS:
                raise RuntimeError(f"the model download stopped {restarts + 1} times; see {NVME / 'download.log'}. "
                                   f"Start it by hand (source .env.vm; {download_command()}), then re-run")
            restarts += 1
            above(f"    the model download stopped without finishing; starting it again ({restarts} of "
                  f"{DOWNLOAD_RESTARTS}, log: {NVME / 'download.log'})")
            sh(step, download_command())
        time.sleep(5)
    bad = base_problems()
    if bad:
        raise RuntimeError(f"{HF_DIR} is not the base every stage assumes (GraniteForCausalLM, llama-style): "
                           + "; ".join(bad))


# The downloaded base's shape. qheal's and gguf_build's tensor maps, the converter's scalar
# head_count_kv and qat's mlp.gate_proj names all assume a llama-style GraniteForCausalLM;
# a hybrid-style checkpoint (layer_types, other projections) would break the export or skip
# tensors silently, so it is refused right after the download, in milliseconds.
BASE_CONFIG = {"architectures": ["GraniteForCausalLM"], "model_type": "granite", "num_hidden_layers": 40,
               "hidden_size": 2560, "intermediate_size": 8192, "num_attention_heads": 40, "num_key_value_heads": 8,
               "vocab_size": 100352, "tie_word_embeddings": False}
BASE_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def safetensors_shapes(p: Path) -> dict[str, list[int]]:
    """{tensor: shape} from a safetensors file's header (8-byte length, then JSON)."""
    with open(p, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    return {k: v.get("shape", []) for k, v in head.items() if k != "__metadata__"}


def base_problems(hf_dir: Path = HF_DIR, want: dict | None = None) -> list[str]:
    """Why hf_dir is not the base the stages assume, or []: its config.json against
    BASE_CONFIG and no layer_types; its weights (the safetensors index, or the one file)
    hold every layer's BASE_LINEARS and lm_head, and no other 2-D weight in a layer."""
    want = BASE_CONFIG if want is None else want
    try:
        cfg = json.loads((hf_dir / "config.json").read_text())
    except (OSError, ValueError) as e:
        return [f"config.json unreadable ({e})"]
    bad = [f"{k} is {cfg.get(k)!r}, expected {v!r}" for k, v in want.items() if cfg.get(k) != v]
    if "layer_types" in cfg:
        bad.append(f"config.json has layer_types ({sorted(set(map(str, cfg['layer_types'])))}): a hybrid-style config")
    index = hf_dir / "model.safetensors.index.json"
    try:
        files = (sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists()
                 else ["model.safetensors"])
        shapes = {k: s for f in files for k, s in safetensors_shapes(hf_dir / f).items()}
    except (OSError, ValueError, KeyError) as e:
        return [*bad, f"the weights' headers are unreadable ({e})"]
    n = cfg.get("num_hidden_layers") if isinstance(cfg.get("num_hidden_layers"), int) else 0
    expect = {f"model.layers.{i}.{s}.weight" for i in range(n) for s in BASE_LINEARS}
    miss = sorted(expect - set(shapes))
    if miss:
        bad.append(f"{len(miss)} expected projection weights missing, e.g. {miss[:2]}")
    if "lm_head.weight" not in shapes:
        bad.append("no lm_head.weight (an untied head is expected)")
    other = sorted(k for k, s in shapes.items() if k.startswith("model.layers.") and len(s) == 2 and k not in expect)
    if other:
        bad.append(f"{len(other)} 2-D layer weights the stages do not map, e.g. {other[:3]}")
    return bad


def download_commit() -> str | None:
    """The commit the base was downloaded at: `hf download --local-dir` records it as the
    first line of each file's .cache/huggingface/download/<file>.metadata."""
    line = (text(HF_DIR / ".cache" / "huggingface" / "download" / "config.json.metadata") or "").split("\n")[0]
    return line if re.fullmatch(r"[0-9a-f]{40}", line) else None


def original_fingerprint() -> str:
    """The original's eval fingerprint (Config.eval_model_fingerprint): its id, the commit
    it was downloaded at and its weight files' sizes, so its partial answers resume on a
    new VM's fresh download of the same weights (eval otherwise keys them by mtime)."""
    files = sorted(HF_DIR.glob("*.safetensors")) if HF_DIR.exists() else []
    return f"{MODEL}@{download_commit() or MODEL_REVISION}:{h([(f.name, f.stat().st_size) for f in files])}"


def download_progress() -> float:
    got = sum(f.stat().st_size for f in HF_DIR.rglob("*") if f.is_file()) if HF_DIR.exists() else 0
    return min(0.99, got / 7.3e9)


def base_job() -> Path:
    return make_job(W / "base", {"teacher": MODEL})


def run_bf16(step: Step) -> None:
    sh(step, f"python -m stages.dense gguf --job {base_job()} --hf {HF_DIR} --bf16 {BF16}")


# S1.2: every check must pass, or the run stops here.

def smoke_dir() -> Path:
    """Keyed by the vLLM and training stacks too (venv freezes, attention backend, sampler),
    so a pass restored from the bucket is not reused after either changes."""
    venvs = [(NVME / "venvs" / f).read_text() if (NVME / "venvs" / f).exists() else "" for f in
             ("venv-vllm.txt", "venv-train.txt")]
    env = [os.environ.get(k, "") for k in ("LOBBOT_VLLM_ATTN_BACKEND", "VLLM_USE_FLASHINFER_SAMPLER")]
    return W / f"smoke-{h(MODEL, LLAMA_CPP_SHA, EVAL, SMOKE_PROBLEMS, SMOKE_TOKENS, *venvs, *env)}"


def smoke_done() -> bool:
    return bool((load(smoke_dir() / "smoke.json") or {}).get("ok"))


def smoke_eval_job(n: int, max_tokens: int, name: str, kind: str) -> Path | None:
    """A few dev problems x 1 through the whole eval path (vLLM, streaming, the sandbox),
    with kind's answers in flight."""
    ids, win = dev_ids(), window("dev")
    if not ids:
        return None
    d = JOBS / f"{name}-{h(EVAL, ids[:n], max_tokens)}"
    return make_job(d, {**EVAL, "vllm_seqs": RECORD_SEQS[kind], "lcb_since": win["since"], "lcb_until": win["until"],
                        "code_eval_ids": ids[:n], "code_eval_samples": 1, "code_eval_max_tokens": max_tokens,
                        "code_eval_budgets": [], "eval_candidates": {d.name: str(HF_DIR)}})


def check_eval_rows(d: Path, n: int) -> list[str]:
    """(c): every answer came back, none errored, and its thinking was parsed off."""
    p = d / "work" / "code_eval" / f"{d.name}.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    bad = [] if len(rows) == n else [f"{len(rows)} answers, not {n}"]
    bad += [f"{x['id']}: errored" for x in rows if tr.errored(x)]
    bad += [f"{x['id']}: no reasoning parsed" for x in rows if not x.get("reasoning_chars")]
    return bad


def run_smoke(step: Step, kind: str) -> None:
    S = smoke_dir()
    S.mkdir(parents=True, exist_ok=True)
    res = load(S / "smoke.json") or {}
    py = "python scripts/event_horizon.py smoke"
    url = f"http://127.0.0.1:{SMOKE_PORT}"

    def check(label: str, what: str, fn: Callable[[], dict | None]) -> None:
        if (res.get(label) or {}).get("ok"):
            return
        above(f"    smoke ({label}) {what}")
        found = fn() or {}
        res[label] = {"ok": True, "what": what, "time": time.strftime("%F %T"), **found}
        dump(S / "smoke.json", res)

    def template() -> dict:
        sh(step, f"{py} template --hf {HF_DIR} --out {S}/template.json")
        t = load(S / "template.json") or {}
        if t.get("default_system_text"):
            above(f"    smoke (b): with no system message the template adds {t['default_system_text'][:200]!r}; "
                  "every prompt the run sends has codebench.SYSTEM, so it never reaches the model")
        return {k: t.get(k) for k in ("default_system_text", "accepted") if t.get(k)}

    def eval_path() -> None:
        d = smoke_eval_job(SMOKE_PROBLEMS, SMOKE_TOKENS, "smoke-eval", kind)
        if d is None:
            raise RuntimeError("no LiveCodeBench dev window in livecodebench.meta.json")
        run_eval(step, d, budgets=False)
        bad = check_eval_rows(d, SMOKE_PROBLEMS)
        if bad:
            raise RuntimeError("smoke (c) failed: " + "; ".join(bad[:8]))

    def pack_and_block() -> None:
        q2, q4 = S / "q2_k.gguf", S / "q4_k.gguf"
        sh(step, f"set -e\nfor t in Q2_K Q4_K; do f={S}/$(echo $t | tr A-Z a-z).gguf; [ -s $f ] || "
                 f"{{ {LLAMA_CPP}/build/bin/llama-quantize --pure {BF16} $f.part $t && mv $f.part $f; }}; done")
        sh(step, f"{py} pack --gguf {q2} {q4}")
        sh(step, f"{py} map --gguf {q2} --out {S}/map.json")
        sh(step, gpu(f"python -m stages.qat block --base {HF_DIR} --map {S}/map.json --gguf {q2} {q4} "
                     f"--traces {S}/traces.jsonl --llama-cpp {LLAMA_CPP} --windows 8 --length 512 --epochs 1 "
                     f"--lr {BLOCK['lr']} --out {S}/block"))

    def dummy_trim() -> None:
        sh(step, f"mkdir -p {S}/vocab && python -m stages.vocab smoke --hf {HF_DIR} --llama-cpp {LLAMA_CPP} "
                 f"--out {S}/vocab > {S}/vocab-smoke.json")

    def kld() -> None:
        sh(step, f"{py} text --bench {CODEBENCH} --out {S}/kld.txt")
        sh(step, gpu(f"python -m stages.horizon ladder --bf16 {BF16} --gguf {S}/q4_k.gguf --text {S}/kld.txt "
                     f"--llama-cpp {LLAMA_CPP} --ctx 512 --chunks 4 --out {S}/ladder.json"))

    def bucket_probe() -> None:
        probe, back = S / "probe.bin", S / "probe.back"
        probe.write_bytes(os.urandom(1024))
        back.unlink(missing_ok=True)
        key = bucket("smoke", S.name, "probe.bin")
        # get tries once (a read doubles as an existence probe); this object is there, and
        # evroc has answered reads with a 500 now and then, so the read back gets 3 tries.
        ok = put(probe, key)
        for k in range(3 if ok else 0):
            if get(key, back):
                break
            time.sleep(10 * (k + 1))
        if not (ok and back.exists() and back.read_bytes() == probe.read_bytes()):
            raise RuntimeError("smoke (h): a 1 KB upload to the bucket and back failed (evroc login?)")

    # (b) first: if the template adds text next to our system message, or opens the answer
    # differently without one, the prompt is fixed before any reading.
    check("b", "chat template with and without codebench.SYSTEM", template)
    check("a+d", "render == vLLM /tokenize; gen on 16 prompts: int prompts, n rows each",
          lambda: sh(step, gpu(vllm_script(HF_DIR, SMOKE_PORT, S / "vllm.log",
                                           f"{py} render --hf {HF_DIR} --url {url} --bench {CODEBENCH} --n 16 "
                                           f"--traces {S}/traces.jsonl"))))
    check("c", f"{SMOKE_PROBLEMS} LiveCodeBench problems x 1 at {SMOKE_TOKENS} through the eval", eval_path)
    check("e", "pack round trip on llama-quantize Q2_K/Q4_K; a short Block-AP from their codes", pack_and_block)
    check("f", "dummy vocab trim: llama-tokenize == HF, special tokens kept", dummy_trim)
    check("g", "llama-perplexity --kl-divergence, parsed as the parity check parses it", kld)
    check("h", "1 KB bucket upload and back", bucket_probe)
    dump(S / "smoke.json", {**res, "ok": True})


def run_gen(step: Step, session: str, kind: str) -> None:
    """The original's thinking on clean CodeContests (+ ~20% APPS functional) problems:
    ternary.gen_command's server (fp8 KV, its own process group, kind's GEN_SEQS) and gen,
    which writes traces.count as it goes and validates the file before it says done.
    session: which traces (measure's 8M, or train's 24M from them)."""
    T, tokens, seqs = traces_dir(session), GEN[session]["tokens"], GEN_SEQS[kind]
    T.mkdir(parents=True, exist_ok=True)
    out = T / "traces.jsonl"
    if session == "train" and not out.exists():  # the 8M from session measure, extended
        tmp = out.with_name("traces.jsonl.part")
        shutil.copyfile(traces_file("measure"), tmp)
        os.replace(tmp, out)
    script = (f"from pathlib import Path\nfrom stages.ternary import gen_command\n"
              f"print(gen_command({str(HF_DIR)!r}, {str(CODEBENCH)!r}, Path({str(out)!r}), {tokens!r}, "
              f"port={GEN_PORT}, seqs={seqs}, samples={GEN_SAMPLES}))")
    sh(step, gpu(f"python - > {T}/gen.sh <<'EOF'\n{script}\nEOF\nbash {T}/gen.sh"))
    c = load(T / "traces.count")
    if not (c and c.get("done")):
        raise RuntimeError(f"gen ended, but {T}/traces.count does not say done: {c}")
    (T / "traces.sha256").write_text(sha256_file(out) + "\n")
    dump(T / ".gen-done", {"tokens": c.get("tokens"), "target": tokens, "time": time.strftime("%F %T")})


def gen_done(session: str) -> bool:
    T = traces_dir(session)
    return bool(T and (T / ".gen-done").exists() and (T / "traces.sha256").exists())


def register(step: Step) -> None:
    """The trimmed tokenizer's hash in the pinned converter (vocab register): the NVMe's
    llama.cpp is new each session, and convert_hf_to_gguf refuses the trimmed tokenizer
    without it. Run after the restore and after the trim. A no-op without a trim, or
    in a session without the trimmed checkpoint (session read converts nothing)."""
    v = vocab()
    if v and v.get("rows") and Path(v["hf"]).exists():
        sh(step, f"python -m stages.vocab register --llama-cpp {LLAMA_CPP} --hf {v['hf']}")


def run_vocab(step: Step) -> None:
    """S1.6 (stages/vocab.py run): trim to the measured set (the ids the traces, statements
    and probe keep; the plan's estimate is ~49,152 rows) if the gates pass on the
    original's held-out thinking, else 65,536, else no trim (its exit code 3). Nothing
    downstream assumes a size: horizon's byte table comes from the trimmed (or untrimmed)
    files, and the preflight from its exact bytes. The vocab's hash in the names
    downstream is the trimmed tokenizer's sha256."""
    V, T = vocab_dir(), traces_file()
    V.mkdir(parents=True, exist_ok=True)
    sh(step, gpu(f"python -m stages.vocab run --hf {HF_DIR} --out {V}/hf --traces {T} --codebench {CODEBENCH} "
                 f"--llama-cpp {LLAMA_CPP} || [ $? -eq 3 ]"))
    rep = load(V / "hf.report.json")
    if rep is None:
        raise RuntimeError(f"vocab run wrote no report ({V}/hf.report.json)")
    index = V / "hf" / "vocab_index.json" if rep.get("trim") else None
    tried = [{"rows": a.get("size"), "pass": bool(a.get("pass")),
              "dropped_mean": ((a.get("model") or {}).get("dropped_mass") or {}).get("mean"),
              "dropped_p999": ((a.get("model") or {}).get("dropped_mass") or {}).get("p999"),
              "nll_per_byte_rise": ((a.get("model") or {}).get("nll_per_byte") or {}).get("rise"),
              "llama_tokenize": (a.get("llama_tokenize") or {}).get("pass")} for a in rep.get("attempts", [])]
    sha = ((load(index) or {}).get("tokenizer_sha256") or sha256_file(index)) if index else "none"
    (V / "vocab.sha256").write_text(sha + "\n")
    dump(V / "vocab.json", {"rows": rep.get("rows") if index else None, "hf": str(V / "hf" if index else HF_DIR),
                            "index": str(index) if index else None, "tried": tried})
    register(step)


def vocab_done() -> bool:
    V = vocab_dir()
    return bool(V and (V / "vocab.json").exists() and (V / "vocab.sha256").exists())


def run_imatrix(step: Step) -> None:
    """S1.7: on the trimmed BF16 from ~1.5M training-split trace tokens, in 8192-token
    chunks with the output head; Event Horizon's own job dir."""
    d = make_job(imatrix_dir(), {"teacher": MODEL})
    bf = trimmed_bf16(step)
    sh(step, f"python -m stages.dense calib --job {d} --traces {traces_file()} --tokenizer {HF_DIR} "
             f"--tokens {IMATRIX_TOKENS}")
    sh(step, gpu(f"python -m stages.dense imatrix --job {d} --bf16 {bf}"))
    (d / "work" / "imatrix.sha256").write_text(sha256_file(d / "work" / "imatrix.gguf") + "\n")


def imatrix_done() -> bool:
    d = imatrix_dir()
    return bool(d and (d / "work" / "imatrix.sha256").exists())


def trimmed_bf16(step: Step) -> Path:
    """The trimmed checkpoint's BF16 GGUF (made again after a restore: it is not uploaded)."""
    d = make_job(imatrix_dir(), {"teacher": MODEL})
    bf = d / "model-bf16.gguf"
    if not bf.exists():
        sh(step, f"python -m stages.dense gguf --job {d} --hf {vocab_hf()} --bf16 {bf}")
    return bf


def run_ptq(step: Step) -> None:
    """S1.8 (stages/horizon.py run): PTQ GGUFs per type, the delta-KL sweep, the knapsack
    up to the budget, the chosen map verified; then the preflight. When even the
    cheapest map at the kept vocab is over the budget (horizon's exit 3, floor.json),
    nothing is swept and the preflight fails on that floor, which stops the run at the
    gate (the upload still runs)."""
    P, d, g0, v = ptq_dir(), imatrix_dir(), gate("gate0"), vocab()
    bf = trimmed_bf16(step)
    teacher = f" --teacher {HF_DIR} --vocab-index {v['index']}" if v.get("index") else ""
    sh(step, gpu(f"python -m stages.horizon run --hf {vocab_hf()} --bf16 {bf} --traces {traces_file()} "
                 f"--llama-cpp {LLAMA_CPP} --imatrix {d}/work/imatrix.gguf --budget-gb {g0['budget_gb']:.6f} "
                 f"--out {P}{teacher} || [ $? -eq 3 ]"))
    hz, floor = load(P / "horizon.json"), load(P / "floor.json")
    if floor and floor.get("over"):
        pf = {**preflight(floor["floor_bytes"], g0["budget_gb"], g0["R0"]), "floor": True,
              "note": "the cheapest map at the kept vocab; nothing was swept"}
    elif hz:
        # horizon's check of the verified map against its prediction (gain_check): a map no
        # better than the floor was replaced by the floor (fallback); a short gain is a warning
        pf = {**preflight(hz["bytes"], g0["budget_gb"], g0["R0"], (P / "map.json").exists()),
              "gain": hz.get("gain"), "fallback": hz.get("fallback"), "warnings": hz.get("warnings") or []}
    else:
        raise RuntimeError(f"horizon run wrote neither {P}/horizon.json nor {P}/floor.json")
    dump(P / "preflight.json", pf)
    for line in ptq_lines(pf):
        above(f"    {line}")
    above(f"    preflight: {pf['gb']:.3f} GB against a budget of {pf['budget_gb']:.3f} GB and R0 {pf['R0']:.1f}: "
          + ("pass" if pf["pass"] else f"fail ({pf['why']})"))


def ptq_lines(pf: dict) -> list[str]:
    """What horizon's gain check said of the map (preflight.json's copy of horizon.json's
    gain, fallback and warnings): a fallback and each warning as a "!" line of their own."""
    out, g = [], pf.get("gain") or {}
    if pf.get("fallback"):
        rej = pf["fallback"].get("rejected") or {}
        out.append(f"!!! PTQ MAP FELL BACK TO THE FLOOR: {pf['fallback'].get('why')}. The floor map ships "
                   f"(map.json's fallback has the rejected {rej.get('bytes', 0) / 1e9:.3f} GB map).")
    out += [f"! PTQ map warning: {w}" for w in pf.get("warnings") or []]
    if g:
        out.append(f"PTQ map gain over the floor (KL, sweep windows): {g.get('real')} measured, {g.get('pred')} "
                   f"predicted, SE {g.get('se')}: {g.get('verdict')}")
    return out


def ptq_done() -> bool:
    P = ptq_dir()
    pf = load(P / "preflight.json") if P else None
    return bool(pf and ((P / "map.json").exists() or pf.get("floor")))


def run_export(step: Step, state: Path, out: Path) -> dict:
    """qat export: the packed GGUF (held against the trainer bit for bit, every type as
    mapped, its size within 0.5% of the map's), the vLLM checkpoint written from its
    gguf-py dequantization; its sha256 beside it."""
    sh(step, gpu(f"python -m stages.qat export --state {state} --out {out} --llama-cpp {LLAMA_CPP} "
                 f"--expect-gb {type_map()['bytes'] / 1e9:.6f}"))
    ex = load(out / "export.json")
    exact = (ex or {}).get("exact") or {}
    if not exact or not all(exact.values()):
        raise RuntimeError(f"the export in {out} is not the trainer's weights bit for bit")
    (out / "model.gguf.sha256").write_text(sha256_file(out / "model.gguf") + "\n")
    return ex


def run_blockap(step: Step) -> None:
    """S1.9: block-wise reconstruction from the BF16 weights and llama-quantize's codes
    (the PTQ files); embedding and head stay at their PTQ codes. A restored state is
    only exported again (its vLLM checkpoint is not uploaded), to the same file."""
    B, P = blockap_dir(), ptq_dir()
    if not (B / "block.json").exists():
        sh(step, gpu(f"python -m stages.qat block --base {vocab_hf()} --map {P}/map.json --gguf {P}/ptq/*.gguf "
                     f"--traces {traces_file()}{vocab_flag()} --llama-cpp {LLAMA_CPP} --windows {BLOCK['windows']} "
                     f"--length {BLOCK['length']} --epochs {BLOCK['epochs']} --lr {BLOCK['lr']} --out {B}"))
    for w in (load(B / "block.json") or {}).get("warnings") or []:  # e.g. fewer windows than asked, and why
        above(f"    ! Block-AP warning: {w}")
    sha = text(B / "export" / "model.gguf.sha256")
    run_export(step, B / "state", B / "export")
    if sha and text(B / "export" / "model.gguf.sha256") != sha:
        raise RuntimeError(f"the export of the restored Block-AP state is not the file it was read as ({sha[:12]})")


def blockap_done() -> bool:
    """Exported, with the vLLM checkpoint its dev screen serves (unless that is read)."""
    B = blockap_dir()
    return bool(B and (B / "export" / "model.gguf.sha256").exists()
                and ((B / "export" / "vllm" / "config.json").exists() or reading_done(READINGS["blockap-dev"])))


def onpolicy_flags(a: dict) -> str:
    """qat onpolicy's run settings (read when its work dir is new): counts and the caps."""
    return (f" --rounds {a['rounds']} --fresh {a['fresh']} --rollouts {a['rollouts']} --held {a['held']}"
            f" --max-minutes {a['max_minutes']} --rollout-tokens-cap {a.get('rollout_tokens_cap', 0):.0f}")


def run_rates(step: Step, kind: str) -> None:
    """S2.2: vLLM on the B200 (a few dev problems x 1 at 16k, streaming, the sandbox); 20
    QAT steps at 16k windows with the codes never frozen (so the step costs what the e2e
    step costs; fit would not freeze a 20-step run at the default anyway, --freeze-steps 0
    says so whatever the step count), timed from step 1 on (qat's tok_s: step 0 allocates
    the optimizer) and with their peak memory; then the plan's on-policy steps: one
    tiny qat onpolicy session from those steps' state in a throwaway work dir, which
    writes the serve dir and serves it on vLLM (fp8 KV at 22.5k, as the rounds do), rolls
    out, labels, judges the start model and trains a round in the trainer's own process.
    So nothing of S2.5 first runs after the e2e step. The end-to-end and on-policy
    guesses are re-based on the QAT tokens/s and on the B200 gen's answer tokens/s
    (minutes())."""
    R, B = rates_dir(), blockap_dir()
    d = smoke_eval_job(RATES["vllm_problems"], LCB_TOKENS, "b200-smoke", kind)
    if d is None:
        raise RuntimeError("no LiveCodeBench dev window in livecodebench.meta.json")
    run_eval(step, d, budgets=False)
    bad = check_eval_rows(d, RATES["vllm_problems"])
    if bad:
        raise RuntimeError("vLLM smoke on the B200 failed: " + "; ".join(bad))
    sh(step, gpu(f"python -m stages.qat e2e --state {B}/state --teacher {HF_DIR} --traces {traces_file()}"
                 f"{vocab_flag()} --tokens {RATES['qat_tokens']} --length {E2E['length']} --eval-every 1000000 "
                 f"--held-rows 1 --held-length 2048 --freeze-steps 0 --out {R}/qat"))
    O = R / "onpolicy"
    sh(step, gpu(f"python -m stages.qat onpolicy --work {O} --state {R}/qat/state --teacher {HF_DIR} "
                 f"--problems {CODEBENCH} --traces {traces_file()}{vocab_flag()}{onpolicy_flags(RATES['onpolicy'])} "
                 "--until-session 1"))
    st = load(O / "state.json") or {}
    train = load(O / "round1" / "train" / "train.json")
    if not (st.get("trained") and train):
        raise RuntimeError(f"the on-policy smoke trained no round: {O}/state.json says {st}")
    rolls = [json.loads(x) for p in ("held.labelled.jsonl", "rollouts.labelled.jsonl")
             for x in ((O / "round1" / p).read_text().splitlines() if (O / "round1" / p).exists() else []) if x.strip()]
    q = load(R / "qat" / "e2e.json") or {}
    ev = ((load(d / "out" / "eval.json") or {}).get("candidates") or [{}])[0].get("code") or {}
    dump(R / "rates.json", {"qat_tok_s": q.get("tok_s"), "qat_steps": q.get("steps"), "qat_minutes": q.get("minutes"),
                            "qat_peak_gb": q.get("peak_gb"),
                            "vllm_smoke_tok_s": ev.get("tok_s_total"), "gpu": (gpu_info() or ("?",))[0],
                            "onpolicy": {"rollouts": len(rolls), "steps": train.get("steps"),
                                         "minutes": train.get("minutes"), "peak_gb": train.get("peak_gb"),
                                         "longest_row": max((len(x.get("prompt") or []) + len(x.get("tokens") or [])
                                                             for x in rolls), default=None)}})


def rates_done() -> bool:
    R = rates_dir()
    return bool(R and (R / "rates.json").exists())


def run_e2e(step: Step) -> None:
    """S2.4: every weight in its grid, KL over the kept vocab to the bf16 original on
    trace rows up to 16k from their prompt, LR by progress, a time cap."""
    sh(step, gpu(f"python -m stages.qat e2e --state {blockap_dir()}/state --teacher {HF_DIR} "
                 f"--traces {traces_file('train')}{vocab_flag()} --tokens {E2E['tokens']:.0f} "
                 f"--length {E2E['length']} --max-minutes {E2E['max_minutes']} --out {e2e_dir()}"))


def e2e_done() -> bool:
    E = e2e_dir()
    return bool(E and (E / "e2e.json").exists() and (E / "state").exists())


def run_onpolicy(step: Step, session: int) -> None:
    """S2.5, one session of qat onpolicy (it resumes from its state.json): the model to
    judge served alone, its held-out and fresh rollouts, their labels, then the judging
    against the last kept model on the same held-out problems and the next round's
    training. One step per session, so the upload hook sends each round to the bucket as
    it ends (a session is ~45 min of B200 time, bounded by ONPOLICY's caps). Once the
    rounds are done, every decision is checked against the plan's rule (audit_rounds)
    before anything ships."""
    O, a = onpolicy_dir(), ONPOLICY
    sh(step, gpu(f"python -m stages.qat onpolicy --work {O} --state {e2e_dir()}/state --teacher {HF_DIR} "
                 f"--problems {CODEBENCH} --traces {traces_file('train')}{vocab_flag()}{onpolicy_flags(a)} "
                 f"--until-session {session}"))
    st = load(O / "state.json") or {}
    if not st.get("done"):
        if session < ONPOLICY_SESSIONS and st.get("session", 0) > session:
            return  # the next session's step goes on
        raise RuntimeError(f"qat onpolicy left session {session} unfinished: {O}/state.json says "
                           f"session {st.get('session')}, done {st.get('done')}")
    bad = audit_rounds(st, a["rounds"])
    dump(O / "audit.json", {"ok": not bad, "problems": bad, "final": st.get("final"),
                            "rounds": [{"round": r.get("round"), "kept": r.get("kept"),
                                        "rule": round_rule(r.get("rules") or {})} for r in st.get("rounds") or []]})
    if bad:
        raise RuntimeError("the on-policy rounds break the plan's rules: " + "; ".join(bad))


def onpolicy_done() -> bool:
    O = onpolicy_dir()
    return bool(O and (O / "onpolicy.json").exists() and (load(O / "audit.json") or {}).get("ok"))


def onpolicy_session_done(session: int) -> bool:
    """Session `session` of the rounds has run (or the rounds ended before it, audited)."""
    O = onpolicy_dir()
    st = load(O / "state.json") if O else None
    if not st:
        return False
    return onpolicy_done() if st.get("done") else st.get("session", 0) > session


def run_final(step: Step) -> None:
    """S2.6: the shipped GGUF from the last kept model, and the parity gates on it."""
    F, P = final_dir(), ptq_dir()
    ex = run_export(step, final_state(), F)
    if not (F / "dequant-bf16.gguf").exists():  # the same weights in BF16, for llama.cpp's KLD
        # the converter must know a trimmed tokenizer's hash (vocab_index.json travels with the checkpoint)
        sh(step, f"python -m stages.vocab register --llama-cpp {LLAMA_CPP} --hf {F}/vllm")
        sh(step, f"python {LLAMA_CPP}/convert_hf_to_gguf.py {F}/vllm --outtype bf16 --outfile {F}/dequant-bf16.part "
                 f"&& mv {F}/dequant-bf16.part {F}/dequant-bf16.gguf")
    sh(step, gpu(f"python -m stages.horizon ladder --bf16 {F}/dequant-bf16.gguf --gguf {F}/model.gguf "
                 f"--text {P}/ladder.txt --llama-cpp {LLAMA_CPP} --ctx {KLD['ctx']} --chunks {KLD['chunks']} "
                 f"--out {F}/kld.json"))
    sh(step, f"python -m stages.vocab tokcheck --hf {vocab_hf()} --gguf {F}/model.gguf --llama-cpp {LLAMA_CPP} "
             f"--traces {traces_file('train')} --codebench {CODEBENCH} > {F}/tokenize.json || true")
    kld = next((r for r in load(F / "kld.json") or [] if Path(r.get("gguf", "")).name == "model.gguf"), {})
    par = parity(ex, type_map(), kld, load(F / "tokenize.json") or {})
    dump(F / "parity.json", par)
    above(f"    parity: {'pass' if par['pass'] else 'fail (' + par['why'] + ')'}")


def final_done() -> bool:
    F = final_dir()
    return bool(F and (F / "parity.json").exists())


VERDICT_RULE = (f"R = 100 mean_i p_i; SE = 100 sqrt(sum_i p_i(1-p_i)/(K_i-1)) / N; GB = shipped bytes / 1e9. "
                f"Reached: (R - {Z_ONE_SIDED} SE)/GB >= {GOAL_PTS_PER_GB} and every check passes; point estimate "
                f"only: R/GB >= {GOAL_PTS_PER_GB} with the lower bound below; not reached: otherwise. One look, "
                "no extension; added answers are never pooled into this reading.")


def run_choice(step: Step) -> None:
    """S2.7: pre-registration, before any reading of the final model. Never rewritten."""
    F, g0, k = final_dir(), gate("gate0"), k_alloc()
    k_file = reading_dir(READINGS["original-ref"]) / "out" / "k_alloc.json"
    gguf = F / "model.gguf"
    sha = text(F / "model.gguf.sha256")
    c = {"gguf": gguf.relative_to(NVME).as_posix(), "gguf_sha256": sha, "gguf_bytes": gguf.stat().st_size,
         "gb": gguf.stat().st_size / 1e9, "record": {"seed_offset": READINGS["final-record"].seed,
                                                     "k_file": k_file.relative_to(NVME).as_posix(),
                                                     "k_sha256": sha256_file(k_file), "answers": sum(k.values()),
                                                     "k_rule": K_BY_OUTCOME},
         "goal_pts_per_gb": GOAL_PTS_PER_GB, "gate2_pts_per_gb": GATE2_MIN_PTS_PER_GB, "verdict_rule": VERDICT_RULE,
         "R0": g0["R0"], "budget_gb": g0["budget_gb"], "test_ids_sha256": window("test")["ids_sha256"],
         "harness": harness_settings(), "harness_sha": h(harness_settings()), "gpu": harness_gpu(), "run": RUN_ID,
         "base": {"model": MODEL, "revision": MODEL_REVISION, "commit": download_commit()},
         "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # The bucket's choice (one per run, whatever the hashes upstream) is checked before
    # anything is written, and a new one is uploaded at once: a second VM, or a re-run
    # after a change upstream of final_dir, cannot pre-register another GGUF.
    if not evroc_ok() and not wait_for_login(LOGIN_WAIT_MINUTES if SHUTDOWN else 0):
        raise RuntimeError("choice.json is checked against the bucket and uploaded before anything reads the model: "
                           "log in (evroc login --oidc-port 8765) and re-run")
    art = choice_art()
    restore_art(art, "train")
    old = choice()
    if old and old.get("gguf_sha256") != sha:
        raise RuntimeError(f"{CHOICE} (from {bucket('choice')}) already pre-registers {old['gguf']}; "
                           "a new choice is a new run (HORIZON_RUN)")
    if not old:
        dump(CHOICE, c)
    upload_art(art)


def choice_done() -> bool:
    F = final_dir()
    c = choice()
    return bool(c and F and c.get("gguf_sha256") == text(F / "model.gguf.sha256"))


def run_rebuild(step: Step) -> None:
    """S3.1: the vLLM checkpoint written again from the pre-registered GGUF's gguf-py
    dequantization, after checking the GGUF is the one choice.json names."""
    c, g, out = choice(), final_gguf(), final_hf()
    if sha256_file(g) != c["gguf_sha256"]:
        raise RuntimeError(f"{g} is not the GGUF choice.json pre-registers (sha256 differs)")
    sh(step, f"python -m stages.gguf_build vllm --gguf {g} --hf {g.with_name('model-hf')} --out {out} "
             f"--llama-cpp {LLAMA_CPP}")
    (out / "gguf.sha256").write_text(c["gguf_sha256"] + "\n")


def rebuild_done() -> bool:
    out, c = final_hf(), choice()
    return bool(out and c and text(out / "gguf.sha256") == c["gguf_sha256"])


def run_rescore(step: Step) -> None:
    """S3.5: the original's and the record's answers graded again on the CPU with every
    test case (the reading itself uses the first LCB_MAX_CASES)."""
    for name in ("original-ref", "final-record"):
        d = reading_dir(READINGS[name])
        if d and (d / ".done" / "eval").exists():
            sh(step, f"python scripts/event_horizon.py rescore --job {d} --name {d.name} --max-cases 0",
               {**nvme_env(), "LOBBOT_SANDBOX_WORKERS": str(max(1, physical_cores() - 2))})


def rescore_files() -> list[Path]:
    out = []
    for name in ("original-ref", "final-record"):
        d = reading_dir(READINGS[name])
        if d:
            out.append(d / "work" / "code_eval" / f"{d.name}-allcases.jsonl")
    return out


def rescore_done() -> bool:
    r = READINGS["final-record"]
    want = [p for p, n in zip(rescore_files(), ("original-ref", "final-record"), strict=False)
            if n == "original-ref" or (reading_dir(r) and (reading_dir(r) / ".done" / "eval").exists())]
    return bool(want) and all(p.exists() for p in want)


# ---------- the upload step and the shutdown ----------

def summary_file() -> Path:
    return W / "summary.txt"


def write_summary() -> Path:
    p, s = summary_file(), summary() + "\n"
    if text(p) != s.strip():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    return p


def session_arts(session: str) -> list[Art]:
    """What the session's steps make, each art once (the on-policy sessions share one)."""
    steps(session)
    out: list[Art] = []
    for (s, _), a in STEP_ARTS.items():
        if s == session and a is not None and a.slug not in {x.slug for x in out}:
            out.append(a)
    return out


def arts_through(session: str) -> list[Art]:
    """The arts of this session and the ones before it (session all's own are all)."""
    if session == ALL:
        return session_arts(ALL)
    out: list[Art] = []
    for s in SESSIONS[: SESSIONS.index(session) + 1]:
        out += [a for a in session_arts(s) if a.slug not in {x.slug for x in out}]
    return out


def session_fingerprint(session: str) -> str:
    return h(sorted((a.slug, a.key(), fingerprint(a.files())) for a in session_arts(session) if a.key() and a.files()))


def uploaded(session: str) -> bool:
    """Everything this session made is in the bucket, as it is on the disk now."""
    m = (load(W / ".uploaded") or {}).get(session)
    return bool(m and m.get("fingerprint") == session_fingerprint(session))


def run_upload(step: Step, session: str) -> None:
    """S1.11 / S2.7 / S3.5 (session all: once, at its end): every result of the session,
    the summary and the logs; then W/.uploaded with their fingerprint. Waits a bounded
    time for a login with --shutdown."""
    if not evroc_ok() and not wait_for_login(LOGIN_WAIT_MINUTES if SHUTDOWN else 0):
        raise Skip("no evroc login. Log in (evroc login --oidc-port 8765) and re-run to upload.")
    for art in session_arts(session):
        upload_art(art)
    pin_venvs()  # when the setup had no login to pin them
    upload_extras(session)
    marks = load(W / ".uploaded") or {}
    marks[session] = {"fingerprint": session_fingerprint(session), "time": time.strftime("%F %T")}
    dump(W / ".uploaded", marks)


def upload_extras(session: str, step: Step | None = None) -> None:
    """The summary and the session's logs (small, so every time)."""
    if not put(write_summary(), bucket(f"summary-{session}.txt")):
        raise RuntimeError("upload of the summary failed")
    logs = [step.log] if step is not None and getattr(step, "log", None) else sorted((LOGS / session).glob("*.log"))
    logs.append(NVME / "versions.txt")  # what this VM's setup installed (its venvs' freeze, nvcc, llama.cpp)
    for log in logs:
        if log and log.exists():
            put(log, bucket("logs", session, log.name))


def make_hook(session: str) -> Callable[[Step | None, bool], None]:
    """runner.run_plan's after_step: after each finished step, its outputs, the summary
    and its log; at the end, however the run ended, the failed step's partial outputs
    (eval and gen resume from them) and anything of the session not yet uploaded. It
    never waits for a login; a failure is reported by the runner and retried next call."""
    def after_step(s: Step | None, final: bool) -> None:
        if not evroc_ok():
            raise RuntimeError("no evroc login, nothing uploaded; the last step waits for one "
                               "(evroc login --oidc-port 8765)")
        if s is not None:
            art = STEP_ARTS.get((session, s.name))
            if art is not None:
                upload_art(art, complete=s.state == "done")
        if final:
            for art in session_arts(session):
                if not in_sync(art):
                    upload_art(art, complete=False)
        upload_extras(session, None if final else s)
    return after_step


def shutdown_after(code: int, session: str) -> None:
    """Powers the VM off, but only after a run that exited 0 and whose results are in the
    bucket as they are on the disk: the NVMe is wiped when the VM stops. A run a gate
    stopped is such a run too (the steps after the gate are skipped, the upload step
    runs and the run exits 0), so a B200 never idles after a decided stop."""
    if code != 0:
        print(f"\nNot shutting down: the run did not finish (exit code {code}), and stopping the VM wipes the NVMe.")
        return
    if not uploaded(session):
        print("\nNot shutting down: the results on the NVMe are not the ones uploaded, and stopping the VM wipes them.")
        return
    code = os.system("sudo shutdown -h +5")
    if code != 0:
        print(f"\n{'!' * 72}\nSHUTDOWN FAILED (`sudo shutdown -h +5` returned {code}): THE VM IS STILL UP AND BILLED.\n"
              f"The results are uploaded; power it off by hand (sudo shutdown -h now) or in the evroc console.\n"
              f"{'!' * 72}")
        return
    print("\nResults uploaded. Shutting the VM down in 5 minutes (cancel with: sudo shutdown -c).")


# ---------- the sessions ----------

def codebench_art() -> Art:
    """The code suites as session 1 fetched them: the run's one copy (run_setup restores it
    on every later VM before anything reads it)."""
    return Art("codebench", lambda: "pinned",
               lambda: [CODEBENCH / f"{s}.{x}" for s in SUITES for x in ("jsonl", "meta.json")],
               SESSIONS, light=lambda: [CODEBENCH / f"{s}.meta.json" for s in SUITES], once=True,
               unfit=codebench_problems)


def choice_art() -> Art:
    """choice.json under one key per run, whatever the hashes upstream of it: every
    session's restore brings back the run's one pre-registration."""
    return Art("choice", lambda: "run", lambda: [CHOICE], SESSIONS, light=lambda: [CHOICE])


def reading_art(r: Reading) -> Art:
    def key() -> str | None:
        d = reading_dir(r)
        return d.name.rsplit("-", 1)[1] if d else None
    return Art(f"reading-{r.name}", key, lambda: [reading_dir(r)] if reading_dir(r) else [], SESSIONS,
               light=lambda: [reading_dir(r)] if reading_dir(r) else [])


def dir_art(slug: str, where: Callable[[], Path | None], sessions: tuple[str, ...], light: tuple[str, ...] = (),
            exclude: tuple[str, ...] = ()) -> Art:
    """An art that is one hash-named dir; its key is the dir's hash."""
    def key() -> str | None:
        d = where()
        return d.name.rsplit("-", 1)[1] if d else None

    def lights() -> list[Path]:
        d = where()
        return [f for x in light for f in (sorted(d.glob(x)) if "*" in x else [d / x])] if d else []
    return Art(slug, key, lambda: [where()] if where() else [], sessions, light=lights, exclude=exclude)


def reading_step(session: str, name: str, kind: str, gates: tuple[str, ...] = ()) -> Step:
    """A reading's step on kind's GPU; its guess is re-based once the reference reading has
    measured the rate (reading_minutes(), read again as each step starts)."""
    r = READINGS[name]
    return step(session, r.title, lambda: reading_minutes(r, kind), lambda s: run_reading(s, r, kind),
                lambda: reading_done(r), reading_art(r), gates)


def common_start(session: str, m: dict) -> list[Step]:
    return [step(session, "VM setup", {"": m["setup"]}, run_setup, setup_done, codebench_art()),
            step(session, "restore from the bucket", {"": m["restore"]}, lambda s: run_restore(s, session),
                 lambda: False)]


def download_step(session: str, m: dict) -> Step:
    return step(session, "Granite 4.2 3B download", {"": m["download"]}, run_download, download_done,
                progress=download_progress)


def upload_step(session: str, m: dict) -> Step:
    return step(session, "upload, then stop", {"": m["upload"]}, lambda s: run_upload(s, session),
                lambda: uploaded(session))


# The gates each part's steps run behind: a failed one skips them.
G0 = ("gate0",)
PRE = (*G0, "preflight")
G1 = (*PRE, "gate1")
CHOSEN = (*G1, "parity")  # choice.json is written


def measure_steps(session: str, m: dict, kind: str) -> list[Step]:
    """Session measure's work (S1.1's download to S1.10): the base, the smoke tests, the
    original's readings (Gate 0), the training text, the trim, the imatrix, PTQ (the
    preflight), Block-AP and its dev screen (Gate 1)."""
    return [
        download_step(session, m),
        step(session, "BF16 GGUF", {"gguf": m["gguf"]}, run_bf16, BF16.exists),
        step(session, "GPU smoke (a)-(h)", {"": m["smoke"]}, lambda s: run_smoke(s, kind), smoke_done,
             dir_art("smoke", smoke_dir, ("measure",), ("smoke.json",), ("*.gguf", "*.kld", "vocab", "block"))),
        reading_step(session, "original-ref", kind),
        reading_step(session, "original-dev", kind, G0),
        step(session, "training text, 8M tokens", lambda: {"": minutes(kind)["gen-measure"]},
             lambda s: run_gen(s, "measure", kind), lambda: gen_done("measure"),
             dir_art("traces", traces_dir, ("measure", "train"), ("traces.sha256", "traces.count", ".gen-done")),
             G0, lambda: traces_progress("measure")),
        step(session, "vocab trim and its gates", {"": m["vocab"]}, run_vocab, vocab_done,
             dir_art("vocab", vocab_dir, ("measure", "train"), ("vocab.json", "vocab.sha256", "hf.report.json")),
             G0),
        step(session, "imatrix on the trimmed BF16", {"": m["imatrix"]}, run_imatrix, imatrix_done,
             dir_art("imatrix", imatrix_dir, ("measure",), ("work/imatrix.sha256", "work/imatrix.json"),
                     ("model-bf16.gguf",)), G0),
        step(session, "PTQ, sensitivity, knapsack, preflight", {"": m["ptq"]}, run_ptq, ptq_done,
             dir_art("ptq", ptq_dir, ("measure",),
                     ("map.json", "horizon.json", "preflight.json", "floor.json", "bytes.json", "ladder.json",
                      "ladder.txt"),
                     ("cache", "teacher-*.pt", "horizon-*.gguf")), G0),
        step(session, "Block-AP and its export", {"": m["blockap"]}, run_blockap, blockap_done,
             dir_art("blockap", blockap_dir, ("measure", "train"),
                     ("block.json", "export/export.json", "export/model.gguf.sha256"),
                     ("export/vllm", "export/model-hf", "layers")), PRE),
        reading_step(session, "blockap-dev", kind, PRE),
    ]


def train_steps(session: str, m: dict, kind: str) -> list[Step]:
    """Session train's work (S2.2 to S2.7's choice.json), behind Gate 1."""
    onpolicy_art = dir_art("onpolicy", onpolicy_dir, ("train",),
                           ("state.json", "onpolicy.json", "audit.json", "config.json", "held.json",
                            "round*/judge.json", "round*/held.metrics.json"), ("serve",))
    return [
        step(session, "B200 smoke and rates", {"": m["rates"]}, lambda s: run_rates(s, kind), rates_done,
             dir_art("rates", rates_dir, ("train",), ("rates.json", "qat/e2e.json"),
                     ("qat/state", "qat/ckpt.pt", "onpolicy/round*/train", "serve")), G1),
        step(session, "training text to 24M tokens", {"": m["gen-train"]}, lambda s: run_gen(s, "train", kind),
             lambda: gen_done("train"),
             dir_art("traces24", lambda: traces_dir("train"), ("train",),
                     ("traces.sha256", "traces.count", ".gen-done")),
             G1, lambda: traces_progress("train")),
        step(session, "end-to-end KD", lambda: {"": minutes(kind)["e2e"]}, run_e2e, e2e_done,
             dir_art("e2e", e2e_dir, ("train",), ("e2e.json",), ("serve",)), G1),
        *[step(session, f"on-policy session {k} of up to {ONPOLICY_SESSIONS}"
               + (" (judges the last round)" if k == ONPOLICY_SESSIONS else f" (round {k})"),
               lambda k=k: {"": minutes(kind)["onpolicy"] / ROUNDS_MAX if k < ONPOLICY_SESSIONS
                            else minutes(kind)["onpolicy-judge"]},
               lambda s, k=k: run_onpolicy(s, k), lambda k=k: onpolicy_session_done(k), onpolicy_art, G1)
          for k in range(1, ONPOLICY_SESSIONS + 1)],
        step(session, "final export and parity", {"": m["final"]}, run_final, final_done,
             dir_art("final", final_dir, ("train", "read"),
                     ("export.json", "kld.json", "tokenize.json", "parity.json", "model.gguf.sha256"),
                     ("vllm", "dequant-bf16.gguf")), G1),
        step(session, "choice.json pre-registration", {"": m["choice"]}, run_choice, choice_done, choice_art(),
             CHOSEN),
    ]


def read_steps(session: str, m: dict, kind: str, before: tuple[str, ...] = ()) -> list[Step]:
    """Session read's work (S3.1's rebuild to S3.5's rescore), behind the gates `before`:
    none in session read (its restore checks choice.json), every one before choice.json in
    session all, where a stop at Gate 2 skips the rescore too (session read rescores the
    original's answers whatever Gate 2 says)."""
    return [
        step(session, "final checkpoint from the shipped GGUF", {"": m["rebuild"]}, run_rebuild, rebuild_done,
             gates=before),
        reading_step(session, "final-dev", kind, before),
        reading_step(session, "final-record", kind, (*before, "gate2")),
        reading_step(session, "final-llama", kind, (*before, "gate2")),
        step(session, "all-cases CPU rescore", {"": m["rescore"]}, run_rescore, rescore_done,
             Art("rescore", lambda: h([p.name for p in rescore_files()]) if rescore_files() else None,
                 rescore_files, ("read",), light=rescore_files), (*before, "gate2") if before else ()),
    ]


def steps(session: str) -> list[Step]:
    """A session's steps. Session all is one B200 VM: the setup, the restore and the
    download once, then measure's, train's and read's work back to back (train's own
    smoke and rates step in it), each part behind every gate before it, and one upload at
    the end (the hook uploads after every step as in any session)."""
    kind = machine(session)
    m = minutes(kind)
    if session == "measure":
        return [*common_start(session, m), *measure_steps(session, m, kind), upload_step(session, m)]
    if session == "train":
        return [*common_start(session, m), download_step(session, m), *train_steps(session, m, kind),
                upload_step(session, m)]
    if session == "read":
        return [*common_start(session, m), *read_steps(session, m, kind), upload_step(session, m)]
    if session == ALL:
        return [*common_start(session, m), *measure_steps(session, m, kind), *train_steps(session, m, kind),
                *read_steps(session, m, kind, CHOSEN), upload_step(session, m)]
    raise ValueError(f"no session {session!r}; one of {', '.join((*SESSIONS, ALL))}")


# ---------- summary ----------

def f1(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def is_harness(key: str) -> bool:
    return key in EVAL or key.startswith(HARNESS_PREFIXES)


def config_diff(d: Path, ref: Path) -> tuple[list[str], list[str]] | None:
    """(harness, other) keys whose values differ between two readings' config.effective.json
    (a key one side lacks differs too), less PER_READING; None when one is missing."""
    a, b = load(d / "work" / "config.effective.json"), load(ref / "work" / "config.effective.json")
    if a is None or b is None:
        return None
    keys = [k for k in sorted((set(a) | set(b)) - PER_READING) if k not in a or k not in b or a[k] != b[k]]
    line = lambda k: f"{k}: {a.get(k, '(none)')!r} vs {b.get(k, '(none)')!r}"
    return [line(k) for k in keys if is_harness(k)], [line(k) for k in keys if not is_harness(k)]


def settings_diff(d: Path, ref: Path) -> list[str]:
    """The record harness settings in which a reading differs from its original's, read
    back from both config.effective.json files (these block the verdict)."""
    diff = config_diff(d, ref)
    return ["config.effective.json missing"] if diff is None else diff[0]


def settings_reported(d: Path, ref: Path) -> list[str]:
    """The other Config fields that differ (reported only): pipeline knobs that the code
    may add or change between sessions do not touch a reading."""
    diff = config_diff(d, ref)
    return [] if diff is None else diff[1]


def provenance_diff(d: Path, ref: Path) -> tuple[list[str], list[str]]:
    """(blocking, reported) differences between two readings' provenance.json."""
    a, b = load(d / "work" / "provenance.json"), load(ref / "work" / "provenance.json")
    if a is None or b is None:
        return ["provenance.json missing"], []
    block = []
    for k in PROVENANCE_BLOCKING:
        x, y = (gpu_names(a), gpu_names(b)) if k == "gpu" else (a.get(k), b.get(k))
        if x != y:
            block.append(f"{k}: {x!r} vs {y!r}")
    other = [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in ("git_head", "git_dirty") if a.get(k) != b.get(k)]
    return block, other


def off_gpu() -> list[str]:
    """The readings made on another GPU name than the original's reference reading, which
    defines the harness (provenance_diff's gpu, against the reference for every reading:
    the dev screens, Gate 2's, the record and the llama-server check). Each blocks the
    verdict."""
    ref = reading_dir(READINGS["original-ref"])
    out = []
    for r in READINGS.values():
        d = reading_dir(r)
        if r.name == "original-ref" or not (ref and d and (d / "work" / "provenance.json").exists()):
            continue
        out += [f"{r.name}: {b}" for b in provenance_diff(d, ref)[0] if b.startswith("gpu:")]
    return out


def ids_sha(r: Reading) -> str | None:
    d = reading_dir(r)
    rep = load(d / "out" / "eval.json") if d else None
    for c in (rep or {}).get("candidates", []):
        return (((c.get("code") or {}).get("suites") or {}).get("livecodebench") or {}).get("ids_sha256")
    return None


def reading_lines(r: Reading) -> list[str]:
    st = reading_stats(r)
    if st is None:
        return [f"{r.name}: not read yet"]
    R, se, lb, gb = st["R"], st["se"], st["lb"], st["gb"]
    boot = st["boot"]
    out = [f"{r.name} ({r.role}): {st['problems']} problems, {st['answers']} answers, first {LCB_MAX_CASES} cases",
           f"  R {R:.1f}  SE {f1(se, 2)}  lower bound {f1(lb)}  problem bootstrap {f1(boot[0])}-{f1(boot[1])}  |  "
           f"{gb:.3f} GB: {R / gb:.1f} points/GB (lower bound {f1(lb and lb / gb)})  |  R/40 = {R / GOAL_PTS_PER_GB:.3f} GB",
           "  shares: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in st["shares"].items())
           + f"  |  8k re-scored {f1(st['budget8k'])} vs {LCB_TOKENS // 1024}k {f1(st['budget16k'])} on the same "
             f"answers (the first per problem)  |  all cases {f1(st['allcases'])}"]
    ref = READINGS[r.ref]
    if ref is not r:
        base = outcomes(rows_of(ref))
        p = paired(st["outs"], base)
        if p:
            kept = share_kept({i: sum(v) / len(v) for i, v in st["outs"].items()},
                              {i: sum(v) / len(v) for i, v in base.items()})
            out.append(f"  vs {ref.name} on {p[2]} problems: {p[0]:+.1f} points (SE {f1(p[1], 2)})"
                       + (f", {kept[0]:.1%} kept ({kept[1]:.1%} to {kept[2]:.1%})" if kept else ""))
        d, dr = reading_dir(r), reading_dir(ref)
        if r.model != "final-gguf" and dr and dr.exists():
            diff = settings_diff(d, dr)
            block, other = provenance_diff(d, dr)
            other = settings_reported(d, dr) + other
            out.append(f"  settings: {'same as ' + ref.name if not diff else 'differ from ' + ref.name + ': ' + '; '.join(diff)}"
                       + (f"; provenance differs: {'; '.join(block)}" if block else "")
                       + (f" (reported only: {'; '.join(other)})" if other else ""))
    off = [x.split(": ", 1)[1] for x in off_gpu() if x.startswith(f"{r.name}: ")]
    if off:
        out.append(f"  not on the original reference reading's GPU ({off[0]}): blocks the verdict")
    return out


def pin_lines() -> list[str]:
    """What the run is pinned to, for the TODOs that pin it in the code after session 1:
    the base's downloaded commit (MODEL_REVISION) and the code suites' dataset commits
    (fetch_codebench.REVISIONS)."""
    out = []
    if HF_DIR.exists():
        out.append(f"Granite 4.2 3B at {download_commit() or 'an unknown commit'} (revision {MODEL_REVISION})")
    metas = {s: load(CODEBENCH / f"{s}.meta.json") for s in SUITES}
    if any(metas.values()):
        out.append("Code suites at: " + ", ".join(f"{s} {(m or {}).get('repo', '?')}@{(m or {}).get('sha', '?')}"
                                                   for s, m in metas.items() if m))
    return out


def gate_lines() -> list[str]:
    out = []
    g = gate("gate0")
    if g:
        out.append(f"Gate 0: R0 {g['R0']:.1f} {'>=' if g['pass'] else '<'} {GATE0_MIN_R0}: "
                   f"{'pass' if g['pass'] else 'STOP'}; budget {g['budget_gb']:.3f} GB")
    v = vocab()
    if v:
        out.append(f"Vocab: {'trimmed to ' + str(v['rows']) + ' rows' if v['rows'] else 'not trimmed (the gates refused)'}"
                   + "".join(f"; {t['rows'] or 'measured set'}: {'pass' if t['pass'] else 'fail'} (dropped mass mean "
                             f"{t.get('dropped_mean')}, p99.9 {t.get('dropped_p999')}, NLL/byte rise "
                             f"{t.get('nll_per_byte_rise')}, llama-tokenize {t.get('llama_tokenize')})"
                             for t in v.get("tried", [])))
    g = gate("preflight")
    if g:
        out.append(f"Preflight: {'cheapest possible map' if g.get('floor') else 'PTQ map'} {g['gb']:.3f} GB, "
                   f"budget {g['budget_gb']:.3f} GB, "
                   f"40 x size = {GOAL_PTS_PER_GB * g['gb']:.1f} vs R0 {g['R0']:.1f}: "
                   + ("pass" if g["pass"] else f"STOP ({g['why']})"))
        out += [f"  {x}" for x in ptq_lines(g)]
    b = blockap_dir()
    out += [f"  ! Block-AP warning: {w}" for w in ((load(b / "block.json") if b else None) or {}).get("warnings") or []]
    for name, n in (("gate1", "Gate 1"), ("gate2", "Gate 2")):
        g = gate(name)
        if g:
            out.append(f"{n}: R0 {g['R0']:.1f} x retention {g['retention']:.1%} / {g['gb']:.3f} GB = "
                       f"{g['predicted_pts_per_gb']:.1f}/GB vs {g['threshold']}: {'pass' if g['pass'] else 'STOP'}")
    o = onpolicy_dir()
    st = load(o / "state.json") if o else None
    for d in (st or {}).get("rounds") or []:
        r = d.get("rules") or {}
        val = lambda m, k, r=r: (r.get(m) or {}).get(k)
        out.append(f"On-policy round {d.get('round')}: {'kept' if d.get('kept') else 'not kept'}: KL "
                   f"{-100 * (val('kl', 'drop') or 0):+.1f}%, loop {f1((val('loop', 'rise') or 0) * 100)} pts "
                   f"(2 SE {f1(2 * (val('loop', 'se') or 0) * 100)}), cap {f1((val('cap', 'rise') or 0) * 100)} pts "
                   f"(2 SE {f1(2 * (val('cap', 'se') or 0) * 100)}), held-out pass {100 * (val('pass', 'diff') or 0):+.1f} "
                   f"pts (2 SE {f1(200 * (val('pass', 'se') or 0))})")
    audit = load(o / "audit.json") if o else None
    if audit and not audit.get("ok"):
        out.append("On-policy rounds break the plan's rules: " + "; ".join(audit.get("problems", [])))
    g = gate("parity")
    if g:
        out.append(f"Parity: KLD {f1(g['kld_mean'], 4)}, top-1 {f1(g['top1'] and 100 * g['top1'])}%, size off by "
                   f"{100 * g['size_rel']:.2f}%: {'pass' if g['pass'] else 'FAIL (' + g['why'] + ')'}")
    c = choice()
    if c:
        out.append(f"choice.json: {c['gguf']} ({c['gb']:.3f} GB, sha256 {c['gguf_sha256'][:12]}), "
                   f"{c['record']['answers']} record answers from seed {c['record']['seed_offset']}")
    return out


def record_checks() -> dict[str, bool]:
    """The checks a verdict needs (section 5); any failing one blocks it."""
    rec, ref, llama = READINGS["final-record"], READINGS["original-ref"], READINGS["final-llama"]
    rows, k, c = rows_of(rec), k_alloc() or {}, choice() or {}
    outs = outcomes(rows)
    d, dr = reading_dir(rec), reading_dir(ref)
    block, _ = provenance_diff(d, dr) if d and dr else (["missing"], [])
    out = {"0 errored answers": bool(rows) and not any(map(tr.errored, rows)),
           "every problem has its K_i answers": bool(k) and {i: len(v) for i, v in outs.items()} == k,
           "the K_i file as pre-registered": bool(k) and sha256_file(reading_dir(ref) / "out" / "k_alloc.json")
           == (c.get("record") or {}).get("k_sha256"),
           "same problems as the original": ids_sha(rec) is not None and ids_sha(rec) == ids_sha(ref)
           == (window("test") or {}).get("ids_sha256"),
           "settings and provenance as the original's": bool(d and dr) and not settings_diff(d, dr) and not block,
           "parity gates pass": bool((gate("parity") or {}).get("pass")),
           "the GGUF is choice.json's": bool(c) and text(final_hf() / "gguf.sha256") == c.get("gguf_sha256"),
           "every reading on the original's GPU": bool(harness_gpu()) and not off_gpu(),
           "the harness and seeds as pre-registered": bool(c) and c.get("harness_sha") == h(harness_settings())
           and (c.get("record") or {}).get("seed_offset") == rec.seed}
    lo = outcomes(rows_of(llama))
    p = paired(lo, outs) if lo else None
    out[f"llama-server within {LLAMA_CHECK_SE:.0f} SE of vLLM"] = bool(p and p[1] is not None
                                                                        and p[0] >= -LLAMA_CHECK_SE * p[1])
    return out


def verdict(R: float, se: float | None, gb: float, checks: dict[str, bool]) -> str:
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        return "blocked (" + "; ".join(failed) + ")"
    lb = lower_bound(R, se)
    if lb is not None and lb / gb >= GOAL_PTS_PER_GB:
        return "reached"
    if R / gb >= GOAL_PTS_PER_GB:
        return "point estimate only"
    return "not reached"


def verdict_lines() -> list[str]:
    st = reading_stats(READINGS["final-record"])
    if st is None:
        return []
    checks = record_checks()
    v = verdict(st["R"], st["se"], st["gb"], checks)
    need = GOAL_PTS_PER_GB * st["gb"] + Z_ONE_SIDED * (st["se"] or 0)
    return [f"Verdict (pre-registered in choice.json, one look): {v.upper()}",
            f"  {st['R']:.1f} points (lower bound {f1(st['lb'])}) at {st['gb']:.3f} GB = {st['R'] / st['gb']:.1f}/GB "
            f"(lower bound {f1(st['lb'] and st['lb'] / st['gb'])}/GB); reached needs R >= {need:.1f}",
            "  checks: " + ", ".join(f"{k} {'ok' if ok else 'FAILED'}" for k, ok in checks.items())]


def summary() -> str:
    g, n = harness_gpu(), record_seqs()
    engine = (f"vLLM with {n} in flight on the {g}, every reading on that one GPU" if g and n else
              "vLLM on the GPU of the original's reference reading, with its RECORD_SEQS in flight, every reading "
              "on that one GPU")
    out = [f"Event Horizon, run {RUN_ID}. Goal: >= {GOAL_PTS_PER_GB} LiveCodeBench points per GB of the shipped "
           f"GGUF (bytes / 1e9), on our harness (LiveCodeBench v6 2025-01..04, thinking on, {LCB_TOKENS} tokens + "
           f"forced answer, temperature 1.0, top_p 0.95, {engine})."]
    stop = stop_line()
    if stop:
        out.append(stop)
    for r in READINGS.values():
        out += reading_lines(r)
    gl, vl = pin_lines() + gate_lines(), verdict_lines()
    if gl:
        out += ["", *gl]
    if vl:
        out += ["", *vl]
    out.append("SE: the reading's own sampling noise on its fixed problems; the bootstrap draws new problems "
               "(reported only). Granite's own pretraining is outside our control.")
    return "\n".join(out)


# ---------- helpers that run in the training venv ----------

def template_problems(render: Callable[[list[dict]], str], system: str, user: str, markup: list[str]
                      ) -> tuple[list[str], str]:
    """S1.2(b): (problems, the default system text). The chat template rendered with and
    without our system message must open the answer the same way (the thinking switch)
    and add no text next to ours. A default system text that it adds only when no system
    message is given (Granite's templates do) is returned to be reported, not a problem:
    every prompt the run sends carries codebench.SYSTEM (eval, gen, the rollouts, budget)."""
    with_sys = render([{"role": "system", "content": system}, {"role": "user", "content": user}])
    without = render([{"role": "user", "content": user}])
    i, j = with_sys.find(user), without.find(user)
    if i < 0 or j < 0:
        return ["the user message is not in the rendered prompt"], ""
    bad = []
    if with_sys[i + len(user):] != without[j + len(user):]:
        bad.append(f"the generation prompt differs with and without the system message: "
                   f"{with_sys[i + len(user):]!r} vs {without[j + len(user):]!r}")
    if system not in with_sys[:i]:
        bad.append("codebench.SYSTEM is not in the rendered prompt")

    def words(s: str) -> list[str]:
        for m in sorted(markup, key=len, reverse=True):
            s = s.replace(m, " ")
        return [w for w in re.findall(r"[A-Za-z]{2,}", s) if w.lower() not in ("system", "user", "assistant")]

    extra, default = words(with_sys[:i].replace(system, " ", 1)), words(without[:j])
    if extra:
        bad.append(f"the template adds text next to our system message: {' '.join(extra)[:300]!r}")
    return bad, " ".join(default)[:300]


def smoke_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="event_horizon.py smoke", description="S1.2's checks that need the venv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--hf", required=True)
    t.add_argument("--out", help="the result as JSON: problems, the default system text, accepted")
    r = sub.add_parser("render")
    r.add_argument("--hf", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--bench", required=True)
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--traces", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--gguf", nargs="+", required=True)
    p.add_argument("--per-type", type=int, default=12)
    mp = sub.add_parser("map")
    mp.add_argument("--gguf", required=True)
    mp.add_argument("--out", required=True)
    tx = sub.add_parser("text")
    tx.add_argument("--bench", required=True)
    tx.add_argument("--out", required=True)
    tx.add_argument("--n", type=int, default=200)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if a.cmd == "template":
        from transformers import AutoTokenizer

        from stages import codebench
        tok = AutoTokenizer.from_pretrained(a.hf)
        render = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                      enable_thinking=True)
        user = codebench.build_prompt({"language": "python", "prompt": "Read two integers and print their sum.",
                                       "tests": [], "id": "smoke"})
        markup = list(tok.all_special_tokens) + list(tok.get_added_vocab())
        bad, default = template_problems(render, codebench.SYSTEM, user, markup)
        print(render([{"role": "system", "content": codebench.SYSTEM}, {"role": "user", "content": user}]))
        if default:
            print(f"(with no system message the template adds a default system text: {default!r}; reported only)")
        accepted = bool(bad) and os.environ.get("HORIZON_TEMPLATE_OK") == "1"
        if a.out:
            Path(a.out).write_text(json.dumps({"problems": bad, "default_system_text": default,
                                               "accepted": accepted}, indent=1) + "\n")
        if bad and not accepted:
            print("smoke (b) failed: " + "\n  ".join(bad) + "\nFix the prompt before any reading, or, after "
                  "reviewing the rendering above, re-run with HORIZON_TEMPLATE_OK=1.", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "render":
        from transformers import AutoTokenizer

        from stages import opd
        tok = AutoTokenizer.from_pretrained(a.hf)
        bad, rows = [], []
        for n, prob in enumerate(opd.problems(a.bench, limit=a.n)):
            ids = opd.render(tok, prob)
            if n == 0:
                print("first prompt:\n" + tok.decode(ids))
            remote, strs = opd.render_remote(a.url, opd.STUDENT, prob, token_strs=True)
            if ids != remote:
                if not any("vLLM /tokenize" in b for b in bad):
                    import tokenizers
                    import transformers
                    pins = [ln for f in ("venv-vllm.txt", "venv-train.txt") if (NVME / "venvs" / f).exists()
                            for ln in [f + ": " + " ".join(x for x in (NVME / "venvs" / f).read_text().split()
                                                           if x.split("==")[0] in ("transformers", "tokenizers"))]]
                    print(f"render() here: transformers {transformers.__version__}, tokenizers "
                          f"{tokenizers.__version__}; " + "; ".join(pins), file=sys.stderr)
                bad.append(f"{prob['id']}: render() != vLLM /tokenize: " + opd.render_diff(tok, ids, remote, strs))
            got = opd.sample(a.url, opd.STUDENT, ids, 2, a.max_tokens)
            if len(got) != 2 or not all(g["tokens"] and all(type(x) is int for x in g["tokens"]) for g in got):
                bad.append(f"{prob['id']}: gen gave {len(got)} rows, not 2 of token ids")
            rows += [{"id": prob["id"], "prompt": ids, "tokens": g["tokens"], "finish": g["finish"], "passed": None,
                      "forced": None} for g in got]
        Path(a.traces).write_text("".join(json.dumps(x) + "\n" for x in rows))
        if len(rows) != 2 * a.n:
            bad.append(f"{len(rows)} rows from {a.n} prompts")
        if bad:
            print("smoke (a)/(d) failed:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        return 0
    if a.cmd == "pack":
        import numpy as np
        import torch
        from gguf import GGUFReader
        from gguf.quants import dequantize

        from stages import kquant
        seen, bad = {}, []
        for path in a.gguf:
            for t in GGUFReader(path).tensors:
                typ = t.tensor_type.name.lower()
                if typ not in kquant.TYPES or seen.get(typ, 0) >= a.per_type:
                    continue
                seen[typ] = seen.get(typ, 0) + 1
                raw = np.ascontiguousarray(t.data)
                shape = tuple(int(x) for x in reversed(t.shape.tolist()))
                kq = kquant.from_gguf(raw, typ, shape)
                if kquant.pack(kq) != raw.tobytes():
                    bad.append(f"{t.name} ({typ}): pack(from_gguf(x)) != x")
                ref = torch.from_numpy(np.asarray(dequantize(raw, t.tensor_type), dtype=np.float32)).reshape(shape)
                if not torch.equal(kquant.dequant(kq).reshape(shape), ref):
                    bad.append(f"{t.name} ({typ}): dequant != gguf-py's")
        bad += [f"no {x} tensor in {' '.join(a.gguf)}" for x in ("q2_k", "q4_k") if not seen.get(x)]
        print(f"pack round trips: {seen}")
        if bad:
            print("smoke (e) failed:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        return 0
    if a.cmd == "map":
        from gguf import GGUFReader
        out = {}
        for t in GGUFReader(a.gguf).tensors:
            if len(t.shape) >= 2 and t.tensor_type.name.lower() in ("q2_k", "q3_k", "q4_k"):
                kv = re.search(r"attn_(k|v)\.weight$", t.name) or t.name == "output.weight"
                out[t.name] = "q4_k" if kv else "q2_k"  # P1's shape: k/v and the head at Q4_K
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        return 0
    if a.cmd == "text":
        lines, n = [], 0
        with open(Path(a.bench) / "codecontests.jsonl") as f:
            for line in f:
                if n >= a.n:
                    break
                lines.append(json.loads(line)["prompt"])
                n += 1
        Path(a.out).write_text("\n\n".join(lines) + "\n")
        return 0
    return 2


def rescore_main(argv: list[str]) -> int:
    """Grades a reading's stored answers again on the CPU (the fixed graders, every case
    with --max-cases 0), into work/code_eval/<name>-allcases.jsonl."""
    ap = argparse.ArgumentParser(prog="event_horizon.py rescore")
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--chunk", type=int, default=32, help="problems whose answers are graded together")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    from stages import codebench, sandbox
    from stages.eval import cap_cases
    cfg = load(a.job / "work" / "config.effective.json")
    src = a.job / "work" / "code_eval" / f"{a.name}.jsonl"
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    ids = {x["id"] for x in rows}
    probs = {p["id"]: p for p in codebench.load_suite("livecodebench", cfg["code_eval_dir"], cfg["lcb_since"], ids=ids)}
    by_id: dict[str, list[int]] = {}
    for k, x in enumerate(rows):
        by_id.setdefault(x["id"], []).append(k)
    # The items are built a few problems at a time and dropped once graded: each carries
    # its problem's whole case list (hundreds of MB on LiveCodeBench's largest), up to 8
    # answers a problem, and all at once they once held every copy in memory.
    results: list = [None] * len(rows)
    order, chunk = sorted(by_id), max(1, a.chunk)
    for c in range(0, len(order), chunk):
        ks = [k for i in order[c:c + chunk] for k in by_id[i]]
        capped = {i: cap_cases(probs[i], a.max_cases) for i in order[c:c + chunk]}
        items = [codebench.job(capped[rows[k]["id"]], rows[k]["answer"], codebench.extract_code(rows[k]["answer"]))
                 for k in ks]
        try:
            got = sandbox.run_many(items, timeout=a.timeout)
        except (OSError, MemoryError) as e:  # one bad problem is recorded, not the end of the step
            got = [sandbox.Result(False, "grader_error", f"{type(e).__name__}: {e}") for _ in items]
        for k, res in zip(ks, got, strict=True):
            results[k] = res
        del items, capped
    out = a.job / "work" / "code_eval" / f"{a.name}-allcases.jsonl"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps({"suite": x.get("suite", "livecodebench"), "id": x["id"], "sample": x["sample"],
                                       "passed": res.passed, "reason": res.reason, "was": x.get("passed"),
                                       "max_cases": a.max_cases}) + "\n" for x, res in zip(rows, results, strict=True)))
    os.replace(tmp, out)
    print(f"{a.name}: {sum(r.passed for r in results)}/{len(results)} pass on "
          f"{'every case' if not a.max_cases else f'the first {a.max_cases} cases'} "
          f"(was {sum(bool(x.get('passed')) for x in rows)})")
    return 0


HELPERS = {"smoke": smoke_main, "rescore": rescore_main}


def main(argv: list[str] | None = None) -> int:
    global SHUTDOWN
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in HELPERS:
        return HELPERS[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=True)
    ap.add_argument("what", nargs="?", choices=["summary"])
    ap.add_argument("--session", choices=(*SESSIONS, ALL))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        sessions = [a.session] if a.session else [*SESSIONS, ALL]
        for kind in dict.fromkeys(machine(s) for s in sessions):
            print(measured(kind))
        for s in sessions:
            print(f"\nsession {s}: on the {machine(s)}")
            run_plan(steps(s), summary, LOGS / s, ["--list"])
        return 0
    if a.what == "summary":
        print(summary())
        return 0
    if not a.session:
        print("Name the session: --session all (one B200), or measure (L40S or B200), train (B200) and read (L40S or "
              "B200). --list shows the steps.")
        return 2
    SHUTDOWN = a.shutdown
    # Every vLLM this run starts inherits these (eval.VLLM_JIT_FREE): no FlashInfer JIT build.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if this_gpu() == "B200":
        os.environ.setdefault("LOBBOT_VLLM_ATTN_BACKEND", "FLASH_ATTN")
    if a.shutdown and not evroc_ok():
        print("Refusing to start with --shutdown: there is no evroc login, so the results could not be uploaded "
              "before the VM powers off. Log in (evroc login --oidc-port 8765) and re-run.")
        return 2
    if not evroc_ok():
        print("WARNING: no evroc login. Nothing is restored or uploaded until there is one (evroc login "
              "--oidc-port 8765); the results stay only on the NVMe, which a VM stop wipes.")
    problem = machine_problem(a.session)
    if problem:
        print(problem)
        return 2
    (LOGS / a.session).mkdir(parents=True, exist_ok=True)
    code = run_plan(steps(a.session), summary, LOGS / a.session, ["run"], after_step=make_hook(a.session))
    if a.shutdown:
        shutdown_after(code, a.session)
    return code


if __name__ == "__main__":
    sys.exit(main())
