#!/usr/bin/env python3
"""Blackhole on one L40S: Granite 4.2 3B at most 1.5 GB, keeping 95% of its LiveCodeBench
score, measured so the result can be trusted. One command.

    python3 scripts/blackhole.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/blackhole.py --list   # the steps and their time guesses
    python3 scripts/blackhole.py summary  # the results table
    python3 scripts/blackhole.py --shutdown  # everything, then power the VM off once the results are uploaded

Oliver, 2026-10-12: "Lets aim for an as reliable result as possible. I can spare up to
1.5GB". Only proven pieces (plan: /mnt/project-files/plan/blackhole.md):

  bh15     llama.cpp's imatrix-guided formats (stages/dense.py FIXED): feed-forward
           IQ2_S, its down projection IQ3_XXS, attention IQ4_XS, head Q4_K,
           embedding Q3_K: ~1.45 GB, runs fast anywhere llama.cpp does
  repair   a ~0.03 GB LoRA trained so the quantized model matches the original
           (stages/qheal.py: KL on the original's own thinking, the quantized weights
           frozen and read back from the GGUF), loaded on top by llama-server --lora

Steps, one job at a time on the whole GPU (two servers starting together miscount each
other's memory): setup, download, BF16 GGUF; the original scored; the training text (the
original's thinking on clean CodeContests problems, never LiveCodeBench); bh15 made;
bh15 scored; the repair adapter trained; bh15 + repair scored; upload.

Every score: LiveCodeBench v6 2025-01..04, thinking on at the card's sampling, 16k
tokens, 4 answers per problem, all three on vLLM with the same settings (llama.cpp
managed ~100 answers an hour, ~7 h a model). bh15 is scored as its GGUF weights
dequantized into a bf16 checkpoint, the repair merged in for the last one: the same
weights llama.cpp runs, minus llama.cpp's 8-bit rounding of activations inside its
matmuls. The summary compares each with the original problem by problem and gives a 95% interval
for the share kept. Logs are in ~/blackhole-logs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import exp05 as e5
from runner import Skip, Step, evroc_ok, read, run_plan, sh

LOGS = Path(os.environ.get("BLACKHOLE_LOGS", Path.home() / "blackhole-logs"))
W = e5.J / "blackhole" / "work"
TRACES = W / "traces.jsonl"
RECIPE = "bh15"
QUANT_JOB = e5.J / "blackhole-bh15"  # dense.py's quantize job; its out/model.gguf
QUANT = QUANT_JOB / "out" / "model.gguf"
HEAL = W / "heal"
LORA = HEAL / "lora.gguf"  # the adapter for llama.cpp (shipped); scoring merges the PEFT one
ADAPTER = HEAL / "adapter"
HF = {"original": e5.HF_DIR, "bh15": W / "bh15-hf", "bh15+repair": W / "bh15-repair-hf"}
# All three are scored by vLLM (llama.cpp managed ~100 answers an hour on the L40S, ~7 h a
# model); the quantized ones as their GGUF weights dequantized into an HF checkpoint.
JOBS = {k: e5.J / f"blackhole-{k.replace('+', '-')}-vllm" for k in HF}
LIMIT_GB = 1.5
SAMPLES = 4
LCB_TOKENS = 16384
GEN_PORT = "8093"
GEN_TOKENS = 12e6  # the training text; the adapter trains on it once
HEAL_CAP_MINUTES = 100
# One job at a time on the whole GPU: two servers starting together miscount each
# other's memory (vLLM saw -13.9 GiB of KV cache on the first run).
MINUTES = {"setup": 45, "download": 5, "gguf": 5, "score": 40, "gen": 45, "quant": 20, "export": 5}


# Every GPU job waits until the last one's memory is back (a stopped vLLM's engine
# process lets go a little after the server; the second run's text vLLM crashed on it).
GPU_FREE = ("for i in $(seq 120); do [ \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "
            "| head -1)\" -lt 2000 ] && break; sleep 5; done")


def gpu(cmd: str) -> str:
    return f"{GPU_FREE}\n{cmd}"


def eval_job(key: str) -> Path:
    d = JOBS[key]
    if not (d / "config.json").exists():
        cfg = {**e5.THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
               "code_eval_max_tokens": LCB_TOKENS, "code_eval_samples": SAMPLES, "code_eval_ref": "",
               "eval_candidates": {key: str(HF[key])}}
        e5.make_job(d, cfg)
    return d


def scored(key: str) -> bool:
    return (JOBS[key] / ".done" / "eval").exists()


def score(step: Step, key: str) -> None:
    """LiveCodeBench on vLLM with the whole GPU; for bh15 and bh15+repair the checkpoint
    is exported from the GGUF first."""
    if key != "original" and not (HF[key] / "config.json").exists():
        sh(step, gpu(f"python -m stages.qheal --hf {e5.HF_DIR} --gguf {QUANT} --llama-cpp {e5.NVME}/llama.cpp "
                     f"--export {HF[key]}" + (f" --adapter {ADAPTER}" if key == "bh15+repair" else "")))
    if not scored(key):
        sh(step, gpu(f"python pipeline.py --job {eval_job(key)} --only eval"), {"LOBBOT_VLLM_GPU_UTIL": "0.9"})


def gen_command() -> str:
    """A vLLM on the GPU for the original's thinking, then the text; it stops after."""
    return "\n".join([
        GPU_FREE,
        "set -e",
        f"mkdir -p {W}",
        f"$LOBBOT_VLLM_PY -m vllm.entrypoints.openai.api_server --model {e5.HF_DIR} --served-model-name student "
        f"--host 127.0.0.1 --port {GEN_PORT} --max-model-len 12288 --gpu-memory-utilization 0.9 "
        f"--enable-prefix-caching > {W}/vllm-gen.log 2>&1 & gen=$!",
        "trap 'kill $gen 2>/dev/null' EXIT",
        f"until curl -sf http://127.0.0.1:{GEN_PORT}/health >/dev/null; do "
        f"kill -0 $gen 2>/dev/null || {{ tail -20 {W}/vllm-gen.log; exit 1; }}; sleep 5; done",
        f"python -m stages.ternary gen --model {e5.HF_DIR} --url http://127.0.0.1:{GEN_PORT} "
        f"--problems {e5.NVME}/codebench --out {TRACES} --tokens {GEN_TOKENS:.0f}",
    ])


# ---------- the steps ----------

def run_setup(step: Step) -> None:
    sh(step, f"NVME={e5.NVME} TEACHER={e5.MODEL} STUDENT= bash scripts/setup_vm.sh")


def setup_done() -> bool:
    return e5.setup_done() and (e5.NVME / "codebench/codecontests.jsonl").exists()


def make_text(step: Step) -> None:
    sh(step, gen_command())
    (W / ".gen-done").write_text("ok")


def text_progress() -> float | None:
    """~7 bytes of the traces file per token (the ids as JSON, plus the prompts)."""
    return min(0.99, TRACES.stat().st_size / (7 * GEN_TOKENS)) if TRACES.exists() else None


def make_quant(step: Step) -> None:
    """Calibration answers, importance matrix and bh15, on llama.cpp."""
    e5.base_job()
    if not (e5.BASE / "work" / "calib-chat.txt").exists():
        sh(step, gpu(f"python -m stages.dense calib --job {e5.BASE} --model {e5.BF16}"),
           e5.env_for(e5.PORTS[1], e5.CALIB_SLOTS))
    if not (e5.BASE / "work" / "imatrix.gguf").exists():
        sh(step, f"python -m stages.dense imatrix --job {e5.BASE} --bf16 {e5.BF16}")
    if not (QUANT_JOB / "config.json").exists():
        e5.make_job(QUANT_JOB, dict(e5.THINKING))
    sh(step, f"python -m stages.dense quantize --job {QUANT_JOB} --recipe {RECIPE} --bf16 {e5.BF16} "
             f"--imatrix {e5.BASE / 'work' / 'imatrix.gguf'}")


def train_repair(step: Step) -> None:
    sh(step, gpu(f"python -m stages.qheal --hf {e5.HF_DIR} --gguf {QUANT} --traces {TRACES} "
                 f"--llama-cpp {e5.NVME}/llama.cpp --tokens {GEN_TOKENS:.0f} --max-minutes {HEAL_CAP_MINUTES} "
                 f"--out {HEAL}"))


def run_upload(step: Step) -> None:
    """Configs, logs, scores and every answer, bh15 and the adapter (not the training text)."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 5) and re-run to upload.")
    (W / "summary.txt").write_text(summary() + "\n")
    tar = e5.NVME / "blackhole.tar"
    dirs = " ".join(f"{e5.J.name}/{d.name}" for d in [W.parent, QUANT_JOB, *JOBS.values()])
    sh(step, f"mkdir -p {W}/logs && cp {LOGS}/*.log {W}/logs/ && cd {e5.J.parent} && "
             f"find {dirs} -type f -size -200M ! -name traces.jsonl ! -name '*.safetensors' -print0 "
             f"| tar cf {tar} --null -T -")
    for src, key in ((W / "summary.txt", "blackhole/summary.txt"), (tar, "blackhole/blackhole.tar"),
                     (QUANT, "gguf/granite42-3b-bh15.gguf"),
                     (LORA, "gguf/granite42-3b-bh15-repair-lora.gguf")):
        if src.exists():
            sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                     f"evroc storage bucket copy --from {src} --to {e5.BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    score_min = {"eval": MINUTES["score"]}
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Granite 4.2 3B download", {"": MINUTES["download"]}, e5.run_download, e5.download_done,
             e5.download_progress),
        Step("BF16 GGUF", {"gguf": MINUTES["gguf"]}, e5.run_ggufs, e5.BF16.exists),
        Step("original scored", score_min, lambda s: score(s, "original"), lambda: scored("original")),
        Step("training text written", {"": MINUTES["gen"]}, make_text, (W / ".gen-done").exists, text_progress),
        Step("bh15 made", {"": MINUTES["quant"]}, make_quant, (QUANT_JOB / ".done" / "quantize").exists),
        Step("bh15 scored", {"": MINUTES["export"], **score_min}, lambda s: score(s, "bh15"), lambda: scored("bh15")),
        Step("repair adapter trained", {"": HEAL_CAP_MINUTES}, train_repair, (ADAPTER / "adapter_config.json").exists),
        Step("bh15 + repair scored", {"": MINUTES["export"], **score_min}, lambda s: score(s, "bh15+repair"),
             lambda: scored("bh15+repair")),
        Step("upload to the bucket", {"": 15}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

def per_problem(d: Path, name: str) -> dict[str, float]:
    """Each LiveCodeBench problem's pass rate over its samples."""
    p = d / "work" / "code_eval" / f"{name}.jsonl"
    if not p.exists():
        return {}
    got: dict[str, list[bool]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("suite") == "livecodebench":
                got.setdefault(r["id"], []).append(bool(r.get("passed")))
    return {k: sum(v) / len(v) for k, v in got.items()}


def share_kept(model: dict[str, float], ref: dict[str, float], n: int = 4000, seed: int = 0
               ) -> tuple[float, float, float] | None:
    """(share of the original's score kept, 95% interval low, high) over the problems both
    answered, by resampling problems (paired: each draw keeps a problem's two scores together)."""
    ids = sorted(set(model) & set(ref))
    if not ids or not sum(ref[i] for i in ids):
        return None
    share = sum(model[i] for i in ids) / sum(ref[i] for i in ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        pick = [rng.choice(ids) for _ in ids]
        r = sum(ref[i] for i in pick)
        if r:
            draws.append(sum(model[i] for i in pick) / r)
    draws.sort()
    return share, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def size_gb() -> dict[str, float]:
    q = QUANT.stat().st_size / 1e9 if QUANT.exists() else None
    fp32 = ADAPTER / "adapter_model.safetensors"  # if the llama.cpp copy failed: ~a quarter as Q8_0
    lora = LORA.stat().st_size / 1e9 if LORA.exists() else fp32.stat().st_size / 4e9 if fp32.exists() else 0.0
    return {"original": e5.BF16.stat().st_size / 1e9 if e5.BF16.exists() else 7.32,
            "bh15": q, "bh15+repair": None if q is None else q + lora}


def summary() -> str:
    ref = per_problem(JOBS["original"], "original")
    sizes = size_gb()
    rows = ["model         size     LCB (4 answers each)  share of original (95% interval)"]
    for key in JOBS:
        sc = per_problem(JOBS[key], key)
        if not sc:
            rows.append(f"{key:12}  {'':7}  not scored")
            continue
        lcb = 100 * sum(sc.values()) / len(sc)
        kept = share_kept(sc, ref) if key != "original" else None
        gb = sizes[key]
        rows.append(f"{key:12}  {f'{gb:.2f} GB' if gb else '':7}  {lcb:5.1f} on {len(sc)} problems    "
                    + (f"{kept[0]:.1%} ({kept[1]:.1%} to {kept[2]:.1%})" if kept else ""))
    out = "\n".join(rows)
    best = per_problem(JOBS["bh15+repair"], "bh15+repair")
    kept = share_kept(best, ref) if best and ref else None
    gb = sizes["bh15+repair"]
    if kept and gb:
        verdict = (f"over the {LIMIT_GB} GB limit" if gb > LIMIT_GB else
                   "reached, and the whole interval is above 95%" if kept[1] >= 0.95 else
                   "likely reached, but the interval dips below 95%" if kept[0] >= 0.95 else "not reached")
        out += (f"\n\nGoal (at most {LIMIT_GB} GB, 95% of the original): {gb:.2f} GB, {kept[0]:.1%} kept: {verdict}. "
                f"{sizes['original'] / gb:.1f}x smaller than BF16.")
    return out + ("\nEvery model: same window, 16k thinking, 4 answers per problem at temperature 1.0, "
                  "same vLLM settings. The interval comes from resampling problems.")


def shutdown_after() -> None:
    """Powers the VM off after a run, but only once the results are safe in the bucket:
    the VM's disk is wiped when it stops."""
    if not (LOGS / ".uploaded").exists():
        print("\nNot shutting down: the results aren't uploaded yet, and stopping the VM wipes them.")
        return
    print("\nResults uploaded. Shutting the VM down in 5 minutes (cancel with: sudo shutdown -c).")
    os.system("sudo shutdown -h +5")


def main() -> int:
    code = run_plan(steps(), summary, LOGS)
    if "--shutdown" in sys.argv and code != 130:
        shutdown_after()
    return code


if __name__ == "__main__":
    sys.exit(main())
