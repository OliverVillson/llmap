#!/usr/bin/env python3
"""Blackhole on one L40S: Granite 4.2 3B at most 1.5 GB, keeping 95% of its LiveCodeBench
score, measured so the result can be trusted. One command.

    python3 scripts/blackhole.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/blackhole.py --list   # the steps and their time guesses
    python3 scripts/blackhole.py summary  # the results table

Oliver, 2026-10-12: "Lets aim for an as reliable result as possible. I can spare up to
1.5GB". Only proven pieces (plan: /mnt/project-files/plan/blackhole.md):

  bh15     llama.cpp's imatrix-guided formats (stages/dense.py FIXED): feed-forward
           IQ2_S, its down projection IQ3_XXS, attention IQ4_XS, head Q4_K,
           embedding Q3_K: ~1.45 GB, runs fast anywhere llama.cpp does
  repair   a ~0.03 GB LoRA trained so the quantized model matches the original
           (stages/qheal.py: KL on the original's own thinking, the quantized weights
           frozen and read back from the GGUF), loaded on top by llama-server

Steps, two jobs sharing the GPU wherever they can:
  1. VM setup, the Granite download, a BF16 GGUF
  2. the original scored, while the training text (the original's thinking on clean
     CodeContests problems, never LiveCodeBench) is written and bh15 is made
  3. bh15 scored as it is, while the repair adapter trains
  4. bh15 with the adapter scored
  5. results to the mugge-library bucket

Every score: LiveCodeBench v6 2025-01..04, thinking on at the card's sampling, 16k
tokens, 4 answers per problem, the same llama-server settings for all three. The
summary compares each with the original problem by problem and gives a 95% interval
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
LORA = HEAL / "lora.gguf"
JOBS = {"original": e5.J / "blackhole-original", "bh15": e5.J / "blackhole-bh15-score",
        "bh15+repair": e5.J / "blackhole-bh15-repair"}
LIMIT_GB = 1.5
SAMPLES = 4
LCB_TOKENS = 16384
GEN_PORT = "8093"
GEN_TOKENS = 12e6  # the training text; the adapter trains on it once
HEAL_CAP_MINUTES = 100
MINUTES = {"setup": 45, "download": 5, "gguf": 5, "phase1": 100, "phase2": 95, "phase3": 70}


def eval_job(key: str) -> Path:
    d = JOBS[key]
    if not (d / "config.json").exists():
        model = {"original": e5.BF16, "bh15": QUANT, "bh15+repair": QUANT}[key]
        cfg = {**e5.THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
               "code_eval_max_tokens": LCB_TOKENS, "code_eval_samples": SAMPLES, "code_eval_ref": "",
               "eval_candidates": {key: str(model)}}
        if key == "bh15+repair":
            cfg["eval_lora"] = str(LORA)
        e5.make_job(d, cfg)
    return d


def score(step: Step, key: str, port: str = e5.PORTS[0]) -> None:
    d = eval_job(key)
    if not (d / ".done" / "eval").exists():
        sh(step, f"python pipeline.py --job {d} --only eval", e5.env_for(port))


def gen_command() -> str:
    """A vLLM on 40% of the GPU for the original's thinking, then the text; it stops after."""
    return "\n".join([
        "set -e",
        f"mkdir -p {W}",
        f"$LOBBOT_VLLM_PY -m vllm.entrypoints.openai.api_server --model {e5.HF_DIR} --served-model-name student "
        f"--host 127.0.0.1 --port {GEN_PORT} --max-model-len 12288 --gpu-memory-utilization 0.4 "
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


def make_text_and_quant(step: Step) -> None:
    """The training text (vLLM, then stopped), then calibration, importance matrix and bh15
    on the second llama-server port."""
    if not (W / ".gen-done").exists():
        sh(step, gen_command())
        W.mkdir(parents=True, exist_ok=True)
        (W / ".gen-done").write_text("ok")
    e5.base_job()
    if not (e5.BASE / "work" / "calib-chat.txt").exists():
        sh(step, f"python -m stages.dense calib --job {e5.BASE} --model {e5.BF16}", e5.env_for(e5.PORTS[1], e5.CALIB_SLOTS))
    if not (e5.BASE / "work" / "imatrix.gguf").exists():
        sh(step, f"python -m stages.dense imatrix --job {e5.BASE} --bf16 {e5.BF16}")
    if not (QUANT_JOB / ".done" / "quantize").exists():
        if not (QUANT_JOB / "config.json").exists():
            e5.make_job(QUANT_JOB, dict(e5.THINKING))
        sh(step, f"python -m stages.dense quantize --job {QUANT_JOB} --recipe {RECIPE} --bf16 {e5.BF16} "
                 f"--imatrix {e5.BASE / 'work' / 'imatrix.gguf'}")


def run_phase1(step: Step) -> None:
    e5.at_once(lambda: score(step, "original"), lambda: make_text_and_quant(step))


def phase1_done() -> bool:
    return (JOBS["original"] / ".done" / "eval").exists() and (QUANT_JOB / ".done" / "quantize").exists() \
        and (W / ".gen-done").exists()


def train_repair(step: Step) -> None:
    if not LORA.exists():
        sh(step, f"python -m stages.qheal --hf {e5.HF_DIR} --gguf {QUANT} --traces {TRACES} "
                 f"--llama-cpp {e5.NVME}/llama.cpp --tokens {GEN_TOKENS:.0f} --max-minutes {HEAL_CAP_MINUTES} "
                 f"--out {HEAL}")


def run_phase2(step: Step) -> None:
    """bh15 scored as it is while the repair adapter trains beside it."""
    e5.at_once(lambda: score(step, "bh15"), lambda: train_repair(step))


def phase2_done() -> bool:
    return (JOBS["bh15"] / ".done" / "eval").exists() and LORA.exists()


def run_upload(step: Step) -> None:
    """Configs, logs, scores and every answer, bh15 and the adapter (not the training text)."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 5) and re-run to upload.")
    tar = e5.NVME / "blackhole.tar"
    dirs = " ".join(f"{e5.J.name}/{d.name}" for d in [W.parent, QUANT_JOB, *JOBS.values()])
    sh(step, f"mkdir -p {W}/logs && cp {LOGS}/*.log {W}/logs/ && cd {e5.J.parent} && "
             f"find {dirs} -type f -size -200M ! -name traces.jsonl ! -name '*.safetensors' -print0 "
             f"| tar cf {tar} --null -T -")
    for src, key in ((tar, "blackhole/blackhole.tar"), (QUANT, "gguf/granite42-3b-bh15.gguf"),
                     (LORA, "gguf/granite42-3b-bh15-repair-lora.gguf")):
        if src.exists():
            sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                     f"evroc storage bucket copy --from {src} --to {e5.BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Granite 4.2 3B download", {"": MINUTES["download"]}, e5.run_download, e5.download_done,
             e5.download_progress),
        Step("BF16 GGUF", {"gguf": MINUTES["gguf"]}, e5.run_ggufs, e5.BF16.exists),
        Step("original scored; text and bh15 made", {"": MINUTES["phase1"]}, run_phase1, phase1_done),
        Step("bh15 scored; repair adapter trained", {"": MINUTES["phase2"]}, run_phase2, phase2_done),
        Step("bh15 + repair scored", {"": MINUTES["phase3"]}, lambda s: score(s, "bh15+repair"),
             lambda: (JOBS["bh15+repair"] / ".done" / "eval").exists()),
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
    lora = LORA.stat().st_size / 1e9 if LORA.exists() else 0.0
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
                  "same llama-server settings. The interval comes from resampling problems.")


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
