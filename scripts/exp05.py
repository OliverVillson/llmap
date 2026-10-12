#!/usr/bin/env python3
"""Experiment 05 on one L40S (or a B200): IBM Granite 4.2 3B with thinking on, in one command.

    python3 scripts/exp05.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp05.py --list   # the steps and their time guesses
    python3 scripts/exp05.py summary  # the results table

Granite 4.2 3B is a small dense model (40 layers, 3.7B weights with its untied
embeddings) whose card gives LiveCodeBench v6 69.7. This run measures it on our
LiveCodeBench window and finds how small it can get before the score drops, as the
baseline for an LCB-per-GB record (plan: /mnt/project-files/plan/granite-record.md).
Recipes (stages/dense.py), each a GGUF scored by llama-server:
  bf16  the model as released, ~7.3 GB
  q8    Q8_0, ~3.9 GB
  q4    r25q4's bits (Q4_K_M-like), ~2.2 GB
  m35   feed-forward ~3.5 bits by saliency, attention q5_k, head q6_k, embedding q4_k, ~1.9 GB
  m28   feed-forward ~2.8 bits, attention q4_k, head q5_k, embedding q3_k, ~1.5 GB

Steps (docs: /mnt/project-files/plan/exp05-granite.md):
  1. VM setup (scripts/setup_vm.sh) and the Granite download
  2. a BF16 GGUF
  3. calibration text: the model answers MultiPL-E and C problems with thinking on
  4. the importance matrix on that text
  5. per recipe: quantize, LiveCodeBench v6 (2025-01..04) with thinking on and a 32k
     budget, then the 8k (the card's recommended cap) and 16k scores by cutting that thinking
  6. the GGUFs and the job results to the mugge-library bucket

Scores are compared with the card's 69.7 (its own window and settings). Runs with the
system python3 (stdlib only); each step sources .env.vm. Logs are in ~/exp05-logs.
The GPU's memory picks the answers run at once (EXP05_VRAM_GB overrides it).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from runner import REPO, Skip, Step, evroc_ok, read, run_plan, sh, state_lock, write

NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
J = Path(os.environ.get("LOBBOT_JOBS", NVME / "jobs"))
LOGS = Path(os.environ.get("EXP05_LOGS", Path.home() / "exp05-logs"))
BUCKET = "bucket://mugge-library"
MODEL = "ibm-granite/granite-4.2-3b"
HF_DIR = NVME / "models" / "granite-4.2-3b"
BF16 = NVME / "models" / "gguf" / "granite-4.2-3b-BF16.gguf"
BASE = J / "granite-base"  # calibration answers and the importance matrix
TASKSPEC = "examples/python-utils.code.taskspec.json"
RECIPES = ["bf16", "q8", "q4", "m35", "m28"]
CARD_LCB = 0.697  # Granite 4.2 3B model card, LiveCodeBench v6 (window and settings not stated)
# The card recommends 8192 new tokens with thinking; 32k leaves room to see what a longer cap buys.
MAX_TOKENS = 32768
BUDGETS = [8192, 16384]
CALIB_SUITES = ["multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "c-set"]


def vram_gb() -> float:
    """The GPU's memory in GiB; EXP05_VRAM_GB overrides it (dry runs)."""
    if os.environ.get("EXP05_VRAM_GB"):
        return float(os.environ["EXP05_VRAM_GB"])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0]) / 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        sys.exit("nvidia-smi found no GPU. On a new VM the driver may still be installing: "
                 "wait for the reboot, SSH back in and re-run.")


# A 3B fits any GPU in BF16 (~7.3 GB), so the model itself writes the calibration
# answers and the importance pass runs on the GPU. Its KV cache is ~80 KiB per token
# (40 layers, 8 KV heads of 64), ~2.7 GB per answer at 32k in 16 bits. An L40S (48 GB)
# scores with an 8-bit KV cache, 16 answers at once next to the BF16 model; a B200
# keeps 16-bit and runs 48.
B200 = vram_gb() >= 150
EVAL_SLOTS, CALIB_SLOTS = ("48", "48") if B200 else ("16", "16")
KV_TYPE = "" if B200 else "q8_0"
MINUTES = ({"setup": 40, "download": 3, "gguf": 3, "calib": 8, "imatrix": 3, "eval": 15, "budgets": 5}
           if B200 else
           {"setup": 45, "download": 5, "gguf": 5, "calib": 15, "imatrix": 5, "eval": 40, "budgets": 10})
# The card's sampling: temperature 1.0, top_p 0.95, no top_k (0 turns it off in llama-server).
THINKING = {"teacher": MODEL, "code_eval_thinking": True, "code_eval_thinking_temperature": 1.0,
            "thinking_top_k": 0, "eval_kv_type": KV_TYPE}


def job_dir(recipe: str) -> Path:
    return J / f"granite-{recipe}"


def make_job(d: Path, cfg: dict, max_size_gb: float = 10.0) -> Path:
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    write(d / "config.json", cfg)
    spec = read(REPO / TASKSPEC)
    spec["target"] = {**spec.get("target", {}), "max_size_gb": max_size_gb}
    write(d / "taskspec.json", spec)
    return d


def skip_part(step: Step, part: str) -> None:
    with state_lock:
        step.finished.add(part)


# ---------- the steps ----------

def run_setup(step: Step) -> None:
    sh(step, f"NVME={NVME} TEACHER={MODEL} STUDENT= bash scripts/setup_vm.sh")


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and (NVME / "llama.cpp/build/bin/llama-server").exists()
            and (NVME / "codebench/livecodebench.jsonl").exists())


def run_download(step: Step) -> None:
    log = NVME / "download.log"
    while not download_done():
        if subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL).returncode != 0:
            time.sleep(10)
            if not download_done():
                raise RuntimeError(f"the model download stopped; see {log} and re-run scripts/setup_vm.sh")
        time.sleep(5)


def download_done() -> bool:
    log = NVME / "download.log"
    return (log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace")
            and (HF_DIR / "config.json").exists())


def download_progress() -> float:
    got = sum(f.stat().st_size for f in HF_DIR.rglob("*") if f.is_file()) if HF_DIR.exists() else 0
    return min(0.99, got / 7.3e9)  # ~3.7B in bf16


def base_job() -> Path:
    if not (BASE / "config.json").exists():
        make_job(BASE, {**THINKING, "code_eval_suites": CALIB_SUITES, "code_eval_max_tokens": 6144})
    return BASE


def run_ggufs(step: Step) -> None:
    base_job()
    sh(step, f"python -m stages.dense gguf --job {BASE} --hf {HF_DIR} --bf16 {BF16}")


def run_calib(step: Step) -> None:
    base_job()
    sh(step, f"python -m stages.dense calib --job {BASE} --model {BF16}", {"LOBBOT_EVAL_SLOTS": CALIB_SLOTS})


def run_imatrix(step: Step) -> None:
    base_job()
    sh(step, f"python -m stages.dense imatrix --job {BASE} --bf16 {BF16}")


def model_of(recipe: str) -> Path:
    """The recipe's GGUF: the BF16 one as released, or its job's quantized copy."""
    return BF16 if recipe == "bf16" else job_dir(recipe) / "out" / "model.gguf"


def run_recipe(recipe: str):
    def run(step: Step) -> None:
        d = job_dir(recipe)
        if not (d / "config.json").exists():
            make_job(d, {**THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
                         "code_eval_max_tokens": MAX_TOKENS, "code_eval_budgets": BUDGETS, "code_eval_ref": "",
                         "eval_candidates": {f"granite-{recipe}": str(model_of(recipe))}})
        env = {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS}
        if recipe == "bf16" or (d / ".done" / "quantize").exists():
            skip_part(step, "quantize")
        else:
            sh(step, f"python -m stages.dense quantize --job {d} --recipe {recipe} --bf16 {BF16} "
                     f"--imatrix {BASE / 'work' / 'imatrix.gguf'}")
        if (d / ".done" / "eval").exists():  # --only always reruns
            skip_part(step, "eval")
        else:
            sh(step, f"python pipeline.py --job {d} --only eval", env)
        sh(step, f"python -m stages.budget --job {d}", env)
    return run


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, thinking, scores, the imatrix) and both GGUFs."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 4) and re-run to upload.")
    tar = NVME / "exp05-jobs.tar"
    sh(step, f"mkdir -p {J}/_logs_exp05 && cp {LOGS}/*.log {J}/_logs_exp05/ && cd {J.parent} && "
             f"find {J.name}/granite-* {J.name}/_logs_exp05 -type f -size -200M -print0 | tar cf {tar} --null -T -")
    up = [(tar, "exp05/exp05-jobs.tar")]
    for r in RECIPES[1:]:  # the BF16 one is on Hugging Face
        gguf = model_of(r)
        if gguf.exists():
            up.append((gguf, f"gguf/granite42-3b-{r}.gguf"))
    for src, key in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Granite 4.2 3B download", {"": MINUTES["download"]}, run_download, download_done, download_progress),
        Step("BF16 GGUF", {"gguf": MINUTES["gguf"]}, run_ggufs, BF16.exists),
        Step("calibration answers (BF16)", {"calib": MINUTES["calib"]},
             run_calib, (BASE / "work" / "calib-chat.txt").exists),
        Step("importance pass (BF16)", {"imatrix": MINUTES["imatrix"]},
             run_imatrix, (BASE / "work" / "imatrix.gguf").exists),
        *[Step(r, {"quantize": 5, "eval": MINUTES["eval"], "budgets": MINUTES["budgets"]}, run_recipe(r),
               (job_dir(r) / ".done" / "budgets").exists) for r in RECIPES],
        Step("upload to the bucket", {"": 25}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

def summary() -> str:
    """LiveCodeBench pass@1 at each thinking budget, the 32k score as a share of the card, and
    points per GB of file (the record metric)."""
    head = ["model", "size", "FFN bits"] + [f"LCB {b // 1024}k" for b in BUDGETS + [MAX_TOKENS]] \
        + ["of card", "%/GB", "hit 32k", "tok/s"]
    table, lost = [head], []
    for r in RECIPES:
        d, name = job_dir(r), f"granite-{r}"
        alloc = read(d / "work" / "allocation.json") if (d / "work" / "allocation.json").exists() else {}
        rep = read(d / "out" / "eval.json") if (d / "out" / "eval.json").exists() else None
        if not rep:
            table.append([name, f"{alloc['size_gb']:.1f} GB" if alloc.get("size_gb") else "not run"]
                         + [""] * (len(head) - 2))
            continue
        c = next((x for x in rep["candidates"] if x["name"] == name), rep["candidates"][0])
        code = c.get("code") or {}
        budgets = code.get("budgets") or {}

        def lcb(b: int) -> float | None:
            if b == MAX_TOKENS:
                return ((code.get("suites") or {}).get("livecodebench") or {}).get("pass@1")
            s = budgets.get(str(b)) or {}
            return ((s.get("suites") or {}).get("livecodebench") or {}).get("pass@1", s.get("mean_pass@1"))

        scores = [lcb(b) for b in BUDGETS + [MAX_TOKENS]]
        top, tps = scores[-1], c.get("tok_s_vm")
        hit = (budgets.get(str(MAX_TOKENS)) or {}).get("how", {}).get("hit the cap")
        lost += [f"{name} {int(b) // 1024}k: {s['errors']}" for b, s in sorted(budgets.items(), key=lambda kv: int(kv[0]))
                 if s.get("errors")]
        size = c.get("size_gb") or alloc.get("size_gb") or (model_of(r).stat().st_size / 1e9 if model_of(r).exists() else 0)
        table.append([name, f"{size:.2f} GB", f"{alloc['ffn_bits']:.2f}" if alloc.get("ffn_bits") else ""]
                     + [f"{s:.0%}" if s is not None else "–" for s in scores]
                     + [f"{top / CARD_LCB:.0%}" if top is not None else "",
                        f"{100 * top / size:.1f}" if top is not None and size else "", "" if hit is None else str(hit),
                        f"{tps:.0f}" if tps else ""])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    out = "\n".join("  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(row)).rstrip()
                    for row in table)
    return out + (f"\n\nof card: the 32k score against the card's LiveCodeBench v6 {CARD_LCB:.1%}, which used its own "
                  f"window and budget, so it is approximate. %/GB: the 32k score in points per GB of file. hit 32k: answers whose thinking reached the cap. "
                  f"tok/s: per answer, with {EVAL_SLOTS} running at once and {'a 16-bit' if not KV_TYPE else 'an 8-bit'} KV cache."
                  + (f"\nAnswers the server never returned, scored as failures: {', '.join(lost)}." if lost else ""))


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
