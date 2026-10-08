#!/usr/bin/env python3
"""Experiment 03 on one B200 (or an L40S): Qwen3.8-27B with thinking on, in one command.

    python3 scripts/exp03.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp03.py --list   # the steps and their time guesses
    python3 scripts/exp03.py summary  # the results table

Qwen3.8-27B is dense (no experts), so REAP has nothing to prune and heal nothing to
repair: each code10x recipe comes down to its bits (stages/dense.py).
  q4    r25q4's bits, ~16 GB
  w18s  r50w95s's bits: feed-forward 2-8 bit by saliency averaging ~3.35, the rest
        q8_0, ~17.6 GB

Steps (docs: /mnt/project-files/plan/exp03-qwen38.md):
  1. VM setup (scripts/setup_vm.sh) and the Qwen3.8-27B download
  2. a BF16 GGUF (text only), plus an 8-bit copy on an L40S
  3. calibration text: the model answers MultiPL-E and C problems with thinking on
     (BF16 on a B200, the 8-bit copy on an L40S)
  4. the importance matrix on that text from the BF16 model (on an L40S split
     between GPU and RAM)
  5. per recipe: quantize, LiveCodeBench v6 (2025-01..04) with thinking on and a 64k
     budget, then the 16k and 32k scores by cutting that thinking (stages/budget.py)
  6. both GGUFs and the job results to the mugge-library bucket

There is no reference run: scores are compared with the model card's LiveCodeBench
v6 90.3. Runs with the system python3 (stdlib only); each step sources .env.vm.
Logs are in ~/exp03-logs. The GPU's memory picks the plan (EXP03_VRAM_GB overrides it).
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
LOGS = Path(os.environ.get("EXP03_LOGS", Path.home() / "exp03-logs"))
BUCKET = "bucket://mugge-library"
MODEL = "Qwen/Qwen3.8-27B"
HF_DIR = NVME / "models" / "Qwen3.8-27B"
BF16 = NVME / "models" / "gguf" / "Qwen3.8-27B-BF16.gguf"
Q8 = NVME / "models" / "gguf" / "Qwen3.8-27B-Q8_0.gguf"
BASE = J / "qwen38-base"  # calibration answers and the importance matrix
TASKSPEC = "examples/python-utils.code.taskspec.json"
RECIPES = ["q4", "w18s"]
CARD_LCB = 0.903  # Qwen3.8-27B model card, LiveCodeBench v6
MAX_TOKENS = 65536
BUDGETS = [16384, 32768]
CALIB_SUITES = ["multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "c-set"]


def vram_gb() -> float:
    """The GPU's memory in GiB; EXP03_VRAM_GB overrides it (dry runs)."""
    if os.environ.get("EXP03_VRAM_GB"):
        return float(os.environ["EXP03_VRAM_GB"])
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0]) / 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        sys.exit("nvidia-smi found no GPU. On a new VM the driver may still be installing: "
                 "wait for the reboot, SSH back in and re-run.")


# The plan follows the GPU. A B200 (~180 GB) holds the BF16 model (~54 GB), so the
# model itself writes the calibration answers and the importance pass runs fully on
# the GPU. Scoring keeps the default 16-bit KV cache, ~4.8 GB per answer at 64k, for
# 24 answers at once next to w18s's 17.6 GB. An L40S (48 GB) writes the calibration
# answers with an 8-bit copy, splits the importance pass between GPU and RAM, and
# scores with an 8-bit KV cache (~2.6 GB per answer) to fit 8 at once.
B200 = vram_gb() >= 150
EVAL_SLOTS, CALIB_SLOTS = ("24", "48") if B200 else ("8", "16")
KV_TYPE = "" if B200 else "q8_0"
CALIB_MODEL = BF16 if B200 else Q8
MINUTES = ({"setup": 40, "download": 15, "gguf": 10, "calib": 15, "imatrix": 10, "eval": 90, "budgets": 10}
           if B200 else
           {"setup": 45, "download": 20, "gguf": 25, "calib": 35, "imatrix": 20, "eval": 360, "budgets": 30})
# The thinking settings from the Qwen3.8 card: temperature 1.0, top_p 0.95, top_k 20.
THINKING = {"teacher": MODEL, "code_eval_thinking": True, "code_eval_thinking_temperature": 1.0, "eval_kv_type": KV_TYPE}


def job_dir(recipe: str) -> Path:
    return J / f"qwen38-{recipe}"


def make_job(d: Path, cfg: dict, max_size_gb: float = 20.0) -> Path:
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
    return min(0.99, got / 56e9)  # ~27B in bf16, with the vision tower


def base_job() -> Path:
    if not (BASE / "config.json").exists():
        make_job(BASE, {**THINKING, "code_eval_suites": CALIB_SUITES, "code_eval_max_tokens": 6144})
    return BASE


def run_ggufs(step: Step) -> None:
    base_job()
    q8 = "" if B200 else f" --q8 {Q8}"
    sh(step, f"python -m stages.dense gguf --job {BASE} --hf {HF_DIR} --bf16 {BF16}{q8}")


def run_calib(step: Step) -> None:
    base_job()
    sh(step, f"python -m stages.dense calib --job {BASE} --model {CALIB_MODEL}", {"LOBBOT_EVAL_SLOTS": CALIB_SLOTS})


def run_imatrix(step: Step) -> None:
    base_job()
    sh(step, f"python -m stages.dense imatrix --job {BASE} --bf16 {BF16}")


def run_recipe(recipe: str):
    def run(step: Step) -> None:
        d = job_dir(recipe)
        if not (d / "config.json").exists():
            make_job(d, {**THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
                         "code_eval_max_tokens": MAX_TOKENS, "code_eval_budgets": BUDGETS, "code_eval_ref": "",
                         "eval_candidates": {f"qwen38-{recipe}": str(d / "out" / "model.gguf")}})
        env = {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS}
        if (d / ".done" / "quantize").exists():
            skip_part(step, "quantize")
        else:
            sh(step, f"python -m stages.dense quantize --job {d} --recipe {recipe} --bf16 {BF16} "
                     f"--imatrix {BASE / 'work' / 'imatrix.gguf'}")
        if (d / ".done" / "eval").exists():  # --only always reruns, and this one takes hours
            skip_part(step, "eval")
        else:
            sh(step, f"python pipeline.py --job {d} --only eval", env)
        sh(step, f"python -m stages.budget --job {d}", env)
    return run


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, thinking, scores, the imatrix) and both GGUFs."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 4) and re-run to upload.")
    tar = NVME / "exp03-jobs.tar"
    sh(step, f"mkdir -p {J}/_logs_exp03 && cp {LOGS}/*.log {J}/_logs_exp03/ && cd {J.parent} && "
             f"find {J.name}/qwen38-* {J.name}/_logs_exp03 -type f -size -200M -print0 | tar cf {tar} --null -T -")
    up = [(tar, "exp03/exp03-jobs.tar")]
    for r in RECIPES:
        gguf = job_dir(r) / "out" / "model.gguf"
        if gguf.exists():
            up.append((gguf, f"gguf/qwen38-27b-{r}.gguf"))
    for src, key in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Qwen3.8-27B download", {"": MINUTES["download"]}, run_download, download_done, download_progress),
        Step("BF16 GGUF" if B200 else "BF16 GGUF and 8-bit copy", {"gguf": MINUTES["gguf"]}, run_ggufs,
             CALIB_MODEL.exists),
        Step(f"calibration answers ({'BF16' if B200 else '8-bit copy'})", {"calib": MINUTES["calib"]},
             run_calib, (BASE / "work" / "calib-chat.txt").exists),
        Step(f"importance pass (BF16, {'GPU' if B200 else 'GPU + RAM'})", {"imatrix": MINUTES["imatrix"]},
             run_imatrix, (BASE / "work" / "imatrix.gguf").exists),
        *[Step(r, {"quantize": 5, "eval": MINUTES["eval"], "budgets": MINUTES["budgets"]}, run_recipe(r),
               (job_dir(r) / ".done" / "budgets").exists) for r in RECIPES],
        Step("upload to the bucket", {"": 25}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

def summary() -> str:
    """LiveCodeBench pass@1 at each thinking budget, and the 64k score as a share of the card."""
    head = ["model", "size", "FFN bits"] + [f"LCB {b // 1024}k" for b in BUDGETS + [MAX_TOKENS]] \
        + ["of card", "hit 64k", "tok/s"]
    table, lost = [head], []
    for r in RECIPES:
        d, name = job_dir(r), f"qwen38-{r}"
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
        table.append([name, f"{c.get('size_gb') or alloc.get('size_gb') or 0:.1f} GB", f"{alloc['ffn_bits']:.2f}" if alloc.get("ffn_bits") else ""]
                     + [f"{s:.0%}" if s is not None else "–" for s in scores]
                     + [f"{top / CARD_LCB:.0%}" if top is not None else "", "" if hit is None else str(hit),
                        f"{tps:.0f}" if tps else ""])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    out = "\n".join("  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(row)).rstrip()
                    for row in table)
    return out + (f"\n\nof card: the 64k score against the card's LiveCodeBench v6 {CARD_LCB:.1%}, which used its own "
                  f"window and budget, so it is approximate. hit 64k: answers whose thinking reached the cap. "
                  f"tok/s: per answer, with {EVAL_SLOTS} running at once and {'a 16-bit' if not KV_TYPE else 'an 8-bit'} KV cache."
                  + (f"\nAnswers the server never returned, scored as failures: {', '.join(lost)}." if lost else ""))


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
