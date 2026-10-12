#!/usr/bin/env python3
"""Blackhole gate 1 on one B200: Granite 4.2 3B trained into ternary weights, in one command.

    python3 scripts/blackhole.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/blackhole.py --list   # the steps and their time guesses
    python3 scripts/blackhole.py summary  # the results table

Goal (plan: /mnt/project-files/plan/blackhole.md): 10x smaller than the BF16 file
(~0.73 GB, ternary layers in llama.cpp's TQ1_0 after a vocab trim) while keeping 95%
of the original's LiveCodeBench score. Gate 1 asks whether the training works at all,
on a small budget, before more is spent:

  1. VM setup and the Granite download
  2. at once: the original scored on LiveCodeBench v6 (2025-01..04, thinking on, 16k,
     vLLM) and the original's own thinking answers to clean CodeContests problems
     (~40M tokens, never LiveCodeBench) as training text
  3. KL of a plain ternary snap, no training (the floor)
  4. layer by layer: each ternary layer trained to reproduce the full one (~4M tokens x 2)
  5. whole model: the ternary model distilled from the full one (~60M tokens, capped)
  6. the ternary model scored like the original, and the share it keeps
  7. results to the mugge-library bucket

Everything trains and serves on the one GPU; scoring uses vLLM on the snapped weights,
which are exactly the ternary model. Logs are in ~/blackhole-logs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import exp05 as e5
from runner import Skip, Step, evroc_ok, read, run_plan, sh

LOGS = Path(os.environ.get("BLACKHOLE_LOGS", Path.home() / "blackhole-logs"))
JOB = e5.J / "blackhole"
W = JOB / "work"
TRACES = W / "traces.jsonl"
BASE_JOB, FINAL_JOB = e5.J / "blackhole-base", e5.J / "blackhole-final"
GEN_PORT = "8093"
GEN_TOKENS = 40e6
E2E_TOKENS, E2E_CAP_MINUTES = 60e6, 75
BLOCK_WINDOWS = 2048  # x 2048 tokens
LCB_TOKENS = 16384
MINUTES = {"setup": 40, "download": 3, "phase1": 35, "snap": 5, "block": 15, "e2e": E2E_CAP_MINUTES + 10, "score": 25}


def eval_job(d: Path, name: str, model: Path) -> Path:
    if not (d / "config.json").exists():
        e5.make_job(d, {**e5.THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
                        "code_eval_max_tokens": LCB_TOKENS, "code_eval_ref": "", "eval_candidates": {name: str(model)}})
    return d


def score(step: Step, d: Path, env: dict | None = None) -> None:
    if not (d / ".done" / "eval").exists():
        sh(step, f"python pipeline.py --job {d} --only eval", env)


def gen_command() -> str:
    """A vLLM on half the GPU for the original's answers, then gen; the server stops after."""
    return "\n".join([
        "set -e",
        f"mkdir -p {W}",
        f"$LOBBOT_VLLM_PY -m vllm.entrypoints.openai.api_server --model {e5.HF_DIR} --served-model-name student "
        f"--host 127.0.0.1 --port {GEN_PORT} --max-model-len 12288 --gpu-memory-utilization 0.45 "
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


def run_phase1(step: Step) -> None:
    """The original scored (vLLM on half the GPU) while a second vLLM writes the training text."""
    d = eval_job(BASE_JOB, "granite-bf16", e5.HF_DIR)

    def write_text() -> None:
        if not (W / ".gen-done").exists():
            sh(step, gen_command())
            W.mkdir(parents=True, exist_ok=True)
            (W / ".gen-done").write_text("ok")

    e5.at_once(lambda: score(step, d, {"LOBBOT_VLLM_GPU_UTIL": "0.45"}), write_text)


def phase1_done() -> bool:
    return (BASE_JOB / ".done" / "eval").exists() and (W / ".gen-done").exists()


def run_snap(step: Step) -> None:
    sh(step, f"python -m stages.ternary kl --student {e5.HF_DIR} --teacher {e5.HF_DIR} --snap "
             f"--traces {TRACES} --out {W}/kl-snap.json")


def run_block(step: Step) -> None:
    sh(step, f"python -m stages.ternary block --base {e5.HF_DIR} --traces {TRACES} --windows {BLOCK_WINDOWS} "
             f"--out {W}/block")


def run_e2e(step: Step) -> None:
    sh(step, f"python -m stages.ternary e2e --student {W}/block/latent --teacher {e5.HF_DIR} --traces {TRACES} "
             f"--tokens {E2E_TOKENS:.0f} --max-minutes {E2E_CAP_MINUTES} --out {W}/e2e")


def run_score(step: Step) -> None:
    score(step, eval_job(FINAL_JOB, "blackhole-g1", W / "e2e" / "snapped"))


def run_upload(step: Step) -> None:
    """Configs, logs, KL curves, scores and answers (not the traces or weights)."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 5) and re-run to upload.")
    tar = e5.NVME / "blackhole-g1.tar"
    sh(step, f"mkdir -p {JOB}/logs && cp {LOGS}/*.log {JOB}/logs/ && cd {e5.J.parent} && "
             f"find {e5.J.name}/blackhole {e5.J.name}/blackhole-base {e5.J.name}/blackhole-final -type f -size -200M "
             f"! -name traces.jsonl ! -name '*.safetensors' -print0 | tar cf {tar} --null -T - && "
             f"evroc storage bucket get-s3-credentials >/dev/null && "
             f"evroc storage bucket copy --from {tar} --to {e5.BUCKET}/blackhole/blackhole-g1.tar")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def gen_progress() -> float | None:
    if not TRACES.exists():
        return None
    n = sum(len(json.loads(l)["tokens"]) for l in TRACES.read_text().splitlines() if l.strip())
    return min(0.99, n / GEN_TOKENS)


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Granite 4.2 3B download", {"": MINUTES["download"]}, e5.run_download, e5.download_done, e5.download_progress),
        Step("original scored; training text written", {"": MINUTES["phase1"]}, run_phase1, phase1_done, gen_progress),
        Step("plain ternary snap (no training)", {"": MINUTES["snap"]}, run_snap, (W / "kl-snap.json").exists),
        Step("layer by layer", {"": MINUTES["block"]}, run_block, (W / "block" / "block.json").exists),
        Step("whole model", {"": MINUTES["e2e"]}, run_e2e, (W / "e2e" / "e2e.json").exists),
        Step("ternary model scored", {"": MINUTES["score"]}, run_score, (FINAL_JOB / ".done" / "eval").exists),
        Step("upload to the bucket", {"": 10}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

def lcb(d: Path) -> float | None:
    p = d / "out" / "eval.json"
    if not p.exists():
        return None
    c = read(p)["candidates"][0]
    return (((c.get("code") or {}).get("suites") or {}).get("livecodebench") or {}).get("pass@1")


def summary() -> str:
    base, final = lcb(BASE_JOB), lcb(FINAL_JOB)
    snap = read(W / "kl-snap.json")["kl"] if (W / "kl-snap.json").exists() else None
    e2e = read(W / "e2e" / "e2e.json") if (W / "e2e" / "e2e.json").exists() else {}
    kls = e2e.get("kl") or []
    f = lambda x, p=4: "–" if x is None else f"{x:.{p}f}"  # noqa: E731
    out = [f"LiveCodeBench v6 (2025-01..04, 16k thinking): original {f(base and 100 * base, 1)}, "
           f"ternary {f(final and 100 * final, 1)}"
           + (f", kept {final / base:.0%}" if base and final is not None else ""),
           f"Held-out KL to the original: plain snap {f(snap)}, after layer by layer {f(kls[0]['kl'] if kls else None)}, "
           f"after whole model {f(kls[-1]['kl'] if kls else None)} ({(e2e.get('tokens') or 0) / 1e6:.0f}M tokens)"]
    if len(kls) > 2:
        out.append("KL curve: " + ", ".join(f"{h['step']}:{h['kl']:.3f}" for h in kls[:: max(1, len(kls) // 8)]))
    if e2e:
        gb = e2e["size_gb_vocab24k"]
        out.append(f"Size as TQ1_0: {e2e['size_gb']:.2f} GB, {gb:.2f} GB after a vocab trim to 24k "
                   f"({7.32 / gb:.1f}x smaller than BF16)"
                   + (f"; {100 * final / gb:.0f} LCB points per GB" if final is not None else ""))
    out.append("Gate 1 goes on when the ternary model scores clearly above 0 and the KL curve is still falling; "
               "the 95% target is for the full run.")
    return "\n".join(out)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
