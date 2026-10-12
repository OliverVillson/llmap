#!/usr/bin/env python3
"""Experiment 06 on two L40S: a short on-policy distillation (OPD) of Granite 4.2 3B
from Granite 4.2 30B, as a cheap check of how much the teacher can add.

    python3 scripts/exp06.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp06.py --list   # the steps and their time guesses
    python3 scripts/exp06.py summary  # the results table

Both models share one tokenizer (vocab 100352, same special ids; configs checked), so
the 30B can score every token the 3B writes (stages/opd.py). The 3B thinks on clean
CodeContests problems (deduped against LiveCodeBench), the 30B scores its tokens, the
3B takes one LoRA step per round toward it. Then the result is scored on LiveCodeBench
exactly like exp05's bf16 (16k thinking, 8k cut), so the two compare directly; run
exp05 first on the same VM (its steps are reused when done).

GPU 0: the 30B teacher in vLLM (FP8 weights, ~30 GB). GPU 1: the 3B in vLLM for the
answers (with runtime LoRA loading) and the trainer beside it. Plan:
/mnt/project-files/plan/granite-record.md. Logs are in ~/exp06-logs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import exp05 as e5
from runner import Skip, Step, evroc_ok, read, run_plan, sh

LOGS = Path(os.environ.get("EXP06_LOGS", Path.home() / "exp06-logs"))
TEACHER = "ibm-granite/granite-4.2-30b"
TEACHER_DIR = e5.NVME / "models" / "granite-4.2-30b"
JOB = e5.J / "granite-opd"
OPD_DIR = JOB / "work" / "opd"
OPD_GGUF = e5.NVME / "models" / "gguf" / "granite-4.2-3b-opd-BF16.gguf"
STUDENT_PORT, TEACHER_PORT = "8093", "8094"
# 20 rounds of 16 problems x 4 answers (1280 answers) at the card's 8k cap: enough to
# see whether the KL falls and the pass rate moves, capped so a slow round can't run up the bill.
ROUNDS, PER_ROUND, SAMPLES, ANSWER_TOKENS, CAP_MINUTES = 20, 16, 4, 8192, 90
CTX = 12288  # prompt + 8k answer
MINUTES = {"setup": 45, "download": 15, "opd": CAP_MINUTES + 15, "gguf": 5, "score": 35}


def vllm(gpu: int, model: Path, name: str, port: str, util: float, args: str = "", env: str = "") -> str:
    """A background vLLM on one GPU, its pid in $<name>; its log in the job."""
    return (f"CUDA_VISIBLE_DEVICES={gpu} {env}{' ' if env else ''}$LOBBOT_VLLM_PY -m vllm.entrypoints.openai.api_server "
            f"--model {model} --served-model-name {name} --host 127.0.0.1 --port {port} "
            f"--max-model-len {CTX} --gpu-memory-utilization {util} --enable-prefix-caching {args} "
            f"> {JOB}/work/vllm-{name}.log 2>&1 & {name}=$!")


def wait_up(name: str, port: str) -> str:
    return (f"until curl -sf http://127.0.0.1:{port}/health >/dev/null; do "
            f"kill -0 ${name} 2>/dev/null || {{ tail -20 {JOB}/work/vllm-{name}.log; exit 1; }}; sleep 5; done")


def opd_command() -> str:
    """Both servers, then the trainer on GPU 1; the servers stop when it ends. One
    command per line: `a && b &` would background the whole list in a subshell."""
    return "\n".join([
        "set -e",
        f"mkdir -p {JOB}/work",
        # FP8 weights keep the 30B at ~30 GB; its KV cache stays 16-bit so its scores are exact
        vllm(0, TEACHER_DIR, "teacher", TEACHER_PORT, 0.92, "--quantization fp8"),
        # half of GPU 1 for answers (8-bit KV, ~20 at once at 8k), half for the trainer
        vllm(1, e5.HF_DIR, "student", STUDENT_PORT, 0.5, "--enable-lora --max-lora-rank 32 --kv-cache-dtype fp8",
             "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True"),
        "trap 'kill $teacher $student 2>/dev/null' EXIT",
        wait_up("teacher", TEACHER_PORT), wait_up("student", STUDENT_PORT),
        f"CUDA_VISIBLE_DEVICES=1 python -m stages.opd --student {e5.HF_DIR} "
        f"--student-url http://127.0.0.1:{STUDENT_PORT} --teacher-url http://127.0.0.1:{TEACHER_PORT} "
        f"--problems {e5.NVME}/codebench --out {OPD_DIR} --rounds {ROUNDS} --problems-per-round {PER_ROUND} "
        f"--samples {SAMPLES} --max-tokens {ANSWER_TOKENS} --max-minutes {CAP_MINUTES}",
    ])


# ---------- the steps ----------

def run_setup(step: Step) -> None:
    sh(step, f"NVME={e5.NVME} TEACHER={TEACHER} STUDENT={e5.MODEL} bash scripts/setup_vm.sh")


def setup_done() -> bool:
    return e5.setup_done() and (e5.NVME / "codebench/codecontests.jsonl").exists()


def download_done() -> bool:
    return e5.download_done() and (TEACHER_DIR / "config.json").exists()


def run_download(step: Step) -> None:
    if not download_done():
        e5.run_download(step)
    if not download_done():
        raise RuntimeError(f"{TEACHER} is missing from {TEACHER_DIR}; re-run scripts/setup_vm.sh "
                           f"with TEACHER={TEACHER} STUDENT={e5.MODEL}")


def run_opd(step: Step) -> None:
    sh(step, opd_command())


def run_gguf(step: Step) -> None:
    e5.base_job()
    sh(step, f"python -m stages.dense gguf --job {e5.BASE} --hf {OPD_DIR} --bf16 {OPD_GGUF}")


def score_job() -> Path:
    if not (JOB / "config.json").exists():
        e5.make_job(JOB, {**e5.THINKING, "code_eval_suites": ["livecodebench"], "lcb_since": "2025-01-01",
                          "code_eval_max_tokens": e5.MAX_TOKENS, "code_eval_budgets": e5.BUDGETS,
                          "code_eval_ref": "", "eval_candidates": {"granite-opd": str(OPD_GGUF)}})
    return JOB


def run_score(step: Step) -> None:
    d, env = score_job(), {**e5.env_for(e5.PORTS[0]), "CUDA_VISIBLE_DEVICES": "0"}
    if not (d / ".done" / "eval").exists():
        sh(step, f"python pipeline.py --job {d} --only eval", env)
    sh(step, f"python -m stages.budget --job {d}", env)


def run_upload(step: Step) -> None:
    """The job files under 200 MB (the OPD log, answers, scores) and the logs; the model
    itself stays on the VM until it beats exp05's bf16."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step 5) and re-run to upload.")
    tar = e5.NVME / "exp06-jobs.tar"
    sh(step, f"mkdir -p {JOB}/logs && cp {LOGS}/*.log {JOB}/logs/ && cd {e5.J.parent} && "
             f"find {e5.J.name}/granite-opd -type f -size -200M -print0 | tar cf {tar} --null -T - && "
             f"evroc storage bucket get-s3-credentials >/dev/null && "
             f"evroc storage bucket copy --from {tar} --to {e5.BUCKET}/exp06/exp06-jobs.tar")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": MINUTES["setup"]}, run_setup, setup_done),
        Step("Granite 3B and 30B download", {"": MINUTES["download"]}, run_download, download_done),
        Step("OPD from the 30B", {"": MINUTES["opd"]}, run_opd, lambda: (OPD_DIR / "config.json").exists(), opd_progress),
        Step("BF16 GGUF of the result", {"gguf": MINUTES["gguf"]}, run_gguf, OPD_GGUF.exists),
        Step("scored on LiveCodeBench", {"": MINUTES["score"]}, run_score, lambda: (JOB / ".done" / "budgets").exists()),
        Step("upload to the bucket", {"": 10}, run_upload, (LOGS / ".uploaded").exists),
    ]


def opd_rounds() -> list[dict]:
    p = JOB / "work" / "opd-log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def opd_progress() -> float | None:
    rows = opd_rounds()
    return min(0.99, len(rows) / ROUNDS) if rows else None


# ---------- summary ----------

def lcb(job: Path, name: str) -> list[float | None]:
    """[8k score, 16k score] of the job's candidate, or Nones when it hasn't been scored."""
    p = job / "out" / "eval.json"
    if not p.exists():
        return [None, None]
    rep = read(p)
    c = next((x for x in rep["candidates"] if x["name"] == name), rep["candidates"][0])
    code = c.get("code") or {}
    cut = ((code.get("budgets") or {}).get(str(e5.BUDGETS[0])) or {})
    return [((cut.get("suites") or {}).get("livecodebench") or {}).get("pass@1", cut.get("mean_pass@1")),
            ((code.get("suites") or {}).get("livecodebench") or {}).get("pass@1")]


def summary() -> str:
    pct = lambda x: "–" if x is None else f"{x:.1%}"  # noqa: E731
    rows = [("granite-bf16 (exp05)", lcb(e5.job_dir("bf16"), "granite-bf16")), ("granite-opd", lcb(JOB, "granite-opd"))]
    out = ["model                 LCB 8k   LCB 16k"] + [f"{n:20}  {pct(a):>6}  {pct(b):>8}" for n, (a, b) in rows]
    (_, base16), (_, opd16) = rows[0][1], rows[1][1]
    if base16 is not None and opd16 is not None:
        out.append(f"\nOPD changed the 16k score by {100 * (opd16 - base16):+.1f} points.")
    r = opd_rounds()
    if r:
        first, last = r[0], r[-1]
        out.append(f"OPD: {len(r)} rounds; reverse KL per token {first['kl_per_token']:.3f} -> {last['kl_per_token']:.3f}; "
                   f"rollout pass rate {first['pass']:.0%} -> {last['pass']:.0%} (CodeContests, 8k, noisy); "
                   f"cut at 8k {first['cut']:.0%} -> {last['cut']:.0%}.")
    return "\n".join(out)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
