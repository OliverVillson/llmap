#!/usr/bin/env python3
"""Experiment 02 on the B200 in one command, with a progress bar and the time left.

    python3 scripts/exp02.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp02.py --list   # the steps and their time guesses
    python3 scripts/exp02.py summary  # the results table

Steps (docs: /mnt/project-files/plan/b200-exp02-thinking.md):
  1. VM setup (scripts/setup_vm.sh), experiment 01's files from the mugge-library
     bucket, the base model download, the code10x jobs and the BF16 GGUF
  2. ref-think: the full model with thinking on, the 100% mark for r50w95s-t. If its
     LiveCodeBench score is far below the model card's, the thinking eval is suspect
     and r50w95s-t is skipped
  3. side tests, thinking off, against r50w95s: 6 routed experts per token instead
     of 8 (r50w95s-k6 vs r50w95s-k8), and REAP keeping 37.5% of the experts at 9.5 GB
     and 7.5 GB (r62w95s, r62w75s), healed on experiment 01's r50mix data. Each also
     scores one fix turn (fix@1, Mugge's write-check-fix loop)
  4. Mugge's own loop: r50w95s asked the way Mugge asks (r50w95s-mugge), and the same
     recipe healed on Mugge-shaped data (r50w95s-h, stages/harness.py). EXP02_MUGGE=0
     leaves it out.
  5. r50w95s-t, the thinking model
  6. the r50w95s-t GGUF and the job results to the bucket

Runs with the system python3 (stdlib only); each step sources .env.vm. Logs are in
~/exp02-logs. The bucket steps need an evroc login first; without one they are
skipped and a re-run picks them up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from runner import REPO, Skip, Step, above, evroc_ok, read, run_plan, sh, write

NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
J = Path(os.environ.get("LOBBOT_JOBS", NVME / "jobs"))
M = NVME / "exp01"  # experiment 01's files from the bucket
LOGS = Path(os.environ.get("EXP02_LOGS", Path.home() / "exp02-logs"))
BUCKET = "bucket://mugge-library"
TEACHER = "Qwen/Qwen3.6-35B-A3B"
TASKSPEC = "examples/python-utils.code.taskspec.json"
SIDE = ["r50w95s-k8", "r50w95s-k6", "r62w95s", "r62w75s"]
# The model card says 80.4 with thinking on and exp01's ref got 45% with it off, so
# a score near the thinking-off one means the thinking eval is broken.
LCB_FLOOR = 0.55
# Step 4 (Mugge's own loop); EXP02_MUGGE=0 leaves it out.
WITH_MUGGE = os.environ.get("EXP02_MUGGE", "1") == "1"


# ---------- job helpers ----------


def new_job(name: str, max_size_gb: float, data: bool = True, **cfg) -> Path:
    """A job with experiment 01 r50mix's config, taskspec and finished data stage, plus cfg.
    Every side test also scores one fix turn (fix@1): if the harness loop wins back
    what compression loses, we can compress harder. data=False makes its own data from
    the same tasks (r50mix's cached task list, when the bucket tar has it)."""
    d, src = J / f"code10x-{name}", M / "jobs" / "code10x-r50mix"
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if not data:
        for f in ("data_scenarios.json", "data_inputs.jsonl"):
            if (src / "work" / f).exists() and not (d / "work" / f).exists():
                shutil.copy(src / "work" / f, d / "work" / f)
    elif not (d / ".done" / "data").exists():
        shutil.copytree(src / "data", d / "data", dirs_exist_ok=True)
        shutil.copy(src / ".done" / "data", d / ".done" / "data")
    write(d / "config.json", {**read(src / "config.json"), "code_eval_fix": True, **cfg})
    spec = read(src / "taskspec.json")
    spec["target"] = {**spec.get("target", {}), "max_size_gb": max_size_gb}
    write(d / "taskspec.json", spec)
    ref = J / "code10x-ref" / "out" / "eval.json"  # exp01's ref (thinking off) is the 100% mark
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(M / "jobs" / "code10x-ref" / "out" / "eval.json", ref)
    return d


def eval_json(name: str) -> dict | None:
    p = J / f"code10x-{name}" / "out" / "eval.json"
    return read(p) if p.exists() else None


def pick(rep: dict, name: str) -> dict:
    cands = {c["name"]: c for c in rep.get("candidates", [])}
    return cands.get(name) or cands.get("lobbot-moe") or cands.get("r50w95s") or cands.get("ref") or rep["candidates"][0]


def lcb(name: str) -> float | None:
    rep = eval_json(name)
    if not rep:
        return None
    return (((pick(rep, name).get("code") or {}).get("suites") or {}).get("livecodebench") or {}).get("pass@1")


# ---------- the steps ----------

def run_setup(step: Step) -> None:
    sh(step, f"NVME={NVME} TEACHER={TEACHER} bash scripts/setup_vm.sh")


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and (NVME / "llama.cpp/build/bin/llama-server").exists()
            and (NVME / "codebench/livecodebench.jsonl").exists())


EXP01_FILES = {"exp01/exp01-jobs.tar": 2.2e9, "gguf/qwen36-r50w95s.gguf": 9.3e9}


def run_fetch(step: Step) -> None:
    if not evroc_ok():
        raise Skip("no evroc login, so the side tests are skipped. Log in (runbook step D) and re-run.")
    M.mkdir(parents=True, exist_ok=True)
    for key in EXP01_FILES:
        dst = M / Path(key).name
        if not dst.exists() or dst.stat().st_size == 0:
            sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                     f"evroc storage bucket copy --from {BUCKET}/{key} --to {dst}.part && mv {dst}.part {dst}")
    sh(step, f"tar xf {M}/exp01-jobs.tar -C {M} jobs/code10x-r50mix jobs/code10x-ref")


def fetch_done() -> bool:
    return (M / "qwen36-r50w95s.gguf").exists() and (M / "jobs/code10x-r50mix/.done/data").exists()


def fetch_progress() -> float:
    got = 0
    for key in EXP01_FILES:
        for f in (M / Path(key).name, M / (Path(key).name + ".part")):
            got += f.stat().st_size if f.exists() else 0
    return min(0.99, got / sum(EXP01_FILES.values()))


def run_download(step: Step) -> None:
    log = NVME / "download.log"
    while not (log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace")):
        if subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL).returncode != 0:
            time.sleep(10)  # between the two downloads
            if subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL).returncode != 0 and \
                    "DOWNLOADS_DONE" not in log.read_text(errors="replace"):
                raise RuntimeError(f"the model download stopped; see {log} and re-run scripts/setup_vm.sh")
        time.sleep(5)


def download_done() -> bool:
    log = NVME / "download.log"
    return log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace")


def download_progress() -> float:
    d = NVME / "models"
    got = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    return min(0.99, got / 86e9)  # the base (~70 GB) and the unused dense student (~16 GB)


# Answers at once in every eval. 16 left the B200 mostly idle (exp03: 46% busy at
# 24). These models are MoE with a small KV cache, ~0.9 GB per answer at the 32k
# thinking budget, so 64 fit next to even the 70 GB BF16 ref.
EVAL_SLOTS = "64"

def run_jobs(step: Step) -> None:
    sh(step, f"python scripts/code10x.py setup --jobs {J} --taskspec {TASKSPEC} && python scripts/code10x.py ref-ggufs")


def jobs_done() -> bool:
    return (J / "code10x-r50w95s-t/config.json").exists() and (NVME / "models/gguf/Qwen3.6-35B-A3B-BF16.gguf").exists()


def eval_done(name: str) -> Callable[[], bool]:
    return lambda: (J / f"code10x-{name}" / ".done" / "eval").exists()


def run_ref_think(step: Step) -> None:
    sh(step, f"python pipeline.py --job {J}/code10x-ref-think --only eval", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})
    score = lcb("ref-think")
    above(f"    ref-think LiveCodeBench pass@1: {score:.0%}" if score is not None else "    ref-think has no LiveCodeBench score")


def needs_exp01() -> None:
    if not fetch_done():
        raise Skip("experiment 01's files are not here (no evroc login when they were fetched); re-run after logging in.")


def run_experts(k: int) -> Callable[[Step], None]:
    def run(step: Step) -> None:
        needs_exp01()
        name = f"r50w95s-k{k}"
        new_job(name, 9.5, eval_candidates={"r50w95s": str(M / "qwen36-r50w95s.gguf")},
                eval_experts_used=0 if k == 8 else k)
        sh(step, f"python pipeline.py --job {J}/code10x-{name} --only eval", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})
    return run


def run_r62w95s(step: Step) -> None:
    needs_exp01()
    d = new_job("r62w95s", 9.5, reap_sparsity=0.625, bit_floor="q2_k", bit_ceiling="q8_0", static_type="q8_0")
    sh(step, f"python pipeline.py --job {d}", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})


def run_r62w75s(step: Step) -> None:
    """r62w95s's pruned and healed model, quantized to 7.5 GB."""
    src, dst = J / "code10x-r62w95s", J / "code10x-r62w75s"
    if not (src / ".done" / "heal").exists():
        raise Skip("r62w95s has not healed, so there is nothing to quantize")
    (dst / "work").mkdir(parents=True, exist_ok=True)
    for sub in ("data", ".done"):
        shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    for f in ("config.json", "taskspec.json"):
        shutil.copy(src / f, dst / f)
    for f in (src / "work").glob("*.json"):
        shutil.copy(f, dst / "work" / f.name)
    if not (dst / "work" / "healed").exists():
        (dst / "work" / "healed").symlink_to(src / "work" / "healed")
    spec = read(dst / "taskspec.json")
    spec["target"]["max_size_gb"] = 7.5
    write(dst / "taskspec.json", spec)
    sh(step, f"python pipeline.py --job {dst} --from quantize", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})


def run_mugge_baseline(step: Step) -> None:
    """r50w95s asked the way Mugge asks (tickets in, files out, Mugge's fix call)."""
    needs_exp01()
    new_job("r50w95s-mugge", 9.5, eval_candidates={"r50w95s": str(M / "qwen36-r50w95s.gguf")},
            code_eval_format="harness", code_eval_ref="")
    sh(step, f"python pipeline.py --job {J}/code10x-r50w95s-mugge --only eval", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})


def run_mugge_model(step: Step) -> None:
    """r50w95s's recipe healed on Mugge-shaped data: half the rows as write calls, plus
    600 fix rows; scored the same way as r50w95s-mugge."""
    needs_exp01()
    d = new_job("r50w95s-h", 9.5, data=False, data_harness_share=0.5, data_fix_rows=600, bit_floor="q2_k",
                bit_ceiling="q8_0", static_type="q8_0", code_eval_format="harness", code_eval_ref="")
    sh(step, f"python pipeline.py --job {d}", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})
    stats = d / "data" / "stats.json"
    if stats.exists():
        st = read(stats)
        above(f"    data: {st.get('harness_write_rows')} write rows, {st.get('fix_rows')} fix rows "
              f"from {st.get('fix_pool')} drafts, dropped {st.get('fix_dropped')}")


def run_thinking_model(step: Step) -> None:
    score = lcb("ref-think")
    if score is None:
        raise Skip("ref-think has no score yet")
    if score < LCB_FLOOR:
        raise Skip(f"ref-think scored {score:.0%} on LiveCodeBench, far below the model card's 80%, "
                   "so the thinking eval looks broken. Send Claude the summary before spending hours here.")
    sh(step, f"python pipeline.py --job {J}/code10x-r50w95s-t", {"LOBBOT_EVAL_SLOTS": EVAL_SLOTS})
    stats = J / "code10x-r50w95s-t/data/stats.json"
    if stats.exists():
        s = read(stats)
        above(f"    data: {s.get('answered')} answered, thinking median {s.get('thinking_chars_median')} chars, "
              f"dropped {s.get('dropped')}")


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, data, scores) and the r50w95s-t and r50w95s-h GGUFs."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step D) and re-run to upload.")
    tar = NVME / "exp02-jobs.tar"
    sh(step, f"mkdir -p {J}/_logs_exp02 && cp {LOGS}/*.log {J}/_logs_exp02/ && cd {J.parent} && "
             f"find {J.name} -type f -size -200M -print0 | tar cf {tar} --null -T -")
    up = [(tar, "exp02/exp02-jobs.tar")]
    for name in ("r50w95s-t", "r50w95s-h"):
        gguf = J / f"code10x-{name}/out/model.gguf"
        if gguf.exists():
            up.append((gguf, f"gguf/qwen36-{name}.gguf"))
    for src, key in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": 40}, run_setup, setup_done),
        Step("experiment 01 files from the bucket", {"": 10}, run_fetch, fetch_done, fetch_progress),
        Step("base model download", {"": 15}, run_download, download_done, download_progress),
        Step("jobs and the BF16 GGUF", {"": 15}, run_jobs, jobs_done),
        Step("ref-think eval", {"eval": 100}, run_ref_think, eval_done("ref-think")),
        Step("r50w95s-k8 eval", {"eval": 20}, run_experts(8), eval_done("r50w95s-k8")),
        Step("r50w95s-k6 eval", {"eval": 20}, run_experts(6), eval_done("r50w95s-k6")),
        Step("r62w95s", {"reap": 10, "heal": 30, "quantize": 10, "eval": 15, "package": 1}, run_r62w95s, eval_done("r62w95s")),
        Step("r62w75s", {"quantize": 10, "eval": 15, "package": 1}, run_r62w75s, eval_done("r62w75s")),
        *([Step("r50w95s-mugge eval", {"eval": 25}, run_mugge_baseline, eval_done("r50w95s-mugge")),
           Step("r50w95s-h", {"data": 45, "reap": 10, "heal": 30, "quantize": 10, "eval": 25, "package": 1},
                run_mugge_model, eval_done("r50w95s-h"))] if WITH_MUGGE else []),
        Step("r50w95s-t", {"data": 90, "reap": 25, "heal": 180, "quantize": 20, "eval": 100, "package": 2},
             run_thinking_model, lambda: (J / "code10x-r50w95s-t/.done/package").exists()),
        Step("upload to the bucket", {"": 15}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

SHORT = {"multipl-e-py": "py", "multipl-e-js": "js", "multipl-e-ts": "ts", "multipl-e-cpp": "cpp",
         "c-set": "C", "livecodebench": "LCB", "heldout": "held"}


def summary() -> str:
    """pass@1 per suite, and the mean share of the reference (and its lowest suite)."""
    head = ["model", "size"] + list(SHORT.values()) + ["mean", "+1 fix", "vs", "of ref", "lowest", "tok/s"]
    table = [head]
    for name, ref in [("ref-think", ""), ("r50w95s-t", "ref-think"), ("r50w95s-k8", "ref"),
                      ("r50w95s-k6", "ref"), ("r62w95s", "ref"), ("r62w75s", "ref"),
                      *([("r50w95s-mugge", ""), ("r50w95s-h", "")] if WITH_MUGGE else [])]:
        rep = eval_json(name)
        if not rep:
            table.append([name, "not run"] + [""] * (len(head) - 2))
            continue
        c = pick(rep, name)
        code = c.get("code") or {}
        suites = code.get("suites", {})
        share = code.get("share_of_ref") or {}
        tps = c.get("tok_s_vm")
        table.append([name, f"{c.get('size_gb', 0):.1f} GB"]
                     + [f"{suites[k]['pass@1']:.0%}" if "pass@1" in suites.get(k, {}) else "–" for k in SHORT]
                     + [f"{code['mean_pass@1']:.0%}" if code.get("mean_pass@1") is not None else "",
                        f"{code['mean_fix@1']:.0%}" if code.get("mean_fix@1") is not None else "",
                        ref if share else "", f"{share['mean']:.0%}" if share else "",
                        f"{share['min']:.0%}" if share else "", f"{tps:.0f}" if tps else ""])
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    return "\n".join("  ".join(x.ljust(widths[i]) if i in (0, len(SHORT) + 4) else x.rjust(widths[i])
                               for i, x in enumerate(r)).rstrip() for r in table)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
