#!/usr/bin/env python3
"""Experiment 04 on the B200 in one command: r59-t, the next Mugge model.

    python3 scripts/exp04.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp04.py --list   # the steps and their time guesses
    python3 scripts/exp04.py summary  # the results table

The bar (Oliver, 2026-10-10): a file under 10 GB, above 65% on LiveCodeBench with
thinking on, and a build faster than r50w95s-t's. Served by vLLM on the GPU, so the
experts are 4-bit (vLLM's MoE kernels go no lower) and fewer experts are kept than
r50w95s's 128: about 104 of 256, int4 experts, FP8 DeltaNet / attention / output
head, BF16 embeddings and router. Plan: /mnt/project-files/plan/next-model.md.

Steps:
  1. VM setup (scripts/setup_vm.sh), with the code suites re-fetched when they
     predate the per-case time limit
  2. the base model download
  3. smoke test (scripts/exp04_smoke.py): a 4-layer copy of the base with random
     weights goes through the real quantize stage and is served by vLLM, once with
     FP8 DeltaNet / attention and, if that fails, with them in BF16. That picks the
     format, and the format and the 10 GB budget pick the expert count
  4. the full model scored on vLLM: LiveCodeBench (2025-01..04, thinking on, 32k,
     4 answers per problem; pass@1 is their mean), then the other code suites once
     each with one fix turn. These are the 100% marks
  5. r59-t: contest + Python thinking data, REAP to the smoke test's expert count,
     heal capped at 45 min, int4 + FP8 quantize with a size check, LiveCodeBench x4
  6. r59-t on the other suites (one fix turn), and in Mugge's format
  7. side scores on LiveCodeBench x4: r59-t before quantizing (pruning loss vs
     4-bit loss), and with 6 experts per token instead of 8 (speed)
  8. throughput: 1, 32 and 128 requests at once
  9. a GGUF for the Mac from the same healed model (EXP04_GGUF=0 leaves it out)
 10. the model, the GGUF and the job results to the bucket

Runs with the system python3 (stdlib only); each step sources .env.vm. Logs are in
~/exp04-logs. The upload needs an evroc login first; without one it is skipped and
a re-run picks it up.
"""

from __future__ import annotations

import json
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
LOGS = Path(os.environ.get("EXP04_LOGS", Path.home() / "exp04-logs"))
BUCKET = "bucket://mugge-library"
TEACHER = "Qwen/Qwen3.6-35B-A3B"
TEACHER_DIR = NVME / "models" / TEACHER.split("/")[-1]
TASKSPEC = "examples/python-utils.code.taskspec.json"
SMOKE = NVME / "exp04-smoke"
CONFIGS = REPO / "configs" / "exp04"
# Oliver's bar is 10 GB. The expert count is chosen so the estimate fits BUDGET_GB;
# the quantize stage refuses a file over MAX_GB.
BUDGET_GB = 9.8
MAX_GB = 9.95
BAR = 0.65
SUITES = ["multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "c-set"]
WITH_GGUF = os.environ.get("EXP04_GGUF", "1") == "1"


# ---------- jobs ----------


def job(name: str) -> Path:
    return J / f"exp04-{name}"


def make_job(name: str, cfg: dict, max_size_gb: float = MAX_GB) -> Path:
    """A job from configs/exp04/<base>.json plus cfg; the size target goes in its taskspec."""
    d = job(name)
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    write(d / "config.json", cfg)
    spec = read(REPO / TASKSPEC)
    spec["target"] = {**spec.get("target", {}), "max_size_gb": max_size_gb}
    write(d / "taskspec.json", spec)
    return d


def config(base: str, **over) -> dict:
    return {**read(CONFIGS / f"{base}.json"), **over}


def smoke() -> dict:
    """The smoke test's verdict: {"fp8_attention", "experts", "size_gb_est", ...}."""
    return read(SMOKE / "result.json")


def r59_config() -> dict:
    s = smoke()
    return config("r59-t", reap_sparsity=round(1 - s["experts"] / 256, 6), quant_fp8_attention=s["fp8_attention"])


def model_dir(name: str = "r59-t") -> Path:
    """The vLLM directory the quantize stage registered for a job."""
    alloc = read(job(name) / "work" / "allocation.json")
    for c in alloc["candidates"].values():
        if (Path(c["path"]) / "config.json").exists():
            return Path(c["path"])
    raise RuntimeError(f"{job(name)} has no vLLM candidate in work/allocation.json")


def eval_only(name: str, candidates: dict[str, str], base: str = "eval", **over) -> Path:
    return make_job(name, config(base, eval_candidates=candidates, **over))


def eval_json(name: str) -> dict | None:
    p = job(name) / "out" / "eval.json"
    return read(p) if p.exists() else None


def suite(name: str, key: str = "livecodebench") -> dict:
    rep = eval_json(name)
    if not rep or not rep.get("candidates"):
        return {}
    c = rep["candidates"][0]
    return ((c.get("code") or {}).get("suites") or {}).get(key) or {}


def lcb(name: str) -> float | None:
    return suite(name).get("pass@1")


def done(name: str, stage: str = "eval") -> Callable[[], bool]:
    return lambda: (job(name) / ".done" / stage).exists()


def pipeline(step: Step, name: str, extra: str = "") -> None:
    sh(step, f"python pipeline.py --job {job(name)} {extra}".rstrip())


# ---------- the steps ----------


def lcb_current() -> bool:
    """Whether the suites were fetched with LeetCode tests as data and the example count."""
    p = NVME / "codebench" / "livecodebench.jsonl"
    if not p.exists():
        return False
    with open(p) as f:
        for line in f:
            if line.strip():
                return "public" in json.loads(line)
    return False


def run_setup(step: Step) -> None:
    if (NVME / "codebench" / "livecodebench.jsonl").exists() and not lcb_current():
        (NVME / "codebench" / "livecodebench.jsonl").rename(NVME / "codebench" / "livecodebench-old.jsonl")
        above("    re-fetching the code suites (older LiveCodeBench rows have no per-case time limit)")
    sh(step, f"NVME={NVME} TEACHER={TEACHER} STUDENT= bash scripts/setup_vm.sh")


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and (NVME / "venv-vllm/bin/python").exists()
            and (NVME / "llama.cpp/build/bin/llama-server").exists() and lcb_current())


def downloading() -> bool:
    return subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL).returncode == 0


def download_done() -> bool:
    log = NVME / "download.log"
    return (log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace")
            and (TEACHER_DIR / "config.json").exists())


def run_download(step: Step) -> None:
    log = NVME / "download.log"
    if not (TEACHER_DIR / "config.json").exists() and not downloading():
        # A VM set up for another experiment has a finished download.log but not this model.
        if log.exists():
            log.rename(log.with_name(f"download-{int(time.time())}.log"))
        sh(step, f"export PATH=$HOME/.local/bin:$PATH && nohup bash -c 'uv tool run --from huggingface_hub hf download {TEACHER} "
                 f"--local-dir {TEACHER_DIR} && echo DOWNLOADS_DONE' > {log} 2>&1 &")
    while not download_done():
        if not downloading():
            time.sleep(10)
            if not downloading() and not download_done():
                raise RuntimeError(f"the model download stopped; see {log} and re-run scripts/setup_vm.sh")
        time.sleep(5)


def download_progress() -> float:
    got = sum(f.stat().st_size for f in TEACHER_DIR.rglob("*") if f.is_file()) if TEACHER_DIR.exists() else 0
    return min(0.99, got / 70e9)


def run_smoke(step: Step) -> None:
    sh(step, f"python scripts/exp04_smoke.py --teacher {TEACHER_DIR} --out {SMOKE} --budget-gb {BUDGET_GB}")
    s = smoke()
    above(f"    vLLM serves int4 experts with {'FP8' if s['fp8_attention'] else 'BF16'} DeltaNet and attention: "
          f"keeping {s['experts']} of 256 experts, {s['size_gb_est']:.2f} GB estimated")
    if not s.get("bf16_loads", True):
        above("    vLLM did not load the unquantized pruned model, so the before-quantizing score is skipped")


def run_ref(step: Step) -> None:
    eval_only("ref", {"ref": str(TEACHER_DIR)}, code_eval_ref="")
    pipeline(step, "ref", "--only eval")
    score = lcb("ref")
    if score is None:
        raise RuntimeError("the full model has no LiveCodeBench score; see the eval log")
    above(f"    full model LiveCodeBench pass@1 (mean of 4): {score:.1%}; the bar is {BAR:.0%}")
    if score < 0.55:
        above("    far below the model card's 80%: the eval may be broken. The run goes on; send Claude the summary.")


def run_ref_suites(step: Step) -> None:
    eval_only("ref-suites", {"ref": str(TEACHER_DIR)}, "eval-suites", code_eval_ref="")
    pipeline(step, "ref-suites", "--only eval")


def run_build(step: Step) -> None:
    if not (job("r59-t") / "config.json").exists():
        make_job("r59-t", r59_config())
    pipeline(step, "r59-t")
    stats = job("r59-t") / "data" / "stats.json"
    if stats.exists():
        s = read(stats)
        c = s.get("contest") or {}
        above(f"    data: {s.get('rows', s.get('answered'))} rows, {c.get('rows_kept', '?')} contest rows, "
              f"{s.get('fix_rows', '?')} fix rows")
    score = lcb("r59-t")
    if score is not None:
        above(f"    r59-t LiveCodeBench pass@1: {score:.1%} ({'above' if score > BAR else 'below'} the {BAR:.0%} bar)")


def run_suites(step: Step) -> None:
    eval_only("r59-t-suites", {"r59-t": str(model_dir())}, "eval-suites")
    pipeline(step, "r59-t-suites", "--only eval")


def run_mugge(step: Step) -> None:
    eval_only("r59-t-mugge", {"r59-t": str(model_dir())}, "eval-suites", code_eval_format="harness", code_eval_ref="",
              code_eval_suites=SUITES + ["livecodebench"])
    pipeline(step, "r59-t-mugge", "--only eval")


def run_bf16(step: Step) -> None:
    if not smoke().get("bf16_loads", True):
        raise Skip("vLLM did not load the unquantized pruned model in the smoke test")
    eval_only("r59-t-bf16", {"r59-t-bf16": str(job("r59-t") / "work" / "healed")})
    pipeline(step, "r59-t-bf16", "--only eval")


def run_k6(step: Step) -> None:
    eval_only("r59-t-k6", {"r59-t-k6": str(model_dir())}, eval_experts_used=6)
    pipeline(step, "r59-t-k6", "--only eval")


BENCH = LOGS / "throughput.json"


def run_bench(step: Step) -> None:
    sh(step, f"$LOBBOT_VLLM_PY scripts/bench_vllm.py --model {model_dir()} --out {BENCH} --concurrency 1 32 128")


def run_gguf(step: Step) -> None:
    """The existing llama.cpp path (2-8 bit by saliency, 8-bit rest) on r59-t's healed model."""
    src, d = job("r59-t"), job("r59-t-gguf")
    if not (src / ".done" / "heal").exists():
        raise Skip("r59-t has not healed")
    make_job("r59-t-gguf", {**r59_config(), "quant_format": "gguf"}, max_size_gb=9.5)
    for sub in ("data", ".done"):
        shutil.copytree(src / sub, d / sub, dirs_exist_ok=True)
    for f in ("eval", "quantize", "package"):
        (d / ".done" / f).unlink(missing_ok=True)
    for f in ("layer_importance.json", "reap_saliency.json"):
        if (src / "work" / f).exists():
            shutil.copy(src / "work" / f, d / "work" / f)
    if not (d / "work" / "healed").exists():
        (d / "work" / "healed").symlink_to(src / "work" / "healed")
    pipeline(step, "r59-t-gguf", "--only quantize")


def gguf_file() -> Path:
    return job("r59-t-gguf") / "work" / "candidates" / "lobbot-moe.gguf"


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, data, scores), the vLLM model as one tar, and the GGUF."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step D) and re-run to upload.")
    tar = NVME / "exp04-jobs.tar"
    sh(step, f"mkdir -p {J}/_logs_exp04 && cp {LOGS}/*.log {LOGS}/*.json {J}/_logs_exp04/ 2>/dev/null; cd {J.parent} && "
             f"find {J.name} -path '*exp04-*' -type f -size -200M -print0 | tar cf {tar} --null -T -")
    up = [(tar, "exp04/exp04-jobs.tar")]
    try:
        m = model_dir()
        mtar = NVME / "qwen36-r59-t-w4a16.tar"
        if not mtar.exists():
            sh(step, f"tar cf {mtar}.part -C {m.parent} {m.name} && mv {mtar}.part {mtar}")
        up.append((mtar, "models/qwen36-r59-t-w4a16.tar"))
    except (FileNotFoundError, RuntimeError):
        above("    no r59-t model to upload")
    if gguf_file().exists():
        up.append((gguf_file(), "gguf/qwen36-r59-t.gguf"))
    for src, key in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{key}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


def steps() -> list[Step]:
    return [
        Step("VM setup", {"": 40}, run_setup, setup_done),
        Step("base model download", {"": 15}, run_download, download_done, download_progress),
        Step("smoke test: vLLM and the 4-bit format", {"": 25}, run_smoke, (SMOKE / "result.json").exists),
        Step("full model, LiveCodeBench x4", {"eval": 40}, run_ref, done("ref")),
        Step("full model, other suites", {"eval": 20}, run_ref_suites, done("ref-suites")),
        Step("r59-t build and LiveCodeBench x4",
             {"data": 25, "reap": 10, "heal": 45, "quantize": 20, "eval": 25, "package": 1}, run_build, done("r59-t", "package")),
        Step("r59-t, other suites", {"eval": 15}, run_suites, done("r59-t-suites")),
        Step("r59-t in Mugge's format", {"eval": 20}, run_mugge, done("r59-t-mugge")),
        Step("r59-t before quantizing", {"eval": 25}, run_bf16, done("r59-t-bf16")),
        Step("r59-t with 6 experts per token", {"eval": 20}, run_k6, done("r59-t-k6")),
        Step("throughput at 1, 32, 128 at once", {"": 15}, run_bench, BENCH.exists),
        *([Step("GGUF for the Mac", {"quantize": 20}, run_gguf, gguf_file().exists)] if WITH_GGUF else []),
        Step("upload to the bucket", {"": 20}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------

SHORT = {"livecodebench": "LCB", "multipl-e-py": "py", "multipl-e-js": "js", "multipl-e-ts": "ts",
         "multipl-e-cpp": "cpp", "c-set": "C"}
ROWS = [("full model", "ref", "ref-suites"), ("r59-t", "r59-t", "r59-t-suites"), ("r59-t Mugge format", None, "r59-t-mugge"),
        ("r59-t before quant", "r59-t-bf16", None), ("r59-t 6 experts", "r59-t-k6", None)]


def pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "–"


def summary() -> str:
    """LiveCodeBench pass@1 (mean of 4) and best-of-4 picked by the examples, the other
    suites, size, the share of answers that hit the 32k cap, GPU busy and speed."""
    head = ["model", "size", "LCB", "pick", "of ref"] + list(SHORT.values())[1:] + ["+1 fix", "capped", "GPU", "tok/s"]
    table = [head]
    ref = lcb("ref")
    for label, lcb_job, suites_job in ROWS:
        reps = [eval_json(n) for n in (lcb_job, suites_job) if n]
        if not any(reps):
            table.append([label, "not run"] + [""] * (len(head) - 2))
            continue
        main = next(r for r in reps if r)["candidates"][0]
        code = main.get("code") or {}
        lc = suite(lcb_job) if lcb_job else {}
        rest = [suite(suites_job, k).get("pass@1") if suites_job else None for k in list(SHORT)[1:]]
        fix = ((eval_json(suites_job) or {}).get("candidates") or [{}])[0].get("code", {}).get("mean_fix@1") if suites_job else None
        gpu = main.get("gpu") or {}
        table.append([label, f"{main.get('size_gb', 0):.1f} GB" if main.get("size_gb") else "",
                      pct(lc.get("pass@1")), pct(lc.get("pick@examples")),
                      pct(lc["pass@1"] / ref) if lc.get("pass@1") is not None and ref else "",
                      *[pct(x) for x in rest], pct(fix), pct(code.get("capped")),
                      f"{gpu['busy_mean']:.0f}%" if gpu.get("busy_mean") is not None else "",
                      f"{code['out_tok_s']:.0f}" if code.get("out_tok_s") else ""])
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    out = ["\n".join("  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(r)).rstrip()
                     for r in table)]
    r = lcb("r59-t")
    if r is not None:
        size = ((eval_json("r59-t") or {}).get("candidates") or [{}])[0].get("size_gb")
        out.append(f"\nThe bar: under 10 GB ({'yes' if size and size < 10 else 'no'}, {size} GB), "
                   f"LiveCodeBench above {BAR:.0%} ({'yes' if r > BAR else 'no'}, {r:.1%}).")
    if BENCH.exists():
        b = read(BENCH)
        out.append("Throughput (output tok/s): " + ", ".join(f"{k} at once {v['output_tok_s']:.0f}" for k, v in b.items()))
    return "\n".join(out)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
