#!/usr/bin/env python3
"""Experiment 04 on the B200 in one command: the next Mugge model, scored on Aider Polyglot.

    python3 scripts/exp04.py          # everything; re-run after a stop and finished steps are skipped
    python3 scripts/exp04.py --list   # the steps and their time guesses
    python3 scripts/exp04.py summary  # the results table

The bar (Oliver, 2026-10-10): a model file under 10 GB, served by vLLM on the GPU, a
faster build than r50w95s-t's, and an Aider Polyglot score (pass rate after the second
attempt) of at least 90% of the best full model's, both measured here. vLLM's MoE
kernels go no lower than 4 bits, so experts are int4, attention / DeltaNet / dense MLP
and the output head FP8, embeddings and routers BF16, and fewer experts are kept.
Plan: /mnt/project-files/plan/next-model.md.

Three bases (plan/moe-bases.md):
  qwen    Qwen3.6-35B-A3B     256 experts, ~104 kept
  ornith  Ornith-1.5-35B-A3B  the same architecture, stronger at agent coding
  gemma   Gemma 4 26B-A4B     128 experts, ~72 kept (less pruning, ~4B active)
Two are built: the better of qwen and ornith on full-model Polyglot (ornith unless it
trails by more than 2 points; they share a shape, so pruning should cost both the
same), and gemma. EXP04_BASES=qwen,gemma (say) runs a subset.

Steps:
  1. VM setup (scripts/setup_vm.sh), Aider's benchmark image and the Exercism
     training exercises (scripts/polyglot.py), with the code suites re-fetched
     when they predate the per-case time limit
  2. the base model downloads
  3. smoke tests (scripts/exp04_smoke.py), one per architecture: a few layers of
     the base with random weights go through the real quantize stage and vLLM,
     with FP8 attention first and BF16 if vLLM refuses it. That picks the format,
     and the format and the 10 GB budget pick the expert count
  4. Aider Polyglot on each full model (thinking on, diff edits, 64 at once).
     This sets the bar and picks qwen or ornith
  5. per build: the base works Exercism exercises Polyglot doesn't use, through
     Aider, logged; its passing transcripts (fix turns too) join the heal data. Then
     data, REAP to the smoke test's expert count, heal capped at 45 min, int4 +
     FP8 quantize with a size check, one LiveCodeBench pass as a reasoning floor,
     and Aider Polyglot on the 4-bit model
  6. side scores on the better build: before quantizing (pruning loss vs 4-bit
     loss), and 6 experts per token instead of 8
  7. throughput of the better build at 1, 32 and 128 requests at once
  8. the models and the job results to the bucket

Runs with the system python3 (stdlib only); each step sources .env.vm. Logs and the
Polyglot results are in ~/exp04-logs. The upload needs an evroc login first; without
one it is skipped and a re-run picks it up.
"""

from __future__ import annotations

import json
import os
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
TASKSPEC = "examples/python-utils.code.taskspec.json"
SMOKE = NVME / "exp04-smoke"
EXERCISM = NVME / "exercism-train"  # Exercism practice exercises outside Polyglot
POLY = LOGS / "polyglot"  # one results JSON per Polyglot run
CONFIGS = REPO / "configs" / "exp04"

BASES = {"qwen": "Qwen/Qwen3.6-35B-A3B", "ornith": "ornith-ai/Ornith-1.5-35B", "gemma": "google/gemma-4-26B-A4B-it"}
LABELS = {"qwen": "Qwen3.6", "ornith": "Ornith-1.5", "gemma": "Gemma 4 26B-A4B"}
SHAPE = {"qwen": "qwen", "ornith": "qwen", "gemma": "gemma"}  # architecture, for the smoke test
PARSER = {"qwen": "qwen3", "ornith": "qwen3", "gemma": "gemma4"}  # vLLM reasoning parser
WANTED = [k for k in os.environ.get("EXP04_BASES", ",".join(BASES)).split(",") if k]
if not WANTED or set(WANTED) - set(BASES):
    sys.exit(f"EXP04_BASES takes a comma list of {', '.join(BASES)}")
ORNITH_MARGIN = 0.02  # ornith is built unless it trails qwen by more than this on Polyglot
SHARE = 0.90  # Oliver's bar: 90% of the best full model's Polyglot score
# 10 GB is the bar. The expert count is the most whose estimate fits BUDGET_GB; the
# quantize stage refuses a file over MAX_GB.
BUDGET_GB = 9.8
MAX_GB = 9.95
CHOICE = LOGS / "builds.json"


# ---------- bases, builds and results ----------


def base_dir(key: str) -> Path:
    return NVME / "models" / BASES[key].split("/")[-1]


def job(name: str) -> Path:
    return J / f"exp04-{name}"


def poly(name: str) -> dict | None:
    p = POLY / f"{name}.json"
    return read(p) if p.exists() else None


def poly_score(name: str) -> float | None:
    r = poly(name)
    return r.get("pass_rate_2") if r else None


def smoke(key: str) -> dict:
    """The smoke test's verdict for key's architecture: {"fp8_attention", "experts", "size_gb_est", ...}."""
    return read(SMOKE / SHAPE[key] / "result.json")


def builds() -> list[str]:
    """The bases to compress: the better of qwen and ornith on full-model Polyglot, and
    gemma. Decided once, after the full models are scored."""
    if CHOICE.exists():
        return read(CHOICE)["builds"]
    full = {k: poly_score(f"{k}-full") for k in WANTED}
    if any(v is None for v in full.values()):
        raise RuntimeError("every full model needs a Polyglot score before the builds are picked")
    out = []
    qwen_shape = [k for k in ("ornith", "qwen") if k in WANTED]
    if len(qwen_shape) == 2:
        out.append("ornith" if full["ornith"] >= full["qwen"] - ORNITH_MARGIN else "qwen")
    else:
        out += qwen_shape
    out += [k for k in WANTED if k == "gemma"]
    LOGS.mkdir(exist_ok=True)
    write(CHOICE, {"builds": out, "full": full, "bar": round(SHARE * max(full.values()), 4)})
    return out


def bar() -> float | None:
    return read(CHOICE)["bar"] if CHOICE.exists() else None


def build_name(key: str) -> str:
    return f"{key}-w4"


def make_job(name: str, cfg: dict, max_size_gb: float = MAX_GB) -> Path:
    d = job(name)
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    write(d / "config.json", cfg)
    spec = read(REPO / TASKSPEC)
    spec["target"] = {**spec.get("target", {}), "max_size_gb": max_size_gb}
    write(d / "taskspec.json", spec)
    return d


def build_config(key: str) -> dict:
    """configs/exp04/build.json for one base: its own teacher, transcripts, expert count and format."""
    s, n = smoke(key), n_experts(key)
    return {**read(CONFIGS / "build.json"), "teacher": BASES[key], "data_extra_rows": str(rows_file(key)),
            "reap_sparsity": round(1 - s["experts"] / n, 6), "quant_fp8_attention": s["fp8_attention"],
            "eval_reasoning_parser": PARSER[key]}


def n_experts(key: str) -> int:
    cfg = read(base_dir(key) / "config.json")
    t = cfg.get("text_config") or cfg
    return t.get("num_experts") or t["num_local_experts"]


def model_dir(name: str) -> Path:
    """The vLLM directory the quantize stage registered for a job."""
    alloc = read(job(name) / "work" / "allocation.json")
    for c in alloc["candidates"].values():
        if (Path(c["path"]) / "config.json").exists():
            return Path(c["path"])
    raise RuntimeError(f"{job(name)} has no vLLM candidate in work/allocation.json")


def lcb(name: str) -> float | None:
    p = job(name) / "out" / "eval.json"
    if not p.exists():
        return None
    c = (read(p).get("candidates") or [{}])[0]
    return (((c.get("code") or {}).get("suites") or {}).get("livecodebench") or {}).get("pass@1")


def best_build() -> str:
    """The build with the higher Polyglot score."""
    scored = [(poly_score(build_name(k)) or -1, k) for k in builds()]
    if not scored or max(scored)[0] < 0:
        raise Skip("no build has a Polyglot score yet")
    return max(scored)[1]


# ---------- commands ----------


def polyglot(step: Step, model: Path, name: str, key: str, *extra: str) -> None:
    """Aider Polyglot (or, with --exercises, another exercises dir) on a model served by vLLM."""
    POLY.mkdir(parents=True, exist_ok=True)
    sh(step, f"python scripts/polyglot.py run --model-dir {model} --name {name} --out {POLY / (name + '.json')} "
             f"--reasoning-parser {PARSER[key]} {' '.join(extra)}".rstrip())


def pipeline(step: Step, name: str, extra: str = "") -> None:
    sh(step, f"python pipeline.py --job {job(name)} {extra}".rstrip())


def rows_file(key: str) -> Path:
    return LOGS / "transcripts" / f"{key}.rows.jsonl"


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
    sh(step, f"NVME={NVME} TEACHER={BASES[WANTED[0]]} STUDENT= bash scripts/setup_vm.sh")
    sh(step, f"python scripts/polyglot.py setup && python scripts/polyglot.py exercism --out {EXERCISM}")


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and (NVME / "venv-vllm/bin/python").exists() and lcb_current()
            and (EXERCISM / ".done").exists())


def downloading() -> bool:
    return subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL).returncode == 0


DONE_MARK = "EXP04_DOWNLOADS_DONE"


def download_done() -> bool:
    log = NVME / "download.log"
    return (log.exists() and DONE_MARK in log.read_text(errors="replace")
            and all((base_dir(k) / "config.json").exists() for k in WANTED))


def run_download(step: Step) -> None:
    """setup_vm.sh starts the first base's download; this waits for it, then fetches every
    base (hf download skips the files already there), so a half-finished download resumes."""
    log = NVME / "download.log"
    while downloading():
        time.sleep(5)
    if log.exists():
        log.rename(log.with_name(f"download-{int(time.time())}.log"))
    fetch = " && ".join(f"uv tool run --from huggingface_hub hf download {BASES[k]} --local-dir {base_dir(k)}"
                        for k in WANTED)
    sh(step, f"export PATH=$HOME/.local/bin:$PATH && nohup bash -c '{fetch} && echo {DONE_MARK}' > {log} 2>&1 &")
    time.sleep(10)
    while not download_done():
        if not downloading():
            time.sleep(10)
            if not downloading() and not download_done():
                raise RuntimeError(f"the model download stopped; see {log} and re-run scripts/setup_vm.sh")
        time.sleep(5)


def download_progress() -> float:
    sizes = {"qwen": 70e9, "ornith": 72e9, "gemma": 52e9}
    got = sum(f.stat().st_size for k in WANTED if base_dir(k).exists() for f in base_dir(k).rglob("*") if f.is_file())
    return min(0.99, got / sum(sizes[k] for k in WANTED))


def run_smoke(key: str) -> Callable[[Step], None]:
    def run(step: Step) -> None:
        out = SMOKE / SHAPE[key]
        sh(step, f"python scripts/exp04_smoke.py --teacher {base_dir(key)} --out {out} --budget-gb {BUDGET_GB}")
        s = smoke(key)
        above(f"    {SHAPE[key]} architecture: int4 experts with {'FP8' if s['fp8_attention'] else 'BF16'} attention, "
              f"keeping {s['experts']} of {n_experts(key)} experts, {s['size_gb_est']:.2f} GB estimated")
    return run


def smoke_keys() -> list[str]:
    """One base per architecture."""
    seen: dict[str, str] = {}
    for k in WANTED:
        seen.setdefault(SHAPE[k], k)
    return list(seen.values())


def run_full(key: str) -> Callable[[Step], None]:
    def run(step: Step) -> None:
        polyglot(step, base_dir(key), f"{key}-full", key)
        above(f"    {LABELS[key]} full model, Aider Polyglot: {poly_score(f'{key}-full'):.1%}")
        if all(poly(f"{k}-full") for k in WANTED):
            b = builds()
            above(f"    bar: {bar():.1%} (90% of the best full model); building {', '.join(LABELS[k] for k in b)}")
    return run


def run_build(slot: int) -> Callable[[Step], None]:
    """The slot-th build: transcripts, the pipeline, then Polyglot on the 4-bit model."""
    def run(step: Step) -> None:
        b = builds()
        if slot >= len(b):
            raise Skip("only one build in this run")
        key, name = b[slot], build_name(b[slot])
        if not rows_file(key).exists():
            log = LOGS / "transcripts" / f"{key}.requests.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            polyglot(step, base_dir(key), f"{key}-exercism", key, f"--exercises {EXERCISM} --log {log}")
            sh(step, f"python scripts/polyglot.py rows --log {log} --results {POLY / (key + '-exercism.json')} "
                     f"--out {rows_file(key)}")
        if not (job(name) / "config.json").exists():
            make_job(name, build_config(key))
        pipeline(step, name)
        polyglot(step, model_dir(name), name, key)
        score, line = poly_score(name), bar()
        above(f"    {LABELS[key]} 4-bit: Aider Polyglot {score:.1%} against the {line:.1%} bar "
              f"({'met' if score >= line else 'missed'}); LiveCodeBench {lcb(name) or 0:.0%}")
    return run


def build_done(slot: int) -> Callable[[], bool]:
    def done() -> bool:
        if not CHOICE.exists():
            return False
        b = builds()
        return slot >= len(b) or poly(build_name(b[slot])) is not None
    return done


def run_before_quant(step: Step) -> None:
    key = best_build()
    if not smoke(key).get("bf16_loads", True):
        raise Skip("vLLM did not load the unquantized pruned model in the smoke test")
    polyglot(step, job(build_name(key)) / "work" / "healed", f"{key}-bf16", key)


def run_k6(step: Step) -> None:
    key = best_build()
    polyglot(step, model_dir(build_name(key)), f"{key}-w4-k6", key, "--experts-used 6")


def side_done(suffix: str) -> Callable[[], bool]:
    return lambda: any(poly(f"{k}-{suffix}") for k in BASES)


BENCH = LOGS / "throughput.json"


def run_bench(step: Step) -> None:
    sh(step, f"$LOBBOT_VLLM_PY scripts/bench_vllm.py --model {model_dir(build_name(best_build()))} --out {BENCH} "
             f"--concurrency 1 32 128")


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, data, scores), the logs and Polyglot results,
    and each 4-bit model as one tar."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step D) and re-run to upload.")
    tar = NVME / "exp04-jobs.tar"
    sh(step, f"rm -rf {J}/_logs_exp04 && cp -r {LOGS} {J}/_logs_exp04 && cd {J.parent} && "
             f"find {J.name} \\( -path '*exp04-*' -o -path '*_logs_exp04*' \\) -type f -size -200M -print0 "
             f"| tar cf {tar} --null -T -")
    up = [(tar, "exp04/exp04-jobs.tar")]
    for key in builds():
        try:
            m = model_dir(build_name(key))
        except (FileNotFoundError, RuntimeError):
            continue
        mtar = NVME / f"{key}-w4.tar"
        if not mtar.exists():
            sh(step, f"tar cf {mtar}.part -C {m.parent} {m.name} && mv {mtar}.part {mtar}")
        up.append((mtar, f"models/exp04-{key}-w4.tar"))
    for src, dst in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{dst}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


BUILD_MINUTES = {"polyglot": 40, "data": 15, "reap": 10, "heal": 45, "quantize": 20, "eval": 10, "package": 1}


def steps() -> list[Step]:
    n_builds = min(2, len({SHAPE[k] for k in WANTED}))
    return [
        Step("VM setup and Aider's benchmark", {"": 50}, run_setup, setup_done),
        Step("base model downloads", {"": 25}, run_download, download_done, download_progress),
        *[Step(f"smoke test: {SHAPE[k]} architecture on vLLM", {"": 20}, run_smoke(k),
               (SMOKE / SHAPE[k] / "result.json").exists) for k in smoke_keys()],
        *[Step(f"{LABELS[k]} full model, Aider Polyglot", {"": 40}, run_full(k), (POLY / f"{k}-full.json").exists)
          for k in WANTED],
        *[Step(f"build {i + 1}: transcripts, compress, Polyglot", {**BUILD_MINUTES, "": 35}, run_build(i), build_done(i))
          for i in range(n_builds)],
        Step("better build before quantizing, Polyglot", {"": 40}, run_before_quant, side_done("bf16")),
        Step("better build with 6 experts, Polyglot", {"": 35}, run_k6, side_done("w4-k6")),
        Step("throughput at 1, 32, 128 at once", {"": 15}, run_bench, BENCH.exists),
        Step("upload to the bucket", {"": 20}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------


def pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "–"


def summary() -> str:
    """Aider Polyglot after 1 and 2 attempts, well-formed edits and per language, for every
    run, with the bar; then size, LiveCodeBench floor, GPU busy and throughput."""
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]
    head = ["run", "size", "pass 2", "pass 1", "formed", *langs, "LCB", "GPU"]
    table = [head]
    names = [f"{k}-full" for k in WANTED] + [build_name(k) for k in BASES] + \
            [f"{k}-bf16" for k in BASES] + [f"{k}-w4-k6" for k in BASES]
    for name in names:
        r = poly(name)
        if not r:
            continue
        size = ""
        if name.endswith("-w4"):
            alloc = job(name) / "work" / "allocation.json"
            if alloc.exists():
                size = f"{next(iter(read(alloc)['candidates'].values()))['size_gb']:.2f} GB"
        per = r.get("per_language") or {}
        gpu = r.get("gpu") or {}
        table.append([name, size, pct(r.get("pass_rate_2")), pct(r.get("pass_rate_1")),
                      pct(r.get("percent_cases_well_formed")),
                      *[pct((per.get(lang) or {}).get("pass_rate_2")) for lang in langs],
                      pct(lcb(name)) if name.endswith("-w4") else "",
                      f"{gpu['busy_pct']:.0f}%" if gpu.get("busy_pct") is not None else ""])
    if len(table) == 1:
        return "No Aider Polyglot results yet."
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    out = ["\n".join("  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(r)).rstrip()
                     for r in table)]
    if CHOICE.exists():
        c = read(CHOICE)
        out.append(f"\nThe bar: Aider Polyglot {c['bar']:.1%} (90% of the best full model), under 10 GB.")
        for key in c["builds"]:
            s = poly_score(build_name(key))
            if s is not None:
                out.append(f"  {LABELS[key]} 4-bit: {s:.1%}, {'meets' if s >= c['bar'] else 'misses'} the bar")
    if BENCH.exists():
        b = read(BENCH)
        out.append("Throughput (output tok/s): " + ", ".join(f"{k} at once {v['output_tok_s']:.0f}" for k, v in b.items()))
    return "\n".join(out)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
