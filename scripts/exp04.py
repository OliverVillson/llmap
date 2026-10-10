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
  qwen    Qwen3.6-35B-A3B     256 experts, ~106 kept
  ornith  Ornith-1.5-35B-A3B  the same architecture, stronger at agent coding
  gemma   Gemma 4 26B-A4B     128 experts, ~70 kept (less pruning, ~4B active)
Two are built: the better of qwen and ornith on full-model Polyglot (ornith unless it
trails by more than 2 points; they share a shape, so pruning should cost both the
same), and gemma. If gemma's full model already scores under the bar, its build could
not pass, so that slot builds the chosen qwen-shape base at ~12 GB instead: what the
10 GB limit costs. EXP04_BASES=qwen,gemma (say) runs a subset.

Steps:
  1. VM setup (scripts/setup_vm.sh), Aider's benchmark image and the Exercism
     training exercises (scripts/polyglot.py), with the code suites re-fetched
     when they predate the per-case time limit; then the harness self-test runs
     reference solutions through the image, so a broken toolchain fails here and
     not as a low score
  2. the base model downloads
  3. smoke tests (scripts/exp04_smoke.py), one per architecture: a few layers of
     the base with random weights go through the real quantize stage and vLLM,
     with FP8 attention first and BF16 if vLLM refuses it. That picks the format,
     and the format and the 10 GB budget pick the expert count
  4. Aider Polyglot on each full model (thinking on, diff edits), after a
     6-exercise pre-flight. This sets the bar and picks qwen or ornith
  5. per build: the base works Exercism exercises Polyglot doesn't use, through
     Aider, logged; its passing transcripts (fix turns too) join the heal data. Then
     data, REAP to the smoke test's expert count, heal capped at 90 min, int4 +
     FP8 quantize with a size check, one LiveCodeBench pass as a reasoning floor,
     Aider Polyglot on the 4-bit model, and the Mugge-format eval (tickets in,
     files out, one fix round) on it and its full base
  6. side scores on the better build: before quantizing (pruning loss vs 4-bit
     loss), and 6 experts per token instead of 8
  7. throughput of the better build at 1, 32 and 128 requests at once
  8. the models and the job results to the bucket

Each build is also compared with the best full model exercise by exercise: the paired
difference and its 95% interval say met, missed, or too close to call, since one
Polyglot score moves about 3 points between runs.

Runs with the system python3 (stdlib only); each step sources .env.vm. Logs and the
Polyglot results are in ~/exp04-logs, and go to the bucket after every scored step
while the evroc login lasts. The final upload needs a login; without one it is
skipped and a re-run picks it up.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from runner import REPO, Skip, Step, above, evroc_ok, read, run_plan, sh, write

from stages import w4a16

NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
J = Path(os.environ.get("LOBBOT_JOBS", NVME / "jobs"))
LOGS = Path(os.environ.get("EXP04_LOGS", Path.home() / "exp04-logs"))
BUCKET = "bucket://mugge-library"
TASKSPEC = "examples/python-utils.code.taskspec.json"
SMOKE = NVME / "exp04-smoke"
EXERCISM = NVME / "exercism-train"  # Exercism practice exercises outside Polyglot
POLY = LOGS / "polyglot"  # one results JSON per Polyglot run
CONFIGS = REPO / "configs" / "exp04"

BASES = {"qwen": "Qwen/Qwen3.6-35B-A3B", "ornith": "ornith-ai/Ornith-1.5-35B-A3B", "gemma": "google/gemma-4-26B-A4B-it"}
LABELS = {"qwen": "Qwen3.6", "ornith": "Ornith-1.5", "gemma": "Gemma 4 26B-A4B"}
SHAPE = {"qwen": "qwen", "ornith": "qwen", "gemma": "gemma"}  # architecture, for the smoke test
PARSER = {"qwen": "qwen3", "ornith": "qwen3", "gemma": "gemma4"}  # vLLM reasoning parser
# Thinking sampling from each model card: Qwen3.6 (and Ornith) 0.6 / top_p 0.95 / top_k 20, Gemma 4 1.0 / 0.95 / 64
SAMPLING = {"qwen": (0.6, 20), "ornith": (0.6, 20), "gemma": (1.0, 64)}
WANTED = [k for k in os.environ.get("EXP04_BASES", ",".join(BASES)).split(",") if k]
if not WANTED or set(WANTED) - set(BASES):
    sys.exit(f"EXP04_BASES takes a comma list of {', '.join(BASES)}")
ORNITH_MARGIN = 0.02  # ornith is built unless it trails qwen by more than this on Polyglot
SHARE = 0.90  # Oliver's bar: 90% of the best full model's Polyglot score
# 10 GB is the bar. The expert count is the most whose estimate fits BUDGET_GB; the
# quantize stage refuses a file over MAX_GB.
BUDGET_GB = 9.8
MAX_GB = 9.95
BIG_BUDGET_GB, BIG_MAX_GB = 11.8, 11.95  # the ~12 GB point that can take gemma's slot
ANCHOR = 0.622  # full Qwen3.6 on Polyglot, third-party (llama.cpp Q8, diff edits)
CHOICE = LOGS / "builds.json"
# Guards against a run that looks fine but isn't. A full model under these on Polyglot is
# broken (reasoning parser, chat template, edit format), not weak: the third-party anchor
# for Qwen3.6 is 62%. Fewer transcript rows than MIN_ROWS means the logging or the row
# parser failed. Either stops the run; EXP04_ACCEPT=1 goes on anyway.
MIN_FULL_PASS, MIN_FORMED, MIN_ROWS = 0.15, 0.70, 60
MAX_TIMEOUTS = 0.05  # test timeouts per exercise above this: the VM fails correct answers
MIN_DISK_GB = 450  # three bases, two builds, Docker and the upload tars
MUGGE_LANGS = ("cpp", "javascript", "python")  # Polyglot's languages nearest Mugge's (C, JS/TS, Python)
ACCEPT = os.environ.get("EXP04_ACCEPT") == "1"


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
    """The builds: the better of qwen and ornith on full-model Polyglot, and gemma, or
    "<base>@12" (the chosen base at ~12 GB) in gemma's slot when gemma's full model is
    already under the bar. Decided once, after the full models are scored."""
    if CHOICE.exists():
        return read(CHOICE)["builds"]
    full = {k: poly_score(f"{k}-full") for k in WANTED}
    if any(v is None for v in full.values()):
        raise RuntimeError("every full model needs a Polyglot score before the builds are picked")
    bad = {k: why for k in WANTED if (why := suspect(f"{k}-full"))}
    if bad and not ACCEPT:
        raise RuntimeError("these full-model Polyglot runs look broken, not weak: "
                           + "; ".join(f"{LABELS[k]} {why}" for k, why in bad.items())
                           + f". Send Claude the step logs and {POLY}; EXP04_ACCEPT=1 builds anyway.")
    out = []
    qwen_shape = [k for k in ("ornith", "qwen") if k in WANTED]
    if len(qwen_shape) == 2:
        out.append("ornith" if full["ornith"] >= full["qwen"] - ORNITH_MARGIN else "qwen")
    else:
        out += qwen_shape
    line = round(SHARE * max(full.values()), 4)
    swapped = ""
    if "gemma" in WANTED:
        if out and full["gemma"] < line:
            out.append(f"{out[0]}@12")
            swapped = (f"{LABELS['gemma']} scored {full['gemma']:.1%}, under the {line:.1%} bar before compressing, "
                       f"so its slot builds {LABELS[out[0]]} at ~12 GB instead")
        else:
            out.append("gemma")
    LOGS.mkdir(exist_ok=True)
    write(CHOICE, {"builds": out, "full": full, "bar": line, "swapped": swapped})
    return out


def base_of(entry: str) -> str:
    return entry.split("@")[0]


def big(entry: str) -> bool:
    return entry.endswith("@12")


def label(entry: str) -> str:
    return LABELS[base_of(entry)] + (" at 12 GB" if big(entry) else "")


def suspect(name: str) -> str:
    """Why a Polyglot result looks broken rather than weak, or ""."""
    r = poly(name) or {}
    why = []
    if (r.get("pass_rate_2") or 0) < MIN_FULL_PASS:
        why.append(f"passed {r.get('pass_rate_2') or 0:.0%}")
    if (r.get("percent_cases_well_formed") or 0) < MIN_FORMED:
        why.append(f"{r.get('percent_cases_well_formed') or 0:.0%} well-formed edits")
    if timeouts(r) > MAX_TIMEOUTS:
        why.append(f"{r.get('test_timeouts')} test timeouts in {r.get('n')} exercises (the VM, not the model)")
    return " and ".join(why)


def timeouts(r: dict) -> float:
    return (r.get("test_timeouts") or 0) / max(1, r.get("n") or 0)


def best_full() -> str | None:
    if not CHOICE.exists():
        return None
    full = read(CHOICE)["full"]
    return f"{max(full, key=full.get)}-full"


def paired(base: str, other: str, share: float = SHARE) -> dict | None:
    """other against share x base, exercise by exercise (pass by the second attempt): the
    mean difference with its 95% interval. One run's score moves ~3 points; pairing on
    the same exercises takes out most of the exercises' own difficulty."""
    a, b = (poly(base) or {}).get("cases"), (poly(other) or {}).get("cases")
    common = sorted(set(a or {}) & set(b or {}))
    if len(common) < 30:
        return None
    d = [float(b[k][1]) - share * float(a[k][1]) for k in common]
    m, se = statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "diff": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def verdict(p: dict) -> str:
    return "met" if p["lo"] >= 0 else "missed" if p["hi"] < 0 else "too close to call"


def mugge_langs(r: dict) -> float | None:
    """Polyglot on the languages nearest Mugge's, weighted by exercise count."""
    per = [(v["n"], v["pass_rate_2"]) for lang, v in (r.get("per_language") or {}).items()
           if lang in MUGGE_LANGS and v.get("pass_rate_2") is not None]
    return sum(n * s for n, s in per) / sum(n for n, _ in per) if per else None


def bar() -> float | None:
    return read(CHOICE)["bar"] if CHOICE.exists() else None


def build_name(entry: str) -> str:
    return f"{base_of(entry)}-w4" + ("-12g" if big(entry) else "")


def make_job(name: str, cfg: dict, max_size_gb: float = MAX_GB) -> Path:
    d = job(name)
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    write(d / "config.json", cfg)
    spec = read(REPO / TASKSPEC)
    spec["target"] = {**spec.get("target", {}), "max_size_gb": max_size_gb}
    write(d / "taskspec.json", spec)
    return d


def build_config(entry: str) -> dict:
    """configs/exp04/build.json for one build: its base's teacher, transcripts and format,
    and the expert count the smoke test found for 10 GB (or the 12 GB budget's)."""
    key, s = base_of(entry), smoke(base_of(entry))
    n = n_experts(key)
    kept = (w4a16.max_experts(read(base_dir(key) / "config.json"), BIG_BUDGET_GB, s["fp8_attention"], 2)
            if big(entry) else s["experts"])
    return {**read(CONFIGS / "build.json"), **sampling(key), "teacher": BASES[key],
            "data_extra_rows": str(rows_file(key)), "reap_sparsity": round(1 - kept / n, 6),
            "quant_fp8_attention": s["fp8_attention"]}


def sampling(key: str) -> dict:
    """The base's reasoning parser and its card's thinking sampling, as job config."""
    return {"eval_reasoning_parser": PARSER[key], "data_thinking_temperature": SAMPLING[key][0],
            "code_eval_thinking_temperature": SAMPLING[key][0], "thinking_top_k": SAMPLING[key][1]}


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
    """The 10 GB build with the higher Polyglot score (the side scores are about it)."""
    scored = [(poly_score(build_name(k)) or -1, k) for k in builds() if not big(k)]
    if not scored or max(scored)[0] < 0:
        raise Skip("no 10 GB build has a Polyglot score yet")
    return max(scored)[1]


def mugge(name: str) -> dict | None:
    """The Mugge-format eval of an eval-only job: mean pass@1 and after one fix round."""
    p = job(f"{name}-mugge") / "out" / "eval.json"
    if not p.exists():
        return None
    code = (read(p).get("candidates") or [{}])[0].get("code") or {}
    return {"pass": code.get("mean_pass@1"), "fix": code.get("mean_fix@1")}


# ---------- commands ----------


def polyglot(step: Step, model: Path, name: str, key: str, *extra: str) -> None:
    """Aider Polyglot (or, with --exercises, another exercises dir) on a model served by vLLM."""
    POLY.mkdir(parents=True, exist_ok=True)
    temp, top_k = SAMPLING[key]
    sh(step, f"python scripts/polyglot.py run --model-dir {model} --name {name} --out {POLY / (name + '.json')} "
             f"--reasoning-parser {PARSER[key]} --temperature {temp} --top-p 0.95 --top-k {top_k} "
             f"{' '.join(extra)}".rstrip())


def pipeline(step: Step, name: str, extra: str = "") -> None:
    sh(step, f"python pipeline.py --job {job(name)} {extra}".rstrip())


def mugge_eval(step: Step, model: Path, name: str, key: str) -> None:
    """The Mugge-format eval (configs/exp04/mugge.json: tickets in, files out, one fix
    round) on one model, as an eval-only job named <name>-mugge."""
    d = job(f"{name}-mugge")
    if (d / "out" / "eval.json").exists():
        return
    make_job(f"{name}-mugge", {**read(CONFIGS / "mugge.json"), **sampling(key), "teacher": BASES[key],
                               "eval_candidates": {name: str(model)}}, max_size_gb=200)
    pipeline(step, f"{name}-mugge", "--only eval")


def upload_results() -> None:
    """The small results (Polyglot JSONs, choices, step logs) to the bucket, best effort,
    so a VM that stops early still leaves them. Never fails the step."""
    try:
        if not evroc_ok():
            return
        tar = NVME / "exp04-results.tar"
        subprocess.run(["tar", "cf", str(tar), "--exclude=*.requests.jsonl", "-C", str(LOGS.parent), LOGS.name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        subprocess.run(["evroc", "storage", "bucket", "copy", "--from", str(tar), "--to",
                        f"{BUCKET}/exp04/exp04-results.tar"], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=600)
    except Exception:  # noqa: BLE001
        pass


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
    disk = shutil.disk_usage(NVME).total / 1e9
    if disk < MIN_DISK_GB and not ACCEPT:
        raise RuntimeError(f"{NVME} has {disk:.0f} GB; the run needs ~{MIN_DISK_GB} GB. EXP04_ACCEPT=1 tries anyway.")
    sh(step, f"python scripts/polyglot.py setup && python scripts/polyglot.py exercism --out {EXERCISM}")
    # reference solutions through aider's image: toolchains, crate and Gradle downloads, test time limits
    ok = " || true" if ACCEPT else ""
    sh(step, f"python scripts/polyglot.py selftest --out {SELFTEST}{ok} && "
             f"python scripts/polyglot.py selftest --exercises {EXERCISM} --out {SELFTEST}-train{ok}")


SELFTEST = LOGS / "selftest"


def setup_done() -> bool:
    return ((REPO / ".env.vm").exists() and (NVME / "venv-vllm/bin/python").exists() and lcb_current()
            and (EXERCISM / ".done").exists() and (SELFTEST / "selftest.json").exists())


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
        preflight(step, key)
        polyglot(step, base_dir(key), f"{key}-full", key)
        score = poly_score(f"{key}-full")
        note = f" (third-party anchor {ANCHOR:.1%}; worth a look)" if key == "qwen" and abs(score - ANCHOR) > 0.10 else ""
        above(f"    {LABELS[key]} full model, Aider Polyglot: {score:.1%}{note}")
        upload_results()
        if all(poly(f"{k}-full") for k in WANTED):
            b = builds()
            above(f"    bar: {bar():.1%} (90% of the best full model); building {', '.join(label(e) for e in b)}")
            if read(CHOICE)["swapped"]:
                above(f"    {read(CHOICE)['swapped']}")
    return run


def preflight(step: Step, key: str) -> None:
    """One exercise per language first: a broken reasoning parser or chat template shows
    in a few minutes instead of after a 40-minute run."""
    name = f"{key}-preflight"
    if not poly(name):
        polyglot(step, base_dir(key), name, key, "--sample-per-language 1")
    r = poly(name)
    if r is None:
        raise RuntimeError(f"{LABELS[key]}'s pre-flight wrote no results; see the step log")
    if (r.get("percent_cases_well_formed") or 0) < 0.5 and not ACCEPT:
        raise RuntimeError(f"{LABELS[key]}'s pre-flight: {r.get('percent_cases_well_formed') or 0:.0%} of "
                           f"{r.get('n')} exercises came back as well-formed edits. Send Claude the step log; "
                           f"delete {POLY / (name + '.json')} to try again, or EXP04_ACCEPT=1 goes on.")


def run_build(slot: int) -> Callable[[Step], None]:
    """The slot-th build: transcripts, the pipeline, then Polyglot on the 4-bit model. A
    failed build is skipped, so the other one still runs; a re-run retries it."""
    def run(step: Step) -> None:
        b = builds()
        if slot >= len(b):
            raise Skip("only one build in this run")
        try:
            build(step, b[slot])
        except Exception as e:  # noqa: BLE001
            raise Skip(f"{label(b[slot])} failed (re-run to retry): {e}") from e
        finally:
            upload_results()
    return run


def build(step: Step, entry: str) -> None:
    """Transcripts, the pipeline, then Polyglot and the Mugge-format eval on the 4-bit
    model, for one build."""
    key, name = base_of(entry), build_name(entry)
    if not rows_file(key).exists():
        log = LOGS / "transcripts" / f"{key}.requests.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        polyglot(step, base_dir(key), f"{key}-exercism", key, f"--exercises {EXERCISM} --log {log}")
        # rows heal would cut short are dropped: ~3 characters a token in code and its thinking
        max_chars = 3 * read(CONFIGS / "build.json")["heal_max_len"]
        sh(step, f"python scripts/polyglot.py rows --log {log} --results {POLY / (key + '-exercism.json')} "
                 f"--out {rows_file(key)} --max-chars {max_chars}")
    n = sum(1 for line in open(rows_file(key)) if line.strip())
    if n < MIN_ROWS and not ACCEPT:
        raise RuntimeError(f"the Aider transcripts gave {n} training rows (expected well over {MIN_ROWS}); "
                           f"see the rows counts in the step log. EXP04_ACCEPT=1 builds anyway.")
    if not (job(name) / "config.json").exists():
        make_job(name, build_config(entry), BIG_MAX_GB if big(entry) else MAX_GB)
    pipeline(step, name)
    polyglot(step, model_dir(name), name, key)
    score, line = poly_score(name), bar()
    p = paired(best_full(), name)
    above(f"    {label(entry)}, 4-bit: Aider Polyglot {score:.1%} against the {line:.1%} bar "
          f"({'met' if score >= line else 'missed'}"
          + (f"; paired {p['diff']:+.1%} [{p['lo']:+.1%}, {p['hi']:+.1%}], {verdict(p)}" if p else "")
          + f"); LiveCodeBench {lcb(name) or 0:.0%}")
    mugge_eval(step, base_dir(key), f"{key}-full", key)
    mugge_eval(step, model_dir(name), name, key)


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
    upload_results()


def run_k6(step: Step) -> None:
    key = best_build()
    polyglot(step, model_dir(build_name(key)), f"{key}-w4-k6", key, "--experts-used 6")
    upload_results()


def side_done(suffix: str) -> Callable[[], bool]:
    return lambda: any(poly(f"{k}-{suffix}") for k in BASES)


BENCH = LOGS / "throughput.json"


def run_bench(step: Step) -> None:
    sh(step, f"$LOBBOT_VLLM_PY scripts/bench_vllm.py --model {model_dir(build_name(best_build()))} --out {BENCH} "
             f"--concurrency 1 32 128")
    upload_results()


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
    for entry in builds():
        try:
            m = model_dir(build_name(entry))
        except (FileNotFoundError, RuntimeError):
            continue
        mtar = NVME / f"{build_name(entry)}.tar"
        if not mtar.exists():
            sh(step, f"tar cf {mtar}.part -C {m.parent} {m.name} && mv {mtar}.part {mtar}")
        up.append((mtar, f"models/exp04-{build_name(entry)}.tar"))
    for src, dst in up:
        sh(step, f"evroc storage bucket get-s3-credentials >/dev/null && "
                 f"evroc storage bucket copy --from {src} --to {BUCKET}/{dst}")
    (LOGS / ".uploaded").write_text(time.strftime("%F %T"))


# "polyglot" is the transcripts run; "" covers Polyglot on the 4-bit model and the two Mugge-format evals
BUILD_MINUTES = {"polyglot": 40, "data": 15, "reap": 10, "heal": 90, "quantize": 20, "eval": 10, "package": 1, "": 70}


def steps() -> list[Step]:
    n_builds = min(2, len({SHAPE[k] for k in WANTED}))
    return [
        Step("VM setup and Aider's benchmark", {"": 65}, run_setup, setup_done),
        Step("base model downloads", {"": 25}, run_download, download_done, download_progress),
        *[Step(f"smoke test: {SHAPE[k]} architecture on vLLM", {"": 20}, run_smoke(k),
               (SMOKE / SHAPE[k] / "result.json").exists) for k in smoke_keys()],
        *[Step(f"{LABELS[k]} full, Polyglot", {"": 45}, run_full(k), (POLY / f"{k}-full.json").exists)
          for k in WANTED],
        *[Step(f"build {i + 1}: heal data, compress, score", BUILD_MINUTES, run_build(i), build_done(i))
          for i in range(n_builds)],
        Step("better build unquantized, Polyglot", {"": 40}, run_before_quant, side_done("bf16")),
        Step("better build with 6 experts, Polyglot", {"": 35}, run_k6, side_done("w4-k6")),
        Step("throughput at 1, 32, 128 at once", {"": 15}, run_bench, BENCH.exists),
        Step("upload to the bucket", {"": 20}, run_upload, (LOGS / ".uploaded").exists),
    ]


# ---------- summary ----------


def pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "–"


def summary() -> str:
    """Aider Polyglot after 1 and 2 attempts, well-formed edits, test timeouts, per language
    and on Mugge's languages, for every run; then the bar with each build's paired verdict,
    the Mugge-format eval, size, LiveCodeBench floor, GPU busy and throughput."""
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]
    head = ["run", "size", "pass 2", "pass 1", "formed", "t/o", *langs, "Mugge langs", "LCB", "GPU"]
    table = [head]
    entries = builds() if CHOICE.exists() else []
    names = ([f"{k}-full" for k in WANTED] + [build_name(e) for e in entries]
             + [f"{k}-bf16" for k in BASES] + [f"{k}-w4-k6" for k in BASES])
    for name in names:
        r = poly(name)
        if not r:
            continue
        size = ""
        alloc = job(name) / "work" / "allocation.json"
        if name in map(build_name, entries) and alloc.exists():
            size = f"{next(iter(read(alloc)['candidates'].values()))['size_gb']:.2f} GB"
        per = r.get("per_language") or {}
        gpu = r.get("gpu") or {}
        table.append([name, size, pct(r.get("pass_rate_2")), pct(r.get("pass_rate_1")),
                      pct(r.get("percent_cases_well_formed")), str(r.get("test_timeouts", "")),
                      *[pct((per.get(lang) or {}).get("pass_rate_2")) for lang in langs], pct(mugge_langs(r)),
                      pct(lcb(name)) if size else "",
                      f"{gpu['busy_pct']:.0f}%" if gpu.get("busy_pct") is not None else ""])
    if len(table) == 1:
        return "No Aider Polyglot results yet."
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    out = ["\n".join("  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(r)).rstrip()
                     for r in table)]
    if CHOICE.exists():
        c, ref = read(CHOICE), best_full()
        out.append(f"\nThe bar: Aider Polyglot {c['bar']:.1%} (90% of the best full model, {ref}), under 10 GB.")
        if c.get("swapped"):
            out.append(f"  {c['swapped']}.")
        for entry in c["builds"]:
            s = poly_score(build_name(entry))
            if s is None:
                continue
            p = paired(ref, build_name(entry))
            line = (f"  {label(entry)}: {s:.1%}, " + ("over the size bar; " if big(entry) else "")
                    + f"{'meets' if s >= c['bar'] else 'misses'} the score bar")
            if p:
                line += (f". Paired with {ref} on {p['n']} exercises: {p['diff']:+.1%} over the bar "
                         f"[{p['lo']:+.1%}, {p['hi']:+.1%}], {verdict(p)}")
            out.append(line)
        rows = []
        for entry in c["builds"]:
            key = base_of(entry)
            full, comp = mugge(f"{key}-full"), mugge(build_name(entry))
            if full and comp and full["fix"] and comp["fix"] is not None:
                rows.append(f"  {label(entry)}: {comp['fix']:.1%} after one fix round vs {full['fix']:.1%} "
                            f"for the full {LABELS[key]} ({comp['fix'] / full['fix']:.0%}); first try "
                            f"{pct(comp['pass'])} vs {pct(full['pass'])}")
        if rows:
            out.append("\nMugge format (tickets in, files out; MultiPL-E py/js/ts/cpp and the C set):")
            out += rows
    if BENCH.exists():
        b = read(BENCH)
        out.append("Throughput (output tok/s): " + ", ".join(f"{k} at once {v['output_tok_s']:.0f}" for k, v in b.items()))
    return "\n".join(out)


def main() -> int:
    return run_plan(steps(), summary, LOGS)


if __name__ == "__main__":
    sys.exit(main())
