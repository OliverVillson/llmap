"""Experiment 01 (docs/exp-01-code10x.md): set up, prepare and compare the code10x jobs.

    python scripts/code10x.py setup --jobs $LOBBOT_JOBS --taskspec <code taskspec.json>
    python scripts/code10x.py ref-ggufs              # BF16 and Q8_0 GGUFs of the base for ref/fp8
    python pipeline.py --job $LOBBOT_JOBS/code10x-ref --only eval
    python pipeline.py --job $LOBBOT_JOBS/code10x-fp8 --only eval
    python pipeline.py --job $LOBBOT_JOBS/code10x-r50mix
    python scripts/code10x.py share-data --jobs $LOBBOT_JOBS   # reuse r50mix's tested data
    python pipeline.py --job $LOBBOT_JOBS/code10x-r25q4
    python pipeline.py --job $LOBBOT_JOBS/code10x-r50mix-gen
    python scripts/code10x.py report --jobs $LOBBOT_JOBS

The job configs are configs/code10x/<variant>.json. Each compressed variant's
size budget lives in its TaskSpec target, so setup writes the taskspec per job
with TARGETS below. Every job also needs the code suites on the VM
(scripts/fetch_codebench.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stages._util import Config, Job  # noqa: E402

VARIANTS = ["ref", "fp8", "r25q4", "r50mix", "r50mix-gen"]
CONFIGS = ROOT / "configs" / "code10x"
# Size budget per compressed variant (GB, with quantize's margin under it). Laptop
# speed is not what this experiment measures, so the tok/s floor is low.
TARGETS = {
    "r25q4": {"max_size_gb": 16.0, "min_tok_s": 10.0},
    "r50mix": {"max_size_gb": 7.5, "min_tok_s": 10.0},
    "r50mix-gen": {"max_size_gb": 7.5, "min_tok_s": 10.0},
}
# The pass bar from the experiment doc: (mean share of ref, lowest suite share).
BARS = {"r50mix": (0.90, 0.80), "r25q4": (0.95, None)}
GGUF_TYPES = {"ref": "bf16", "fp8": "q8_0"}  # llama.cpp has no FP8 type; Q8_0 is its 8-bit stand-in


def job_dir(jobs: str, variant: str) -> Path:
    return Path(jobs) / f"code10x-{variant}"


def config(variant: str) -> dict:
    return json.loads((CONFIGS / f"{variant}.json").read_text())


def setup(args) -> None:
    spec = json.loads(Path(args.taskspec).read_text())
    for v in VARIANTS:
        d = job_dir(args.jobs, v)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(config(v), indent=2) + "\n")
        s = json.loads(json.dumps(spec))
        s["target"] = {**s.get("target", {}), **TARGETS.get(v, {})}
        (d / "taskspec.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        Job(d).spec  # validates config and taskspec
        print(f"{d}: ready")


def ref_ggufs(args) -> None:
    """Convert the base model to the GGUFs the ref and fp8 jobs evaluate."""
    for v, outtype in GGUF_TYPES.items():
        cfg = Config(**config(v))
        out = Path(next(iter(cfg.eval_candidates.values())))
        if out.exists():
            print(f"{out}: exists")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        lc = Path(cfg.llama_cpp)
        if outtype == "bf16":
            src = Path(cfg.models_dir) / cfg.teacher.split("/")[-1]
            cmd = [sys.executable, str(lc / "convert_hf_to_gguf.py"), str(src), "--outtype", "bf16", "--outfile", str(out)]
        else:
            bf16 = Path(next(iter(Config(**config("ref")).eval_candidates.values())))
            cmd = [str(lc / "build" / "bin" / "llama-quantize"), str(bf16), str(out), outtype.upper()]
        print("$ " + " ".join(cmd), flush=True)
        if not args.print_only:
            subprocess.run(cmd, check=True)


def share_data(args) -> None:
    """Copy the finished data stage of one job into the other compressed jobs, so
    the ref model answers and tests the ~2k coding tasks once."""
    src = job_dir(args.jobs, args.source)
    if not (src / ".done" / "data").exists():
        sys.exit(f"{src} has not finished its data stage yet")
    for v in TARGETS:
        if v == args.source:
            continue
        d = job_dir(args.jobs, v)
        if (d / ".done" / "data").exists():
            print(f"{d}: data already done")
            continue
        shutil.copytree(src / "data", d / "data", dirs_exist_ok=True)
        (d / ".done").mkdir(exist_ok=True)
        shutil.copy(src / ".done" / "data", d / ".done" / "data")
        print(f"{d}: data copied from {args.source}")


def pick(rep: dict, variant: str) -> dict | None:
    cands = {c["name"]: c for c in rep.get("candidates", [])}
    return cands.get(variant) or cands.get("lobbot-moe") or cands.get(rep.get("winner"))


def report(args) -> int:
    rows, suites = [], []
    for v in VARIANTS:
        p = job_dir(args.jobs, v) / "out" / "eval.json"
        if not p.exists():
            rows.append((v, None))
            continue
        c = pick(json.loads(p.read_text()), v)
        rows.append((v, c))
        for s in ((c or {}).get("code") or {}).get("suites", {}):
            if s not in suites:
                suites.append(s)
    head = ["variant", "size"] + suites + ["mean", "vs ref", "lowest", "bar"]
    table = [head]
    failed = False
    for v, c in rows:
        code = (c or {}).get("code") or {}
        if not code:
            table.append([v, "–"] + ["not run"] + [""] * (len(head) - 3))
            continue
        share = code.get("share_of_ref") or {}
        cells = [v, f"{c['size_gb']:.1f} GB"]
        for s in suites:
            r = code["suites"].get(s) or {}
            cells.append("skipped" if r.get("skipped") else f"{r['pass@1']:.0%}" if "pass@1" in r else "–")
        mean_p = code.get("mean_pass@1")
        cells.append(f"{mean_p:.0%}" if mean_p is not None else "–")
        cells += [f"{share['mean']:.0%}" if share else ("100%" if v == "ref" else "–"),
                  f"{share['min']:.0%}" if share else "–"]
        bar = BARS.get(v)
        if bar and share:
            ok = share["mean"] >= bar[0] and (bar[1] is None or share["min"] >= bar[1])
            failed |= not ok
            cells.append(("pass" if ok else "MISS") + f" (≥{bar[0]:.0%}" + (f", none <{bar[1]:.0%})" if bar[1] else ")"))
        else:
            cells.append("")
        table.append(cells)
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    for r in table:
        print("  ".join(x.ljust(widths[i]) if i < 2 else x.rjust(widths[i]) for i, x in enumerate(r)).rstrip())
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    jobs = os.environ.get("LOBBOT_JOBS", "/mnt/nvme/jobs")
    s = sub.add_parser("setup")
    s.add_argument("--jobs", default=jobs)
    s.add_argument("--taskspec", required=True, help="a code TaskSpec (task_type code)")
    g = sub.add_parser("ref-ggufs")
    g.add_argument("--print-only", action="store_true")
    d = sub.add_parser("share-data")
    d.add_argument("--jobs", default=jobs)
    d.add_argument("--source", default="r50mix", choices=list(TARGETS))
    r = sub.add_parser("report")
    r.add_argument("--jobs", default=jobs)
    args = ap.parse_args()
    return {"setup": setup, "ref-ggufs": ref_ggufs, "share-data": share_data, "report": report}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
