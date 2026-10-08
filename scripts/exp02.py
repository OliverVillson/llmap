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
  4. r50w95s-t, the thinking model
  5. the r50w95s-t GGUF and the job results to the bucket

Runs with the system python3 (stdlib only); each step sources .env.vm. Logs are in
~/exp02-logs. The bucket steps need an evroc login first; without one they are
skipped and a re-run picks them up.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.progress import parse  # noqa: E402

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
WIDTH = 30
TTY = sys.stdout.isatty()


class Skip(Exception):
    """A step that cannot run now; the rest of the run goes on."""


@dataclass
class Step:
    name: str
    minutes: dict[str, float]  # time guess per part: pipeline stages, or {"": n}
    run: Callable[["Step"], None]
    done: Callable[[], bool] = lambda: False
    progress: Callable[[], float | None] = lambda: None  # for steps that send no events
    # live state
    state: str = "waiting"  # waiting, running, done, skipped, failed
    part: str = ""
    frac: float = 0.0  # of the current part
    finished: set = field(default_factory=set)
    started: float = 0.0
    part_started: float = 0.0
    log: Path | None = None

    @property
    def total(self) -> float:
        return sum(self.minutes.values())


# ---------- job helpers ----------

def read(p: Path) -> dict:
    return json.loads(p.read_text())


def write(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def new_job(name: str, max_size_gb: float, **cfg) -> Path:
    """A job with experiment 01 r50mix's config, taskspec and finished data stage, plus cfg.
    Every side test also scores one fix turn (fix@1): if the harness loop wins back
    what compression loses, we can compress harder."""
    d, src = J / f"code10x-{name}", M / "jobs" / "code10x-r50mix"
    for sub in (".done", "work", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if not (d / ".done" / "data").exists():
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


def evroc_ok() -> bool:
    if not shutil.which("evroc"):
        return False
    return subprocess.run("evroc storage bucket get-s3-credentials", shell=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


# ---------- running commands ----------

state_lock = threading.RLock()  # the step state and the terminal


def above(text: str) -> None:
    """Prints a line above the bar (the bar is redrawn on the next tick)."""
    with state_lock:
        sys.stdout.write(("\r\033[K" if TTY else "") + text + "\n")
        sys.stdout.flush()


def sh(step: Step, cmd: str, env: dict | None = None) -> None:
    """Runs cmd in bash with .env.vm sourced; its progress events move the bar."""
    full = f"cd {REPO} && {{ [ -f .env.vm ] && source .env.vm; true; }} && {cmd}"
    with open(step.log, "a") as log:
        log.write(f"\n===== {time.strftime('%F %T')} $ {cmd}\n")
        log.flush()
        proc = subprocess.Popen(["bash", "-c", full], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace", env={**os.environ, **(env or {})})
        tail: list[str] = []
        for line in proc.stdout:
            log.write(line)
            log.flush()
            on_line(step, line)
            if line.strip() and parse(line) is None:
                tail = (tail + [line.rstrip()])[-15:]
        if proc.wait() != 0:
            raise RuntimeError(f"`{cmd}` failed:\n" + "\n".join("    " + t for t in tail[-8:]))


def on_line(step: Step, line: str) -> None:
    e = parse(line)
    if e is None:
        return
    stage, status = e["stage"], e["status"]
    with state_lock:
        if stage in step.minutes and stage not in step.finished:
            if status in ("done", "skipped"):
                step.finished.add(stage)
                if status == "done":
                    took = hm(time.time() - step.part_started) if step.part == stage else ""
                    above(f"    {stage} done{f' in {took}' if took else ''}: {e.get('msg', '')}")
                step.part, step.frac = "", 0.0
                return
            if step.part != stage:
                step.part, step.frac, step.part_started = stage, 0.0, time.time()
            if "answered" in e and e.get("total"):
                step.frac = max(step.frac, 0.1 + 0.85 * e["answered"] / e["total"])
            elif e.get("pct") is not None:
                step.frac = max(step.frac, e["pct"] / 100)
        if status == "error":
            above(f"    {stage} error: {e.get('msg', '')}")


# ---------- the bar ----------

def hm(seconds: float) -> str:
    if seconds < 60:
        return "<1m"
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


class Run:
    def __init__(self, steps: list[Step]):
        self.steps, self.t0 = steps, time.time()
        self.done_est = 0.0  # minutes of guesses finished in this run
        self.done_real = 0.0  # and how long they really took

    def fraction_and_left(self) -> tuple[float, float]:
        total = sum(s.total for s in self.steps)
        speed = min(3.0, max(0.5, self.done_real / self.done_est)) if self.done_est > 10 else 1.0
        got, left = 0.0, 0.0
        for s in self.steps:
            if s.state in ("done", "skipped"):
                got += s.total
                continue
            for part, mins in s.minutes.items():
                if s.state == "running" and part in s.finished:
                    got += mins
                elif s.state == "running" and part == s.part:
                    frac = s.frac
                    got += mins * frac
                    ran = time.time() - (s.part_started or s.started)
                    if frac >= 0.05 and ran > 120:
                        left += ran / frac * (1 - frac)
                    else:
                        left += mins * 60 * speed * (1 - frac)
                else:
                    left += mins * 60 * speed
        return got / total, left

    def line(self) -> str:
        frac, left = self.fraction_and_left()
        n = len(self.steps)
        cur = next((s for s in self.steps if s.state == "running"), None)
        what = ""
        if cur:
            i = self.steps.index(cur) + 1
            part = f" · {cur.part} {cur.frac:.0%}" if cur.part else ""
            what = f"  | {i}/{n} {cur.name}{part}"
        filled = int(WIDTH * frac)
        eta = "done" if frac >= 1 else f"~{hm(left)} left" if left >= 60 else "<1m left"
        return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {frac:.0%}  {hm(time.time() - self.t0)} in, {eta}{what}"


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


def run_jobs(step: Step) -> None:
    sh(step, f"python scripts/code10x.py setup --jobs {J} --taskspec {TASKSPEC} && python scripts/code10x.py ref-ggufs")


def jobs_done() -> bool:
    return (J / "code10x-r50w95s-t/config.json").exists() and (NVME / "models/gguf/Qwen3.6-35B-A3B-BF16.gguf").exists()


def eval_done(name: str) -> Callable[[], bool]:
    return lambda: (J / f"code10x-{name}" / ".done" / "eval").exists()


def run_ref_think(step: Step) -> None:
    sh(step, f"python pipeline.py --job {J}/code10x-ref-think --only eval", {"LOBBOT_EVAL_SLOTS": "16"})
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
        sh(step, f"python pipeline.py --job {J}/code10x-{name} --only eval", {"LOBBOT_EVAL_SLOTS": "16"})
    return run


def run_r62w95s(step: Step) -> None:
    needs_exp01()
    d = new_job("r62w95s", 9.5, reap_sparsity=0.625, bit_floor="q2_k", bit_ceiling="q8_0", static_type="q8_0")
    sh(step, f"python pipeline.py --job {d}", {"LOBBOT_EVAL_SLOTS": "16"})


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
    sh(step, f"python pipeline.py --job {dst} --from quantize", {"LOBBOT_EVAL_SLOTS": "16"})


def run_thinking_model(step: Step) -> None:
    score = lcb("ref-think")
    if score is None:
        raise Skip("ref-think has no score yet")
    if score < LCB_FLOOR:
        raise Skip(f"ref-think scored {score:.0%} on LiveCodeBench, far below the model card's 80%, "
                   "so the thinking eval looks broken. Send Claude the summary before spending hours here.")
    sh(step, f"python pipeline.py --job {J}/code10x-r50w95s-t", {"LOBBOT_EVAL_SLOTS": "16"})
    stats = J / "code10x-r50w95s-t/data/stats.json"
    if stats.exists():
        s = read(stats)
        above(f"    data: {s.get('answered')} answered, thinking median {s.get('thinking_chars_median')} chars, "
              f"dropped {s.get('dropped')}")


def run_upload(step: Step) -> None:
    """Every job file under 200 MB (configs, data, scores) and the r50w95s-t GGUF."""
    if not evroc_ok():
        raise Skip("no evroc login. Log in (runbook step D) and re-run to upload.")
    tar = NVME / "exp02-jobs.tar"
    sh(step, f"mkdir -p {J}/_logs_exp02 && cp {LOGS}/*.log {J}/_logs_exp02/ && cd {J.parent} && "
             f"find {J.name} -type f -size -200M -print0 | tar cf {tar} --null -T -")
    up = [(tar, "exp02/exp02-jobs.tar")]
    gguf = J / "code10x-r50w95s-t/out/model.gguf"
    if gguf.exists():
        up.append((gguf, "gguf/qwen36-r50w95s-t.gguf"))
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
                      ("r50w95s-k6", "ref"), ("r62w95s", "ref"), ("r62w75s", "ref")]:
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
    plan = steps()
    if "--list" in sys.argv:
        for i, s in enumerate(plan, 1):
            parts = ", ".join(f"{k} {v:.0f}m" for k, v in s.minutes.items() if k)
            print(f"{i:2}. {s.name:38} ~{hm(s.total * 60):>6}" + (f"  ({parts})" if parts else "")
                  + ("  [done]" if s.done() else ""))
        print(f"    total ~{hm(sum(s.total for s in plan) * 60)} (guesses)")
        return 0
    if "summary" in sys.argv:
        print(summary())
        return 0

    LOGS.mkdir(exist_ok=True)
    run = Run(plan)
    for s in plan:
        s.state = "done" if s.done() else "waiting"
    stop = threading.Event()

    def draw():
        while not stop.wait(1):
            with state_lock:
                for s in plan:  # steps without progress events: their own measure, else the clock
                    if s.state == "running" and "" in s.minutes:
                        p = s.progress()
                        s.frac = p if p is not None else min(0.95, (time.time() - s.started) / (s.total * 60))
                if TTY:
                    sys.stdout.write("\r\033[K" + run.line())
                    sys.stdout.flush()

    def heartbeat():  # without a terminal (e.g. piped to a file), one line every 5 minutes
        while not stop.wait(300):
            above(run.line())

    threading.Thread(target=draw, daemon=True).start()
    if not TTY:
        threading.Thread(target=heartbeat, daemon=True).start()
    skipped, failed = [], None
    for i, s in enumerate(plan, 1):
        if s.state == "done":
            above(f"✓ {i}. {s.name} (done before)")
            continue
        s.log = LOGS / f"{i:02d}-{re.sub(r'[^a-z0-9]+', '-', s.name.lower()).strip('-')}.log"
        with state_lock:
            s.state, s.started, s.part_started = "running", time.time(), time.time()
            s.part = "" if "" in s.minutes else next(iter(s.minutes))
        above(f"▶ {i}. {s.name}  (log: {s.log})")
        try:
            s.run(s)
            took = time.time() - s.started
            with state_lock:
                s.state = "done"
                run.done_est += s.total
                run.done_real += took / 60
            above(f"✓ {i}. {s.name} in {hm(took)}")
        except Skip as e:
            with state_lock:
                s.state = "skipped"
            skipped.append(f"{s.name}: {e}")
            above(f"– {i}. {s.name} skipped: {e}")
        except KeyboardInterrupt:
            stop.set()
            above(f"\nStopped during {s.name}. Re-run the same command to carry on from there.")
            return 130
        except Exception as e:  # noqa: BLE001
            with state_lock:
                s.state = "failed"
            failed = f"{s.name}: {e}"
            above(f"✗ {i}. {s.name} failed: {e}\n  Full log: {s.log}")
            break
    stop.set()
    time.sleep(0.1)
    if TTY:
        sys.stdout.write("\r\033[K" + run.line() + "\n")
    print("\n" + summary())
    if skipped:
        print("\nSkipped:\n" + "\n".join("  " + x for x in skipped))
    if failed:
        print(f"\nStopped at {failed}\nSend Claude this output. Re-running the same command resumes.")
        return 1
    print(f"\nFinished in {hm(time.time() - run.t0)}. Send Claude the table above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
