#!/usr/bin/env python3
"""Project Centipede's code agent, run 1, on one B200: Granite 4.2 3B taught by Granite
4.2 30B to work as a Mugge-format coding agent with short thinking, then compressed to
one GGUF of at most 1.5 GB that a single llama-server serves to 40 agents at once.

    python3 scripts/centipede.py             # everything; re-run after a stop and finished steps are skipped
    python3 scripts/centipede.py --shutdown  # ... then power off once the results are uploaded
    python3 scripts/centipede.py --list      # the steps and their time guesses
    python3 scripts/centipede.py summary     # the results table

The agent's job is Mugge's harness loop (stages/harness.py): a ticket in, each owned file
in full out, and after a failed check one fix call with the failing output. Thinking is
on but short, AGENT_TOKENS per call (then the answer is forced), so 40 agents stay fast.

  1-3  VM setup (scripts/setup_vm.sh without Aider Polyglot), both models, and the
       teacher's rows from the bucket when an earlier VM made them
  4    agent eval of the 3B as released and of the 30B teacher (the ceiling)
  5-8  teacher data, one job per language (stages/data.py): the 30B writes tasks with
       tests in Python, C, JavaScript and TypeScript, answers each as a Mugge ticket with
       short thinking and fixes failing drafts; the Python job adds CodeContests problems.
       Only answers that pass their tests are kept
  9    the rows merged, and the tasks with tests as the loop round's problem set
  10-11 SFT of the 3B on the rows (LoRA on every projection, merged; stages/heal.py), eval
  12-14 one round in the loop: the SFT model answers each task 4 times, its shortest
       passing answer per task joins the teacher's rows, SFT again from the 3B, eval
  15   the agent: the better of the two by mean fix@1; its training rows as trace text
  16-19 compression with Event Horizon's stages: BF16 GGUF and imatrix, the horizon PTQ
       map at BUDGET_GB, Block-AP, end-to-end KD from the bf16 agent, export
  20   agent eval of the shipped GGUF on llama-server
  21   40 agents at once on one llama-server (16k context each, 8-bit KV cache)
  22   upload

Agent eval: MultiPL-E (Python, JS, TS, C++), our C set and LiveCodeBench 2025-01..04, every
problem asked as a Mugge ticket: one answer (pass@1), and one fix call with the failing
output (fix@1, the loop's score).

Results go to bucket://mugge-library/centipede/<run>/ after every step (CENTIPEDE_RUN names
the run): results.tar (every result and log), the teacher's rows and the round's tasks
(restored on a fresh VM, so the teacher's data is made once), agent.gguf. Logs are in
~/centipede-logs. Runs with the system python3 (stdlib only); the stages run in the
venvs .env.vm sets up.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from runner import REPO, Skip, Step, evroc_ok, run_plan, sh

NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
W = Path(os.environ.get("LOBBOT_JOBS", NVME / "jobs")) / "centipede"
LOGS = Path(os.environ.get("CENTIPEDE_LOGS", Path.home() / "centipede-logs"))
RUN = os.environ.get("CENTIPEDE_RUN", "cp1")
BUCKET = "bucket://mugge-library"
STUDENT, TEACHER = "ibm-granite/granite-4.2-3b", "ibm-granite/granite-4.2-30b"
MODELS = NVME / "models"
STUDENT_DIR, TEACHER_DIR = MODELS / "granite-4.2-3b", MODELS / "granite-4.2-30b"
LLAMA_CPP = NVME / "llama.cpp"
LLAMA_CPP_SHA = "af50f2f248b475d9760f8e67503d01c0f194f5f4"  # Event Horizon's
UV_EXCLUDE_NEWER = "2026-10-12T00:00:00Z"  # Event Horizon's: PyPI as it stood then
CODEBENCH = NVME / "codebench"
SETUP_SUITES = ("livecodebench", "codecontests", "apps")  # each with its meta; multipl-e is checked by file
LLAMA_BINS = ("llama-server", "llama-quantize", "llama-imatrix", "llama-perplexity", "libllama-perplexity-impl.so")

AGENT_TOKENS = 4096  # thinking plus answer per call; a longer thinking is closed and the answer forced
LANGS = {"python": "python-utils", "c": "c-strings", "javascript": "javascript-utils", "typescript": "typescript-utils"}
BUDGET_GB = 1.5  # the shipped GGUF, in 1e9 bytes
SLOTS, SLOT_CTX = 40, 16384  # the serving check: 40 agents with 16k tokens each
ROUND_PER_LANGUAGE, ROUND_SAMPLES = 600, 4

EVAL = {"teacher": STUDENT, "models_dir": str(MODELS), "code_eval_dir": str(CODEBENCH),
        "code_eval_suites": ["multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "c-set", "livecodebench"],
        "lcb_since": "2025-01-01", "lcb_until": "2025-04-30", "lcb_max_cases": 50,
        "code_eval_format": "harness", "code_eval_fix": True, "code_eval_samples": 1,
        "code_eval_thinking": True, "code_eval_thinking_temperature": 1.0, "thinking_top_k": 0, "eval_min_p": 0.0,
        "code_eval_max_tokens": AGENT_TOKENS, "eval_reasoning_parser": "qwen3"}
# Granite's card sampling for thinking (temperature 1.0, top_p 0.95, no top_k), as the eval
DATA = {"teacher": TEACHER, "models_dir": str(MODELS), "code_eval_dir": str(CODEBENCH),
        "n_generate": 800, "n_heldout": 40, "data_thinking": True, "data_thinking_temperature": 1.0,
        "thinking_top_k": 0, "data_answer_max_tokens": AGENT_TOKENS, "data_max_len": 12288,
        "data_harness_share": 1.0, "data_fix_rows": 300}
CONTEST = {"data_contest_rows": 300, "data_contest_samples": 2, "data_contest_source": "codecontests",
           "data_contest_harness_share": 1.0}  # the Python job only
SFT = {"teacher": TEACHER, "student": str(STUDENT_DIR), "models_dir": str(MODELS), "student_epochs": 2.0,
       "student_lr": 1e-4, "heal_max_len": 8192, "student_max_minutes": 90}
BLOCK = {"windows": 4096, "length": 1024, "epochs": 2, "lr": 1e-4}  # the agent's rows are ~1-4k tokens
E2E = {"tokens": 20e6, "length": 8192, "max_minutes": 60}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                   "chat_template.json", "generation_config.json", "vocab.json", "merges.txt")

ROWS = W / "rows" / "teacher.jsonl"  # the teacher's rows, merged
POOL = W / "round" / "bench" / "agent-train.jsonl"  # the round's tasks, a code suite
PICK = W / "pick.json"
C = W / "compress"
FINAL = C / "final" / "model.gguf"
SERVE = W / "serve" / "serve.json"


# ---------- small helpers ----------

def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def dump(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def make_job(d: Path, cfg: dict, spec: str = "python-utils") -> Path:
    """A pipeline job dir with this config.json and a code taskspec; a changed config
    clears the job's finished stages, so nothing made under the old one is reused."""
    for sub in (".done", "work", "out", "data"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if load(d / "config.json") != cfg:
        for f in (d / ".done").iterdir():
            f.unlink()
        dump(d / "config.json", cfg)
    if not (d / "taskspec.json").exists():
        s = json.loads((REPO / "examples" / f"{spec}.code.taskspec.json").read_text())
        s["target"] = {**s.get("target", {}), "max_size_gb": 10.0}
        dump(d / "taskspec.json", s)
    return d


def gpu_free(tries: int = 120, every: int = 5, limit_mib: int = 2000) -> str:
    """Shell that waits for the GPU's memory to come back after a stopped server, and
    fails, naming the holders, if it does not."""
    used = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1"
    return (f"for i in $(seq {tries}); do m=$({used}); [ -n \"$m\" ] && [ \"$m\" -lt {limit_mib} ] && break; "
            f"sleep {every}; done\n"
            f"m=$({used}); if [ -z \"$m\" ] || [ \"$m\" -ge {limit_mib} ]; then "
            f"echo \"GPU not free after {tries * every}s: ${{m:-no reading of}} MiB in use. Holders:\"; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; fi")


def gpu(cmd: str) -> str:
    return f"set -e\n{gpu_free()}\n{cmd}"


# ---------- setup ----------

def run_setup(step: Step) -> None:
    sh(step, f"NVME={NVME} TEACHER={TEACHER} STUDENT={STUDENT} LLAMA_CPP_SHA={LLAMA_CPP_SHA} "
             f"UV_EXCLUDE_NEWER={UV_EXCLUDE_NEWER} SKIP_POLYGLOT=1 bash scripts/setup_vm.sh")
    missing = setup_missing()
    if missing:
        raise RuntimeError("setup_vm.sh finished without " + ", ".join(missing) + "; re-run to fetch what is missing")


def setup_missing() -> list[str]:
    out = [] if (REPO / ".env.vm").exists() else [".env.vm"]
    out += [b for b in LLAMA_BINS if not (LLAMA_CPP / "build" / "bin" / b).exists()]
    out += [f"{s} suite" for s in SETUP_SUITES if not (CODEBENCH / f"{s}.meta.json").exists()]
    out += [f"{s} suite" for s in ("multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp")
            if not (CODEBENCH / f"{s}.jsonl").exists()]
    out += [] if (NVME / "node" / "bin" / "tsc").exists() or shutil.which("tsc") else ["tsc (TypeScript)"]
    return out


def setup_done() -> bool:
    return not setup_missing()


DOWNLOAD_RESTARTS = 3


def downloaded() -> bool:
    log = NVME / "download.log"
    return (log.exists() and "DOWNLOADS_DONE" in log.read_text(errors="replace")
            and (STUDENT_DIR / "config.json").exists() and (TEACHER_DIR / "config.json").exists())


def run_download(step: Step) -> None:
    """Waits for setup_vm.sh's background download of both models, starting it again (up
    to DOWNLOAD_RESTARTS times) when it stopped without finishing."""
    restarts = 0
    while not downloaded():
        if subprocess.run(["pgrep", "-f", "hf download"], stdout=subprocess.DEVNULL, check=False).returncode != 0:
            time.sleep(10)
            if downloaded():
                break
            if restarts >= DOWNLOAD_RESTARTS:
                raise RuntimeError(f"the model download stopped {restarts + 1} times; see {NVME / 'download.log'}")
            restarts += 1
            sh(step, "export PATH=$HOME/.local/bin:$PATH; setsid nohup bash -c '"
                     f"uv tool run --from huggingface_hub hf download {TEACHER} --local-dir {TEACHER_DIR} && "
                     f"uv tool run --from huggingface_hub hf download {STUDENT} --local-dir {STUDENT_DIR} && "
                     f"echo DOWNLOADS_DONE' >> {NVME / 'download.log'} 2>&1 < /dev/null &")
        time.sleep(5)
    for d in (STUDENT_DIR, TEACHER_DIR):
        arch = (load(d / "config.json") or {}).get("architectures")
        if arch != ["GraniteForCausalLM"]:
            raise RuntimeError(f"{d} is {arch}, not GraniteForCausalLM: the stages assume Granite's dense layout")


def download_progress() -> float:
    got = sum(f.stat().st_size for d in (STUDENT_DIR, TEACHER_DIR) if d.exists() for f in d.rglob("*") if f.is_file())
    return min(0.99, got / 70e9)


# ---------- the bucket ----------

def bucket(name: str) -> str:
    return f"{BUCKET}/centipede/{RUN}/{name}"


def evroc_copy(src: str, dst: str) -> bool:
    """One `evroc storage bucket copy` either way, after refreshing the S3 credentials;
    an upload is tried three times."""
    if not shutil.which("evroc"):
        return False
    for k in range(1 if src.startswith("bucket://") else 3):
        ok = all(subprocess.run(["evroc", "storage", "bucket", *a], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                check=False).returncode == 0
                 for a in (["get-s3-credentials"], ["copy", "--from", src, "--to", dst]))
        if ok:
            return True
        time.sleep(5 * (k + 1))
    return False


def get(name: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if not evroc_copy(bucket(name), str(tmp)) or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def run_restore(step: Step) -> None:
    """The teacher's rows and the round's tasks from an earlier VM's run, when the bucket
    has both: the data steps then count as done."""
    if not evroc_ok():
        raise Skip("no evroc login: nothing restored (the teacher's data is made on this VM)")
    if get("teacher-rows.jsonl", ROWS) and get("agent-train.jsonl", POOL):
        print(f"restored {ROWS} and {POOL} from {bucket('')}", flush=True)
    dump(W / ".restored", {"time": time.strftime("%F %T"), "rows": ROWS.exists()})


# ---------- agent eval ----------

def eval_dir(name: str) -> Path:
    return W / "eval" / name


def run_eval(step: Step, name: str, models: dict[str, Path]) -> None:
    d = make_job(eval_dir(name), {**EVAL, "eval_candidates": {k: str(v) for k, v in models.items()}})
    if not (d / ".done" / "eval").exists():
        sh(step, gpu(f"python pipeline.py --job {d} --only eval"))


def evaled(name: str) -> bool:
    return (eval_dir(name) / ".done" / "eval").exists()


def scores(name: str) -> dict[str, dict]:
    """{candidate: its code results} from an eval job, {} before it ran."""
    rep = load(eval_dir(name) / "out" / "eval.json") or {}
    return {c["name"]: {**(c.get("code") or {}), "size_gb": c.get("size_gb")} for c in rep.get("candidates", [])}


# ---------- teacher data ----------

def data_dir(lang: str) -> Path:
    return W / "data" / lang


def run_data(step: Step, lang: str) -> None:
    d = make_job(data_dir(lang), {**DATA, **(CONTEST if lang == "python" else {})}, LANGS[lang])
    # 1024 prompts per vLLM call, not 256: each call waits for its longest answer
    sh(step, gpu(f"LOBBOT_DATA_BATCH=1024 python pipeline.py --job {d} --only data"))


def data_done(lang: str) -> bool:
    return merged() or (data_dir(lang) / ".done" / "data").exists()


def merged() -> bool:
    return ROWS.exists() and POOL.exists()


def run_merge(step: Step) -> None:
    """All languages' train rows in one file, shuffled with a fixed seed, and their
    tasks with tests (less each job's held-out ones) as the round's problem set."""
    jobs = [data_dir(lang) for lang in LANGS]
    rows = [line for d in jobs for line in (d / "data" / "train.jsonl").read_text().splitlines() if line.strip()]
    random.Random(0).shuffle(rows)
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROWS.with_name(ROWS.name + ".tmp")
    tmp.write_text("\n".join(rows) + "\n")
    sh(step, f"python -m stages.agent pool --jobs {' '.join(map(str, jobs))} --per-language {ROUND_PER_LANGUAGE} "
             f"--out {POOL}")
    tmp.replace(ROWS)


# ---------- SFT ----------

def sft_dir(name: str) -> Path:
    return W / "sft" / name


def sft_model(name: str) -> Path:
    return sft_dir(name) / "work" / "dense"


def sft_rows(name: str) -> Path:
    return sft_dir(name) / "data" / "train.jsonl"


def run_sft(step: Step, name: str, extra: Path | None = None) -> None:
    """LoRA SFT of the 3B (attention and MLP, merged) on the teacher's rows plus extra;
    then the 3B's own tokenizer files over the saved ones, so the GGUF converter and the
    servers read exactly the released tokenizer and chat template."""
    d = make_job(sft_dir(name), SFT)
    out = sft_model(name)
    if not (out / "config.json").exists():
        lines = [x for p in (ROWS, extra) if p for x in p.read_text().splitlines() if x.strip()]
        random.Random(1).shuffle(lines)
        sft_rows(name).write_text("\n".join(lines) + "\n")
        sh(step, gpu(f"python -m stages.heal --job {d} --part dense"))
    if not (out / "config.json").exists():
        raise RuntimeError(f"SFT wrote no model to {out}")
    for f in TOKENIZER_FILES:
        if (STUDENT_DIR / f).exists():
            shutil.copyfile(STUDENT_DIR / f, out / f)
    arch = (load(out / "config.json") or {}).get("architectures")
    if arch != ["GraniteForCausalLM"]:
        raise RuntimeError(f"the SFT model in {out} is {arch}, not GraniteForCausalLM")
    dump(sft_dir(name) / "sft.json", {"rows": len(sft_rows(name).read_text().splitlines()), "time": time.strftime("%F %T")})


def sft_done(name: str) -> bool:
    return (sft_dir(name) / "sft.json").exists()


# ---------- the loop round ----------

ROUND = W / "round"


def run_round(step: Step) -> None:
    """The SFT model answers every task of the pool ROUND_SAMPLES times as a Mugge ticket
    (thinking kept); its passing answers become training rows (stages/agent.py round)."""
    d = make_job(ROUND / "answers", {**EVAL, "code_eval_dir": str(POOL.parent), "code_eval_suites": [POOL.stem],
                                     "code_eval_fix": False, "code_eval_samples": ROUND_SAMPLES,
                                     "code_eval_budgets": [AGENT_TOKENS],  # keeps each answer's thinking
                                     "eval_candidates": {"sft": str(sft_model("sft"))}})
    if not (d / ".done" / "eval").exists():
        sh(step, gpu(f"python pipeline.py --job {d} --only eval"))
    sh(step, f"python -m stages.agent round --answers {d}/work/code_eval/sft.jsonl --suite {POOL} "
             f"--out {ROUND / 'rows.jsonl'}")


def round_done() -> bool:
    return (ROUND / "rows.jsonl").exists()


# ---------- the agent and its compression ----------

def fix1(name: str, cand: str) -> float | None:
    return (scores(name).get(cand) or {}).get("mean_fix@1")


def run_pick(step: Step) -> None:
    """The agent is the round model when its mean fix@1 is at least the SFT model's;
    its training rows, tokenized as at inference, are the compression's trace text."""
    sft, rnd = fix1("sft", "sft"), fix1("round", "round")
    if sft is None or rnd is None:
        raise RuntimeError(f"no mean fix@1 to pick by (sft {sft}, round {rnd})")
    name = "round" if rnd >= sft else "sft"
    dump(PICK, {"agent": name, "dir": str(sft_model(name)), "rows": str(sft_rows(name)),
                "fix@1": {"sft": sft, "round": rnd}})
    C.mkdir(parents=True, exist_ok=True)
    sh(step, f"python -m stages.agent traces --hf {sft_model(name)} --rows {sft_rows(name)} --out {C / 'traces.jsonl'}")


def pick() -> dict:
    p = load(PICK)
    if not p:
        raise RuntimeError(f"{PICK} missing: the pick step has not run")
    return p


def pick_done() -> bool:
    return PICK.exists() and (C / "traces.json").exists()


def run_imatrix_ptq(step: Step) -> None:
    """The agent's BF16 GGUF and imatrix (from its trace text), then the horizon run: a
    PTQ file per type, the KL sweep on held-out text, the knapsack to BUDGET_GB."""
    a, j = pick()["dir"], make_job(C / "job", {"teacher": STUDENT, "models_dir": str(MODELS)})
    bf, t, p = C / "agent-bf16.gguf", C / "traces.jsonl", C / "ptq"
    sh(step, f"python -m stages.dense gguf --job {j} --hf {a} --bf16 {bf}")
    sh(step, f"python -m stages.dense calib --job {j} --traces {t} --tokenizer {a}")
    sh(step, gpu(f"python -m stages.dense imatrix --job {j} --bf16 {bf}"))
    sh(step, gpu(f"python -m stages.horizon run --hf {a} --bf16 {bf} --traces {t} --llama-cpp {LLAMA_CPP} "
                 f"--imatrix {j}/work/imatrix.gguf --budget-gb {BUDGET_GB} --out {p} || [ $? -eq 3 ]"))
    m = load(p / "map.json")
    if not m:
        floor = load(p / "floor.json") or {}
        raise RuntimeError(f"no PTQ map within {BUDGET_GB} GB: the cheapest map is "
                           f"{floor.get('floor_bytes', 0) / 1e9:.3f} GB ({p}/floor.json)")
    if m["bytes"] > BUDGET_GB * 1e9:
        raise RuntimeError(f"the PTQ map is {m['bytes'] / 1e9:.3f} GB, over the {BUDGET_GB} GB budget")


def ptq_done() -> bool:
    return (C / "ptq" / "map.json").exists()


def run_blockap(step: Step) -> None:
    b = BLOCK
    sh(step, gpu(f"python -m stages.qat block --base {pick()['dir']} --map {C}/ptq/map.json --gguf {C}/ptq/ptq/*.gguf "
                 f"--traces {C}/traces.jsonl --llama-cpp {LLAMA_CPP} --windows {b['windows']} --length {b['length']} "
                 f"--epochs {b['epochs']} --lr {b['lr']} --out {C}/blockap"))


def blockap_done() -> bool:
    return (C / "blockap" / "block.json").exists()


def run_e2e(step: Step) -> None:
    e = E2E
    sh(step, gpu(f"python -m stages.qat e2e --state {C}/blockap/state --teacher {pick()['dir']} "
                 f"--traces {C}/traces.jsonl --tokens {e['tokens']:.0f} --length {e['length']} "
                 f"--max-minutes {e['max_minutes']} --out {C}/e2e"))


def e2e_done() -> bool:
    return (C / "e2e" / "e2e.json").exists()


def run_export(step: Step) -> None:
    """The packed GGUF of the trained weights, held against the trainer bit for bit and
    against the map's size (stages/qat.py export)."""
    m = load(C / "ptq" / "map.json") or {}
    sh(step, gpu(f"python -m stages.qat export --state {C}/e2e/state --out {FINAL.parent} --llama-cpp {LLAMA_CPP} "
                 f"--no-vllm --expect-gb {m['bytes'] / 1e9:.6f}"))
    exact = (load(FINAL.parent / "export.json") or {}).get("exact") or {}
    if not exact or not all(exact.values()):
        raise RuntimeError(f"the export in {FINAL.parent} is not the trainer's weights bit for bit")
    if FINAL.stat().st_size > BUDGET_GB * 1e9:
        raise RuntimeError(f"{FINAL} is {FINAL.stat().st_size / 1e9:.3f} GB, over the {BUDGET_GB} GB budget")


def export_done() -> bool:
    return FINAL.exists() and (FINAL.parent / "export.json").exists()


# ---------- 40 agents ----------

def run_serve(step: Step) -> None:
    sh(step, gpu(f"python -m stages.agent serve-check --gguf {FINAL} --llama-cpp {LLAMA_CPP} --bench {CODEBENCH} "
                 f"--slots {SLOTS} --ctx {SLOT_CTX} --kv q8_0 --max-tokens {AGENT_TOKENS} --out {SERVE}"))


# ---------- upload ----------

STATE = {"final_upload": False}


def upload(step: Step | None, final: bool) -> None:
    """After every step: every small result and the logs as results.tar; the teacher's
    rows and the round's tasks once, when they exist; the agent GGUF once it is exported."""
    if not evroc_ok():
        raise RuntimeError("no evroc login: nothing uploaded (log in with evroc login --oidc-port 8765)")
    W.mkdir(parents=True, exist_ok=True)
    tar = NVME / "centipede-results.tar"
    big = ("*.safetensors", "*.gguf", "*.pt", "*.bin", "*.part", "*.tmp")
    names = " ".join(f"! -name '{g}'" for g in big)
    sh_quiet(f"cd {W.parent} && {{ find {W.name} -type f -size -300M {names} -print0; "
             f"find {LOGS} -type f -print0; }} | tar cf {tar}.part --null -T - && mv {tar}.part {tar}")
    if not evroc_copy(str(tar), bucket("results.tar")):
        raise RuntimeError(f"upload of {tar} failed")
    for src, name in ((ROWS, "teacher-rows.jsonl"), (POOL, "agent-train.jsonl"), (FINAL, "agent.gguf")):
        mark = W / ".uploaded" / name
        if src.exists() and (not mark.exists() or mark.read_text() != str(src.stat().st_size)):
            if (src is not FINAL or export_done()) and evroc_copy(str(src), bucket(name)):
                mark.parent.mkdir(exist_ok=True)
                mark.write_text(str(src.stat().st_size))
    STATE["final_upload"] = final


def sh_quiet(cmd: str) -> None:
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"`{cmd}` failed: {r.stderr.strip()[-400:]}")


def run_upload(step: Step) -> None:
    upload(step, False)
    if export_done() and not (W / ".uploaded" / "agent.gguf").exists():
        raise RuntimeError(f"the agent GGUF did not reach {bucket('agent.gguf')}")
    dump(W / ".uploaded" / "done.json", {"time": time.strftime("%F %T")})


# ---------- the plan ----------

def steps() -> list[Step]:
    s = [
        Step("VM setup", {"": 45}, run_setup, setup_done),
        Step("Granite 3B and 30B download", {"": 15}, run_download, downloaded, download_progress),
        Step("the teacher's rows from the bucket", {"": 2}, run_restore, lambda: (W / ".restored").exists()),
        Step("agent eval: the 3B and the 30B", {"eval": 40},
             lambda st: run_eval(st, "base", {"granite-3b": STUDENT_DIR, "granite-30b": TEACHER_DIR}),
             lambda: evaled("base")),
    ]
    for lang in LANGS:
        s.append(Step(f"teacher data: {lang}", {"data": 40 if lang == "python" else 25},
                      lambda st, lang=lang: run_data(st, lang), lambda lang=lang: data_done(lang)))
    s += [
        Step("rows merged, the round's tasks", {"": 2}, run_merge, merged),
        Step("SFT of the 3B", {"heal": 45}, lambda st: run_sft(st, "sft"), lambda: sft_done("sft")),
        Step("agent eval: SFT", {"eval": 20}, lambda st: run_eval(st, "sft", {"sft": sft_model("sft")}),
             lambda: evaled("sft")),
        Step("loop round: the SFT model's own answers", {"eval": 25}, run_round, round_done),
        Step("SFT with the round's rows", {"heal": 45}, lambda st: run_sft(st, "round", ROUND / "rows.jsonl"),
             lambda: sft_done("round")),
        Step("agent eval: round", {"eval": 20}, lambda st: run_eval(st, "round", {"round": sft_model("round")}),
             lambda: evaled("round")),
        Step("pick the agent, its trace text", {"": 5}, run_pick, pick_done),
        Step(f"imatrix and PTQ map at {BUDGET_GB} GB", {"": 40}, run_imatrix_ptq, ptq_done),
        Step("Block-AP", {"": 20}, run_blockap, blockap_done),
        Step("end-to-end KD", {"": E2E["max_minutes"] + 5}, run_e2e, e2e_done),
        Step("export the GGUF", {"": 10}, run_export, export_done),
        Step(f"agent eval: the {BUDGET_GB} GB GGUF", {"eval": 30},
             lambda st: run_eval(st, "agent", {"agent": FINAL}), lambda: evaled("agent")),
        Step(f"{SLOTS} agents on one llama-server", {"": 10}, run_serve, SERVE.exists),
        Step("upload", {"": 10}, run_upload, lambda: (W / ".uploaded" / "done.json").exists()),
    ]
    return s


# ---------- summary ----------

def pct(x) -> str:
    return "  –  " if x is None else f"{100 * x:5.1f}"


def summary() -> str:
    out = ["Agent eval: pass@1 / fix@1 (one fix call), thinking capped at "
           f"{AGENT_TOKENS} tokens, every problem as a Mugge ticket", ""]
    rows = [("base", "granite-3b", "3B as released"), ("base", "granite-30b", "30B teacher"), ("sft", "sft", "SFT"),
            ("round", "round", "SFT + round"), ("agent", "agent", "agent GGUF")]
    suites = EVAL["code_eval_suites"]
    out.append(f"{'model':16} {'GB':>6}  {'mean':>11}  " + "  ".join(f"{s.replace('multipl-e-', 'mpl-'):>11}" for s in suites))
    for job, cand, label in rows:
        c = scores(job).get(cand)
        if not c:
            continue
        per = [((c.get("suites") or {}).get(s) or {}) for s in suites]
        out.append(f"{label:16} {c.get('size_gb') or 0:6.2f}  {pct(c.get('mean_pass@1'))}/{pct(c.get('mean_fix@1'))}  "
                   + "  ".join(f"{pct(p.get('pass@1'))}/{pct(p.get('fix@1'))}" for p in per))
    for lang in LANGS:
        st = load(data_dir(lang) / "data" / "stats.json")
        if st:
            out.append(f"teacher data {lang}: {st.get('train')} rows ({st.get('harness_write_rows')} write, "
                       f"{st.get('fix_rows')} fix, {st.get('contest_rows', 0)} contest); pass rate "
                       f"{st.get('sandbox_pass_rate')}")
    if (ROUND / "rows.jsonl").exists():
        out.append(f"loop round: {len((ROUND / 'rows.jsonl').read_text().splitlines())} rows from the SFT model's own answers")
    p = load(PICK)
    if p:
        out.append(f"agent: the {p['agent']} model (mean fix@1 SFT {p['fix@1']['sft']}, round {p['fix@1']['round']})")
    m = load(C / "ptq" / "map.json")
    if m:
        out.append(f"PTQ map: {m['bytes'] / 1e9:.3f} GB of a {BUDGET_GB} GB budget")
    if FINAL.exists():
        out.append(f"shipped GGUF: {FINAL} ({FINAL.stat().st_size / 1e9:.3f} GB)")
    s = load(SERVE)
    if s:
        g = s.get("gpu_mib") or {}
        out.append(f"{s['slots']} agents x {s['ctx']} tokens ({s['kv']} KV) on one llama-server: {s['completed']}/"
                   f"{s['requests']} answered in {s['wall_s']} s, {s['tok_s_all']} tok/s over all, "
                   f"{s['tok_s_per_agent_median']} tok/s per agent; GPU memory {g.get('peak')} MiB "
                   f"({g.get('before')} before); {s['wrote_file']} wrote their file, {s['passed']} passed")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    shutdown = "--shutdown" in argv
    if "--list" in argv or "summary" in argv:
        return run_plan(steps(), summary, LOGS, argv)
    if shutdown and not evroc_ok():
        print("Refusing to start with --shutdown: no evroc login, so nothing could be uploaded before the VM "
              "powers off. Log in (evroc login --oidc-port 8765) and re-run.")
        return 2
    if not evroc_ok():
        print("WARNING: no evroc login. Nothing is restored or uploaded until there is one; the results stay on "
              "the NVMe, which a VM stop wipes.")
    # Every vLLM this run starts: no FlashInfer JIT build (the VMs have a minimal CUDA
    # toolkit), and on the B200 a prebuilt attention backend (stages/eval.py VLLM_JIT_FREE).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("LOBBOT_VLLM_ATTN_BACKEND", "FLASH_ATTN")
    LOGS.mkdir(parents=True, exist_ok=True)
    code = run_plan(steps(), summary, LOGS, ["run"], after_step=upload)
    if shutdown:
        if code == 0 and STATE["final_upload"] and (W / ".uploaded" / "done.json").exists():
            print("Results uploaded; powering off in 1 minute (sudo shutdown -c cancels).")
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)
        else:
            print("Not powering off: the run did not finish with every result uploaded.")
    return code


if __name__ == "__main__":
    sys.exit(main())
