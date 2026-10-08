"""Shared helpers for pipeline stages: job layout, config, subprocess runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.progress import emit
from common.taskspec import TaskSpec

# Set LOBBOT_DRY_RUN=1 to walk every stage without a GPU: stages skip the
# heavy work and write small placeholder outputs. Useful for testing the
# pipeline contract and the TUI end to end on a laptop.
DRY_RUN = os.environ.get("LOBBOT_DRY_RUN") == "1"


@dataclass
class Config:
    """Backend knobs. Override per job with <job>/config.json."""

    teacher: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    student: str = "google/gemma-4-E4B-it"  # dense fallback; Qwen/Qwen3-4B-Instruct-2507 also works
    models_dir: str = os.environ.get("LOBBOT_MODELS", "/mnt/nvme/models")
    llama_cpp: str = os.environ.get("LOBBOT_LLAMA_CPP", "/mnt/nvme/llama.cpp")
    # Data
    n_generate: int = 2000
    n_heldout: int = 100
    # Length limits for long-output tasks (code). None uses the env knob or built-in default:
    data_answer_max_tokens: int | None = None  # LOBBOT_DATA_ANSWER_MAX_TOKENS or 1536; eval follows it
    data_max_len: int | None = None  # vLLM context, LOBBOT_DATA_MAX_LEN or 8192
    # Teacher answers with thinking on, and the thinking is kept in the training rows
    # (the assistant message's reasoning_content), so REAP calibrates on it and heal
    # teaches the model to think. Makes a thinking model: raise data_answer_max_tokens,
    # data_max_len, heal_max_len and reap_max_seq to fit the thinking (see r50w95s-t).
    data_thinking: bool = False
    # Mugge-shaped heal data for code specs (stages/harness.py): this share of the train
    # rows is asked the way Mugge's harness asks (context pack in, files out), and
    # data_fix_rows fix rows are added: a failed draft and its error in, the teacher's
    # passing fix out. Drafts are the teacher's own failed answers, topped up with
    # extra answers sampled at temperature 1.0. 0 and 0 = plain rows only.
    data_harness_share: float = 0.0
    data_fix_rows: int = 0
    # Gemini writes the held-out test inputs when GEMINI_API_KEY is set ("" = teacher writes them)
    testgen_model: str = "gemini-3.8-flash"
    # REAP: fraction of experts removed per layer. 0.5 keeps 64 of 128.
    # Raising it frees bytes for more bits per remaining expert.
    reap_sparsity: float = 0.5
    reap_calib_samples: int = 512
    reap_max_seq: int = 2048
    # REAP calibration source: "task" = data/train.jsonl; "general" = reap_calib_path,
    # a .jsonl (rows with "text", or chat/prompt-answer rows) or a plain-text file
    # (documents split on blank lines). A relative path is resolved in the job dir.
    reap_calib: str = "task"
    reap_calib_path: str = ""
    # Heal (LoRA SFT on teacher answers)
    heal_epochs: float = 1.0
    heal_lr: float = 1e-4
    heal_lora_r: int = 16
    heal_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    heal_train_experts: bool = True  # also LoRA the experts, not just attention and router
    heal_kd_weight: float = 0.0  # >0 adds logit KL to the unpruned teacher (needs it in GPU memory)
    heal_max_minutes: float | None = None  # wall-clock cap on heal; None uses LOBBOT_HEAL_MAX_MINUTES or 60
    heal_max_len: int | None = None  # tokens per example for heal and the dense student; None uses LOBBOT_HEAL_MAX_LEN or 2048
    dense_fallback: bool = True
    student_epochs: float = 2.0
    student_lr: float | None = None  # dense student LoRA lr; None uses LOBBOT_STUDENT_LR or 1e-4
    student_max_minutes: float | None = None  # same for the dense student (LOBBOT_STUDENT_MAX_MINUTES)
    # Quantize: aim below the TaskSpec max size by this margin.
    size_margin_gb: float = 0.5
    bit_floor: str = "q2_k"
    bit_ceiling: str = "q6_k"
    # One llama.cpp type (e.g. "q8_0") for every non-expert tensor; "" keeps bits.STATIC.
    static_type: str = ""
    # Eval
    judge_model: str = "gemini-3.8-flash"  # gemini-* needs GEMINI_API_KEY, claude-* ANTHROPIC_API_KEY
    laptop_bandwidth_gb_s: float = 120.0  # MacBook Air M4; M5 is ~153
    # Execution-based code eval (stages/codebench.py), next to the judge. Empty = off.
    # Suites: multipl-e-py/js/ts/cpp, c-set, livecodebench, heldout.
    code_eval_suites: list[str] = field(default_factory=list)
    code_eval_dir: str = os.environ.get("LOBBOT_CODEBENCH", "/mnt/nvme/codebench")  # scripts/fetch_codebench.py
    code_eval_samples: int = 1  # answers per problem; 1 is greedy pass@1, more sample at code_eval_temperature
    code_eval_k: list[int] = field(default_factory=lambda: [1])
    code_eval_temperature: float = 0.2
    code_eval_limit: int | None = None  # problems per suite (quick runs); None = all
    code_eval_max_tokens: int = 4096
    # Let the model think before answering (chat template enable_thinking). Thinking
    # runs for thousands of tokens, so raise code_eval_max_tokens to ~24k-32k with it.
    code_eval_thinking: bool = False
    # After a greedy answer fails, ask once more with the failing output (Mugge's fix
    # call) and report fix@1 next to pass@1: what the harness loop gets in one repair.
    code_eval_fix: bool = False
    # "plain": benchmark prompts as published. "harness": every problem is asked the way
    # Mugge's harness asks (stages/harness.py: a ticket in, files out), fixes too.
    code_eval_format: str = "plain"
    # Route each token to this many experts instead of the model's own count (0 keeps
    # it). A llama-server override at serve time, so a top-k test needs no rebuild.
    eval_experts_used: int = 0
    lcb_since: str = "2026-01-01"  # LiveCodeBench problems published on or after this date only
    # Models to evaluate instead of work/allocation.json: {name: gguf path}. Lets an
    # eval-only job (pipeline.py --only eval) score an uncompressed reference.
    eval_candidates: dict[str, str] = field(default_factory=dict)
    # Reference job for code scores: its out/eval.json's pass@1 is the 100% mark
    # (experiment 01's `ref`). Relative paths resolve against this job's parent dir.
    code_eval_ref: str = ""


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "work", "out", ".done"):
            (self.root / sub).mkdir(exist_ok=True)
        cfg_path = self.root / "config.json"
        overrides = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.config = Config(**overrides)

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec.load(self.root / "taskspec.json")

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def model_path(self, hf_id: str) -> str:
        """Local snapshot if setup_vm.sh downloaded it, else the hub id."""
        local = Path(self.config.models_dir) / hf_id.split("/")[-1]
        return str(local) if local.exists() else hf_id

    def is_done(self, stage: str) -> bool:
        return (self.root / ".done" / stage).exists()

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        (self.root / ".done" / stage).write_text(json.dumps(info or {}))

    def clear_from(self, stages: list[str], start: str) -> None:
        for s in stages[stages.index(start):]:
            (self.root / ".done" / s).unlink(missing_ok=True)

    def save_config(self) -> None:
        """Record the effective config of this run. config.json stays the user's
        overrides only, so later default changes still apply on resume."""
        (self.root / "work" / "config.effective.json").write_text(json.dumps(asdict(self.config), indent=2))


def run(cmd: list[str], stage: str, cwd: str | Path | None = None, env: dict | None = None) -> None:
    """Run a command, forwarding its output as log lines (stderr is merged)."""
    emit(stage, msg="$ " + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else ""))
    proc = subprocess.Popen(
        cmd, cwd=cwd, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout
    for line in proc.stdout:
        sys.stdout.write("[" + stage + "] " + line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
