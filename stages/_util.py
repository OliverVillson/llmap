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
    # Sampling for thinking answers: the temperature here, top_k in the data stage and
    # the eval. The Qwen3 cards say 0.6 and 20; Gemma 4's says 1.0 and 64.
    data_thinking_temperature: float = 0.6
    thinking_top_k: int = 20
    # Mugge-shaped heal data for code specs (stages/harness.py): this share of the train
    # rows is asked the way Mugge's harness asks (context pack in, files out), and
    # data_fix_rows fix rows are added: a failed draft and its error in, the teacher's
    # passing fix out. Drafts are the teacher's own failed answers, topped up with
    # extra answers sampled at temperature 1.0. 0 and 0 = plain rows only.
    data_harness_share: float = 0.0
    data_fix_rows: int = 0
    # Contest rows for code specs: contest problems answered by the teacher (thinking per
    # data_thinking) and kept when they pass their tests. Train only, never held out.
    data_contest_rows: int = 0  # contest rows to add; 0 = off
    # "codecontests": deepmind/code_contests (fetched to code_eval_dir), every problem from
    # before 2022, so none of LiveCodeBench's (v6 starts 2023-05), and any whose statement
    # shares a long word run with a LiveCodeBench problem is dropped too. "livecodebench":
    # LiveCodeBench problems released before data_contest_before; those are LCB v6
    # problems, so never use it when LCB v6 is a score that counts.
    data_contest_source: str = "codecontests"
    data_contest_before: str = "2024-10-01"  # livecodebench source: only problems released before this date
    data_contest_samples: int = 2  # teacher answers per problem; the shortest passing one is kept
    data_contest_harness_share: float = 0.0  # share asked as a Mugge ticket instead of a plain prompt
    # A .jsonl of ready chat rows added to train as they are, for code specs (train only, no
    # teacher): e.g. aider transcripts from scripts/polyglot.py rows. A relative path is
    # resolved in the job dir. The data stage marks them in train.jsonl with "meta": {"extra": true}
    # (training ignores meta).
    data_extra_rows: str = ""
    # Each extra row appears this many times in train, so heal sees it that often per epoch.
    data_extra_weight: int = 1
    # Share of the calibration samples drawn from the extra rows, for REAP (reap_calib "task",
    # reap_calib_samples) and w4a16 GPTQ (quant_calib_samples): each distinct row at most once,
    # taken from each language (meta.language) in turn; the rest are the other train rows.
    # 0 = train rows as they come, copies included.
    calib_extra_share: float = 0.0
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
    # Quantize: "gguf" = llama.cpp mixed-bit GGUF (below); "w4a16" = compressed-tensors
    # for vLLM on a GPU (stages/quantize_ct.py): int4 experts, FP8 attention, BF16 rest.
    quant_format: str = "gguf"
    quant_calib_samples: int = 128  # w4a16: train.jsonl rows GPTQ calibrates on (see calib_extra_share)
    quant_calib_len: int = 2048  # w4a16: tokens per calibration row
    quant_fp8_attention: bool = True  # w4a16: False keeps attention, DeltaNet, lm_head, Gemma's MLP BF16
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
    # A llama.cpp LoRA GGUF loaded on top of the evaluated GGUF (llama-server --lora). Nothing
    # on the Event Horizon path sets it (qheal ships no adapter any more). Empty = none.
    eval_lora: str = ""
    code_eval_samples: int = 1  # answers per problem; 1 is greedy pass@1, more sample at code_eval_temperature
    code_eval_k: list[int] = field(default_factory=lambda: [1])
    code_eval_temperature: float = 0.2
    code_eval_limit: int | None = None  # problems per suite (quick runs); None = all
    code_eval_max_tokens: int = 4096
    # Let the model think before answering (chat template enable_thinking). Thinking
    # runs for thousands of tokens, so raise code_eval_max_tokens to ~24k-32k with it.
    code_eval_thinking: bool = False
    # Sampling temperature for one thinking answer per problem (greedy decoding loops
    # when thinking). The Qwen3 cards say 0.6; Qwen3.8's says 1.0.
    code_eval_thinking_temperature: float = 0.6
    # Keep each answer's whole thinking in work/code_eval/<name>.jsonl and re-score it
    # cut at these token budgets (stages/budget.py), so one long run also gives the
    # scores at every shorter budget.
    code_eval_budgets: list[int] = field(default_factory=list)
    # After a greedy answer fails, ask once more with the failing output (Mugge's fix
    # call) and report fix@1 next to pass@1: what the harness loop gets in one repair.
    code_eval_fix: bool = False
    # "plain": benchmark prompts as published. "harness": every problem is asked the way
    # Mugge's harness asks (stages/harness.py: a ticket in, files out), fixes too.
    code_eval_format: str = "plain"
    # Route each token to this many experts instead of the model's own count (0 keeps
    # it). An override at serve time (llama-server --override-kv, vLLM --hf-overrides),
    # so a top-k test needs no rebuild.
    eval_experts_used: int = 0
    # llama-server KV cache type at eval ("q8_0" halves it, so more long thinking
    # answers run at once); "" keeps llama.cpp's f16. vLLM: --kv-cache-dtype in eval_vllm_args.
    eval_kv_type: str = ""
    # vLLM serves the candidates that are HF checkpoint dirs: its --reasoning-parser
    # ("" = none; the <think> tags then stay in the content and are split off there;
    # "gemma4" for Gemma 4, whose thought channel needs it: without it vLLM drops the
    # channel's special tokens and the thinking runs into the answer),
    # and extra `vllm serve` args, e.g. ["--speculative-config", "{...}"].
    eval_reasoning_parser: str = "qwen3"
    eval_vllm_args: list[str] = field(default_factory=list)
    # vLLM's --max-num-seqs, which is also how many answers the eval keeps in flight;
    # 0 = LOBBOT_VLLM_SEQS or 256. On an L40S with 16k thinking, ~96: past what the KV
    # cache holds, requests only wait inside vLLM. A --max-num-seqs in eval_vllm_args wins.
    vllm_seqs: int = 0
    # min_p sent with every eval request, explicitly and with a repetition penalty of 1,
    # so llama-server's own default (0.05) never applies and both engines sample alike.
    eval_min_p: float = 0.0
    lcb_since: str = "2026-01-01"  # LiveCodeBench problems published on or after this date only
    # ... and on or before this one ("" = no end), so a later release cannot grow the window.
    lcb_until: str = ""
    # Test cases per LiveCodeBench problem: the first N (examples first), 0 = all.
    # scripts/fetch_codebench.py stores every case; 50 is what readings so far used.
    lcb_max_cases: int = 50
    # Seed of answer i to a problem is code_eval_seed_offset + i, so a new reading of the
    # same model draws fresh answers (e.g. 1000 for a record, 50 for a screen).
    code_eval_seed_offset: int = 0
    # Only these problem ids, from every suite ([] = all). An id missing from the suites
    # is an error: a fixed problem set must be the one scored.
    code_eval_ids: list[str] = field(default_factory=list)
    # Answers per problem by id, e.g. 2/8/4 on problems the reference went 0/4, mixed, 4/4;
    # ids not listed get code_eval_samples (at least max(code_eval_k)), and 0 leaves a
    # problem out. pass@k is reported for the k every problem has answers for.
    code_eval_samples_by_id: dict[str, int] = field(default_factory=dict)
    # Size in GB (1e9 bytes) to report per candidate name in eval.json instead of its
    # files' size: the shipped GGUF's bytes for a candidate scored as a bf16 checkpoint.
    size_gb_override: dict[str, float] = field(default_factory=dict)
    # A trimmed vocabulary's vocab_index.json (stages/vocab.py): {"kept": [old ids in new
    # order], "old_to_new": {old: new}}, so training maps the original's ids. "" = none.
    vocab_index: str = ""
    # Models to evaluate instead of work/allocation.json: {name: path}, a GGUF (served
    # with llama-server) or an HF checkpoint dir with a config.json (BF16 or
    # compressed-tensors, served with vLLM). Lets an eval-only job (pipeline.py
    # --only eval) score an uncompressed reference.
    eval_candidates: dict[str, str] = field(default_factory=dict)
    # What the model being scored is, by content, from the caller ("" = not given): e.g.
    # scripts/event_horizon.py's hash key of the reading's model. When set it stands in
    # for the model files' modification time in the eval's run key, so answers uploaded
    # from a failed reading resume on a new VM, where the same model was downloaded or
    # rebuilt with new mtimes. One fingerprint per job, so set it for one candidate only.
    eval_model_fingerprint: str = ""
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


def safetensors_keys(path: Path) -> list[str]:
    """The tensor names in a safetensors file's header (8-byte length, then JSON)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    return [k for k in head if k != "__metadata__"]


def weight_files(hf_dir: str | Path) -> dict[str, Path]:
    """{tensor name: the safetensors file holding it}, from each file's own header. The
    index only names the files: its name -> file map is not used, because the Hub's for
    Granite 4.2 3B puts model.layers.4.self_attn.k_proj.weight in a shard that does not
    hold it (2026-10-14; convert_hf_to_gguf, vLLM and transformers read every shard's keys
    and never notice)."""
    d = Path(hf_dir)
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    where: dict[str, Path] = {}
    for f in files:
        for k in safetensors_keys(d / f):
            if k in where:
                raise ValueError(f"{k} is in both {where[k].name} and {f} of {d}")
            where[k] = d / f
    return where
