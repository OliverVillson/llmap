"""The train harness's one config: the base, the teacher ladder, the pins, and a profile
per place it runs (G0 on the 4060, B200 sessions 1 and 2, the laptops in between).

A run is one profile plus optional overrides from a JSON file (--config) and --set
key=value, in that order. Everything a stage reads comes from the Config below, so
a changed setting is visible in config.json beside the results.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Base:
    """A student base model."""
    key: str
    hf: str
    # architectures the text-only load may report (config.json of the saved SFT model)
    archs: tuple[str, ...]
    family: str  # "qwen3.5" (hybrid Gated DeltaNet + full attention) or "llama" (MiniCPM5)
    teacher_tokenizer: bool  # same tokenizer as the teacher: logit KL from its top-20 log-probs
    min_p: float | None  # sent on every request when set (MiniCPM5: 0, its 0.05 default repeats)
    quant_ladder: tuple[str, ...]  # best first; the footprint step ships the first that fits
    vocab_cut: bool  # try stages/vocab.py's trim before the GGUF


BASES = {
    "qwen3.5-2b": Base("qwen3.5-2b", "Qwen/Qwen3.5-2B", ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
                       "qwen3.5", True, None, ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"), True),
    "minicpm5-2b": Base("minicpm5-2b", "openbmb/MiniCPM5-2B", ("LlamaForCausalLM", "MiniCPMForCausalLM"),
                        "llama", False, 0.0, ("Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS"), False),
    # G0's student: the plumbing check, not a candidate
    "qwen3.5-0.8b": Base("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
                         "qwen3.5", True, None, ("Q8_0", "Q4_K_M"), False),
}


@dataclass(frozen=True)
class Teacher:
    hf: str
    licence: str
    vllm_args: tuple[str, ...] = ()  # extra `vllm serve` flags (quantization is read from the checkpoint)


TEACHERS = {
    "qwen3.8-27b-nvfp4": Teacher("PassingByPixels/Qwen3.8-27B-NVFP4", "Apache-2.0"),
    "qwen3.6-35b-a3b-nvfp4": Teacher("nvidia/Qwen3.6-35B-A3B-NVFP4", "Apache-2.0"),
    "qwen3.6-35b-a3b-fp8": Teacher("Qwen/Qwen3.6-35B-A3B-FP8", "Apache-2.0"),
    # G0's teacher on the 4060: bf16, no FP4 there
    "qwen3.5-2b": Teacher("Qwen/Qwen3.5-2B", "Apache-2.0", ("--enforce-eager",)),
}

# Lowercase: the profile names on the command line.
PROFILES = ("g0", "s1", "s2", "laptop")


@dataclass
class Config:
    profile: str = "s1"
    run: str = "bh01"  # bucket://mugge-library/blackhole/<run>/
    version: str = "0.1"  # BH<version>-<GB>
    where: str = "b200"  # b200 | laptop: which stages may run here (preflight refuses the rest)
    # ---- models
    base: str = "qwen3.5-2b"
    teachers: tuple[str, ...] = ("qwen3.8-27b-nvfp4", "qwen3.6-35b-a3b-nvfp4", "qwen3.6-35b-a3b-fp8")
    # ---- pins (preflight refuses a paid session without exact pins)
    vllm: str = "vllm==0.31.0"  # setup_vm.sh's; the boot dry-run proves the teacher loads in it
    vllm_image: str = ""  # the real image's tag or digest when the B200 runs one; recorded in teacher.json
    transformers_major: int = 5
    llama_cpp_sha: str = "af50f2f248b475d9760f8e67503d01c0f194f5f4"
    uv_exclude_newer: str = "2026-10-12T00:00:00Z"
    # ---- serving and sampling (every server, every bench run)
    thinking: bool = False  # the worker is trained thinking-off (BlackHole, 2026-10-15)
    temperature: float = 0.7  # Qwen's non-thinking card: 0.7 / 0.8 / 20
    top_p: float = 0.8
    top_k: int = 20  # finite everywhere: run 1's top_k 0 was one of the three bugs that scored 0%
    slot_ctx: int = 16384  # one agent's working memory, the footprint's 16K
    max_call_tokens: int = 4096  # per call; thinking-off answers are ~1-2K
    kv_type: str = "q8_0"
    sandbox: str = "docker"  # bench/run.ts --sandbox: docker | podman | local
    vllm_seqs: int = 256
    vllm_gpu_util: float = 0.90
    teacher_logprobs: int = 20  # top-k log-probs stored per answer token (0: none)
    # ---- data
    kits: str = ""  # bhdata's package dir (BHDATA); empty: bhdata_io.CANDIDATES
    limit_kits: int = 0  # keep the first N bench projects of each split after export (G0); 0 = all
    extra_rows: bool = True  # Config.data_extra_rows: bhdata's free mutation rows
    # ---- G2: the teacher's gate, first minutes of session 1
    g2_tickets: int = 200
    g2_min_pass: float = 55.0  # % of the G2 tickets solved; None-like 0 = report only
    g2_min_apm: float = 250.0  # answers (model calls) per minute
    g2_minutes: float = 15.0
    # ---- teacher answers
    samples: int = 3  # seeds per train ticket (2-4); the shortest passing answer is kept
    bench_concurrency: int = 192
    answer_minutes: float = 75.0  # stop starting new shards after this
    shard_projects: int = 25
    drop_warn: float = 0.20  # warn when more than this share of answers drop for length or parsing
    # ---- SFT
    sft_mode: str = "full"  # full | lora
    epochs: float = 2.0
    lr: float = 1e-5  # full FT; LoRA uses lora_lr
    lora_lr: float = 2e-4
    lora_r: int = 64
    kd_weight: float = 0.5  # logit KL share of the loss on the teacher-tokenizer path
    max_len: int = 16384
    batch_tokens: int = 65536
    sft_minutes: float = 60.0
    save_minutes: float = 20.0  # a resumable checkpoint this often
    # ---- smoke and score
    smoke_seeds: int = 1
    smoke_minutes: float = 10.0
    smoke_tickets: int = 0  # 0: the whole spike (15)
    score_seeds: int = 3
    score_bench: str = ""  # a bench dir (the dev set); empty: the spike
    score_bar: float = 50.0
    slots: int = 12  # llama-server -np for score and round-2 sampling on the laptops
    # ---- export and footprint
    student: str = "sft"  # the model the laptop stages export, measure and score: sft or round2
    footprint_limit: float = 2.0e9
    footprint_prompt: int = 12000
    imatrix_chunks: int = 200
    # ---- round 2
    round2_tickets: int = 0  # 0: every held-out ticket
    round2_seeds: int = 2
    round2_minutes: float = 30.0
    repair_samples: int = 2
    rft_minutes: float = 50.0
    # ---- guards
    gpu_idle_abort: bool = True
    gpu_watch_minutes: float = 5.0
    session_minutes: float = 0.0  # 0: no session clock; else stop starting steps that cannot finish in it
    # ---- paths (filled from the environment by load())
    nvme: str = "/mnt/nvme"
    work: str = ""
    logs: str = ""
    bucket: str = "bucket://mugge-library"
    codebench: str = ""
    muggebench_sh: str = ""  # BlackHole's eval/muggebench/muggebench.sh, for the dev set on a laptop
    footprint_sh: str = ""  # BlackHole's serving/footprint.sh, the reference measurement
    extra: dict = field(default_factory=dict)

    # ---- derived
    @property
    def base_model(self) -> Base:
        return BASES[self.base]

    @property
    def kd(self) -> bool:
        return self.base_model.teacher_tokenizer and self.teacher_logprobs > 0 and self.kd_weight > 0

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


PROFILE_SETTINGS: dict[str, dict] = {
    # G0: Qwen3.5-0.8B from Qwen3.5-2B, 20 kits, tiny token counts, end to end on the 4060
    # (8 GB). It proves the plumbing, so its gate only reports.
    "g0": {"where": "laptop", "base": "qwen3.5-0.8b", "teachers": ("qwen3.5-2b",), "run": "g0",
           "limit_kits": 20, "g2_tickets": 20, "g2_min_pass": 0.0, "g2_min_apm": 0.0, "g2_minutes": 5.0,
           "samples": 2, "bench_concurrency": 8, "answer_minutes": 20.0, "shard_projects": 10,
           "max_call_tokens": 1024, "slot_ctx": 8192, "vllm_seqs": 16, "vllm_gpu_util": 0.80,
           "sft_mode": "lora", "epochs": 1.0, "max_len": 8192, "batch_tokens": 8192, "sft_minutes": 10.0,
           "save_minutes": 5.0, "smoke_tickets": 3, "smoke_minutes": 5.0, "score_seeds": 1, "slots": 4,
           "footprint_prompt": 6000, "imatrix_chunks": 20, "round2_tickets": 5, "round2_seeds": 1,
           "round2_minutes": 5.0, "repair_samples": 1, "rft_minutes": 5.0, "gpu_idle_abort": False},
    # B200 session 1 (about 3 h 20): teacher, G2, answers, SFT, spike smoke
    "s1": {"where": "b200", "session_minutes": 200.0},
    # B200 session 2 (about 2 h 30): the own-mistakes round
    "s2": {"where": "b200", "session_minutes": 150.0},
    # the laptops: export, footprint, score, student sampling between sessions
    "laptop": {"where": "laptop", "gpu_idle_abort": False, "vllm_seqs": 32},
}

# Which stages a place may run.
WHERE = {
    "b200": {"setup", "data", "teacher-serve", "teacher-answers", "sft", "smoke", "round2-sample",
             "round2-repair", "round2-rft", "upload"},
    "laptop": {"setup", "data", "teacher-serve", "teacher-answers", "sft", "smoke", "export", "footprint", "score",
               "round2-sample", "round2-repair", "round2-rft", "upload"},
}


def _coerce(cur, raw: str):
    if isinstance(cur, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(raw)
    if isinstance(cur, float):
        return float(raw)
    if isinstance(cur, tuple):
        return tuple(x for x in raw.split(",") if x)
    if isinstance(cur, dict):
        return json.loads(raw)
    return raw


def load(profile: str, env: dict, config_file: Path | None = None, sets: list[str] = ()) -> Config:
    """The profile's config, then the file's keys, then each --set key=value; paths from
    the environment (NVME, BH_WORK, BH_LOGS, BH_BUCKET, CODEBENCH, MUGGEBENCH_SH,
    FOOTPRINT_SH) unless set already."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}, not {profile!r}")
    over: dict = {"profile": profile, **PROFILE_SETTINGS[profile]}
    if config_file:
        over.update(json.loads(Path(config_file).read_text()))
    cfg = Config()
    names = {f.name for f in dataclasses.fields(Config)}
    for k, v in over.items():
        if k not in names:
            raise ValueError(f"unknown setting {k!r}")
        setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    for s in sets:
        k, sep, raw = s.partition("=")
        if not sep or k not in names:
            raise ValueError(f"--set wants key=value; unknown key or no value in {s!r}")
        setattr(cfg, k, _coerce(getattr(cfg, k), raw))
    cfg.nvme = env.get("NVME", cfg.nvme)
    cfg.work = cfg.work or env.get("BH_WORK") or str(Path(cfg.nvme) / "jobs" / "bh" / cfg.run)
    cfg.logs = cfg.logs or env.get("BH_LOGS") or str(Path(env.get("HOME", "/tmp")) / "bh-logs" / cfg.run)
    cfg.bucket = env.get("BH_BUCKET", cfg.bucket)
    cfg.codebench = cfg.codebench or env.get("CODEBENCH") or str(Path(cfg.nvme) / "codebench")
    cfg.muggebench_sh = cfg.muggebench_sh or env.get("MUGGEBENCH_SH", "")
    cfg.footprint_sh = cfg.footprint_sh or env.get("FOOTPRINT_SH", "")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.base not in BASES:
        raise ValueError(f"base must be one of {', '.join(BASES)}, not {cfg.base!r}")
    for t in cfg.teachers:
        if t not in TEACHERS:
            raise ValueError(f"teacher {t!r} is not one of {', '.join(TEACHERS)}")
    if cfg.top_k <= 0:
        raise ValueError("top_k must be finite and positive on every server (run 1's top_k 0)")
    if cfg.sft_mode not in ("full", "lora"):
        raise ValueError(f"sft_mode must be full or lora, not {cfg.sft_mode!r}")
    if cfg.student not in ("sft", "round2"):
        raise ValueError(f"student must be sft or round2, not {cfg.student!r}")
    if cfg.where not in WHERE:
        raise ValueError(f"where must be one of {', '.join(WHERE)}")
    if not 2 <= cfg.samples <= 4 and cfg.profile != "g0":
        raise ValueError(f"samples is 2-4 per ticket, not {cfg.samples}")


def pins_problems(cfg: Config) -> list[str]:
    """What keeps the versions from being exact: a paid session refuses to start on any."""
    out = []
    if not re.fullmatch(r"vllm(==[\w.+-]+|\s*@\s*\S+@[0-9a-f]{40})", cfg.vllm.strip()):
        out.append(f"vllm {cfg.vllm!r} is not an exact version (vllm==X) or commit (vllm @ git+...@<sha>)")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.llama_cpp_sha):
        out.append(f"llama_cpp_sha {cfg.llama_cpp_sha!r} is not a full commit")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cfg.uv_exclude_newer):
        out.append(f"uv_exclude_newer {cfg.uv_exclude_newer!r} is not a UTC timestamp")
    return out
