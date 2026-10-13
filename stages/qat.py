"""Event Horizon QAT: the dense model trained inside the llama.cpp K-quant grids it ships
in, so the packed GGUF is exactly the model that was trained and scored.

Every layer linear, the embedding and the head get a kquant.QuantWeight at the type the
map gives it (horizon's knapsack: Q2_K, Q3_K or Q4_K per tensor). Its forward pass is
the weight llama.cpp will hold (integer codes, sub-block scales and mins, fp16 d and
dmin, dequantized in ggml's order), and gradients reach the fp32 latent weight and the
scales straight through. Norms train in fp32 and ship as F32. Nothing is re-quantized
after training: export packs each QuantWeight's own integers, and gguf-py's
dequantization of the packed file must equal the trainer's fake-quant bit for bit, so
the vLLM checkpoint written from it is the shipped weights. This replaces the LoRA
repair, which froze the weights where post-training quantization left them and could
not move one weight to another grid point.

Phases (plan: /mnt/project-files/plan/event-horizon-qat.md, S1.9, S2.4-S2.6):

  block     Block-AP on the L40S: the embedding and head go into their grid at
            llama-quantize's codes and stay there; then layer by layer, each block's
            QuantWeights (also from llama-quantize's codes) and norms learn to reproduce
            the bf16 block's output from what the quantized blocks before it produce.
            One block's optimizer state at a time (~1.3 GB), hidden states on the CPU,
            as many windows as its free memory holds (fit_windows)
  e2e       teacher-forced KD on the B200: every weight in its grid, embedding and head
            included, fp32 master weights and AdamW, trace rows of up to 16k tokens from
            their prompt (traces.batches), KL over the kept vocab to the bf16 teacher
            (sft.kd_kl with index=), prompt tokens at weight 0.1
  onpolicy  rounds on the student's own 16k-token thinking: the trainer checkpoints and
            exits, vLLM serves the student alone, it answers fresh problems (with the
            eval's forced answer), the sandbox labels them, and the trainer resumes on
            them: dense reverse KL to the teacher on every rollout token plus 0.1 x
            cross-entropy on passing rollouts (not forced). Rounds are kept by coded rules
  export    pack every QuantWeight (gguf_build.build), hold the packed file against the
            trainer bit for bit, and write the vLLM checkpoint from its dequantization

The teacher of e2e and onpolicy runs in the trainer's process, not in a vLLM: a dense KL
needs the teacher's whole distribution at every position, and vLLM's prompt_logprobs
returns only the top few per token (too many to cache besides: one 20k-token rollout's
logits over the kept vocab are ~2 GB). It fits: the bf16 teacher is ~7 GB of the
~97 GB e2e_memory() estimates for the trimmed model at 16k tokens (~125 GB untrimmed at
a whole 22.5k-token rollout) on a 180 GB B200, and a teacher server would need GPU
memory carved out of the trainer's while it runs.

Robustness, the same in every phase: the learning rate follows progress
p = max(step/total, training seconds/cap), so a slow or throttled GPU still anneals to
zero at the time cap; held-out evals and checkpoints run off that clock. The time cap
is checked before any forward; a non-finite loss or gradient norm skips the update
(too many in a row, or in all, raise); nothing non-finite is ever saved. Checkpoints
are written atomically and a resume carries the step, the optimizer and the training
seconds, so the schedule continues where it stopped. Models load with
device_map={"": 0}, and with a GPU every tensor must be on it: an occupied GPU fails at
once instead of spilling layers to the CPU.

    python -m stages.qat block --base <trimmed hf> --map map.json --gguf q2k.gguf q3k.gguf q4k.gguf \\
        --traces traces.jsonl --vocab-index vocab_index.json --llama-cpp <dir> --out <dir>
    python -m stages.qat e2e --state <block dir> --teacher <hf> --traces traces.jsonl \\
        --vocab-index vocab_index.json --out <dir> --tokens 40e6 --max-minutes 45 [--freeze-steps 50]
    python -m stages.qat onpolicy --work <dir> --state <e2e dir> --teacher <hf> --problems <codebench> \\
        --traces traces.jsonl --vocab-index vocab_index.json --llama-cpp <dir> [--until-session N] \\
        [--max-minutes M] [--rollout-tokens-cap T]
    python -m stages.qat export --state <dir> --out <dir> --llama-cpp <dir>
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import itertools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stages import kquant, opd, traces
from stages.qheal import hf_name as gguf_to_hf
from stages.sft import KD_CHUNK, NonFiniteTraining, kd_kl, log
from stages.ternary import FORCED_MARGIN, MAX_MODEL_LEN, MAX_TOKENS, SEQS, decoder_layers, layer_inputs, run_layer

PROMPT_WEIGHT = 0.1  # weight of a prompt token's KL in e2e (an answer token's is 1)
CE_WEIGHT = 0.1  # on-policy: cross-entropy on passing rollouts, beside the reverse KL
FREEZE_STEPS = 50  # the codes stop moving for about the last steps, while the scales settle
# AdamW's step is about lr whatever a tensor's size. d and dmin are ~10x smaller than a
# weight (d * sc spans a sub-block), so they move at a tenth of the rate; scales and mins
# count in steps of their integer grid, so they need a larger one to ever cross a .5.
D_LR = 0.1
SCALE_LR = 10.0
B200_GB = 180.0
L40S_GB = 46.0
# Block-AP keeps two hidden-state caches (2 x windows x length x hidden): 2048 x 2048 x 2560
# in bf16 is ~43 GB, more than some VMs have free. fit_windows() keeps as many windows as
# the cache device's free memory holds after this reserve, and refuses fewer than
# MIN_BLOCK_WINDOWS of a request it had to cut.
CACHE_RESERVE_GB = {"cpu": 8.0, "cuda": 10.0}
MIN_BLOCK_WINDOWS = 256
# Keep rules of an on-policy round (plan S2.5), against the last kept model on the same
# held-out problems: on-policy KL drops by at least KEEP_KL_DROP; loop%, cap% and pass@1
# move no more than KEEP_SE standard errors of the paired difference the wrong way.
# Stop after MAX_NOT_KEPT rounds in a row are not kept, or after MAX_ROUNDS.
KEEP_KL_DROP = 0.05
KEEP_SE = 2.0
MAX_ROUNDS = 4
MAX_NOT_KEPT = 2
LOOP_RATIO = 0.2  # scripts/thinking_report.py: a thinking whose last 1500 chars compress below this loops
STATE, META = "latent.safetensors", "qat.json"
NON_QUANT = ("f32", "f16", "bf16")  # map entries of norms and the like, which train as they are


# ---------- the type map ----------

def load_type_map(src) -> dict[str, str]:
    """{HF weight name: K-quant type} from a dict or a JSON file (horizon's map.json, whose
    "types" are by GGUF name, or a bare map, or one under "map"/"type_map"). GGUF tensor
    names (blk.3.ffn_down.weight, token_embd.weight, output.weight) become HF ones; entries
    at f32/f16/bf16 (norms) are dropped, they train as they are. A weight at a type QAT
    cannot train into exactly (anything but kquant.TYPES), or a GGUF name that maps to no
    HF weight, raises."""
    if not isinstance(src, dict):
        src = json.loads(Path(src).read_text())
    for key in ("types", "map", "type_map"):
        if isinstance(src.get(key), dict):
            src = src[key]
            break
    out = {}
    for name, qtype in src.items():
        qtype = str(qtype).lower()
        if qtype in NON_QUANT:
            continue
        if qtype not in kquant.TYPES:
            raise ValueError(f"type map: {name} is {qtype}; QAT trains exactly into {', '.join(kquant.TYPES)} only")
        hf = gguf_to_hf(name) if name.startswith("blk.") or name in ("token_embd.weight", "output.weight") else name
        if hf is None:
            raise ValueError(f"type map: {name} is no layer, embedding or head weight")
        out[hf] = qtype
    if not out:
        raise ValueError("type map has no quantized weight")
    return out


# ---------- quantized modules ----------

class QuantLinear(nn.Module):
    """A linear layer whose weight is a kquant.QuantWeight: the forward pass multiplies by
    its fake-quantized weight, which is what the GGUF will hold."""

    def __init__(self, qw: kquant.QuantWeight, bias: nn.Parameter | None = None):
        super().__init__()
        self.qw, self.bias = qw, bias
        self.out_features, self.in_features = qw.weight.shape

    @property
    def weight(self) -> torch.Tensor:
        return self.qw()

    def forward(self, x):
        return F.linear(x, self.qw().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class QuantEmbedding(nn.Module):
    """An embedding whose table is a kquant.QuantWeight; rows come out in the dtype the
    plain embedding had (bf16 on the GPU), as the served model's do."""

    def __init__(self, qw: kquant.QuantWeight, padding_idx: int | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.qw, self.padding_idx, self.out_dtype = qw, padding_idx, dtype
        self.num_embeddings, self.embedding_dim = qw.weight.shape

    @property
    def weight(self) -> torch.Tensor:
        return self.qw()

    def forward(self, ids):
        return F.embedding(ids, self.qw(), self.padding_idx).to(self.out_dtype)


QUANT_MODULES = (QuantLinear, QuantEmbedding)


def embed_names(model) -> tuple[str, str | None]:
    """(HF name of the embedding weight, of the head weight or None)."""
    mods = {m: n for n, m in model.named_modules()}
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    return mods[emb] + ".weight", (mods[head] + ".weight" if head is not None else None)


def quant_modules(model) -> dict[str, nn.Module]:
    """{HF weight name: QuantLinear or QuantEmbedding} under model."""
    return {n + ".weight": m for n, m in model.named_modules() if isinstance(m, QUANT_MODULES)}


def quant_weights(module) -> list[kquant.QuantWeight]:
    return [m for m in module.modules() if isinstance(m, kquant.QuantWeight)]


def tied(model) -> bool:
    head = model.get_output_embeddings()
    return head is not None and not isinstance(head, QUANT_MODULES) \
        and head.weight is model.get_input_embeddings().weight


def _untie(model) -> None:
    """A head sharing the embedding's table gets its own copy: the two ship as separate
    tensors at their own types (token_embd and output)."""
    if tied(model):
        head = model.get_output_embeddings()
        head.weight = nn.Parameter(model.get_input_embeddings().weight.detach().clone())
        model.config.tie_word_embeddings = False


def _layers_prefix(model) -> str:
    layers = decoder_layers(model)
    return next(n for n, m in model.named_modules() if m is layers)


def _wrap(model, names, type_map: dict[str, str], codes: dict | None = None) -> dict[str, nn.Module]:
    """Swaps the modules of the named weights for quantized ones in place, each from its
    llama-quantize codes when codes has them (kquant's own rounding otherwise)."""
    codes = codes or {}
    emb, head = embed_names(model)
    if tied(model) and {emb, head} & set(names):
        if not {emb, head} <= set(type_map):
            raise ValueError(f"{emb} and {head} are tied: the map needs a type for each (they ship as two tensors)")
        _untie(model)
    mods = dict(model.named_modules())
    out = {}
    for name in names:
        path = name.removesuffix(".weight")
        mod = mods.get(path)
        if isinstance(mod, QUANT_MODULES):
            continue
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            raise ValueError(f"{name}: no linear or embedding weight of that name in the model")
        qw = kquant.QuantWeight(mod.weight.detach().float(), type_map[name], codes.get(name))
        if qw.nudged > 0.5 * qw.weight.numel():
            raise ValueError(f"{name}: {qw.nudged} of {qw.weight.numel()} weights do not round to the given codes; "
                             "the GGUF is not of this checkpoint")
        new = (QuantLinear(qw, mod.bias) if isinstance(mod, nn.Linear)
               else QuantEmbedding(qw, mod.padding_idx, mod.weight.dtype))
        parent, _, attr = path.rpartition(".")
        setattr(mods[parent] if parent else model, attr, new)
        out[name] = new
    return out


def coverage(model, type_map: dict[str, str]) -> None:
    """Raises unless the map names exactly the model's layer matrices, embedding and head:
    a weight left out would ship at whatever type the converter picks, not as trained."""
    emb, head = embed_names(model)
    lp = _layers_prefix(model) + "."
    want = {n for n, p in model.named_parameters() if p.dim() == 2 and n.startswith(lp) and ".qw." not in n}
    want |= {n for n in quant_modules(model) if n.startswith(lp)} | {emb} | ({head} if head else set())
    missing, extra = sorted(want - set(type_map)), sorted(set(type_map) - want)
    if missing == [head] and tied(model):  # the converter writes no output tensor for a tied head
        raise ValueError(f"{head} is tied to {emb} and the map has no type for it: QAT trains and ships the head "
                         "as its own tensor (output.weight), so the map needs one (and its bytes)")
    if missing or extra:
        raise ValueError(f"type map does not cover the model: {len(missing)} weights without a type "
                         f"(e.g. {missing[:3]}), {len(extra)} names not in it (e.g. {extra[:3]})")


def quant_model(model, type_map, init: dict | None = None, *, strict: bool = True) -> dict[str, nn.Module]:
    """Wraps every weight the type map names in a kquant.QuantWeight at its type, in place:
    layer linears become QuantLinear, the embedding a QuantEmbedding, the head a
    QuantLinear (untied from the embedding first). init: {HF name: kquant.KQ},
    llama-quantize's own codes (gguf_codes) to start from. strict: the map must cover
    every layer matrix, the embedding and the head. Returns {HF name: module}."""
    type_map = load_type_map(type_map)
    if strict:
        coverage(model, type_map)
    return _wrap(model, list(type_map), type_map, init)


def freeze_codes(module) -> None:
    """Fixes the integer codes of every QuantWeight under module (kquant: the latent
    weights stop training, the scales go on)."""
    for q in quant_weights(module):
        if not q.frozen:
            q.freeze_codes()


def thaw_codes(module) -> int:
    """Lets the codes of every frozen QuantWeight under module move again, for the next
    phase: each latent weight whose rounding (at the current scales) is not its frozen
    code moves to that code's value, as kquant places a weight on llama-quantize's codes,
    so the forward pass is unchanged bit for bit. Returns how many weights moved."""
    moved = 0
    with torch.no_grad():
        for q in quant_weights(module):
            if not q.frozen:
                continue
            want = q()
            q.codes = None
            off = q() != want
            q.weight[off] = want[off]
            if not torch.equal(q(), want):
                raise ValueError(f"cannot thaw the codes of a {q.type} weight {tuple(q.weight.shape)} exactly")
            q.weight.requires_grad_(True)
            moved += int(off.sum())
    return moved


# ---------- llama-quantize's codes ----------

def _gguf(llama_cpp: str | Path | None = None):
    """gguf-py, from the pinned llama.cpp checkout when given (the converter's own)."""
    from stages import gguf_build
    return gguf_build._gguf(llama_cpp or None)


def gguf_codes(paths, hf_dir: str | Path, model, type_map, gguf=None) -> dict[str, kquant.KQ]:
    """{HF name: kquant.KQ}: each mapped weight's integers as llama-quantize wrote them
    (gguf_build.read_kq: rows in HF order), from the first GGUF holding that tensor at
    its planned type. The GGUFs are of hf_dir, the checkpoint model was loaded from
    (S1.8 quantizes it once per type). A head tied to the embedding in hf_dir starts
    from token_embd at the head's type. Raises on any weight no GGUF has at its type."""
    from stages import gguf_build
    type_map = load_type_map(type_map)
    emb, head = embed_names(model)
    got, embd = {}, {}
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        for name, kq in gguf_build.read_kq(path, hf_dir, gguf).items():
            if name == emb:
                embd.setdefault(kq.type, kq)
            if type_map.get(name) == kq.type and name not in got:
                got[name] = kq
    if head in type_map and head not in got and tied(model) and type_map[head] in embd:
        log(f"qat: {head} starts from token_embd's {type_map[head]} codes (tied in {hf_dir})")
        got[head] = embd[type_map[head]]
    missing = sorted(set(type_map) - set(got))
    if missing:
        raise ValueError(f"{len(missing)} weights are in no GGUF at their planned type, e.g. "
                         f"{[(n, type_map[n]) for n in missing[:3]]}")
    return got


# ---------- loading and saving ----------

def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def on_cuda(*models) -> None:
    """With a GPU, every parameter and buffer must be on it: accelerate's offload would
    train at a fraction of the speed until the time cap fired."""
    if not torch.cuda.is_available():
        return
    for m in models:
        off = [n for n, t in itertools.chain(m.named_parameters(), m.named_buffers()) if t.device.type != "cuda"]
        if off:
            raise RuntimeError(f"{len(off)} tensors are not on the GPU (e.g. {off[0]}); is it shared with another job?")


def load_model(path: str | Path, device: str | None = None):
    """bf16 on GPU 0 (device_map={"": 0}: an occupied GPU fails at once with an out of
    memory error instead of offloading layers to the CPU), fp32 on the CPU."""
    import transformers
    from transformers import AutoModelForCausalLM
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        m = AutoModelForCausalLM.from_pretrained(str(path), **{key: torch.bfloat16}, device_map={"": 0})
        on_cuda(m)
    else:
        m = AutoModelForCausalLM.from_pretrained(str(path), **{key: torch.float32})
    m.config.use_cache = False
    return m


def check_finite(model) -> None:
    """Never save a NaN/inf weight: a bad state would be resumed, exported and scored."""
    bad = [n for n, p in model.named_parameters() if p.is_floating_point() and not torch.isfinite(p).all()]
    if bad:
        raise NonFiniteTraining(f"{len(bad)} tensors are not finite, e.g. {bad[0]}; nothing saved")


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return (t.detach().cpu() if t.is_cuda else t.detach().clone()).contiguous()


def _replace_dir(tmp: Path, out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def _state_dir(path) -> Path:
    p = Path(path)
    return p / "state" if (p / "state" / META).exists() else p


def save_state(model, out: str | Path, meta: dict) -> Path:
    """The trained model as QAT state: every tensor (fp32 latent weights, scales and mins,
    frozen codes, norms) in latent.safetensors and meta (base checkpoint, type map,
    phase) in qat.json, written to a .tmp dir and renamed, so a stop never leaves half a
    state. ~15 GB for the 3.4B model: what session 2 restores."""
    from safetensors.torch import save_file
    check_finite(model)
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: _cpu(v) for k, v in model.state_dict().items()}, str(tmp / STATE))
    (tmp / META).write_text(json.dumps(meta, indent=1) + "\n")
    _replace_dir(tmp, out)
    return out


def load_state(path, base: str | None = None, device: str | None = None):
    """(model, meta) from a state dir (or a phase's output dir holding state/): the base
    checkpoint (meta["base"], or base when the files moved), quantized by meta's type
    map, then every tensor from latent.safetensors (frozen codes included)."""
    from safetensors.torch import load_file
    d = _state_dir(path)
    meta = json.loads((d / META).read_text())
    model = load_model(base or meta["base"], device)
    quant_model(model, meta["type_map"])
    for p in model.parameters():  # norms were trained and saved in fp32: load them so
        if p.is_floating_point() and p.dtype != torch.float32:
            p.data = p.data.float()
    dev = next(model.parameters()).device
    model.load_state_dict(load_file(str(d / STATE), device=str(dev)), strict=True)
    on_cuda(model)
    return model, meta


def plain_state_dict(model, *, keep_fp32: bool = False) -> dict[str, torch.Tensor]:
    """The model's tensors under their HF names, on the CPU: each quantized weight as its
    fake-quant value (what the GGUF holds) in bf16, the rest bf16 too, or fp32 with
    keep_fp32 (the converter then writes the trained norms exactly)."""
    q = quant_modules(model)
    prefixes = tuple(n.removesuffix("weight") for n in q)
    sd = {}
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if not k.startswith(prefixes):
                sd[k] = _cpu(v.float() if keep_fp32 else v.to(torch.bfloat16))
        for n, m in q.items():
            sd[n] = _cpu(m.qw().to(torch.bfloat16))
            if getattr(m, "bias", None) is not None:
                sd[n.removesuffix("weight") + "bias"] = _cpu(m.bias.float() if keep_fp32 else m.bias.to(torch.bfloat16))
    return sd


def write_hf(model, base: str | Path, out: str | Path, *, keep_fp32: bool = False) -> Path:
    """A plain HF checkpoint of the model as trained (plain_state_dict), with base's
    tokenizer and other files and the model's own config (untied head, bf16, cache on):
    vLLM serves it, export converts it."""
    from safetensors.torch import save_file
    check_finite(model)
    out, base = Path(out), Path(base)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in base.iterdir():
        if f.is_file() and f.suffix not in (".safetensors", ".bin", ".pt", ".gguf") and not f.name.endswith(".index.json"):
            shutil.copy2(f, tmp / f.name)
    cfg = model.config.__class__.from_dict(model.config.to_dict())
    cfg.use_cache = True
    cfg.tie_word_embeddings = False
    cfg.dtype = torch.bfloat16
    cfg.save_pretrained(tmp)
    save_file(plain_state_dict(model, keep_fp32=keep_fp32), str(tmp / "model.safetensors"), metadata={"format": "pt"})
    _replace_dir(tmp, out)
    return out


# ---------- the schedule, the clock and the non-finite guard ----------

class Schedule:
    """lr from progress p = max(step/total, training seconds/cap): linear warm-up over the
    first `warm` steps, then cosine to 0 at p = 1, whichever of steps and time runs out
    first. Training stops at p >= 1, so it always ends annealed."""

    def __init__(self, lr: float, total: int, warm: int = 0, cap_s: float = 0.0):
        self.peak, self.total, self.cap_s = lr, max(1, total), cap_s
        self.warm = min(warm, self.total - 1)

    def progress(self, step: int, seconds: float) -> float:
        return max(step / self.total, seconds / self.cap_s if self.cap_s else 0.0)

    def lr(self, step: int, seconds: float) -> float:
        if step < self.warm:
            return self.peak * (step + 1) / self.warm
        pw = self.warm / self.total
        q = min(1.0, max(0.0, (self.progress(step, seconds) - pw) / max(1e-9, 1 - pw)))
        return self.peak * 0.5 * (1 + math.cos(math.pi * q))


class Clock:
    """Training seconds: wall time minus what ran inside off() (held-out evals,
    checkpoints), plus what earlier attempts trained (a resume carries it)."""

    def __init__(self, elapsed: float = 0.0):
        self.base, self.start, self.paused = elapsed, time.monotonic(), 0.0

    def seconds(self) -> float:
        return self.base + time.monotonic() - self.start - self.paused

    @contextlib.contextmanager
    def off(self):
        t = time.monotonic()
        try:
            yield
        finally:
            self.paused += time.monotonic() - t


def reset_peak() -> None:
    """Starts peak_gb()'s count afresh (no GPU: nothing to do)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float | None:
    """The most GPU memory torch has held since reset_peak(), in GB (1e9 bytes), for the
    fit reports: what a phase really needed, beside e2e_memory()'s estimate. None
    without a GPU."""
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None


class Guard:
    """Applies an update only when its loss and gradient norm are finite: one NaN
    gradient would turn every trained weight into NaN through the optimizer. Skips are
    counted; `max_run` in a row, or more than max(max_run, total // 20) in all, raise.
    After each update the QuantWeights' scales are kept within their range (project_)."""

    def __init__(self, total: int, qws=(), max_run: int = 8, skipped: int = 0, run: int = 0):
        self.total, self.qws, self.max_run, self.skipped, self.run = total, list(qws), max_run, skipped, run

    def step(self, loss: torch.Tensor, params: list, opt, where: str) -> bool:
        finite = bool(torch.isfinite(loss))
        if finite:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        ok = finite and bool(torch.isfinite(gnorm))
        if ok:
            opt.step()
            for q in self.qws:
                q.project_()
            self.run = 0
        else:
            self.skipped += 1
            self.run += 1
            log(f"qat: {where}: non-finite {'loss' if not finite else 'gradient'}, update skipped")
        opt.zero_grad(set_to_none=True)
        if not ok and (self.run >= self.max_run or self.skipped > max(self.max_run, self.total // 20)):
            raise NonFiniteTraining(f"training diverged: {self.skipped} non-finite steps ({self.run} in a row) at {where}")
        return ok


def train_params(module) -> list[nn.Parameter]:
    """The trainable tensors under module, in fp32: each QuantWeight's own, as kquant marks
    them (latent weight, d and dmin, scales and mins), and every other floating
    parameter (norms)."""
    qp = {id(p) for q in quant_weights(module) for p in q.parameters()}
    out = []
    for p in module.parameters():
        if id(p) not in qp and p.is_floating_point():
            if p.dtype != torch.float32:
                p.data = p.data.float()
            p.requires_grad_(True)
        if p.requires_grad:
            out.append(p)
    return out


def param_groups(module, *, d_lr: float = D_LR, scale_lr: float = SCALE_LR) -> list[dict]:
    """AdamW groups over train_params(module): latent weights and norms at the base rate,
    the QuantWeights' d and dmin at d_lr times it, their scales and mins (in units of
    their integer grid) at scale_lr times it. lr_scale is each group's factor."""
    params = train_params(module)
    live = {id(p) for p in params}
    ds = [p for q in quant_weights(module) for p in (q.d, q.dmin) if p is not None and id(p) in live]
    scs = [p for q in quant_weights(module) for p in (q.scales, q.mins) if p is not None and id(p) in live]
    special = {id(p) for p in ds + scs}
    groups = [{"params": [p for p in params if id(p) not in special], "lr_scale": 1.0},
              {"params": ds, "lr_scale": d_lr}, {"params": scs, "lr_scale": scale_lr}]
    return [g for g in groups if g["params"]]


def _optimizer(module, lr: float, betas=(0.9, 0.95)):
    groups = param_groups(module)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.0, betas=betas), [p for g in groups for p in g["params"]]


def _set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr * g["lr_scale"]


# ---------- Block-AP ----------

def block_windows(rows: list[dict], length: int, n: int = 0, seed: int = 0, vocab_map=None, *,
                  report: dict | None = None) -> torch.Tensor:
    """(n, length) windows for Block-AP, in the trimmed model's ids with vocab_map (a
    window holding a dropped id is left out), shuffled. Each trace row of at least
    `length` tokens gives its first window, from its prompt on. When those are fewer than
    n, the rows long enough give more: tokens [length, 2 x length) of the same trace, then
    [2 x length, 3 x length), and so on until n. A window is never two traces spliced
    and never overlaps another. One that starts mid-answer is still good calibration:
    Block-AP fits each layer to the bf16 layer on the same inputs, whatever they are. A
    row is one whole thinking answer, often several windows long, so one window per row
    can fall well short of the count asked. n = 0: each row's first window. report, when
    given, gets the counts: asked, rows_long_enough, windows, by_offset (windows taken
    at each start k x length)."""
    long = [r["prompt"] + r["tokens"] for r in rows if len(r["prompt"]) + len(r["tokens"]) >= length]
    if not long:
        raise ValueError(f"no trace of {length} tokens or more")
    lut = traces.vocab_lut(vocab_map) if vocab_map is not None else None
    picked, dropped = [], 0
    for k in itertools.count():
        tier = [s[k * length:(k + 1) * length] for s in long if len(s) >= (k + 1) * length]
        if not tier:
            break
        w = torch.tensor(tier)
        if lut is not None:
            w = torch.where(w < len(lut), lut[w.clamp(max=len(lut) - 1)], -1)
            ok = (w >= 0).all(1)
            dropped += int((~ok).sum())
            w = w[ok]
        w = w[torch.randperm(len(w), generator=torch.Generator().manual_seed(seed + k))]
        picked.append(w[: n - sum(map(len, picked))] if n else w)
        if not n or sum(map(len, picked)) >= n:
            break
    if dropped:
        log(f"block: {dropped} windows hold ids the trimmed vocab dropped; left out")
    w = torch.cat(picked)
    if len(picked) > 1:  # later windows mixed in among the first ones
        w = w[torch.randperm(len(w), generator=torch.Generator().manual_seed(seed))]
    if report is not None:
        report.update(asked=n, rows_long_enough=len(long), windows=len(w), by_offset=[len(x) for x in picked])
    return w


def free_bytes(device) -> int | None:
    """Memory the hidden-state caches can take on device: MemAvailable of /proc/meminfo
    for the CPU, torch's free VRAM for a GPU; None when it cannot be read."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return int(torch.cuda.mem_get_info(dev)[0])
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def fit_windows(n: int, length: int, hidden: int, elem: int, device, *, free: int | None = None) -> dict:
    """How many of n Block-AP windows of `length` tokens the two hidden-state caches
    (2 x windows x length x hidden x elem bytes) leave room for on device, after
    CACHE_RESERVE_GB. free: the device's free bytes (default free_bytes(device)).
    Returns {"windows", "asked", "cache_gb", "free_gb", "device"}; raises when fewer
    than MIN_BLOCK_WINDOWS (or n, if smaller) fit."""
    dev = torch.device(device)
    per = 2 * length * hidden * elem
    free = free_bytes(dev) if free is None else free
    keep = n
    if free is not None:
        room = free - CACHE_RESERVE_GB["cuda" if dev.type == "cuda" else "cpu"] * 1e9
        keep = max(0, min(n, int(room // per)))
        if keep < min(n, MIN_BLOCK_WINDOWS):
            raise RuntimeError(f"block: {free / 1e9:.1f} GB free on {dev} holds the hidden states of {keep} windows "
                               f"of {length} tokens ({per / 1e9:.3f} GB each, {n} asked, at least "
                               f"{min(n, MIN_BLOCK_WINDOWS)} needed); free memory, or pass --cache-device")
    return {"windows": keep, "asked": n, "cache_gb": round(keep * per / 1e9, 2),
            "free_gb": None if free is None else round(free / 1e9, 2), "device": str(dev)}


def _advance(layer, xs: list, kw_of, device, cache, amp) -> None:
    """Replaces each cached hidden state in xs by the layer's output on it, in place, one
    batch at a time, so xs never holds more than one cache plus the batch being made. A
    new list built beside the old one holds the old cache whole until the new one is
    finished: with the other cache, three caches where fit_windows budgets two."""
    with torch.no_grad(), amp:
        for j in range(len(xs)):
            xs[j] = run_layer(layer, xs[j].to(device), kw_of[len(xs[j])]).to(cache)


def _rel_err(layer, xs, ys, kw_of, device, amp) -> float:
    with torch.no_grad(), amp:
        num = sum(float((run_layer(layer, x.to(device), kw_of[len(x)]).float() - y.to(device).float()).pow(2).sum())
                  for x, y in zip(xs[:4], ys[:4]))
    return num / max(sum(float(y.float().pow(2).sum()) for y in ys[:4]), 1e-12)


def block_ap(model, wins: torch.Tensor, type_map, codes: dict | None = None, *, batch: int = 4, epochs: int = 2,
             lr: float = 1e-4, cache_device: str | None = None, work: str | Path | None = None,
             freeze_steps: int = 0, progress=lambda pct, msg: None) -> list[dict]:
    """Block-AP (EfficientQAT's block phase) in the K-quant grids, in place: the embedding
    and head go into their grid at their codes and stay there untrained; then each
    decoder layer in turn is quantized (from codes) and trained, its QuantWeights and
    norms only, to map the quantized layers' own hidden states to the bf16 layer's
    output on the bf16 layers' hidden states (relative MSE). Only one layer's optimizer
    state exists at a time; hidden states live on cache_device. With work, each finished
    layer is saved there and a rerun resumes after the last one. The codes stay free
    (freeze_steps 0): this state is trained further, not shipped. Returns per layer the
    relative error at the codes it started from and after training."""
    from safetensors.torch import load_file, save_file
    type_map = load_type_map(type_map)
    codes = codes or {}
    device = next(model.parameters()).device
    cache = torch.device(cache_device) if cache_device else device
    amp = _amp(device)
    for p in model.parameters():
        p.requires_grad_(False)
    work = Path(work) if work else None
    report = json.loads((work / "block.json").read_text())["layers"] if work and (work / "block.json").exists() else []
    if report:
        log(f"block: resuming after layer {len(report)}")
    kw_of: dict[int, dict] = {}  # the layer kwargs per batch size (windows are all one length)

    def inputs():
        xs = []
        with torch.no_grad(), amp:
            for i in range(0, len(wins), batch):
                h, kw = layer_inputs(model, wins[i:i + batch].to(device))
                xs.append(h.to(cache))
                kw_of.setdefault(h.shape[0], kw)
        return xs

    x_t = inputs()  # the bf16 model's hidden states entering layer 0
    log(f"block: two hidden-state caches of {2 * sum(x.numel() * x.element_size() for x in x_t) / 1e9:.1f} GB on {cache}")
    emb, head = embed_names(model)
    for m in _wrap(model, [n for n in (emb, head) if n in type_map], type_map, codes).values():
        m.qw.requires_grad_(False)
    x_s = inputs() if emb in type_map else [x.clone() for x in x_t]  # what the quantized embedding gives
    lp, layers = _layers_prefix(model), decoder_layers(model)
    for li, layer in enumerate(layers):
        _advance(layer, x_t, kw_of, device, cache, amp)  # the bf16 layer's output replaces its input
        new = _wrap(model, [n for n in type_map if n.startswith(f"{lp}.{li}.")], type_map, codes)
        if li < len(report):
            for p in layer.parameters():  # its norms were trained and saved in fp32
                if p.is_floating_point():
                    p.data = p.data.float()
            layer.load_state_dict(load_file(str(work / f"layer{li}.safetensors"), device=str(device)), strict=True)
        else:
            row = _train_block(layer, li, x_s, x_t, kw_of, device, amp, epochs, lr, freeze_steps)
            row["nudged"] = sum(m.qw.nudged for m in new.values())
            report.append(row)
            for p in layer.parameters():
                p.requires_grad_(False)
            if work:
                work.mkdir(parents=True, exist_ok=True)
                check_finite(layer)
                save_file({k: _cpu(v) for k, v in layer.state_dict().items()}, str(work / f"layer{li}.safetensors"))
                tmp = work / "block.json.tmp"
                tmp.write_text(json.dumps({"layers": report}, indent=1))
                os.replace(tmp, work / "block.json")
            log(f"block: layer {li + 1}/{len(layers)} relative error {row['rel_err_ptq']:.5f} -> "
                f"{row['rel_err_trained']:.5f}")
        _advance(layer, x_s, kw_of, device, cache, amp)  # the quantized layer's, likewise: two caches in all
        progress(100 * (li + 1) / len(layers), f"layer {li + 1}/{len(layers)}")
    return report


def _train_block(layer, li, x_s, y_t, kw_of, device, amp, epochs, lr, freeze_steps) -> dict:
    opt, params = _optimizer(layer, lr, betas=(0.9, 0.99))
    steps = epochs * len(x_s)
    sched, guard = Schedule(lr, steps, warm=steps // 20), Guard(steps, quant_weights(layer))
    before = _rel_err(layer, x_s, y_t, kw_of, device, amp)
    order, s = list(range(len(x_s))), 0
    for ep in range(epochs):
        random.Random(li * 100 + ep).shuffle(order)
        for j in order:
            if freeze_steps and steps - s == freeze_steps:
                freeze_codes(layer)
            _set_lr(opt, sched.lr(s, 0.0))
            with amp:
                out = run_layer(layer, x_s[j].to(device), kw_of[len(x_s[j])])
                y = y_t[j].to(device).float()
                loss = (out.float() - y).pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)
            guard.step(loss, params, opt, f"layer {li} step {s + 1}/{steps}")
            s += 1
    after = _rel_err(layer, x_s, y_t, kw_of, device, amp)
    return {"layer": li, "rel_err_ptq": round(before, 6), "rel_err_trained": round(after, 6), "steps": steps,
            "skipped": guard.skipped}


# ---------- KL to the teacher ----------

def vocab_index(path_or_dict) -> dict | None:
    """vocab_index.json ({"kept": [old ids in new order], "old_to_new": ...}), or None."""
    if not path_or_dict:
        return None
    return path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text())


def teacher_index(student, teacher, vocab: dict | None) -> torch.Tensor | None:
    """The teacher ids the student keeps, in the student's order, when the teacher has the
    untrimmed vocab (sft.kd_kl's index; it also maps student ids back to the teacher's);
    None when both have one vocab (the trimmed bf16 teacher)."""
    ns, nt = student.config.vocab_size, teacher.config.vocab_size
    if ns == nt:
        return None
    if not vocab:
        raise ValueError(f"student vocab {ns} != teacher vocab {nt}: give the vocab index")
    kept = torch.tensor(vocab["kept"], dtype=torch.long)
    if len(kept) != ns or int(kept.max()) >= nt:
        raise ValueError(f"vocab index keeps {len(kept)} ids (max {int(kept.max())}); student has {ns}, teacher {nt}")
    return kept.to(next(student.parameters()).device)


def kd_loss(student, teacher, ids, mask, w, *, index=None, reverse: bool = False):
    """(weighted KL between teacher and student over the positions with weight, the
    student's logits there, which positions). Forward KL(teacher || student) on
    teacher-forced text, reverse on the student's own rollouts; with index the teacher
    reads the same text in its own ids and its distribution is cut to the kept vocab."""
    device = next(student.parameters()).device
    ids, mask, w = ids.to(device), mask.to(device), w.to(device)
    t_ids = ids if index is None else index[ids]
    with torch.no_grad(), _amp(device):
        t = teacher(input_ids=t_ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    with _amp(device):
        s = student(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    keep = w > 0
    s_keep, t_keep = s[keep], t[keep]
    del s, t
    return kd_kl(s_keep, t_keep, weights=w[keep], index=index, reverse=reverse), s_keep, keep


def mean_kl(student, teacher, batches, *, index=None, reverse: bool = False) -> float:
    """Weighted mean per-token KL over batches (the held-out gate metrics)."""
    was = student.training
    student.eval()
    total = n = 0.0
    with torch.no_grad():
        for ids, mask, w, *_ in batches:
            kl, _, keep = kd_loss(student, teacher, ids, mask, w, index=index, reverse=reverse)
            wsum = float(w.to(keep.device)[keep].sum())
            total += float(kl) * wsum
            n += wsum
    student.train(was)
    return total / max(n, 1e-12)


# ---------- the whole-model training loop ----------

def _save_ckpt(path: Path, model, opt, st: dict) -> None:
    """Atomic: written beside, then renamed over the last one; never a non-finite model."""
    check_finite(model)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, tmp)
    os.replace(tmp, path)


def fit(model, data, n_per_epoch: int, total: int, step_loss, *, lr: float, warm: int = 0, max_minutes: float = 0.0,
        out: str | Path | None = None, ckpt_minutes: float = 20.0, ckpt_every: int = 0, eval_fn=None,
        eval_every: int = 0, freeze_steps: int = FREEZE_STEPS, name: str = "qat") -> dict:
    """The loop of every whole-model phase. data(epoch) iterates that epoch's batches
    (n_per_epoch of them, the same ones for the same epoch); step_loss(batch) -> (loss,
    real tokens). Every trainable tensor is updated by AdamW (param_groups) at the
    Schedule's lr. The time cap is checked before each step; held-out evals (eval_fn ->
    dict, every eval_every steps) and checkpoints (out/ckpt.pt every ckpt_minutes of
    training, or every ckpt_every steps) run off the training clock. A rerun with
    out/ckpt.pt resumes its step, optimizer, counters and training seconds. The codes
    freeze for about the last freeze_steps steps, but only in a run of at least
    2 x freeze_steps steps (by steps or by time at the rate so far): in a shorter one
    the freeze would come at step 1, and a rate measured on it would leave out the
    latent weights' gradients and updates. Returns the report: steps, real tokens,
    minutes trained, whether the cap fired, skips, the held-out history, tokens per
    training second after this attempt's first step (which allocates the optimizer's
    state), and peak_gb, the GPU memory torch held (None without a GPU). The caller
    saves the model."""
    device = next(model.parameters()).device
    reset_peak()
    opt, params = _optimizer(model, lr)
    st = {"step": 0, "train_s": 0.0, "eval_s": 0.0, "skipped": 0, "run": 0, "ema": None, "tokens": 0,
          "tokens_applied": 0, "hist": [], "frozen": False, "cap_fired": False}
    ckpt = Path(out) / "ckpt.pt" if out else None
    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt and ckpt.exists():
        saved = torch.load(ckpt, map_location=device)
        model.load_state_dict(saved["model"], strict=True)  # kquant makes room for frozen codes
        opt.load_state_dict(saved["opt"])
        st.update(saved["state"])
        del saved
        log(f"{name}: resumed at step {st['step']}/{total} after {st['train_s'] / 60:.1f} min of training")
    sched = Schedule(lr, total, warm, 60 * max_minutes)
    guard = Guard(total, quant_weights(model), skipped=st["skipped"], run=st["run"])
    clock = Clock(st["train_s"])
    last_ckpt = st["train_s"]

    def evaluate():
        t = time.monotonic()
        with clock.off():
            st["hist"].append({"step": st["step"], **eval_fn(), "train": st["ema"]})
        st["eval_s"] += time.monotonic() - t
        log(f"{name}: step {st['step']}/{total} held-out {st['hist'][-1]}")

    def save():
        st.update(train_s=clock.seconds(), skipped=guard.skipped, run=guard.run)
        with clock.off():
            _save_ckpt(ckpt, model, opt, st)

    if eval_fn and not st["hist"]:
        evaluate()
    model.train()
    it, it_epoch = None, -1
    first, too_short = None, False  # (training seconds, tokens) after this attempt's first step
    while True:
        p = sched.progress(st["step"], clock.seconds())
        if p >= 1:  # checked first, whatever the last step did
            st["cap_fired"] = st["step"] < total
            break
        left = (1 - p) * st["step"] / p if p > 0 else math.inf  # steps left at the rate so far
        if freeze_steps and not st["frozen"] and left <= freeze_steps:
            if st["step"] + left >= 2 * freeze_steps - 1e-6:  # the whole run, at the rate so far
                freeze_codes(model)
                st["frozen"] = True
                log(f"{name}: codes frozen at step {st['step']} (about {left:.0f} steps left)")
            elif not too_short:
                too_short = True
                log(f"{name}: about {st['step'] + left:.0f} steps in all, under 2 x {freeze_steps}: "
                    "the codes stay free")
        epoch, pos = divmod(st["step"], n_per_epoch)
        if epoch != it_epoch:
            it, it_epoch = iter(data(epoch)), epoch
            for _ in range(pos):
                next(it)
        batch = next(it)
        _set_lr(opt, sched.lr(st["step"], clock.seconds()))
        loss, n_tok = step_loss(batch)
        if guard.step(loss, params, opt, f"{name} step {st['step'] + 1}/{total}"):
            v = float(loss.detach())
            st["ema"] = v if st["ema"] is None else 0.95 * st["ema"] + 0.05 * v
            st["tokens_applied"] += n_tok
        st["tokens"] += n_tok
        st["step"] += 1
        if first is None:
            first = (clock.seconds(), st["tokens"])
        if eval_fn and eval_every and st["step"] % eval_every == 0:
            evaluate()
            model.train()
        if ckpt and ((ckpt_every and st["step"] % ckpt_every == 0)
                     or (ckpt_minutes and clock.seconds() - last_ckpt >= 60 * ckpt_minutes)):
            save()
            last_ckpt = clock.seconds()
    st.update(train_s=clock.seconds(), skipped=guard.skipped, run=guard.run)
    if eval_fn and st["hist"][-1]["step"] != st["step"]:
        evaluate()
    if st["tokens_applied"] == 0:
        raise NonFiniteTraining(f"{name}: no finite training step")
    check_finite(model)
    if st["cap_fired"]:
        log(f"{name}: time cap of {max_minutes:g} min reached at step {st['step']}/{total}")
    after = (st["tokens"] - first[1], st["train_s"] - first[0]) if first else (0, 0.0)
    return {"steps": st["step"], "total": total, "tokens": st["tokens"], "tokens_applied": st["tokens_applied"],
            "minutes": round(st["train_s"] / 60, 2), "eval_minutes": round(st["eval_s"] / 60, 2),
            "cap_fired": st["cap_fired"], "skipped": guard.skipped, "frozen": st["frozen"],
            "freeze_steps": freeze_steps, "lr": lr, "tok_s": round(after[0] / after[1], 1) if after[1] > 0 else None,
            "peak_gb": peak_gb(), "held": st["hist"]}


# ---------- end-to-end KD ----------

def e2e_memory(weights: float = 3.66e9, *, layers: int = 40, hidden: int = 2560, inter: int = 8192,
               vocab: int = 100352, teacher_weights: float = 3.66e9, teacher_vocab: int = 100352,
               tokens: int = 16384, table: int = 100352 * 2560, qparams: float = 34 / 256,
               chunk: int = KD_CHUNK) -> dict[str, float]:
    """GB on the GPU for a whole-model QAT step of `tokens` tokens with gradient
    checkpointing: fp32 master weights, gradients and AdamW's two moments (16 bytes per
    weight) and the same for the learnable d, scales and mins (Q2_K's 34 per 256
    weights, the most), the frozen int8 codes, the bf16 teacher, the layer inputs
    checkpointing keeps, one layer's activations recomputed with their gradients, the
    fake-quant copies of the largest tensor, both models' logits (and the copies at the
    weighted positions) with the student's gradient and one KD chunk in fp32, and ~6 GB
    of CUDA context and allocator slack. The defaults are the student before its vocab
    trim (3.66B weights, 100k vocab: the conservative case) against the untrimmed
    teacher, at 16k tokens."""
    gb = 1e9
    parts = {
        "weights_grads_adam": weights * 16 / gb,
        "scales_mins": weights * qparams * 16 / gb,
        "frozen_codes": weights / gb,
        "teacher_bf16": teacher_weights * 2 / gb,
        "checkpointed_inputs": layers * tokens * hidden * 2 / gb,
        "layer_recompute": 2 * tokens * (12 * hidden + 4 * inter) * 2 / gb,
        "fake_quant_table": table * 4 * 3 / gb,
        "logits": tokens * (3 * vocab + 2 * teacher_vocab) * 2 / gb + chunk * (vocab + teacher_vocab) * 4 * 3 / gb,
        "cuda_slack": 6.0,
    }
    parts = {k: round(v, 2) for k, v in parts.items()}
    parts["total"] = round(sum(parts.values()), 1)
    return parts


def model_memory(student, teacher, tokens: int) -> dict[str, float]:
    """e2e_memory for these two models."""
    c = student.config
    emb = student.get_input_embeddings()
    return e2e_memory(sum(q.weight.numel() for q in quant_weights(student)) or sum(p.numel() for p in student.parameters()),
                      layers=c.num_hidden_layers, hidden=c.hidden_size, inter=c.intermediate_size, vocab=c.vocab_size,
                      teacher_weights=sum(p.numel() for p in teacher.parameters()),
                      teacher_vocab=teacher.config.vocab_size, tokens=tokens, table=emb.weight.numel())


def _trainable_setup(student) -> None:
    student.config.use_cache = False
    if next(student.parameters()).device.type == "cuda":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def _frozen_teacher(teacher):
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def e2e_kd(student, teacher, train_rows: list[dict], held_rows: list[dict], out: str | Path, *, vocab=None,
           length: int = 16384, batch: int = 1, tokens: float = 40e6, lr: float = 2e-5, warm_frac: float = 0.02,
           max_minutes: float = 0.0, eval_every: int = 200, held_n: int = 16, held_length: int = 8192,
           ckpt_minutes: float = 20.0, ckpt_every: int = 0, prompt_weight: float = PROMPT_WEIGHT,
           freeze_steps: int = FREEZE_STEPS, seed: int = 0) -> dict:
    """Teacher-forced KD of the whole quantized student (S2.4): KL(teacher || student)
    over the kept vocab on trace rows from their prompt, up to `length` tokens (prompt
    tokens at prompt_weight), until `tokens` real tokens (over as many epochs as that
    takes) or the time cap. Held-out KL on held_n rows of held_length tokens of the
    held-out problems, off the clock. vocab: vocab_index.json when the student is
    trimmed (traces are in the original's ids). Checkpoints and resumes in out."""
    vocab = vocab_index(vocab)
    thaw_codes(student)
    _trainable_setup(student)
    _frozen_teacher(teacher)
    on_cuda(student, teacher)
    if vocab and len(vocab["kept"]) != student.config.vocab_size:
        raise ValueError(f"vocab index keeps {len(vocab['kept'])} ids, the student has {student.config.vocab_size}")
    index = teacher_index(student, teacher, vocab)
    held = list(itertools.islice(traces.batches(held_rows, held_length, 1, vocab_map=vocab,
                                                prompt_weight=prompt_weight, seed=0), held_n))

    def data(epoch):
        return traces.batches(train_rows, length, batch, vocab_map=vocab, prompt_weight=prompt_weight,
                              seed=seed + 1 + epoch)

    n_per_epoch = tok_per_epoch = 0
    for _, mask, _ in data(0):
        n_per_epoch += 1
        tok_per_epoch += int(mask.sum())
    if not n_per_epoch:
        raise ValueError(f"no trace row with an answer token inside {length} tokens")
    total = max(1, math.ceil(n_per_epoch * tokens / tok_per_epoch)) if tokens else n_per_epoch
    mem = model_memory(student, teacher, length * batch)
    log(f"e2e: {total} steps of {batch} x <= {length} tokens ({tok_per_epoch / 1e6:.1f}M tokens per epoch, "
        f"{total / n_per_epoch:.2f} epochs), {len(held)} held-out rows; GPU estimate {mem['total']} GB")
    if torch.cuda.is_available() and mem["total"] > torch.cuda.get_device_properties(0).total_memory / 1e9:
        log(f"e2e: the estimate is above this GPU's memory; expect an out-of-memory error ({mem})")

    def step_loss(b):
        ids, mask, w = b
        loss, _, _ = kd_loss(student, teacher, ids, mask, w, index=index)
        return loss, int(mask.sum())

    report = fit(student, data, n_per_epoch, total, step_loss, lr=lr, warm=max(1, int(warm_frac * total)),
                 max_minutes=max_minutes, out=out, ckpt_minutes=ckpt_minutes, ckpt_every=ckpt_every,
                 eval_fn=lambda: {"kl": mean_kl(student, teacher, held, index=index)}, eval_every=eval_every,
                 freeze_steps=freeze_steps, name="e2e")
    report.update(length=length, batch=batch, epochs=round(report["steps"] / n_per_epoch, 3), memory_gb=mem)
    return report


# ---------- on-policy rounds: rollouts, labels, keep rules ----------

@dataclasses.dataclass
class OnPolicy:
    """The settings of an on-policy run, kept in <work>/config.json."""
    teacher: str
    problems: str  # codebench dir (codecontests.jsonl, apps.jsonl; livecodebench.jsonl for the dedupe)
    traces: str = ""  # the e2e text: its problems are not fresh
    vocab_index: str = ""
    base: str = ""  # the student's base checkpoint, when it moved since the state was saved
    llama_cpp: str = ""  # for the final export
    rounds: int = MAX_ROUNDS
    fresh: int = 250  # fresh problems per round
    rollouts: int = 4  # per fresh problem
    held: int = 128  # held-out problems, the same every round
    held_rollouts: int = 2
    max_tokens: int = MAX_TOKENS  # thinking budget, as the eval's; then the forced answer
    length: int = MAX_MODEL_LEN  # training window: a whole rollout
    batch: int = 1
    epochs: int = 2
    lr: float = 1e-5
    ce_weight: float = CE_WEIGHT
    # A session's budgets; qat onpolicy --max-minutes and --rollout-tokens-cap set them for
    # the sessions that follow. max_minutes caps a round's training (fit anneals the rate
    # by time to it, and freezes the codes before it); rollout_tokens_cap stops the fresh
    # rollouts once they hold that many answer tokens (the held-out ones are never cut:
    # the keep rules compare the same problems every round). 0: no cap.
    max_minutes: float = 0.0
    rollout_tokens_cap: int = 0
    freeze_steps: int = FREEZE_STEPS
    seqs: int = SEQS["b200"]
    port: int = 8095
    seed: int = 0


def _complete_rows(path: Path, n: int) -> list[dict]:
    """The rows of complete problems in path (n each); a line cut by a stop and the rows of
    an incomplete problem are dropped from the file."""
    if not path.exists():
        return []
    text = path.read_text()
    whole = text[: text.rfind("\n") + 1]
    rows = [json.loads(line) for line in whole.splitlines() if line.strip()]
    per = Counter(r["id"] for r in rows)
    keep = [r for r in rows if per[r["id"]] == n]
    if whole != text or len(keep) != len(rows):
        log(f"onpolicy: dropping {len(rows) - len(keep)} rows of incomplete problems and a cut line from {path}")
        _write_jsonl(path, keep)
    return keep


def sample_rollouts(url: str, tok, probs: list[dict], n: int, out: str | Path, *, max_tokens: int = MAX_TOKENS,
                    workers: int = 64, max_fail: float = 0.03, max_model_len: int = MAX_MODEL_LEN,
                    token_cap: int = 0) -> list[dict]:
    """n rollouts per problem from the served student, asked and sampled as the eval asks
    (opd.render, opd.sample); a thinking that ends without an answer gets the eval's
    forced answer (opd.continue_forced), as in ternary.gen. Rows in traces' format
    (passed None until labelled), appended to out per problem; a rerun skips the problems
    already there. A problem whose request fails after opd's retries is skipped; more
    than max_fail of them, or a dead server, raise. With token_cap, no problem is started
    once out's rows hold that many answer tokens, those of an earlier attempt included:
    sampling time follows the tokens, and a model that loops or thinks to its budget
    makes far more of them than a time guess assumes. The problems already running
    finish and are kept, so the file ends over the cap by at most what was in flight."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _complete_rows(out, n)
    seen = {r["id"] for r in rows}
    todo = [p for p in probs if p["id"] not in seen]
    max_prompt = max_model_len - max_tokens - opd.ANSWER_TOKENS - FORCED_MARGIN
    toks = sum(len(r["tokens"]) for r in rows)
    if todo and token_cap and toks >= token_cap:
        log(f"onpolicy: {out.name} holds {toks} answer tokens, the cap of {token_cap}: {len(todo)} problems not asked")
        return rows
    if not todo:
        return rows
    opd.client(workers)
    if opd.render_remote(url, opd.STUDENT, todo[0]) != opd.render(tok, todo[0]):
        raise RuntimeError(f"render() and the server's chat template disagree on {todo[0]['id']}: "
                           "the rollouts would not be asked as the eval asks")

    def ask(p):
        prompt = opd.render(tok, p)
        if len(prompt) > max_prompt:
            return None
        got = opd.sample(url, opd.STUDENT, prompt, n, max_tokens)
        for r in got:
            r["forced"] = opd.forced_reason(tok, r["tokens"], r["finish"])
            if r["forced"]:
                think = opd.open_thinking(tok, r["tokens"])
                r["tokens"] = think + opd.continue_forced(url, opd.STUDENT, prompt, think, why=r["forced"])
        return [{"id": p["id"], "prompt": prompt, "tokens": r["tokens"], "finish": r["finish"], "passed": None,
                 "forced": r["forced"]} for r in got]

    asked = failed = too_long = 0
    capped = False
    t0, toks0 = time.time(), toks
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, p): p for p in todo}
        try:
            for f in as_completed(futs):
                if f.cancelled():  # not started: the token cap was reached
                    continue
                asked += 1
                try:
                    got = f.result()
                except opd.ServerDead:
                    raise
                except Exception as e:
                    failed += 1
                    log(f"onpolicy: {futs[f]['id']} skipped ({type(e).__name__}: {e})")
                    if failed > max(2, max_fail * asked):
                        raise RuntimeError(f"onpolicy: {failed} of {asked} problems failed; stopping") from e
                    continue
                if got is None:
                    too_long += 1
                    continue
                with out.open("a") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in got))
                rows += got
                toks += sum(len(r["tokens"]) for r in got)
                if token_cap and toks >= token_cap and not capped:
                    capped = True
                    left = sum(x.cancel() for x in futs)
                    log(f"onpolicy: {out.name}: {toks} answer tokens reach the cap of {token_cap}; "
                        f"{left} problems not asked, those running finish")
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    log(f"onpolicy: {out.name}: {len(rows)} rollouts of {len(rows) // n} problems, {toks / 1e6:.2f}M tokens, "
        f"{(toks - toks0) / max(time.time() - t0, 1e-3):.0f} tok/s, {failed} failed, {too_long} prompts too long")
    return rows


def label(tok, rows: list[dict], probs: dict[str, dict]) -> list[dict]:
    """rows with passed set by the problem's tests (opd.passes: codebench.job, then
    sandbox.run_many, graded as the eval grades)."""
    ok = opd.passes(tok, [(probs[r["id"]], r["tokens"]) for r in rows])
    return [{**r, "passed": bool(k)} for r, k in zip(rows, ok)]


def thinking_text(tok, toks: list[int]) -> str:
    """A rollout's thinking as text, as scripts/thinking_report.py reads rollout rows: up
    to the first closing tag, without the forced answer's early-stop line, stripped."""
    from stages.eval import EARLY_STOP
    text = tok.decode(toks, skip_special_tokens=False)
    cut = [i for i in (text.find(t) for t in ("</think>", "<channel|>")) if i >= 0]
    text = text[:min(cut)] if cut else text
    stop = EARLY_STOP.strip()
    return (text[:text.rfind(stop)] if stop in text else text).strip()


def looped(text: str) -> bool:
    """thinking_report.py's rule: the thinking's last 1500 chars compress below LOOP_RATIO."""
    tail = text[-1500:]
    return len(tail) >= 500 and len(zlib.compress(tail.encode())) / len(tail.encode()) < LOOP_RATIO


def held_metrics(tok, rows: list[dict]) -> dict:
    """Per held-out problem, and their mean: pass@1 (over its rollouts), loop share and
    cap share (the thinking hit its budget, finish == "length")."""
    per: dict[str, dict[str, list]] = {}
    for r in rows:
        d = per.setdefault(r["id"], {"pass": [], "loop": [], "cap": []})
        d["pass"].append(float(bool(r["passed"])))
        d["loop"].append(float(looped(thinking_text(tok, r["tokens"]))))
        d["cap"].append(float(r["finish"] == "length"))
    per_p = {k: {m: sum(v) / len(v) for m, v in d.items()} for k, d in per.items()}
    out = {"problems": len(per_p), "rollouts": len(rows), "per_problem": per_p}
    for m in ("pass", "loop", "cap"):
        out[m] = round(sum(d[m] for d in per_p.values()) / max(len(per_p), 1), 4)
    return out


def keep_round(new: dict, best: dict, *, kl_drop: float = KEEP_KL_DROP, k: float = KEEP_SE) -> tuple[bool, dict]:
    """The plan's keep rules for a round's model against the last kept one, on the
    held-out problems both answered: on-policy KL drops by at least kl_drop; loop% and
    cap% rise by no more than k standard errors of the paired (per problem) difference;
    pass@1 falls by no more than k of them. Returns (kept, every rule with its numbers)."""
    common = sorted(set(new["per_problem"]) & set(best["per_problem"]))

    def paired(m):
        d = [new["per_problem"][i][m] - best["per_problem"][i][m] for i in common]
        mean = sum(d) / len(d) if d else 0.0
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
        return round(mean, 5), round(se, 5)

    drop = 1 - new["kl"] / best["kl"] if best["kl"] > 0 else 0.0
    rules = {"kl": {"new": new["kl"], "kept": best["kl"], "drop": round(drop, 5), "need": kl_drop,
                    "ok": drop >= kl_drop}}
    for m in ("loop", "cap"):
        rise, se = paired(m)
        rules[m] = {"rise": rise, "se": se, "ok": rise <= k * se}
    diff, se = paired("pass")
    rules["pass"] = {"diff": diff, "se": se, "ok": diff >= -k * se}
    rules["problems"] = {"n": len(common), "ok": len(common) > 0}
    return all(r["ok"] for r in rules.values()), rules


# ---------- on-policy rounds: the server ----------

def gpu_free(limit_mib: int = 2000, wait_s: float = 600.0, poll: float = 5.0) -> None:
    """Waits until the GPU's memory is free, the last job's processes gone; raises naming
    the processes still holding it when it is not by wait_s. No nvidia-smi: no GPU."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return
    deadline = time.monotonic() + wait_s
    while True:
        r = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=False)
        try:
            used = int(r.stdout.split()[0])
        except (IndexError, ValueError):
            used = None
        if used is not None and used < limit_mib:
            return
        if time.monotonic() > deadline:
            apps = subprocess.run([smi, "--query-compute-apps=pid,used_memory", "--format=csv"],
                                  capture_output=True, text=True, check=False).stdout
            raise RuntimeError(f"GPU not free after {wait_s:.0f} s ({used} MiB used); holders:\n{apps}")
        time.sleep(poll)


@contextlib.contextmanager
def serve_student(model_dir: str | Path, *, port: int, seqs: int, log_path: str | Path,
                  max_model_len: int = MAX_MODEL_LEN, gpu_util: float = 0.9, health_s: float = 1200.0):
    """vLLM serving model_dir alone, after the GPU-free wait: the eval's served name, an
    fp8 KV cache for rollouts (records stay bf16), its own session, so eval.stop()
    reaches the engine core too. Yields the server's URL."""
    import httpx

    from stages import eval as ev
    gpu_free()
    cmd, env = ev.vllm_command(str(model_dir))
    cmd += ["--served-model-name", opd.STUDENT, "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", str(max_model_len), "--max-num-seqs", str(seqs), "--kv-cache-dtype", "fp8",
            "--gpu-memory-utilization", str(gpu_util), "--enable-prefix-caching"]
    url = f"http://127.0.0.1:{port}"
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + health_s
        while True:
            if proc.poll() is not None:
                raise RuntimeError("vLLM exited:\n" + "\n".join(Path(log_path).read_text().splitlines()[-30:]))
            try:
                if opd.client().get(url + "/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"vLLM not healthy after {health_s:.0f} s; see {log_path}")
            time.sleep(5)
        opd.watch(url, proc.pid)
        yield url
    finally:
        ev.stop(proc)


# ---------- on-policy rounds: state and the driver ----------

def read_state(work: str | Path) -> dict:
    return json.loads((Path(work) / "state.json").read_text())


def write_state(work: str | Path, st: dict) -> None:
    p = Path(work) / "state.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    os.replace(tmp, p)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, p)


def init_onpolicy(work: str | Path, start: str | Path, cfg: OnPolicy) -> dict:
    """The run's config and state in work, the start model (e2e's state) the first kept
    one. An existing state is kept, so a rerun resumes."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1) + "\n")
    if not (work / "state.json").exists():
        write_state(work, {"best": str(_state_dir(start)), "best_held": None, "candidate": None, "trained": 0,
                           "session": 1, "streak": 0, "used": [], "rounds": [], "done": False, "final": None})
    return read_state(work)


def _config(work: str | Path) -> OnPolicy:
    return OnPolicy(**json.loads((Path(work) / "config.json").read_text()))


def _held_ids(work: Path, pool: dict, cfg: OnPolicy) -> list[str]:
    """The held-out problems, the same every round: the pool's CodeContests problems that
    traces.split holds out (crc32 of the id), so no trace trained on them, lowest bucket
    first."""
    from stages import codebench
    p = work / "held.json"
    if p.exists():
        return json.loads(p.read_text())
    cc = {q["id"] for q in codebench.load_suite("codecontests", cfg.problems)}
    ids = sorted((i for i in pool if i in cc and traces.is_held(i)), key=traces.bucket)[: cfg.held]
    if not ids:
        raise RuntimeError("no held-out problem in the pool")
    if len(ids) < cfg.held:
        log(f"onpolicy: only {len(ids)} held-out problems in the pool (wanted {cfg.held})")
    p.write_text(json.dumps(ids) + "\n")
    return ids


def _fresh_ids(rd: Path, work: Path, pool: dict, held: list[str], st: dict, cfg: OnPolicy) -> list[str]:
    """This session's fresh problems, fixed when first chosen: none held out, none in the
    e2e text, none rolled out in an earlier session."""
    p = rd / "problems.json"
    if p.exists():
        return json.loads(p.read_text())
    skip = set(held) | set(st["used"])
    if cfg.traces:
        with open(cfg.traces) as fh:
            skip |= {json.loads(line)["id"] for line in fh if line.strip()}
    ids = [i for i in pool if i not in skip][: cfg.fresh]
    if not ids:
        raise RuntimeError("no fresh problem left in the pool")
    if len(ids) < cfg.fresh:
        log(f"onpolicy: only {len(ids)} fresh problems left (wanted {cfg.fresh})")
    st["used"] = st["used"] + ids
    write_state(work, st)
    p.write_text(json.dumps(ids) + "\n")
    return ids


def run_step(step: str, work: Path, target: Path | None = None) -> None:
    """The driver's GPU steps, each in its own process, so the GPU is free once it exits:
    "serve-dir" writes target's servable checkpoint, "train" judges and trains,
    "export" packs the final model into work/final."""
    py = [sys.executable, "-m", "stages.qat"]
    work = Path(work).resolve()
    target = Path(target).resolve() if target else None
    cfg = _config(work)
    base = ["--base", cfg.base] if cfg.base else []
    if step == "serve-dir":
        cmd = py + ["export", "--state", str(target), "--hf-only", "--out", str(target / "serve")] + base
    elif step == "train":
        cmd = py + ["onpolicy", "--work", str(work), "--step", "train"]
    elif step == "export":
        cmd = py + ["export", "--state", str(target), "--out", str(work / "final"), "--llama-cpp", cfg.llama_cpp] + base
    else:
        raise ValueError(f"unknown step {step!r}")
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])


def onpolicy_round(work: str | Path, *, serve=serve_student, run=run_step) -> dict:
    """One session of the on-policy loop (S2.5). The trainer has checkpointed and exited;
    vLLM serves the model to judge (the last round's candidate, or the start model) alone,
    after the GPU-free wait; it answers the held-out problems (the same ones every
    session) and, unless the rounds are used up, the session's fresh problems, with
    forced answers; the server stops and the sandbox labels everything; then the trainer
    resumes in its own process (onpolicy_train): it judges the candidate by the keep
    rules and trains the kept model on the fresh rollouts. Each part is skipped when its
    output exists, so a rerun continues the session. Returns the state."""
    from transformers import AutoTokenizer
    work = Path(work)
    cfg, st = _config(work), read_state(work)
    if st["done"]:
        return st
    rd = work / f"round{st['session']}"
    rd.mkdir(parents=True, exist_ok=True)
    served = Path(st["candidate"] or st["best"])
    pool = {p["id"]: p for p in opd.problems(cfg.problems, seed=cfg.seed)}
    held = _held_ids(work, pool, cfg)
    fresh = [] if st["trained"] >= cfg.rounds else _fresh_ids(rd, work, pool, held, st, cfg)
    if not (rd / "rollouts.done").exists():
        if not (served / "serve" / "config.json").exists():
            run("serve-dir", work, served)
        tok = AutoTokenizer.from_pretrained(str(served / "serve"))
        log(f"onpolicy: session {st['session']}: serving {served} for {len(held)} held-out and {len(fresh)} "
            f"fresh problems")
        with serve(served / "serve", port=cfg.port, seqs=cfg.seqs, log_path=rd / "vllm.log") as url:
            kw = {"max_tokens": cfg.max_tokens, "max_model_len": cfg.length}
            sample_rollouts(url, tok, [pool[i] for i in held], cfg.held_rollouts, rd / "held.jsonl",
                            workers=max(1, cfg.seqs // cfg.held_rollouts), **kw)
            info = {"served": str(served)}
            if fresh:
                got = sample_rollouts(url, tok, [pool[i] for i in fresh], cfg.rollouts, rd / "rollouts.jsonl",
                                      workers=max(1, cfg.seqs // cfg.rollouts), token_cap=cfg.rollout_tokens_cap, **kw)
                toks = sum(len(r["tokens"]) for r in got)
                info.update(fresh_problems=len(fresh), fresh_sampled=len({r["id"] for r in got}), fresh_tokens=toks,
                            rollout_tokens_cap=cfg.rollout_tokens_cap,
                            capped=bool(cfg.rollout_tokens_cap and toks >= cfg.rollout_tokens_cap))
        (rd / "rollouts.done").write_text(json.dumps(info) + "\n")
    tok = AutoTokenizer.from_pretrained(str(served / "serve"))
    if not (rd / "held.metrics.json").exists():  # the server has stopped: the sandbox has the CPU
        rows = label(tok, _read_jsonl(rd / "held.jsonl"), pool)
        _write_jsonl(rd / "held.labelled.jsonl", rows)
        (rd / "held.metrics.json").write_text(json.dumps(held_metrics(tok, rows)) + "\n")
    if fresh and not (rd / "rollouts.labelled.jsonl").exists():
        _write_jsonl(rd / "rollouts.labelled.jsonl", label(tok, _read_jsonl(rd / "rollouts.jsonl"), pool))
    gpu_free()
    run("train", work)
    return read_state(work)


def onpolicy_train(work: str | Path, *, load=load_state, teacher=None) -> dict:
    """The trainer's part of a session. The served model's on-policy KL (reverse KL on
    every token of its held-out rollouts, teacher in this process) joins its held-out
    metrics; a candidate is kept or dropped by keep_round against the last kept model
    (the start model only sets the baseline). The run is done after MAX_NOT_KEPT rounds
    in a row not kept, or `rounds` trained. Otherwise the kept model trains `epochs`
    epochs on the session's fresh rollouts (dense reverse KL plus ce_weight x
    cross-entropy on the passing ones, _ce), and is saved as the next candidate with its
    servable checkpoint. The judging is done once per session; training resumes from
    its checkpoint."""
    work = Path(work)
    cfg, st = _config(work), read_state(work)
    if st["done"]:
        return st
    rd = work / f"round{st['session']}"
    vocab = vocab_index(cfg.vocab_index)
    t_model = _frozen_teacher(teacher if teacher is not None else load_model(cfg.teacher))
    student = None
    if not (rd / "judge.json").exists():
        served = st["candidate"] or st["best"]
        student, _ = load(served, cfg.base or None)
        held = _read_jsonl(rd / "held.labelled.jsonl")
        metrics = json.loads((rd / "held.metrics.json").read_text())
        metrics["kl"] = mean_kl(student, t_model, list(traces.batches(held, cfg.length, 1, prompt_weight=0.0)),
                                index=teacher_index(student, t_model, vocab), reverse=True)
        brief = {k: v for k, v in metrics.items() if k != "per_problem"}
        judge = {"session": st["session"], "served": served, "held": brief}
        if st["candidate"]:
            kept, rules = keep_round(metrics, st["best_held"])
            judge.update(kept=kept, rules=rules)
            st["rounds"].append({"round": st["trained"], "kept": kept, "rules": rules, "held": brief})
            if kept:
                st.update(best=st["candidate"], best_held=metrics, streak=0)
            else:
                st["streak"] += 1
                student = None  # the kept model trains next, not this one
            st["candidate"] = None
            log(f"onpolicy: round {st['trained']} {'kept' if kept else 'not kept'}: "
                + ", ".join(f"{k} {'ok' if v['ok'] else 'FAIL'}" for k, v in rules.items()))
        else:
            st["best_held"] = metrics
            log(f"onpolicy: start model: held-out on-policy KL {metrics['kl']:.4f}, pass {metrics['pass']:.3f}, "
                f"loop {metrics['loop']:.3f}, cap {metrics['cap']:.3f}")
        if st["streak"] >= MAX_NOT_KEPT or st["trained"] >= cfg.rounds:
            st.update(done=True, final=st["best"])
            log(f"onpolicy: done after {st['trained']} rounds; the kept model is {st['best']}")
        (rd / "judge.json").write_text(json.dumps(judge, indent=1) + "\n")
        write_state(work, st)
    if st["done"]:
        return st
    if student is None:
        student, _ = load(st["best"], cfg.base or None)
    meta = json.loads((_state_dir(st["best"]) / META).read_text())
    out = rd / "train"
    report = onpolicy_fit(student, t_model, _read_jsonl(rd / "rollouts.labelled.jsonl"), out,
                          index=teacher_index(student, t_model, vocab), length=cfg.length, batch=cfg.batch,
                          epochs=cfg.epochs, lr=cfg.lr, ce_weight=cfg.ce_weight, max_minutes=cfg.max_minutes,
                          freeze_steps=cfg.freeze_steps, seed=cfg.seed + st["session"])
    cand = save_state(student, out / "state", {**meta, "phase": f"onpolicy{st['trained'] + 1}"})
    write_hf(student, cfg.base or meta["base"], cand / "serve")
    (out / "train.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "ckpt.pt").unlink(missing_ok=True)
    st.update(candidate=str(cand), trained=st["trained"] + 1, session=st["session"] + 1)
    write_state(work, st)
    log(f"onpolicy: round {st['trained']} trained: {report['steps']} steps, {report['tokens'] / 1e6:.2f}M tokens, "
        f"{report['minutes']} min{' (time cap)' if report['cap_fired'] else ''}; judged next session")
    return st


def _ce(r: dict) -> bool:
    """Whether a rollout gets the cross-entropy: it passed, and its answer is the student's
    own (a forced one carries the eval's inserted stop text, which the student should not
    learn to write)."""
    return bool(r["passed"]) and not r.get("forced")


def rollout_batches(rows: list[dict], length: int, batch: int, seed: int) -> list[tuple]:
    """One epoch of (ids, mask, weights, ce) over rollouts: the rows that get the
    cross-entropy (_ce) and the rest batched apart (traces.batches each), then mixed;
    weight 0 on the prompt, 1 on every token after it."""
    out = [(*b, True) for b in traces.batches([r for r in rows if _ce(r)], length, batch,
                                               prompt_weight=0.0, seed=seed)]
    out += [(*b, False) for b in traces.batches([r for r in rows if not _ce(r)], length, batch,
                                                 prompt_weight=0.0, seed=seed)]
    random.Random(seed).shuffle(out)
    return out


def onpolicy_fit(student, teacher, rows: list[dict], out: str | Path, *, index=None, length: int = MAX_MODEL_LEN,
                 batch: int = 1, epochs: int = 2, lr: float = 1e-5, ce_weight: float = CE_WEIGHT,
                 warm_frac: float = 0.05, max_minutes: float = 0.0, ckpt_minutes: float = 20.0, ckpt_every: int = 0,
                 freeze_steps: int = FREEZE_STEPS, seed: int = 0) -> dict:
    """An on-policy round's training: dense reverse KL(student || teacher) over the kept
    vocab on every token of the student's own rollouts, plus ce_weight x cross-entropy on
    the rollouts that passed their tests with an answer of their own (_ce), `epochs`
    epochs (fit: schedule, guard, resume).
    The codes the last phase froze move again until this phase's own last steps."""
    thaw_codes(student)
    _trainable_setup(student)
    _frozen_teacher(teacher)
    on_cuda(student, teacher)
    device = next(student.parameters()).device
    n_per_epoch = len(rollout_batches(rows, length, batch, seed))
    if not n_per_epoch:
        raise ValueError("no rollout with a token inside the window")

    def step_loss(b):
        ids, mask, w, ce = b
        loss, s_keep, keep = kd_loss(student, teacher, ids, mask, w, index=index, reverse=True)
        if ce and ce_weight:
            loss = loss - ce_weight * opd.token_logprobs(s_keep, ids.to(device)[:, 1:][keep]).mean()
        return loss, int(mask.sum())

    total = n_per_epoch * epochs
    log(f"onpolicy: {len(rows)} rollouts ({sum(bool(r['passed']) for r in rows)} passing), {total} steps")
    report = fit(student, lambda e: rollout_batches(rows, length, batch, seed + e), n_per_epoch, total, step_loss,
                 lr=lr, warm=max(1, int(warm_frac * total)), max_minutes=max_minutes, out=out,
                 ckpt_minutes=ckpt_minutes, ckpt_every=ckpt_every, freeze_steps=freeze_steps, name="onpolicy")
    report.update(rollouts=len(rows), passing=sum(bool(r["passed"]) for r in rows), ce_rows=sum(map(_ce, rows)))
    return report


# ---------- export ----------

def size_bytes(model, type_map) -> int:
    """The GGUF's tensor bytes: each quantized weight at its type (kquant.bytes_for), every
    other parameter F32, as the converter writes norms; metadata and tokenizer on top."""
    type_map = load_type_map(type_map)
    q = quant_modules(model)
    n = sum(kquant.bytes_for(type_map[name], m.qw.weight.numel()) for name, m in q.items())
    qp = {id(p) for m in q.values() for p in m.qw.parameters()}
    return n + sum(4 * p.numel() for p in model.parameters() if id(p) not in qp)


def check_exact(model, gguf_path: str | Path, type_map, gguf=None) -> dict[str, int]:
    """Every quantized weight of the packed GGUF, dequantized by gguf-py, equals the
    trainer's fake-quant bit for bit (signed zeros included), at its planned type, q/k
    in the converter's row order. Raises on any difference or missing tensor; returns
    {type: tensors checked}."""
    import numpy as np

    from stages import gguf_build
    gguf = gguf or _gguf()
    type_map = load_type_map(type_map)
    q = quant_modules(model)
    heads = {"q_proj": model.config.num_attention_heads, "k_proj": model.config.num_key_value_heads}
    seen: dict[str, str] = {}
    for t in gguf.GGUFReader(str(gguf_path)).tensors:
        name = gguf_to_hf(t.name)
        if name not in q:
            continue
        qtype = t.tensor_type.name.lower()
        if qtype != type_map[name]:
            raise ValueError(f"{t.name} is {qtype} in {gguf_path}, planned {type_map[name]}")
        with torch.no_grad():
            want = q[name].qw().float().cpu().numpy()
        got = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(want.shape)
        same = np.array_equal(got.view(np.uint32), want.view(np.uint32))
        kind = name.split(".")[-2]
        if not same and kind in heads:  # the converter permutes q/k rows for llama.cpp's rotary layout
            want = np.ascontiguousarray(want[gguf_build.qk_rows(want.shape[0], heads[kind])])
            same = np.array_equal(got.view(np.uint32), want.view(np.uint32))
        if not same:
            diff = np.abs(got - want)
            raise ValueError(f"{t.name}: the packed weights are not the trained ones ({int((diff > 0).sum())} of "
                             f"{diff.size} differ, max {float(diff.max()):.3g})")
        seen[name] = qtype
    missing = sorted(set(q) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} trained weights are not in {gguf_path}, e.g. {missing[:3]}")
    return dict(Counter(seen.values()))


def export(model, type_map, hf_dir: str | Path, out_gguf: str | Path, llama_cpp: str | Path, *,
           vllm_dir: str | Path | None = None, expect_gb: float | None = None) -> dict:
    """The packed GGUF of the trained model (S2.6): its HF checkpoint (trained norms in
    fp32; hf_dir gives tokenizer and files) is converted by gguf_build.build, which puts
    every QuantWeight's own packed integers in place of the converted weight; check_exact
    then holds the file against the trainer bit for bit, and gguf_build checks types and
    (with expect_gb) the size. With vllm_dir, the bf16 checkpoint vLLM scores is written
    from gguf-py's dequantization of the file (gguf_build.vllm_checkpoint)."""
    from stages import gguf_build
    gguf = _gguf(llama_cpp)
    type_map = load_type_map(type_map)
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    trained = write_hf(model, hf_dir, out_gguf.with_name(out_gguf.stem + "-hf"), keep_fp32=True)
    with torch.no_grad():
        tensors = {name: (type_map[name], kquant.pack(m.qw.export())) for name, m in quant_modules(model).items()}
    built = gguf_build.build(str(trained), tensors, str(out_gguf), str(llama_cpp), expect_gb=expect_gb)
    del tensors
    exact = check_exact(model, out_gguf, type_map, gguf)
    report = {"gguf": str(out_gguf), "bytes": built["bytes"], "gb": round(built["gb"], 6),
              "tensor_bytes_estimate": size_bytes(model, type_map), "types": dict(Counter(type_map.values())),
              "exact": exact, "tensors": built["tensors"]}
    if vllm_dir:
        report["vllm"] = gguf_build.vllm_checkpoint(str(out_gguf), str(trained), str(vllm_dir))
    log(f"export: {out_gguf} {built['gb']:.4f} GB, {sum(exact.values())} weights equal to the trainer bit for bit")
    return report


# ---------- the commands ----------

def cmd_block(a) -> None:
    torch.manual_seed(a.seed)
    out = Path(a.out)
    if (out / "state" / META).exists():
        log(f"block: {out / 'state'} exists; nothing to do")
        return
    model = load_model(a.base)
    type_map = load_type_map(a.map)
    coverage(model, type_map)
    codes = gguf_codes(a.gguf, a.base, model, type_map, _gguf(a.llama_cpp))
    train, _ = traces.split(traces.load(a.traces))
    vmap = vocab_index(a.vocab_index)
    got: dict = {}
    wins = block_windows(train, a.length, a.windows, a.seed, vmap, report=got)
    cache_device = a.cache_device or ("cpu" if torch.cuda.is_available() else None)
    emb = model.get_input_embeddings().weight
    fit = fit_windows(len(wins), a.length, emb.shape[1], emb.element_size(), cache_device or emb.device)
    used = got
    if fit["windows"] < len(wins):
        log(f"block: only {fit['windows']} of {len(wins)} windows fit {fit['free_gb']} GB free on {fit['device']}")
        used = {}  # picked again, so the rows' first windows still come first
        wins = block_windows(train, a.length, fit["windows"], a.seed, vmap, report=used)
    warnings = []
    if len(wins) < a.windows:
        warnings.append(f"{len(wins)} windows of the {a.windows} asked: the traces give {got['windows']} "
                        f"({got['rows_long_enough']} rows of {a.length} tokens or more, windows by offset "
                        f"{got['by_offset']})" + (f", and {fit['free_gb']} GB free on {fit['device']} holds "
                                                   f"{fit['windows']}" if fit["windows"] < got["windows"] else ""))
        log(f"block: warning: {warnings[-1]}")
    log(f"block: {len(wins)} windows of {a.length} tokens ({len(wins) * a.length / 1e6:.1f}M tokens), {a.epochs} epochs; "
        f"hidden-state caches {fit['cache_gb']} GB on {fit['device']} ({fit['free_gb']} GB free)")
    reset_peak()
    t0 = time.time()
    report = block_ap(model, wins, type_map, codes, batch=a.batch, epochs=a.epochs, lr=a.lr,
                      cache_device=cache_device, work=out / "layers")
    save_state(model, out / "state", {"base": str(Path(a.base).resolve()), "type_map": type_map, "phase": "block"})
    (out / "block.json").write_text(json.dumps({"layers": report, "asked": a.windows, "windows": len(wins),
                                                "rows_long_enough": got["rows_long_enough"],
                                                "windows_by_offset": used["by_offset"], "warnings": warnings,
                                                "length": a.length, "epochs": a.epochs, "lr": a.lr, "cache": fit,
                                                "minutes": round((time.time() - t0) / 60, 1), "peak_gb": peak_gb(),
                                                "tensor_bytes": size_bytes(model, type_map)}, indent=1))
    shutil.rmtree(out / "layers", ignore_errors=True)
    log(f"block: saved to {out / 'state'}")


def cmd_e2e(a) -> None:
    torch.manual_seed(a.seed)
    out = Path(a.out)
    if (out / "e2e.json").exists() and (out / "state" / META).exists():
        log(f"e2e: {out} is done")
        return
    student, meta = load_state(a.state, a.base or None)
    teacher = load_model(a.teacher)
    train, held = traces.split(traces.load(a.traces))
    report = e2e_kd(student, teacher, train, held, out, vocab=a.vocab_index, length=a.length, batch=a.batch,
                    tokens=a.tokens, lr=a.lr, max_minutes=a.max_minutes, eval_every=a.eval_every,
                    held_n=a.held_rows, held_length=a.held_length, ckpt_minutes=a.ckpt_minutes,
                    freeze_steps=a.freeze_steps, seed=a.seed)
    save_state(student, out / "state", {**meta, "phase": "e2e"})
    (out / "e2e.json").write_text(json.dumps(report, indent=1))
    (out / "ckpt.pt").unlink(missing_ok=True)
    log(f"e2e: {report['steps']} steps, {report['tokens'] / 1e6:.1f}M tokens, {report['minutes']} min"
        f"{' (time cap)' if report['cap_fired'] else ''}; saved to {out / 'state'}")


def set_budgets(work: str | Path, **budgets) -> OnPolicy:
    """A session's budgets (OnPolicy.max_minutes, rollout_tokens_cap) in an existing run's
    config.json, for the sessions that follow (and the one a rerun continues); the
    run's other settings stay as it started."""
    cfg = _config(work)
    new = dataclasses.replace(cfg, **budgets)
    if new != cfg:
        p = Path(work) / "config.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(new), indent=1) + "\n")
        os.replace(tmp, p)
        log("onpolicy: " + ", ".join(f"{k} {getattr(cfg, k)} -> {v}" for k, v in budgets.items()
                                     if getattr(cfg, k) != v))
    return new


def cmd_onpolicy(a) -> None:
    work = Path(a.work)
    if a.step == "train":
        onpolicy_train(work)
        return
    budgets = {k: v for k, v in (("max_minutes", a.max_minutes), ("rollout_tokens_cap", a.rollout_tokens_cap))
               if v is not None}
    if not (work / "state.json").exists():
        if not (a.state and a.teacher and a.problems):
            raise SystemExit("onpolicy: a new run needs --state, --teacher and --problems")
        init_onpolicy(work, a.state, OnPolicy(
            teacher=a.teacher, problems=a.problems, traces=a.traces or "", vocab_index=a.vocab_index,
            base=a.base, llama_cpp=a.llama_cpp, rounds=a.rounds, fresh=a.fresh, rollouts=a.rollouts, held=a.held,
            lr=a.lr, seqs=a.seqs, port=a.port, seed=a.seed, **budgets))
    elif budgets:
        set_budgets(work, **budgets)
    st = read_state(work)
    while not st["done"] and not (a.until_session and st["session"] > a.until_session):
        st = onpolicy_round(work)
    if not st["done"]:
        log(f"onpolicy: stopped after session {st['session'] - 1} (--until-session); a rerun continues")
        return
    if _config(work).llama_cpp and not (work / "final" / "export.json").exists():
        run_step("export", work, Path(st["final"]))
    (work / "onpolicy.json").write_text(json.dumps({k: st.get(k) for k in ("final", "trained", "rounds")}, indent=1))
    log(f"onpolicy: the final model is {st['final']}")


def cmd_export(a) -> None:
    model, meta = load_state(a.state, a.base or None)
    base = a.base or meta["base"]
    out = Path(a.out)
    if a.hf_only:
        write_hf(model, base, out)
        log(f"export: {out} written (fake-quant weights in bf16)")
        return
    if not a.llama_cpp:
        raise SystemExit("export: --llama-cpp is needed for the GGUF")
    report = export(model, meta["type_map"], base, out / "model.gguf", a.llama_cpp,
                    vllm_dir=None if a.no_vllm else out / "vllm", expect_gb=a.expect_gb)
    (out / "export.json").write_text(json.dumps({**report, "state": str(_state_dir(a.state)),
                                                 "phase": meta.get("phase")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Event Horizon QAT: training in the K-quant grids")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("block")
    b.add_argument("--base", required=True, help="the (trimmed) bf16 HF checkpoint")
    b.add_argument("--map", required=True, help="type map JSON (horizon's knapsack)")
    b.add_argument("--gguf", nargs="+", required=True, help="llama-quantize GGUFs of --base (their codes start QAT)")
    b.add_argument("--windows", type=int, default=2048)
    b.add_argument("--length", type=int, default=2048)
    b.add_argument("--epochs", type=int, default=2)
    b.add_argument("--lr", type=float, default=1e-4)
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--cache-device", default=None, help="where hidden states are kept (default: the CPU with a GPU)")
    e = sub.add_parser("e2e")
    e.add_argument("--state", required=True, help="Block-AP's output (or any QAT state)")
    e.add_argument("--teacher", required=True)
    e.add_argument("--tokens", type=float, default=40e6)
    e.add_argument("--length", type=int, default=16384)
    e.add_argument("--batch", type=int, default=1)
    e.add_argument("--lr", type=float, default=2e-5)
    e.add_argument("--max-minutes", type=float, default=0.0)
    e.add_argument("--eval-every", type=int, default=200)
    e.add_argument("--held-rows", type=int, default=16)
    e.add_argument("--held-length", type=int, default=8192)
    e.add_argument("--ckpt-minutes", type=float, default=20.0)
    e.add_argument("--freeze-steps", type=int, default=FREEZE_STEPS,
                   help="the codes freeze for about the last N steps (only in a run of at least 2 x N); 0: never")
    o = sub.add_parser("onpolicy")
    o.add_argument("--work", required=True)
    o.add_argument("--step", choices=["all", "train"], default="all", help="train: the trainer's part of a session")
    o.add_argument("--until-session", type=int, default=0,
                   help="stop once this session is done (0: run to the end); a caller that uploads between "
                        "sessions runs one at a time")
    o.add_argument("--state", help="the start model (e2e's output)")
    o.add_argument("--teacher")
    o.add_argument("--problems", help="codebench dir")
    o.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    o.add_argument("--fresh", type=int, default=250)
    o.add_argument("--rollouts", type=int, default=4)
    o.add_argument("--held", type=int, default=128)
    o.add_argument("--lr", type=float, default=1e-5)
    o.add_argument("--max-minutes", type=float, default=None,
                   help="training minutes of each session's round, the rate annealed by time to it (0: no cap); "
                        "a later call's value holds from its session on, no value keeps the last")
    o.add_argument("--rollout-tokens-cap", type=int, default=None,
                   help="stop starting fresh rollouts once a session's hold this many answer tokens (0: no cap); "
                        "kept like --max-minutes")
    o.add_argument("--seqs", type=int, default=SEQS["b200"])
    o.add_argument("--port", type=int, default=8095)
    x = sub.add_parser("export")
    x.add_argument("--state", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--hf-only", action="store_true", help="only the servable bf16 checkpoint of the fake-quant weights")
    x.add_argument("--no-vllm", action="store_true")
    x.add_argument("--expect-gb", type=float, default=None)
    for p in (b, e, o):
        p.add_argument("--traces", required=p is not o, default="")
        p.add_argument("--vocab-index", default="", help="vocab_index.json of the trim")
        p.add_argument("--seed", type=int, default=0)
    for p in (b, e):
        p.add_argument("--out", required=True)
    for p in (b, o, x):
        p.add_argument("--llama-cpp", default="")
    for p in (e, o, x):
        p.add_argument("--base", default="", help="the base checkpoint, when it moved since the state was saved")
    a = ap.parse_args()
    {"block": cmd_block, "e2e": cmd_e2e, "onpolicy": cmd_onpolicy, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
