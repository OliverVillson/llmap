"""The code10x recipes on a dense model (experiment 03, Qwen3.8-27B; experiment 05 and
Event Horizon, Granite 4.2 3B).

A dense model has no experts, so REAP has nothing to prune and heal nothing to
repair: a recipe comes down to its bits.

  q4    r25q4's bits: every feed-forward tensor q4_k, attention q5_k, output head
        q6_k, token embedding q4_k, the rest (Gated DeltaNet) as llama.cpp's Q4_K_M
        mix had them
  q8    every tensor q8_0
  w18s  r50w95s's bits: each layer's feed-forward tensors get 2 to 8 bits by
        saliency (stages/bits.py), averaging FFN_BITS like r50w95s's experts did,
        and every other tensor is q8_0
  m35, m28  the same allocator at fewer bits, other tensors by kind (MIXED)
  m22, bh15 llama.cpp's IQ formats, by kind (FIXED)
  --map     a type map from a file, e.g. stages/horizon.py's knapsack

Every recipe becomes a type map, {GGUF tensor name: llama.cpp type}, that names every
tensor llama-quantize would quantize. llama-quantize gets one --tensor-type flag per
name, anchored and escaped (^blk\\.1\\.ffn_up\\.weight$), because it reads each flag
as an ECMAScript regex, lowercased, and gives a tensor the first flag whose pattern
regex_search-es its name (src/llama-quant.cpp, llama_tensor_get_type). Unanchored,
output.weight also matches every blk.N.attn_output.weight, and blk.1 matches
blk.10-19; checked on the pinned build, where an earlier broad flag beat a later exact
one. --pure gives anything left unnamed the base type instead of a mix's choice.
After quantizing, every tensor is read back and must have its planned type, and the
file must be within SIZE_TOLERANCE of the planned bytes, so a silent fallback (a row
width the type's blocks don't divide, a renamed tensor, a new llama.cpp default) stops
the run in seconds instead of after hours of scoring.

Subcommands, each skipping work already on disk that matches its inputs:
  gguf      the HF model to a BF16 GGUF and an 8-bit copy. Text only: llama.cpp
            writes a vision tower to a separate mmproj file, which is never made
  calib     the calibration text, work/calib-chat.txt.
            --traces: the training split of gen's traces (stages/traces.py), one
            sample per problem, decoded with the special tokens kept, up to
            CALIB_TOKENS. That is the text the model is scored on: its own long
            thinking on contest problems, deduplicated against LiveCodeBench, and
            never the held-out problems that judge the quantized model. It goes in its
            own job dir: a job whose text came from llama-server is refused.
            --model (experiment 05): the 8-bit copy answers code problems with thinking
            on (MultiPL-E and the C set, never LiveCodeBench); its thinking and
            answers, in the chat format, are the calibration text
  imatrix   the BF16 model, its layers split between GPU and RAM, measures the
            importance matrix on that text in IMATRIX_CTX-token chunks (answers think
            for 8-16k tokens; 512-token chunks never saw a long position), the output
            head included. work/imatrix.json records what it was made from, and an
            imatrix that doesn't match is deleted and made again
  quantize  one recipe into <job>/out/model.gguf, plus work/allocation.json with the
            flags, the map's hash, the imatrix's sha256 and llama.cpp's commit; the
            model is rebuilt when any of them differ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from common.progress import emit
from stages import bits, codebench, quantize
from stages._util import DRY_RUN, Job, run, write_jsonl

STAGE = "quantize"
# r50w95s's experts averaged ~3.35 bits with every other tensor at q8_0
# (/mnt/project-files/plan/model-rebuilds.md); w18s gives the FFN the same.
FFN_BITS = 3.35
FFN = {"ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}
# llama.cpp types as ggml lays them out: (weights per block, bytes per block). A row
# must divide into whole blocks, or llama-quantize silently picks another type.
BLOCK = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q8_0": (32, 34), "q6_k": (256, 210), "q5_k": (256, 176),
         "q4_k": (256, 144), "q3_k": (256, 110), "q2_k": (256, 84), "iq4_xs": (256, 136), "iq3_s": (256, 110),
         "iq3_xxs": (256, 98), "iq2_s": (256, 82), "iq2_xs": (256, 74), "iq2_xxs": (256, 66)}
# Small-model recipes (experiment 05, Granite 4.2 3B): the feed-forward by saliency at an
# average of these bits, and attention, output head and token embedding at fixed types.
# A 3B's attention and embeddings are ~30% of its weights, so q8_0 there (w18s) would
# cost more than the feed-forward saves.
MIXED = {
    "m35": (3.5, {"attn": "q5_k", "output": "q6_k", "token_embd": "q4_k"}),
    "m28": (2.8, {"attn": "q4_k", "output": "q5_k", "token_embd": "q3_k"}),
}
# Recipes by kind of tensor (kind_type). m22 and bh15 go below q2_k, where the
# allocator's ladder starts: llama.cpp's imatrix-guided 2-bit codebooks (IQ2) for the
# feed-forward, ffn_down a step up as it writes straight into the residual stream.
FIXED = {
    # r25q4: attn_qkv q5_k and the other Gated DeltaNet tensors q4_k, as Q4_K_M chose them
    "q4": {"ffn_gate_up": "q4_k", "ffn_down": "q4_k", "attn": "q5_k", "output": "q6_k", "token_embd": "q4_k",
           "attn_qkv": "q5_k", "rest": "q4_k"},
    "q8": {"ffn_gate_up": "q8_0", "ffn_down": "q8_0", "attn": "q8_0", "output": "q8_0", "token_embd": "q8_0"},
    # ~2.2-bit feed-forward, ~1.2 GB for Granite 3B
    "m22": {"ffn_gate_up": "iq2_xxs", "ffn_down": "iq2_s", "attn": "iq3_xxs", "output": "q4_k", "token_embd": "q3_k"},
    # blackhole's ~1.45 GB file, retired: 40 points per GB needs about 1.17 GB
    # (plan/event-horizon-qat.md). Kept so run 2's file can be made again.
    "bh15": {"ffn_gate_up": "iq2_s", "ffn_down": "iq3_xxs", "attn": "iq4_xs", "output": "q4_k", "token_embd": "q3_k"},
}
RECIPES = ("w18s", *MIXED, *FIXED)
SIZE_TOLERANCE = 0.03  # |file - planned bytes| / planned bytes
# Calibration problems per suite for --model, spread over each suite; LiveCodeBench is the test.
CALIB_SUITES = {"multipl-e-py": 10, "multipl-e-js": 10, "multipl-e-ts": 10, "multipl-e-cpp": 10, "c-set": 8}
CALIB_MAX_TOKENS = 6144
# Trace text for the imatrix: ~180 chunks of IMATRIX_CTX, a few minutes of BF16 prompt
# processing for a 3B on an L40S (plan S1.7).
CALIB_TOKENS = 1_500_000
IMATRIX_CTX = 8192


@dataclass
class Tensor:
    name: str
    n: int  # elements
    dims: int
    shape: tuple[int, ...] = ()  # (out, in) for a 2D weight
    type: str = ""  # its type in the source GGUF ("bf16", "f32"); "" for the synthetic shapes
    nbytes: int = 0  # its bytes in the source GGUF


def ffn_slot(name: str) -> tuple[int, str] | None:
    """(layer, gate|up|down) for a dense feed-forward weight."""
    m = re.fullmatch(r"blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight", name)
    return (int(m[1]), FFN[m[2]]) if m else None


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def llama_cpp_sha(job: Job) -> str:
    """The commit of the llama.cpp checkout that makes the files ("" without git metadata)."""
    try:
        r = subprocess.run(["git", "-C", str(job.config.llama_cpp), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


# ---------- the model's tensors ----------

def read_tensors(job: Job, path: Path) -> tuple[list[Tensor], int]:
    """The GGUF's tensors and its number of main layers (without the MTP layer)."""
    gguf = quantize.gguf_lib(job)
    r = gguf.GGUFReader(str(path))
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")

    def kv(key: str) -> int:
        f = r.fields.get(f"{arch}.{key}")
        return int(f.parts[-1][0]) if f else 0

    tensors = [Tensor(t.name, int(t.n_elements), len(t.shape), tuple(int(x) for x in reversed(t.shape.tolist())),
                      t.tensor_type.name.lower(), int(t.n_bytes))
               for t in r.tensors]
    return tensors, kv("block_count") - kv("nextn_predict_layers")


def synthetic_tensors(layers: int = 64, hidden: int = 5120, ffn: int = 17408, vocab: int = 248320) -> list[Tensor]:
    """Qwen3.8-27B's tensors from its model card: per block of four layers, three Gated
    DeltaNet (16 QK heads, 48 V heads of 128) and one gated attention (24 Q heads,
    4 KV heads of 256). For dry runs and tests."""
    out = [Tensor("token_embd.weight", vocab * hidden, 2, (vocab, hidden)),
           Tensor("output.weight", vocab * hidden, 2, (vocab, hidden)),
           Tensor("output_norm.weight", hidden, 1)]

    def w(name: str, o: int, i: int) -> None:
        out.append(Tensor(name, o * i, 2, (o, i)))

    for l in range(layers):
        b = f"blk.{l}."
        w(b + "ffn_gate.weight", ffn, hidden)
        w(b + "ffn_up.weight", ffn, hidden)
        w(b + "ffn_down.weight", hidden, ffn)
        out += [Tensor(b + "attn_norm.weight", hidden, 1), Tensor(b + "post_attention_norm.weight", hidden, 1)]
        if l % 4 == 3:
            w(b + "attn_q.weight", 2 * 24 * 256, hidden)  # queries and their output gate
            w(b + "attn_k.weight", 4 * 256, hidden)
            w(b + "attn_v.weight", 4 * 256, hidden)
            w(b + "attn_output.weight", hidden, 24 * 256)
        else:
            w(b + "attn_qkv.weight", 2 * 16 * 128 + 48 * 128, hidden)
            w(b + "attn_gate.weight", 48 * 128, hidden)
            w(b + "ssm_ba.weight", 2 * 48, hidden)
            w(b + "ssm_out.weight", hidden, 48 * 128)
            out += [Tensor(b + "ssm_a", 48, 1), Tensor(b + "ssm_dt.bias", 48, 1), Tensor(b + "ssm_norm.weight", 128, 1)]
    return out


def quantizable(t: Tensor) -> bool:
    """Whether llama-quantize quantizes this tensor (tensor_allows_quantization, the
    parts that apply to dense text models). The read-back check catches any case this
    misses: such a tensor would come out at a type the plan doesn't have."""
    n = t.name
    return (t.dims >= 2 and n.endswith("weight") and "_norm.weight" not in n
            and not any(s in n for s in ("ffn_gate_inp.weight", "ssm_conv1d", "shortconv.conv.weight", "altup",
                                         "laurel", "per_layer_model_proj", "attn_rel_b.weight", ".position_embd"))
            and n not in ("position_embd.weight", "token_types.weight"))


def source_type(t: Tensor) -> str:
    return t.type or ("f32" if t.dims < 2 else "bf16")


def bpw(qtype: str) -> float:
    block, size = BLOCK[qtype]
    return size * 8 / block


def tensor_bytes(t: Tensor, qtype: str) -> int:
    """Exact bytes at a llama.cpp type (whole blocks per row, as check_map requires)."""
    block, size = BLOCK[qtype]
    return t.n // block * size


def tensor_gb(t: Tensor, qtype: str) -> float:
    """Size at a llama.cpp type; 1D tensors (norms, biases) stay F32."""
    return t.n * (32 if t.dims < 2 else bpw(qtype)) / 8e9


def ffn_params(tensors: list[Tensor], layers: int) -> int:
    return sum(t.n for t in tensors if (s := ffn_slot(t.name)) and s[0] < layers)


def static_type(name: str, types: str | dict[str, str]) -> str:
    """A tensor's type in a recipe by kind: one type for all, or by kind. The kinds are
    token_embd, output, attn (q/k/v/output), ffn_gate_up, ffn_down, a block tensor's own
    name (attn_qkv) and rest, which falls back to attn's type."""
    if isinstance(types, str):
        return types
    part = name.split(".")[2] if name.startswith("blk.") else name.split(".")[0]
    if part in types:
        return types[part]
    if part in ("attn_q", "attn_k", "attn_v", "attn_output"):
        return types["attn"]
    if part in ("ffn_gate", "ffn_up") and "ffn_gate_up" in types:
        return types["ffn_gate_up"]
    return types.get("rest", types["attn"])


def static_gb(tensors: list[Tensor], layers: int, qtype: str | dict[str, str] = "q8_0") -> float:
    """Everything outside the main layers' feed-forward tensors, at one type or by kind."""
    return sum(tensor_gb(t, static_type(t.name, qtype)) for t in tensors
               if not ((s := ffn_slot(t.name)) and s[0] < layers))


def shape_of(tensors: list[Tensor], layers: int) -> bits.MoEShape:
    """The dense model as bits.MoEShape sees it: one always-active "expert" per layer.
    Only the feed-forward sizes matter; the size of the rest is passed as static_gb."""
    gate = next(t for t in tensors if t.name == "blk.0.ffn_gate.weight")
    vocab = next((t.shape[0] for t in tensors if t.name == "token_embd.weight"), 0)
    return bits.MoEShape(n_layers=layers, hidden=gate.shape[1], moe_intermediate=gate.shape[0], n_experts=1,
                         n_experts_active=1, vocab=vocab, n_heads=1, n_kv_heads=1, head_dim=1)


def allocate_w(tensors: list[Tensor], layers: int, energy: dict[str, list[float]] | None,
               ffn_bits: float = FFN_BITS, static: str | dict[str, str] = "q8_0"
               ) -> tuple[list[bits.LayerBits], float, float]:
    """w18s and MIXED: (per-layer FFN types, estimated size in GB, FFN average bits)."""
    fixed = static_gb(tensors, layers, static)
    n_ffn = ffn_params(tensors, layers)
    shape = shape_of(tensors, layers)
    plan = bits.allocate(shape, None, fixed + n_ffn * ffn_bits / 8e9, "q2_k", "q8_0", proj_energy=energy,
                         static=bits.static_types("q8_0"), static_gb=fixed)
    proj = shape.expert_params_per_proj
    ffn_gb = sum(proj * (2 * bits.BPW[l.gate_up] + bits.BPW[l.down]) / 8e9 for l in plan)
    return plan, fixed + ffn_gb, ffn_gb * 8e9 / n_ffn


# ---------- type maps ----------

def by_kind(tensors: list[Tensor], types: str | dict[str, str]) -> dict[str, str]:
    """A FIXED recipe (or one type for everything) as a type map."""
    return {t.name: static_type(t.name, types) for t in tensors if quantizable(t)}


def ffn_map(tensors: list[Tensor], layers: int, plan: list[bits.LayerBits],
            static: str | dict[str, str] = "q8_0") -> dict[str, str]:
    """w18s and MIXED as a type map: the main layers' feed-forward tensors from the
    plan, every other tensor (the MTP layer's feed-forward included) by kind."""
    per = {l.layer: l for l in plan}
    out = {}
    for t in tensors:
        if not quantizable(t):
            continue
        s = ffn_slot(t.name)
        if s and s[0] < layers:
            out[t.name] = per[s[0]].down if s[1] == "down" else per[s[0]].gate_up
        else:
            out[t.name] = static_type(t.name, static)
    return out


def check_map(tensors: list[Tensor], tmap: dict[str, str]) -> None:
    """Raises, listing every problem, unless the map names each tensor llama-quantize
    quantizes, and only those, at a type whose blocks divide the tensor's rows."""
    q = {t.name: t for t in tensors if quantizable(t)}
    names = {t.name for t in tensors}
    bad = [f"{n}: not in the model" for n in tmap if n not in names]
    bad += [f"{n}: llama-quantize leaves it as it is" for n in tmap if n in names and n not in q]
    bad += [f"{n}: no type planned" for n in q if n not in tmap]
    for n, ty in tmap.items():
        if not re.fullmatch(r"[a-z0-9_.]+", n):
            bad.append(f"{n}: not a name llama-quantize can match literally")
        if ty not in BLOCK:
            bad.append(f"{n}: unknown type {ty!r}")
        elif n in q and q[n].shape and q[n].shape[-1] % BLOCK[ty][0]:
            bad.append(f"{n}: rows of {q[n].shape[-1]} don't divide into {ty} blocks of {BLOCK[ty][0]}; "
                       "llama-quantize would fall back to another type")
    if bad:
        raise ValueError("the type map doesn't fit the model:\n  " + "\n  ".join(bad))


def exact(name: str) -> str:
    """A llama-quantize --tensor-type pattern that matches this one tensor name only."""
    return "^" + name.replace(".", r"\.") + "$"


def map_args(tmap: dict[str, str]) -> tuple[list[str], str]:
    """(llama-quantize flags, base type) that write every tensor of the map at its type.
    The head and embedding have their own flags, which llama-quantize applies before
    any --tensor-type; every other name gets one anchored flag, in name order, so no
    flag can match a second tensor and their order cannot matter. The base type is the
    map's most common quantized type (ties by name)."""
    args = ["--pure"]
    if "output.weight" in tmap:
        args += ["--output-tensor-type", tmap["output.weight"]]
    if "token_embd.weight" in tmap:
        args += ["--token-embedding-type", tmap["token_embd.weight"]]
    for name in sorted(tmap):
        if name not in ("output.weight", "token_embd.weight"):
            args += ["--tensor-type", f"{exact(name)}={tmap[name]}"]
    counts = Counter(ty for ty in tmap.values() if BLOCK[ty][0] > 1)
    if not counts:
        raise ValueError("a type map needs at least one quantized type")
    base = min(counts, key=lambda ty: (-counts[ty], ty))
    return args, base.upper()


def map_sha(tmap: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(tmap, sort_keys=True).encode()).hexdigest()


def planned_bytes(tensors: list[Tensor], tmap: dict[str, str]) -> int:
    """Tensor bytes of the quantized file: mapped tensors at their type, the rest as
    they are in the source."""
    return sum(tensor_bytes(t, tmap[t.name]) if t.name in tmap else t.nbytes or tensor_bytes(t, source_type(t))
               for t in tensors)


def overhead_bytes(path: Path, tensors: list[Tensor]) -> int:
    """The source file's bytes outside tensor data (metadata, tokenizer, padding), which
    llama-quantize carries over."""
    return path.stat().st_size - sum(t.nbytes for t in tensors) if path.exists() else 0


def ffn_summary(tensors: list[Tensor], layers: int, types: dict[str, str]) -> dict:
    """The feed-forward bits per tensor and on average, for these types."""
    widths, total, n = [], 0.0, 0
    for t in tensors:
        slot = ffn_slot(t.name)
        if not slot or slot[0] >= layers or t.name not in types:
            continue
        b = bpw(types[t.name])
        widths.append({"layer": slot[0], "tensor": t.name.split(".")[2], "type": types[t.name], "bits": round(b, 3)})
        total, n = total + b * t.n, n + t.n
    widths.sort(key=lambda w: (w["layer"], w["tensor"]))
    return {"ffn_bits": round(total / n, 2) if n else None, "bit_widths": widths}


def verify(job: Job, path: Path, tensors: list[Tensor], tmap: dict[str, str], est_bytes: int) -> dict[str, str]:
    """Reads every tensor of the written file back. Raises, listing each one, if a
    tensor's type differs from the plan (the map's type, or the source's for tensors
    llama-quantize leaves alone) or a tensor is missing or extra; raises if the file is
    SIZE_TOLERANCE or more off the planned bytes. Returns {tensor: type written}."""
    gguf = quantize.gguf_lib(job)
    got = {t.name: t.tensor_type.name.lower() for t in gguf.GGUFReader(str(path)).tensors}
    want = {t.name: tmap.get(t.name, source_type(t)) for t in tensors}
    bad = [f"{n}: planned {ty}, written {got.get(n, 'nothing')}" for n, ty in want.items() if got.get(n) != ty]
    bad += [f"{n}: written as {ty}, not in the model" for n, ty in got.items() if n not in want]
    if bad:
        raise RuntimeError(f"{path} doesn't hold the planned types ({len(bad)} tensors):\n  " + "\n  ".join(bad))
    size = path.stat().st_size
    if abs(size - est_bytes) / est_bytes >= SIZE_TOLERANCE:
        raise RuntimeError(f"{path} is {size / 1e9:.4f} GB, planned {est_bytes / 1e9:.4f} GB "
                           f"({size / est_bytes - 1:+.1%}, more than {SIZE_TOLERANCE:.0%} off)")
    return got


def quantize_map(job: Job, bf16: Path, imatrix: Path, out: Path, tmap: dict[str, str],
                 tensors: list[Tensor] | None = None) -> dict:
    """llama-quantize bf16 into out at exactly the map's types, then check the file.
    Returns {"tensors": {name: type}, "size_bytes", "est_bytes"}; raises on any
    mismatch (and leaves out in place to look at)."""
    tensors = tensors if tensors is not None else read_tensors(job, bf16)[0]
    check_map(tensors, tmap)
    args, base = map_args(tmap)
    est = planned_bytes(tensors, tmap) + overhead_bytes(bf16, tensors)
    quantize.quantize(job, bf16, imatrix, out, args, base)
    got = verify(job, out, tensors, tmap, est)
    return {"tensors": got, "size_bytes": out.stat().st_size, "est_bytes": est}


def imatrix_energy(job: Job, bf16: Path, imatrix: Path, layers: int) -> dict[str, list[float]]:
    """Expected output energy of each feed-forward tensor on the calibration text,
    sum_j E[x_j^2] * ||W[:, j]||^2 (the dense form of quantize.imatrix_energy)."""
    import numpy as np

    gguf = quantize.gguf_lib(job)
    im = {t.name: t for t in gguf.GGUFReader(str(imatrix)).tensors}
    energy = {k: [float("nan")] * layers for k in FFN.values()}
    for t in gguf.GGUFReader(str(bf16)).tensors:
        slot = ffn_slot(t.name)
        s2 = im.get(t.name + ".in_sum2")
        if not slot or slot[0] >= layers or s2 is None:
            continue
        n_in, n_out = (int(x) for x in t.shape.tolist()[:2])
        col2 = (quantize._f32(gguf, t.tensor_type, t.data).reshape(n_out, n_in) ** 2).sum(axis=0)
        energy[slot[1]][slot[0]] = float(np.asarray(s2.data, dtype=np.float32).reshape(-1)[:n_in] @ col2)
    if all(x != x for v in energy.values() for x in v):
        raise RuntimeError("the imatrix has no feed-forward tensors")
    return energy


# ---------- subcommands ----------

def make_ggufs(job: Job, hf_dir: Path, bf16: Path, q8: Path | None) -> None:
    """The BF16 GGUF, and the 8-bit copy that writes the calibration answers when the
    BF16 model doesn't fit the GPU (q8 is None when it does)."""
    stage = "gguf"
    lc = Path(job.config.llama_cpp)
    bf16.parent.mkdir(parents=True, exist_ok=True)
    if not bf16.exists():
        emit(stage, pct=5, msg=f"converting {hf_dir.name} to a BF16 GGUF")
        tmp = bf16.with_name(bf16.name + ".part")
        if DRY_RUN:
            tmp.write_bytes(b"GGUF dry run")
        else:
            from stages import vocab  # imports torch
            from stages.moe_utils import gguf_convert_flags

            vocab.register(lc, hf_dir)  # a trimmed checkpoint's tokenizer; a no-op on any other
            run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16",
                 "--outfile", str(tmp), *gguf_convert_flags(hf_dir)], stage)
        tmp.rename(bf16)
    if q8 and not q8.exists():
        emit(stage, pct=70, msg="the 8-bit copy")
        tmp = q8.with_name("tmp-" + q8.name)
        if DRY_RUN:
            tmp.write_bytes(b"GGUF dry run")
        else:
            run([quantize.bin_path(job, "llama-quantize"), str(bf16), str(tmp), "Q8_0"], stage)
        tmp.rename(q8)
    emit(stage, "done", 100, f"{bf16.name}, {q8.name}" if q8 else bf16.name)


def trace_calibration(job: Job, traces_path: Path, tokenizer_dir: Path, max_tokens: int = CALIB_TOKENS,
                      held_frac: float | None = None, seed: int = 0) -> Path:
    """work/calib-chat.txt from the training split of the traces (traces.split with its
    own fixed seed, the split every other step uses; `seed` only shuffles): one sample
    per problem (the first in the file), shuffled by seed, until max_tokens, each decoded
    from its prompt and answer ids with the special tokens kept, so llama-imatrix
    (--parse-special) sees the chat format the model is scored in. tokenizer_dir is the
    tokenizer that wrote the ids (the original model's, before any vocab trim).
    work/calib.json records the source and the token count."""
    from stages import traces  # imports torch

    stage = "calib"
    out, meta_path = job.path("work", "calib-chat.txt"), job.path("work", "calib.json")
    held_frac = traces.HELD_FRAC if held_frac is None else held_frac
    key = {"traces_sha256": sha256_file(traces_path), "held_frac": held_frac, "seed": seed, "max_tokens": max_tokens}
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if out.exists() and old is None:
        raise RuntimeError(f"{out} holds llama-server answers (experiment 05's calib step); "
                           "give the trace calibration its own job dir")
    if out.exists() and old.get("key") == key:
        emit(stage, "done", 100, f"{out} exists, from the same traces")
        return out
    rows = traces.load(traces_path)
    train, held = traces.split(rows, held_frac)
    first: dict[str, dict] = {}
    for r in train:
        first.setdefault(r["id"], r)
    picks = list(first.values())
    random.Random(seed).shuffle(picks)
    chosen, total = [], 0
    for r in picks:
        if total >= max_tokens:
            break
        chosen.append(r)
        total += len(r["prompt"]) + len(r["tokens"])
    if total < 2 * IMATRIX_CTX:
        raise ValueError(f"{traces_path}: {total} training tokens; llama-imatrix needs at least two chunks "
                         f"of {IMATRIX_CTX}")
    emit(stage, pct=20, msg=f"{len(chosen)} of {len(first)} training problems, {total} tokens")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    text = "".join(tok.decode(r["prompt"] + r["tokens"], skip_special_tokens=False,
                              clean_up_tokenization_spaces=False) for r in chosen)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(text)
    tmp.replace(out)
    meta_path.write_text(json.dumps({
        "key": key, "traces": str(traces_path), "tokenizer": str(tokenizer_dir), "problems": len(chosen),
        "train_problems": len(first), "held_out_problems": len({r["id"] for r in held}), "tokens": total,
        "chars": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    emit(stage, "done", 100, f"{len(chosen)} conversations, {total // 1000}k tokens")
    return out


def calib_problems(bench_dir: str) -> list[dict]:
    """Spread over each suite, and interleaved, so the imatrix's first chunks already
    cover every language when the text is longer than it reads."""
    picks = []
    for suite, n in CALIB_SUITES.items():
        rows = codebench.load_suite(suite, bench_dir)
        picks.append(rows[:: max(1, len(rows) // n)][:n])
    return [p for i in range(max(map(len, picks))) for rows in picks if i < len(rows) for p in [rows[i]]]


def chat_text(prompt: str, reasoning: str, answer: str, end: str) -> str:
    """One conversation as the model sees it: the template's prompt (thinking opened),
    the thinking, the answer and the template's end of turn."""
    return f"{prompt}{reasoning.strip()}\n</think>\n\n{answer.strip()}{end}"


def end_of_turn() -> str:
    """What the served model's chat template writes after an assistant message (Granite's
    <|end_of_text|>, Qwen's <|im_end|>), from llama-server's /apply-template. Raises
    when the template can't be read: a guessed marker can be a token the model doesn't
    have, and the imatrix would be measured on it."""
    import httpx

    from stages import eval as ev

    mark = "END-OF-ANSWER-MARK"
    try:
        text = httpx.post(f"http://127.0.0.1:{ev.PORT}/apply-template", timeout=60, json={
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": mark},
                         {"role": "user", "content": "again"}]}).json()["prompt"]
    except Exception as e:
        raise RuntimeError(f"llama-server's /apply-template failed ({e!r}); the end-of-turn marker is unknown") from e
    if mark not in text:
        raise RuntimeError(f"the chat template dropped the assistant message: {text!r}")
    tail = text.split(mark, 1)[1]
    end = tail[:tail.index("again")] if "again" in tail else tail
    # keep the end marker, not the next turn's opening role tag
    cut = re.match(r"\s*(<\|[^|]+\|>)\s*", end)
    if not cut:
        raise RuntimeError(f"no end-of-turn marker after the assistant message: {end!r}")
    return cut.group(0)


def calibration(job: Job, model: Path) -> Path:
    from stages import eval as ev

    stage = "calib"
    out = job.path("work", "calib-chat.txt")
    if out.exists():
        emit(stage, "done", 100, f"{out} exists")
        return out
    cfg = job.config
    probs = calib_problems(cfg.code_eval_dir)
    emit(stage, msg=f"{len(probs)} problems, thinking on, up to {CALIB_MAX_TOKENS} tokens each", answered=0, total=len(probs))
    if DRY_RUN:
        out.write_text("".join(chat_text("user: " + codebench.build_prompt(p) + "\nassistant: <think>\n",
                                         "Let me think.", "```\npass\n```", "\n") for p in probs))
        emit(stage, "done", 100, f"{len(probs)} conversations")
        return out
    done, lock = 0, threading.Lock()

    def answered():
        nonlocal done
        with lock:
            done += 1
            emit(stage, msg=f"{done}/{len(probs)} answers", answered=done, total=len(probs))

    proc = ev.serve(job, str(model), "calib")
    try:
        end = end_of_turn()  # before the answers: a template that can't be read fails in seconds
        infos: list[dict] = []
        answers, tps = ev.generate(codebench.SYSTEM, [codebench.build_prompt(p) for p in probs], CALIB_MAX_TOKENS,
                                   cfg.code_eval_thinking_temperature, 0, True, answered, infos=infos,
                                   keep_reasoning=True)
        texts = [chat_text(ev.thinking_prompt([{"role": "system", "content": codebench.SYSTEM},
                                               {"role": "user", "content": codebench.build_prompt(p)}]),
                           info.get("reasoning", ""), a, end)
                 for p, a, info in zip(probs, answers, infos)]
    finally:
        ev.stop(proc)
    write_jsonl(job.path("work", "calib-answers.jsonl"),
                [{"suite": p["suite"], "id": p["id"], "answer": a, **i} for p, a, i in zip(probs, answers, infos)])
    out.write_text("".join(texts))
    emit(stage, "done", 100, f"{len(texts)} conversations, {sum(map(len, texts)) // 1000}k characters, {tps} tok/s")
    return out


def free_vram_gb() -> float:
    """Free GPU memory, 0 without a GPU (llama-imatrix then runs on the CPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    return int(out.split()[0]) / 1024


def gpu_layers(tensors: list[Tensor], free_gb: float, reserve_gb: float = 6.0) -> int:
    """How many BF16 layers fit on the GPU; llama.cpp runs the rest from RAM. 999
    when the whole model fits, so the output head is on the GPU too."""
    per: dict[int, float] = {}
    rest = 0.0
    for t in tensors:
        gb = t.n * (2 if t.dims > 1 else 4) / 1e9
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            per[int(m[1])] = per.get(int(m[1]), 0.0) + gb
        else:
            rest += gb
    if not per:
        return 0
    if sum(per.values()) + rest <= free_gb - reserve_gb:
        return 999
    layer = max(per.values())
    return max(0, min(len(per), int((free_gb - reserve_gb) / layer)))


def importance(job: Job, bf16: Path, ctx: int = IMATRIX_CTX) -> Path:
    """work/imatrix.gguf from work/calib-chat.txt, in chunks of ctx tokens, the output
    head included. Its inputs (the text's and model's sha256, ctx, chunks, llama.cpp's
    commit) go to work/imatrix.json; an imatrix made from anything else is deleted,
    with any partial save, and made again."""
    stage = "imatrix"
    out, meta_path = job.path("work", "imatrix.gguf"), job.path("work", "imatrix.json")
    calib = job.path("work", "calib-chat.txt")
    if DRY_RUN:
        if not out.exists():
            out.write_bytes(b"GGUF dry run")
        emit(stage, "done", 100, "dry run")
        return out
    if not calib.exists():
        raise FileNotFoundError(f"{calib} missing; run the calib step first")
    calib_meta = job.path("work", "calib.json")
    tokens = json.loads(calib_meta.read_text())["tokens"] if calib_meta.exists() else None
    if tokens is not None and tokens < 2 * ctx:
        raise ValueError(f"{calib} holds {tokens} tokens; llama-imatrix needs at least {2 * ctx}")
    # llama-imatrix reads every whole chunk when --chunks isn't given (the llama-server text has no count)
    chunks = tokens // ctx if tokens is not None else None
    emit(stage, pct=1, msg="hashing the text and the model")
    key = {"calib_sha256": sha256_file(calib), "model_sha256": sha256_file(bf16), "ctx": ctx, "chunks": chunks,
           "process_output": True, "llama_cpp": llama_cpp_sha(job)}
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if out.exists() and old.get("key") == key:
        emit(stage, "done", 100, f"{out} exists, from the same text, model and settings")
        return out
    tmp = out.with_name("tmp-" + out.name)  # imatrix wants a .gguf suffix; it saves partial results every 10 chunks
    stale = [p for p in (out, meta_path, tmp, *out.parent.glob(out.name + ".at_*"), *out.parent.glob(tmp.name + ".at_*"))
             if p.exists()]
    for p in stale:
        p.unlink()
    if stale:
        changed = [k for k in key if old.get("key", {}).get(k) != key[k]] if old else ["no record of its inputs"]
        emit(stage, pct=2, msg=f"deleted a stale imatrix ({', '.join(changed)})")
    tensors, _ = read_tensors(job, bf16)
    ngl = gpu_layers(tensors, free_vram_gb())
    where = "the whole model on the GPU" if ngl == 999 else f"{ngl} layers on the GPU, the rest in RAM"
    emit(stage, pct=5, msg=f"{where}; {chunks or 'all'} chunks of {ctx} tokens")
    cmd = [quantize.bin_path(job, "llama-imatrix"), "-m", str(bf16), "-f", str(calib), "-o", str(tmp),
           "-ngl", str(ngl), "-c", str(ctx), *(["--chunks", str(chunks)] if chunks else []),
           "--process-output", "--parse-special", "--no-ppl"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(f"[{stage}] {line}")
        if m := re.search(r"computing over (\d+) chunks", line):
            emit(stage, pct=10, msg=f"{m[1]} chunks of {ctx} tokens")
        elif m := re.search(r"seconds per pass - ETA\s+(.*\S)", line):
            emit(stage, pct=12, msg=f"ETA {m[1]}")
    if proc.wait() != 0:
        raise RuntimeError(f"llama-imatrix exited with {proc.returncode}")
    gguf = quantize.gguf_lib(job)
    r = gguf.GGUFReader(str(tmp))
    entries = {t.name.rsplit(".", 1)[0] for t in r.tensors if t.name.endswith(".in_sum2")}
    head = "output.weight" if any(t.name == "output.weight" for t in tensors) else "token_embd.weight"
    if head not in entries:
        raise RuntimeError(f"{tmp} has no entry for {head}; --process-output had no effect")
    written = {k: int(r.fields[f"imatrix.{k}"].parts[-1][0]) for k in ("chunk_count", "chunk_size")
               if f"imatrix.{k}" in r.fields}
    tmp.rename(out)
    meta_path.write_text(json.dumps({"key": key, "entries": len(entries), **written}, indent=2))
    emit(stage, "done", 100, (f"{written.get('chunk_count', '?')} chunks of {ctx}, "
                              + ("all on the GPU" if ngl == 999 else f"{ngl} GPU layers")))
    return out


def recipe_map(job: Job, recipe: str, tensors: list[Tensor], layers: int, bf16: Path, imatrix: Path
               ) -> tuple[dict[str, str], dict]:
    """A recipe's type map, and what the report keeps about how it was chosen."""
    if recipe in FIXED:
        return by_kind(tensors, FIXED[recipe]), {}
    target, static = MIXED.get(recipe, (FFN_BITS, "q8_0"))
    energy = None
    if not DRY_RUN:
        emit(STAGE, pct=5, msg="feed-forward sensitivity from the imatrix")
        energy = imatrix_energy(job, bf16, imatrix, layers)
    plan, _, _ = allocate_w(tensors, layers, energy, target, static)
    return ffn_map(tensors, layers, plan, static), {"layers": [asdict(l) for l in plan],
                                                    "sensitivity": bits.sensitivities(layers, None, energy)}


def quantize_recipe(job: Job, recipe: str, bf16: Path, imatrix: Path, tmap: dict[str, str] | None = None) -> dict:
    """recipe (or the given type map, recipe "map") into out/model.gguf, checked type by
    type, with work/allocation.json. A model already there is kept only when
    allocation.json's key (the flags, the map's hash, the imatrix's sha256, llama.cpp's
    commit, the BF16's size) matches."""
    out = job.path("out", "model.gguf")
    alloc = job.path("work", "allocation.json")
    # the marker stands only after a call that ends with a model matching its inputs,
    # so a failed or changed run never leaves an old model looking done
    job.path(".done", STAGE).unlink(missing_ok=True)
    if DRY_RUN:
        tensors, layers = synthetic_tensors(), 64
    else:
        tensors, layers = read_tensors(job, bf16)
    report: dict = {"recipe": recipe, "layers": None}
    if tmap is None:
        tmap, how = recipe_map(job, recipe, tensors, layers, bf16, imatrix)
        report.update(how)
    check_map(tensors, tmap)
    args, base = map_args(tmap)
    est = planned_bytes(tensors, tmap) + overhead_bytes(bf16, tensors)
    planned = ffn_summary(tensors, layers, tmap)
    key = {"args": args + [base], "map_sha256": map_sha(tmap),
           "imatrix_sha256": sha256_file(imatrix) if imatrix.exists() else "",
           "llama_cpp": llama_cpp_sha(job), "bf16": {"name": bf16.name, "bytes": bf16.stat().st_size if bf16.exists() else 0}}
    old = json.loads(alloc.read_text()) if alloc.exists() else {}
    if out.exists() and old.get("key") == key:
        job.mark_done(STAGE, {"size_gb": old["size_gb"], "ffn_bits": old["ffn_bits"]})
        emit(STAGE, "done", 100, f"{out} exists, made from the same map, imatrix and llama.cpp")
        return old
    if out.exists() or old:
        changed = [k for k in key if old.get("key", {}).get(k) != key[k]] if old.get("key") else ["no record of its inputs"]
        emit(STAGE, pct=2, msg=f"rebuilding a stale model ({', '.join(changed)})")
    alloc.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    report.update(key=key, base=base, est_gb=round(est / 1e9, 4), est_bytes=est, est_ffn_bits=planned["ffn_bits"])
    emit(STAGE, pct=20, msg=f"{recipe} plan: ~{est / 1e9:.2f} GB, feed-forward ~{planned['ffn_bits']} bits")
    emit(STAGE, pct=25, msg=f"llama-quantize {recipe}")
    if DRY_RUN:
        out.write_bytes(b"GGUF dry run")
        report.update(size_gb=report["est_gb"], ffn_bits=planned["ffn_bits"], bit_widths=planned["bit_widths"],
                      tensors=tmap)
    else:
        q = quantize_map(job, bf16, imatrix, out, tmap, tensors)
        report.update(size_gb=round(q["size_bytes"] / 1e9, 4), size_bytes=q["size_bytes"], tensors=q["tensors"],
                      **ffn_summary(tensors, layers, q["tensors"]))
    alloc.write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"size_gb": report["size_gb"], "ffn_bits": report["ffn_bits"]})
    emit(STAGE, "done", 100, f"{recipe}: {report['size_gb']} GB, feed-forward {report['ffn_bits']} bits")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gguf")
    g.add_argument("--job", required=True)
    g.add_argument("--hf", required=True)
    g.add_argument("--bf16", required=True)
    g.add_argument("--q8", help="also make this 8-bit copy, for calibration when BF16 doesn't fit the GPU")
    c = sub.add_parser("calib")
    c.add_argument("--job", required=True)
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="llama-server answers from this GGUF (experiment 05)")
    src.add_argument("--traces", help="the training split of this traces.jsonl")
    c.add_argument("--tokenizer", help="with --traces: the HF tokenizer that wrote the trace ids")
    c.add_argument("--tokens", type=int, default=CALIB_TOKENS, help="with --traces: about this many tokens")
    i = sub.add_parser("imatrix")
    i.add_argument("--job", required=True)
    i.add_argument("--bf16", required=True)
    i.add_argument("--ctx", type=int, default=IMATRIX_CTX)
    q = sub.add_parser("quantize")
    q.add_argument("--job", required=True)
    which = q.add_mutually_exclusive_group(required=True)
    which.add_argument("--recipe", choices=RECIPES)
    which.add_argument("--map", help="a JSON type map {GGUF tensor name: type} naming every quantized tensor")
    q.add_argument("--bf16", required=True)
    q.add_argument("--imatrix", required=True)
    a = ap.parse_args()
    job = Job(a.job)
    if a.cmd == "gguf":
        make_ggufs(job, Path(a.hf), Path(a.bf16), Path(a.q8) if a.q8 else None)
    elif a.cmd == "calib" and a.traces:
        if not a.tokenizer:
            ap.error("calib --traces needs --tokenizer")
        trace_calibration(job, Path(a.traces), Path(a.tokenizer), a.tokens)
    elif a.cmd == "calib":
        calibration(job, Path(a.model))
    elif a.cmd == "imatrix":
        importance(job, Path(a.bf16), a.ctx)
    elif a.map:
        quantize_recipe(job, "map", Path(a.bf16), Path(a.imatrix), json.loads(Path(a.map).read_text()))
    else:
        quantize_recipe(job, a.recipe, Path(a.bf16), Path(a.imatrix))


if __name__ == "__main__":
    main()
