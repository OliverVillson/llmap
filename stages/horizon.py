"""Measured bit allocation: which llama.cpp type each tensor group gets, by KL per byte.

The old recipes gave each tensor kind a fixed type by rule of thumb, and no cheaper
mix was ever measured. Here every group's choice is measured on held-out thinking and
then bought by an exact knapsack:

1. PTQ GGUFs. llama-quantize writes the model once per type (Q2_K, Q3_K, Q4_K: what
   stages/kquant.py trains and packs exactly), every tensor named by an anchored
   exact-name --tensor-type flag under --pure (flags()), so no mixture rule moves a
   tensor to another type. Each file is read back and every tensor's type checked, and
   a file is only reused when made from the same BF16 GGUF, imatrix and binary.
2. Byte table. Each tensor's bytes at each type from ggml's block sizes
   (gguf.constants.GGML_QUANT_SIZES), padded to the GGUF alignment, plus the file's
   fixed part (header, metadata, tokenizer, norms) measured from the PTQ files. A map's
   file size is then known to the byte before it is written; the PTQ files must agree
   with the table exactly or the step fails. (bits.REL_MSE is a rule of thumb from
   another model and plays no part.) When even the floor map (step 5) is over the
   budget, e.g. at an untrimmed vocab, the run stops here: floor.json, exit 3.
3. Dequantization cache. gguf-py dequantizes every tensor of every PTQ file once, in
   a process pool, to .npy files in HF row order (q/k un-permuted) and the student's
   dtype, so a swap is a memory-mapped copy, not a GGUF read.
4. Teacher log-probs. The teacher runs once over held-out trace windows (traces.split's
   held-out problems, not the imatrix text or any training window) and its log-probs
   at every weighted position are cached, sliced to the kept vocab and renormalised as
   sft.kd_kl(index=...) does. The held-out problems are halved by traces.half (a
   sha256, the halving vocab uses too): one half chooses (the sweep), the other checks
   (verify), so the check is not the text the choice was fitted on.
5. Sweep. From the floor map (every group at its cheapest allowed type), each group in
   turn is swapped to each alternative type and the KL measured: layer x {gate+up,
   down, q+o, k+v} x alternatives, plus the embedding and the head. gain = KL(floor) -
   KL(swap), with a per-position standard error. Rows are appended as they finish, so
   a restart resumes.
6. Knapsack. Multiple-choice knapsack on the exact byte table: one type per group,
   most total gain within the byte budget. Dynamic programming over byte costs in units
   of their greatest common divisor, exact unless the budget needs more than
   MAX_STATES units (then costs round up: never over budget, maybe slightly short of
   optimal; map.json says so). The same table gives the predicted bytes -> KL curve
   from the floor to the budget.
7. Verify. The chosen map is quantized by llama-quantize; every tensor's type and the
   file's size (exactly the table's) are checked, and every tensor's bytes must equal
   the PTQ file's of that type, so the sweep measured exactly the shipped weights. Its
   true KL is measured on the sweep windows and on the check windows. On the sweep
   windows the gain it bought, KL(floor) - KL(map), paired by position with the floor's
   own KL there, is held against the knapsack's prediction (the swaps' gains added up):
   a map that does not beat the floor by GAIN_SE standard errors bought nothing for
   its bytes, so the floor map is shipped instead (map.json and horizon.json say why);
   one that gets under GAIN_SHORT of its predicted gain is shipped with a warning, the
   swaps' gains did not add up.
8. Ladder. llama-perplexity --kl-divergence of every PTQ file and the map against the
   BF16 GGUF on the check text, written to ladder.json (skipped when the binary is
   missing).

Memory on an L40S (48 GB) for a 3.66B model (3.40B after the vocab trim to 49,152):
the teacher (7.3 GB bf16) runs alone and is freed once its log-probs are cached. The
student (6.8 GB) is the only model on the GPU during the sweep, with the sweep's
log-probs resident (32,768 positions x 49,152 kept ids x fp16 = 3.2 GB) and one 4096-
token window in flight with no gradients (0.4 GB of bf16 logits, fp32 KL in chunks of
2048 positions: ~1.2 GB). The dequantized alternatives (3 types x 3.40B weights x bf16
= 20 GB) stay on the NVMe, memory-mapped; a swap copies one group in (at most 2 x 21M
weights, 84 MB) and the floor back. The check windows' log-probs (6.4 GB) stream from
the NVMe batch by batch. Peak ~12 GB (the teacher phase ~10 GB); ~18 GB without the
trim (100,352 ids). peak_gb() is this arithmetic.

    python -m stages.horizon run --hf <trimmed hf> --bf16 <its BF16 gguf> --traces traces.jsonl \\
        --llama-cpp <dir> --imatrix imatrix.gguf --budget-gb 1.172 --out W/horizon \\
        [--teacher <original hf> --vocab-index vocab_index.json]
    python -m stages.horizon knapsack --out W/horizon --budget-gb 1.2     # re-plan, CPU only
    python -m stages.horizon ladder --bf16 <BF16 gguf> --gguf a.gguf b.gguf --text held.txt \\
        --llama-cpp <dir> --out ladder.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.progress import emit
from stages import qheal, traces
from stages.sft import KD_CHUNK, load_causal_lm, log

STAGE = "horizon"
TYPES = ("q2_k", "q3_k", "q4_k")  # what stages/kquant.py trains and packs bit-exactly
# The head is never cut below Q4_K: it is read for every token's distribution, and
# llama.cpp's own 2-bit presets keep it higher still. The sweep still measures the cut.
HEAD_MIN = "q4_k"
# A layer's tensors are swapped in these groups, each group at one type.
KINDS = {"gate_up": ("ffn_gate", "ffn_up"), "down": ("ffn_down",), "q_o": ("attn_q", "attn_output"),
         "k_v": ("attn_k", "attn_v")}
SWEEP_LENGTH, SWEEP_TOKENS = 4096, 8 * 4096  # held-out window length, scored positions
CHECK_LENGTH, CHECK_TOKENS = 8192, 8 * 8192
PROMPT_WEIGHT = 0.1  # as in training (traces.batches)
LOGP_MIN = -1e4  # cached log-probs are clamped here, so a -inf never meets fp16
MAX_STATES = 200_000  # knapsack capacity in cost units
LADDER_CTX, LADDER_CHUNKS = 4096, 16
# The verified map against its prediction (gain_check): it must beat the floor by
# GAIN_SE standard errors of the paired difference, or the floor ships; a measured gain
# under GAIN_SHORT of the predicted one is a warning. Both are relative to the gain the
# bytes above the floor were to buy: the floor's KL itself is far larger than that gain
# at ~2.6 bits, so a tolerance on the KL would never fire.
GAIN_SE = 2.0
GAIN_SHORT = 0.5
KLD_TRIES = 3  # plain llama-perplexity runs per ladder rung when its summary is lost (see _kld_run)


# ---------- the model's tensors and groups ----------

def quantizable(name: str, n_dims: int) -> bool:
    """llama-quantize's rule (tensor_allows_quantization) as it applies to these models:
    2-D weights that are not norms or a router."""
    return n_dims >= 2 and name.endswith("weight") and "_norm.weight" not in name and "ffn_gate_inp" not in name


def gguf_info(path: Path, gguf) -> tuple[dict[str, tuple[int, ...]], int]:
    """({name: (rows, cols)} of the quantizable tensors in file order, the data alignment)."""
    r = gguf.GGUFReader(str(path))
    f = r.fields.get("general.alignment")
    align = int(f.parts[-1][0]) if f else 32
    shapes = {t.name: tuple(int(x) for x in reversed(t.shape.tolist())) for t in r.tensors
              if quantizable(t.name, len(t.shape))}
    return shapes, align


def _group_key(g: str) -> tuple:
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)", g)
    return (1, int(m[1]), list(KINDS).index(m[2])) if m else (0 if g == "token_embd" else 2, 0, 0)


def groups(names) -> dict[str, list[str]]:
    """GGUF tensor names by swap group: token_embd, blk.N.{gate_up,down,q_o,k_v}, output.
    Raises on a quantizable tensor no group holds: its type would be llama-quantize's
    choice, not ours."""
    out: dict[str, list[str]] = {}
    for n in names:
        if n in ("token_embd.weight", "output.weight"):
            out[n.split(".")[0]] = [n]
            continue
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", n)
        kind = next((k for k, parts in KINDS.items() if m and m[2] in parts), None)
        if kind is None:
            raise ValueError(f"no swap group for {n}; horizon knows {sorted(p for v in KINDS.values() for p in v)}")
        out.setdefault(f"blk.{m[1]}.{kind}", []).append(n)
    return {g: out[g] for g in sorted(out, key=_group_key)}


# ---------- exact bytes ----------

def _qtype(t: str):
    from gguf.constants import GGMLQuantizationType
    return GGMLQuantizationType[t.upper()]


def bits(t: str) -> float:
    from gguf.constants import GGML_QUANT_SIZES
    block, size = GGML_QUANT_SIZES[_qtype(t)]
    return 8 * size / block


def by_bits(types) -> list[str]:
    return sorted(set(types), key=bits)


def byte_table(shapes: dict[str, tuple[int, ...]], types, alignment: int = 32) -> dict[str, dict[str, int]]:
    """{tensor: {type: bytes it takes in the GGUF}}: rows x blocks per row x the type's
    block bytes (ggml's GGML_QUANT_SIZES), padded to the file's data alignment as every
    GGUF writer pads each tensor. A type whose block does not divide a row is left out
    for that tensor, because llama-quantize would silently write another type."""
    from gguf.constants import GGML_QUANT_SIZES
    out = {}
    for name, shape in shapes.items():
        cols, rows = shape[-1], math.prod(shape[:-1])
        row = {}
        for t in types:
            block, size = GGML_QUANT_SIZES[_qtype(t)]
            if cols % block == 0:
                row[t] = -(-(rows * (cols // block) * size) // alignment) * alignment
        out[name] = row
    return out


def group_bytes(table: dict[str, dict[str, int]], grp: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """{group: {type: bytes}} for the types every member tensor can take."""
    out = {}
    for g, names in grp.items():
        ok = set.intersection(*(set(table[n]) for n in names))
        out[g] = {t: sum(table[n][t] for n in names) for t in by_bits(ok)}
    return out


def fixed_bytes(ptq: dict[str, Path], table: dict[str, dict[str, int]]) -> int:
    """The file bytes outside the quantizable tensors (header, metadata, tokenizer, norms),
    from each PTQ file's size minus its tensors' table bytes. They must all agree, or the
    table is not exact."""
    fixed = {t: Path(p).stat().st_size - sum(row[t] for row in table.values()) for t, p in ptq.items()}
    if len(set(fixed.values())) != 1:
        raise ValueError(f"the byte table does not match llama-quantize's files: fixed bytes by type {fixed}")
    return next(iter(fixed.values()))


def predict_bytes(types: dict[str, str], table: dict[str, dict[str, int]], fixed: int) -> int:
    return fixed + sum(table[n][t] for n, t in types.items())


class OverBudget(ValueError):
    """Even the cheapest map is over the byte budget (the CLI exits 3)."""


def floor_bytes(gbytes: dict[str, dict[str, int]], fixed: int, head_min: str = HEAD_MIN) -> int:
    """The smallest file any map makes: every group at its cheapest type (the head at
    head_min or above), plus the fixed bytes. Known from the PTQ files, before the sweep."""
    return fixed + sum(min(b for t, b in by_t.items() if g != "output" or not head_min or bits(t) >= bits(head_min))
                       for g, by_t in gbytes.items())


# ---------- llama-quantize ----------

def flags(types: dict[str, str]) -> list[str]:
    """llama-quantize flags for exactly this map. --tensor-type patterns are unanchored
    regexes (regex_search) and the first match wins, so each tensor gets its own anchored
    pattern with its dots escaped (an unanchored blk.1 also takes blk.10, output.weight
    every attn_output.weight). --pure gives any tensor no pattern names the base type,
    not the file type's mixture rule (Q2_K bumps attn_v and ffn_down). The head and
    embedding flags, which llama-quantize applies before any pattern, say the same as
    their patterns."""
    out = ["--pure"]
    if "output.weight" in types:
        out += ["--output-tensor-type", types["output.weight"]]
    if "token_embd.weight" in types:
        out += ["--token-embedding-type", types["token_embd.weight"]]
    for name, t in types.items():
        out += ["--tensor-type", f"^{re.escape(name)}$={t}"]
    return out


def tool(llama_cpp: str | Path, name: str) -> Path:
    return Path(llama_cpp) / "build" / "bin" / name


def quant_key(bf16: Path, types: dict[str, str], llama_cpp: str | Path, imatrix: Path | None) -> dict:
    """What a quantized file was made from: the BF16 GGUF, the imatrix, the llama-quantize
    binary (a rebuild at another commit changes it) and the flags."""
    return {"bf16": _stamp(bf16), "imatrix": _stamp(imatrix) if imatrix else None,
            "llama_quantize": _stamp(tool(llama_cpp, "llama-quantize")),
            "flags": hashlib.sha256(json.dumps(flags(types)).encode()).hexdigest()}


def reusable(out: Path, key: dict, types: dict[str, str], gguf, expect_bytes: int | None = None) -> bool:
    """A file on disk is used again only when it was made from the same inputs (its
    .key.json) and still reads back as planned."""
    k = out.with_suffix(".key.json")
    if not (out.exists() and k.exists()) or json.loads(k.read_text()) != key:
        return False
    try:
        check(out, types, gguf, expect_bytes)
        return True
    except (ValueError, OSError) as e:
        log(f"horizon: {out.name} is redone ({e})")
        return False


def quantize(bf16: Path, out: Path, types: dict[str, str], llama_cpp: str | Path, *,
             imatrix: Path | None = None, threads: int = 0) -> Path:
    """llama-quantize bf16 -> out with every tensor of `types` at its type (written via a
    temporary file; its log and its quant_key next to out)."""
    exe = tool(llama_cpp, "llama-quantize")
    if not exe.exists():
        raise FileNotFoundError(f"{exe} missing; build llama.cpp (scripts/setup_vm.sh)")
    out.with_suffix(".key.json").unlink(missing_ok=True)
    base = max(set(types.values()), key=list(types.values()).count).upper()
    tmp = out.with_name("tmp-" + out.name)
    cmd = [str(exe), *(["--imatrix", str(imatrix)] if imatrix else []), *flags(types), str(bf16), str(tmp), base,
           *([str(threads)] if threads else [])]
    out.parent.mkdir(parents=True, exist_ok=True)
    logf = out.with_suffix(".log")
    with open(logf, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = logf.read_text()[-3000:]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"llama-quantize exited with {rc} for {out.name}:\n{tail}")
    tmp.replace(out)
    out.with_suffix(".key.json").write_text(json.dumps(quant_key(bf16, types, llama_cpp, imatrix)))
    return out


def check(path: Path, types: dict[str, str], gguf, expect_bytes: int | None = None) -> dict:
    """Reads a written GGUF back: every quantizable tensor is in `types` at its type, and
    the file is expect_bytes long. Raises listing every difference."""
    r = gguf.GGUFReader(str(path))
    got = {t.name: t.tensor_type.name.lower() for t in r.tensors if quantizable(t.name, len(t.shape))}
    bad = {n: (types.get(n), got.get(n)) for n in set(got) | set(types) if types.get(n) != got.get(n)}
    if bad:
        shown = ", ".join(f"{n}: planned {a}, written {b}" for n, (a, b) in sorted(bad.items())[:8])
        raise ValueError(f"{path.name}: {len(bad)} tensors not as planned: {shown}")
    size = Path(path).stat().st_size
    if expect_bytes is not None and size != expect_bytes:
        raise ValueError(f"{path.name} is {size} bytes, the byte table says {expect_bytes}")
    counts: dict[str, int] = {}
    for t in got.values():
        counts[t] = counts.get(t, 0) + 1
    return {"bytes": size, "gb": round(size / 1e9, 4), "types": counts}


def ptq(bf16: Path, out_dir: Path, types, names, llama_cpp, gguf, *, imatrix: Path | None = None,
        table: dict[str, dict[str, int]] | None = None) -> dict[str, Path]:
    """One GGUF per type with every quantizable tensor at it (reusable() ones are kept)."""
    out = {}
    for t in types:
        if table is not None and (miss := [n for n in names if t not in table[n]]):
            raise ValueError(f"{t} does not fit {miss[0]} (its rows are not whole blocks)")
        p = out_dir / f"{t}.gguf"
        plan = {n: t for n in names}
        if reusable(p, quant_key(bf16, plan, llama_cpp, imatrix), plan, gguf):
            out[t] = p
            continue
        log(f"horizon: llama-quantize, every tensor at {t}")
        quantize(bf16, p, plan, llama_cpp, imatrix=imatrix)
        check(p, plan, gguf)
        out[t] = p
    return out


# ---------- the dequantization cache ----------

def bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), rounded to nearest even as torch's
    .to(torch.bfloat16) does."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _hf_tensor(hf_dir: str | Path, name: str) -> torch.Tensor:
    from safetensors import safe_open
    for f in sorted(Path(hf_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            if name in s.keys():
                return s.get_tensor(name)
    raise KeyError(f"{name} is in no safetensors file of {hf_dir}")


def qk_heads(bf16: Path, hf_dir: str | Path, gguf) -> dict[str, int]:
    """{gguf q/k tensor: heads to un-permute it with}, empty when the converter left q/k
    in HF order. Decided once from layer 0 of the BF16 GGUF against the HF checkpoint;
    raises when neither order matches (the GGUF is not of this checkpoint)."""
    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    tensors = {t.name: t for t in gguf.GGUFReader(str(bf16)).tensors}
    w = qheal.dequant(gguf, tensors["blk.0.attn_q.weight"])
    ref = _hf_tensor(hf_dir, "model.layers.0.self_attn.q_proj.weight").float().numpy()
    plain, unperm = qheal.rel_err(w, ref), qheal.rel_err(qheal.unpermute(w, nh), ref)
    if min(plain, unperm) > 1e-3:
        raise ValueError(f"{bf16.name} blk.0.attn_q is {min(plain, unperm):.0%} off {hf_dir}'s q_proj")
    if plain <= unperm:
        return {}
    return {n: (nh if ".attn_q." in n else nkv) for n in tensors if re.fullmatch(r"blk\.\d+\.attn_[qk]\.weight", n)}


def _stamp(p: str | Path) -> dict:
    st = Path(p).stat()
    return {"path": str(p), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def _cache_one(src: str, ref: str, name: str, dst: str, heads: int, dtype: str) -> tuple[str, str, float]:
    """In a pool worker: one tensor of one PTQ file, dequantized, checked against the BF16
    GGUF's, put in HF row order and saved."""
    gguf = qheal._W["gguf"]
    w = qheal.dequant(gguf, qheal.worker_tensor(src, name))
    e = qheal.rel_err(w, qheal.dequant(gguf, qheal.worker_tensor(ref, name)))
    if e > 0.5:
        raise ValueError(f"{src}: {name} is {e:.0%} off the BF16 GGUF's")
    if heads:
        w = qheal.unpermute(w, heads)
    tmp = dst[:-4] + ".tmp.npy"
    np.save(tmp, bf16_bits(w) if dtype == "bf16" else w)
    os.replace(tmp, dst)
    return Path(src).stem, name, e


class Cache:
    """The dequantized PTQ tensors on disk: get(type, gguf name) -> tensor in HF order."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "cache.json").read_text())
        self.dtype = self.info["meta"]["dtype"]
        self.key = self.info["meta"]["ggufs"]  # which PTQ files these are

    def get(self, qtype: str, name: str) -> torch.Tensor:
        a = np.array(np.load(self.root / qtype / f"{name}.npy", mmap_mode="r"))
        if self.dtype == "bf16":
            return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(a)


def build_cache(ptq_files: dict[str, Path], bf16: Path, names, out: Path, gguf, *, heads: dict[str, int],
                dtype: str = "bf16", workers: int | None = None) -> Cache:
    """Dequantizes every tensor of every PTQ file into out/<type>/<name>.npy (bf16 bits or
    float32), workers processes in parallel; reused while the files are unchanged."""
    meta = {"dtype": dtype, "heads": heads, "bf16": _stamp(bf16), "ggufs": {t: _stamp(p) for t, p in ptq_files.items()},
            "names": list(names)}
    info = out / "cache.json"
    if info.exists() and json.loads(info.read_text()).get("meta") == meta:
        return Cache(out)
    info.unlink(missing_ok=True)
    for t in ptq_files:
        (out / t).mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(bf16), n, str(out / t / f"{n}.npy"), heads.get(n, 0), dtype)
            for t, p in ptq_files.items() for n in names]
    n_weights = sum(int(t.n_elements) for t in gguf.GGUFReader(str(bf16)).tensors) * len(ptq_files)
    n = qheal.auto_workers(n_weights, workers)
    t0, errs = time.time(), {t: {} for t in ptq_files}
    if n > 1:
        with qheal.pool(gguf, n) as ex:
            for i, (t, name, e) in enumerate(qheal.ordered(ex, _cache_one, jobs, 4 * n)):
                errs[t][name] = round(e, 6)
                if i % 50 == 0:
                    emit(STAGE, msg=f"dequantization cache {i}/{len(jobs)}")
    else:
        qheal._W.update(gguf=gguf, readers={})
        try:
            for j in jobs:
                t, name, e = _cache_one(*j)
                errs[t][name] = round(e, 6)
        finally:
            qheal._W["readers"] = {}
    tmp = info.with_name("cache.json.tmp")
    tmp.write_text(json.dumps({"meta": meta, "rel_err": errs}, indent=1))
    os.replace(tmp, info)
    log(f"horizon: {len(jobs)} tensors dequantized with {max(n, 1)} processes in {time.time() - t0:.0f}s")
    return Cache(out)


# ---------- held-out windows and the teacher's log-probs ----------

@dataclass
class Held:
    """Held-out batches as the student sees them (ids, mask, weights; traces.batches) and
    the teacher's log-probs at each batch's weighted positions (fp16, kept vocab)."""
    batches: list
    logp: list
    meta: dict

    @property
    def positions(self) -> int:
        return sum(int(lp.shape[0]) for lp in self.logp)

    def to(self, device) -> "Held":
        return Held([tuple(x.to(device) for x in b) for b in self.batches], [lp.to(device) for lp in self.logp],
                    self.meta)


def half(pid: str) -> int:
    """0 or 1 for a problem id, stable across processes: traces.half, the rule vocab's
    selection and gate halves use too. Not a second crc32: crc32 is linear, so among the
    ids whose crc32 put them in the held-out share, a crc32 of a prefixed id clusters
    (5 of 6 held-out ids landed on one side in a test)."""
    return traces.half(pid)


def held_rows(traces_path: str | Path, role: str) -> list[dict]:
    """The held-out trace rows (traces.split) of one half: "sweep" chooses, "check" checks,
    halved by half(problem id); "all" is both. Both halves get every held-out row when one
    would be empty."""
    if role not in ("sweep", "check", "all"):
        raise ValueError(f"role {role!r}")
    _, held = traces.split(traces.load(traces_path))
    if role == "all":
        return held
    side = [r for r in held if half(r["id"]) == (0 if role == "sweep" else 1)]
    if not side or len(side) == len(held):
        log(f"horizon: too few held-out problems to halve; {role} uses all {len(held)} held-out rows")
        return held
    return side


def _stamp_dir(d: str | Path) -> list:
    return [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in sorted(Path(d).glob("*.safetensors"))]


def held_out(traces_path: str | Path, teacher_dir: str, *, length: int, tokens: int, batch: int = 1,
             vocab_index: str = "", role: str = "sweep", path: Path | None = None) -> Held:
    """The teacher's log-probs on `tokens` weighted positions of held-out windows of
    `length`, cached at `path` and reused while the inputs are unchanged. With
    vocab_index (stages/vocab.py's vocab_index.json) the teacher is the untrimmed model:
    it reads the traces' original ids, its logits are sliced to the kept ids and
    renormalised (sft.kd_kl's index), and the student's batches carry the new ids with
    weight 0 where a target was dropped. The teacher is freed before returning."""
    meta = {"teacher": str(teacher_dir), "weights": _stamp_dir(teacher_dir), "traces": _stamp(traces_path),
            "vocab_index": _stamp(vocab_index) if vocab_index else None, "length": length, "tokens": tokens,
            "batch": batch, "role": role, "prompt_weight": PROMPT_WEIGHT}
    if path is not None and Path(path).exists():
        d = torch.load(path, mmap=True, weights_only=True)
        if d["meta"] == meta:
            return Held(d["batches"], d["logp"], meta)
    rows = held_rows(traces_path, role)
    vmap = json.loads(Path(vocab_index).read_text()) if vocab_index else None
    index = torch.tensor(vmap["kept"], dtype=torch.long) if vmap else None
    teacher = load_causal_lm(teacher_dir).eval()
    dev = next(teacher.parameters()).device
    if index is not None and int(index.max()) >= teacher.config.vocab_size:
        raise ValueError(f"vocab_index keeps id {int(index.max())} but {teacher_dir} has {teacher.config.vocab_size}"
                         " ids; the teacher must be the untrimmed model")
    t_it = traces.batches(rows, length, batch, prompt_weight=PROMPT_WEIGHT)
    s_it = traces.batches(rows, length, batch, vocab_map=vmap, prompt_weight=PROMPT_WEIGHT) if vmap else None
    out_b, out_lp, n = [], [], 0
    with torch.no_grad():
        for tb in t_it:
            sb = next(s_it) if s_it is not None else tb
            keep = sb[2] > 0
            if not keep.any():
                continue
            logits = teacher(input_ids=tb[0].to(dev), attention_mask=tb[1].to(dev), use_cache=False).logits[:, :-1]
            lt = logits[keep.to(dev)]
            del logits
            if index is not None:
                lt = lt.index_select(-1, index.to(dev))
            out_lp.append(torch.cat([F.log_softmax(lt[i:i + KD_CHUNK].float(), -1).clamp_min(LOGP_MIN).half().cpu()
                                     for i in range(0, len(lt), KD_CHUNK)]))
            out_b.append(sb)
            n += int(keep.sum())
            if n >= tokens:
                break
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not out_b:
        raise ValueError(f"no held-out window with a scored position in {traces_path}")
    if n < tokens:
        log(f"horizon: only {n} held-out positions for the {role} windows ({tokens} asked)")
    if path is None:
        return Held(out_b, out_lp, meta)
    tmp = Path(str(path) + ".tmp")
    torch.save({"meta": meta, "batches": out_b, "logp": out_lp}, tmp)
    os.replace(tmp, path)
    del out_b, out_lp  # read back memory-mapped, so the host keeps no copy
    d = torch.load(path, mmap=True, weights_only=True)
    return Held(d["batches"], d["logp"], meta)


def kl_positions(model, held: Held, chunk: int = KD_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """(KL(teacher || model) at every weighted held-out position, its weight), on the CPU.
    The same quantity sft.kd_kl averages: the cached log-probs are renormalised by a
    log_softmax as kd_kl does to teacher logits."""
    dev = next(model.parameters()).device
    kls, ws = [], []
    with torch.no_grad():
        for (ids, mask, w), lp in zip(held.batches, held.logp):
            ids, mask, w = ids.to(dev), mask.to(dev), w.to(dev)
            s = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
            keep = w > 0
            s = s[keep]
            if s.shape[-1] != lp.shape[-1]:
                raise ValueError(f"student vocab {s.shape[-1]} != the teacher cache's {lp.shape[-1]} ids")
            for i in range(0, len(s), chunk):
                ls = F.log_softmax(s[i:i + chunk].float(), -1)
                lt = F.log_softmax(lp[i:i + chunk].to(dev).float(), -1)
                kls.append((lt.exp() * (lt - ls)).sum(-1).cpu())
            ws.append(w[keep].float().cpu())
    return torch.cat(kls), torch.cat(ws)


def mean_kl(kl: torch.Tensor, w: torch.Tensor) -> float:
    return float((kl * w).sum() / w.sum().clamp_min(1e-12))


def weighted_se(d: torch.Tensor, w: torch.Tensor) -> float:
    """Standard error of the weighted mean of d, taking positions as independent (they
    are not inside a window, so this is a lower bound)."""
    m = (d * w).sum() / w.sum()
    var = (w * (d - m) ** 2).sum() / w.sum()
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return float((var / n_eff).sqrt())


# ---------- the sweep ----------

def set_types(model, cache: Cache, types: dict[str, str], params: dict | None = None) -> None:
    """Puts the cached dequantized weights of each GGUF tensor at its type into the model."""
    params = params if params is not None else dict(model.named_parameters())
    with torch.no_grad():
        for name, t in types.items():
            p = params[qheal.hf_name(name)]
            p.copy_(cache.get(t, name).to(p.device, p.dtype))


def plan_choices(grp: dict[str, list[str]], types, head_min: str = HEAD_MIN) -> tuple[dict, dict]:
    """({group: types to measure}, {group: floor type}): every group measures every type;
    the floor is the cheapest type, or head_min for the head."""
    types = by_bits(types)
    choices = {g: list(types) for g in grp}
    base = {g: types[0] for g in grp}
    if "output" in grp and head_min:
        choices["output"] = by_bits(set(types) | {head_min})
        base["output"] = head_min
    return choices, base


def _sweep_rows(path: Path | None, header: dict) -> list[dict]:
    """The rows already measured under the same header (base map, windows, choices)."""
    if path is None or not path.exists():
        return []
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not lines or lines[0].get("header") != header:
        path.rename(path.with_name(path.name + ".old"))
        return []
    return [r for r in lines[1:] if "group" in r]


def sweep(model, cache: Cache, held: Held, grp: dict[str, list[str]], choices: dict[str, list[str]],
          base: dict[str, str], out: Path | None = None) -> dict:
    """KL of the floor map and of every single-group swap away from it, on the held-out
    windows. Appends each row to `out` (JSON lines after a header) and resumes from it.
    Returns {base_kl, positions, rows: [{group, type, kl, gain, se}], base_kl_positions:
    the floor's KL at each position (a tensor, for verify's paired gain; not saved)}."""
    params = dict(model.named_parameters())
    set_types(model, cache, {n: base[g] for g, names in grp.items() for n in names}, params)
    header = {"base": base, "choices": choices, "held": held.meta, "weights": getattr(cache, "key", None)}
    rows = _sweep_rows(out, header)
    done = {(r["group"], r["type"]) for r in rows}
    kl0, w = kl_positions(model, held)
    base_kl = mean_kl(kl0, w)
    if out is not None and not out.exists():
        out.write_text(json.dumps({"header": header, "base_kl": base_kl, "positions": len(w)}) + "\n")
    todo = [(g, t) for g in grp for t in choices[g] if t != base[g]]
    log(f"horizon: floor KL {base_kl:.5f} on {len(w)} positions; {len(todo) - len(done)} of {len(todo)} swaps to go")
    t0 = time.time()
    for i, (g, t) in enumerate(todo):
        if (g, t) in done:
            continue
        set_types(model, cache, {n: t for n in grp[g]}, params)
        k, _ = kl_positions(model, held)
        set_types(model, cache, {n: base[g] for n in grp[g]}, params)
        d = kl0 - k
        row = {"group": g, "type": t, "kl": round(mean_kl(k, w), 7), "gain": round(mean_kl(d, w), 7),
               "se": round(weighted_se(d, w), 7)}
        rows.append(row)
        if out is not None:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if i % 8 == 0 or i == len(todo) - 1:
            eta = (time.time() - t0) / (len(rows) - len(done)) * (len(todo) - i - 1) / 60
            emit(STAGE, pct=100 * (i + 1) / len(todo), msg=f"sweep {i + 1}/{len(todo)}: {g} at {t} gains "
                 f"{row['gain']:.5f}, ~{eta:.0f} min left")
    return {"base_kl": base_kl, "positions": len(w), "rows": rows, "base_kl_positions": kl0}


# ---------- the knapsack ----------

def knapsack(table: dict[str, dict[str, tuple[int, float]]], budget_bytes: int, *,
             max_states: int = MAX_STATES) -> dict[str, str]:
    """Multiple-choice knapsack: {group: {type: (bytes, gain)}} -> {group: type}, one type
    per group, the most total gain with total bytes <= budget_bytes, and the fewest bytes
    among equal gains. Costs above each group's cheapest type are counted in units of
    their greatest common divisor, which is exact; when the budget needs more than
    max_states units they round up instead (feasible, maybe slightly short of optimal).
    Raises when even the cheapest map is over the budget."""
    return _knapsack(table, budget_bytes, max_states)["map"]


def _knapsack(table, budget_bytes: int, max_states: int = MAX_STATES) -> dict:
    gs = list(table)
    opts = {g: sorted(table[g].items(), key=lambda kv: (kv[1][0], -kv[1][1])) for g in gs}
    floor = {g: opts[g][0] for g in gs}
    floor_bytes = sum(b for _, (b, _) in floor.values())
    cap = budget_bytes - floor_bytes
    if cap < 0:
        raise ValueError(f"the cheapest map is {floor_bytes} bytes, over the budget of {budget_bytes}")
    deltas = [b - floor[g][1][0] for g in gs for _, (b, _) in opts[g] if b > floor[g][1][0]]
    unit = reduce(math.gcd, deltas, 0) or 1
    exact = cap // unit <= max_states
    if not exact:
        unit = -(-cap // max_states)
    C = cap // unit
    dp = np.zeros(C + 1)
    pick = np.zeros((len(gs), C + 1), dtype=np.int16)
    costs = {}
    for gi, g in enumerate(gs):
        f_bytes, f_gain = floor[g][1]
        cs = [-(-(b - f_bytes) // unit) for _, (b, _) in opts[g]]
        costs[g] = cs
        best = np.full(C + 1, -np.inf)
        arg = np.zeros(C + 1, dtype=np.int16)
        for k, (c, (_, (_, gain))) in enumerate(zip(cs, opts[g])):
            if c > C:
                continue
            cand = np.full(C + 1, -np.inf)
            cand[c:] = dp[:C + 1 - c] + (gain - f_gain)
            better = cand > best
            best[better], arg[better] = cand[better], k
        dp, pick[gi] = best, arg
    # the fewest units that reach the best gain (dp is non-decreasing in capacity)
    c = int(np.flatnonzero(dp >= dp[C] - 1e-12)[0])
    out = {}
    for gi in range(len(gs) - 1, -1, -1):
        g = gs[gi]
        k = int(pick[gi][c])
        out[g] = opts[g][k][0]
        c -= costs[g][k]
    base_gain = sum(gain for _, (_, gain) in floor.values())
    curve = [{"bytes": floor_bytes + int(i * unit), "gain": float(dp[i] + base_gain)}
             for i in np.unique(np.linspace(0, C, min(C + 1, 41)).astype(int))]
    return {"map": {g: out[g] for g in gs}, "exact": exact, "unit": unit, "floor_bytes": floor_bytes,
            "curve": curve}


def knapsack_table(gbytes: dict[str, dict[str, int]], rows: list[dict], base: dict[str, str],
                   head_min: str = HEAD_MIN) -> dict[str, dict[str, tuple[int, float]]]:
    """{group: {type: (bytes, gain)}} from the byte table and the sweep: the floor type at
    gain 0, every measured swap at its gain, the head never below head_min."""
    gain = {(r["group"], r["type"]): r["gain"] for r in rows}
    out = {}
    for g, by_t in gbytes.items():
        out[g] = {t: (b, 0.0 if t == base[g] else gain[(g, t)]) for t, b in by_t.items()
                  if t == base[g] or (g, t) in gain}
        if g == "output" and head_min:
            out[g] = {t: v for t, v in out[g].items() if bits(t) >= bits(head_min)}
    return out


def plan(out: Path, budget_bytes: int, head_min: str = HEAD_MIN, *, force: dict[str, str] | None = None,
         note: dict | None = None) -> dict:
    """The knapsack on out/bytes.json and out/sweep.jsonl; writes out/map.json. force
    ({group: type}) is written instead of the knapsack's map, with note (why) under
    "fallback": the floor map, when the knapsack's bought nothing measurable."""
    info = json.loads((out / "bytes.json").read_text())
    lines = [json.loads(x) for x in (out / "sweep.jsonl").read_text().splitlines() if x.strip()]
    head, rows = lines[0], [r for r in lines[1:] if "group" in r]
    base = head["header"]["base"]
    want = {(g, t) for g, ts in head["header"]["choices"].items() for t in ts if t != base[g]}
    if missing := want - {(r["group"], r["type"]) for r in rows}:
        raise ValueError(f"the sweep is not finished: {len(missing)} of {len(want)} swaps not measured")
    grp = info["groups"]
    res = _knapsack(knapsack_table(info["group_bytes"], rows, base, head_min), budget_bytes - info["fixed"])
    by_group = dict(force) if force else res["map"]
    types = {n: by_group[g] for g, names in grp.items() for n in names}
    gain = {(r["group"], r["type"]): r["gain"] for r in rows}
    pred = head["base_kl"] - sum(0.0 if t == base[g] else gain[(g, t)] for g, t in by_group.items())
    size = predict_bytes(types, info["tensors"], info["fixed"])
    sha = hashlib.sha256(json.dumps(types, sort_keys=True).encode()).hexdigest()
    counts: dict[str, int] = {}
    for t in by_group.values():
        counts[t] = counts.get(t, 0) + 1
    m = {"budget_bytes": budget_bytes, "bytes": size, "gb": round(size / 1e9, 6),
         "floor_bytes": info["fixed"] + res["floor_bytes"], "base_kl": head["base_kl"], "pred_kl": pred,
         "exact": res["exact"], "unit_bytes": res["unit"], "sha256": sha, "group_types": counts,
         "groups": by_group, "types": types, "hf_types": {qheal.hf_name(n): t for n, t in types.items()},
         "flags": flags(types),
         "curve": [{"gb": round((info["fixed"] + p["bytes"]) / 1e9, 6), "pred_kl": head["base_kl"] - p["gain"]}
                   for p in res["curve"]]}
    if note:
        m["fallback"] = note
    tmp = out / "map.json.tmp"
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, out / "map.json")
    log(f"horizon: map {sha[:8]}: {size / 1e9:.4f} GB of a {budget_bytes / 1e9:.4f} GB budget "
        f"(floor {m['floor_bytes'] / 1e9:.4f}), predicted KL {pred:.5f} (floor {head['base_kl']:.5f}), {counts}")
    return m


# ---------- verification ----------

def identical(path: Path, ptq_files: dict[str, Path], types: dict[str, str], gguf) -> int:
    """Raises unless every tensor of `path` has exactly the bytes of the PTQ file of its
    type: then the swaps measured the very weights the map ships. Returns the count."""
    mine = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
    theirs = {t: {x.name: x for x in gguf.GGUFReader(str(p)).tensors} for t, p in ptq_files.items()}
    bad = [n for n, t in types.items() if not np.array_equal(np.asarray(mine[n].data), np.asarray(theirs[t][n].data))]
    if bad:
        raise ValueError(f"{len(bad)} tensors of {path.name} differ from the PTQ files', e.g. {bad[0]}: "
                         "llama-quantize did not reproduce them, so the sweep measured other weights")
    return len(types)


def verify(hf_dir: str, bf16: Path, types: dict[str, str], out_gguf: Path, llama_cpp, gguf, *,
           expect_bytes: int, held: dict[str, Held], imatrix: Path | None = None,
           ptq_files: dict[str, Path] | None = None, workers: int | None = None,
           base: dict[str, torch.Tensor] | None = None) -> dict:
    """Quantizes the map, checks every tensor's type and the exact size (and, given the
    PTQ files, that every tensor is byte-identical to the one the sweep used), then
    reads it into a fresh student (qheal.load_quantized: strict) and measures its true KL
    on each of `held` ({name: Held}). base ({name: per-position KL of the floor on that
    Held, sweep()'s base_kl_positions}) adds "paired": {name: {gain, se}}, the mean of
    KL(floor) - KL(map) over the same positions and its standard error."""
    if not reusable(out_gguf, quant_key(bf16, types, llama_cpp, imatrix), types, gguf, expect_bytes):
        quantize(bf16, out_gguf, types, llama_cpp, imatrix=imatrix)
    rep = check(out_gguf, types, gguf, expect_bytes)
    if ptq_files:
        rep["identical_tensors"] = identical(out_gguf, ptq_files, types, gguf)
    student = load_causal_lm(hf_dir).eval()
    rep["quant_rel_err"] = qheal.load_quantized(student, out_gguf, gguf, workers=workers)
    rep["kl"], paired = {}, {}
    for k, h in held.items():
        kl, w = kl_positions(student, h)
        rep["kl"][k] = round(mean_kl(kl, w), 7)
        if base is not None and k in base:
            if base[k].shape != kl.shape:
                raise ValueError(f"the floor's KL has {len(base[k])} positions on {k}, the map's {len(kl)}")
            d = base[k] - kl
            paired[k] = {"gain": round(mean_kl(d, w), 7), "se": round(weighted_se(d, w), 7)}
    if paired:
        rep["paired"] = paired
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # llama-perplexity runs next, on the same GPU
    return {"gguf": str(out_gguf), **rep}


def gain_check(floor_kl: float, pred_kl: float, true_kl: float, se: float, *, is_floor: bool = False,
               k: float = GAIN_SE, short: float = GAIN_SHORT) -> dict:
    """The verified map against the knapsack's prediction, on the sweep windows. pred =
    KL(floor) - the predicted KL (the swaps' gains added up), real = KL(floor) - the
    map's measured KL, se = the standard error of their paired per-position difference.
    verdict: "floor" when the map is the floor (nothing bought, nothing to check); "no
    better" when real <= k x se, so the bytes above the floor bought nothing measurable
    (run() ships the floor instead); "short" when real < short x pred, so the swaps did
    not add up (a warning); "ok" otherwise. The tolerance is on the gain, not on the KL:
    the floor's KL is many times the gain at ~2.6 bits."""
    pred, real = floor_kl - pred_kl, floor_kl - true_kl
    out = {"pred": round(pred, 7), "real": round(real, 7), "se": round(se, 7),
           "ratio": round(real / pred, 4) if pred > 0 else None,
           "rule": f"real > {k:g} x se, or the floor ships; real >= {short:g} x pred, or a warning"}
    if is_floor:
        verdict, why = "floor", "the map is the floor"
    elif real <= k * se:
        verdict, why = "no better", (f"the map's KL {true_kl:.5f} is within {k:g} SE ({k * se:.5f}) of the floor's "
                                     f"{floor_kl:.5f}: its bytes above the floor bought nothing measurable")
    elif real < short * pred:
        verdict, why = "short", (f"the map gains {real:.5f} of the {pred:.5f} predicted ({real / pred:.0%}, under "
                                 f"{short:.0%}): the swaps' gains did not add up")
    else:
        verdict, why = "ok", None
    return {**out, "verdict": verdict, "why": why}


# ---------- the llama-perplexity ladder ----------

_LADDER = {"kld": r"Mean\s+KLD:\s+([-\d.e+]+)\s+±\s+([\d.e+-]+)", "top1": r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)",
           "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s+([\d.e+-]+)", "kld_p99": r"99\.0%\s+KLD:\s+([\d.e+-]+)",
           "kld_median": r"Median\s+KLD:\s+([\d.e+-]+)"}


def parse_kld(text: str) -> dict:
    """The summary of a llama-perplexity --kl-divergence run (top1 as a share). Both lines
    the parity check reads must be there: the top-1 share is the summary's last line, so
    an output cut short can have the mean KLD without it, and that is a lost summary too.
    (llama-perplexity prints no summary at all when fewer than 100 positions were scored.)"""
    out = {}
    for k, pat in _LADDER.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m[1])
            if k == "kld":
                out["kld_se"] = float(m[2])
            if k == "top1":
                out[k], out["top1_se"] = float(m[1]) / 100, float(m[2]) / 100
    if "kld" not in out or "top1" not in out:
        raise ValueError(f"no KLD summary in llama-perplexity's output ({'top-1 line' if 'kld' in out else 'mean KLD'} "
                         f"missing):\n{text[-2000:]}")
    return out


# llama-perplexity prints everything, its KLD summary included, through llama.cpp's
# asynchronous log (common/log.cpp): a worker thread prints a queue of lines. The pinned
# tool returns from main without common_log_flush(), so the process can exit with the
# last lines still queued: one run in ten lost its summary on an idle 4-core machine,
# nearly every run with the cores busy. No flag or environment variable makes that log
# synchronous (--log-file is written by the same thread), and the KLD statistics go to
# no file. So the tool's own entry point, llama_perplexity() in the
# libllama-perplexity-impl.so it is built on, is called from a small Python process that
# then calls common_log_flush(common_log_main()), as log.h asks of a program that needs
# all of its log, and exits with the tool's return code. Python's SIGINT and SIGPIPE
# handling is put back to the defaults first, so the tool stops as the binary would.
_FLUSHED = """\
import ctypes, os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
lib = ctypes.CDLL(sys.argv[1])
main, log_main, flush = lib._Z16llama_perplexityiPPc, lib._Z15common_log_mainv, lib._Z16common_log_flushP10common_log
main.argtypes, main.restype = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int
log_main.restype, flush.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
argv = [os.fsencode(a) for a in sys.argv[2:]]
rc = main(len(argv), (ctypes.c_char_p * (len(argv) + 1))(*argv, None))
flush(log_main())
ctypes.CDLL(None).fflush(None)
os._exit(rc & 0xFF)
"""
PERPLEXITY_LIB = "libllama-perplexity-impl.so"


def perplexity_lib(exe: str | Path) -> Path | None:
    """The library llama-perplexity is built on, beside the binary, or None."""
    lib = Path(exe).parent / PERPLEXITY_LIB
    return lib if lib.exists() else None


def perplexity_cmd(cmd: list[str]) -> list[str]:
    """cmd (llama-perplexity and its arguments) as a command whose output is the tool's
    whole log (_FLUSHED): through the tool's library when it is built beside the binary,
    else the binary itself (_kld_run then repeats a run that lost its summary)."""
    lib = perplexity_lib(cmd[0])
    return [sys.executable, "-I", "-c", _FLUSHED, str(lib), *map(str, cmd)] if lib else [*map(str, cmd)]


def _kld_run(cmd: list[str], tries: int = KLD_TRIES) -> dict:
    """One llama-perplexity --kl-divergence run (cmd: the binary and its arguments),
    parsed, its log flushed before it exits (perplexity_cmd). Without the tool's library
    only the plain binary can run, whose log can lose its last lines: a run that exits 0
    without the summary is then repeated (up to `tries`)."""
    model = Path(cmd[cmd.index("-m") + 1]).name if "-m" in cmd else "?"
    run, flushed = perplexity_cmd(cmd), perplexity_lib(cmd[0]) is not None
    for i in range(1 if flushed else tries):
        r = subprocess.run(run, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llama-perplexity --kl-divergence on {model} exited with "
                               f"{r.returncode}:\n{(r.stdout + r.stderr)[-3000:]}")
        try:
            return parse_kld(r.stdout + r.stderr)
        except ValueError:
            if flushed or i == tries - 1:
                raise
            log(f"horizon: llama-perplexity's output on {model} lost its summary; running it again")
    raise AssertionError("unreachable")


def ladder_text(traces_path: str | Path, tokenizer_dir: str, path: Path, role: str = "check") -> Path:
    """The held-out rows of one half as text (prompt and answer, special tokens kept),
    for llama-perplexity, which tokenizes it with the GGUF's own vocab."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    rows = held_rows(traces_path, role)
    path.write_text("\n\n".join(tok.decode(r["prompt"] + r["tokens"], skip_special_tokens=False) for r in rows))
    return path


def ladder(bf16: Path, ggufs: list[Path], text: Path, out: Path, llama_cpp, *, ctx: int = LADDER_CTX,
           chunks: int = LADDER_CHUNKS, ngl: int | None = None) -> list[dict] | None:
    """llama-perplexity --kl-divergence of each GGUF against the BF16 GGUF's logits on
    `text` (ctx-token chunks, at most `chunks`), written to `out` as a JSON list, smallest
    file first: [{gguf, bytes, gb, kld, kld_se, top1, top1_se (shares), ppl_ratio,
    kld_p99, kld_median, ctx, chunks, text}]. None when llama-perplexity is not built."""
    exe = tool(llama_cpp, "llama-perplexity")
    if not exe.exists():
        log(f"horizon: {exe} missing, no llama.cpp KL ladder")
        return None
    ngl = (999 if torch.cuda.is_available() else 0) if ngl is None else ngl
    common = ["-c", str(ctx), "--chunks", str(chunks), "-ngl", str(ngl)]
    base = out.with_name(out.stem + "-base.kld")
    stamp = out.with_name(out.stem + "-base.json")
    want = {"bf16": _stamp(bf16), "text": _stamp(text), "ctx": ctx, "chunks": chunks}
    if not (base.exists() and stamp.exists() and json.loads(stamp.read_text()) == want):
        tmp = Path(str(base) + ".tmp")
        r = subprocess.run(perplexity_cmd([str(exe), "-m", str(bf16), "-f", str(text), "--kl-divergence-base",
                                           str(tmp), *common]), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llama-perplexity on {bf16.name} exited with {r.returncode}:\n"
                               f"{(r.stdout + r.stderr)[-3000:]}")
        os.replace(tmp, base)
        stamp.write_text(json.dumps(want))
    rows = []
    for g in ggufs:
        size = Path(g).stat().st_size
        rows.append({"gguf": str(g), "bytes": size, "gb": round(size / 1e9, 4),
                     **_kld_run([str(exe), "-m", str(g), "--kl-divergence-base", str(base), "--kl-divergence", *common]),
                     "ctx": ctx, "chunks": chunks, "text": str(text)})
        log(f"horizon: ladder {Path(g).name}: {size / 1e9:.3f} GB, KLD {rows[-1]['kld']:.5f}, "
            f"top-1 {rows[-1].get('top1', float('nan')):.3%}")
    rows.sort(key=lambda r: r["bytes"])
    out.write_text(json.dumps(rows, indent=1))
    return rows


# ---------- the whole step ----------

def peak_gb(student_params: float, teacher_params: float, vocab: int, teacher_vocab: int | None = None,
            positions: int = SWEEP_TOKENS, length: int = SWEEP_LENGTH, hidden: int = 2560, ffn: int = 8192) -> dict:
    """GPU memory of the two phases in GB (bf16 weights, no gradients). Teacher phase: the
    teacher, one window's logits over its whole vocab and their weighted rows, the fp32
    log_softmax chunk. Sweep: the student, the resident log-probs (fp16, kept vocab), one
    window's logits and their weighted rows, three fp32 KL temporaries per chunk, and the
    window's largest activations."""
    tv = teacher_vocab or vocab
    acts = 4 * length * max(hidden, ffn) * 2
    teacher = teacher_params * 2 + 2 * length * tv * 2 + KD_CHUNK * tv * 4 + acts
    student = student_params * 2 + positions * vocab * 2 + 2 * length * vocab * 2 + 3 * KD_CHUNK * vocab * 4 + acts
    return {"teacher_phase": round(teacher / 1e9, 1), "sweep_phase": round(student / 1e9, 1)}


def ensure_bf16(hf_dir: str, bf16: Path, llama_cpp) -> Path:
    if not bf16.exists():
        tmp = bf16.with_name("tmp-" + bf16.name)
        subprocess.run([sys.executable, str(Path(llama_cpp) / "convert_hf_to_gguf.py"), hf_dir, "--outtype", "bf16",
                        "--outfile", str(tmp)], check=True)
        tmp.replace(bf16)
    return bf16


def run(hf_dir: str, bf16: Path, traces_path: Path, out: Path, llama_cpp, budget_bytes: int, *,
        imatrix: Path | None = None, teacher: str = "", vocab_index: str = "", types=TYPES,
        head_min: str = HEAD_MIN, sweep_window: tuple[int, int] = (SWEEP_LENGTH, SWEEP_TOKENS),
        check_window: tuple[int, int] = (CHECK_LENGTH, CHECK_TOKENS), do_ladder: bool = True,
        ladder_window: tuple[int, int] = (LADDER_CTX, LADDER_CHUNKS), workers: int | None = None) -> dict:
    """Steps 1-8 of the module docstring; each one reuses its finished outputs in `out`.
    Writes out/horizon.json (the summary) and returns it."""
    gguf = qheal.gguf_module(str(llama_cpp))
    out.mkdir(parents=True, exist_ok=True)
    teacher = teacher or hf_dir
    ensure_bf16(hf_dir, bf16, llama_cpp)
    shapes, align = gguf_info(bf16, gguf)
    grp = groups(shapes)
    choices, base = plan_choices(grp, types, head_min)
    all_types = by_bits({t for c in choices.values() for t in c})
    table = byte_table(shapes, all_types, align)
    emit(STAGE, pct=0, msg=f"PTQ GGUFs at {', '.join(all_types)}")
    ptq_files = ptq(bf16, out / "ptq", all_types, list(shapes), llama_cpp, gguf, imatrix=imatrix, table=table)
    fixed = fixed_bytes(ptq_files, table)
    gbytes = group_bytes(table, grp)
    (out / "bytes.json").write_text(json.dumps({"alignment": align, "fixed": fixed, "groups": grp,
                                                "group_bytes": gbytes, "tensors": table}, indent=1))
    log(f"horizon: byte table exact against {len(ptq_files)} PTQ files; fixed part {fixed} bytes")
    floor = floor_bytes(gbytes, fixed, head_min)
    if floor > budget_bytes:  # e.g. an untrimmed vocab: refuse before the hour of cache, teacher and sweep
        (out / "floor.json").write_text(json.dumps({"floor_bytes": floor, "floor_gb": round(floor / 1e9, 6),
                                                    "budget_bytes": budget_bytes, "over": True}, indent=1))
        raise OverBudget(f"the cheapest map is {floor / 1e9:.4f} GB, over the {budget_bytes / 1e9:.4f} GB budget; "
                         "nothing to sweep")
    (out / "floor.json").unlink(missing_ok=True)
    emit(STAGE, pct=0, msg="dequantization cache")
    dtype = "bf16" if torch.cuda.is_available() else "f32"
    cache = build_cache(ptq_files, bf16, list(shapes), out / "cache", gguf, heads=qk_heads(bf16, hf_dir, gguf),
                        dtype=dtype, workers=workers)
    emit(STAGE, pct=0, msg="teacher log-probs")
    held = {role: held_out(traces_path, teacher, length=w[0], tokens=w[1], vocab_index=vocab_index, role=role,
                           path=out / f"teacher-{role}.pt")
            for role, w in (("sweep", sweep_window), ("check", check_window))}
    student = load_causal_lm(hf_dir).eval()
    dev = next(student.parameters()).device
    n_student = sum(p.numel() for p in student.parameters())
    vocab = held["sweep"].logp[0].shape[-1]
    log(f"horizon: GPU plan {peak_gb(n_student, n_student, vocab, positions=held['sweep'].positions)} GB")
    sw = sweep(student, cache, held["sweep"].to(dev), grp, choices, base, out / "sweep.jsonl")
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = plan(out, budget_bytes, head_min)

    def verified(m: dict) -> dict:
        emit(STAGE, pct=100, msg=f"verifying the {m['gb']:.4f} GB map")
        return verify(hf_dir, bf16, m["types"], out / f"horizon-{m['sha256'][:8]}.gguf", llama_cpp, gguf,
                      expect_bytes=m["bytes"], held=held, imatrix=imatrix, ptq_files=ptq_files, workers=workers,
                      base={"sweep": sw["base_kl_positions"]})

    v = verified(m)
    gain = gain_check(sw["base_kl"], m["pred_kl"], v["kl"]["sweep"], v["paired"]["sweep"]["se"],
                      is_floor=m["groups"] == base, k=GAIN_SE, short=GAIN_SHORT)
    warnings, fallback = [], None
    if gain["verdict"] == "no better":
        fallback = {"why": gain["why"], "rejected": {"sha256": m["sha256"], "gguf": v["gguf"], "bytes": v["bytes"],
                                                     "pred_kl": m["pred_kl"], "true_kl": v["kl"]["sweep"],
                                                     "check_kl": v["kl"]["check"], "group_types": m["group_types"]}}
        log(f"horizon: {gain['why']}; the floor map ships instead")
        m = plan(out, budget_bytes, head_min, force=base, note=fallback)
        v = verified(m)
    elif gain["verdict"] == "short":
        warnings.append(gain["why"])
        log(f"horizon: warning: {gain['why']}")
    lad = None
    if do_ladder:
        text = ladder_text(traces_path, teacher, out / "ladder.txt")
        lad = ladder(bf16, [*ptq_files.values(), Path(v["gguf"])], text, out / "ladder.json", llama_cpp,
                     ctx=ladder_window[0], chunks=ladder_window[1])
    summary = {"budget_bytes": budget_bytes, "gguf": v["gguf"], "bytes": v["bytes"], "gb": v["gb"],
               "floor_bytes": m["floor_bytes"], "floor_kl": sw["base_kl"], "pred_kl": m["pred_kl"],
               "true_kl": v["kl"]["sweep"], "check_kl": v["kl"]["check"], "gain": gain, "fallback": fallback,
               "warnings": warnings, "swaps": len(sw["rows"]),
               "positions": {k: h.positions for k, h in held.items()}, "group_types": m["group_types"],
               "map": str(out / "map.json"), "ladder": str(out / "ladder.json") if lad else None,
               "verify": v}
    (out / "horizon.json").write_text(json.dumps(summary, indent=1))
    emit(STAGE, "done", 100, f"{v['gb']:.4f} GB, KL {v['kl']['check']:.5f} on the check windows "
         f"(predicted {m['pred_kl']:.5f} on the sweep's, measured {v['kl']['sweep']:.5f}); gain {gain['verdict']}"
         + (", the floor ships" if fallback else ""))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="measured bit allocation (PTQ sweep + knapsack)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="PTQ files, cache, teacher, sweep, knapsack, verify, ladder")
    r.add_argument("--hf", required=True, help="the (trimmed) bf16 HF checkpoint being quantized")
    r.add_argument("--bf16", required=True, help="its BF16 GGUF (converted here when missing)")
    r.add_argument("--traces", required=True)
    r.add_argument("--llama-cpp", required=True)
    r.add_argument("--imatrix", default="")
    r.add_argument("--teacher", default="", help="default --hf; the untrimmed original with --vocab-index")
    r.add_argument("--vocab-index", default="")
    r.add_argument("--types", default=",".join(TYPES))
    r.add_argument("--head-min", default=HEAD_MIN, help="'' lets the knapsack cut the head too")
    r.add_argument("--sweep", default=f"{SWEEP_LENGTH}x{SWEEP_TOKENS}", help="window length x positions")
    r.add_argument("--check", default=f"{CHECK_LENGTH}x{CHECK_TOKENS}")
    r.add_argument("--ladder", default=f"{LADDER_CTX}x{LADDER_CHUNKS}", help="context x chunks; 'off' skips it")
    r.add_argument("--workers", type=int, default=None)
    k = sub.add_parser("knapsack", help="re-plan from a finished sweep (CPU only)")
    k.add_argument("--head-min", default=HEAD_MIN)
    for p in (r, k):
        p.add_argument("--out", required=True)
        p.add_argument("--budget-gb", type=float, required=True, help="the whole file, in 1e9 bytes")
    lad = sub.add_parser("ladder", help="llama-perplexity KLD of GGUFs against a BF16 GGUF")
    lad.add_argument("--bf16", required=True)
    lad.add_argument("--gguf", required=True, nargs="+")
    lad.add_argument("--text", required=True)
    lad.add_argument("--llama-cpp", required=True)
    lad.add_argument("--out", required=True)
    lad.add_argument("--ctx", type=int, default=LADDER_CTX)
    lad.add_argument("--chunks", type=int, default=LADDER_CHUNKS)
    a = ap.parse_args()
    if a.cmd == "knapsack":
        plan(Path(a.out), int(a.budget_gb * 1e9), a.head_min)
    elif a.cmd == "ladder":
        ladder(Path(a.bf16), [Path(g) for g in a.gguf], Path(a.text), Path(a.out), a.llama_cpp, ctx=a.ctx,
               chunks=a.chunks)
    else:
        def window(s: str) -> tuple[int, int]:
            x, y = s.lower().split("x")
            return int(x), int(y)
        try:
            run(a.hf, Path(a.bf16), Path(a.traces), Path(a.out), a.llama_cpp, int(a.budget_gb * 1e9),
                imatrix=Path(a.imatrix) if a.imatrix else None, teacher=a.teacher, vocab_index=a.vocab_index,
                types=tuple(a.types.split(",")), head_min=a.head_min, sweep_window=window(a.sweep),
                check_window=window(a.check), do_ladder=a.ladder != "off",
                ladder_window=window(a.ladder) if a.ladder != "off" else (LADDER_CTX, LADDER_CHUNKS),
                workers=a.workers)
        except OverBudget as e:
            log(f"horizon: {e} ({Path(a.out) / 'floor.json'})")
            sys.exit(3)


if __name__ == "__main__":
    main()
