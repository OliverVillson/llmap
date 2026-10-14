"""llama.cpp's Q2_K, Q3_K and Q4_K weight formats in torch: read from llama-quantize
output, written back byte for byte, and trained in, so that the weights the trainer's
forward uses are exactly the weights the GGUF stores.

A K-quant tensor is rows of super-blocks of 256 weights (QK_K). Each super-block has an
fp16 d (and dmin) and an integer scale (and min) per sub-block of 16 weights (Q2_K,
Q3_K) or 32 (Q4_K); each weight is an integer code:

    q2_k  w = (d * sc) * q - (dmin * m)    q 0..3    sc, m 0..15 (4 bits)    84 B per 256
    q3_k  w = (d * sc) * q                 q -4..3   sc -32..31 (6 bits)    110 B per 256
    q4_k  w = (d * sc) * q - (dmin * m)    q 0..15   sc, m 0..63 (6 bits)   144 B per 256

Byte layout of one block (ggml-common.h, little endian):

    q2_k  scales[16] (sc | m << 4), qs[64] (four 2-bit codes per byte), d, dmin (d at byte 80)
    q3_k  hmask[32] (bit 2 of q + 4), qs[64] (its bits 0-1), scales[12] (sc + 32, 6-bit packed), d
    q4_k  d, dmin, scales[12] (sc and m, 6-bit packed), qs[128] (two 4-bit codes per byte)

None of the three layouts has a spare bit, so from_gguf and pack are exact inverses on
every byte string. dequant follows ggml's order of operations (dequantize_row_q*_K, gguf-py's
quants.py): sub-block scale d * sc, then times q, minus dmin * m, all in fp32. Every
product there is exact (an fp16 d has 11 significant bits, sc at most 6, q at most 4:
at most 21 of fp32's 24), so the one rounding is the final subtraction. The fp32 value is
therefore the same whatever order the multiplies run in and whether a compiler fuses
the multiply and subtract: ggml's CPU code, its CUDA kernels (which nvcc contracts into
FMAs) and gguf-py's numpy all give the same fp32, and dequant here equals
gguf.quants.dequantize bit for bit. llama.cpp's CUDA dequantize into an fp16 buffer
rounds that fp32 once more, as the bf16 vLLM checkpoint (gguf_build.vllm_checkpoint)
and the trainer's bf16 matmuls do. vLLM's own GGUF kernels (vllm-gguf-plugin's
dequantize.cuh) are not this: they multiply and subtract in fp16 (__hmul, __hsub), up
to three fp16 roundings per weight, and their small-batch matmuls never dequantize
(8-bit activations against the codes, as llama.cpp's CPU matmuls do). Scoring reads the
bf16 checkpoint, not the GGUF, for that reason.

Codes come from the same rounding llama-quantize ends with: q = clamp(round((w + dmin *
m) / (d * sc))) with ties to even (ggml's nearest_int), on the fp16-rounded d and dmin.
So a QuantWeight made from the original weights and llama-quantize's scales starts with
llama-quantize's codes (exactly, when the GGUF was converted at bf16 or f32 from the same
weights); only sub-blocks whose scale is 0 hold leftover codes that no rounding gives,
and those do not change any value.

QuantWeight trains one tensor in its grid (QAT): an fp32 latent weight and continuous
d, dmin, sub-block scales and mins; forward() rounds them as the GGUF will (fp16 d and
dmin, integer scales, mins and codes, straight-through gradients, LSQ-style for the
scales) and returns the weight dequant(export()) gives, bit for bit. Its state per
weight, in bytes:

    latent weight (fp32)              4
    scales + mins (fp32)              q2_k 0.5, q3_k 0.25 (no mins), q4_k 0.25
    d + dmin (fp32)                   0.03 (q3_k 0.016)
    frozen codes (int8)               1, only after freeze_codes()

about 4.3-4.5 B/weight, ~15 GB for a 3.4B model before the trainer's gradients and
optimizer moments (another ~4.5 + 9 B/weight in fp32 AdamW): whole-model training of a
3B model does not fit a 48 GB L40S, one decoder block at a time does. The forward saves
nothing for backward beyond its parameters (the backward recomputes the rounding), so
autograd holds no per-weight intermediates across a whole-model forward; the one
full-size tensor per call is the returned weight (fp32, or its bf16 copy under autocast)
that the matmul keeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

QK_K = 256
TYPES = ("q2_k", "q3_k", "q4_k")
BLOCK_BYTES = {"q2_k": 84, "q3_k": 110, "q4_k": 144}
F16_MAX = 65504.0


@dataclass(frozen=True)
class Grid:
    sub: int                       # weights per sub-block
    q: tuple[int, int]             # code range
    sc: tuple[int, int]            # sub-block scale range
    m: tuple[int, int] | None      # sub-block min range; None: the type has no mins


GRID = {"q2_k": Grid(16, (0, 3), (0, 15), (0, 15)),
        "q3_k": Grid(16, (-4, 3), (-32, 31), None),
        "q4_k": Grid(32, (0, 15), (0, 63), (0, 63))}


def grid(type: str) -> Grid:
    if type not in GRID:
        raise ValueError(f"{type!r} is not one of {TYPES}")
    return GRID[type]


def bytes_for(type: str, n_weights: int) -> int:
    """Bytes of n_weights stored as type (a multiple of 256 weights: K-quant rows are)."""
    grid(type)
    if n_weights % QK_K:
        raise ValueError(f"{n_weights} weights is not a whole number of {QK_K}-weight {type} blocks")
    return n_weights // QK_K * BLOCK_BYTES[type]


@dataclass
class KQ:
    """One K-quant tensor as integers, in the weight's own row order (rows of the GGUF
    tensor; blocks run along the last dimension):

        codes   int8     (*shape)                 q of each weight (q3_k: -4..3)
        scales  int8     (*shape[:-1], n_sub)     sc of each sub-block (q3_k: -32..31)
        mins    int8     like scales, or None for q3_k
        d       float16  (*shape[:-1], n_super)
        dmin    float16  like d, or None for q3_k
    """
    type: str
    codes: torch.Tensor
    scales: torch.Tensor
    mins: torch.Tensor | None
    d: torch.Tensor
    dmin: torch.Tensor | None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.codes.shape)

    def to(self, device) -> KQ:
        return KQ(self.type, *(None if t is None else t.to(device)
                               for t in (self.codes, self.scales, self.mins, self.d, self.dmin)))


def take_rows(kq: KQ, index) -> KQ:
    """kq with its rows (first dimension) reordered: row i of the result is row index[i].
    Rows are whole blocks, so this is the GGUF's q/k row permutation (or its inverse)."""
    index = torch.as_tensor(np.asarray(index), dtype=torch.long, device=kq.codes.device)
    return KQ(kq.type, *(None if t is None else t[index] for t in (kq.codes, kq.scales, kq.mins, kq.d, kq.dmin)))


# ---------- bytes <-> integers ----------

_SH2 = np.array([0, 2, 4, 6], dtype=np.uint8)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16)


def from_gguf(raw, type: str, shape) -> KQ:
    """The integers of a K-quant tensor as llama-quantize wrote it. raw is its bytes (a
    GGUFReader tensor's .data, or bytes); shape is the weight's numpy/torch shape, the
    GGUF's ne reversed: (rows, columns) for a matrix."""
    g = grid(type)
    shape = tuple(int(s) for s in shape)
    if not shape or shape[-1] % QK_K:
        raise ValueError(f"{type} needs rows that are a multiple of {QK_K} weights, got shape {shape}")
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(raw, dtype=np.uint8)
    else:
        buf = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    n = math.prod(shape)
    if buf.size != bytes_for(type, n):
        raise ValueError(f"{buf.size} bytes is not a {type} tensor of shape {shape} ({bytes_for(type, n)} bytes)")
    blocks = buf.reshape(-1, BLOCK_BYTES[type])
    nb = blocks.shape[0]
    mins = dmin = None
    if type == "q2_k":
        sc = blocks[:, :16]
        scales, mins = sc & 0xF, sc >> 4
        codes = (blocks[:, 16:80].reshape(nb, 2, 1, 32) >> _SH2.reshape(1, 1, 4, 1)) & 3
        d, dmin = _f16(blocks[:, 80:82]), _f16(blocks[:, 82:84])
    elif type == "q3_k":
        hmask, qs, sc = blocks[:, :32], blocks[:, 32:96], blocks[:, 96:108]
        low = (qs.reshape(nb, 2, 1, 32) >> _SH2.reshape(1, 1, 4, 1)) & 3          # element 128n + 32j + l
        high = (hmask.reshape(nb, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1  # bit 4n + j
        codes = (low.reshape(nb, QK_K) | (high.reshape(nb, QK_K) << 2)).astype(np.int8) - 4
        lo4 = np.concatenate([sc[:, :8] & 0xF, sc[:, :8] >> 4], axis=1)            # sub-blocks 0-7, 8-15
        hi2 = (sc[:, 8:12].reshape(nb, 1, 4) >> _SH2.reshape(1, 4, 1)) & 3          # sub-block 4s + k
        scales = (lo4 | (hi2.reshape(nb, 16) << 4)).astype(np.int8) - 32
        d = _f16(blocks[:, 108:110])
    else:
        d, dmin = _f16(blocks[:, 0:2]), _f16(blocks[:, 2:4])
        a, b, c = blocks[:, 4:8], blocks[:, 8:12], blocks[:, 12:16]
        scales = np.concatenate([a & 63, (c & 0xF) | ((a >> 6) << 4)], axis=1)
        mins = np.concatenate([b & 63, (c >> 4) | ((b >> 6) << 4)], axis=1)
        codes = (blocks[:, 16:144].reshape(nb, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0xF
    lead, n_sub, n_super = shape[:-1], shape[-1] // g.sub, shape[-1] // QK_K

    def t(a, *s):
        return torch.from_numpy(np.array(a, copy=True)).reshape(*lead, *s)

    return KQ(type, t(codes.astype(np.int8), shape[-1]), t(scales.astype(np.int8), n_sub),
              None if mins is None else t(mins.astype(np.int8), n_sub), t(d, n_super),
              None if dmin is None else t(dmin, n_super))


def _check(kq: KQ) -> None:
    g = grid(kq.type)
    shape = kq.shape
    if not shape or shape[-1] % QK_K:
        raise ValueError(f"{kq.type} needs rows that are a multiple of {QK_K} weights, got shape {shape}")
    sub, sup = (*shape[:-1], shape[-1] // g.sub), (*shape[:-1], shape[-1] // QK_K)
    for name, rng, want in (("codes", g.q, shape), ("scales", g.sc, sub), ("mins", g.m, sub)):
        v = getattr(kq, name)
        if (v is None) != (rng is None):
            raise ValueError(f"{kq.type}: {name} should be {'None' if rng is None else 'set'}")
        if v is None:
            continue
        if tuple(v.shape) != want:
            raise ValueError(f"{kq.type}: {name} has shape {tuple(v.shape)} for a weight of shape {shape}")
        if v.is_floating_point() or int(v.min()) < rng[0] or int(v.max()) > rng[1]:
            raise ValueError(f"{kq.type}: {name} must be integers in {rng[0]}..{rng[1]}, "
                             f"got {v.dtype} in {v.min().item()}..{v.max().item()}")
    for name, v in (("d", kq.d), ("dmin", kq.dmin)):
        if name == "dmin" and g.m is None:
            if v is not None:
                raise ValueError(f"{kq.type} has no dmin")
            continue
        if v is None or v.dtype != torch.float16 or tuple(v.shape) != sup:
            raise ValueError(f"{kq.type}: {name} must be float16 of shape {sup}")


def pack(kq: KQ) -> bytes:
    """The GGUF bytes of kq, exactly what from_gguf read. Raises on any code, scale or min
    outside its grid instead of clipping it."""
    _check(kq)

    def a(t):
        return t.detach().cpu().contiguous().numpy()

    nb = kq.codes.numel() // QK_K
    codes = a(kq.codes).reshape(nb, QK_K)
    d = a(kq.d).reshape(nb, 1).view(np.uint8)
    if kq.type == "q2_k":
        q = codes.astype(np.uint8).reshape(nb, 2, 4, 32)
        qs = (q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)).reshape(nb, 64)
        sc = a(kq.scales).astype(np.uint8).reshape(nb, 16) | (a(kq.mins).astype(np.uint8).reshape(nb, 16) << 4)
        out = [sc, qs, d, a(kq.dmin).reshape(nb, 1).view(np.uint8)]
    elif kq.type == "q3_k":
        L = (codes.astype(np.int16) + 4).astype(np.uint8)
        low = (L & 3).reshape(nb, 2, 4, 32)
        qs = (low[:, :, 0] | (low[:, :, 1] << 2) | (low[:, :, 2] << 4) | (low[:, :, 3] << 6)).reshape(nb, 64)
        high = (L >> 2).reshape(nb, 8, 32)
        hmask = np.zeros((nb, 32), dtype=np.uint8)
        for bit in range(8):
            hmask |= high[:, bit] << bit
        S = (a(kq.scales).astype(np.int16).reshape(nb, 16) + 32).astype(np.uint8)
        hi2 = (S >> 4).reshape(nb, 4, 4)
        sc = np.concatenate([(S[:, :8] & 0xF) | ((S[:, 8:] & 0xF) << 4),
                             hi2[:, 0] | (hi2[:, 1] << 2) | (hi2[:, 2] << 4) | (hi2[:, 3] << 6)], axis=1)
        out = [hmask, qs, sc, d]
    else:
        S = a(kq.scales).astype(np.uint8).reshape(nb, 8)
        M = a(kq.mins).astype(np.uint8).reshape(nb, 8)
        sc = np.concatenate([S[:, :4] | ((S[:, 4:] >> 4) << 6), M[:, :4] | ((M[:, 4:] >> 4) << 6),
                             (S[:, 4:] & 0xF) | ((M[:, 4:] & 0xF) << 4)], axis=1)
        q = codes.astype(np.uint8).reshape(nb, 4, 2, 32)
        qs = (q[:, :, 0] | (q[:, :, 1] << 4)).reshape(nb, 128)
        out = [d, a(kq.dmin).reshape(nb, 1).view(np.uint8), sc, qs]
    return np.concatenate(out, axis=1).tobytes()


# ---------- values ----------

def _sub_scale(dh: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
    """d * sc per sub-block (exact in fp32): dh (..., n_super), s (..., n_sub)."""
    return (dh.unsqueeze(-1) * s.unflatten(-1, (-1, k))).flatten(-2)


def _values(dl: torch.Tensor, ml: torch.Tensor | None, q: torch.Tensor, sub: int) -> torch.Tensor:
    """ggml's dequantization: (d * sc) * q - (dmin * m), in fp32."""
    w = dl.unsqueeze(-1) * q.unflatten(-1, (-1, sub))
    if ml is not None:
        w = w - ml.unsqueeze(-1)
    return w.flatten(-2)


def dequant(kq: KQ) -> torch.Tensor:
    """The fp32 weight kq stores, bit-identical to gguf.quants.dequantize(pack(kq))."""
    g = grid(kq.type)
    k = QK_K // g.sub
    dl = _sub_scale(kq.d.float(), kq.scales.float(), k)
    ml = None if g.m is None else _sub_scale(kq.dmin.float(), kq.mins.float(), k)
    return _values(dl, ml, kq.codes.float(), g.sub)


def _fp16(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-F16_MAX, F16_MAX).half().float()


def _snap(d, s, dmin, m, g: Grid):
    """fp16 d and dmin, integer scales and mins, and the per-sub-block d * sc and dmin * m.

    + 0.0 turns round's -0.0 (a scale or min in (-0.5, 0)) into the +0.0 the int8 it is
    packed as dequantizes from, as _code does for the codes. Without it d * sc is -0.0
    where ggml has +0.0, every weight of that sub-block differs from the GGUF's in its
    sign bit, and the bitwise export check fails. A Q3_K scale that shrinks past -0.5
    stays there: its codes all round to 0 and its gradient is exactly 0."""
    k = QK_K // g.sub
    dh, sc = _fp16(d), s.round().clamp(*g.sc) + 0.0
    dl = _sub_scale(dh, sc, k)
    if g.m is None:
        return dh, sc, dl, None, None, None
    mh, mi = _fp16(dmin), m.round().clamp(*g.m) + 0.0
    return dh, sc, dl, mh, mi, _sub_scale(mh, mi, k)


def _target(w, dl, ml, g: Grid) -> torch.Tensor:
    """(w + dmin * m) / (d * sc) per weight, shaped (..., n_sub, sub): what the code rounds.
    A sub-block whose d * sc is 0 divides by 1 instead; its codes do not change its value."""
    x = w.unflatten(-1, (-1, g.sub))
    if ml is not None:
        x = x + ml.unsqueeze(-1)
    return x / torch.where(dl == 0, torch.ones_like(dl), dl).unsqueeze(-1)


def _code(x: torch.Tensor, g: Grid) -> torch.Tensor:
    """The integer code of each target, ties to even like ggml's nearest_int; + 0.0 turns
    round's -0.0 into the +0.0 an integer code dequantizes from."""
    return x.round().clamp(*g.q) + 0.0


def quantize(w: torch.Tensor, type: str) -> KQ:
    """A plain round-to-nearest K-quant of w (min/max scales, no search, no importance
    weights): a starting point when there is no llama-quantize output, and for tests."""
    g = grid(type)
    w = w.detach().float()
    if w.shape[-1] % QK_K:
        raise ValueError(f"{type} needs rows that are a multiple of {QK_K} weights, got shape {tuple(w.shape)}")
    k = QK_K // g.sub
    x = w.unflatten(-1, (-1, g.sub))

    def super_of(v, top):  # fp16 super-block scale so that the largest |v| maps to top, and v / it rounded
        big = v.unflatten(-1, (-1, k))
        i = big.abs().argmax(-1, keepdim=True)
        mx = big.gather(-1, i).squeeze(-1)
        dh = _fp16(mx / top)
        q = big / torch.where(dh == 0, torch.ones_like(dh), dh).unsqueeze(-1)
        return dh, q.flatten(-2)

    if g.m is None:  # symmetric: the sub-block's largest |w| maps to code -4, as ggml's make_qx_quants
        i = x.abs().argmax(-1, keepdim=True)
        scale = -x.gather(-1, i).squeeze(-1) / -g.q[0]
        dh, s = super_of(scale, g.sc[0])
        dmin = m = None
    else:  # asymmetric: [min(w, 0), max(w)] spread over the codes
        lo = x.amin(-1).clamp(max=0)
        scale, mn = (x.amax(-1) - lo) / g.q[1], -lo
        dh, s = super_of(scale, g.sc[1])
        dmin, m = super_of(mn, g.m[1])
    dh, sc, dl, mh, mi, ml = _snap(dh, s, dmin, m, g)
    codes = _code(_target(w, dl, ml, g), g).flatten(-2)
    return KQ(type, codes.to(torch.int8), sc.to(torch.int8), None if mi is None else mi.to(torch.int8),
              dh.half(), None if mh is None else mh.half())


# ---------- training ----------

class _FakeQuant(torch.autograd.Function):
    """The quantized weight with straight-through gradients. Saves only its inputs
    (parameters, alive anyway) and recomputes the rounding in backward.

    Gradients (x = (w + dmin * m) / (d * sc), inside = x within the code range):
        w      g * inside                     (0 once codes are frozen)
        d*sc   g * (q - x) inside, g * q outside  (LSQ)
        dmin*m -g outside, 0 inside
    then through d * sc and dmin * m to the fp16 and integer roundings unchanged
    (scales and mins only while within half a step of their range)."""

    @staticmethod
    def forward(ctx, w, d, s, dmin, m, codes, g):
        ctx.g = g
        ctx.save_for_backward(w, d, s, dmin, m, codes)
        _, _, dl, _, _, ml = _snap(d, s, dmin, m, g)
        if codes is None:
            q = _code(_target(w, dl, ml, g), g).flatten(-2)
        else:
            q = codes.float()
        return _values(dl, ml, q, g.sub)

    @staticmethod
    def backward(ctx, gout):
        w, d, s, dmin, m, codes = ctx.saved_tensors
        g = ctx.g
        k = QK_K // g.sub
        dh, sc, dl, mh, mi, ml = _snap(d, s, dmin, m, g)
        G = gout.float().unflatten(-1, (-1, g.sub))
        if codes is None:
            x = _target(w, dl, ml, g)
            q = _code(x, g)
            inside = (x >= g.q[0]) & (x <= g.q[1]) & (dl != 0).unsqueeze(-1)
            gw = (G * inside).flatten(-2) if ctx.needs_input_grad[0] else None
            gdl = (G * torch.where(inside, q - x, q)).sum(-1)
            gml = -(G * ~inside).sum(-1)
            del x, q, inside
        else:
            gw = None
            gdl = (G * codes.float().unflatten(-1, (-1, g.sub))).sum(-1)
            gml = -G.sum(-1)

        def through(gl, dh, si, s, rng):  # gl = d(loss)/d(d * sc) per sub-block
            gl = gl.unflatten(-1, (-1, k))
            gd = (gl * si.unflatten(-1, (-1, k))).sum(-1)
            gs = (gl * dh.unsqueeze(-1)).flatten(-2) * ((s >= rng[0] - 0.5) & (s <= rng[1] + 0.5))
            return gd, gs

        gd, gs = through(gdl, dh, sc, s, g.sc)
        gdm = gm = None
        if g.m is not None:
            gdm, gm = through(gml, mh, mi, m, g.m)
        return gw, gd, gs, gdm, gm, None, None


class QuantWeight(nn.Module):
    """One weight trained in a K-quant grid. forward() is the quantized weight (fp32),
    equal bit for bit to dequant(self.export()).

    weight is the latent fp32 copy; d and dmin (per super-block) and scales and mins (per
    sub-block, in units of their integer grid, so a scale moves a step only when it
    crosses a .5) are the learnable quantization parameters. With kq (from_gguf of
    llama-quantize's output, rows in the weight's order) they start at llama-quantize's
    values and forward() starts at dequant(kq); without it, at quantize(weight).

    freeze_codes() fixes the integer codes at their current values (for the last steps,
    so the codes are settled while the scales still adjust); the latent weight then gets
    no gradient. project_() keeps scales and mins within half a step of their range;
    call it after optimizer steps."""

    def __init__(self, weight: torch.Tensor, type: str, kq: KQ | None = None):
        super().__init__()
        g = grid(type)
        w = weight.detach().float()
        if w.dim() != 2 or w.shape[1] % QK_K:
            raise ValueError(f"QuantWeight needs a matrix with rows a multiple of {QK_K}, got {tuple(w.shape)}")
        if kq is None:
            kq = quantize(w, type)
        elif kq.type != type or kq.shape != tuple(w.shape):
            raise ValueError(f"kq is {kq.type} {kq.shape}, the weight {type} {tuple(w.shape)}")
        _check(kq)
        kq = kq.to(w.device)
        self.type, self.grid = type, g
        self.weight = nn.Parameter(w.clone())
        self.d = nn.Parameter(kq.d.float())
        self.scales = nn.Parameter(kq.scales.float())
        self.dmin = nn.Parameter(kq.dmin.float()) if g.m else None
        self.mins = nn.Parameter(kq.mins.float()) if g.m else None
        self.register_buffer("codes", None)
        # Where rounding the latent weight does not give kq's code (its sub-block scale is
        # 0, or the weight is not what llama-quantize read), the latent weight moves to the
        # value of kq's code, so the module starts exactly at dequant(kq).
        with torch.no_grad():
            dl = _snap(self.d, self.scales, self.dmin, self.mins, g)[2]
            live = (dl != 0).repeat_interleave(g.sub, -1)
            want = kq.codes.float()
            off = (self._round() != want) & live
            self.nudged = int(off.sum())
            if self.nudged:
                self.weight[off] = dequant(kq)[off]
                if ((self._round() != want) & live).any():
                    raise ValueError("cannot place the latent weight on kq's codes (d * sc too small against dmin * m)")

    def _round(self) -> torch.Tensor:
        _, _, dl, _, _, ml = _snap(self.d, self.scales, self.dmin, self.mins, self.grid)
        return _code(_target(self.weight, dl, ml, self.grid), self.grid).flatten(-2)

    @property
    def frozen(self) -> bool:
        return self.codes is not None

    def forward(self) -> torch.Tensor:
        return _FakeQuant.apply(self.weight, self.d, self.scales, self.dmin, self.mins, self.codes, self.grid)

    def quant_parameters(self) -> list[nn.Parameter]:
        """d, scales, dmin and mins (the latent weight is .weight), for their own learning rate."""
        return [p for p in (self.d, self.scales, self.dmin, self.mins) if p is not None]

    @torch.no_grad()
    def project_(self) -> None:
        for p, rng in ((self.scales, self.grid.sc), (self.mins, self.grid.m)):
            if p is not None:
                p.clamp_(rng[0] - 0.49, rng[1] + 0.49)
        for p in (self.d, self.dmin):
            if p is not None:
                p.clamp_(-F16_MAX, F16_MAX)

    @torch.no_grad()
    def freeze_codes(self) -> None:
        self.codes = self.export().codes
        self.weight.requires_grad_(False)

    @torch.no_grad()
    def export(self) -> KQ:
        """The integers forward() uses, ready for pack."""
        g = self.grid
        dh, sc, _, mh, mi, _ = _snap(self.d, self.scales, self.dmin, self.mins, g)
        if not (torch.isfinite(self.d).all() and (self.dmin is None or torch.isfinite(self.dmin).all())):
            raise ValueError("non-finite d or dmin")
        codes = self.codes if self.frozen else self._round().to(torch.int8)
        return KQ(self.type, codes.clone(), sc.to(torch.int8), None if mi is None else mi.to(torch.int8),
                  dh.half(), None if mh is None else mh.half())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # a checkpoint saved after freeze_codes() carries the codes buffer; make room for it
        key = prefix + "codes"
        if key in state_dict and self.codes is None:
            self.codes = torch.empty_like(state_dict[key], device=self.weight.device)
            self.weight.requires_grad_(False)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        return f"{self.type}, shape={tuple(self.weight.shape)}{', codes frozen' if self.frozen else ''}"
