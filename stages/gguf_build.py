"""The shipped GGUF, built from the integers QAT trained instead of by llama-quantize, and
the vLLM checkpoint that is scored in its place.

build() converts the HF checkpoint with the pinned llama.cpp's convert_hf_to_gguf.py, so
metadata, tokenizer, norms and tensor names are exactly what that llama.cpp writes and
loads, then writes a new GGUF with gguf-py in which every weight matrix is replaced by
the bytes stages/kquant.py packed. It is strict where llama-quantize runs were not
checked (F49, F59):

  - every matrix of the converted file must be replaced, and every given tensor must
    land on one, with the right number of bytes for its type and shape;
  - each converted matrix must equal the HF weight it replaces (to the converted
    dtype), which also settles the q/k row order: convert_hf_to_gguf (LlamaModel, so
    GraniteModel) permutes q and k rows for llama.cpp's rotary layout, and the packed
    rows are permuted the same way. K-quant blocks run along rows, so a row
    permutation moves whole packed rows and leaves every block intact;
  - the written file is read back: every tensor's type, the replaced bytes, and with
    expect_gb the file size to within 0.5%.

vllm_checkpoint() writes the bf16 HF checkpoint vLLM scores from gguf-py's
dequantization of that file (stages/qheal.load_quantized), so the scored weights are
the shipped ones, to bf16 rounding (which the trainer's bf16 matmuls also apply). The
other tensors (norms) come from hf_dir and are checked against the file's.

read_kq() is the other direction: llama-quantize's output as kquant.KQ integers in HF
row order, the starting point of QAT.

    python -m stages.gguf_build vllm --gguf <packed gguf> --hf <hf dir> --out <hf out>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from stages import kquant
from stages._util import weight_files
from stages.sft import log

SIZE_TOL = 0.005  # |file - estimate| / estimate allowed by expect_gb
MATCH_TOL = 1e-4  # relative squared error of a converted matrix against its HF weight (bf16 rounding is ~1e-5)


def _gguf(llama_cpp=None):
    """gguf-py, from the llama.cpp checkout when given (the one its converter uses)."""
    if llama_cpp is not None:
        where = str(Path(llama_cpp) / "gguf-py")
        if Path(where).is_dir() and where not in sys.path:
            sys.path.insert(0, where)
    import gguf
    return gguf


def qtype(gguf, name: str):
    """The GGMLQuantizationType called name ("q2_k", "f32", ...)."""
    try:
        return gguf.GGMLQuantizationType[name.upper()]
    except KeyError:
        raise ValueError(f"{name!r} is not a ggml type") from None


def nbytes(gguf, name: str, shape) -> int:
    """Bytes of a tensor of this type and (rows, ..., columns) shape."""
    block, size = gguf.GGML_QUANT_SIZES[qtype(gguf, name)]
    if shape[-1] % block:
        raise ValueError(f"rows of {shape[-1]} weights are not whole {name} blocks of {block}")
    return int(np.prod(shape[:-1], dtype=np.int64)) * (shape[-1] // block) * size


def shape_of(t) -> tuple[int, ...]:
    """A GGUF tensor's numpy/torch shape (its ne reversed)."""
    return tuple(int(x) for x in reversed(t.shape.tolist()))


def qk_rows(n_rows: int, n_head: int) -> np.ndarray:
    """convert_hf_to_gguf's LlamaModel.permute as a row index: GGUF row r is HF row idx[r]."""
    return np.arange(n_rows).reshape(n_head, 2, n_rows // n_head // 2).swapaxes(1, 2).reshape(-1)


def _field(reader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def name_map(gguf, reader):
    """gguf-py's HF -> GGUF tensor name map for the converted file's architecture (the map
    convert_hf_to_gguf itself uses)."""
    arch = _field(reader, gguf.Keys.General.ARCHITECTURE)
    archs = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    if arch not in archs:
        raise ValueError(f"unknown architecture {arch!r} in the converted GGUF")
    blocks = _field(reader, f"{arch}.block_count")
    return gguf.get_tensor_name_map(archs[arch], int(blocks)), arch


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum() / max(float((b ** 2).sum()), 1e-30))


class _HF:
    """Tensors of an HF checkpoint's safetensors, read one at a time as float32 numpy."""

    def __init__(self, hf_dir: Path):
        self.file = weight_files(hf_dir)  # from the shards' headers: a Hub index can be wrong

    def __contains__(self, name: str) -> bool:
        return name in self.file

    def __getitem__(self, name: str) -> np.ndarray:
        from safetensors import safe_open
        with safe_open(str(self.file[name]), "pt") as f:
            return f.get_tensor(name).float().numpy()


def _row_order(gguf, t, hf: np.ndarray, gname: str, heads: dict[str, int]) -> np.ndarray | None:
    """How the converter laid out hf's rows in GGUF tensor t: None (as is) or the q/k row
    index. Raises when t is not hf in either order."""
    conv = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(hf.shape)
    tries: list[np.ndarray | None] = [None]
    kind = gname.split(".")[-2]
    if kind in heads and hf.ndim == 2:
        tries.append(qk_rows(hf.shape[0], heads[kind]))
    errs = []
    for idx in tries:
        e = _rel_err(conv, hf if idx is None else hf[idx])
        if e < MATCH_TOL:
            return idx
        errs.append(e)
    raise ValueError(f"{gname} of the converted GGUF is not its HF weight in any known row order "
                     f"(relative errors {', '.join(f'{e:.3g}' for e in errs)}); the converter changed it")


def convert(hf_dir: str | Path, out: Path, llama_cpp: str | Path, outtype: str = "bf16") -> Path:
    """hf_dir as a GGUF by the llama.cpp checkout's convert_hf_to_gguf.py. A trimmed
    checkpoint's tokenizer is registered with that converter first (vocab.register, a
    no-op on an untrimmed one): the converter refuses a tokenizer hash it does not know,
    and the VM's checkout is a fresh clone every session."""
    from stages import vocab
    vocab.register(llama_cpp, hf_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(llama_cpp) / "convert_hf_to_gguf.py"), str(hf_dir),
                    "--outtype", outtype, "--outfile", str(out)], check=True)
    return out


def _file_type(gguf, counts: dict[str, int]):
    """general.file_type for the type holding the most weights, as llama-quantize labels it."""
    top = max(counts, key=counts.get)
    names = {"q2_k": "MOSTLY_Q2_K", "q3_k": "MOSTLY_Q3_K_M", "q4_k": "MOSTLY_Q4_K_M", "q5_k": "MOSTLY_Q5_K_M",
             "q6_k": "MOSTLY_Q6_K", "q8_0": "MOSTLY_Q8_0", "f16": "MOSTLY_F16", "bf16": "MOSTLY_BF16", "f32": "ALL_F32"}
    return gguf.LlamaFileType[names[top]] if top in names else None


def build(hf_dir, tensors: dict[str, tuple[str, bytes]], out_gguf, llama_cpp, *,
          expect_gb: float | None = None) -> dict:
    """The GGUF of hf_dir with every weight matrix replaced by packed bytes.

    tensors: {HF parameter name: (ggml type, bytes)}, the bytes in HF row order (as
    kquant.pack(kq) of a weight's kq gives them). Raises if a matrix of the converted
    file is not replaced, a name lands on no tensor, a size or type is wrong, or the
    written file does not read back as planned. Returns {"tensors": {GGUF name: {"hf",
    "type", "bytes"}}, "bytes": file size, "gb": file size / 1e9, "path"}."""
    gguf = _gguf(llama_cpp)
    hf_dir, out_gguf = Path(hf_dir), Path(out_gguf)
    work = out_gguf.parent / f".{out_gguf.name}.build"
    shutil.rmtree(work, ignore_errors=True)
    try:
        converted = convert(hf_dir, work / "converted.gguf", llama_cpp)
        reader = gguf.GGUFReader(str(converted))
        by_name = {t.name: t for t in reader.tensors}
        nmap, arch = name_map(gguf, reader)
        heads = {"attn_q": int(_field(reader, f"{arch}.attention.head_count")),
                 "attn_k": int(_field(reader, f"{arch}.attention.head_count_kv") or
                               _field(reader, f"{arch}.attention.head_count"))}

        plan: dict[str, tuple[str, str, np.ndarray]] = {}  # GGUF name -> (HF name, type, rows of bytes)
        hf = _HF(hf_dir)
        for name, (typ, data) in tensors.items():
            gname = nmap.get_name(name, try_suffixes=(".weight", ".bias"))
            if gname is None or gname not in by_name:
                raise ValueError(f"{name} maps to {gname or 'no GGUF name'}, which the converted GGUF does not have")
            if gname in plan:
                raise ValueError(f"{name} and {plan[gname][0]} both map to {gname}")
            t = by_name[gname]
            shape = shape_of(t)
            if len(data) != nbytes(gguf, typ, shape):
                raise ValueError(f"{name}: {len(data)} bytes, but {typ} of shape {shape} is {nbytes(gguf, typ, shape)}")
            if name not in hf:
                raise ValueError(f"{name} is not a tensor of {hf_dir}")
            w = hf[name]
            if w.shape != shape:
                raise ValueError(f"{name} is {w.shape} in {hf_dir}, {gname} is {shape}")
            rows = np.frombuffer(data, dtype=np.uint8).reshape(int(np.prod(shape[:-1])), -1)
            idx = _row_order(gguf, t, w, gname, heads)
            plan[gname] = (name, typ, rows if idx is None else rows[idx])
            del w
        left = [t.name for t in reader.tensors if len(t.shape) >= 2 and t.name not in plan]
        if left:
            raise ValueError(f"{len(left)} matrices of the converted GGUF have no packed bytes and would ship as "
                             f"converted: {', '.join(left[:8])}{' ...' if len(left) > 8 else ''}")

        counts: dict[str, int] = {}
        for gname, (_, typ, _) in plan.items():
            counts[typ] = counts.get(typ, 0) + int(by_name[gname].n_elements)
        _write(gguf, reader, plan, work / "out.gguf", arch, _file_type(gguf, counts))
        del reader, by_name
        report = _verify(gguf, work / "out.gguf", plan, expect_gb)  # before it replaces anything at out_gguf
        (work / "out.gguf").replace(out_gguf)
        report["path"] = str(out_gguf)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log(f"gguf_build: {out_gguf} {report['gb']:.4f} GB, {len(plan)} matrices replaced "
        f"({', '.join(f'{k} {v / 1e6:.1f}M' for k, v in sorted(counts.items()))} weights)")
    return report


def _write(gguf, reader, plan: dict, path: Path, arch: str, file_type) -> None:
    """A copy of reader's file with plan's tensors swapped in (gguf-py's gguf_new_metadata pattern)."""
    w = gguf.GGUFWriter(str(path), arch)
    align = _field(reader, gguf.Keys.General.ALIGNMENT)
    if align is not None:
        w.data_alignment = int(align)
    keys = set()
    for f in reader.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        val_type = f.types[0]
        sub = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = f.contents()
        if f.name == gguf.Keys.General.FILE_TYPE and file_type is not None:
            value = int(file_type)
        w.add_key_value(f.name, value, val_type, sub_type=sub)
        keys.add(f.name)
    if file_type is not None and gguf.Keys.General.FILE_TYPE not in keys:
        w.add_file_type(int(file_type))
    if gguf.Keys.General.QUANTIZATION_VERSION not in keys:
        w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    for t in reader.tensors:
        if t.name in plan:
            _, typ, rows = plan[t.name]
            w.add_tensor_info(t.name, rows.shape, np.dtype(np.uint8), rows.nbytes, qtype(gguf, typ))
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in reader.tensors:
        if t.name in plan:
            w.write_tensor_data(np.ascontiguousarray(plan[t.name][2]))
        else:
            w.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    w.close()


def _verify(gguf, path: Path, plan: dict, expect_gb: float | None) -> dict:
    """Reads the written file back: every replaced tensor has its type and its bytes."""
    size = path.stat().st_size
    out = {}
    for t in gguf.GGUFReader(str(path)).tensors:
        if t.name in plan:
            name, typ, rows = plan[t.name]
            if t.tensor_type != qtype(gguf, typ):
                raise ValueError(f"{path}: {t.name} is {t.tensor_type.name}, planned {typ}")
            if not np.array_equal(np.ascontiguousarray(t.data).view(np.uint8).reshape(rows.shape), rows):
                raise ValueError(f"{path}: {t.name} does not hold the packed bytes")
            out[t.name] = {"hf": name, "type": typ, "bytes": int(t.n_bytes)}
        else:
            out[t.name] = {"hf": None, "type": t.tensor_type.name.lower(), "bytes": int(t.n_bytes)}
    missing = set(plan) - set(out)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    gb = size / 1e9
    if expect_gb is not None and abs(gb - expect_gb) > SIZE_TOL * expect_gb:
        raise ValueError(f"{path} is {gb:.4f} GB, the estimate {expect_gb:.4f} GB (more than {SIZE_TOL:.1%} off)")
    return {"tensors": out, "bytes": size, "gb": gb, "path": str(path)}


def read_kq(gguf_path, hf_dir, gguf=None) -> dict[str, kquant.KQ]:
    """The K-quant tensors of a GGUF (llama-quantize's output) as kquant.KQ, keyed by HF
    parameter name, rows in HF order: q/k rows un-permuted when that matches hf_dir's
    weight better (a wrong order or tensor is ~100% off, a K-quant ~1-20%). Tensors of
    other types are left out."""
    gguf = gguf or _gguf()
    hf_dir = Path(hf_dir)
    reader = gguf.GGUFReader(str(gguf_path))
    nmap, arch = name_map(gguf, reader)
    heads = {"attn_q": int(_field(reader, f"{arch}.attention.head_count")),
             "attn_k": int(_field(reader, f"{arch}.attention.head_count_kv") or
                           _field(reader, f"{arch}.attention.head_count"))}
    hf = _HF(hf_dir)
    to_hf = {}
    for name in hf.file:
        g = nmap.get_name(name, try_suffixes=(".weight", ".bias"))
        if g is not None:
            to_hf[g] = name
    out = {}
    for t in reader.tensors:
        typ = t.tensor_type.name.lower()
        if typ not in kquant.TYPES:
            continue
        if t.name not in to_hf:
            raise ValueError(f"{t.name} of {gguf_path} maps to no tensor of {hf_dir}")
        name = to_hf[t.name]
        kq = kquant.from_gguf(t.data, typ, shape_of(t))
        kind = t.name.split(".")[-2]
        if kind in heads:
            ref = hf[name]
            w = kquant.dequant(kq).numpy()
            inv = np.argsort(qk_rows(w.shape[0], heads[kind]))
            if _rel_err(w[inv], ref) < _rel_err(w, ref):
                kq = kquant.take_rows(kq, inv)
        out[name] = kq
    return out


def vllm_checkpoint(gguf_path, hf_dir, out_dir) -> dict:
    """The bf16 HF checkpoint of a packed GGUF: hf_dir's config, tokenizer and model with
    every quantized weight replaced by gguf-py's dequantization of the file
    (qheal.load_quantized, which raises on any expected weight it cannot map), rounded
    to bf16. Every other GGUF tensor (norms) must equal hf_dir's at bf16, so nothing
    vLLM scores differs from the file but that rounding. Returns {"rel_err": relative
    error to hf_dir by kind, "path"}."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stages import qheal
    from stages.sft import save_checkpoint

    gguf = _gguf()
    gguf_path, out_dir = Path(gguf_path), Path(out_dir)
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), **{key: torch.bfloat16})
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    err = qheal.load_quantized(model, gguf_path, gguf)

    reader = gguf.GGUFReader(str(gguf_path))
    nmap, _ = name_map(gguf, reader)
    params = dict(model.named_parameters())
    to_hf = {nmap.get_name(n, try_suffixes=(".weight", ".bias")): n for n in params}
    covered = set()
    for t in reader.tensors:
        if qheal.hf_name(t.name) in params:
            covered.add(qheal.hf_name(t.name))  # a weight load_quantized put in
            continue
        name = to_hf.get(t.name)
        if name is None:
            raise ValueError(f"{t.name} of {gguf_path} has no parameter in {hf_dir}; vLLM would not run it")
        p = params[name]
        g = torch.from_numpy(np.array(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32))
        if not torch.equal(g.reshape(p.shape).to(p.dtype), p.detach()):
            raise ValueError(f"{t.name} of {gguf_path} is not {name} of {hf_dir}; build the GGUF from this checkpoint")
        covered.add(name)
    extra = sorted(set(params) - covered)
    if extra:
        raise ValueError(f"{len(extra)} parameters of {hf_dir} are not in {gguf_path}, so vLLM would score weights "
                         f"llama.cpp does not have: {', '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}")
    save_checkpoint(model, tok, out_dir)
    index = Path(hf_dir) / "vocab_index.json"  # a trimmed tokenizer's record, which vocab.register reads
    if index.exists():
        shutil.copy2(index, out_dir / index.name)
    log(f"gguf_build: {out_dir} written from {gguf_path.name}, relative error by kind {err}")
    return {"rel_err": err, "path": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description="packed GGUF -> vLLM checkpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vllm", help="write the bf16 HF checkpoint of a packed GGUF")
    v.add_argument("--gguf", required=True)
    v.add_argument("--hf", required=True, help="the checkpoint the GGUF was built from")
    v.add_argument("--out", required=True)
    v.add_argument("--llama-cpp", default="", help="take gguf-py from this checkout")
    a = ap.parse_args()
    if a.llama_cpp:
        _gguf(a.llama_cpp)
    print(json.dumps(vllm_checkpoint(a.gguf, a.hf, a.out), indent=2))


if __name__ == "__main__":
    main()
