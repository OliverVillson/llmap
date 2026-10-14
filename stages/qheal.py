"""A quantized GGUF read back into its HF checkpoint, and its KL to the teacher.

gguf-py dequantizes every llama.cpp type, so the weights put into the HF model are
exactly the ones llama.cpp runs (to the model's own dtype: bf16 on the GPU, which is
also what vLLM serves). Two uses:

  export   the GGUF as a plain HF checkpoint, so vLLM scores the shipped weights.
           The GGUF is read once here; later steps load the checkpoint, not the GGUF.
  kl       the KL ladder: held-out teacher-forced KL of several GGUFs. The teacher runs
           once, into a log-prob cache (stages/horizon.py); one student is loaded once
           and each GGUF is read into it in memory, once.

Reading is strict. Every layer projection, the embedding and the head must come from
the GGUF; a tensor the name map misses (a fused attn_qkv, say) would otherwise keep
the original's weights and score the quantized model too high. Every 2-D GGUF weight
with no HF destination is logged. Dequantization runs in a process pool: gguf-py's
numpy code is mostly single-threaded, ~10-60M weights/s per core.

The LoRA repair (heal) is retired from the shipped path: a 16-rank adapter cost ~48 MB,
was trained toward the original (so could not beat it) and was rounded away by the bf16
merge. heal() stays for reference only; nothing merges, converts or ships its adapter.

    python -m stages.qheal export --hf <hf dir> --gguf <quantized gguf> --llama-cpp <dir> --out <hf out>
    python -m stages.qheal kl --hf <hf dir> --gguf a.gguf b.gguf ... --traces traces.jsonl \\
        --llama-cpp <dir> --out kl.json [--teacher <original hf> --vocab-index vocab_index.json]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from stages.sft import kd_kl, load_causal_lm, log

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_GGUF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
         "ffn_down": "mlp.down_proj"}
EMBED, HEAD = "model.embed_tokens.weight", "lm_head.weight"
# Below this many weights a GGUF is read in-process: starting workers costs more.
PARALLEL_MIN = 64_000_000


def hf_name(gguf_name: str) -> str | None:
    """The HF parameter a llama-style GGUF weight came from (None for norms and the rest)."""
    if gguf_name == "token_embd.weight":
        return EMBED
    if gguf_name == "output.weight":
        return HEAD
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf_name)
    if m and m.group(2) in _GGUF:
        return f"model.layers.{m.group(1)}.{_GGUF[m.group(2)]}.weight"
    return None


def expected(params) -> set[str]:
    """The HF parameters a quantized GGUF must replace: every layer projection, the
    embedding and the head (absent when tied to the embedding)."""
    return {n for n in params if n.endswith(".weight") and (n.split(".")[-2] in PROJ or n in (EMBED, HEAD))}


def permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """convert_hf_to_gguf.py's reordering of q/k rows for llama-style rotary embeddings."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w: np.ndarray, n_head: int) -> np.ndarray:
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum() / max(float((b ** 2).sum()), 1e-30))


# ---------- reading GGUF tensors, in parallel ----------

def dequant(gguf, t) -> np.ndarray:
    """A GGUF tensor as float32 in (rows, cols) order, the HF (out, in) layout."""
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(
        tuple(int(x) for x in reversed(t.shape.tolist())))


_W: dict = {}  # a pool worker's gguf module and open readers


def _init_worker(gguf_path_dir: str) -> None:
    if gguf_path_dir not in sys.path:
        sys.path.insert(0, gguf_path_dir)
    import gguf
    _W["gguf"], _W["readers"] = gguf, {}
    torch.set_num_threads(1)


def worker_tensor(path: str, name: str):
    """In a pool worker: a tensor of a GGUF, its reader opened (memory-mapped) once per worker."""
    readers = _W["readers"]
    if path not in readers:
        readers[path] = {t.name: t for t in _W["gguf"].GGUFReader(path).tensors}
    return readers[path][name]


def _read(path: str, name: str) -> tuple[str, np.ndarray]:
    return name, dequant(_W["gguf"], worker_tensor(path, name))


def pool(gguf, workers: int) -> ProcessPoolExecutor:
    """Worker processes with gguf-py from the same place as `gguf` (the llama.cpp checkout's)."""
    where = str(Path(gguf.__file__).resolve().parent.parent)
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_worker, initargs=(where,))


def ordered(ex: ProcessPoolExecutor, fn, jobs, window: int):
    """fn(*job) for each job on the pool, results in job order, at most `window` in flight
    (each result can be a whole dequantized tensor)."""
    it = iter(jobs)
    futs = deque(ex.submit(fn, *j) for j in islice(it, window))
    while futs:
        yield futs.popleft().result()
        for j in islice(it, 1):
            futs.append(ex.submit(fn, *j))


def auto_workers(n_weights: int, workers: int | None) -> int:
    if workers is not None:
        return workers
    return min(16, os.cpu_count() or 1) if n_weights >= PARALLEL_MIN else 0


def load_quantized(model, gguf_path: Path, gguf, *, workers: int | None = None, ref: dict | None = None) -> dict:
    """Overwrites model's weights with the GGUF's dequantized ones in place, reading the
    GGUF once. q/k come back in HF row order: un-permuted when that matches the reference
    better, which also checks the name mapping (a wrong tensor would be ~100% off).
    ref: {hf name: tensor} to compare against (default: the model's current weights).
    Raises when any expected parameter (expected()) has no GGUF tensor; logs every 2-D
    GGUF weight that maps to nothing. Returns the relative error per tensor kind."""
    params = dict(model.named_parameters())
    cfg = model.config
    heads = {"q_proj": cfg.num_attention_heads, "k_proj": cfg.num_key_value_heads}
    want = expected(params)
    tensors = gguf.GGUFReader(str(gguf_path)).tensors
    unmapped = [t.name for t in tensors if len(t.shape) >= 2 and t.name.endswith(".weight")
                and (hf_name(t.name) is None or hf_name(t.name) not in params)]
    if unmapped:
        log(f"qheal: {len(unmapped)} 2-D GGUF weights of {Path(gguf_path).name} map to no parameter: "
            f"{', '.join(unmapped[:8])}{' ...' if len(unmapped) > 8 else ''}")
    jobs = [(str(gguf_path), t.name) for t in tensors if hf_name(t.name) in params]
    n = auto_workers(sum(int(t.n_elements) for t in tensors), workers)
    errs: dict[str, list[float]] = {}
    done: set[str] = set()

    def put(gname: str, w: np.ndarray) -> None:
        name = hf_name(gname)
        p = params[name]
        if tuple(w.shape) != tuple(p.shape):
            raise ValueError(f"{gname} is {tuple(w.shape)}, {name} is {tuple(p.shape)}")
        r = (ref[name] if ref is not None else p).detach().float().cpu().numpy()
        kind = name.split(".")[-2]
        e = rel_err(w, r)
        if kind in heads:
            w2 = unpermute(w, heads[kind])
            e2 = rel_err(w2, r)
            if e2 < e:
                w, e = w2, e2
        if e > 0.5:
            raise ValueError(f"{gname} -> {name}: {e:.0%} off the original; the GGUF doesn't match this checkpoint")
        with torch.no_grad():
            p.copy_(torch.from_numpy(w).to(p.dtype))
        errs.setdefault(kind, []).append(e)
        done.add(name)

    if n > 1:
        with pool(gguf, n) as ex:
            for gname, w in ordered(ex, _read, jobs, 2 * n):
                put(gname, w)
    else:
        by_name = {t.name: t for t in tensors}
        for _, gname in jobs:
            put(gname, dequant(gguf, by_name[gname]))
    missing = sorted(want - done)
    if missing:
        raise ValueError(f"{len(missing)} parameters have no tensor in {gguf_path} and would keep their original "
                         f"weights: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    return {k: round(sum(v) / len(v), 5) for k, v in errs.items()}


def export(hf_dir: str, gguf_path: Path, out: Path, gguf, *, workers: int | None = None) -> dict:
    """The quantized model as a plain bf16 HF checkpoint: the GGUF's weights dequantized,
    so vLLM scores exactly what llama.cpp runs. Loaded in bf16 on the CPU too (as
    gguf_build.vllm_checkpoint does): load_causal_lm's fp32 there would write a
    checkpoint twice the size of the one vLLM serves. Returns the relative error by kind."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stages.sft import save_checkpoint

    tok = AutoTokenizer.from_pretrained(hf_dir)
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(hf_dir, **{key: torch.bfloat16})
    err = load_quantized(model, gguf_path, gguf, workers=workers)
    save_checkpoint(model, tok, out)
    log(f"qheal: {out} written from {Path(gguf_path).name}, relative error by kind {err}")
    return err


# ---------- the KL ladder ----------

def kl(hf_dir: str, ggufs: list[Path], traces_path: Path, out: Path, gguf, *, teacher: str = "",
       vocab_index: str = "", length: int = 8192, tokens: int = 16 * 8192, batch: int = 1,
       cache: Path | None = None, workers: int | None = None) -> list[dict]:
    """Held-out teacher-forced KL of each GGUF to the teacher (default: hf_dir itself; with
    vocab_index, the untrimmed original over the kept ids). The teacher runs once into a
    log-prob cache and is freed; one student is loaded, and each GGUF is read into it once.
    Writes (a JSON list) and returns [{gguf, bytes, gb, kl, positions, length,
    quant_rel_err}], smallest file first, as horizon's ladder() does."""
    from stages import horizon

    out.parent.mkdir(parents=True, exist_ok=True)
    held = horizon.held_out(traces_path, teacher or hf_dir, length=length, tokens=tokens, batch=batch,
                            vocab_index=vocab_index, role="all",
                            path=cache or out.with_name(out.stem + "-teacher.pt"))
    student = load_causal_lm(hf_dir).eval()
    params = dict(student.named_parameters())
    ref = {n: params[n].detach().to("cpu", copy=True) for n in expected(params)}  # the original's weights
    rows = []
    for g in sorted(ggufs, key=lambda p: Path(p).stat().st_size):
        t0 = time.time()
        err = load_quantized(student, Path(g), gguf, workers=workers, ref=ref)
        k = horizon.mean_kl(*horizon.kl_positions(student, held))
        size = Path(g).stat().st_size
        rows.append({"gguf": str(g), "bytes": size, "gb": round(size / 1e9, 4), "kl": round(k, 6),
                     "positions": held.positions, "length": length, "quant_rel_err": err})
        log(f"qheal: {Path(g).name}: {size / 1e9:.3f} GB, held-out KL {k:.5f} ({time.time() - t0:.0f}s)")
    out.write_text(json.dumps(rows, indent=2))
    return rows


# ---------- retired: the LoRA repair, for reference ----------

def heal(hf_dir: str, gguf_path: Path, traces: Path, out: Path, gguf, *, r: int = 16, lr: float = 1e-4,
         tokens: float = 20e6, length: int = 8192, batch: int = 1, held_windows: int = 16,
         max_minutes: float = 0.0, log_every: int = 50, seed: int = 0) -> dict:
    """The retired repair LoRA on the frozen quantized weights; saves the PEFT adapter to
    out/adapter and returns a report. Reference only: nothing merges, converts or ships it."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from stages.ternary import kl_to_teacher, load_traces, split, windows

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    teacher = load_causal_lm(hf_dir).eval()
    student = load_causal_lm(hf_dir)
    q_err = load_quantized(student, gguf_path, gguf)
    log(f"qheal: quantized weights loaded, relative error by kind {q_err}")
    for p in list(student.parameters()) + list(teacher.parameters()):
        p.requires_grad_(False)
    student.config.use_cache = False
    device = next(student.parameters()).device
    if device.type == "cuda":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=PROJ,
                                                 task_type="CAUSAL_LM"))
    params = [p for p in student.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()

    train, held = split(load_traces(traces))
    wins = windows(train, length, seed + 1)[: max(batch, int(tokens) // length)]
    held_w = windows(held, length, 0)[:held_windows]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    total = max(1, len(wins) // batch)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    student.eval()
    hist = [{"step": 0, "kl": kl_to_teacher(student, teacher, held_w, batch=1)}]
    log(f"qheal: {total} steps of {batch} x {length} tokens, {sum(p.numel() for p in params) / 1e6:.1f}M LoRA params; "
        f"held-out KL of the quantized model {hist[0]['kl']:.4f}")
    student.train()
    t0, ema = time.time(), None
    for step in range(1, total + 1):
        ids = wins[(step - 1) * batch: step * batch].to(device)
        with torch.no_grad(), amp:
            t = teacher(input_ids=ids.to(teacher.device), use_cache=False).logits[:, :-1].flatten(0, 1).to(device)
        with amp:
            s = student(input_ids=ids, use_cache=False).logits[:, :-1].flatten(0, 1)
            loss = kd_kl(s, t)
        del t
        if not torch.isfinite(loss):
            log(f"qheal: step {step}: non-finite loss, skipped")
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        timed_out = bool(max_minutes) and time.time() - t0 > 60 * max_minutes
        if step % log_every == 0 or step == total or timed_out:
            student.eval()
            hist.append({"step": step, "kl": kl_to_teacher(student, teacher, held_w, batch=1), "train_kl": ema})
            student.train()
            el = time.time() - t0
            log(f"qheal: step {step}/{total} train KL {ema:.4f} held-out KL {hist[-1]['kl']:.4f} "
                f"eta {el / step * (total - step) / 60:.0f}m")
        if timed_out:
            log(f"qheal: time cap of {max_minutes:g} min reached at step {step}/{total}")
            break
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    report = {"quant_rel_err": q_err, "kl": hist, "tokens": len(wins) * length, "lora_r": r,
              "lora_params": sum(p.numel() for p in params)}
    (out / "qheal.json").write_text(json.dumps(report, indent=2))
    return report


def gguf_module(llama_cpp: str):
    """gguf-py from the llama.cpp checkout that wrote the files."""
    p = str(Path(llama_cpp) / "gguf-py")
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)
    import gguf
    return gguf


def main() -> None:
    ap = argparse.ArgumentParser(description="read quantized GGUFs back into the HF checkpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="the GGUF's weights as a plain HF checkpoint (bf16 on the GPU)")
    e.add_argument("--hf", required=True, help="the HF checkpoint the GGUF was made from")
    e.add_argument("--gguf", required=True)
    e.add_argument("--out", required=True)
    k = sub.add_parser("kl", help="held-out KL of several GGUFs, teacher run once")
    k.add_argument("--hf", required=True, help="the HF checkpoint the GGUFs were made from (the student)")
    k.add_argument("--gguf", required=True, nargs="+")
    k.add_argument("--traces", required=True)
    k.add_argument("--out", required=True, help="JSON written here")
    k.add_argument("--teacher", default="", help="default: --hf; the untrimmed original with --vocab-index")
    k.add_argument("--vocab-index", default="", help="vocab_index.json of a trimmed --hf (stages/vocab.py)")
    k.add_argument("--length", type=int, default=8192)
    k.add_argument("--tokens", type=int, default=16 * 8192, help="held-out positions to score")
    for p in (e, k):
        p.add_argument("--llama-cpp", required=True)
        p.add_argument("--workers", type=int, default=None, help="dequantization processes (default: by size)")
    a = ap.parse_args()
    gguf = gguf_module(a.llama_cpp)
    if a.cmd == "export":
        export(a.hf, Path(a.gguf), Path(a.out), gguf, workers=a.workers)
    else:
        kl(a.hf, [Path(g) for g in a.gguf], Path(a.traces), Path(a.out), gguf, teacher=a.teacher,
           vocab_index=a.vocab_index, length=a.length, tokens=a.tokens, workers=a.workers)


if __name__ == "__main__":
    main()
