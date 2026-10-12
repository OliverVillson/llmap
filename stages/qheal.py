"""Blackhole's repair adapter: a LoRA that makes a quantized GGUF behave like the original.

The quantized model's weights are read back from the GGUF (gguf-py dequantizes every
llama.cpp type) into the original's HF checkpoint, so training sees exactly the
weights llama.cpp will run. They stay frozen; a small LoRA on every layer projection
learns to close the gap to the original (KL on every token of the original's own
thinking, as in stages/ternary.py, 8192-token windows). llama.cpp loads the LoRA on
top of the unchanged quantized GGUF (llama-server --lora), so nothing is
re-quantized and the repair is exactly what was trained (QLoRA-style healing).

    python -m stages.qheal --hf <original hf dir> --gguf <quantized gguf> --traces traces.jsonl \\
        --llama-cpp <dir> --out <dir>          # writes <dir>/adapter and <dir>/lora.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from stages.sft import kd_kl, load_causal_lm, log
from stages.ternary import kl_to_teacher, load_traces, split, windows

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_GGUF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
         "ffn_down": "mlp.down_proj"}


def hf_name(gguf_name: str) -> str | None:
    """The HF parameter a llama-style GGUF weight came from (None for norms and the rest)."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf_name)
    if m and m.group(2) in _GGUF:
        return f"model.layers.{m.group(1)}.{_GGUF[m.group(2)]}.weight"
    return None


def permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """convert_hf_to_gguf.py's reordering of q/k rows for llama-style rotary embeddings."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w: np.ndarray, n_head: int) -> np.ndarray:
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).sum() / max(float((b ** 2).sum()), 1e-30))


def load_quantized(model, gguf_path: Path, gguf) -> dict:
    """Overwrites model's weights with the GGUF's dequantized ones in place. q/k come back
    in HF row order: un-permuted when that matches the original better, which also
    checks the name mapping (a wrong tensor would be ~100% off). Returns the relative
    error per tensor kind."""
    params = dict(model.named_parameters())
    cfg = model.config
    heads = {"q_proj": cfg.num_attention_heads, "k_proj": cfg.num_key_value_heads}
    errs: dict[str, list[float]] = {}
    n = 0
    for t in gguf.GGUFReader(str(gguf_path)).tensors:
        name = hf_name(t.name)
        if name is None or name not in params:
            continue
        p = params[name]
        w = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(tuple(p.shape))
        ref = p.detach().float().cpu().numpy()
        kind = name.split(".")[-2]
        e = rel_err(w, ref)
        if kind in heads:
            w2 = unpermute(w, heads[kind])
            e2 = rel_err(w2, ref)
            if e2 < e:
                w, e = w2, e2
        if e > 0.5:
            raise ValueError(f"{t.name} -> {name}: {e:.0%} off the original; the GGUF doesn't match this checkpoint")
        with torch.no_grad():
            p.copy_(torch.from_numpy(w).to(p.dtype))
        errs.setdefault(kind, []).append(e)
        n += 1
    if n == 0:
        raise ValueError(f"no weights of {gguf_path} map to the checkpoint")
    return {k: round(sum(v) / len(v), 5) for k, v in errs.items()}


def heal(hf_dir: str, gguf_path: Path, traces: Path, out: Path, gguf, *, r: int = 16, lr: float = 1e-4,
         tokens: float = 20e6, length: int = 8192, batch: int = 1, held_windows: int = 16,
         max_minutes: float = 0.0, log_every: int = 50, seed: int = 0) -> dict:
    """Trains the repair LoRA; saves the PEFT adapter to out/adapter and returns a report."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

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


def export(hf_dir: str, gguf_path: Path, out: Path, gguf, adapter: Path | None = None) -> dict:
    """The quantized model as a plain bf16 HF checkpoint (the GGUF's weights dequantized),
    with the repair adapter merged in when given, so vLLM scores exactly what llama.cpp
    runs and what qheal trained on (training also saw bf16 copies of these weights)."""
    from transformers import AutoTokenizer

    from stages.sft import save_checkpoint

    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = load_causal_lm(hf_dir)
    err = load_quantized(model, gguf_path, gguf)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    save_checkpoint(model, tok, out)
    log(f"qheal: {out} written ({'with' if adapter else 'without'} the repair), relative error by kind {err}")
    return err


def to_gguf(llama_cpp: Path, hf_dir: str, adapter: Path, out: Path, outtype: str = "q8_0") -> Path:
    """The PEFT adapter as a llama.cpp LoRA GGUF (convert_lora_to_gguf.py)."""
    subprocess.run([sys.executable, str(llama_cpp / "convert_lora_to_gguf.py"), str(adapter), "--base", hf_dir,
                    "--outtype", outtype, "--outfile", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="blackhole repair adapter")
    ap.add_argument("--hf", required=True, help="the original HF checkpoint")
    ap.add_argument("--gguf", required=True, help="the quantized GGUF")
    ap.add_argument("--traces", default="")
    ap.add_argument("--export", default="", help="only write the quantized model (plus --adapter) as an HF checkpoint here")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    a = ap.parse_args()
    lc = Path(a.llama_cpp)
    sys.path.insert(0, str(lc / "gguf-py"))
    import gguf
    if a.export:
        export(a.hf, Path(a.gguf), Path(a.export), gguf, Path(a.adapter) if a.adapter else None)
        return
    out = Path(a.out)
    heal(a.hf, Path(a.gguf), Path(a.traces), out, gguf, r=a.r, lr=a.lr, tokens=a.tokens, length=a.length,
         max_minutes=a.max_minutes)
    try:  # the llama.cpp copy of the adapter is for shipping; scoring uses the PEFT one
        to_gguf(lc, a.hf, out / "adapter", out / "lora.gguf")
        log(f"qheal: adapter at {out / 'lora.gguf'} ({(out / 'lora.gguf').stat().st_size / 1e6:.0f} MB)")
    except subprocess.CalledProcessError as e:
        log(f"qheal: convert_lora_to_gguf failed ({e}); the PEFT adapter is at {out / 'adapter'}")


if __name__ == "__main__":
    main()
