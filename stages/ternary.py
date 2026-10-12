"""Blackhole: a dense model trained into ternary weights, {-s, 0, +s} per block of 256,
by distillation from its own full-precision self.

Every decoder-layer linear weight keeps a full-precision latent copy; the forward pass
uses its ternary snap

    s = mean |w| over the block (rounded to bf16),  w_t = s * clamp(round(w / s), -1, 1)

and gradients pass straight through to the latent copy (STE, as in BitNet b1.58).
Activations stay 16-bit. A snapped block's largest |w| is s itself, so llama.cpp's
TQ1_0 (1.69 bits per weight: one fp16 scale per 256 ternary digits) stores it
exactly. Embedding and head stay as they are here (q4_k in the GGUF, after a vocab
trim); norms train in full precision.

Steps (plan: /mnt/project-files/plan/blackhole.md):

  gen     the teacher's own thinking answers on clean CodeContests problems (never
          LiveCodeBench), from a vLLM serving it: the training text
  block   layer by layer (EfficientQAT's block phase): each ternary layer learns to
          reproduce the full layer's output on cached hidden states, its input being
          what the ternary layers before it actually produce. Cheap: one layer's
          forward and backward per step, no teacher pass
  e2e     the whole ternary model against the full model's next-token distribution
          (KL, sft.kd_kl), so errors that build up across layers are fixed too
  kl      mean per-token KL to the teacher on held-out answers (the gate metric)

block and e2e save the latent weights (to continue from) and a snapped copy (to
serve with vLLM and score; it is exactly the ternary model).

    python -m stages.ternary gen --model <hf> --url http://127.0.0.1:8093 --problems <codebench> --out traces.jsonl --tokens 40e6
    python -m stages.ternary block --base <hf> --traces traces.jsonl --out <dir>
    python -m stages.ternary e2e --student <dir>/latent --teacher <hf> --traces traces.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stages.sft import NonFiniteTraining, kd_kl, load_causal_lm, log

BLOCK = 256
TQ1_0_BPW = 1.6875
HELD_OUT = 64  # trace rows kept out of training, for the KL gate


# ---------- ternary weights ----------

def block_scale(w: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """mean |w| per block of `block` along each row, rounded to bf16 (as the snapped
    weights are saved), shape (out, in // block, 1)."""
    b = w.float().reshape(w.shape[0], -1, block)
    return b.abs().mean(-1, keepdim=True).clamp_min(1e-8).to(torch.bfloat16).float()


def snap(w: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """The ternary weight s * {-1, 0, 1}, same shape and dtype as w."""
    b = w.float().reshape(w.shape[0], -1, block)
    s = block_scale(w, block)
    return (s * (b / s).round().clamp(-1, 1)).reshape(w.shape).to(w.dtype)


class TernaryLinear(nn.Linear):
    """nn.Linear whose forward uses the ternary snap of its latent weight (STE)."""

    def forward(self, x):
        w = self.weight
        q = w + (snap(w) - w).detach()
        return F.linear(x, q.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def to_ternary(module: nn.Module) -> int:
    """Swaps every nn.Linear under module for a TernaryLinear in place (same weights).
    Returns how many; rows not a multiple of BLOCK are left alone."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear and child.in_features % BLOCK == 0:
            child.__class__ = TernaryLinear
            n += 1
        else:
            n += to_ternary(child)
    return n


def to_linear(module: nn.Module, snapped: bool) -> None:
    """Back to nn.Linear in place: with snapped=True the weights become their ternary snap."""
    for child in module.modules():
        if type(child) is TernaryLinear:
            if snapped:
                with torch.no_grad():
                    child.weight.copy_(snap(child.weight))
            child.__class__ = nn.Linear


def decoder_layers(model) -> nn.ModuleList:
    return model.model.layers


def ternary_model(model) -> int:
    """Every decoder-layer linear made ternary (not the embedding or head)."""
    return sum(to_ternary(l) for l in decoder_layers(model))


def size_gb(model, vocab: int | None = None, embd_bits: float = 4.5) -> float:
    """The TQ1_0 GGUF's size: ternary layer linears, embedding and head at embd_bits
    (with their rows cut to `vocab` when given), everything else F32."""
    total = 0.0
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    rows = emb.shape[0] if vocab is None else vocab
    for n, p in model.named_parameters():
        if p is emb or p is head:
            total += rows * p.shape[1] * embd_bits
        elif p.dim() >= 2 and ".layers." in n and p.shape[-1] % BLOCK == 0:
            total += p.numel() * TQ1_0_BPW
        else:
            total += p.numel() * 32
    if head is emb:
        total -= rows * emb.shape[1] * embd_bits  # tied: counted once
    return total / 8e9


# ---------- training text ----------

def load_traces(path: Path) -> list[list[int]]:
    """Each trace row's prompt + answer tokens."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r["prompt"] + r["tokens"] for r in rows]


def windows(seqs: list[list[int]], length: int, seed: int = 0) -> torch.Tensor:
    """The sequences packed end to end and cut into (n, length) windows, shuffled."""
    flat = [t for s in seqs for t in s]
    n = len(flat) // length
    if n == 0:
        raise ValueError(f"only {len(flat)} tokens of text, less than one window of {length}")
    w = torch.tensor(flat[: n * length]).reshape(n, length)
    return w[torch.randperm(n, generator=torch.Generator().manual_seed(seed))]


def split(seqs: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """(train, held out): the last HELD_OUT rows are held out (gen writes them in random order)."""
    k = min(HELD_OUT, len(seqs) // 10) or 1
    return seqs[:-k], seqs[-k:]


# ---------- gen ----------

def gen(model_dir: str, url: str, bench_dir: str, out: Path, tokens: float, samples: int = 2,
        max_tokens: int = 8192, workers: int = 128, seed: int = 0) -> int:
    """Samples the served model's thinking answers to clean CodeContests problems until
    `tokens` answer tokens are written. Resumes: rows already in `out` count."""
    from transformers import AutoTokenizer

    from stages import opd

    tok = AutoTokenizer.from_pretrained(model_dir)
    probs = opd.problems(bench_dir, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(len(json.loads(l)["tokens"]) for l in out.read_text().splitlines() if l.strip()) if out.exists() else 0
    start = sum(1 for _ in out.open()) // samples if out.exists() else 0
    lock = threading.Lock()
    state = {"tokens": done, "t0": time.time()}

    def one(i):
        if state["tokens"] >= tokens:
            return
        p = probs[i % len(probs)]
        prompt = opd.render(tok, p)
        if len(prompt) > 4096:
            return
        rows = opd.sample(url, opd.STUDENT, prompt, samples, max_tokens, 1.0, 0.95)
        with lock:
            with out.open("a") as f:
                for r in rows:
                    f.write(json.dumps({"id": p.get("id"), "prompt": prompt, "tokens": r["tokens"],
                                        "finish": r["finish"]}) + "\n")
            state["tokens"] += sum(len(r["tokens"]) for r in rows)
            if i % 50 == 0:
                rate = (state["tokens"] - done) / max(time.time() - state["t0"], 1)
                log(f"gen: {state['tokens'] / 1e6:.1f}M of {tokens / 1e6:.0f}M tokens ({rate:.0f}/s)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        i = start
        while state["tokens"] < tokens:
            list(ex.map(one, range(i, i + workers * 4)))
            i += workers * 4
            if i - start > 20 * len(probs):
                raise RuntimeError("ran out of problems before reaching the token target")
    log(f"gen: {state['tokens'] / 1e6:.1f}M answer tokens in {out}")
    return state["tokens"]


# ---------- block ----------

class _Stop(Exception):
    pass


def layer_inputs(model, ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """(hidden states entering layer 0, the keyword arguments the model passes each
    layer: masks, rotary embeddings, ...) for a batch of windows, captured by a hook
    so any transformers version's layer signature works."""
    got = {}

    def pre(mod, args, kwargs):
        got["h"] = args[0] if args else kwargs["hidden_states"]
        got["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _Stop

    hook = decoder_layers(model)[0].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        model(input_ids=ids, use_cache=False)
    except _Stop:
        pass
    finally:
        hook.remove()
    kw = {k: v for k, v in got["kw"].items() if k not in ("past_key_value", "past_key_values")}
    if "use_cache" in kw:
        kw["use_cache"] = False
    return got["h"], kw


def run_layer(layer, h: torch.Tensor, kw: dict) -> torch.Tensor:
    out = layer(h, **kw)
    return out[0] if isinstance(out, tuple) else out


def block_phase(model, wins: torch.Tensor, *, batch: int = 4, epochs: int = 2, lr: float = 1e-4,
                cache_device: str | None = None, progress=lambda pct, msg: None) -> list[dict]:
    """Makes the decoder layers ternary one at a time (in place), each trained to map
    the ternary layers' own hidden states to the full layer's output on the full
    layers' hidden states. Returns per-layer relative errors before and after."""
    device = next(model.parameters()).device
    cache = torch.device(cache_device) if cache_device else device
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    kw_of: dict[int, dict] = {}  # the layer kwargs per batch size (windows are all one length)
    xs = []
    with torch.no_grad(), amp:
        for i in range(0, len(wins), batch):
            h, kw = layer_inputs(model, wins[i:i + batch].to(device))
            xs.append(h.to(cache))
            kw_of.setdefault(h.shape[0], kw)

    def kwargs(n):
        return kw_of[n]

    x_t = xs
    x_s = [x.clone() for x in xs]
    report = []
    layers = decoder_layers(model)
    for li, layer in enumerate(layers):
        with torch.no_grad(), amp:
            y_t = [run_layer(layer, x.to(device), kwargs(len(x))).to(cache) for x in x_t]
        to_ternary(layer)
        params = list(layer.parameters())
        for p in params:
            p.requires_grad_(True)
            if p.dtype != torch.float32:
                p.data = p.data.float()
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
        steps = epochs * len(x_s)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))

        def rel_err():
            with torch.no_grad(), amp:
                num = sum(float((run_layer(layer, x.to(device), kwargs(len(x))).float() - y.to(device).float()).pow(2).sum())
                          for x, y in zip(x_s[:4], y_t[:4]))
            den = sum(float(y.float().pow(2).sum()) for y in y_t[:4])
            return num / max(den, 1e-12)

        before = rel_err()
        order = list(range(len(x_s)))
        for ep in range(epochs):
            random.Random(li * 100 + ep).shuffle(order)
            for j in order:
                with amp:
                    out = run_layer(layer, x_s[j].to(device), kwargs(len(x_s[j])))
                    y = y_t[j].to(device).float()
                    loss = (out.float() - y).pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)
                if not torch.isfinite(loss):
                    raise NonFiniteTraining(f"block phase: non-finite loss in layer {li}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        after = rel_err()
        for p in params:
            p.requires_grad_(False)
        with torch.no_grad(), amp:
            x_s = [run_layer(layer, x.to(device), kwargs(len(x))).to(cache) for x in x_s]
        x_t = y_t
        del opt
        report.append({"layer": li, "rel_err_snap": round(before, 5), "rel_err_trained": round(after, 5)})
        log(f"block: layer {li + 1}/{len(layers)} relative error {before:.4f} -> {after:.4f}")
        progress(100 * (li + 1) / len(layers), f"layer {li + 1}/{len(layers)}")
    return report


# ---------- e2e ----------

def kl_to_teacher(student, teacher, wins: torch.Tensor, batch: int = 2) -> float:
    """Mean per-token KL(teacher || student) on the windows."""
    device = next(student.parameters()).device
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    total, n = 0.0, 0
    with torch.no_grad(), amp:
        for i in range(0, len(wins), batch):
            ids = wins[i:i + batch].to(device)
            s = student(input_ids=ids, use_cache=False).logits[:, :-1].flatten(0, 1)
            t = teacher(input_ids=ids.to(teacher.device), use_cache=False).logits[:, :-1].flatten(0, 1).to(device)
            total += float(kd_kl(s, t)) * s.shape[0]
            n += s.shape[0]
    return total / max(n, 1)


def e2e_phase(student, teacher, wins: torch.Tensor, held: torch.Tensor, *, batch: int = 4, lr: float = 2e-5,
              epochs: float = 1.0, max_minutes: float = 0.0, log_every: int = 20,
              progress=lambda pct, msg: None) -> list[dict]:
    """KL(teacher || student) on every position of the training windows, all decoder
    weights trained (latent copies of the ternary ones, plus norms); embedding and head
    frozen. Returns the held-out KL along the way."""
    device = next(student.parameters()).device
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for p in decoder_layers(student).parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()
        p.requires_grad_(True)
        params.append(p)
    final_norm = getattr(student.model, "norm", None)
    if final_norm is not None:
        for p in final_norm.parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
            params.append(p)
    student.config.use_cache = False
    if device.type == "cuda":
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    total = max(1, int(len(wins) * epochs) // batch)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist = [{"step": 0, "kl": kl_to_teacher(student, teacher, held)}]
    log(f"e2e: {total} steps of {batch} x {wins.shape[1]} tokens; held-out KL {hist[0]['kl']:.4f}")
    t0, step, ema = time.time(), 0, None
    student.train()
    while step < total:
        for i in range(0, len(wins) - batch + 1, batch):
            ids = wins[i:i + batch].to(device)
            with torch.no_grad(), amp:
                t = teacher(input_ids=ids.to(teacher.device), use_cache=False).logits[:, :-1].flatten(0, 1).to(device)
            with amp:
                s = student(input_ids=ids, use_cache=False).logits[:, :-1].flatten(0, 1)
                loss = kd_kl(s, t)
            del t
            if not torch.isfinite(loss):
                raise NonFiniteTraining(f"e2e: non-finite loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if step % log_every == 0 or step == total:
                student.eval()
                hist.append({"step": step, "kl": kl_to_teacher(student, teacher, held), "train_kl": ema})
                student.train()
                el = time.time() - t0
                log(f"e2e: step {step}/{total} train KL {ema:.4f} held-out KL {hist[-1]['kl']:.4f} "
                    f"eta {el / step * (total - step) / 60:.0f}m")
                progress(100 * step / total, f"step {step}/{total} KL {hist[-1]['kl']:.4f}")
            if step >= total or (max_minutes and time.time() - t0 > 60 * max_minutes):
                break
        if max_minutes and time.time() - t0 > 60 * max_minutes:
            log(f"e2e: time cap of {max_minutes:g} min reached at step {step}/{total}")
            break
    if hist[-1]["step"] != step:
        student.eval()
        hist.append({"step": step, "kl": kl_to_teacher(student, teacher, held), "train_kl": ema})
    return hist


# ---------- saving ----------

def save(model, tok, out: Path, snapped: bool) -> None:
    """latent (snapped=False): the trainable weights, to continue from; snapped: the
    ternary model itself as plain bf16 linears, which vLLM serves as is."""
    m = copy.deepcopy(model)
    to_linear(m, snapped)
    m.to(torch.bfloat16)
    m.config.use_cache = True
    tmp = out.parent / (out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    m.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    del m


def check_ternary(model_dir: Path, samples: int = 4) -> None:
    """Every layer linear of a snapped checkpoint really is s * {-1, 0, 1} per block."""
    m = load_causal_lm(str(model_dir))
    lin = [mod for mod in decoder_layers(m).modules() if isinstance(mod, nn.Linear) and mod.in_features % BLOCK == 0]
    for mod in lin[:: max(1, len(lin) // samples)]:
        w = mod.weight.float().reshape(mod.weight.shape[0], -1, BLOCK)
        s = w.abs().amax(-1, keepdim=True)
        levels = torch.where(s > 0, w / s.clamp_min(1e-30), torch.zeros_like(w))
        if not torch.all((levels == 0) | (levels.abs() == 1)):
            raise ValueError(f"{model_dir}: a layer weight is not ternary per {BLOCK}-block")


# ---------- the commands ----------

def _load(path: str):
    from transformers import AutoTokenizer
    return load_causal_lm(path), AutoTokenizer.from_pretrained(path)


def cmd_block(a) -> None:
    torch.manual_seed(a.seed)
    model, tok = _load(a.base)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    train, _ = split(load_traces(Path(a.traces)))
    wins = windows(train, a.length, a.seed)[: a.windows]
    log(f"block: {len(wins)} windows of {a.length} tokens ({len(wins) * a.length / 1e6:.1f}M tokens), {a.epochs} epochs")
    report = block_phase(model, wins, batch=a.batch, epochs=a.epochs, lr=a.lr, cache_device=a.cache_device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save(model, tok, out / "latent", snapped=False)
    save(model, tok, out / "snapped", snapped=True)
    (out / "block.json").write_text(json.dumps({"layers": report, "size_gb": size_gb(model),
                                                "size_gb_vocab24k": size_gb(model, 24576)}, indent=2))
    log(f"block: saved to {out}")


def cmd_e2e(a) -> None:
    torch.manual_seed(a.seed)
    student, tok = _load(a.student)
    ternary_model(student)
    teacher, _ = _load(a.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    train, held = split(load_traces(Path(a.traces)))
    wins = windows(train, a.length, a.seed + 1)
    if a.tokens:
        wins = wins[: max(a.batch, int(a.tokens) // a.length)]
    held_w = windows(held, a.length, 0)[: a.held_windows]
    log(f"e2e: {len(wins) * a.length / 1e6:.0f}M training tokens, {len(held_w)} held-out windows")
    hist = e2e_phase(student, teacher, wins, held_w, batch=a.batch, lr=a.lr, max_minutes=a.max_minutes,
                     log_every=a.log_every)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save(student, tok, out / "latent", snapped=False)
    save(student, tok, out / "snapped", snapped=True)
    check_ternary(out / "snapped")
    (out / "e2e.json").write_text(json.dumps({"kl": hist, "tokens": len(wins) * a.length,
                                              "size_gb": size_gb(student), "size_gb_vocab24k": size_gb(student, 24576)},
                                             indent=2))
    log(f"e2e: saved to {out}")


def cmd_kl(a) -> None:
    """Held-out KL of a model to the teacher; --snap makes its layers ternary first
    (the "no training" point of gate 1)."""
    student, _ = _load(a.student)
    if a.snap:
        ternary_model(student)
    teacher, _ = _load(a.teacher)
    _, held = split(load_traces(Path(a.traces)))
    kl = kl_to_teacher(student, teacher, windows(held, a.length, 0)[: a.held_windows])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"kl": kl, "snap": a.snap}) + "\n")
    log(f"kl: {kl:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="blackhole: ternary distillation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--url", required=True)
    g.add_argument("--problems", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=float, default=40e6)
    b = sub.add_parser("block")
    b.add_argument("--base", required=True)
    b.add_argument("--windows", type=int, default=2048)
    b.add_argument("--epochs", type=int, default=2)
    b.add_argument("--lr", type=float, default=1e-4)
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--cache-device", default=None, help="where hidden states are kept (default: the GPU)")
    e = sub.add_parser("e2e")
    e.add_argument("--student", required=True)
    e.add_argument("--teacher", required=True)
    e.add_argument("--tokens", type=float, default=60e6)
    e.add_argument("--lr", type=float, default=2e-5)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--max-minutes", type=float, default=0.0)
    e.add_argument("--log-every", type=int, default=50)
    k = sub.add_parser("kl")
    k.add_argument("--student", required=True)
    k.add_argument("--teacher", required=True)
    k.add_argument("--snap", action="store_true")
    for p in (b, e, k):
        p.add_argument("--traces", required=True)
        p.add_argument("--length", type=int, default=2048)
        p.add_argument("--seed", type=int, default=0)
    for p in (b, e, k):
        p.add_argument("--out", required=True)
    for p in (e, k):
        p.add_argument("--held-windows", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.model, a.url, a.problems, Path(a.out), a.tokens)
    else:
        {"block": cmd_block, "e2e": cmd_e2e, "kl": cmd_kl}[a.cmd](a)


if __name__ == "__main__":
    main()
