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

The ternary route is parked; Event Horizon trains into llama.cpp's K-quant grids
instead (stages/qat.py). gen is the training text of both.

Steps (plan: /mnt/project-files/plan/blackhole.md):

  gen     the teacher's own thinking answers on clean problems (CodeContests and
          functional ones, never LiveCodeBench), from a vLLM serving it, asked
          exactly as the eval asks them: up to 16,384 thinking tokens, then the
          eval's forced answer, each row labelled pass/fail by its tests. Rows are
          stages/traces.py's format; gen_command() is the step's shell script
  block   layer by layer (EfficientQAT's block phase): each ternary layer learns to
          reproduce the full layer's output on cached hidden states, its input being
          what the ternary layers before it actually produce. Cheap: one layer's
          forward and backward per step, no teacher pass
  e2e     the whole ternary model against the full model's next-token distribution
          (KL, sft.kd_kl), so errors that build up across layers are fixed too
  kl      mean per-token KL to the teacher on held-out answers (the gate metric)

Training reads the traces one row per window, starting at its prompt (block: fixed
windows of the rows long enough; e2e and kl: traces.batches with padding masked and
prompt tokens at a small weight), never packed end to end. The held-out problems are
picked by a crc32 of their id (traces.split).

block and e2e save the latent weights (to continue from) and a snapped copy (to
serve with vLLM and score; it is exactly the ternary model).

    python -m stages.ternary gen --model <hf> --url http://127.0.0.1:8093 --problems <codebench> --out traces.jsonl --tokens 8e6
    python -m stages.ternary block --base <hf> --traces traces.jsonl --out <dir>
    python -m stages.ternary e2e --student <dir>/latent --teacher <hf> --traces traces.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import random
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stages import traces
from stages.sft import NonFiniteTraining, kd_kl, load_causal_lm, log

BLOCK = 256
TQ1_0_BPW = 1.6875
# Attention is not ternary (Oliver, 2026-10-12: up to 0.2 GB more for it). Long thinking
# needs exact looking-back over thousands of tokens, which coarse q/k blur. It trains in
# full precision and is quantized to these llama.cpp types at export (with an imatrix,
# where 4-bit costs little): ~0.32 GB instead of ~0.13 GB ternary on Granite 3B.
ATTENTION = {"q_proj": ("q4_k", 4.5), "k_proj": ("q4_k", 4.5), "v_proj": ("q4_k", 4.5), "o_proj": ("q3_k", 3.4375)}


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
    Returns how many; attention (ATTENTION) and rows not a multiple of BLOCK are left alone."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear and child.in_features % BLOCK == 0 and name not in ATTENTION:
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
    """The GGUF's size: ternary layer linears in TQ1_0, attention at its ATTENTION types,
    embedding and head at embd_bits (with their rows cut to `vocab` when given), everything
    else F32."""
    total = 0.0
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    rows = emb.shape[0] if vocab is None else vocab
    for n, p in model.named_parameters():
        if p is emb or p is head:
            total += rows * p.shape[1] * embd_bits
        elif p.dim() >= 2 and ".layers." in n and n.split(".")[-2] in ATTENTION:
            total += p.numel() * ATTENTION[n.split(".")[-2]][1]
        elif p.dim() >= 2 and ".layers." in n and p.shape[-1] % BLOCK == 0:
            total += p.numel() * TQ1_0_BPW
        else:
            total += p.numel() * 32
    if head is emb:
        total -= rows * emb.shape[1] * embd_bits  # tied: counted once
    return total / 8e9


# ---------- training text ----------

def load_traces(path: Path) -> list[dict]:
    """The trace rows (traces.load: problem ids, int prompts and answers)."""
    return traces.load(path)


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(train, held out) by problem, from a crc32 of its id (traces.split)."""
    return traces.split(rows)


def windows(rows: list[dict], length: int, seed: int = 0) -> torch.Tensor:
    """(n, length) windows for the fixed-shape block phase, shuffled: one per row of at
    least `length` tokens, starting at its prompt and cropped. No window starts
    mid-answer or holds two traces; shorter rows are left out (traces.batches pads
    them instead, with a mask)."""
    seqs = [(r["prompt"] + r["tokens"])[:length] for r in rows if len(r["prompt"]) + len(r["tokens"]) >= length]
    if not seqs:
        raise ValueError(f"no trace of {length} tokens or more")
    w = torch.tensor(seqs)
    return w[torch.randperm(len(w), generator=torch.Generator().manual_seed(seed))]


def trace_batches(rows: list[dict], length: int, batch: int, tokens: float = 0, seed: int = 0) -> list[tuple]:
    """traces.batches as a list, cut after `tokens` real tokens when given."""
    out, n = [], 0
    for b in traces.batches(rows, length, batch, seed=seed):
        if tokens and n >= tokens:
            break
        out.append(b)
        n += int(b[1].sum())
    if not out:
        raise ValueError(f"no trace with an answer token inside {length} tokens")
    return out


# ---------- gen ----------

MAX_TOKENS = 16384  # thinking per answer, as the eval allows before it forces an answer
# The gen server's --max-model-len: a prompt of up to ~2k tokens, MAX_TOKENS of thinking,
# the early-stop line and closing tag, and the forced answer (opd.ANSWER_TOKENS).
MAX_MODEL_LEN = 22528
FORCED_MARGIN = 64  # tokens kept free for the early-stop line and the closing tag
SEQS = {"l40s": 160, "b200": 384}  # --max-num-seqs per GPU with an fp8 KV cache


def gen_command(model_dir: str, problems: str, out: Path, tokens: float, *, port: int = 8093,
                seqs: int = SEQS["l40s"], samples: int = 2, gpu_util: float = 0.9,
                python: str = "$LOBBOT_VLLM_PY", health_minutes: int = 20) -> str:
    """The shell script of a gen step: a vLLM serving model_dir for the traces (fp8 KV
    cache, seqs sequences, its own process group), then gen with workers x samples =
    seqs, so the client never queues more than the server runs. The server must be
    healthy within health_minutes; it is stopped, the whole group, however gen ends."""
    from stages.opd import STUDENT
    w = Path(out).parent
    log_path = w / "vllm-gen.log"
    return "\n".join([
        "set -e",
        f"mkdir -p {w}",
        # The vLLM venv's bin first on PATH: FlashInfer JIT-builds kernels with its ninja.
        f'PATH="$(dirname "{python}"):$PATH" '
        f"setsid {python} -m vllm.entrypoints.openai.api_server --model {model_dir} "
        f"--served-model-name {STUDENT} --host 127.0.0.1 --port {port} --max-model-len {MAX_MODEL_LEN} "
        f"--max-num-seqs {seqs} --kv-cache-dtype fp8 --gpu-memory-utilization {gpu_util} "
        f"--enable-prefix-caching > {log_path} 2>&1 & gen=$!",
        "trap 'kill -TERM -- -$gen 2>/dev/null; for i in $(seq 30); do kill -0 $gen 2>/dev/null || break; sleep 1; "
        "done; kill -KILL -- -$gen 2>/dev/null; true' EXIT",
        f"for i in $(seq {health_minutes * 12}); do curl -sf http://127.0.0.1:{port}/health >/dev/null && break; "
        f"kill -0 $gen 2>/dev/null || {{ tail -50 {log_path}; exit 1; }}; sleep 5; done",
        f"curl -sf http://127.0.0.1:{port}/health >/dev/null || "
        f"{{ echo 'vLLM not healthy after {health_minutes} min'; tail -50 {log_path}; exit 1; }}",
        f"python -m stages.ternary gen --model {model_dir} --url http://127.0.0.1:{port} --problems {problems} "
        f"--out {out} --tokens {tokens:.0f} --samples {samples} --workers {max(1, seqs // samples)} "
        f"--server-pid $gen",
    ])


class _Workers:
    """n daemon threads applying fn to the argument tuples put on .todo; (args, result or
    the exception) come back on .done. Daemon threads, so an abort does not wait for
    the requests still in flight."""

    def __init__(self, n: int, fn):
        self.n, self.todo, self.done = n, queue.Queue(), queue.Queue()
        for _ in range(n):
            threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn):
        while (args := self.todo.get()) is not None:
            try:
                self.done.put((args, fn(*args)))
            except Exception as e:
                self.done.put((args, e))

    def close(self):
        for _ in range(self.n):
            self.todo.put(None)


def _resume(out: Path, samples: int) -> list[dict]:
    """The rows already in out, of complete problems only. A line cut by a stop
    mid-write, and the rows of a problem without exactly `samples` of them, are dropped
    from the file; anything but token ids raises (traces.load)."""
    if not out.exists():
        return []
    text = out.read_text()
    if text and not text.endswith("\n"):
        log(f"gen: dropping a line cut short at the end of {out}")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text[: text.rfind("\n") + 1])
        os.replace(tmp, out)
    rows = traces.load(out)
    per = Counter(r["id"] for r in rows)
    bad = {k for k, v in per.items() if v != samples}
    if bad:
        log(f"gen: dropping the rows of {len(bad)} problems without {samples} rows each from {out}")
        rows = [r for r in rows if r["id"] not in bad]
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, out)
    return rows


def gen(model_dir: str, url: str, bench_dir: str, out: Path, tokens: float, samples: int = 2,
        max_tokens: int = MAX_TOKENS, workers: int = 80, seed: int = 0, *, max_model_len: int = MAX_MODEL_LEN,
        server_pid: int | None = None, max_fail: float = 0.03, min_median: int = traces.MIN_MEDIAN) -> int:
    """Samples the served model's thinking answers to clean training problems
    (opd.problems: CodeContests plus functional ones, never LiveCodeBench) until `tokens`
    answer tokens are in `out`; returns how many are.

    Each problem is asked as the eval asks it (opd.render, opd.sample), `samples` times
    in one request. A thinking that ends without an answer gets the eval's forced answer
    (opd.continue_forced), so the text covers the whole MAX_TOKENS regime and the
    forced-answer pattern. Every row is labelled with whether it passes the problem's
    tests (opd.passes, on a CPU pool beside the requests): {id, prompt, tokens, finish,
    passed, forced}.

    `workers` requests stay open (workers x samples = the server's --max-num-seqs),
    refilled as each one finishes. No new problem starts once the tokens written, those
    being labelled and an estimate for those in flight reach the target. A problem
    whose request still fails after opd's retries is skipped; more than max_fail of
    them, or a dead server, aborts. Resumes by problem id, never asks one twice, and
    raises when the pool runs out before the target. traces.count beside out tracks
    progress; the whole file is validated (traces.validate) before this returns."""
    from transformers import AutoTokenizer

    from stages import opd, sandbox

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_prompt = max_model_len - max_tokens - opd.ANSWER_TOKENS - FORCED_MARGIN
    if max_prompt < 256:
        raise ValueError(f"--max-model-len {max_model_len} leaves {max_prompt} tokens for the prompt")
    tok = AutoTokenizer.from_pretrained(model_dir)
    opd.client(workers)
    opd.watch(url, server_pid)
    rows = _resume(out, samples)
    seen = {r["id"] for r in rows}
    probs = [p for p in opd.problems(bench_dir, seed=seed) if p["id"] not in seen]
    lens = [len(r["tokens"]) for r in rows]
    st = {"tokens": sum(lens), "rows": len(rows), "problems": len(seen), "pending": 0, "asked": 0, "failed": 0,
          "too_long": 0, "sampled": 0, "answers": len(lens), "answer_tokens": sum(lens), "cut": 0, "forced": 0}
    lock, t0 = threading.Lock(), time.time()

    def rate() -> float:
        return st["sampled"] / max(time.time() - t0, 1e-3)

    def count(**extra) -> None:
        traces.write_count(out, tokens=st["tokens"], target=int(tokens), rows=st["rows"], problems=st["problems"],
                           failed=st["failed"], tok_s=round(rate()), **extra)

    log(f"gen: {len(seen)} problems ({st['tokens'] / 1e6:.1f}M answer tokens) already in {out}, "
        f"{len(probs)} left; target {tokens / 1e6:.1f}M, {workers} x {samples} in flight, prompts <= {max_prompt}")
    count(done=False)
    if probs and st["tokens"] < tokens:
        first = opd.render(tok, probs[0])
        log(f"gen: first prompt ({probs[0]['id']}, {len(first)} tokens):\n{tok.decode(first)[:3000]}")
        theirs = opd.render_remote(url, opd.STUDENT, probs[0])
        if theirs != first:
            at = next((i for i, (a, b) in enumerate(zip(first, theirs)) if a != b), min(len(first), len(theirs)))
            raise RuntimeError(f"render() and the server's chat template disagree on {probs[0]['id']} from token "
                               f"{at}: the training prompts would not be the eval's")

    def ask(p: dict, prompt: list[int]) -> list[dict]:
        got = opd.sample(url, opd.STUDENT, prompt, samples, max_tokens)
        for r in got:
            r["forced"] = opd.forced_reason(tok, r["tokens"], r["finish"])
            if r["forced"]:
                think = opd.open_thinking(tok, r["tokens"])
                r["tokens"] = think + opd.continue_forced(url, opd.STUDENT, prompt, think, why=r["forced"])
        return got

    def label(p: dict, prompt: list[int], got: list[dict]) -> None:
        try:
            ok = opd.passes(tok, [(p, r["tokens"]) for r in got])
        except Exception as e:  # unlabelled rather than lost
            log(f"gen: labelling {p['id']} failed: {e}")
            ok = [None] * len(got)
        lines = "".join(json.dumps({"id": p["id"], "prompt": prompt, "tokens": r["tokens"], "finish": r["finish"],
                                    "passed": k, "forced": r["forced"]}) + "\n" for r, k in zip(got, ok))
        n = sum(len(r["tokens"]) for r in got)
        with lock:
            with out.open("a") as f:
                f.write(lines)
            st["tokens"] += n
            st["pending"] -= n
            st["rows"] += len(got)
            st["problems"] += 1
            count(done=False)

    asking, labelling = _Workers(workers, ask), _Workers(max(1, sandbox.WORKERS // samples), label)
    pool, inflight, to_label, last = iter(probs), 0, 0, time.time()
    try:
        while True:
            while inflight < workers:
                mean = st["answer_tokens"] / st["answers"] if st["answers"] else max_tokens / 2
                with lock:
                    if st["tokens"] + st["pending"] + inflight * samples * mean >= tokens:
                        break
                p = next(pool, None)
                if p is None:
                    break
                prompt = opd.render(tok, p)
                if len(prompt) > max_prompt:
                    st["too_long"] += 1
                    continue
                asking.todo.put((p, prompt))
                inflight += 1
            if not inflight:
                break
            (p, prompt), got = asking.done.get()
            inflight -= 1
            st["asked"] += 1
            if isinstance(got, opd.ServerDead):
                raise got
            if isinstance(got, Exception):
                st["failed"] += 1
                log(f"gen: {p['id']} skipped ({type(got).__name__}: {got})")
                if st["failed"] > max(2, max_fail * st["asked"]):
                    raise RuntimeError(f"gen: {st['failed']} of {st['asked']} problems failed; stopping") from got
                continue
            n = sum(len(r["tokens"]) for r in got)
            with lock:
                st["pending"] += n
            st["sampled"] += n
            st["answers"] += len(got)
            st["answer_tokens"] += n
            st["cut"] += sum(r["finish"] == "length" for r in got)
            st["forced"] += sum(bool(r["forced"]) for r in got)
            labelling.todo.put((p, prompt, got))
            to_label += 1
            while not labelling.done.empty():
                _, res = labelling.done.get()
                to_label -= 1
                if isinstance(res, Exception):
                    raise res
            if time.time() - last > 60:
                last = time.time()
                new = max(st["answers"] - len(lens), 1)
                log(f"gen: {st['tokens'] / 1e6:.2f}M of {tokens / 1e6:.1f}M answer tokens, {rate():.0f} tok/s, "
                    f"cut {st['cut'] / new:.0%}, forced {st['forced'] / new:.0%}, {st['failed']} failed, "
                    f"eta {(tokens - st['tokens']) / max(rate(), 1e-3) / 60:.0f} min")
        while to_label:
            _, res = labelling.done.get()
            to_label -= 1
            if isinstance(res, Exception):
                raise res
    finally:
        asking.close()
        labelling.close()
    if st["tokens"] < tokens:
        raise RuntimeError(f"gen: the problem pool ran out at {st['tokens'] / 1e6:.2f}M of {tokens / 1e6:.1f}M "
                           f"answer tokens ({st['problems']} problems written, {st['failed']} failed, "
                           f"{st['too_long']} prompts too long); add problems or lower --tokens")
    report = traces.validate(traces.load(out), samples, min_median)
    report.update(tok_s=round(rate()), failed=st["failed"], too_long=st["too_long"])
    count(done=True, report=report)
    log(f"gen: {st['tokens'] / 1e6:.2f}M answer tokens in {out}: {report['problems']} problems, median answer "
        f"{report['median_answer']:.0f} tokens, cut {report['cut']:.0%}, forced {report['forced']:.0%}, "
        f"passed {report['passed'] if report['passed'] is None else format(report['passed'], '.0%')}, "
        f"{report['tok_s']} tok/s")
    return st["tokens"]


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

def _amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def _batch_kl(student, teacher, ids, mask, w, device):
    """(weighted mean KL(teacher || student) over a batch's targets, their total weight)."""
    ids, mask, w = ids.to(device), mask.to(device), w.to(device)
    with torch.no_grad(), _amp(device):
        t = teacher(input_ids=ids.to(teacher.device), attention_mask=mask.to(teacher.device),
                    use_cache=False).logits[:, :-1].to(device)
    with _amp(device):
        s = student(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    keep = w > 0
    return kd_kl(s[keep], t[keep], weights=w[keep]), w[keep].sum()


def kl_to_teacher(student, teacher, data, batch: int = 2) -> float:
    """Mean per-token KL(teacher || student) over trace batches (trace_batches: padding
    left out, prompt tokens at their weight), or over plain (n, length) windows."""
    device = next(student.parameters()).device
    if isinstance(data, torch.Tensor):
        data = [(data[i:i + batch], torch.ones_like(data[i:i + batch]), torch.ones(len(data[i:i + batch]),
                 data.shape[1] - 1)) for i in range(0, len(data), batch)]
    total, n = 0.0, 0.0
    with torch.no_grad():
        for ids, mask, w in data:
            kl, wsum = _batch_kl(student, teacher, ids, mask, w, device)
            total += float(kl) * float(wsum)
            n += float(wsum)
    return total / max(n, 1e-12)


def e2e_phase(student, teacher, data: list[tuple], held: list[tuple], *, lr: float = 2e-5,
              epochs: float = 1.0, max_minutes: float = 0.0, log_every: int = 20,
              progress=lambda pct, msg: None) -> list[dict]:
    """KL(teacher || student) on every answer position of the training batches (and
    prompt positions at their small weight), all decoder weights trained (latent copies
    of the ternary ones, plus norms); embedding and head frozen. data and held are
    trace batches (trace_batches). Returns the held-out KL along the way."""
    device = next(student.parameters()).device
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
    total = max(1, int(len(data) * epochs))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    hist = [{"step": 0, "kl": kl_to_teacher(student, teacher, held)}]
    log(f"e2e: {total} steps of up to {data[0][0].shape[0]} x {max(b[0].shape[1] for b in data)} tokens; "
        f"held-out KL {hist[0]['kl']:.4f}")
    t0, step, ema = time.time(), 0, None
    student.train()
    while step < total:
        for ids, mask, w in data:
            loss, _ = _batch_kl(student, teacher, ids, mask, w, device)
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
    lin = [mod for name, mod in decoder_layers(m).named_modules() if isinstance(mod, nn.Linear)
           and mod.in_features % BLOCK == 0 and name.split(".")[-1] not in ATTENTION]
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
    data = trace_batches(train, a.length, a.batch, a.tokens, a.seed + 1)
    held_b = trace_batches(held, a.length, 1)[: a.held_windows]
    n_tok = sum(int(b[1].sum()) for b in data)
    log(f"e2e: {n_tok / 1e6:.1f}M training tokens in {len(data)} batches, {len(held_b)} held-out traces")
    hist = e2e_phase(student, teacher, data, held_b, lr=a.lr, max_minutes=a.max_minutes, log_every=a.log_every)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save(student, tok, out / "latent", snapped=False)
    save(student, tok, out / "snapped", snapped=True)
    check_ternary(out / "snapped")
    (out / "e2e.json").write_text(json.dumps({"kl": hist, "tokens": n_tok,
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
    kl = kl_to_teacher(student, teacher, trace_batches(held, a.length, 1)[: a.held_windows])
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
    g.add_argument("--tokens", type=float, default=8e6)
    g.add_argument("--samples", type=int, default=2)
    g.add_argument("--workers", type=int, default=SEQS["l40s"] // 2, help="requests in flight; x samples = --max-num-seqs")
    g.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    g.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    g.add_argument("--server-pid", type=int, default=None, help="the vLLM's pid, to tell a dead server from a slow one")
    g.add_argument("--seed", type=int, default=0)
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
        gen(a.model, a.url, a.problems, Path(a.out), a.tokens, a.samples, a.max_tokens, a.workers, a.seed,
            max_model_len=a.max_model_len, server_pid=a.server_pid)
    else:
        {"block": cmd_block, "e2e": cmd_e2e, "kl": cmd_kl}[a.cmd](a)


if __name__ == "__main__":
    main()
