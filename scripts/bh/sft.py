#!/usr/bin/env python3
"""BH SFT: the student learns the engine's write and fix calls from the teacher's checked
answers. Runs in the training venv (torch, transformers); bh_run.py's `sft` stage and
round 2's RFT call it.

    python scripts/bh/sft.py --base <hf dir> --rows rows.jsonl [--rows more.jsonl] --out <dir>
        [--topk topk.jsonl --kd-weight 0.5] [--mode full|lora] [--probe probe.json] ...

  - Every row is one engine call (system, user, assistant); the prompt is masked, the loss
    is on the answer and its end of turn (rows.tokenize_row: the prompt rendered by the
    model's own template, thinking off, as the server renders it).
  - full: full fine-tune, fp32 master weights under bf16 autocast, AdamW. lora: LoRA on
    every projection of every layer, the Gated DeltaNet layers' too (Qwen3.5 has full
    attention in only 6 of 24 layers, so attention-only targets would leave 3 of 4
    layers untouched); the run fails if any decoder layer gets no adapter. Merged before
    saving either way, so the output is a plain HF checkpoint.
  - Logit KL (Qwen path: same tokenizer as the teacher): from the teacher's stored top-k
    log-probs per answer token (the recording proxy's topk.jsonl), KL(teacher || student)
    over the teacher's top k, the teacher renormalised over them and the student's full
    softmax. Rows whose answer ids are not the teacher's own (no log entry, or the ids do
    not decode to the answer) get CE only; the end of turn gets CE only.
  - Text-only load: Qwen3.5's checkpoints carry a vision tower; AutoModelForCausalLM
    loads the text decoder alone (stages/sft.py load_causal_lm), and only it is saved.
  - One sequence per micro-step (no padding: right padding through a recurrent layer is
    a bug class not worth having), gradient accumulation to --batch-tokens per update.
  - Resumable: a checkpoint every --save-minutes (weights and the step), picked up by a
    re-run with the same --out; the optimizer restarts its moments (said in the log).
  - Time cap: --max-minutes stops early, saves, and says so in sft.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: bh.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo: stages.*

from bh import rows as R  # noqa: E402
from bh.wire import call_key  # noqa: E402

STAGE = "sft"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                   "chat_template.json", "generation_config.json", "vocab.json", "merges.txt", "tokenizer.model")
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
          # Qwen3.5/3.6 Gated DeltaNet (linear attention) projections
          "in_proj_qkvz", "in_proj_qkv", "in_proj_z", "in_proj_ba", "in_proj_a", "in_proj_b", "out_proj")
CHUNK = 2048  # answer positions per logits chunk: 2048 x 248k fp32 is 2 GB, recomputed in backward


def emit(status: str = "running", pct: float | None = None, msg: str = "") -> None:
    e = {"stage": STAGE, "status": status, "ts": round(time.time(), 3)}
    if pct is not None:
        e["pct"] = round(max(0.0, min(100.0, pct)), 1)
    if msg:
        e["msg"] = msg
    print(json.dumps(e), flush=True)


def log(msg: str) -> None:
    print(f"sft: {msg}", flush=True)


# ---------- data ----------

def load_rows(paths: list[Path], thinking: bool) -> tuple[list[dict], dict]:
    raw = [json.loads(x) for p in paths for x in Path(p).read_text().splitlines() if x.strip()]
    keep, dropped = R.filter_rows(raw, thinking)
    return keep, {"rows_in": len(raw), "rows": len(keep), "dropped": dropped}


def load_topk(path: Path | None, keys: set[str]) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("key") in keys:
                out[r["key"]] = r
    return out


def examples(tok, rows: list[dict], topk: dict[str, dict], max_len: int, thinking: bool) -> tuple[list[dict], dict]:
    """Token ids, the answer's start, and per answer position the teacher's top-k (or
    None). Rows longer than max_len are dropped, never cut (an answer cut short teaches
    one that never ends)."""
    out, too_long, kd_rows, own = [], 0, 0, 0
    for r in rows:
        msgs = r["messages"]
        tk = topk.get(call_key(msgs[1]["content"], msgs[2]["content"]))
        t = R.tokenize_row(tok, msgs, thinking=thinking, answer_ids=tk["ids"] if tk else None)
        ids = t["prompt"] + t["answer"]
        if len(ids) > max_len:
            too_long += 1
            continue
        kd = None
        if tk and t["teacher_ids"] and len(tk["top"]) >= t["n_answer"]:
            kd = (tk["top"][:t["n_answer"]], tk["lp"][:t["n_answer"]])
            kd_rows += 1
        own += t["teacher_ids"]
        out.append({"ids": ids, "start": len(t["prompt"]), "kd": kd, "source": r.get("source", "")})
    return out, {"too_long": too_long, "kd_rows": kd_rows, "teacher_id_rows": own}


def groups(data: list[dict], batch_tokens: int, seed: int) -> list[list[int]]:
    """One epoch's updates: the rows shuffled, then taken in turn until each update holds
    batch_tokens tokens."""
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    out, cur, n = [], [], 0
    for i in idx:
        cur.append(i)
        n += len(data[i]["ids"])
        if n >= batch_tokens:
            out.append(cur)
            cur, n = [], 0
    if cur:
        out.append(cur)
    return out


# ---------- model ----------

def lora_targets(model) -> tuple[list[str], dict]:
    """The Linear leaf names LoRA goes on (every projection the model has of LINEAR), and
    how many decoder layers that reaches. Raises when a layer would get none: that is how
    attention-only targets silently skip a hybrid model's DeltaNet layers."""
    import torch.nn as nn
    present = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    targets = [t for t in LINEAR if t in present]
    layers = decoder_layers(model)
    reached = [any(isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets for n, m in layer.named_modules())
               for layer in layers]
    if not layers or not all(reached):
        missing = [i for i, ok in enumerate(reached) if not ok]
        raise RuntimeError(f"LoRA targets {targets} reach no projection in decoder layers {missing}; "
                           f"Linear names here: {sorted(present)}")
    return targets, {"targets": targets, "layers": len(layers)}


def decoder_layers(model) -> list:
    import torch.nn as nn
    for _, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > 1 and type(m[0]).__name__.endswith(("DecoderLayer", "Layer")):
            return list(m)
    return []


def parts(model):
    """(the decoder that gives hidden states, the output head), through a PEFT wrapper."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    dec = m.get_decoder() if hasattr(m, "get_decoder") else m.model
    return dec, m.get_output_embeddings()


def answer_loss(hidden, head, targets, kd, kd_weight: float):
    """CE over the answer positions (sum) and the top-k KL over those that have the
    teacher's (sum), each chunk recomputed in backward so fp32 logits never pile up."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    n = hidden.shape[0]
    ce_total = hidden.new_zeros((), dtype=torch.float32)
    kl_total = hidden.new_zeros((), dtype=torch.float32)
    kl_n = 0

    def chunk_ce(h, t):
        return F.cross_entropy(head(h).float(), t, reduction="sum")

    def chunk_kl(h, ids, lp):
        ls = F.log_softmax(head(h).float(), -1).gather(-1, ids)
        lt = F.log_softmax(lp, -1)  # renormalised over the teacher's top k
        return (lt.exp() * (lt - ls)).sum()

    for i in range(0, n, CHUNK):
        ce_total = ce_total + checkpoint(chunk_ce, hidden[i:i + CHUNK], targets[i:i + CHUNK], use_reentrant=False)
    if kd is not None and kd_weight > 0:
        ids, lp, pos = kd
        for i in range(0, len(pos), CHUNK):
            p = pos[i:i + CHUNK]
            kl_total = kl_total + checkpoint(chunk_kl, hidden[p], ids[i:i + CHUNK], lp[i:i + CHUNK],
                                             use_reentrant=False)
        kl_n = len(pos)
    return ce_total, kl_total, kl_n


def kd_tensors(ex: dict, device):
    """The example's teacher top-k as (ids [N,k], log-probs [N,k], answer positions [N])
    over positions whose k is full (a short list is padded with its own last entry at
    -inf, which renormalising ignores)."""
    import torch
    if not ex["kd"]:
        return None
    top, lps = ex["kd"]
    k = max((len(t) for t in top), default=0)
    if not k:
        return None
    ids, lp, pos = [], [], []
    for j, (t, l) in enumerate(zip(top, lps)):
        if not t:
            continue
        ids.append(t + [t[-1]] * (k - len(t)))
        lp.append(l + [-1e9] * (k - len(l)))
        pos.append(j)
    if not pos:
        return None
    return (torch.tensor(ids, device=device), torch.tensor(lp, device=device, dtype=torch.float32),
            torch.tensor(pos, device=device))


def save(model, tok, base: Path, out: Path, mode: str) -> None:
    import torch
    m = model.merge_and_unload() if mode == "lora" else model
    m = m.to(torch.bfloat16)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    m.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    for f in TOKENIZER_FILES:  # the base's own files: the converter and servers read exactly these
        if (base / f).exists():
            shutil.copyfile(base / f, tmp / f)
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)


# ---------- training ----------

def train(a) -> dict:
    import torch
    from transformers import AutoTokenizer
    from stages.sft import load_causal_lm

    t_start = time.time()
    out = Path(a.out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows, rstats = load_rows([Path(p) for p in a.rows], a.thinking)
    if not rows:
        raise SystemExit("sft: no trainable rows after the guards: " + json.dumps(rstats))
    probe = json.loads(Path(a.probe).read_text()) if a.probe and Path(a.probe).exists() else {}
    served = probe.get("prompt_ids") if probe.get("tokenizer") == "same" else None
    rt = R.roundtrip(tok, probe.get("row") or rows[0], a.thinking, served)
    log(f"tokenizer round-trip on one real row: ok ({rt['prompt_tokens']} prompt, {rt['answer_tokens']} answer tokens"
        + (", prompt ids equal the teacher server's" if served else "") + ")")
    keys = {call_key(r["messages"][1]["content"], r["messages"][2]["content"]) for r in rows}
    topk = load_topk(Path(a.topk) if a.topk else None, keys) if a.kd_weight > 0 else {}
    data, estats = examples(tok, rows, topk, a.max_len, a.thinking)
    log(f"{len(data)} rows ({rstats['rows_in']} in; dropped {rstats['dropped']}; {estats['too_long']} over "
        f"{a.max_len} tokens); logit KL on {estats['kd_rows']} rows")
    if not data:
        raise SystemExit("sft: every row is longer than --max-len")

    emit(pct=1, msg=f"loading {a.base}")
    full = a.mode == "full"
    model = load_causal_lm(a.base)
    arch = (getattr(model.config, "architectures", None) or [type(model).__name__])[0]
    if full and torch.cuda.is_available():
        model = model.float()  # fp32 master weights; the forward runs under bf16 autocast
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lstats = {}
    if not full:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        targets, lstats = lora_targets(model)
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
        log(f"LoRA r={a.lora_r} on {targets} in all {lstats['layers']} decoder layers")
    ckpt = work / "ckpt.pt"
    step0 = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        step0 = state["step"]
        log(f"resumed at update {step0} from {ckpt} (optimizer moments restart)")
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    plan = [g for e in range(math.ceil(a.epochs)) for g in groups(data, a.batch_tokens, a.seed + e)]
    total = max(1, round(len(plan) * a.epochs / math.ceil(a.epochs)))
    warm = max(1, int(0.03 * total))
    lr_at = lambda s: a.lr * ((s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    dec, head = parts(model)
    model.train()
    t0, last_save, ema_ce, ema_kl, skipped, stopped = time.time(), time.time(), None, None, 0, ""
    step = step0
    for step in range(step0, total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        group = plan[step]
        n_tok = sum(len(data[i]["ids"]) - data[i]["start"] for i in group)
        ce_sum = kl_sum = 0.0
        bad = False
        for i in group:
            ex = data[i]
            ids = torch.tensor([ex["ids"]], device=device)
            with amp:
                hidden = dec(input_ids=ids).last_hidden_state[0, :-1]
            s = ex["start"] - 1
            ans_h, tgt = hidden[s:], ids[0, ex["start"]:]
            kd = kd_tensors(ex, device)
            with amp:
                ce, kl, kl_n = answer_loss(ans_h, head, tgt, kd, a.kd_weight)
            loss = ce / n_tok
            if kl_n:
                loss = (1 - a.kd_weight) * loss + a.kd_weight * kl / n_tok
            if not torch.isfinite(loss):
                bad = True
                continue
            loss.backward()
            ce_sum += float(ce)
            kl_sum += float(kl)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad or not torch.isfinite(gnorm):
            skipped += 1
            log(f"update {step + 1}: non-finite loss or gradient, skipped")
            if skipped > max(8, total // 20):
                raise RuntimeError(f"training diverged: {skipped} non-finite updates by {step + 1}/{total}")
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        ce_t = ce_sum / max(1, n_tok)
        ema_ce = ce_t if ema_ce is None else 0.95 * ema_ce + 0.05 * ce_t
        ema_kl = (kl_sum / max(1, n_tok)) if ema_kl is None else 0.95 * ema_kl + 0.05 * kl_sum / max(1, n_tok)
        done = step + 1
        if done % max(1, total // 100) == 0 or done == total:
            el = time.time() - t0
            emit(pct=3 + 92 * done / total, msg=f"update {done}/{total} ce {ema_ce:.3f} kl {ema_kl:.3f} "
                                                f"lr {lr_at(step):.1e} eta {el / max(1, done - step0) * (total - done) / 60:.0f}m")
        if a.save_minutes and time.time() - last_save > 60 * a.save_minutes and done < total:
            torch.save({"model": {k: v for k, v in model.state_dict().items()
                                  if full or "lora_" in k}, "step": done}, work / "ckpt.tmp")
            (work / "ckpt.tmp").replace(ckpt)
            last_save = time.time()
            log(f"checkpoint at update {done}")
        if a.max_minutes and time.time() - t_start > 60 * a.max_minutes:
            stopped = f"time cap of {a.max_minutes:g} min at update {done}/{total}"
            log(stopped)
            break
    emit(pct=96, msg="saving")
    save(model, tok, Path(a.base), out, a.mode)
    rep = {"base": a.base, "arch": arch, "mode": a.mode, "updates": step + 1, "planned": total, "stopped": stopped,
           "ce": ema_ce, "kl": ema_kl, "skipped": skipped, "minutes": round((time.time() - t_start) / 60, 1),
           "rows": rstats, "examples": estats, "lora": lstats, "roundtrip": rt,
           "kd_weight": a.kd_weight if estats["kd_rows"] else 0.0}
    (out / "sft.json").write_text(json.dumps(rep, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    emit("done", 100, f"{len(data)} rows, {step + 1} updates, ce {ema_ce:.3f}")
    return rep


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BH SFT on engine calls")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", default="")
    ap.add_argument("--probe", default="", help="teacher-serve's probe.json: one row and the server's prompt ids")
    ap.add_argument("--mode", choices=("full", "lora"), default="full")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--kd-weight", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--save-minutes", type=float, default=20.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
