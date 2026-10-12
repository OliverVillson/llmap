"""On-policy distillation (OPD): a small student learns from a bigger teacher that
shares its tokenizer, on the student's own answers.

Each round the student (served by vLLM with the current LoRA) thinks and answers a
batch of clean contest problems; the teacher (a second vLLM) reads every answer and
gives the log-probability of each token the student wrote (prompt_logprobs, one
prefill, no sampling); the student then takes one step that lowers the reverse KL
to the teacher on those tokens. Per token the signal is

    advantage = log p_teacher(token) - log p_student(token)

and the loss is -advantage * log p_student (advantage held fixed), the policy-gradient
form of reverse KL with no look-ahead (GKD, Agarwal et al. 2023; Thinking Machines
2025). The student is corrected where it actually goes wrong, on its own thinking,
which SFT on someone else's traces never sees. Answers cut by max_tokens train too:
every token has a target, so no answer has to finish.

The new LoRA is loaded into the student's vLLM after each step, so the next round
samples from it. Rollout pass rates (sandbox, the problem's tests) are logged as a
free check, never used in the loss. At the end the LoRA is merged and saved like
sft.py does.

    python -m stages.opd --student <hf dir> --student-url http://127.0.0.1:8093 \\
        --teacher-url http://127.0.0.1:8094 --problems <codebench dir> --out <dir>

The student vLLM must run with --enable-lora --max-lora-rank >= r and
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True; scripts/exp06.py starts both servers.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from stages.sft import ATTN, KD_CHUNK, NonFiniteTraining, _lora_model, check_finite, load_causal_lm, log, save_merged

TOKEN_ID = "token_id:"
STUDENT = "student"  # the base model's served name in the student vLLM (scripts/exp06.py)


# ---------- the two servers ----------

def _post(url: str, path: str, body: dict, timeout: float = 3600) -> dict:
    import httpx
    r = httpx.post(url.rstrip("/") + path, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.content else {}


def choice_tokens(choice: dict) -> list[int]:
    """The sampled token ids of one vLLM completion choice: its token_ids when the
    server returns them, else its logprobs tokens as "token_id:<n>"."""
    if choice.get("token_ids") is not None:
        return list(choice["token_ids"])
    toks = (choice.get("logprobs") or {}).get("tokens") or []
    if not all(t.startswith(TOKEN_ID) for t in toks):
        raise ValueError("vLLM returned text tokens, not ids; it needs return_tokens_as_token_ids")
    return [int(t[len(TOKEN_ID):]) for t in toks]


def sample(url: str, model: str, prompt: list[int], n: int, max_tokens: int,
           temperature: float, top_p: float) -> list[dict]:
    """n answers to one prompt (token ids in, token ids out), with why each stopped."""
    out = _post(url, "/v1/completions", {
        "model": model, "prompt": prompt, "n": n, "max_tokens": max_tokens, "temperature": temperature,
        "top_p": top_p, "logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True,
        "skip_special_tokens": False})
    return [{"tokens": choice_tokens(c), "finish": c.get("finish_reason")}
            for c in sorted(out["choices"], key=lambda c: c["index"])]


def prompt_token_logprobs(entries: list, ids: list[int], start: int) -> list[float]:
    """Log-probabilities of ids[start:] from vLLM's prompt_logprobs (one entry per prompt
    position, None for the first; each a {token id: {"logprob": x}} map that holds the
    actual token)."""
    out = []
    for i in range(start, len(ids)):
        e = entries[i] or {}
        v = e.get(str(ids[i]))
        if v is None:
            raise ValueError(f"teacher returned no logprob for token {ids[i]} at position {i}")
        out.append(float(v["logprob"] if isinstance(v, dict) else v))
    return out


def teacher_logprobs(url: str, model: str, prompt: list[int], answer: list[int]) -> list[float]:
    """The teacher's log p of each answer token given the prompt and the answer before it."""
    ids = prompt + answer
    out = _post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                         "prompt_logprobs": 0})
    return prompt_token_logprobs(out["choices"][0]["prompt_logprobs"], ids, len(prompt))


def load_adapter(url: str, name: str, path: Path, old: str | None) -> None:
    """Loads the saved LoRA into the student's vLLM under a new name, then drops the old one."""
    _post(url, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}, timeout=600)
    if old:
        _post(url, "/v1/unload_lora_adapter", {"lora_name": old}, timeout=600)


# ---------- the loss ----------

def token_logprobs(logits: torch.Tensor, targets: torch.Tensor, chunk: int = KD_CHUNK) -> torch.Tensor:
    """log p(target) per position, in fp32 chunks recomputed in the backward pass, so a
    long answer never holds a whole tokens x vocab fp32 softmax."""
    def part(l, t):
        return torch.log_softmax(l.float(), -1).gather(-1, t[:, None]).squeeze(-1)

    return torch.cat([checkpoint(part, logits[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
                      for i in range(0, logits.shape[0], chunk)])


def opd_loss(student_lp: torch.Tensor, teacher_lp: torch.Tensor, clip: float = 5.0) -> tuple[torch.Tensor, float]:
    """(sum over tokens of -advantage * log p_student, summed reverse KL estimate).
    The advantage, log p_teacher - log p_student, is held fixed and clipped to
    [-clip, clip] so one token the teacher finds absurd can't swamp the step."""
    adv = (teacher_lp - student_lp.detach()).clamp(-clip, clip)
    return -(adv * student_lp).sum(), float((student_lp.detach() - teacher_lp).sum())


# ---------- problems and rollouts ----------

def problems(bench_dir: str, limit: int = 0, seed: int = 0) -> list[dict]:
    """CodeContests problems with no 13-word run in common with LiveCodeBench (data.py's dedupe)."""
    from stages.data import contest_problems
    rows = contest_problems(SimpleNamespace(data_contest_source="codecontests", code_eval_dir=bench_dir))
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def render(tok, p: dict) -> list[int]:
    """The problem as the eval asks it, in the chat template with thinking on."""
    from stages.codebench import build_prompt
    msgs = [{"role": "user", "content": build_prompt(p)}]
    try:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=True))
    except TypeError:
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))


def passes(tok, items: list[tuple[dict, list[int]]]) -> list[bool]:
    """Whether each answer closed its thinking and passed the problem's tests."""
    from stages import codebench, sandbox
    from stages.data import split_think
    jobs, idx = [], []
    for i, (p, toks) in enumerate(items):
        text = tok.decode(toks, skip_special_tokens=True)
        if "</think>" not in text:
            continue
        _, answer = split_think(text)
        jobs.append((p["language"], *codebench.assemble(p, answer), codebench.time_limit(p)))
        idx.append(i)
    ok = [False] * len(items)
    for i, r in zip(idx, sandbox.run_many(jobs)):
        ok[i] = r.passed
    return ok


def rollouts(probs: list[dict], prompts: list[list[int]], student: tuple[str, str], teacher: tuple[str, str],
             n: int, max_tokens: int, temperature: float, top_p: float, workers: int = 32) -> list[dict]:
    """n student answers per problem, each scored by the teacher as soon as it exists;
    one thread per problem, so sampling and scoring overlap."""
    def one(i):
        out = []
        for a in sample(*student, prompts[i], n, max_tokens, temperature, top_p):
            if a["tokens"]:
                out.append({"problem": i, "prompt": prompts[i], **a,
                            "teacher": teacher_logprobs(*teacher, prompts[i], a["tokens"])})
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [r for rs in ex.map(one, range(len(probs))) for r in rs]


# ---------- training ----------

def train_opd(base: str, probs: list[dict], out_dir: Path, student_url: str, teacher_url: str, *,
              teacher_model: str = "teacher", rounds: int = 20, problems_per_round: int = 16, samples: int = 4,
              max_tokens: int = 8192, temperature: float = 1.0, top_p: float = 0.95, lr: float = 5e-5,
              r: int = 32, alpha: int = 64, clip: float = 5.0, seed: int = 0, max_minutes: float = 0.0,
              max_prompt_tokens: int = 4096) -> Path:
    """Runs OPD rounds on the student and saves the merged model to out_dir.
    work/opd-log.jsonl (beside out_dir) gets one line per round."""
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = load_causal_lm(base)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = _lora_model(model, r, alpha, ATTN, train_experts=True, train_router=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = max(1, rounds // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    device = next(model.parameters()).device
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    log(f"opd: {len(probs)} problems, {rounds} rounds of {problems_per_round} x {samples} answers, "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params")

    adapters = out_dir.parent / (out_dir.name + ".adapters")
    shutil.rmtree(adapters, ignore_errors=True)
    hist = out_dir.parent / "opd-log.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    name, t0, skipped = None, time.time(), 0
    order = list(range(len(probs)))
    rng.shuffle(order)
    for rnd in range(rounds):
        batch = [probs[order[(rnd * problems_per_round + j) % len(order)]] for j in range(problems_per_round)]
        prompts = [render(tok, p) for p in batch]
        # prompt + answer must fit the servers' context (scripts/exp06.py: 12288 = 4k + 8k)
        keep = [i for i, q in enumerate(prompts) if len(q) <= max_prompt_tokens]
        batch, prompts = [batch[i] for i in keep], [prompts[i] for i in keep]
        t_round = time.time()
        rolls = rollouts(batch, prompts, (student_url, name or STUDENT), (teacher_url, teacher_model),
                         samples, max_tokens, temperature, top_p)
        t_roll = time.time() - t_round
        ok = passes(tok, [(batch[x["problem"]], x["tokens"]) for x in rolls])
        n_tok = sum(len(x["tokens"]) for x in rolls)
        model.train()
        kl_sum, bad = 0.0, False
        for x in rolls:
            ids = torch.tensor([x["prompt"] + x["tokens"]], device=device)
            with amp:
                logits = model(input_ids=ids).logits[0, len(x["prompt"]) - 1:-1]
            s_lp = token_logprobs(logits, ids[0, len(x["prompt"]):])
            loss, kl = opd_loss(s_lp, torch.tensor(x["teacher"], device=device), clip)
            loss = loss / max(n_tok, 1)
            if not torch.isfinite(loss):
                bad = True
                continue
            loss.backward()
            kl_sum += kl
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad or not torch.isfinite(gnorm):
            skipped += 1
            log(f"opd: round {rnd + 1}: non-finite loss or gradient, update skipped")
            if skipped >= 3:
                raise NonFiniteTraining(f"OPD diverged: {skipped} non-finite rounds by round {rnd + 1}")
        else:
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        new = f"opd{rnd + 1}"
        model.save_pretrained(adapters / new)
        load_adapter(student_url, new, adapters / new, name)
        if name:
            shutil.rmtree(adapters / name, ignore_errors=True)
        name = new
        row = {"round": rnd + 1, "answers": len(rolls), "tokens": n_tok, "kl_per_token": kl_sum / max(n_tok, 1),
               "pass": sum(ok) / max(len(ok), 1), "cut": sum(x["finish"] == "length" for x in rolls) / max(len(rolls), 1),
               "grad_norm": float(gnorm), "rollout_s": round(t_roll), "round_s": round(time.time() - t_round)}
        with open(hist, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"opd: round {rnd + 1}/{rounds} kl/token {row['kl_per_token']:.3f} pass {row['pass']:.0%} "
            f"cut {row['cut']:.0%} {n_tok // 1000}k tokens {row['round_s']}s")
        if max_minutes and time.time() - t0 > 60 * max_minutes:
            log(f"opd: time cap of {max_minutes:g} min reached after round {rnd + 1}; stopping")
            break
    check_finite(model)
    save_merged(model, tok, out_dir)
    shutil.rmtree(adapters, ignore_errors=True)
    log(f"opd: saved merged model to {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student", required=True, help="the student's HF checkpoint dir")
    ap.add_argument("--student-url", required=True)
    ap.add_argument("--teacher-url", required=True)
    ap.add_argument("--problems", required=True, help="codebench dir with codecontests.jsonl (and livecodebench.jsonl for the dedupe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--problems-per-round", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probs = problems(a.problems, limit=a.rounds * a.problems_per_round, seed=a.seed)
    train_opd(a.student, probs, Path(a.out), a.student_url, a.teacher_url, rounds=a.rounds,
              problems_per_round=a.problems_per_round, samples=a.samples, max_tokens=a.max_tokens,
              lr=a.lr, r=a.r, alpha=2 * a.r, seed=a.seed, max_minutes=a.max_minutes)


if __name__ == "__main__":
    main()
