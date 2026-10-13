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

The request side here is shared with the training text (stages/ternary.py gen) and
the QAT on-policy rounds, so every sampled answer is asked the way the eval asks:
render() is the eval's messages (codebench.SYSTEM, then the problem) in the chat
template with thinking on, sample() sends the eval's sampling fields explicitly, and
continue_forced() adds the forced answer eval.force_answer adds when a thinking runs
out of budget. Requests share one HTTP client and are retried on timeouts, dropped
connections and 5xx, but not once the server is dead.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from stages.sft import ATTN, KD_CHUNK, NonFiniteTraining, _lora_model, check_finite, load_causal_lm, log, save_merged

TOKEN_ID = "token_id:"
STUDENT = "student"  # the base model's served name in the student vLLM (scripts/exp06.py)
# The eval's sampling (the record harness): temperature 1, top_p 0.95, top_k off (vLLM
# reads 0 as all tokens), no min_p, no repetition penalty. Sent on every request, so
# no default from the served model's generation_config can leak in.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
ANSWER_TOKENS = 4096  # eval.ANSWER_TOKENS: the forced answer's cap
# Functional (LeetCode-style) problems in the training pool: LiveCodeBench scores both
# stdin programs and starter-code functions, and CodeContests only has the first.
FUNCTIONAL_SHARE = 0.2

# A request waits behind a full server and then writes up to ~20k tokens: the read
# timeout covers that; connect and write are short, and a free connection is waited for.
READ_TIMEOUT = 7200.0
RETRIES = 3  # after the first try, on timeouts, broken connections and 5xx
BACKOFF = 5.0  # seconds before the first retry, doubling
HEALTH_WAIT = 10.0  # seconds between the /health tries that decide a server is dead


# ---------- the servers ----------

class ServerDead(RuntimeError):
    """The server's process exited or /health stopped answering: retrying cannot help."""


_http: dict = {"client": None, "size": 0}
_http_lock = threading.Lock()
_pids: dict[str, int] = {}
_tok_lock = threading.Lock()  # one fast tokenizer is not safe to share between threads


def client(workers: int | None = None):
    """The HTTP client every request shares, made on first use; its pool holds `workers`
    requests at once (default 128) plus 8 for health checks and the like, and grows when
    a caller asks for more. httpx's default pool of 100 connections would hold back 128
    threads and fail them with PoolTimeout; pool=None waits for a connection instead."""
    import httpx
    with _http_lock:
        size = (workers or 128) + 8
        if _http["client"] is None or workers and _http["size"] < size:
            _http["client"] = httpx.Client(
                timeout=httpx.Timeout(connect=30, write=60, pool=None, read=READ_TIMEOUT),
                limits=httpx.Limits(max_connections=size, max_keepalive_connections=size - 8))
            _http["size"] = size
        return _http["client"]


def watch(url: str, pid: int | None) -> None:
    """Tells alive() the server at url runs as process pid (gen gets it from its shell)."""
    if pid:
        _pids[url.rstrip("/")] = pid


def alive(url: str) -> bool:
    """False when the server at url is gone: its watched process has exited, or /health
    failed three times in a row, HEALTH_WAIT apart."""
    import httpx
    pid = _pids.get(url.rstrip("/"))
    if pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
    for i in range(3):
        try:
            if client().get(url.rstrip("/") + "/health", timeout=30).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if i < 2:
            time.sleep(HEALTH_WAIT)
    return False


def _post(url: str, path: str, body: dict, timeout: float | None = None) -> dict:
    """POSTs body as JSON and returns the reply. Timeouts, broken connections and 5xx
    are retried RETRIES times with backoff while the server is alive; once it is dead
    this raises ServerDead, and a 4xx raises at once (the request itself is wrong)."""
    import httpx
    kw = {"timeout": httpx.Timeout(connect=30, write=60, pool=None, read=timeout)} if timeout else {}
    for attempt in range(RETRIES + 1):
        try:
            r = client().post(url.rstrip("/") + path, json=body, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json() if r.content else {}
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except httpx.TransportError as e:  # timeouts, refused and dropped connections
            err = f"{type(e).__name__}: {e}"
        if not alive(url):
            raise ServerDead(f"{url} is down ({path}: {err})")
        if attempt == RETRIES:
            raise RuntimeError(f"{path} failed {RETRIES + 1} times; last: {err}")
        log(f"opd: {path} failed ({err}); retry {attempt + 1}/{RETRIES}")
        time.sleep(BACKOFF * 2 ** attempt)


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
           temperature: float = SAMPLING["temperature"], top_p: float = SAMPLING["top_p"]) -> list[dict]:
    """n answers to one prompt (token ids in, token ids out), with why each stopped,
    sampled as the eval samples (SAMPLING)."""
    if not prompt or not all(type(t) is int for t in prompt):
        raise ValueError(f"prompt must be token ids, not {prompt[:4]!r}")
    out = _post(url, "/v1/completions", {
        "model": model, "prompt": prompt, "n": n, "max_tokens": max_tokens,
        **SAMPLING, "temperature": temperature, "top_p": top_p,
        "logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True, "skip_special_tokens": False})
    choices = sorted(out["choices"], key=lambda c: c["index"])
    if len(choices) != n:  # e.g. a list of strings read as a batch of text prompts
        raise ValueError(f"asked for {n} answers, got {len(choices)}")
    return [{"tokens": choice_tokens(c), "finish": c.get("finish_reason")} for c in choices]


def _tokenize(url: str, model: str, text: str) -> list[int]:
    return _post(url, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]


_forced_ids: dict[tuple, list[int]] = {}


def continue_forced(url: str, model: str, prompt_ids: list[int], toks: list[int],
                    max_answer: int = ANSWER_TOKENS, *, why: str = "length") -> list[int]:
    """What eval.force_answer adds to a thinking that ended without an answer, as token
    ids to append to toks: Qwen's early-stop line (eval.EARLY_STOP; only when the
    budget ran out, why="length"), the template's closing thinking tag, then the
    model's answer on top of that, up to max_answer tokens, sampled as the eval does.
    toks is the thinking as open_thinking() leaves it (no end-of-text, no closing tag).
    The line and the tag are tokenized by the server, as eval's text prompt is; the
    prompt, toks, both and max_answer must fit the server's --max-model-len."""
    from stages.eval import EARLY_STOP
    from stages.taskdata import think_tags
    key = (url, model, why)
    if key not in _forced_ids:
        template = _post(url, "/detokenize", {"model": model, "tokens": prompt_ids})["prompt"]
        _forced_ids[key] = _tokenize(url, model, (EARLY_STOP if why == "length" else "") + think_tags(template)[1])
    close = _forced_ids[key]
    out = _post(url, "/v1/completions", {
        "model": model, "prompt": prompt_ids + toks + close, "n": 1, "max_tokens": max_answer, **SAMPLING,
        "logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True, "skip_special_tokens": False})
    return close + choice_tokens(out["choices"][0])


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

def _functional(bench_dir: str) -> list[dict]:
    """apps.jsonl's LeetCode-style problems (APPS/TACO call-based, in LiveCodeBench's
    functional format), with the same length cut and LiveCodeBench dedupe as CodeContests;
    [] when the file is not there."""
    from stages import codebench
    from stages.data import MAX_INPUT_CHARS, _shingles
    if not codebench.suite_path("apps", bench_dir).exists():
        return []
    rows = [p for p in codebench.load_suite("apps", bench_dir) if isinstance(p.get("tests"), dict)]
    lcb = codebench.suite_path("livecodebench", bench_dir)
    seen = set().union(*(_shingles(p["prompt"]) for p in codebench.load_suite("livecodebench", bench_dir,
                                                                             fields=("id", "prompt")))
                       ) if lcb.exists() else set()
    return [p for p in rows if len(p["prompt"]) <= MAX_INPUT_CHARS and not (_shingles(p["prompt"]) & seen)]


def problems(bench_dir: str, limit: int = 0, seed: int = 0, functional_share: float = FUNCTIONAL_SHARE) -> list[dict]:
    """CodeContests problems with no 13-word run in common with LiveCodeBench (data.py's
    dedupe), plus functional problems from apps.jsonl (same dedupe) as about
    functional_share of the pool, shuffled by seed; each id once."""
    from stages.data import contest_problems
    rows = contest_problems(SimpleNamespace(data_contest_source="codecontests", code_eval_dir=bench_dir))
    rng = random.Random(seed)
    fn = _functional(bench_dir) if functional_share > 0 else []
    rng.shuffle(fn)
    fn = fn[: int(len(rows) * functional_share / (1 - functional_share))]
    log(f"opd: {len(rows)} CodeContests and {len(fn)} functional problems"
        + ("" if fn or not functional_share else " (no apps.jsonl: stdin programs only)"))
    rows = list({p["id"]: p for p in rows + fn}.values())
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def messages(p: dict) -> list[dict]:
    """The eval's messages for a problem: its system prompt, then the problem."""
    from stages.codebench import SYSTEM, build_prompt
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(p)}]


def render(tok, p: dict) -> list[int]:
    """The problem as the eval asks it (messages()), in the chat template with thinking
    on, as token ids. transformers 5 returns a BatchEncoding unless return_dict=False,
    and list() of that is its keys; anything but a list of ints raises."""
    kw = {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
    with _tok_lock:
        try:
            ids = tok.apply_chat_template(messages(p), enable_thinking=True, **kw)
        except TypeError:
            ids = tok.apply_chat_template(messages(p), **kw)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids)
    if len(ids) <= 20 or not all(type(t) is int for t in ids):
        raise ValueError(f"render() of {p.get('id')} gave {ids[:4]!r} ({len(ids)} items), not token ids")
    return ids


def render_remote(url: str, model: str, p: dict) -> list[int]:
    """The ids the server's own chat template gives the same messages (vLLM's /tokenize,
    as /v1/chat/completions renders them for the eval)."""
    return _post(url, "/tokenize", {"model": model, "messages": messages(p), "add_generation_prompt": True,
                                    "add_special_tokens": False,
                                    "chat_template_kwargs": {"enable_thinking": True}})["tokens"]


def answer_text(tok, toks: list[int]) -> str | None:
    """The text after the thinking closed, without special tokens; None when the
    thinking never closed. Decoded with special tokens kept, so a closing tag that is a
    special token is still seen."""
    from stages.data import split_think
    with _tok_lock:
        text = tok.decode(toks, skip_special_tokens=False)
        special = [s for s in tok.all_special_tokens if s]
    if "</think>" not in text and "<channel|>" not in text:
        return None
    answer = split_think(text)[1]
    for s in special:
        answer = answer.replace(s, "")
    return answer


def forced_reason(tok, toks: list[int], finish: str | None) -> str | None:
    """Why the eval would force an answer onto this answer (it does when no answer text
    follows the thinking): "length" (out of budget) or "stop" (it ended inside its
    thinking). None when an answer follows the closed thinking, even one cut by the cap:
    the eval counts that one, it does not continue it."""
    answer = answer_text(tok, toks)
    if answer is not None and answer.strip():
        return None
    return "length" if finish == "length" else "stop"


def open_thinking(tok, toks: list[int]) -> list[int]:
    """toks cut back to the thinking, for continue_forced: without a trailing
    end-of-text, and without a closing tag that no answer followed (the eval closes
    the thinking again its own way)."""
    def closes(t):
        with _tok_lock:
            text = tok.decode(t, skip_special_tokens=False)
        return text.count("</think>") + text.count("<channel|>")

    toks, special = list(toks), set(tok.all_special_ids)
    while toks and toks[-1] in special and not closes(toks[-1:]):
        toks.pop()
    n = closes(toks)
    while n and toks and closes(toks) == n:
        toks.pop()
    return toks


def passes(tok, items: list[tuple[dict, list[int]]]) -> list[bool]:
    """Whether each answer closed its thinking and passed the problem's tests (the
    eval's grading: codebench.job, then the sandbox)."""
    from stages import codebench, sandbox
    jobs, idx = [], []
    for i, (p, toks) in enumerate(items):
        answer = answer_text(tok, toks)
        if answer is None:
            continue
        jobs.append(codebench.job(p, answer))
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
