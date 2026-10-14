"""Code scores at shorter thinking budgets, from one long run.

A thinking budget of B tokens means: think for at most B tokens, then close the
thinking with Qwen's early-stop line and answer, which is what stages/eval.py does
when an answer runs out of max_tokens. So one run at the longest budget, with each
answer's whole thinking kept (Config.code_eval_budgets), gives every shorter one:

  same       the answer fit in B tokens, or the model stopped inside its thinking
             before B: the long run's result stands
  cut        the thinking reached B tokens: it is cut there and the model answers
  truncated  the thinking ended before B but the answer ran past it: the answer is
             cut at B, as a capped run would have returned it

Writes work/code_eval/<name>-<B>.jsonl per budget and adds code.budgets to
out/eval.json: {"<B>": {"mean_pass@1", "suites", "how": {same, cut, truncated},
"errors"}}, with the long run itself under its own budget, scored on the same
answers ("how" there counts the answers whose thinking hit the cap, and
"all_samples_mean_pass@1" is the eval's own number over every sample, see
long_run). An answer the server never returned is tried
once more, and if it still fails the step stops rather than score it ("errors" is
0 in every written result); so does a server that died. Redone answers are kept
in work/code_eval/<name>-<B>.redo.jsonl as they come, keyed by the long run's
answer they were made from, so a re-run after a stop only redoes the rest.

It covers the first answer to each problem (sample 0, seeded
code_eval_seed_offset) of any eval that kept its thinking: set code_eval_budgets
(e.g. [8192], the model card's cap) in every eval job, whatever its window, ids or
answers per problem. Each candidate is served again (the forced answers need the
GPU), so this is a GPU step; thinking is cut in tokens of the candidate's own HF
tokenizer when vLLM serves it, and by llama-server's /tokenize for a GGUF.

    python -m stages.budget --job <job>    # after the eval stage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common.progress import emit
from stages import codebench, sandbox
from stages import eval as ev
from stages._util import DRY_RUN, Job, read_jsonl, write_jsonl

STAGE = "budgets"
CLOSE_TOKENS = 2  # "\n</think>\n\n" between the thinking and the answer


def classify(row: dict, reasoning_tokens: int, budget: int) -> str:
    """What a run capped at `budget` would have done with this answer."""
    if row.get("forced") == "no_reasoning":
        return "same"
    if reasoning_tokens >= budget:
        return "cut"
    if row.get("forced") or (row.get("tokens") or 0) <= budget:
        return "same"
    return "truncated"


def tokenize(text: str) -> list[int]:
    """text's token ids by the served candidate's tokenizer: its own HF tokenizer when
    vLLM serves a checkpoint dir, else llama-server's /tokenize."""
    if not text:
        return []
    if ev.VLLM:
        tok = ev.tokenizer(ev.VLLM["dir"])
        with ev.tokenizer_lock:
            return tok.encode(text, add_special_tokens=False)
    return ev.request("/tokenize", {"content": text})["tokens"]


def detokenize(tokens: list[int]) -> str:
    if not tokens:
        return ""
    if ev.VLLM:
        tok = ev.tokenizer(ev.VLLM["dir"])
        with ev.tokenizer_lock:
            return tok.decode(tokens)
    return ev.request("/detokenize", {"tokens": tokens})["content"]


def rescore(job: Job, name: str, rows: list[dict], problems: list[dict], budgets: list[int],
            release=None) -> dict:
    """{str(B): scores} for each budget; needs this candidate's server running until every
    budget's answers are redone, then calls release() (run_stage stops the server there)
    before the sandbox runs, as the eval does."""
    cfg = job.config
    from stages.data import system_prompt

    system_of, prompt, code_of = ev.prompting(cfg, system_prompt(job.spec))
    sampling = ev.sampling_params(cfg.code_eval_seed_offset, True)  # sample 0's seed
    with ThreadPoolExecutor(ev.workers()) as ex:
        toks = list(ex.map(lambda r: tokenize(r.get("reasoning") or ""), rows))
    how = {b: [classify(r, len(t), b) for r, t in zip(rows, toks)] for b in budgets}
    total = sum(h != "same" for b in budgets for h in how[b])
    done, lock = 0, threading.Lock()
    job.path("work", "code_eval").mkdir(exist_ok=True)
    kept = {b: job.path("work", "code_eval", f"{name}-{b}.redo.jsonl") for b in budgets}
    cached = {b: {x["key"]: x["answer"] for x in read_jsonl(f)} if f.exists() else {} for b, f in kept.items()}
    errors = dict.fromkeys(budgets, 0)
    emit(STAGE, msg=f"{name}: {total} answers to redo across {len(budgets)} budgets"
                    + (f", {sum(map(len, cached.values()))} kept from before" if any(cached.values()) else ""),
         answered=0, total=max(total, 1))

    def answer_at(b: int, j: int) -> str:
        r, p = rows[j], problems[j]
        if how[b][j] == "cut":
            messages = [{"role": "system", "content": system_of(p)}, {"role": "user", "content": prompt(p)}]
            answer, _ = ev.force_answer(messages, detokenize(toks[j][:b]), "length",
                                        cfg.code_eval_thinking_temperature, sampling)
            return answer
        return detokenize(tokenize(r.get("answer") or "")[:max(0, b - len(toks[j]) - CLOSE_TOKENS)])

    def key_of(r: dict) -> str:
        """The long run's answer a redone one is made from: its problem and a hash of its
        thinking and answer, so a re-run of the eval never reuses answers redone before it."""
        made = hashlib.sha256(f"{r.get('reasoning') or ''}\0{r.get('answer') or ''}".encode()).hexdigest()[:16]
        return f"{r['suite']}|{r['id']}|{made}"

    def redo(b: int, j: int) -> str:
        nonlocal done
        key = key_of(rows[j])
        answer = cached[b].get(key)
        for attempt in range(2 if answer is None else 0):
            try:
                answer = answer_at(b, j)
                break
            except ev.ServerDead:
                raise
            except Exception as e:  # e.g. a 5xx under load: once more, then the step stops
                print(f"[{STAGE}] {key} at {b} tokens, try {attempt + 1}: {e}", flush=True)
        with lock:
            if answer is None:
                errors[b] += 1
            elif key not in cached[b]:
                with open(kept[b], "a") as f:
                    f.write(json.dumps({"key": key, "answer": answer}) + "\n")
            done += 1
            emit(STAGE, msg=f"{name}: {done}/{total} redone", answered=done, total=total)
        return answer or ""

    redone: dict[int, dict[int, str]] = {}
    for b in budgets:
        todo = [j for j, h in enumerate(how[b]) if h != "same"]
        with ThreadPoolExecutor(ev.workers()) as ex:
            redone[b] = dict(zip(todo, ex.map(lambda j: redo(b, j), todo)))
    if any(errors.values()):
        raise RuntimeError(f"{name}: {sum(errors.values())} answers failed twice (by budget: {errors}); the others "
                           "redone are kept, and a re-run asks only for the rest")
    if release:
        release()
    out = {}
    for b in budgets:
        todo = list(redone[b])
        results = [sandbox.Result(bool(r.get("passed")), r.get("reason") or "", r.get("output") or "") for r in rows]
        by_timeout: dict[float, list[int]] = {}
        for j in todo:
            by_timeout.setdefault(ev.SUITE_TIMEOUT.get(problems[j]["suite"], sandbox.TIMEOUT), []).append(j)
        for timeout, idx in by_timeout.items():
            items = [codebench.job(problems[j], redone[b][j], code_of(problems[j], redone[b][j])) for j in idx]
            for j, res in zip(idx, sandbox.run_many(items, timeout=timeout)):
                results[j] = res
        write_jsonl(job.path("work", "code_eval", f"{name}-{b}.jsonl"),
                    [{"suite": r["suite"], "id": r["id"], "how": how[b][j], "passed": results[j].passed,
                      "reason": results[j].reason, **({"answer": redone[b][j]} if j in redone[b] else {})}
                     for j, r in enumerate(rows)])
        suites = codebench.summarize(problems, [[x] for x in results], [1])
        out[str(b)] = {"mean_pass@1": codebench.mean_pass1(suites), "suites": suites, "how": dict(Counter(how[b])),
                       "errors": errors[b]}
    return out


def long_run(code: dict, rows: list[dict], problems: list[dict]) -> dict:
    """The entry for the long run's own budget, scored on the same answers as the
    shorter budgets (sample 0 of each problem, as rescore covers): its mean_pass@1 is
    those answers' own results, so the gap to a shorter budget is the budget's effect
    alone, not that plus sample 0's draw against the per-problem mean over every
    sample. The eval's number over every sample stays beside it as
    all_samples_mean_pass@1. A dry run has no rows and keeps the eval's number."""
    if rows:
        suites = codebench.summarize(problems, [[sandbox.Result(bool(r.get("passed")), r.get("reason") or "")]
                                                for r in rows], [1])
        mean = codebench.mean_pass1(suites)
    else:
        suites, mean = code.get("suites"), code.get("mean_pass@1")
    return {"mean_pass@1": mean, "suites": suites, "all_samples_mean_pass@1": code.get("mean_pass@1"),
            "how": {"hit the cap": sum(r.get("forced") == "length" for r in rows)},
            "errors": sum(bool(r.get("error")) for r in rows)}


def run_stage(job: Job) -> None:
    cfg = job.config
    budgets = sorted(b for b in set(cfg.code_eval_budgets) if b < cfg.code_eval_max_tokens)
    report_path = job.path("out", "eval.json")
    if not report_path.exists():
        raise RuntimeError("run the eval stage first")
    report = json.loads(report_path.read_text())
    held_path = job.path("data", "heldout.jsonl")
    by_id = {(p["suite"], p["id"]): p for p in ev.code_problems(job, read_jsonl(held_path) if held_path.exists() else [])}
    cands = ev.candidates_to_eval(job)
    for i, (name, c) in enumerate(cands.items()):
        cand = next(x for x in report["candidates"] if x["name"] == name)
        code = cand.get("code") or {}
        if DRY_RUN:
            top = code.get("mean_pass@1") or 0.0
            scores = {str(b): {"mean_pass@1": round(top * (0.8 + 0.2 * b / cfg.code_eval_max_tokens), 4),
                               "how": {"same": 150, "cut": 20, "truncated": 2}} for b in budgets}
            rows, problems = [], []
        else:
            rows = [r for r in read_jsonl(job.path("work", "code_eval", f"{name}.jsonl")) if r.get("sample", 0) == 0]
            if any(r.get("reasoning_chars") and "reasoning" not in r for r in rows):
                raise RuntimeError(f"{name}: the eval kept no thinking; set code_eval_budgets before the eval")
            problems = [by_id[(r["suite"], r["id"])] for r in rows]
            emit(STAGE, pct=100 * i / len(cands), msg=f"{name}: thinking cut at {', '.join(map(str, budgets))} tokens")
            proc = ev.serve(job, c["path"], f"{name}-budgets")
            try:
                scores = rescore(job, name, rows, problems, budgets, release=lambda: ev.stop(proc))
            finally:
                ev.stop(proc)
        scores[str(cfg.code_eval_max_tokens)] = long_run(code, rows, problems)
        code["budgets"] = scores
        cand["code"] = code
    report_path.write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {name: {b: s["mean_pass@1"] for b, s in c["code"]["budgets"].items()}
                          for name, c in ((x["name"], x) for x in report["candidates"]) if c.get("code")})
    emit(STAGE, "done", 100, "; ".join(
        f"{x['name']}: " + ", ".join(f"{int(b) // 1024}k {s['mean_pass@1']:.0%}" for b, s in
                                     sorted(x["code"]["budgets"].items(), key=lambda kv: int(kv[0]))
                                     if s.get("mean_pass@1") is not None)
        for x in report["candidates"] if (x.get("code") or {}).get("budgets")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ev.raise_on_sigterm()
    run_stage(Job(ap.parse_args().job))
