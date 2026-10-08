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
out/eval.json: {"<B>": {"mean_pass@1", "suites", "how": {same, cut, truncated}}},
with the long run itself under its own budget ("how" there counts the answers
whose thinking hit the cap).

    python -m stages.budget --job <job>    # after the eval stage
"""

from __future__ import annotations

import argparse
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


def _post(path: str, body: dict) -> dict:
    import httpx

    r = httpx.post(f"http://127.0.0.1:{ev.PORT}{path}", json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def tokenize(text: str) -> list[int]:
    return _post("/tokenize", {"content": text})["tokens"] if text else []


def detokenize(tokens: list[int]) -> str:
    return _post("/detokenize", {"tokens": tokens})["content"] if tokens else ""


def rescore(job: Job, name: str, rows: list[dict], problems: list[dict], budgets: list[int]) -> dict:
    """{str(B): scores} for each budget; needs this candidate's llama-server running."""
    cfg = job.config
    from stages.data import system_prompt

    system_of, prompt, code_of = ev.prompting(cfg, system_prompt(job.spec))
    sampling = ev.sampling_params(0, True)
    with ThreadPoolExecutor(ev.SLOTS) as ex:
        toks = list(ex.map(lambda r: tokenize(r.get("reasoning") or ""), rows))
    how = {b: [classify(r, len(t), b) for r, t in zip(rows, toks)] for b in budgets}
    total = sum(h != "same" for b in budgets for h in how[b])
    done, lock = 0, threading.Lock()
    emit(STAGE, msg=f"{name}: {total} answers to redo across {len(budgets)} budgets", answered=0, total=max(total, 1))

    def redo(b: int, j: int) -> str:
        nonlocal done
        r, p = rows[j], problems[j]
        if how[b][j] == "cut":
            messages = [{"role": "system", "content": system_of(p)}, {"role": "user", "content": prompt(p)}]
            answer, _ = ev.force_answer(messages, detokenize(toks[j][:b]), "length",
                                        cfg.code_eval_thinking_temperature, sampling)
        else:
            answer = detokenize(tokenize(r.get("answer") or "")[:max(0, b - len(toks[j]) - CLOSE_TOKENS)])
        with lock:
            done += 1
            emit(STAGE, msg=f"{name}: {done}/{total} redone", answered=done, total=total)
        return answer

    out = {}
    for b in budgets:
        todo = [j for j, h in enumerate(how[b]) if h != "same"]
        with ThreadPoolExecutor(ev.SLOTS) as ex:
            redone = dict(zip(todo, ex.map(lambda j: redo(b, j), todo)))
        results = [sandbox.Result(bool(r.get("passed")), r.get("reason") or "", r.get("output") or "") for r in rows]
        by_timeout: dict[float, list[int]] = {}
        for j in todo:
            by_timeout.setdefault(ev.SUITE_TIMEOUT.get(problems[j]["suite"], sandbox.TIMEOUT), []).append(j)
        for timeout, idx in by_timeout.items():
            items = []
            for j in idx:
                code, tests = codebench.assemble(problems[j], redone[j], code_of(problems[j], redone[j]))
                items.append((problems[j]["language"], code, tests))
            for j, res in zip(idx, sandbox.run_many(items, timeout=timeout)):
                results[j] = res
        write_jsonl(job.path("work", "code_eval", f"{name}-{b}.jsonl"),
                    [{"suite": r["suite"], "id": r["id"], "how": how[b][j], "passed": results[j].passed,
                      "reason": results[j].reason, **({"answer": redone[j]} if j in redone else {})}
                     for j, r in enumerate(rows)])
        suites = codebench.summarize(problems, [[x] for x in results], [1])
        out[str(b)] = {"mean_pass@1": codebench.mean_pass1(suites), "suites": suites, "how": dict(Counter(how[b]))}
    return out


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
            rows = []
        else:
            rows = [r for r in read_jsonl(job.path("work", "code_eval", f"{name}.jsonl")) if r.get("sample", 0) == 0]
            if any(r.get("reasoning_chars") and "reasoning" not in r for r in rows):
                raise RuntimeError(f"{name}: the eval kept no thinking; set code_eval_budgets before the eval")
            problems = [by_id[(r["suite"], r["id"])] for r in rows]
            emit(STAGE, pct=100 * i / len(cands), msg=f"{name}: thinking cut at {', '.join(map(str, budgets))} tokens")
            proc = ev.serve(job, c["path"], f"{name}-budgets")
            try:
                scores = rescore(job, name, rows, problems, budgets)
            finally:
                ev.stop(proc)
        scores[str(cfg.code_eval_max_tokens)] = {"mean_pass@1": code.get("mean_pass@1"), "suites": code.get("suites"),
                                                 "how": {"hit the cap": sum(r.get("forced") == "length" for r in rows)}}
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
    run_stage(Job(ap.parse_args().job))
