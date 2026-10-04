"""Stage 5: evaluate teacher vs. candidates on the held-out set.

Each candidate GGUF is served with llama-server on the VM GPU and answers the
held-out inputs with the same system prompt the training data used. Every
answer is scored two ways:
  agreement  match with the teacher's reference answer (field-level for JSON
             outputs, token F1 otherwise). Needs no API key.
  judge      an LLM judge (Gemini by default, cfg.judge_model) scores each
             answer 0-10 against the TaskSpec criteria (teacher references
             included). Used as `score` when available.
  pass@k     when Config.code_eval_suites is set, every candidate also answers
             the code benchmarks (stages/codebench.py) and each answer is
             compiled and run against its tests (stages/sandbox.py). The mean
             pass@1 over suites then becomes `score` (score_method "pass@1"),
             with judge and agreement still reported next to it.
Also records the decode speed measured on the VM. Writes out/eval.json, the
contract the scoreboard screen reads, and work/code_eval/<candidate>.jsonl
with every code answer and its test status.

Config.eval_candidates replaces work/allocation.json with named GGUFs, so an
eval-only job (pipeline.py --only eval) can score an uncompressed reference;
without data/heldout.jsonl such a job is scored on code alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.progress import emit
from stages import codebench, data, sandbox
from stages._util import DRY_RUN, Job, read_jsonl, write_jsonl

STAGE = "eval"
PORT = int(os.environ.get("LOBBOT_EVAL_PORT", "8091"))
SLOTS = 8
# Must cover the longest teacher answer the data stage keeps
# (LOBBOT_DATA_ANSWER_MAX_TOKENS), or long correct answers are judged as truncated.
MAX_TOKENS = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.ANSWER_MAX_TOKENS)))
# Room per slot for the system prompt, an input of up to data.MAX_INPUT_CHARS
# (~2k tokens) and the answer.
CTX_PER_SLOT = int(os.environ.get("LOBBOT_EVAL_CTX", MAX_TOKENS + 4096))


def limits(cfg) -> tuple[int, int]:
    """(max answer tokens, context per slot) for this job: follows the job's
    data answer cap (Config data_answer_max_tokens) unless LOBBOT_EVAL_* is set."""
    max_tokens = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.answer_max_tokens(cfg))))
    return max_tokens, int(os.environ.get("LOBBOT_EVAL_CTX", max_tokens + 4096))


def serve(job: Job, gguf: str, name: str) -> subprocess.Popen:
    import httpx

    binary = Path(job.config.llama_cpp) / "build/bin/llama-server"
    log_path = job.path("work", f"llama-server-{name}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [str(binary), "-m", gguf, "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT),
         "-c", str(SLOTS * limits(job.config)[1]), "-np", str(SLOTS), "--jinja"],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited for {name}:\n" + "\n".join(log_path.read_text().splitlines()[-20:]))
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"llama-server did not become healthy for {name}; see {log_path}")


def generate(system: str, inputs: list[str], max_tokens: int = MAX_TOKENS,
             temperature: float = 0.0, seed: int | None = None) -> tuple[list[str], float | None]:
    """Answers plus the median decode speed (tok/s) the server reported."""
    import httpx

    speeds: list[float] = []

    def one(text: str) -> str:
        for attempt in range(2):
            try:
                r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=600, json={
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                    "temperature": temperature, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                    **({"seed": seed, "top_p": 0.95} if seed is not None else {}),
                })
                r.raise_for_status()
                body = r.json()
                tps = (body.get("timings") or {}).get("predicted_per_second")
                if tps:
                    speeds.append(float(tps))
                content = body["choices"][0]["message"].get("content") or ""
                return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            except Exception as e:
                if attempt:
                    print(f"[eval] generation failed: {e}", flush=True)
        return ""

    with ThreadPoolExecutor(SLOTS) as ex:
        answers = list(ex.map(one, inputs))
    return answers, (round(statistics.median(speeds), 1) if speeds else None)


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _f1(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    common = sum((ta & tb).values())
    if not ta or not tb:
        return float(ta == tb)
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def _as_json(s: str):
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def agreement(answer: str, reference: str) -> float:
    """1.0 means the answer matches the teacher. For JSON objects: short fields
    must match exactly, long text fields score by token F1; invalid JSON is 0."""
    ref = _as_json(reference)
    if isinstance(ref, dict) and ref:
        ans = _as_json(answer)
        if not isinstance(ans, dict):
            return 0.0
        scores = []
        for k, v in ref.items():
            a = ans.get(k)
            if isinstance(v, str) and len(v.split()) > 3:
                scores.append(_f1(str(a or ""), v))
            else:
                scores.append(float(str(a).strip().lower() == str(v).strip().lower()))
        return sum(scores) / len(scores)
    return _f1(answer, reference)


def judge_key(model: str) -> str | None:
    """The API key the judge model needs, or None if it is not set."""
    if model.startswith("gemini"):
        from stages.gemini import api_key

        return api_key()
    return os.environ.get("ANTHROPIC_API_KEY")


def judge(spec, model: str, inputs: list[str], answers: list[str]) -> float | None:
    """Mean judge score in [0, 1]; None if no answer could be judged. gemini-*
    models go to the Gemini API, claude-* to Anthropic; both through
    condense.chat when CONDENSE_API_KEY is set (stages/gemini.py, stages/condense.py)."""
    if model.startswith("gemini"):
        from stages import gemini

        ask = lambda prompt: gemini.chat(model, [{"role": "user", "content": prompt}])
    else:
        from stages.condense import anthropic_client

        client = anthropic_client(max_retries=6)
        ask = lambda prompt: client.messages.create(
            model=model, max_tokens=20, messages=[{"role": "user", "content": prompt}]).content[0].text

    broken = threading.Event()  # a 4xx (bad key or model id) will not fix itself: stop calling

    def one(pair) -> float | None:
        text, answer = pair
        if broken.is_set():
            return None
        try:
            reply = ask(
                f"Task: {spec.description}\nOutput format: {spec.output_format}\nCriteria: {spec.eval_criteria}\n\n"
                f"<input>\n{text}\n</input>\n<answer>\n{answer}\n</answer>\n\n"
                "Score how well the answer meets the criteria from 0 to 10. Reply with the number only.")
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429 and not broken.is_set():
                broken.set()
                print(f"[eval] judge disabled after HTTP {status}: {e}", flush=True)
            elif not broken.is_set():
                print(f"[eval] judge call failed: {e}", flush=True)
            return None
        m = re.search(r"\d+(\.\d+)?", reply)
        return min(10.0, float(m.group(0))) / 10 if m else None

    with ThreadPoolExecutor(16) as ex:
        scores = [s for s in ex.map(one, zip(inputs, answers)) if s is not None]
    return round(sum(scores) / len(scores), 3) if scores else None


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def code_problems(job: Job, held: list[dict]) -> list[dict]:
    """Every problem of the configured code suites."""
    cfg = job.config
    problems: list[dict] = []
    for suite in cfg.code_eval_suites:
        if suite == "heldout":
            rows = codebench.heldout_problems(held)[: cfg.code_eval_limit or None]
        else:
            since = cfg.lcb_since if suite == "livecodebench" else None
            rows = codebench.load_suite(suite, cfg.code_eval_dir, since, cfg.code_eval_limit)
        if not rows:
            emit(STAGE, msg=f"code suite {suite} has no problems; skipping it")
        problems += rows
    return problems


# LiveCodeBench runs every test case in one process, so it gets a longer budget.
SUITE_TIMEOUT = {"livecodebench": max(sandbox.TIMEOUT, 60.0)}


def code_eval(job: Job, name: str, problems: list[dict], task_system: str) -> dict:
    """Answer every problem code_eval_samples times on the running server, run
    the tests in the sandbox and return per-suite pass@k. Per-answer results go
    to work/code_eval/<name>.jsonl. Held-out task rows are asked with the task's
    own system prompt (the one the model was healed on), benchmarks with
    codebench.SYSTEM."""
    cfg = job.config
    n = max(cfg.code_eval_samples, max(cfg.code_eval_k))
    groups: dict[tuple[str, float], list[int]] = {}
    for j, p in enumerate(problems):
        # Held-out rows of a merged multi-language job carry their own task prompt.
        key = ((p.get("system") or task_system) if p["suite"] == "heldout" else codebench.SYSTEM,
               SUITE_TIMEOUT.get(p["suite"], sandbox.TIMEOUT))
        groups.setdefault(key, []).append(j)
    results: list[list] = [[] for _ in problems]
    rows, speeds = [], []
    for i in range(n):
        greedy = n == 1
        answers = [""] * len(problems)
        for (system, _), idx in groups.items():
            got, tps = generate(system, [codebench.build_prompt(problems[j]) for j in idx], cfg.code_eval_max_tokens,
                                0.0 if greedy else cfg.code_eval_temperature, None if greedy else i)
            speeds += [tps] if tps else []
            for j, a in zip(idx, got):
                answers[j] = a
        ran: list = [None] * len(problems)
        for (_, timeout), idx in groups.items():
            items = []
            for j in idx:
                code, tests = codebench.assemble(problems[j], answers[j])
                items.append((problems[j]["language"], code, tests))
            for j, r in zip(idx, sandbox.run_many(items, timeout=timeout)):
                ran[j] = r
        for j, (p, a, r) in enumerate(zip(problems, answers, ran)):
            results[j].append(r)
            rows.append({"suite": p["suite"], "id": p["id"], "sample": i, "passed": r.passed,
                         "reason": r.reason, "output": r.output[-500:], "answer": a})
    out_dir = job.path("work", "code_eval")
    out_dir.mkdir(exist_ok=True)
    write_jsonl(out_dir / f"{name}.jsonl", rows)
    suites = codebench.summarize(problems, results, cfg.code_eval_k)
    for suite, v in suites.items():
        broken = sum(v["status"].get(r, 0) for r in ("missing_toolchain", "unsupported_language"))
        if broken:
            emit(STAGE, msg=f"{suite}: {broken} answers could not run ({', '.join(k for k in v['status'] if k in ('missing_toolchain', 'unsupported_language'))}); check the VM toolchains")
    return {"suites": suites, "mean_pass@1": codebench.mean_pass1(suites), "samples_per_problem": n,
            "tok_s_vm": round(statistics.median(speeds), 1) if speeds else None}


def load_code_ref(job: Job) -> dict:
    """The reference job's per-suite code results ({} when unset or not run yet)."""
    if not job.config.code_eval_ref:
        return {}
    p = Path(job.config.code_eval_ref)
    p = p if p.is_absolute() else job.root.parent / p
    if p.is_dir():
        p = p / "out" / "eval.json"
    if not p.exists():
        emit(STAGE, msg=f"reference results {p} not found; run the ref job first for scores as a share of ref")
        return {}
    return codebench.ref_suites(json.loads(p.read_text()))


def candidates_to_eval(job: Job) -> dict[str, dict]:
    """{name: {path, size_gb, tok_s_est}} from Config.eval_candidates or allocation.json."""
    cfg = job.config
    if cfg.eval_candidates:
        out = {}
        for name, path in cfg.eval_candidates.items():
            p = Path(path)
            if not p.exists() and not DRY_RUN:
                raise FileNotFoundError(f"eval candidate {name}: {p} not found")
            parts = sorted(p.parent.glob(p.name.replace("-00001-of-", "-*-of-"))) if "-00001-of-" in p.name else [p]
            size = sum(f.stat().st_size for f in parts if f.exists())
            out[name] = {"path": str(p), "size_gb": round(size / 1e9, 2), "tok_s_est": None}
        return out
    return json.loads(job.path("work", "allocation.json").read_text())["candidates"]


def teacher_size_gb(job: Job) -> float:
    d = Path(job.model_path(job.config.teacher))
    if d.is_dir():
        n = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        if n:
            return round(n / 1e9, 1)
    return 61.0  # Qwen3-30B-A3B in bf16


def run_stage(job: Job) -> None:
    from stages.data import system_prompt

    cfg, spec = job.config, job.spec
    alloc_path = job.path("work", "allocation.json")
    alloc = json.loads(alloc_path.read_text()) if alloc_path.exists() and not cfg.eval_candidates else {}
    cands = candidates_to_eval(job)
    held_path = job.path("data", "heldout.jsonl")
    held = read_jsonl(held_path) if held_path.exists() else []
    inputs = [h["input"] for h in held]
    refs = [h["reference"] for h in held]
    system = system_prompt(spec)
    teacher_name = cfg.teacher.split("/")[-1]
    problems = code_problems(job, held) if cfg.code_eval_suites else []
    if not inputs and not problems:
        raise RuntimeError("nothing to evaluate: no data/heldout.jsonl and no code_eval_suites")
    use_judge = bool(inputs) and bool(judge_key(cfg.judge_model)) and not DRY_RUN
    if inputs and not use_judge and not DRY_RUN:
        need = "GEMINI_API_KEY" if cfg.judge_model.startswith("gemini") else "ANTHROPIC_API_KEY"
        emit(STAGE, msg=f"{need} not set on the VM; scoring by agreement with the teacher")
    if problems:
        emit(STAGE, msg=f"code eval: {len(problems)} problems from {', '.join(cfg.code_eval_suites)}")

    results: dict[str, dict] = {}
    teacher_judge = None
    if DRY_RUN:
        for i, name in enumerate(cands):
            code = None
            if problems:
                langs = {p["suite"]: p["language"] for p in problems}
                suites = {s: {"language": lang, "n": sum(p["suite"] == s for p in problems), "status": {},
                              **{f"pass@{k}": round(0.8 - 0.07 * i, 4) for k in cfg.code_eval_k}}
                          for s, lang in langs.items()}
                code = {"suites": suites, "mean_pass@1": codebench.mean_pass1(suites), "samples_per_problem": 1}
            results[name] = {"agreement": 0.85 - 0.07 * i if inputs else None, "judge": None, "tok_s_vm": None,
                             "code": code, "samples": []}
    else:
        if use_judge:
            emit(STAGE, pct=5, msg=f"judging teacher references on {len(inputs)} held-out inputs")
            teacher_judge = judge(spec, cfg.judge_model, inputs, refs)
        n = len(cands)
        for i, (name, c) in enumerate(cands.items()):
            emit(STAGE, pct=10 + 85 * i / n, msg=f"running {name}")
            proc = serve(job, c["path"], name)
            agree = score = tps = code = None
            answers: list[str] = []
            try:
                if inputs:
                    answers, tps = generate(system, inputs, limits(cfg)[0])
                if problems:
                    emit(STAGE, pct=10 + 85 * (i + 0.3) / n, msg=f"{name}: answering {len(problems)} code problems")
                    code = code_eval(job, name, problems, system)
            finally:
                stop(proc)
            tps = tps or (code or {}).get("tok_s_vm")
            if inputs:
                agree = round(sum(agreement(a, r) for a, r in zip(answers, refs)) / max(len(refs), 1), 3)
                emit(STAGE, pct=10 + 85 * (i + 0.6) / n, msg=f"{name}: {agree:.0%} agreement with teacher, {tps} tok/s on the VM")
                score = judge(spec, cfg.judge_model, inputs, answers) if use_judge else None
            if code:
                per = ", ".join(f"{k} {v['pass@1']:.0%}" for k, v in code["suites"].items() if "pass@1" in v)
                emit(STAGE, pct=10 + 85 * (i + 0.9) / n, msg=f"{name}: pass@1 {per}")
            results[name] = {
                "agreement": agree, "judge": score, "tok_s_vm": tps, "code": code,
                "samples": [{"input": a, "output": b, "reference": r} for a, b, r in list(zip(inputs, answers, refs))[:3]],
            }

    # pass@1 when code was run for every model; else judge when every model has
    # one; else agreement (teacher = 1.0 by definition). The teacher itself is
    # not run on code here: an eval-only `ref` job scores it.
    coded = bool(problems) and all(r["code"] and r["code"]["mean_pass@1"] is not None for r in results.values())
    judged = teacher_judge is not None and all(r["judge"] is not None for r in results.values())
    method = "pass@1" if coded else "judge" if judged else "agreement"
    teacher_score = None if coded else teacher_judge if judged else 1.0

    candidates = []
    for name, c in cands.items():
        r = results[name]
        score = r["code"]["mean_pass@1"] if coded else r["judge"] if judged else r["agreement"]
        tok_s_est = c.get("tok_s_est")
        candidates.append({
            "name": name, "size_gb": c["size_gb"], "score": score,
            "agreement": r["agreement"], "judge_score": r["judge"], "code": r["code"],
            "tok_s_est": tok_s_est, "tok_s_vm": r["tok_s_vm"],
            "meets_target": c["size_gb"] <= spec.target.max_size_gb and (tok_s_est or 0) >= spec.target.min_tok_s,
        })
    ref = load_code_ref(job) if coded else {}
    for c in candidates:
        if ref and c["code"]:
            c["code"]["share_of_ref"] = codebench.share_of_ref(c["code"]["suites"], ref)
    eligible = [c for c in candidates if c["meets_target"]] or candidates
    best = max(eligible, key=lambda c: c["score"] if c["score"] is not None else -1)
    # Prefer the compressed MoE when it is within 2 points of the best.
    moe = next((c for c in eligible if c["name"] == "lobbot-moe"), None)
    winner = moe if moe and moe["score"] is not None and moe["score"] >= best["score"] - 0.02 else best

    report = {
        "teacher": {"name": teacher_name, "size_gb": teacher_size_gb(job), "score": teacher_score},
        "candidates": candidates,
        "winner": winner["name"],
        "score_method": method,
        "bit_widths": alloc.get("bit_widths", []),
        "samples": {k: v["samples"] for k, v in results.items()},
    }
    job.path("out", "eval.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"winner": winner["name"]})
    emit(STAGE, "done", 100, f"winner {winner['name']}: score {winner['score']} vs teacher {teacher_score} ({method})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
