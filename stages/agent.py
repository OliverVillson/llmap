"""Project Centipede's code agent (scripts/centipede.py): the helpers that run in the
training venv.

The agent is one small model behind an OpenAI-style API, with many contexts at once. It
speaks Mugge's harness format (stages/harness.py): a ticket in, each owned file in full
out, and one fix call with the failing command's output when a check fails.

    python -m stages.agent pool --jobs D1 D2 ... --out bench/agent-train.jsonl
    python -m stages.agent round --answers work/code_eval/NAME.jsonl --suite bench/agent-train.jsonl --out rows.jsonl
    python -m stages.agent traces --hf MODEL_DIR --rows train.jsonl --out traces.jsonl
    python -m stages.agent serve-check --gguf MODEL.gguf --llama-cpp DIR --bench DIR --out serve.json

pool: the data jobs' own tasks with their tests (work/data_inputs.jsonl, minus each job's
held-out inputs) as one code suite, for the loop round.
round: the round model's passing answers to that suite as training rows (its shortest
passing answer per ticket, thinking kept, never one whose thinking was cut short).
traces: chat rows as trace rows (stages/traces.py), the training text Event Horizon's
compression stages read: the prompt exactly as at inference, then the thinking and answer.
serve-check: one llama-server with `slots` contexts answers that many tickets at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import threading
import time
import zlib
from pathlib import Path

from stages import codebench, harness, sandbox


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Via a .tmp file, so a stop never leaves half a file under the real name."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(p)


# ---------- the loop round ----------

def pool(jobs: list[str | Path], per_language: int = 0, seed: int = 0) -> list[dict]:
    """Code suite rows (codebench's shape, asked as written) from data jobs: each task
    the teacher wrote with its tests, minus the job's held-out inputs; at most
    per_language per language (0 = all), drawn with a fixed seed."""
    out = []
    for d in map(Path, jobs):
        lang = codebench.normalize_language(json.loads((d / "taskspec.json").read_text())["language"])
        held_path = d / "data" / "heldout.jsonl"
        held = {r["input"] for r in read_jsonl(held_path)} if held_path.exists() else set()
        rows, seen = [], set()
        for r in read_jsonl(d / "work" / "data_inputs.jsonl"):
            pid = f"{lang}-{zlib.crc32(r['input'].encode()):08x}"
            if r.get("tests") and r["input"] not in held and pid not in seen:
                seen.add(pid)
                rows.append({"id": pid, "language": lang, "prompt": r["input"], "tests": r["tests"], "raw_prompt": True})
        random.Random(seed).shuffle(rows)
        out += rows[:per_language or None]
    if not out:
        raise ValueError(f"no tasks with tests in {[str(j) for j in jobs]}")
    return out


def round_rows(answers: list[dict], problems: list[dict]) -> list[dict]:
    """Write-call training rows from a code eval's answers (work/code_eval/<name>.jsonl,
    kept with their thinking): per problem, the shortest answer (thinking plus answer)
    that passed, finished its thinking itself and was not cut. The answer is rendered the
    way the teacher's rows are (the owned file, then its note)."""
    by_id = {p["id"]: p for p in problems}
    best: dict[str, dict] = {}
    for a in answers:
        if not (a.get("id") in by_id and a.get("passed") and (a.get("reasoning") or "").strip()
                and not a.get("forced") and not a.get("answer_cut")):
            continue
        cost = len(a["reasoning"]) + len(a.get("answer") or "")
        if a["id"] not in best or cost < len(best[a["id"]]["reasoning"]) + len(best[a["id"]].get("answer") or ""):
            best[a["id"]] = a
    rows = []
    for pid in sorted(best):
        p, a = by_id[pid], best[pid]
        src = sandbox.layout(p["language"])[0]
        code = codebench.harness_code(p, a["answer"])
        note = harness.parse_files(a["answer"], [src])[1]
        rows.append({"messages": [*harness.write_messages(codebench.ticket(p)),
                                  {"role": "assistant", "content": harness.render_files({src: code}, note),
                                   "reasoning_content": a["reasoning"].strip()}],
                     "meta": {"round": 1, "language": p["language"]}})
    return rows


# ---------- training text for the compression ----------

def trace_rows(tok, rows: list[dict], max_len: int = 8192) -> tuple[list[dict], dict]:
    """(trace rows, counts): each chat row tokenized as training sees it
    (taskdata.tokenize_example: the prompt as at inference, thinking opened, then the
    thinking, the answer and the end of turn), split into prompt and answer ids. A row's
    id is its user message's hash, so a task's rows fall on one side of the held-out
    split. Rows cut at max_len are left out (their thinking would never close)."""
    from stages.taskdata import to_messages, tokenize_example

    out, cut = [], 0
    for r in rows:
        msgs = to_messages(r)
        if not msgs:
            continue
        ex = tokenize_example(tok, msgs, max_len)
        ids, labels = ex["input_ids"], ex["labels"]
        k = next((i for i, x in enumerate(labels) if x != -100), len(labels))
        if len(ids) >= max_len:
            cut += 1
            continue
        if k == len(ids):
            continue
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        out.append({"id": hashlib.sha256(user.encode()).hexdigest()[:16], "prompt": ids[:k], "tokens": ids[k:],
                    "finish": "stop", "passed": True, "forced": None})
    lens = [len(r["tokens"]) for r in out]
    return out, {"rows": len(out), "cut": cut, "answer_tokens": sum(lens),
                 "median_answer": statistics.median(lens) if lens else 0}


# ---------- 40 agents on one server ----------

SERVE_SUITES = ("multipl-e-py", "multipl-e-js", "multipl-e-ts", "c-set")
PORT = 8097


def serve_problems(bench: str | Path, n: int) -> list[dict]:
    """n problems, taken in turn from SERVE_SUITES' first rows."""
    per = -(-n // len(SERVE_SUITES))
    suites = [codebench.load_suite(s, bench, limit=per) for s in SERVE_SUITES]
    mixed = [p for group in zip(*suites) for p in group] if all(suites) else [p for s in suites for p in s]
    if len(mixed) < n:
        raise ValueError(f"only {len(mixed)} problems in {SERVE_SUITES} under {bench}, not {n}")
    return mixed[:n]


def gpu_mem_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def serve_check(gguf: str, llama_cpp: str, bench: str, out: str, *, slots: int = 40, ctx: int = 16384,
                kv: str = "q8_0", max_tokens: int = 4096, port: int = PORT, health_s: int = 900) -> dict:
    """One llama-server with `slots` contexts of `ctx` tokens (KV cache in `kv`) answers
    `slots` tickets at once, thinking on and capped at max_tokens. Reports wall time,
    tokens per second over all agents and per agent, the GPU memory in use while they run,
    and how many answers came back whole, wrote their file and pass their tests. Raises
    when a request fails."""
    import httpx

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    problems = serve_problems(bench, slots)
    cmd = [str(Path(llama_cpp) / "build" / "bin" / "llama-server"), "-m", gguf, "-ngl", "999", "--host", "127.0.0.1",
           "--port", str(port), "-c", str(slots * ctx), "-np", str(slots), "--jinja", "-fa", "on"]
    if kv:
        cmd += ["-ctk", kv, "-ctv", kv]
    log = open(out_p.with_suffix(".server.log"), "w")
    idle = gpu_mem_mib()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + health_s
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({proc.returncode}); see {log.name}")
            try:
                if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server not healthy after {health_s} s; see {log.name}")
            time.sleep(2)
        loaded = gpu_mem_mib()
        peak, stop = [loaded or 0], threading.Event()

        def watch():
            while not stop.wait(2):
                peak.append(gpu_mem_mib() or 0)

        threading.Thread(target=watch, daemon=True).start()
        results: list[dict | None] = [None] * len(problems)

        def ask(i: int) -> None:
            body = {"messages": harness.write_messages(codebench.ticket(problems[i])), "max_tokens": max_tokens,
                    "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
                    "chat_template_kwargs": {"enable_thinking": True}}
            t = time.monotonic()
            r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=httpx.Timeout(3600, connect=30))
            r.raise_for_status()
            d = r.json()
            results[i] = {"seconds": time.monotonic() - t, "tokens": d["usage"]["completion_tokens"],
                          "tok_s": (d.get("timings") or {}).get("predicted_per_second"),
                          "finish": d["choices"][0]["finish_reason"],
                          "content": d["choices"][0]["message"].get("content") or ""}

        errors: list[str] = []

        def run(i: int) -> None:
            try:
                ask(i)
            except Exception as e:  # noqa: BLE001  (counted and reported)
                errors.append(f"{problems[i]['id']}: {type(e).__name__}: {e}"[:300])

        t0 = time.monotonic()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(problems))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.monotonic() - t0
        stop.set()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
    done = [(p, r) for p, r in zip(problems, results) if r]
    files = [(p, r, codebench.harness_code(p, r["content"])) for p, r in done
             if sandbox.layout(p["language"])[0] in harness.parse_files(r["content"], [sandbox.layout(p["language"])[0]])[0]]
    passed = sandbox.run_many([codebench.job(p, r["content"], code) for p, r, code in files])
    tokens = sum(r["tokens"] for _, r in done)
    rep = {"slots": slots, "ctx": ctx, "kv": kv or "f16", "max_tokens": max_tokens, "gguf": gguf,
           "gguf_gb": round(Path(gguf).stat().st_size / 1e9, 3), "requests": len(problems), "completed": len(done),
           "errors": errors, "wall_s": round(wall, 1), "tokens": tokens, "tok_s_all": round(tokens / wall, 1) if wall else None,
           "tok_s_per_agent_median": round(statistics.median(r["tok_s"] for _, r in done if r["tok_s"]), 1)
           if any(r["tok_s"] for _, r in done) else None,
           "cut": sum(r["finish"] == "length" for _, r in done), "wrote_file": len(files),
           "passed": sum(x.passed for x in passed),
           "gpu_mib": {"before": idle, "loaded": loaded, "peak": max(peak) or None}}
    out_p.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "errors"}), flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(problems)} requests failed, e.g. {errors[0]}")
    return rep


# ---------- the command line ----------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--per-language", type=int, default=0)
    p.add_argument("--out", required=True)
    r = sub.add_parser("round")
    r.add_argument("--answers", required=True)
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    t = sub.add_parser("traces")
    t.add_argument("--hf", required=True)
    t.add_argument("--rows", required=True)
    t.add_argument("--max-len", type=int, default=8192)
    t.add_argument("--out", required=True)
    s = sub.add_parser("serve-check")
    s.add_argument("--gguf", required=True)
    s.add_argument("--llama-cpp", required=True)
    s.add_argument("--bench", required=True)
    s.add_argument("--slots", type=int, default=40)
    s.add_argument("--ctx", type=int, default=16384)
    s.add_argument("--kv", default="q8_0")
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "pool":
        rows = pool(a.jobs, a.per_language)
        write_jsonl(a.out, rows)
        langs = {lang: sum(r["language"] == lang for r in rows) for lang in dict.fromkeys(r["language"] for r in rows)}
        print(f"pool: {len(rows)} tasks {langs} -> {a.out}", flush=True)
    elif a.cmd == "round":
        suite = Path(a.suite)  # read as the eval read it, so each ticket is the one it asked
        rows = round_rows(read_jsonl(a.answers), codebench.load_suite(suite.stem, suite.parent))
        if not rows:
            raise SystemExit(f"round: no passing answer in {a.answers}")
        write_jsonl(a.out, rows)
        print(f"round: {len(rows)} rows -> {a.out}", flush=True)
    elif a.cmd == "traces":
        from transformers import AutoTokenizer

        rows, rep = trace_rows(AutoTokenizer.from_pretrained(a.hf), read_jsonl(a.rows), a.max_len)
        if not rows:
            raise SystemExit(f"traces: no usable rows in {a.rows}")
        write_jsonl(a.out, rows)
        Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=2) + "\n")
        print(f"traces: {json.dumps(rep)} -> {a.out}", flush=True)
    else:
        serve_check(a.gguf, a.llama_cpp, a.bench, a.out, slots=a.slots, ctx=a.ctx, kv=a.kv, max_tokens=a.max_tokens)


if __name__ == "__main__":
    main()
