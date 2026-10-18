#!/usr/bin/env python3
"""
Checks a running vLLM server the way a Centipede crew will use it (deploy/README.md).
Standard library only, so it runs on a fresh box and on a Mac at the end of an ssh tunnel.

  check.py --url http://127.0.0.1:8000 --model qwen3.8-27b          the full check
  check.py --url http://127.0.0.1:8000 --status                       what the server is doing now

The full check, in order, stopping at the first failure:
  1. /health answers and /v1/models lists --model
  2. a plain call with thinking off: an answer, no thinking text in it
  3. a call with thinking on: the thinking comes back apart from the answer (reasoning parser)
  4. a call constrained to a JSON schema, as the planner sends them: it parses
  5. --concurrency calls at once (default 13: 10 workers, 2 leads, 1 orchestrator) sharing a
     long prompt prefix: all answer, the server ran them together (peak running requests from
     /metrics), and none was preempted (the KV cache held them all)
Exit 0 when every step passes, 1 otherwise.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# The model card's non-thinking sampling for Qwen3.8; the engine sends its own on every call.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def request(url, body=None, key=None, timeout=600):
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
        return r.status, raw


def metrics(base, key=None):
    """vLLM's Prometheus counters and gauges we care about, summed over labels; {} if /metrics is not there."""
    try:
        _, text = request(f"{base}/metrics", key=key, timeout=5)
    except Exception:
        return {}
    want = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_use",
        "vllm:num_preemptions_total": "preemptions",
        "vllm:prefix_cache_hits_total": "prefix_hits",
        "vllm:prefix_cache_queries_total": "prefix_queries",
        "vllm:generation_tokens_total": "generated",
    }
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)", line)
        if m and m.group(1) in want:
            k = want[m.group(1)]
            out[k] = out.get(k, 0.0) + float(m.group(3))
    return out


def chat(base, model, key, messages, max_tokens, thinking, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    t = time.monotonic()
    _, raw = request(f"{base}/v1/chat/completions", body, key)
    data = json.loads(raw)
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "finish": choice.get("finish_reason"),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
        "s": time.monotonic() - t,
    }


class Check:
    def __init__(self):
        self.failed = False

    def ok(self, what, detail=""):
        print(f"ok    {what}{'  ' + detail if detail else ''}")

    def fail(self, what, detail=""):
        print(f"FAIL  {what}{'  ' + detail if detail else ''}")
        self.failed = True


def status(base, key):
    try:
        request(f"{base}/health", key=key, timeout=5)
        print(f"health  ok ({base})")
    except Exception as e:
        print(f"health  no answer from {base}: {e}")
        return 1
    try:
        _, raw = request(f"{base}/v1/models", key=key, timeout=5)
        print("models  " + ", ".join(m["id"] for m in json.loads(raw)["data"]))
    except Exception as e:
        print(f"models  {e}")
    m = metrics(base, key)
    if m:
        hit = f"{100 * m['prefix_hits'] / m['prefix_queries']:.0f}%" if m.get("prefix_queries") else "n/a"
        print(f"now     {m.get('running', 0):.0f} running, {m.get('waiting', 0):.0f} waiting, KV cache {100 * m.get('kv_use', 0):.0f}% used")
        print(f"total   {m.get('generated', 0):.0f} tokens generated, {m.get('preemptions', 0):.0f} preemptions, prefix cache hit {hit}")
    return 0


# A long shared head, as every worker call shares the engine's system prompt and pack layout.
SHARED = (
    "You are one of several coding agents working on the same repository. Each of you owns a few "
    "files. Write only the file you are asked for, complete, in one fenced block. "
) * 20
TASKS = [
    "a Python function slugify(s) that lowercases and joins words with dashes",
    "a C function int clamp(int x, int lo, int hi)",
    "a TypeScript function chunk<T>(xs: T[], n: number): T[][]",
    "a Python function is_prime(n) with a short docstring",
    "a Go function Reverse(s string) string that handles UTF-8",
    "a Rust function fn mean(xs: &[f64]) -> Option<f64>",
    "a Python function merge_intervals(xs) for a list of (start, end) pairs",
    "a TypeScript function debounce(fn, ms)",
    "a C function size_t count_char(const char *s, char c)",
    "a Python function parse_kv(line) for 'a=1;b=2' into a dict",
    "a Bash function that prints the newest file in a directory",
    "a Python class LRU with get and put, capacity n",
    "a TypeScript function parseDuration('1h30m') returning seconds",
    "a Python function flatten(xs) for nested lists",
    "a C function void reverse(int *xs, size_t n)",
    "a Python function word_count(text) returning the top 3 words",
]


def full(base, model, key, n, max_tokens):
    c = Check()
    # 1
    try:
        request(f"{base}/health", key=key, timeout=10)
        _, raw = request(f"{base}/v1/models", key=key, timeout=10)
        ids = [m["id"] for m in json.loads(raw)["data"]]
        (c.ok if model in ids else c.fail)("models", f"serves {', '.join(ids)}" + ("" if model in ids else f"; {model} is not among them"))
    except Exception as e:
        c.fail("health", f"no answer from {base}: {e}")
        return 1
    if c.failed:
        return 1
    # 2
    try:
        r = chat(base, model, key, [{"role": "user", "content": "Write a Python function add(a, b). Only the code."}], 256, False)
        leaked = "<think>" in r["content"] or "</think>" in r["content"]
        if r["content"].strip() and not leaked:
            c.ok("plain call, thinking off", f"{r['completion_tokens']} tokens in {r['s']:.1f}s")
        else:
            c.fail("plain call, thinking off", "thinking text in the answer" if leaked else f"empty answer (finish {r['finish']})")
    except Exception as e:
        c.fail("plain call, thinking off", str(e))
    # 3
    try:
        r = chat(base, model, key, [{"role": "user", "content": "Is 391 prime? Answer yes or no, then one line why."}], 4096, True)
        if r["reasoning"] and "</think>" not in r["content"] and r["content"].strip():
            c.ok("thinking call", f"{len(r['reasoning'])} chars of thinking kept apart from a {len(r['content'])}-char answer")
        elif r["reasoning"] and r["finish"] == "length":
            c.ok("thinking call", "thinking kept apart; it ran out of tokens before answering (fine for this check)")
        else:
            c.fail("thinking call", "no reasoning_content: start the server with --reasoning-parser qwen3" if not r["reasoning"] else "thinking text in the answer")
    except Exception as e:
        c.fail("thinking call", str(e))
    # 4
    schema = {"type": "object", "properties": {"tickets": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "owns": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "owns"], "additionalProperties": False}}}, "required": ["tickets"], "additionalProperties": False}
    try:
        r = chat(base, model, key, [{"role": "user", "content": "Split 'a CLI that counts words in files' into two tickets, each owning one file."}], 512, False,
                 response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}})
        v = json.loads(r["content"])
        (c.ok if isinstance(v.get("tickets"), list) else c.fail)("JSON schema call", f"{len(v.get('tickets', []))} tickets")
    except Exception as e:
        c.fail("JSON schema call", str(e))
    # 5
    before = metrics(base, key)
    peak = {"running": 0.0, "waiting": 0.0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            m = metrics(base, key)
            for k in peak:
                peak[k] = max(peak[k], m.get(k, 0.0))
            stop.wait(0.25)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    t0 = time.monotonic()
    results, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(chat, base, model, key, [{"role": "system", "content": SHARED}, {"role": "user", "content": f"Write {TASKS[i % len(TASKS)]}."}],
                            max_tokens, False, ignore_eos=True) for i in range(n)]
        for f in cf.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(str(e))
    wall = time.monotonic() - t0
    stop.set()
    w.join(2)
    after = metrics(base, key)
    if errors:
        c.fail(f"{n} calls at once", f"{len(errors)} failed: {errors[0][:200]}")
    else:
        toks = sum(r["completion_tokens"] for r in results)
        lat = sorted(r["s"] for r in results)
        c.ok(f"{n} calls at once", f"{toks} tokens in {wall:.1f}s = {toks / wall:.0f} tok/s together, {toks / len(results) / statistics.median(lat):.0f} tok/s per call, latency median {statistics.median(lat):.1f}s max {lat[-1]:.1f}s")
    if before or after:
        if peak["running"] >= n:
            c.ok("ran together", f"peak {peak['running']:.0f} running, {peak['waiting']:.0f} waiting")
        else:
            c.fail("ran together", f"peak only {peak['running']:.0f} running and {peak['waiting']:.0f} waiting: raise --max-num-seqs or the KV cache (MAX_NUM_SEQS, GPU_UTIL, MAX_MODEL_LEN)")
        pre = after.get("preemptions", 0) - before.get("preemptions", 0)
        (c.ok if pre == 0 else c.fail)("no preemptions", "" if pre == 0 else f"{pre:.0f} preempted: the KV cache is too small for {n} calls; lower MAX_MODEL_LEN or use KV_DTYPE=fp8")
        if after.get("prefix_queries"):
            dq = after["prefix_queries"] - before.get("prefix_queries", 0)
            dh = after.get("prefix_hits", 0) - before.get("prefix_hits", 0)
            print(f"info  prefix cache hit {100 * dh / dq:.0f}% during the burst" if dq else "info  no prefix cache queries counted")
    else:
        print("info  no /metrics: concurrency and preemptions not checked")
    print("PASS" if not c.failed else "FAIL")
    return 1 if c.failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server root, without /v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--concurrency", type=int, default=13)
    p.add_argument("--max-tokens", type=int, default=384, help="tokens each burst call writes (ignore_eos, so all write this many)")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    base = a.url.rstrip("/").removesuffix("/v1")
    key = os.environ.get("VLLM_API_KEY") or None
    sys.exit(status(base, key) if a.status else full(base, a.model, key, a.concurrency, a.max_tokens))


if __name__ == "__main__":
    main()
