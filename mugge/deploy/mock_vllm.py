#!/usr/bin/env python3
"""
A stand-in for `vllm serve` with no GPU, for dry runs of qwen.sh and check.py (deploy/test.sh, CI).
It takes the same command line (`mock_vllm.py serve MODEL --port N --served-model-name A B
--max-num-seqs N ...`, other flags ignored) and answers what the crew and check.py use: /health,
/v1/models, /v1/chat/completions (thinking split into reasoning_content as --reasoning-parser
qwen3 does, JSON-schema answers, ignore_eos), and /metrics with vLLM's metric names. At most
--max-num-seqs requests run at once; the rest wait, as in vLLM's scheduler.

  VLLM=deploy/mock_vllm.py deploy/qwen.sh start
  MOCK_LOAD_S  seconds of pretend loading before /health answers (default 2)
  MOCK_TPS     tokens per second per request (default 400)
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def args(argv):
    out = {"port": 8000, "host": "127.0.0.1", "names": None, "seqs": 256, "model": None, "max_len": 4096}
    i = 0
    if argv and argv[0] == "serve":
        argv = argv[1:]
    if argv and not argv[0].startswith("-"):
        out["model"] = argv[0]
        argv = argv[1:]
    while i < len(argv):
        a = argv[i]
        if a == "--port":
            out["port"] = int(argv[i + 1])
        elif a == "--host":
            out["host"] = argv[i + 1]
        elif a == "--max-num-seqs":
            out["seqs"] = int(argv[i + 1])
        elif a == "--max-model-len":
            out["max_len"] = int(argv[i + 1])
        elif a == "--served-model-name":
            names, j = [], i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                names.append(argv[j])
                j += 1
            out["names"] = names
            i = j
            continue
        i += 1
    out["names"] = out["names"] or [out["model"] or "mock"]
    return out


A = args(sys.argv[1:])
LOCK = threading.Lock()
SLOTS = threading.Semaphore(A["seqs"])
STATS = {"running": 0, "waiting": 0, "generated": 0, "prompt": 0, "queries": 0, "hits": 0}
READY_AT = time.time() + float(os.environ.get("MOCK_LOAD_S", "2"))
TPS = float(os.environ.get("MOCK_TPS", "400"))


def instance(schema):
    """The smallest value that satisfies a simple JSON schema (enough for check.py and the planner's shapes)."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        return {k: instance(props[k]) for k in schema.get("required", list(props))}
    if t == "array":
        n = max(schema.get("minItems", 0), 2)
        return [instance(schema.get("items", {"type": "string"})) for _ in range(n)]
    if t in ("integer", "number"):
        return 1
    if t == "boolean":
        return True
    return "x"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        key = os.environ.get("VLLM_API_KEY")
        return not key or self.headers.get("authorization") == f"Bearer {key}"

    def do_GET(self):
        if time.time() < READY_AT:
            return self.send(503, {"error": "loading"})
        if self.path == "/health":
            return self.send(200, b"", "text/plain")
        if self.path == "/metrics":
            with LOCK:
                s = dict(STATS)
            m = A["names"][0]
            lines = [
                f'vllm:num_requests_running{{model_name="{m}"}} {s["running"]}',
                f'vllm:num_requests_waiting{{model_name="{m}"}} {s["waiting"]}',
                f'vllm:kv_cache_usage_perc{{model_name="{m}"}} {min(1.0, s["running"] / max(A["seqs"], 1)) * 0.5}',
                f'vllm:num_preemptions_total{{model_name="{m}"}} 0.0',
                f'vllm:prefix_cache_queries_total{{model_name="{m}"}} {s["queries"]}',
                f'vllm:prefix_cache_hits_total{{model_name="{m}"}} {s["hits"]}',
                f'vllm:generation_tokens_total{{model_name="{m}"}} {s["generated"]}',
            ]
            return self.send(200, "\n".join(lines) + "\n", "text/plain")
        if not self.authorized():
            return self.send(401, {"error": "Unauthorized"})
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": n, "object": "model", "max_model_len": A["max_len"]} for n in A["names"]]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if time.time() < READY_AT:
            return self.send(503, {"error": "loading"})
        if not self.authorized():
            return self.send(401, {"error": "Unauthorized"})
        if self.path != "/v1/chat/completions":
            return self.send(404, {"error": "not found"})
        if body.get("model") not in A["names"]:
            return self.send(404, {"error": {"message": f"The model `{body.get('model')}` does not exist."}})
        prompt = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4
        max_tokens = int(body.get("max_tokens") or 256)
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        rf = body.get("response_format") or {}
        with LOCK:
            STATS["waiting"] += 1
        SLOTS.acquire()
        with LOCK:
            STATS["waiting"] -= 1
            STATS["running"] += 1
            STATS["queries"] += prompt
            STATS["hits"] += prompt // 2
        try:
            if rf.get("type") == "json_schema":
                content = json.dumps(instance(rf["json_schema"]["schema"]))
                toks = min(max_tokens, len(content) // 4 + 1)
            else:
                content = "```python\ndef add(a, b):\n    return a + b\n```"
                toks = max_tokens if body.get("ignore_eos") else min(max_tokens, 24)
            reasoning = "The user asks something; think it through briefly." if thinking else None
            if thinking:
                toks = min(max_tokens, toks + 16)
            time.sleep(toks / TPS)
        finally:
            with LOCK:
                STATS["running"] -= 1
                STATS["generated"] += toks
                STATS["prompt"] += prompt
            SLOTS.release()
        msg = {"role": "assistant", "content": content}
        if reasoning:
            # vLLM 0.31 answers with `reasoning`; `reasoning_content` is only accepted on input now
            msg["reasoning"] = reasoning
        self.send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "length" if body.get("ignore_eos") else "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": toks, "total_tokens": prompt + toks},
        })


if __name__ == "__main__":
    print(f"mock vLLM for {', '.join(A['names'])} on {A['host']}:{A['port']}, max {A['seqs']} running", flush=True)
    print(f"Available KV cache memory: {os.environ.get('MOCK_KV_GIB', '120.00')} GiB (mock)", flush=True)
    print(f"GPU KV cache size: 1,966,080 tokens, Maximum concurrency for {A['max_len']:,} tokens per request: {1966080 / A['max_len']:.2f}x (mock)", flush=True)
    ThreadingHTTPServer.request_queue_size = 128
    ThreadingHTTPServer((A["host"], A["port"]), H).serve_forever()
