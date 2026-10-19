#!/usr/bin/env python3
"""
A stand-in for llama-server with no Mac, for dry runs of mac.sh and check.py (deploy/test.sh, CI).
It takes llama-server's command line (-m FILE --alias NAME --port N -np SLOTS -c CTX, other flags
ignored), prints a --help that lists the flags mac.sh checks for, and answers what the crew and
check.py use the way a real llama-server does (test/fixtures/llama-server): /health, /v1/models,
/v1/chat/completions (thinking in reasoning_content, cut at thinking_budget_tokens, JSON-schema
answers, ignore_eos, the 400 exceed_context_size_error when a prompt does not fit a slot), and
/metrics with llama.cpp's names. At most -np requests run at once; the rest wait.

  LLAMA_SERVER=deploy/mock_llama.py deploy/mac.sh start
  MOCK_LOAD_S  seconds of pretend loading before /health answers (default 1)
  MOCK_TPS     tokens per second per request (default 400)
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--help" in sys.argv[1:]:
    print("usage: llama-server [options]\n  --reasoning-format FORMAT\n  --reasoning-budget-message MESSAGE\n"
          "  -ctxcp, --ctx-checkpoints, --swa-checkpoints N\n  -cram, --cache-ram N\n  -np, --parallel N")
    sys.exit(0)
if "--version" in sys.argv[1:]:
    print("version: 0 (mock)")
    sys.exit(0)


def args(argv):
    out = {"port": 8080, "host": "127.0.0.1", "alias": None, "np": 4, "ctx": 4096, "model": "mock.gguf"}
    names = {"--port": "port", "--host": "host", "--alias": "alias", "-a": "alias", "-np": "np", "--parallel": "np", "-c": "ctx", "--ctx-size": "ctx", "-m": "model", "--model": "model"}
    for i, a in enumerate(argv[:-1]):
        if a in names:
            out[names[a]] = argv[i + 1]
    out["port"], out["np"], out["ctx"] = int(out["port"]), int(out["np"]), int(out["ctx"])
    out["alias"] = out["alias"] or os.path.basename(out["model"])
    return out


A = args(sys.argv[1:])
SLOT_CTX = A["ctx"] // A["np"]
LOCK = threading.Lock()
SLOTS = threading.Semaphore(A["np"])
STATS = {"running": 0, "waiting": 0, "generated": 0, "prompt": 0, "cached": 0}
READY_AT = time.time() + float(os.environ.get("MOCK_LOAD_S", "1"))
TPS = float(os.environ.get("MOCK_TPS", "400"))


def instance(schema):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        return {k: instance(props[k]) for k in schema.get("required", list(props))}
    if t == "array":
        return [instance(schema.get("items", {"type": "string"})) for _ in range(max(schema.get("minItems", 0), 2))]
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
        data = (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if time.time() < READY_AT:
            return self.send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        if self.path == "/health":
            return self.send(200, {"status": "ok"})
        if self.path == "/metrics":
            with LOCK:
                s = dict(STATS)
            lines = [f"llamacpp:requests_processing {s['running']}", f"llamacpp:requests_deferred {s['waiting']}",
                     f"llamacpp:tokens_predicted_total {s['generated']}", f"llamacpp:prompt_tokens_total {s['prompt']}",
                     f"llamacpp:prompt_tokens_cached_total {s['cached']}"]
            return self.send(200, "\n".join(lines) + "\n", "text/plain")
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": A["alias"], "object": "model", "owned_by": "llamacpp"}]})
        self.send(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if time.time() < READY_AT:
            return self.send(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        if self.path != "/v1/chat/completions":
            return self.send(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
        prompt = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4 + 8
        if prompt >= SLOT_CTX:
            return self.send(400, {"error": {"code": 400, "message": f"request ({prompt} tokens) exceeds the available context size ({SLOT_CTX} tokens), try increasing it",
                                             "type": "exceed_context_size_error", "n_prompt_tokens": prompt, "n_ctx": SLOT_CTX}})
        # llama-server writes until the slot is full, whatever max_tokens says beyond it
        max_tokens = min(int(body.get("max_tokens") or 256), SLOT_CTX - prompt)
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        budget = body.get("thinking_budget_tokens", body.get("reasoning_budget_tokens", -1))
        rf = body.get("response_format") or {}
        with LOCK:
            STATS["waiting"] += 1
        SLOTS.acquire()
        with LOCK:
            STATS["waiting"] -= 1
            STATS["running"] += 1
        try:
            think = 0 if not thinking else (24 if budget is None or budget < 0 else min(24, budget))
            if rf.get("type") == "json_schema":
                content = json.dumps(instance(rf["json_schema"]["schema"]))
                answer = len(content) // 4 + 1
            else:
                content = "```python\ndef add(a, b):\n    return a + b\n```"
                answer = max_tokens if body.get("ignore_eos") else 24
            toks = min(max_tokens, think + answer)
            cut = toks < think + answer or body.get("ignore_eos")
            if thinking and toks <= think:
                content = ""
            time.sleep(toks / TPS)
        finally:
            with LOCK:
                STATS["running"] -= 1
                STATS["generated"] += toks
                STATS["prompt"] += prompt // 2
                STATS["cached"] += prompt - prompt // 2
            SLOTS.release()
        msg = {"role": "assistant", "content": content}
        if thinking:
            msg["reasoning_content"] = "The user asks something; think it through briefly."
        self.send(200, {
            "choices": [{"finish_reason": "length" if cut else "stop", "index": 0, "message": msg}],
            "created": int(time.time()), "model": A["alias"], "object": "chat.completion", "id": "chatcmpl-mock",
            "usage": {"completion_tokens": toks, "prompt_tokens": prompt, "total_tokens": prompt + toks, "prompt_tokens_details": {"cached_tokens": prompt - prompt // 2}},
            "timings": {"cache_n": prompt - prompt // 2, "prompt_n": prompt // 2, "predicted_n": toks},
        })


if __name__ == "__main__":
    print(f"mock llama-server for {A['alias']} on {A['host']}:{A['port']}, {A['np']} slots of {SLOT_CTX} tokens", flush=True)
    print(f"srv    load_model: initializing, n_slots = {A['np']}, n_ctx_slot = {SLOT_CTX}, kv_unified = 'false'", flush=True)
    ThreadingHTTPServer.request_queue_size = 128
    ThreadingHTTPServer((A["host"], A["port"]), H).serve_forever()
