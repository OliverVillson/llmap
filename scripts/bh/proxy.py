"""The recording proxy: a local OpenAI-style endpoint between bench/run.ts and the server,
used by every bench run of the train harness (teacher G2 and answers, student smoke,
score and round-2 sampling). For each chat call it

  - sets the sampling the config says, whatever the caller sent: a finite top_k (run 1's
    top_k 0 scored the SFT agent 0%), top_p, min_p when the base needs it (MiniCPM5: 0),
    thinking on/off (chat_template_kwargs.enable_thinking), and the per-call token cap;
  - asks the teacher for its top-k log-probs of every answer token, as token ids, when
    recording them (the logit KL of the Qwen path), and strips them from the reply, so the
    engine sees an ordinary answer;
  - logs one line per call to calls.jsonl: wire.call_key(user, answer), finish_reason and
    whether the answer was cut at the cap, tokens in and out, whether the answer opened a
    thinking block, and what it overrode. The bench transcripts do not keep
    finish_reason, so this log is how cut answers are kept out of the rows.

Long reads: no timeout under an hour per call (Bun's 300 s fetch cap killed run 1's long
calls; the engine's side is MUGGE_TIMEOUT_S, set by the bench command).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bh.wire import call_key


class Recorder:
    def __init__(self, upstream: str, out: Path, *, top_k: int, top_p: float, temperature: float | None,
                 thinking: bool, max_tokens: int, min_p: float | None = None, logprobs: int = 0):
        self.upstream = upstream.rstrip("/")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.settings = {"top_k": top_k, "top_p": top_p, "temperature": temperature, "thinking": thinking,
                         "max_tokens": max_tokens, "min_p": min_p, "logprobs": logprobs}
        self.lock = threading.Lock()
        self.counts = {"calls": 0, "cut": 0, "errors": 0, "think": 0, "top_k_set": 0, "thinking_set": 0,
                       "capped": 0, "logprob_calls": 0}
        self.calls_f = open(self.out / "calls.jsonl", "a")
        self.topk_f = open(self.out / "topk.jsonl", "a") if logprobs else None

    def close(self) -> None:
        self.calls_f.close()
        if self.topk_f:
            self.topk_f.close()
        (self.out / "proxy.json").write_text(json.dumps({"settings": self.settings, "counts": self.counts}, indent=2))

    def shape(self, body: dict) -> tuple[dict, dict]:
        """The request as it goes upstream, and what was overridden."""
        s, notes = self.settings, {}
        body = dict(body)
        if not isinstance(body.get("top_k"), int) or body["top_k"] <= 0 or body["top_k"] != s["top_k"]:
            notes["top_k"] = body.get("top_k")
            body["top_k"] = s["top_k"]
        body["top_p"] = s["top_p"]
        if s["temperature"] is not None:
            body["temperature"] = s["temperature"]
        if s["min_p"] is not None:
            body["min_p"] = s["min_p"]
        kw = dict(body.get("chat_template_kwargs") or {})
        if kw.get("enable_thinking") is not s["thinking"]:
            notes["thinking"] = kw.get("enable_thinking")
            kw["enable_thinking"] = s["thinking"]
        body["chat_template_kwargs"] = kw
        if int(body.get("max_tokens") or 0) <= 0 or int(body["max_tokens"]) > s["max_tokens"]:
            notes["max_tokens"] = body.get("max_tokens")
            body["max_tokens"] = s["max_tokens"]
        body.pop("stream", None)
        if s["logprobs"]:
            body.update(logprobs=True, top_logprobs=s["logprobs"], return_tokens_as_token_ids=True)
        return body, notes

    def handle(self, body: dict) -> dict:
        up, notes = self.shape(body)
        t0 = time.monotonic()
        req = urllib.request.Request(f"{self.upstream}/v1/chat/completions", data=json.dumps(up).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            reply = json.loads(r.read())
        ms = round(1000 * (time.monotonic() - t0))
        choice = (reply.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        answer = msg.get("content") or ""
        finish = choice.get("finish_reason")
        usage = reply.get("usage") or {}
        user = (body.get("messages") or [{}])[-1].get("content") or ""
        key = call_key(user, answer)
        lp = choice.pop("logprobs", None)
        rec = {"key": key, "finish": finish, "cut": finish == "length", "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"), "ms": ms,
               "think": "<think>" in answer or bool(msg.get("reasoning_content") or msg.get("reasoning")),
               "overrode": notes, "seed": body.get("seed"), "t": round(time.time(), 1)}
        top = topk_record(key, lp) if lp and self.topk_f else None
        with self.lock:
            c = self.counts
            c["calls"] += 1
            c["cut"] += rec["cut"]
            c["think"] += rec["think"]
            c["top_k_set"] += "top_k" in notes
            c["thinking_set"] += "thinking" in notes
            c["capped"] += "max_tokens" in notes
            self.calls_f.write(json.dumps(rec) + "\n")
            self.calls_f.flush()
            if top:
                c["logprob_calls"] += 1
                self.topk_f.write(json.dumps(top) + "\n")
                self.topk_f.flush()
        return reply


def _tid(tok: str) -> int | None:
    """vLLM's return_tokens_as_token_ids form, "token_id:123"."""
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return int(tok[9:])
        except ValueError:
            return None
    return None


def topk_record(key: str, logprobs: dict) -> dict | None:
    """{key, ids: the answer's token ids, top: per token its top-k ids, lp: their
    log-probs} from a chat reply's logprobs (OpenAI shape, ids as token_id:N)."""
    ids, top, lps = [], [], []
    for e in logprobs.get("content") or []:
        t = _tid(e.get("token"))
        if t is None:
            return None
        ids.append(t)
        alts = [(_tid(a.get("token")), a.get("logprob")) for a in e.get("top_logprobs") or []]
        alts = [(i, float(v)) for i, v in alts if i is not None and v is not None]
        top.append([i for i, _ in alts])
        lps.append([round(v, 4) for _, v in alts])
    return {"key": key, "ids": ids, "top": top, "lp": lps}


@contextmanager
def recording(upstream: str, out: Path, **settings):
    """A Recorder behind a local server; yields (its /v1 URL, the recorder)."""
    rec = Recorder(upstream, out, **settings)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    raise LookupError(f"no endpoint {self.path}")
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
                status, reply = 200, rec.handle(body)
            except Exception as e:  # noqa: BLE001  (the engine counts a failed call and goes on)
                with rec.lock:
                    rec.counts["errors"] += 1
                status = 404 if isinstance(e, LookupError) else 502
                reply = {"error": f"{type(e).__name__}: {e}"[:500]}
            data = json.dumps(reply).encode()
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass

        def do_GET(self):  # /v1/models and the like, passed through
            try:
                with urllib.request.urlopen(f"{rec.upstream}{self.path}", timeout=60) as r:
                    status, data = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, data = e.code, e.read()
            except OSError as e:
                status, data = 502, json.dumps({"error": str(e)}).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 512

    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", rec
    finally:
        server.shutdown()
        server.server_close()
        rec.close()
