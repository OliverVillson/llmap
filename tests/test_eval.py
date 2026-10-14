"""The eval stage's reading rules: streamed requests and what a dead or stalled server
does, errored answers, the partial file and resume, which problems and seeds are asked,
the sandbox items, provenance and the size override, llama-server's slots, and stopping
the server's whole process group. A local fake server and fakes; no GPU, vLLM or
llama.cpp needed."""

import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stages import codebench, sandbox
from stages import eval as ev
from stages._util import Config, Job, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Module state back to a fresh eval, and no waiting between tries."""
    for name, value in {"VLLM": {}, "IN_FLIGHT": {}, "_http": {}, "_SERVER": {}, "_last_event": [0.0],
                        "_stopping": threading.Event(), "BACKOFF": 0.0, "HEALTH_WAIT": 0.0}.items():
        monkeypatch.setattr(ev, name, value)
    monkeypatch.delenv("LOBBOT_EVAL_SLOTS", raising=False)


class FakeServer:
    """A local server. GET /health answers self.health. POST calls reply(path, body): a
    dict is sent as JSON, (status, dict) with that status, and anything else is
    iterated as a server-sent-events stream: a dict is one event, a number a pause of
    that many seconds, and the stream ends without [DONE] unless the reply sends it."""

    def __init__(self, reply):
        self.reply, self.health, self.bodies = reply, 200, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(fake.health)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append((self.path, body))
                got = fake.reply(self.path, body)
                status, got = got if isinstance(got, tuple) else (200, got)
                self.send_response(status)
                if isinstance(got, dict):
                    data = json.dumps(got).encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for e in got:
                        if isinstance(e, (int, float)):
                            time.sleep(e)
                            continue
                        self.wfile.write(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def sent(self, path: str) -> list[dict]:
        return [b for p, b in self.bodies if p == path]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    made = []

    def start(reply) -> FakeServer:
        fake = FakeServer(reply)
        made.append(fake)
        monkeypatch.setattr(ev, "PORT", fake.port)
        return fake

    yield start
    for fake in made:
        fake.close()


def chat_stream(content="", reasoning="", finish="stop", tokens=10):
    """A chat completion as llama-server streams it."""
    return [{"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {"choices": [], "usage": {"completion_tokens": tokens}, "timings": {"predicted_per_second": 50.0}},
            "[DONE]"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------- requests ----------


def test_a_streamed_answer_comes_together_with_its_reasoning_usage_and_sampling(server):
    fake = server(lambda path, body: chat_stream("```py\nx = 1\n```", "plan it", tokens=42))
    infos = []
    answers, tps = ev.generate("sys", ["a"], 4096, 0.6, 7, thinking=True, infos=infos, keep_reasoning=True)
    assert answers == ["```py\nx = 1\n```"] and tps == 50.0
    assert infos == [{"finish": "stop", "tokens": 42, "reasoning_chars": 7, "forced": None, "reasoning": "plan it"}]
    (body,) = fake.sent("/v1/chat/completions")
    assert body["stream"] and body["stream_options"] == {"include_usage": True}
    # llama-server's own min_p (0.05) never applies: both engines sample alike
    assert (body["seed"], body["top_p"], body["top_k"], body["min_p"], body["repeat_penalty"]) == (7, 0.95, 20, 0.0, 1.0)


def test_sampling_names_the_penalty_in_each_engines_field(monkeypatch):
    monkeypatch.setattr(ev, "MIN_P", 0.0)
    assert ev.sampling_params(3, False) == {"seed": 3, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0}
    monkeypatch.setattr(ev, "VLLM", {"name": "m", "dir": "/m"})
    assert ev.sampling_params(None, True) == {"top_k": ev.THINK_TOP_K, "min_p": 0.0, "repetition_penalty": 1.0}


def test_an_answer_cut_after_its_thinking_is_counted_and_left_as_it_is(server):
    fake = server(lambda path, body: chat_stream("```py\nx = ", "done thinking", finish="length", tokens=4096))
    infos = []
    answers, _ = ev.generate("sys", ["a"], 4096, 0.6, 0, thinking=True, infos=infos)
    assert answers == ["```py\nx ="] and infos[0]["answer_cut"] and infos[0]["forced"] is None
    assert [p for p, _ in fake.bodies] == ["/v1/chat/completions"]  # nothing forced or continued


def test_nothing_listening_raises_instead_of_an_empty_answer(monkeypatch):
    monkeypatch.setattr(ev, "PORT", free_port())
    with pytest.raises(ev.ServerDead):
        ev.generate("sys", ["a", "b"], 4096)


def test_an_exited_server_process_raises_even_while_the_port_answers(server):
    fake = server(lambda path, body: [{"choices": [{"delta": {"content": "x"}}]}])  # cut short, no finish
    ev._SERVER["proc"] = subprocess.Popen(["true"])
    ev._SERVER["proc"].wait()
    with pytest.raises(ev.ServerDead, match="StreamCut"):
        ev.generate("sys", ["a"], 4096)
    assert len(fake.bodies) == 1  # no retry once the process is gone


def test_a_cut_stream_or_5xx_is_retried_while_the_server_lives_and_a_4xx_is_not(server):
    tries = {"cut": 0, "busy": 0}

    def reply(path, body):
        q = body["messages"][1]["content"]
        if q == "bad":
            return 400, {"error": "prompt too long"}
        tries[q] += 1
        if tries[q] == 1:
            return [{"choices": [{"delta": {"content": "x"}}]}] if q == "cut" else (503, {"error": "loading"})
        return chat_stream("ok")

    server(reply)
    infos = []
    answers, _ = ev.generate("sys", ["cut", "busy", "bad"], 4096, infos=infos)
    assert answers == ["ok", "ok", ""] and tries == {"cut": 2, "busy": 2}
    assert infos[2]["finish"] == "error" and "HTTP 400" in infos[2]["error"] and ev.errored(infos[2])


def test_a_request_queued_behind_others_is_asked_again_without_counting(server, monkeypatch):
    """vLLM sends nothing to a request it has queued: going IDLE_TIMEOUT s without a byte
    while another answer streams tokens is a wait, not a failure (with no retries left
    it still gets its answer); the same silence with nothing streaming anywhere is a
    stall, which counts."""
    monkeypatch.setattr(ev, "IDLE_TIMEOUT", 1.0)
    monkeypatch.setattr(ev, "RETRIES", 0)
    tries = {"busy": 0, "queued": 0, "stalled": 0}

    def reply(path, body):
        q = body["messages"][1]["content"]
        tries[q] += 1
        if q == "busy":  # a long answer, a token every 0.1 s
            return [x for _ in range(40) for x in (0.1, {"choices": [{"delta": {"content": "."}}]})] + chat_stream("ok")
        if tries[q] == 1:
            return [3.0] + chat_stream("late")
        return chat_stream("ok")

    server(reply)
    answers, _ = ev.generate("sys", ["busy", "queued"], 4096)
    assert answers == ["." * 40 + "ok", "ok"] and tries["queued"] == 2
    time.sleep(ev.IDLE_TIMEOUT)  # nothing streams any more
    infos = []
    answers, _ = ev.generate("sys", ["stalled"], 4096, infos=infos)
    assert answers == [""] and infos[0]["finish"] == "error" and tries["stalled"] == 1


def test_answer_all_asks_errored_answers_again_then_refuses_to_hand_them_back(monkeypatch):
    calls = []

    def fake_generate(system, inputs, max_tokens, temperature=0.0, seed=None, thinking=False, on_answer=None,
                      infos=None, **kw):
        calls.append(list(inputs))
        bad = {"x"} if len(calls) < 3 else {"never"}
        infos.extend({"finish": "error", "error": "500"} if q in bad | {"never"} else {"finish": "stop"} for q in inputs)
        return [("" if q in bad | {"never"} else q.upper()) for q in inputs], None

    monkeypatch.setattr(ev, "generate", fake_generate)
    assert ev.answer_all("sys", ["a", "x", "b"])[0] == ["A", "X", "B"] and calls == [["a", "x", "b"], ["x"], ["x"]]
    calls.clear()
    with pytest.raises(RuntimeError, match="1 answers still errored"):
        ev.answer_all("sys", ["a", "never"])
    assert calls == [["a", "never"]] + [["never"]] * ev.REASK


# ---------- code_eval: errors, the partial file, selection ----------


def lcb(i: int, **kw) -> dict:
    return {"suite": "livecodebench", "id": f"p{i}", "language": "python", "prompt": f"q{i}", "stub": "", "entry": "",
            "tests": [{"input": "1\n", "output": "1\n"}], **kw}


def thinking_job(tmp_path, **cfg) -> Job:
    job = Job(tmp_path)
    for k, v in {"code_eval_thinking": True, "code_eval_max_tokens": 1000, **cfg}.items():
        setattr(job.config, k, v)
    return job


def pass_good(monkeypatch, seen: list | None = None):
    """Answers starting with "good" pass; seen gets each run_many's items."""
    def run_many(items, timeout=0):
        if seen is not None:
            seen.append(items)
        return [sandbox.Result(it[1].startswith("good"), "" if it[1].startswith("good") else "test_failed")
                for it in items]

    monkeypatch.setattr(ev.codebench, "job", lambda p, a, code=None: (p["language"], a, "", None, {}))
    monkeypatch.setattr(sandbox, "run_many", run_many)


def answering(asked: list, answer=lambda q, seed: "good", fail=lambda q, seed: False, die=lambda q, seed: False):
    """A fake generate: asked gets (problem text, seed) per answer, and each answer goes
    through on_result as it is made. An answer die() picks never comes: once the others
    are in, the call raises ServerDead, as when the server dies with answers in flight."""
    def fake_generate(system, inputs, max_tokens, temperature=0.0, seed=None, thinking=False, on_answer=None,
                      infos=None, keep_reasoning=False, on_result=None):
        got, dead = [], False
        for k, q in enumerate(inputs):
            q = q.split()[0]  # the problem's own text, without build_prompt's instructions
            if die(q, seed):
                dead = True
                continue
            asked.append((q, seed))
            info = ({"finish": "error", "error": "HTTP 500", "forced": None} if fail(q, seed)
                    else {"finish": "stop", "tokens": 100, "forced": None, **({"reasoning": "r"} if keep_reasoning else {})})
            a = "" if fail(q, seed) else answer(q, seed)
            got.append((a, info))
            if on_result:
                on_result(k, a, info)
            if on_answer:
                on_answer()
        if dead:
            raise ev.ServerDead("refused")
        infos.extend(i for _, i in got)
        return [a for a, _ in got], 50.0

    return fake_generate


def test_errored_answers_are_asked_again_and_never_scored(tmp_path, monkeypatch):
    job = thinking_job(tmp_path)
    asked = []
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering(asked, fail=lambda q, seed: q == "q1" and asked.count((q, seed)) == 1))
    out = ev.code_eval(job, "cand", [lcb(0), lcb(1)], "task")
    assert sorted(asked) == [("q0", 0), ("q1", 0), ("q1", 0)] and out["reasked"] == 1
    rows = read_jsonl(tmp_path / "work/code_eval/cand.jsonl")
    assert [r["passed"] for r in rows] == [True, True] and not any(ev.errored(r) for r in rows)
    assert out["error_share"] == 0.0 and out["suites"]["livecodebench"]["pass@1"] == 1.0
    partial = read_jsonl(tmp_path / "work/code_eval/cand.partial.jsonl")
    assert [(r["id"], "error" in r) for r in partial] == [("p0", False), ("p1", True), ("p1", False)]


def test_answers_that_keep_erroring_stop_the_eval_and_a_rerun_asks_only_for_them(tmp_path, monkeypatch):
    asked = []
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering(asked, fail=lambda q, seed: q == "q1"))
    with pytest.raises(RuntimeError, match="1 answers still errored"):
        ev.code_eval(thinking_job(tmp_path), "cand", [lcb(0), lcb(1)], "task")
    assert asked == [("q0", 0)] + [("q1", 0)] * (ev.REASK + 1)
    assert not (tmp_path / "work/code_eval/cand.jsonl").exists()
    asked.clear()
    monkeypatch.setattr(ev, "generate", answering(asked))
    ev.code_eval(thinking_job(tmp_path), "cand", [lcb(0), lcb(1)], "task")
    assert asked == [("q1", 0)]


def test_a_stopped_reading_resumes_per_problem_and_sample(tmp_path, monkeypatch):
    """The server dies after some answers: a re-run asks for exactly the (problem, sample)
    pairs not in the partial file, with their own seeds; rows from other settings, and a
    line cut by the stop, are not used."""
    probs = [lcb(i) for i in range(4)]
    asked = []
    pass_good(monkeypatch)
    first = {("q0", 0), ("q2", 0), ("q1", 1), ("q3", 1)}
    monkeypatch.setattr(ev, "generate", answering(asked, die=lambda q, seed: (q, seed) not in first))
    with pytest.raises(ev.ServerDead):
        ev.code_eval(thinking_job(tmp_path, code_eval_samples=2, code_eval_k=[1, 2]), "cand", probs, "task")
    made = set(asked)
    assert {("q0", 0), ("q2", 0)} <= made <= first  # pass 1 is dropped when it has not begun as pass 0 raises
    partial = tmp_path / "work/code_eval/cand.partial.jsonl"
    with partial.open("a") as f:
        f.write('{"suite": "livecodebench", "id": "p1", "sam')  # cut mid-line
    asked.clear()
    monkeypatch.setattr(ev, "generate", answering(asked))
    out = ev.code_eval(thinking_job(tmp_path, code_eval_samples=2, code_eval_k=[1, 2]), "cand", probs, "task")
    assert sorted(asked) == sorted({(f"q{j}", i) for j in range(4) for i in range(2)} - made)
    rows = read_jsonl(tmp_path / "work/code_eval/cand.jsonl")
    assert [(r["id"], r["sample"], r["seed"]) for r in rows] == [(f"p{j}", i, i) for i in range(2) for j in range(4)]
    assert out["answers"] == 8 and out["suites"]["livecodebench"]["pass@2"] == 1.0
    # Answers made at another temperature are a different reading: all asked again.
    asked.clear()
    ev.code_eval(thinking_job(tmp_path, code_eval_samples=2, code_eval_k=[1, 2], code_eval_thinking_temperature=1.0),
                 "cand", probs, "task")
    assert len(asked) == 8


def test_a_fingerprinted_model_resumes_on_a_new_vm(tmp_path, monkeypatch):
    """orchestration#6: a new VM downloads or rebuilds the same model with new mtimes. With
    the caller's eval_model_fingerprint the partial answers still resume there; a different
    fingerprint, or none and a new mtime, is another model and asks everything again."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"w" * 64)
    probs = [lcb(i) for i in range(3)]
    asked = []
    pass_good(monkeypatch)

    def read(**cfg):
        return ev.code_eval(thinking_job(tmp_path / "job", **cfg), "cand", probs, "task", path=str(model))

    def new_vm():
        os.utime(model / "model.safetensors", (time.time() + 3600, time.time() + 3600))

    monkeypatch.setattr(ev, "generate", answering(asked, die=lambda q, seed: q == "q2"))
    with pytest.raises(ev.ServerDead):
        read(eval_model_fingerprint="hf@abc:123")
    new_vm()
    asked.clear()
    monkeypatch.setattr(ev, "generate", answering(asked))
    read(eval_model_fingerprint="hf@abc:123")
    assert asked == [("q2", 0)]
    cfg = Config(eval_model_fingerprint="hf@abc:123")
    key = ev.run_key(cfg, str(model), 0.6, False)
    new_vm()
    assert ev.run_key(cfg, str(model), 0.6, False) == key
    assert ev.run_key(Config(eval_model_fingerprint="hf@abd:123"), str(model), 0.6, False) != key
    (model / "model.safetensors").write_bytes(b"w" * 65)  # the size still counts
    assert ev.run_key(cfg, str(model), 0.6, False) != key
    # Without a fingerprint the mtime is what tells models apart, as before.
    plain = ev.run_key(Config(), str(model), 0.6, False)
    new_vm()
    assert ev.run_key(Config(), str(model), 0.6, False) != plain


def test_answers_per_problem_seed_offset_and_k(tmp_path, monkeypatch):
    """code_eval_samples_by_id gives p0 three answers and p3 one, and leaves p1 out; p2
    gets code_eval_samples. Answer i draws with seed code_eval_seed_offset + i, and pass@k
    is reported only for k every problem has answers for."""
    job = thinking_job(tmp_path, code_eval_samples=2, code_eval_k=[1, 2], code_eval_seed_offset=1000,
                       code_eval_samples_by_id={"p0": 3, "p1": 0, "p3": 1})
    asked = []
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering(asked, answer=lambda q, seed: "good" if seed == 1000 else "bad"))
    out = ev.code_eval(job, "cand", [lcb(i) for i in range(4)], "task")
    want = [("p0", 1000), ("p0", 1001), ("p0", 1002), ("p2", 1000), ("p2", 1001), ("p3", 1000)]
    assert sorted(asked) == [(p.replace("p", "q"), seed) for p, seed in want]
    rows = read_jsonl(tmp_path / "work/code_eval/cand.jsonl")
    assert sorted((r["id"], r["seed"]) for r in rows) == want
    s = out["suites"]["livecodebench"]
    assert s["n"] == 3 and s["pass@1"] == round((1 / 3 + 1 / 2 + 1) / 3, 4) and "pass@2" not in s
    assert out["samples_per_problem"] == 3 and out["seed_offset"] == 1000 and out["answers"] == 6
    assert s["ids_sha256"] == hashlib.sha256(b"p0\np2\np3").hexdigest()  # fetch_codebench's definition


def test_the_sandbox_gets_codebench_jobs_once_the_server_is_stopped(tmp_path, monkeypatch):
    """Every answer is in and release() has stopped the server before the first test runs;
    each item is codebench.job's 5-tuple, its cases in files["cases.json"]."""
    job = thinking_job(tmp_path, code_eval_samples=2)
    events, seen = [], []
    cases = [{"input": f"{i}\n", "output": f"{i}\n"} for i in range(3)]
    probs = [lcb(0, tests=cases, public=1), lcb(1, tests={"func": "f", "cases": [["1", "1"]]}, public=1)]

    def run_many(items, timeout=0):
        events.append("sandbox")
        seen.extend(items)
        return [sandbox.Result(True, "") for _ in items]

    monkeypatch.setattr(sandbox, "run_many", run_many)
    monkeypatch.setattr(ev, "generate", answering([], answer=lambda q, seed: "```python\nprint(input())\n```"))
    ev.code_eval(job, "cand", probs, "task", start=lambda: events.append("start"),
                 release=lambda: events.append("release"))
    assert events[:2] == ["start", "release"] and set(events[2:]) == {"sandbox"}
    assert len(seen) == 4 and all(len(it) == 5 for it in seen)
    stdin = [it for it in seen if it[2] == ""]
    assert len(stdin) == 2 and json.loads(stdin[0][4][codebench.CASES_FILE]) == [[c["input"], c["output"]] for c in cases]
    assert stdin[0][3] == codebench.CASE_TIMEOUT * 3 + 10 and "print(input())" in stdin[0][1]


def write_lcb(bench: Path, rows: list[dict]) -> None:
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "livecodebench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_lcb_window_case_cap_and_ids(tmp_path):
    many = [{"input": f"{i}\n", "output": f"{i}\n"} for i in range(60)]
    rows = [{**lcb(i), "suite": None, "language": "py", "date": d, "tests": many, "public": 2}
            for i, d in enumerate(["2024-12-31", "2025-01-01", "2025-04-30", "2025-05-01"])]
    rows.append({**lcb(4), "date": "2025-02-01T00:00:00", "tests": {"func": "f", "cases": [["1", "1"]] * 60}, "public": 3})
    write_lcb(tmp_path / "bench", rows)
    job = Job(tmp_path / "job")
    job.config = Config(code_eval_suites=["livecodebench"], code_eval_dir=str(tmp_path / "bench"),
                        lcb_since="2025-01-01", lcb_until="2025-04-30")
    got = ev.code_problems(job, [])
    assert [p["id"] for p in got] == ["p1", "p2", "p4"] and {p["language"] for p in got} == {"python"}
    assert all(p["suite"] == "livecodebench" for p in got)
    assert len(got[0]["tests"]) == 50 and got[0]["public"] == 2 and len(got[2]["tests"]["cases"]) == 50
    job.config.lcb_max_cases = 0
    assert len(ev.code_problems(job, [])[0]["tests"]) == 60
    job.config.code_eval_ids = ["p4", "p1"]
    assert [p["id"] for p in ev.code_problems(job, [])] == ["p1", "p4"]
    job.config.code_eval_ids = ["p1", "p3"]  # p3 is past lcb_until: a fixed set must be the one scored
    with pytest.raises(ValueError, match="p3"):
        ev.code_problems(job, [])


# ---------- run_stage: the reading as a whole ----------


def code_job(tmp_path, **cfg) -> Job:
    """An eval-only job on two LiveCodeBench problems and one GGUF candidate."""
    root = tmp_path / "job"
    root.mkdir()
    (root / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    write_lcb(tmp_path / "bench", [{**lcb(i), "date": "2025-02-01"} for i in range(2)])
    (tmp_path / "bench/livecodebench.meta.json").write_text(json.dumps({
        "repo": "livecodebench/code_generation_lite", "sha": "abc", "max_cases": 0,
        "windows": {"test": {"since": "2025-01-01", "until": "2025-04-30", "n": 2, "ids": ["p0", "p1"]}}}))
    gguf = tmp_path / "m.gguf"
    with open(gguf, "wb") as f:
        f.truncate(7_300_000_000)  # sparse: the bf16-sized file
    (root / "config.json").write_text(json.dumps({
        "eval_candidates": {"m": str(gguf)}, "code_eval_suites": ["livecodebench"], "code_eval_dir": str(tmp_path / "bench"),
        "lcb_since": "2025-01-01", "lcb_until": "2025-04-30", "code_eval_thinking": True, **cfg}))
    return Job(root)


def fake_serving(monkeypatch) -> list:
    """serve() and stop() that only record what they were asked."""
    log = []
    monkeypatch.setattr(ev, "serve", lambda job, path, name: log.append(("serve", name)) or f"proc-{len(log)}")
    monkeypatch.setattr(ev, "stop", lambda proc, wait=None: log.append(("stop", proc)))
    monkeypatch.setattr(ev, "watch_gpu", lambda: lambda: {})
    return log


def test_a_reading_is_not_done_while_answers_errored(tmp_path, monkeypatch):
    job = code_job(tmp_path)
    log, asked = fake_serving(monkeypatch), []
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering(asked, fail=lambda q, seed: q == "q1"))
    with pytest.raises(RuntimeError, match="still errored"):
        ev.run_stage(job)
    assert not job.is_done("eval") and not job.path("out", "eval.json").exists()
    assert log == [("serve", "m"), ("stop", "proc-1")]  # stopped on the way out
    partial = job.path("work", "code_eval", "m.partial.jsonl")
    assert partial.exists()
    asked.clear()
    monkeypatch.setattr(ev, "generate", answering(asked))
    ev.run_stage(job)
    assert asked == [("q1", 0)] and job.is_done("eval") and not partial.exists()


def test_eval_json_has_the_size_override_provenance_and_the_problem_set(tmp_path, monkeypatch):
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    git = ["git", "-C", str(llama), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "pin"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    job = code_job(tmp_path, size_gb_override={"m": 1.172}, llama_cpp=str(llama))
    fake_serving(monkeypatch)
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering([]))
    ev.run_stage(job)
    rep = json.loads(job.path("out", "eval.json").read_text())
    (cand,) = rep["candidates"]
    assert cand["size_gb"] == 1.172  # not the 7.3 GB of the file served
    prov = json.loads(job.path("work", "provenance.json").read_text())
    assert rep["provenance"] == prov and prov["llama_cpp"] == sha and prov["nproc"] == os.cpu_count()
    assert prov["git_head"] == subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                              text=True).stdout.strip()
    assert {"gpu", "physical_cores", "sandbox_workers", "eval_venv", "vllm_venv", "env"} <= set(prov)
    assert prov["eval_venv"]["transformers"]  # installed here
    meta = prov["code_suites"]["livecodebench"]
    assert meta["sha"] == "abc" and meta["windows"]["test"] == {"since": "2025-01-01", "until": "2025-04-30", "n": 2}
    s = cand["code"]["suites"]["livecodebench"]
    assert s["ids_sha256"] == hashlib.sha256(b"p0\np1").hexdigest()
    assert (s["since"], s["until"], s["max_cases"]) == ("2025-01-01", "2025-04-30", 50)
    assert {"error_share", "cap_hit_share", "forced_share", "answer_cut_share", "case_timeout_share"} <= set(cand["code"])
    assert json.loads(job.path("work", "config.effective.json").read_text())["size_gb_override"] == {"m": 1.172}


def test_every_answer_in_the_partial_file_starts_no_server(tmp_path, monkeypatch):
    """A stop during the sandbox leaves every answer in the partial file: the re-run tests
    them without loading the model again."""
    job = code_job(tmp_path)
    log = fake_serving(monkeypatch)
    monkeypatch.setattr(ev, "generate", answering([]))
    monkeypatch.setattr(ev.codebench, "job", lambda p, a, code=None: (p["language"], a, "", None, {}))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        ev.run_stage(job)
    assert log == [("serve", "m"), ("stop", "proc-1")]
    log.clear()
    pass_good(monkeypatch)
    monkeypatch.setattr(ev, "generate", lambda *a, **k: pytest.fail("asked again"))
    ev.run_stage(job)
    assert log == [] and job.is_done("eval")
    assert json.loads(job.path("out", "eval.json").read_text())["candidates"][0]["gpu"] == {}


def test_a_server_that_dies_is_started_again_and_the_answers_so_far_kept(tmp_path, monkeypatch):
    job = code_job(tmp_path)
    log, asked = fake_serving(monkeypatch), []
    pass_good(monkeypatch)
    deaths = []

    def die(q, seed):
        if q == "q1" and not deaths:
            deaths.append(q)
            return True
        return False

    monkeypatch.setattr(ev, "generate", answering(asked, die=die))
    ev.run_stage(job)
    assert log == [("serve", "m"), ("stop", "proc-1"), ("serve", "m"), ("stop", "proc-3")]
    assert asked == [("q0", 0), ("q1", 0)] and job.is_done("eval")
    monkeypatch.setattr(ev, "RESTARTS", 0)  # with no restarts left the death stops the stage
    deaths.clear()
    (job.root / ".done/eval").unlink()
    with pytest.raises(ev.ServerDead):
        ev.run_stage(job)


# ---------- servers: slots, vLLM args, stopping ----------


def small_gguf(path: Path, size: int) -> Path:
    """A GGUF header with what sizes the KV cache (a small model's shape), padded to size bytes."""
    s = lambda b: struct.pack("<Q", len(b)) + b
    u32 = lambda k, v: s(k.encode()) + struct.pack("<II", 4, v)
    kvs = [s(b"general.architecture") + struct.pack("<I", 8) + s(b"arch"),
           u32("arch.block_count", 40), u32("arch.attention.head_count", 40),
           u32("arch.attention.head_count_kv", 8), u32("arch.embedding_length", 2560)]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs))
    with open(path, "r+b") as f:
        f.truncate(size)
    return path


def test_llama_server_slots_come_from_free_gpu_memory(tmp_path, monkeypatch):
    """40 layers x 8 KV heads x (64 + 64) x 2 bytes = 80 KiB a token in f16; a slot holds
    16384 thinking + 4096 forced + 4096 prompt tokens. On a free 46 GB GPU with a 1.172 GB
    file that is 21 slots (about 20, as planned), 40 with a q8_0 cache."""
    monkeypatch.delenv("LOBBOT_EVAL_CTX", raising=False)
    monkeypatch.delenv("LOBBOT_EVAL_MAX_TOKENS", raising=False)
    gguf = str(small_gguf(tmp_path / "m.gguf", 1_172_000_000))
    cfg = Config(code_eval_suites=["livecodebench"], code_eval_max_tokens=16384, code_eval_thinking=True)
    assert ev.kv_bytes_per_token(ev.gguf_meta(gguf)) == 81920 and ev.limits(cfg)[1] == 24576
    monkeypatch.setattr(ev, "free_vram_bytes", lambda: 46_000_000_000)
    assert ev.llama_slots(cfg, gguf) == int((46e9 - 1.172e9 - ev.SLOT_RESERVE_GB * 1e9) // (24576 * 81920)) == 21
    args = ev.server_args(cfg, gguf)
    assert args[args.index("-np") + 1] == "21" and args[args.index("-c") + 1] == str(21 * 24576)
    cfg.eval_kv_type = "q8_0"
    assert ev.llama_slots(cfg, gguf) == 40
    monkeypatch.setattr(ev, "free_vram_bytes", lambda: None)  # no nvidia-smi: the fixed count
    assert ev.llama_slots(cfg, gguf) == ev.SLOTS
    monkeypatch.setattr(ev, "free_vram_bytes", lambda: 46_000_000_000)
    monkeypatch.setenv("LOBBOT_EVAL_SLOTS", "16")  # set by hand: it wins
    assert ev.llama_slots(cfg, gguf) == ev.SLOTS


def test_vllm_seqs_come_from_config_and_eval_vllm_args_pass_through(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    cfg = Config(vllm_seqs=96)
    args = ev.vllm_args(cfg, str(ckpt), "m")
    assert args[args.index("--max-num-seqs") + 1] == "96" and ev.vllm_seqs(cfg) == 96
    assert ev.vllm_seqs(Config()) == ev.VLLM_SEQS
    cfg.eval_vllm_args = ["--kv-cache-dtype", "fp8", "--max-num-seqs", "160"]
    args = ev.vllm_args(cfg, str(ckpt), "m")
    assert args[-4:] == cfg.eval_vllm_args and ev.vllm_seqs(cfg) == 160  # vLLM takes the last one
    cfg.eval_vllm_args = ["--max-num-seqs=200"]
    assert ev.vllm_seqs(cfg) == 200


def group_alive(pgid: int) -> bool:
    """Whether any process of the group is still running (zombies are not: pid 1 here may
    never reap them)."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            return True
    return False


# The leader exits on SIGTERM; its child ignores it, as an engine core that hangs would.
STUBBORN = "(trap '' TERM; exec sleep 300) & exec sleep 300"
needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads /proc")


@needs_proc
def test_stop_terms_the_whole_group_then_kills_what_is_left(monkeypatch):
    proc = subprocess.Popen(["bash", "-c", STUBBORN], start_new_session=True)
    time.sleep(0.3)
    assert group_alive(proc.pid)
    start = time.monotonic()
    ev.stop(proc, wait=1.0)
    assert not group_alive(proc.pid) and proc.returncode == -signal.SIGTERM and time.monotonic() - start >= 1.0


@needs_proc
def test_a_server_that_never_comes_up_is_stopped_with_its_group(tmp_path, monkeypatch):
    binary = tmp_path / "llama.cpp/build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text(f"#!/bin/bash\n{STUBBORN}\n")
    binary.chmod(0o755)
    job = Job(tmp_path / "job")
    job.config.llama_cpp = str(tmp_path / "llama.cpp")
    monkeypatch.setattr(ev, "PORT", free_port())
    monkeypatch.setattr(ev, "HEALTH_TIMEOUT", {"llama-server": 1, "vllm": 1})
    monkeypatch.setattr(ev, "STOP_WAIT", 1.0)
    started = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(ev.subprocess, "Popen", lambda *a, **k: started.append(real_popen(*a, **k)) or started[-1])
    with pytest.raises(RuntimeError, match="did not become healthy"):
        ev.serve(job, str(tmp_path / "m.gguf"), "m")
    assert not group_alive(started[0].pid)


def test_a_server_left_on_the_port_is_refused(tmp_path, server):
    """A server an earlier eval left running would pass /health for the new one and answer
    in its place: serve() refuses to start."""
    server(lambda path, body: {})
    job = Job(tmp_path / "job")
    with pytest.raises(RuntimeError, match="already in use"):
        ev.serve(job, str(tmp_path / "m.gguf"), "m")
    assert not ev._SERVER


@needs_proc
def test_sigterm_stops_the_server_group_and_runs_the_finally_blocks(tmp_path):
    script = (
        "import subprocess, time\n"
        "from stages import eval as ev\n"
        "ev.SIGTERM_WAIT = 1.0\n"
        "ev.raise_on_sigterm()\n"
        f"proc = subprocess.Popen(['bash', '-c', {STUBBORN!r}], start_new_session=True)\n"
        "ev._SERVER['proc'] = proc\n"
        "print(proc.pid, flush=True)\n"
        "try:\n"
        "    time.sleep(60)\n"
        "finally:\n"
        "    print('finally', ev._stopping.is_set(), flush=True)\n")
    child = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    pgid = int(child.stdout.readline())
    time.sleep(0.3)
    child.send_signal(signal.SIGTERM)
    out, _ = child.communicate(timeout=30)
    assert child.returncode == 128 + signal.SIGTERM and out.strip() == "finally True"
    assert not group_alive(pgid)
