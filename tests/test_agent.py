"""Centipede's agent helpers (stages/agent.py) without a GPU: the round's task pool from
data jobs, the round model's passing answers as training rows, chat rows as trace text,
and the 40-agent serving check against a stand-in llama-server."""

import json
import stat
import sys
import zlib
from pathlib import Path

import pytest

from stages import agent, codebench, harness, sandbox, traces

ROOT = Path(__file__).resolve().parents[1]


def data_job(d: Path, lang: str, tasks: list[tuple[str, str]], held: list[str]) -> Path:
    (d / "work").mkdir(parents=True)
    (d / "data").mkdir()
    (d / "taskspec.json").write_text(json.dumps({"language": lang}))
    agent.write_jsonl(d / "work" / "data_inputs.jsonl", [{"input": s, "tests": t} for s, t in tasks])
    agent.write_jsonl(d / "data" / "heldout.jsonl", [{"input": s, "tests": "x"} for s in held])
    return d


TASK = "Write a Python function `double(x)` that returns twice x."
TESTS = "assert double(2) == 4\nassert double(-1) == -2"


def test_pool_takes_the_tasks_with_tests_but_not_the_held_out_ones(tmp_path):
    py = data_job(tmp_path / "py", "python", [(TASK, TESTS), ("Held task", "assert True"), ("No tests", ""),
                                              (TASK, TESTS)], held=["Held task"])
    c = data_job(tmp_path / "c", "c", [(f"C task {i}", "int main(void){return 0;}") for i in range(5)], held=[])
    rows = agent.pool([py, c], per_language=3)
    assert [r["language"] for r in rows] == ["python"] + ["c"] * 3
    assert rows[0] == {"id": f"python-{zlib.crc32(TASK.encode()):08x}", "language": "python", "prompt": TASK,
                       "tests": TESTS, "raw_prompt": True}
    assert rows == agent.pool([py, c], per_language=3)  # the same draw every time
    with pytest.raises(ValueError):
        agent.pool([data_job(tmp_path / "e", "python", [], held=[])])


def test_round_keeps_the_shortest_passing_unforced_answer_per_task(tmp_path):
    p = {"id": "python-1", "language": "python", "prompt": TASK, "tests": TESTS, "raw_prompt": True}
    agent.write_jsonl(tmp_path / "agent-train.jsonl", [p])
    problems = codebench.load_suite("agent-train", tmp_path)  # as the eval reads the suite
    good = "prog.py\n```python\ndef double(x):\n    return 2 * x\n```\n\nNote: doubles x"
    answers = [
        {"id": "python-1", "sample": 0, "passed": True, "answer": good, "reasoning": "a long way round " * 20},
        {"id": "python-1", "sample": 1, "passed": True, "answer": good, "reasoning": "Twice x."},
        {"id": "python-1", "sample": 2, "passed": True, "answer": good, "reasoning": "x", "forced": "length"},
        {"id": "python-1", "sample": 3, "passed": False, "answer": "prog.py\n```python\nboom\n```", "reasoning": "y"},
        {"id": "other", "sample": 0, "passed": True, "answer": good, "reasoning": "z"},
    ]
    rows = agent.round_rows(answers, problems)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert msgs[:2] == harness.write_messages(codebench.ticket(problems[0]))
    assert msgs[2]["reasoning_content"] == "Twice x."
    assert msgs[2]["content"] == harness.render_files({"prog.py": "def double(x):\n    return 2 * x\n"}, "doubles x")
    # the row's file passes the task's tests, as the eval found
    code = codebench.harness_code(problems[0], msgs[2]["content"])
    assert sandbox.run_many([codebench.job(problems[0], msgs[2]["content"], code)])[0].passed


def test_trace_rows_are_the_prompt_as_at_inference_then_the_thinking_and_answer(tmp_path):
    pytest.importorskip("transformers")
    from test_opd import chat_tokenizer

    tok = chat_tokenizer(tmp_path / "tok")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Read an integer n."},
            {"role": "assistant", "content": "```python\nprint(1)\n```", "reasoning_content": "Let me think."}]
    rows = [{"messages": msgs}, {"messages": msgs, "meta": {"round": 1}},
            {"messages": [msgs[0], {"role": "user", "content": "Read " * 400}, msgs[2]]}]
    out, rep = agent.trace_rows(tok, rows, max_len=256)
    assert rep["rows"] == 2 and rep["cut"] == 1
    prompt = tok(tok.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True, enable_thinking=True),
                 add_special_tokens=False)["input_ids"]
    assert out[0]["prompt"] == prompt and out[0]["id"] == out[1]["id"]
    assert tok.decode(out[0]["tokens"]) == "Let me think.\n</think>\n\n```python\nprint(1)\n```<|end_of_text|>"
    # the file Event Horizon's stages read back
    agent.write_jsonl(tmp_path / "traces.jsonl", out)
    assert traces.load(tmp_path / "traces.jsonl")[0]["tokens"] == out[0]["tokens"]


FAKE_SERVER = r'''#!PYTHON
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
assert sys.argv[sys.argv.index("-np") + 1] == "4" and sys.argv[sys.argv.index("-ctk") + 1] == "q8_0"
canon = [json.loads(l) for l in open(CSET) if l.strip()]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, d):
        b = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.send({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        user = body["messages"][1]["content"]
        p = next(r for r in canon if r["prompt"].strip() in user)
        text = "<think>easy</think>prog.c\n```c\n" + p["canonical"] + "```\n"
        self.send({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 100}, "timings": {"predicted_per_second": 50.0}})

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def test_serve_check_runs_every_agent_at_once_and_scores_their_files(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    exe = llama / "build" / "bin" / "llama-server"
    exe.write_text(FAKE_SERVER.replace("PYTHON", sys.executable).replace("CSET", repr(str(ROOT / "benchmarks" / "c-set.jsonl"))))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    gguf = tmp_path / "agent.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 1000)
    monkeypatch.setattr(agent, "SERVE_SUITES", ("c-set",))
    rep = agent.serve_check(str(gguf), str(llama), str(tmp_path), str(tmp_path / "serve.json"), slots=4, port=18097,
                            health_s=30)
    assert rep["completed"] == rep["requests"] == 4 and not rep["errors"]
    assert rep["wrote_file"] == 4 and rep["passed"] == 4 and rep["tokens"] == 400
    assert rep["tok_s_per_agent_median"] == 50.0 and rep["kv"] == "q8_0"
    assert json.loads((tmp_path / "serve.json").read_text())["passed"] == 4
