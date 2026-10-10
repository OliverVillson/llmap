"""scripts/polyglot.py on fakes: no docker, GPU or vLLM. The logging proxy against a fake
upstream, the training exercises' slug exclusion, aider's results and a logged run turned into
chat rows, the container command and the run's wiring."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import polyglot as pg  # noqa: E402

# aider benchmark/prompts.py, verbatim: what the harness appends to the exercise and to the test output.
INSTRUCTIONS_ADDENDUM = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""  # noqa: E501
TEST_FAILURES = """
####

See the testing errors above.
The tests are correct, don't try and change them.
Fix the code in {file_list} to resolve the errors.
"""


# ---------- the logging proxy ----------


class Upstream(BaseHTTPRequestHandler):
    """A fake vLLM: a chat completion with its reasoning apart, a model list, an SSE stream, a 500."""
    seen: list = []

    def log_message(self, *a):
        pass

    def reply(self, status, body: bytes, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        Upstream.seen.append(("GET", self.path, None))
        self.reply(200, json.dumps({"data": [{"id": "cand"}]}).encode())

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Upstream.seen.append(("POST", self.path, req))
        if req.get("fail"):
            return self.reply(500, b'{"error": "boom"}')
        if req.get("stream"):
            return self.reply(200, b'data: {"choices": [{"delta": {"content": "x"}}]}\n\ndata: [DONE]\n\n',
                              "text/event-stream")
        self.reply(200, json.dumps({
            "id": "c1", "object": "chat.completion", "model": req["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "the edit", "reasoning_content": "the plan"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}}).encode())


@pytest.fixture
def upstream():
    Upstream.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_proxy_forwards_and_logs_chat_completions(tmp_path, upstream):
    log = tmp_path / "log.jsonl"
    proxy = pg.LoggingProxy(upstream, log)
    try:
        req = {"model": "cand", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
               "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768,
               "chat_template_kwargs": {"enable_thinking": True}}
        r = httpx.post(proxy.url + "/v1/chat/completions", json=req, headers={"Authorization": "Bearer local"})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "the edit"
        assert Upstream.seen[-1] == ("POST", "/v1/chat/completions", req)  # forwarded as sent
        assert httpx.get(proxy.url + "/v1/models").json() == {"data": [{"id": "cand"}]}
        streamed = httpx.post(proxy.url + "/v1/chat/completions", json={**req, "stream": True})
        assert streamed.text.startswith("data: ") and streamed.text.endswith("data: [DONE]\n\n")  # relayed
        assert httpx.post(proxy.url + "/v1/chat/completions", json={**req, "fail": True}).status_code == 500
        assert httpx.get(proxy.url + "/health").status_code == 404  # only /v1/*
    finally:
        proxy.close()
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 1  # the one whole completion: not the model list, the stream or the error
    e = lines[0]
    assert e["messages"] == req["messages"] and e["model"] == "cand"
    assert (e["reasoning_content"], e["content"], e["finish_reason"]) == ("the plan", "the edit", "stop")
    assert e["usage"]["completion_tokens"] == 34
    assert e["sampling"] == {k: req[k] for k in ("temperature", "top_p", "top_k", "max_tokens", "chat_template_kwargs")}


def test_log_entry_reads_newer_vllm_reasoning_field():
    e = pg.log_entry({"messages": []}, {"choices": [{"finish_reason": "length",
                                                     "message": {"content": None, "reasoning": "long"}}]}, 1.0)
    assert (e["reasoning_content"], e["content"], e["finish_reason"]) == ("long", "", "length")


# ---------- exercises ----------


def exercise(root: Path, lang: str, slug: str, files: dict[str, str] | None = None, docs: str | None = None,
             solution: list[str] | None = None, tests: list[str] | None = None) -> Path:
    """An exercise in Exercism's layout: .docs, .meta/config.json, solution and test files."""
    ext = {"python": ".py", "go": ".go", "java": ".java", "javascript": ".js", "rust": ".rs", "cpp": ".cpp"}[lang]
    name = slug.replace("-", "_")
    solution = solution or [name + ext]
    tests = tests or [f"{name}_test{ext}"]
    ex = root / lang / "exercises" / "practice" / slug
    for f, text in {**{f: "stub\n" for f in solution + tests}, **(files or {})}.items():
        (ex / f).parent.mkdir(parents=True, exist_ok=True)
        (ex / f).write_text(text)
    if docs is not None:
        (ex / ".docs").mkdir(parents=True, exist_ok=True)
        (ex / ".docs" / "instructions.md").write_text(docs)
    (ex / ".meta").mkdir(parents=True, exist_ok=True)
    (ex / ".meta" / "config.json").write_text(json.dumps({"files": {
        "solution": solution, "test": tests, "example": [".meta/example" + ext]}}))
    return ex


def track(root: Path, lang: str, slugs: dict[str, str]) -> Path:
    """An Exercism track: config.json with each practice slug's status, and its exercises."""
    t = root / lang
    t.mkdir(parents=True, exist_ok=True)
    (t / "config.json").write_text(json.dumps({"exercises": {"practice": [
        {"slug": s, "status": st} for s, st in slugs.items() if st != "unlisted"]}}))
    return t


def test_exercism_drops_every_polyglot_slug_in_every_language(tmp_path):
    poly = tmp_path / "polyglot"
    exercise(poly, "python", "two-fer", docs="x")
    exercise(poly, "java", "bob", docs="x")
    src = tmp_path / "src"
    tracks = {"python": track(src, "python", {"two-fer": "active", "bob": "active", "leap": "active",
                                              "old": "deprecated", "nodocs": "active", "stray": "unlisted"}),
              "java": track(src, "java", {"bob": "active", "two-fer": "active", "leap": "beta", "nowrapper": "active"})}
    for slug in ("two-fer", "bob", "leap", "old", "stray"):
        exercise(src, "python", slug, docs=f"# {slug}", files={".gitignore": "x"})
    exercise(src, "python", "nodocs")
    for slug in ("bob", "two-fer", "leap", "nowrapper"):
        ex = exercise(src, "java", slug, docs=f"# {slug}",
                      solution=["src/main/java/A.java"], tests=["src/test/java/ATest.java"])
        if slug != "nowrapper":
            (ex / "gradlew").write_text("#!/bin/sh\n")
            (ex / "gradlew").chmod(0o755)
    out = tmp_path / "out"
    (out / "stale").mkdir(parents=True)  # a rebuild starts clean

    counts = pg.build_exercism(tracks, poly, out)
    assert counts == {"python": {"kept": 1, "dropped": {"in_polyglot": 2, "deprecated": 1, "no_instructions": 1,
                                                        "not_in_track": 1}},
                      "java": {"kept": 1, "dropped": {"in_polyglot": 2, "no_build_file": 1}}}
    assert sorted(str(p.relative_to(out)) for p in out.glob("*/exercises/practice/*")) == [
        "java/exercises/practice/leap", "python/exercises/practice/leap"]
    assert json.loads((out / "counts.json").read_text()) == counts and not (out / "stale").exists()
    assert (out / ".done").exists()  # exp04's mark of a finished build
    leap = out / "python/exercises/practice/leap"
    assert (leap / ".docs/instructions.md").read_text() == "# leap" and not (leap / ".gitignore").exists()
    assert os.access(out / "java/exercises/practice/leap/gradlew", os.X_OK)


def test_why_skip_needs_a_test_command_and_the_solution(tmp_path):
    ex = exercise(tmp_path, "python", "odd", docs="x", tests=["odd_test.txt"])
    assert pg.why_skip("python", ex, "active", set()) == "no_test_command"
    ex = exercise(tmp_path, "python", "nosol", docs="x", solution=["CMakeLists.txt"])  # the harness never edits it
    assert pg.why_skip("python", ex, "active", set()) == "no_solution"
    ex = exercise(tmp_path, "cpp", "nocmake", docs="x")
    assert pg.why_skip("cpp", ex, "active", set()) == "no_build_file"
    (ex / "CMakeLists.txt").write_text("")
    assert pg.why_skip("cpp", ex, "active", set()) == ""


# ---------- aider's results ----------


def results(bench: Path, lang: str, slug: str, **r) -> None:
    d = bench / lang / "exercises" / "practice" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / ".aider.results.json").write_text(json.dumps(r))


def test_summarize_matches_aiders_stats(tmp_path):
    bench = tmp_path / "bench"
    results(bench, "python", "a", tests_outcomes=[True], duration=10, prompt_tokens=5, completion_tokens=7)
    results(bench, "python", "b", tests_outcomes=[False, True], duration=20, num_malformed_responses=2)
    results(bench, "python", "c", tests_outcomes=[False, False], duration=30, test_timeouts=1)
    results(bench, "rust", "d", tests_outcomes=[False, True], num_exhausted_context_windows=1, num_error_outputs=3)
    results(bench, "rust", "e", exception="Traceback ...")  # the harness crashed: a failure, as in aider
    (bench / "rust/exercises/practice/f").mkdir()  # not finished
    (bench / "go/exercises/practice/g").mkdir(parents=True)
    (bench / "go/exercises/practice/g/.aider.results.json").write_text('{"tests_out')  # being written

    s = pg.summarize(bench)
    assert (s["pass_rate_1"], s["pass_rate_2"], s["percent_cases_well_formed"]) == (0.2, 0.6, 0.8)
    assert s["per_language"] == {"python": {"n": 3, "pass_rate_1": 0.3333, "pass_rate_2": 0.6667},
                                 "rust": {"n": 2, "pass_rate_1": 0.0, "pass_rate_2": 0.5}}
    assert (s["n"], s["total"], s["errors"], s["test_timeouts"]) == (5, 7, 1, 1)
    assert (s["exhausted_context_windows"], s["error_outputs"], s["malformed_responses"]) == (1, 3, 2)
    assert (s["prompt_tokens"], s["completion_tokens"], s["seconds_per_case"]) == (5, 7, 12.0)
    assert pg.summarize(tmp_path / "none")["pass_rate_2"] is None


# ---------- a logged run into rows ----------

SYSTEM = "Act as an expert software developer."
EXAMPLES = [{"role": "user", "content": "Change get_factorial() to use math.factorial"},
            {"role": "assistant", "content": "mathweb/flask/app.py\n<<<<<<< SEARCH\n..."}]


def ask(text: str, files: list[str]) -> str:
    return text + INSTRUCTIONS_ADDENDUM.format(file_list=" ".join(files)) + "\n\nReminder: use SEARCH/REPLACE."


def entry(messages, content, reasoning="I think.", finish="stop") -> dict:
    return {"messages": messages, "content": content, "reasoning_content": reasoning, "finish_reason": finish,
            "usage": {"completion_tokens": 9}}


def logged_run(tmp_path):
    """A run dir as aider leaves it, and the log of what the model was asked: leap (python)
    passes at once after a format retry, leap (go, same text, other files) is fixed on the second
    try, pangram fails twice, raindrops is cut at max_tokens, isogram's thinking never closed,
    bob passed but is not in the log, crash is a harness exception."""
    bench = tmp_path / "bench"
    leap_text = "# Instructions\n\nIs the year a leap year?\n"
    for lang, slug, text, outcome in [
        ("python", "leap", leap_text, [True]), ("go", "leap", leap_text, [False, True]),
        ("python", "pangram", "# Instructions\n\nPangram?\n", [False, False]),
        ("python", "raindrops", "# Instructions\n\nRaindrops.\n", [True]),
        ("python", "isogram", "# Instructions\n\nIsogram?\n", [True]),
        ("python", "bob", "# Instructions\n\nBob.\n", [True]),
    ]:
        exercise(bench, lang, slug, docs=text)
        results(bench, lang, slug, tests_outcomes=outcome)
    exercise(bench, "python", "crash", docs="# Crash\n")
    results(bench, "python", "crash", exception="Traceback")
    (bench / "python/exercises/practice/raindrops/.docs/introduction.md").write_text("# Introduction\n\nPling.\n")

    head = [{"role": "system", "content": SYSTEM}, *EXAMPLES]
    files = lambda f: [{"role": "user", "content": f"I have *added these files to the chat*\n\n{f}\n```\nstub\n```"},  # noqa: E731
                       {"role": "assistant", "content": "Ok."}]
    py_leap = {"role": "user", "content": ask(leap_text, ["leap.py"])}
    go_leap = {"role": "user", "content": ask(leap_text, ["leap.go"])}
    go_first = [*head, *files("leap.go"), go_leap]
    go_retry = [*head, {**go_leap, "content": leap_text + INSTRUCTIONS_ADDENDUM.format(file_list="leap.go")},
                {"role": "assistant", "content": "leap.go\n<<<<<<< SEARCH\nwrong"},
                {"role": "user", "content": "I updated the files."}, {"role": "assistant", "content": "Ok."},
                *files("leap.go"),
                {"role": "user", "content": "--- FAIL: TestLeap\n" + TEST_FAILURES.format(file_list="leap.go")}]
    rain = {"role": "user", "content": ask("# Introduction\n\nPling.\n# Instructions\n\nRaindrops.\n", ["raindrops.py"])}
    log = [
        entry([*head, *files("leap.py"), py_leap], "leap.py\n<<<<<<< SEARCH\nnot matching"),
        entry(go_first, "leap.go\n<<<<<<< SEARCH\nwrong"),
        entry([*head, *files("leap.py"), py_leap, {"role": "assistant", "content": "leap.py\n<<<<<<< SEARCH\nnot matching"},
               {"role": "user", "content": "# 1 SEARCH/REPLACE block failed to match!"}], "leap.py\n<<<<<<< SEARCH\nright"),
        entry(go_retry, "<think>\nFix it.\n</think>\n\nleap.go\n<<<<<<< SEARCH\nfixed", reasoning=""),
        entry([*head, *files("raindrops.py"), rain], "raindrops.py\n<<<<<<< SEARCH", finish="length"),
        entry([*head, *files("isogram.py"), {"role": "user", "content": ask("# Instructions\n\nIsogram?\n", ["isogram.py"])}],
              "I keep thinking and never close", reasoning=""),
        entry([*head, *files("pangram.py"), {"role": "user", "content": ask("# Instructions\n\nPangram?\n", ["pangram.py"])}],
              "pangram.py\n<<<<<<< SEARCH\nwrong"),
        entry([{"role": "system", "content": "*Briefly* summarize this partial conversation about programming."},
               {"role": "user", "content": "# USER\n" + py_leap["content"]}], "A summary."),
    ]
    path = tmp_path / "log.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in log))
    return bench, path, head


def test_logged_run_becomes_rows(tmp_path):
    bench, log, head = logged_run(tmp_path)
    rows, stats = pg.rows_from_run(bench, log)
    assert len(rows) == 2
    first, fix = sorted(rows, key=lambda r: len(r["messages"]))
    # leap (python): the last request of its passing try, the format retry and its answer
    assert first["messages"][:3] == head and first["messages"][-2]["content"].startswith("# 1 SEARCH/REPLACE")
    assert first["messages"][-1] == {"role": "assistant", "content": "leap.py\n<<<<<<< SEARCH\nright",
                                     "reasoning_content": "I think."}
    # leap (go): the whole second try, with the failed answer and the test output; tags split off
    roles = [m["role"] for m in fix["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant",
                     "user", "assistant"]
    assert "--- FAIL: TestLeap" in fix["messages"][-2]["content"] and "leap.go" in fix["messages"][-2]["content"]
    assert fix["messages"][-1] == {"role": "assistant", "content": "leap.go\n<<<<<<< SEARCH\nfixed",
                                   "reasoning_content": "Fix it."}
    # each row names its exercise, so calibration can spread across languages (calib_extra_share)
    assert first["meta"] == {"source": "aider", "language": "python", "exercise": "leap"}
    assert fix["meta"] == {"source": "aider", "language": "go", "exercise": "leap"}
    assert stats["rows"] == 2 and stats["fix_rows"] == 1 and stats["requests"] == 8
    assert stats["unmatched_requests"] == 1  # the history summary
    assert stats["per_language"] == {
        "go": {"exercises": 1, "kept": 1, "first_try": 0, "fix": 1, "dropped": {}},
        "python": {"exercises": 6, "kept": 1, "first_try": 1, "fix": 0,
                   "dropped": {"not_logged": 1, "error": 1, "no_thinking_end": 1, "failed": 1, "truncated": 1}}}
    from stages.taskdata import to_messages

    assert all(to_messages(r) == r["messages"] for r in rows)  # heal reads them as they are
    _, stats = pg.rows_from_run(bench, log, max_chars=50)
    assert stats["rows"] == 0 and sum(s["dropped"].get("too_long", 0) for s in stats["per_language"].values()) == 2


def test_rows_cli_writes_rows_and_stats(tmp_path, capsys):
    bench, log, _ = logged_run(tmp_path)
    out = tmp_path / "rows.jsonl"
    res = tmp_path / "teacher.json"  # the run's results name its run dir, whose bench/ the rows come from
    res.write_text(json.dumps({"run_dir": str(tmp_path), "pass_rate_2": 0.7}))
    assert pg.main(["rows", "--results", str(res), "--log", str(log), "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 2
    assert json.loads((tmp_path / "rows.stats.json").read_text())["rows"] == 2
    assert "kept    1 of    6" in capsys.readouterr().out


# ---------- the container and the run ----------


def test_docker_command_and_model_settings(tmp_path):
    cmd = pg.docker_command("qwen36", tmp_path / "run", tmp_path / "ex", "http://127.0.0.1:8092/v1", "diff", 64, 10,
                            "1000:1000", tmp_path / "gradle", ("sudo", "docker"))
    assert cmd[:6] == ["sudo", "docker", "run", "--rm", "--name", "polyglot-qwen36"]
    assert cmd[cmd.index("--network") + 1] == "host" and cmd[-4:-2] == [pg.IMAGE, "bash"]
    mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-v"]
    assert mounts == [f"{tmp_path}/run:/benchmarks", f"{tmp_path}/ex:/exercises:ro", f"{tmp_path}/gradle:/root/.gradle"]
    env = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "-e")
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8092/v1" and env["AIDER_DOCKER"] == "1"
    assert env["AIDER_BENCHMARK_DIR"] == "/benchmarks"
    inner = cmd[-1]
    assert inner.startswith(f'python3 -c "{pg.BOOT}" /benchmarks/bench --model openai/qwen36 --edit-format diff ')
    assert "--threads 64 --exercises-dir /exercises --read-model-settings /benchmarks/model-settings.yml --num-tests 10;" in inner
    assert inner.endswith("chown -R 1000:1000 /benchmarks; exit $rc")
    assert "--num-tests" not in pg.docker_command("q", tmp_path, tmp_path, "http://x/v1")[-1]

    (s,) = pg.model_settings("qwen36", "diff")
    assert s["name"] == "openai/qwen36" and s["edit_format"] == "diff" and s["use_temperature"] == 0.6
    p = s["extra_params"]
    assert (p["max_tokens"], p["top_p"], p["timeout"]) == (32768, 0.95, pg.TIMEOUT) and pg.TIMEOUT > 600
    assert p["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    (s,) = pg.model_settings("gemma", "diff", 1, 0.95, 64)  # Gemma 4's card
    assert s["use_temperature"] == 1.0 and isinstance(s["use_temperature"], float)  # True would mean 0 to aider
    assert s["extra_params"]["extra_body"]["top_k"] == 64
    assert pg.model_metadata("qwen36", 65536)["openai/qwen36"]["max_input_tokens"] == 65536


def test_context_follows_the_models_maximum(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 32768}}))
    assert pg.context(str(tmp_path)) == 32768
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
    assert pg.context(str(tmp_path)) == pg.CTX


def fake_bench(monkeypatch, server_exits=False):
    """serve, the GPU watch and docker as fakes; the container writes two results, runs for one
    check of the server and exits (unless the server has died by then)."""
    calls = {"serve": [], "docker": [], "run": []}

    class Server:
        def poll(self):
            return 1 if server_exits else None

    def serve(job, path, name):
        calls["serve"].append((job, path, name, os.environ.get("LOBBOT_EVAL_CTX")))
        return Server()

    class Container:
        returncode = 0

        def __init__(self, cmd, stdout=None, stderr=None):
            calls["docker"].append(cmd)
            run_dir = Path(next(m for m in cmd if m.endswith(":/benchmarks")).rsplit(":", 1)[0])
            results(run_dir / "bench", "python", "a", tests_outcomes=[True])
            results(run_dir / "bench", "go", "b", tests_outcomes=[False, False])

            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    monkeypatch.setattr(pg.ev, "serve", serve)
    monkeypatch.setattr(pg.ev, "stop", lambda proc: None)
    monkeypatch.setattr(pg.ev, "watch_gpu", lambda: lambda: {"busy_pct": 91.5, "mem_peak_mib": 150000})
    monkeypatch.setattr(pg, "image_exists", lambda: True)
    monkeypatch.setattr(pg, "docker", lambda: ["docker"])
    monkeypatch.setattr(pg.subprocess, "Popen", Container)
    monkeypatch.setattr(pg.subprocess, "run", lambda cmd, **k: calls["run"].append(cmd))  # docker rm -f
    monkeypatch.setattr(pg.time, "sleep", lambda s: None)
    monkeypatch.setenv("LOBBOT_EVAL_CTX", "")
    monkeypatch.delenv("LOBBOT_EVAL_CTX")
    return calls


def model_and_exercises(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 65536}))
    ex = tmp_path / "ex"
    for slug in ("a", "b", "c"):
        (ex / "python/exercises/practice" / slug).mkdir(parents=True)
    return model, ex


def test_run_serves_benchmarks_and_writes_results(tmp_path, monkeypatch, capsys):
    calls = fake_bench(monkeypatch)
    model, ex = model_and_exercises(tmp_path)
    (model / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 65536,
                                                                   "top_k_experts": 8}}))  # Gemma 4's names
    run_dir = tmp_path / "runs/cand"
    args = ["run", "--model-dir", str(model), "--name", "cand", "--exercises", str(ex), "--dir", str(run_dir),
            "--experts-used", "6", "--reasoning-parser", "", "--threads", "8", "--log", str(tmp_path / "logs/cand.jsonl"),
            "--temperature", "1.0", "--top-k", "64"]
    assert pg.main(args) == 0
    assert '"answered": 2, "total": 3' in capsys.readouterr().out  # progress for the runner's bar

    (job, path, name, ctx), = calls["serve"]
    assert (path, name, ctx) == (str(model), "cand", "65536")
    assert job.config.eval_experts_used == 6 and job.config.eval_reasoning_parser == ""
    vllm = pg.ev.vllm_args(job.config, str(model), "cand")  # what serve() starts vLLM with
    assert vllm[vllm.index("--hf-overrides") + 1] == '{"text_config": {"top_k_experts": 6}}'
    assert "--reasoning-parser" not in vllm
    (cmd,) = calls["docker"]
    env = dict(cmd[i + 1].split("=", 1) for i, x in enumerate(cmd) if x == "-e")
    assert env["OPENAI_API_BASE"].startswith("http://127.0.0.1:") and not env["OPENAI_API_BASE"].startswith(
        f"http://127.0.0.1:{pg.ev.PORT}/")  # through the logging proxy
    assert "--threads 8" in cmd[-1]
    (settings,) = json.loads((run_dir / "model-settings.yml").read_text())
    assert settings["name"] == "openai/cand" and settings["use_temperature"] == 1.0
    assert settings["extra_params"]["top_p"] == 0.95 and settings["extra_params"]["extra_body"]["top_k"] == 64
    assert json.loads((run_dir / "model-metadata.json").read_text())["openai/cand"]["max_input_tokens"] == 65536
    res = json.loads((run_dir / "out/polyglot.json").read_text())
    assert {"pass_rate_1", "pass_rate_2", "percent_cases_well_formed", "per_language", "n", "errors", "seconds",
            "gpu"} <= set(res)
    assert (res["pass_rate_1"], res["pass_rate_2"], res["n"], res["errors"]) == (0.5, 0.5, 2, 0)
    assert res["per_language"]["go"] == {"n": 1, "pass_rate_1": 0.0, "pass_rate_2": 0.0}
    assert res["gpu"]["busy_pct"] == 91.5 and res["seconds"] >= 0
    assert res["run_dir"] == str(run_dir.resolve())  # where `rows --results` finds bench/
    assert res["sampling"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

    # a finished run is skipped unless forced
    assert pg.main(args) == 0 and len(calls["serve"]) == 1
    assert pg.main(args + ["--force"]) == 0 and len(calls["serve"]) == 2


def test_run_refuses_a_gguf(tmp_path, monkeypatch):
    fake_bench(monkeypatch)
    ex = tmp_path / "ex/python/exercises/practice/a"
    ex.mkdir(parents=True)
    with pytest.raises(SystemExit, match="not a checkpoint dir"):
        pg.main(["run", "--model-dir", str(tmp_path / "m.gguf"), "--name", "cand", "--exercises", str(tmp_path / "ex"),
                 "--dir", str(tmp_path)])


def test_run_stops_when_vllm_dies(tmp_path, monkeypatch):
    calls = fake_bench(monkeypatch, server_exits=True)
    model, ex = model_and_exercises(tmp_path)
    with pytest.raises(RuntimeError, match="vLLM exited during the benchmark"):
        pg.main(["run", "--model-dir", str(model), "--name", "cand", "--exercises", str(ex), "--dir", str(tmp_path / "run")])
    assert calls["run"][-1] == ["docker", "rm", "-f", "polyglot-cand"]  # the container goes too
    assert not (tmp_path / "run/out/polyglot.json").exists()  # a rerun resumes
