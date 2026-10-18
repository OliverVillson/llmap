"""MuggeBench from the eval stage: the command it runs against the served candidate, and
the bench CLI itself on the engine's fixture project with the fake model (needs bun)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from stages import eval as ev
from stages._util import Job

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "mugge/test/fixtures/bench-mini"


def job(tmp_path, **config) -> Job:
    root = tmp_path / "job"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"mugge_bench_dir": "/bench", **config}))
    return Job(root)


def test_command_targets_the_served_candidate_with_the_eval_sampling(tmp_path):
    j = job(tmp_path, code_eval_thinking=True, code_eval_thinking_temperature=1.0, thinking_top_k=64,
            mugge_bench_seeds=2, mugge_bench_split="dev")
    cmd = ev.mugge_bench_command(j, "build", tmp_path / "out")
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert cmd[:2] == ["bun", str(ROOT / "mugge/bench/run.ts")]
    assert flags["--url"] == f"http://127.0.0.1:{ev.PORT}/v1" and flags["--model"] == "build"
    assert flags["--bench"] == "/bench" and flags["--seeds"] == "2" and flags["--split"] == "dev"
    assert flags["--thinking"] == "on" and flags["--temperature"] == "1.0" and flags["--top-k"] == "64"
    # each call is capped as a code answer is, and the server's context fits a bench prompt over that
    assert flags["--max-tokens"] == str(ev.call_max_tokens(j.config)) and "--tickets" not in cmd
    quick = ev.mugge_bench_command(job(tmp_path / "q", mugge_bench_tickets=20), "x", tmp_path)
    assert quick[quick.index("--tickets") + 1] == "20"
    # and a forced answer after a call still thinking at its cap
    assert ev.limits(j.config)[1] >= ev.call_max_tokens(j.config) + ev.ANSWER_TOKENS + ev.BENCH_PROMPT_TOKENS
    assert ev.mugge_bench_command(j, "b", tmp_path, "http://127.0.0.1:9/v1")[ev.mugge_bench_command(j, "b", tmp_path).index("--url") + 1] == "http://127.0.0.1:9/v1"
    off = ev.mugge_bench_command(job(tmp_path / "b", code_eval_temperature=0.2), "x", tmp_path)
    assert "--top-k" not in off and off[off.index("--thinking") + 1] == "off" and off[off.index("--temperature") + 1] == "0.2"


def test_report_comes_back_and_a_failed_run_is_none(tmp_path, monkeypatch):
    j = job(tmp_path)

    def ok(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "report.json").write_text(json.dumps({"score": 61.5, "tickets": 300, "seeds": 3, "solvedPerGpuHour": 420.0}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ev.subprocess, "run", ok)
    assert ev.mugge_bench(j, "build")["score"] == 61.5
    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    assert ev.mugge_bench(j, "other") is None


THINKING = {"model": "build", "messages": [{"role": "user", "content": "Ticket: t"}], "max_tokens": 100,
            "temperature": 1.0, "seed": 3, "chat_template_kwargs": {"enable_thinking": True}}


def fake_chat(message, finish):
    return lambda messages, max_tokens, temperature, sampling, thinking: {
        "message": message, "finish_reason": finish, "usage": {"prompt_tokens": 40, "completion_tokens": max_tokens}}


def test_a_call_still_thinking_at_its_cap_gets_a_forced_answer(monkeypatch):
    seen = []
    monkeypatch.setattr(ev, "chat", fake_chat({"content": "", "reasoning_content": "so the loop must"}, "length"))
    monkeypatch.setattr(ev, "force_answer", lambda m, r, why, t, s: seen.append((r, why, t)) or ("a.py\n```\nx = 1\n```", 7))
    calls = {"calls": 0, "forced": 0}
    got = ev.bench_answer(THINKING, calls)
    assert got["choices"][0]["message"]["content"] == "a.py\n```\nx = 1\n```" and seen == [("so the loop must", "length", 1.0)]
    assert got["usage"] == {"prompt_tokens": 40, "completion_tokens": 107} and calls == {"calls": 1, "forced": 1}
    assert got["choices"][0]["forced"] == "length"  # the engine's per-call log keeps it out of training rows
    # an answer that came is passed on as it is, without its thinking
    monkeypatch.setattr(ev, "chat", fake_chat({"content": "<think>hm</think>b.py\n```\ny\n```", "reasoning_content": ""}, "stop"))
    got = ev.bench_answer(THINKING, calls)
    assert got["choices"][0]["message"]["content"] == "b.py\n```\ny\n```" and len(seen) == 1 and calls["forced"] == 1
    assert "forced" not in got["choices"][0]
    # thinking off: nothing is forced, even an empty answer
    monkeypatch.setattr(ev, "chat", fake_chat({"content": ""}, "length"))
    assert ev.bench_answer({**THINKING, "chat_template_kwargs": {"enable_thinking": False}}, calls)["choices"][0]["message"]["content"] == ""
    assert len(seen) == 1 and calls == {"calls": 3, "forced": 1}


def post(url, body):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_bench_endpoint_answers_over_http_and_reports_a_dead_server(monkeypatch):
    monkeypatch.setattr(ev, "chat", fake_chat({"content": "", "reasoning_content": "hmm"}, "length"))
    monkeypatch.setattr(ev, "force_answer", lambda *a: ("c.py\n```\nz\n```", 5))
    with ev.bench_proxy() as (url, calls):
        status, got = post(url + "/chat/completions", THINKING)
        assert status == 200 and got["choices"][0]["message"]["content"] == "c.py\n```\nz\n```"
        assert post(url + "/models", {})[0] == 404

        def dead(*a, **k):
            raise ev.ServerDead("refused")

        monkeypatch.setattr(ev, "chat", dead)
        status, got = post(url + "/chat/completions", THINKING)
        assert status == 502 and "refused" in got["error"]
    assert calls == {"calls": 1, "forced": 1, "errors": 2, "dead": "refused"}


def test_a_server_that_dies_during_the_bench_gives_no_score(tmp_path, monkeypatch):
    j = job(tmp_path)

    def dead(*a, **k):
        raise ev.ServerDead("gone")

    def bench(cmd, **kw):  # one call, then a report as if every ticket failed
        assert kw["env"]["MUGGE_TIMEOUT_S"] == "3600"
        post(cmd[cmd.index("--url") + 1] + "/chat/completions", THINKING)
        (Path(cmd[cmd.index("--out") + 1]) / "report.json").write_text(json.dumps({"score": 0, "tickets": 1, "seeds": 1}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ev, "chat", dead)
    monkeypatch.setattr(ev.subprocess, "run", bench)
    with pytest.raises(ev.ServerDead, match="during MuggeBench"):
        ev.mugge_bench(j, "build")
    assert "failed calls 1" in (tmp_path / "job/work/mugge-bench-build.log").read_text()


@pytest.mark.skipif(not shutil.which("bun"), reason="needs bun")
def test_bench_cli_on_the_fixture_with_the_fake_model(tmp_path):
    out = tmp_path / "run"
    r = subprocess.run(["bun", "bench/run.ts", "--bench", str(FIXTURE), "--out", str(out), "--fake", "--seeds", "1"],
                       cwd=ROOT / "mugge", capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["score"] == 100 and report["tickets"] == 2
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert {row["key"] for row in rows} == {"calc/add", "calc/mul"} and all(row["solved"] for row in rows)
    cmp = subprocess.run(["bun", "bench/report.ts", str(out), "--ref", str(out)], cwd=ROOT / "mugge",
                         capture_output=True, text=True, timeout=120)
    assert cmp.returncode == 0 and "of reference: MET" in cmp.stdout


def test_rows_holding_a_bench_canary_never_reach_training():
    from stages import data

    stats = {}
    rows = [{"messages": [{"role": "user", "content": "ok"}]},
            {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "MUGGE-BENCH-1234 leaked"}]}]
    assert data.drop_bench_rows(rows, stats) == rows[:1] and stats == {"bench_rows_dropped": 1}
