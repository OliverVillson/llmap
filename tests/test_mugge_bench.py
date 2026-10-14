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
    assert ev.limits(j.config)[1] >= ev.call_max_tokens(j.config) + ev.BENCH_PROMPT_TOKENS
    off = ev.mugge_bench_command(job(tmp_path / "b", code_eval_temperature=0.2), "x", tmp_path)
    assert "--top-k" not in off and off[off.index("--thinking") + 1] == "off" and off[off.index("--temperature") + 1] == "0.2"


def test_report_comes_back_and_a_failed_run_is_none(tmp_path, monkeypatch):
    j = job(tmp_path)

    def ok(cmd, stdout=None, stderr=None, cwd=None):
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "report.json").write_text(json.dumps({"score": 61.5, "tickets": 300, "seeds": 3, "solvedPerGpuHour": 420.0}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ev.subprocess, "run", ok)
    assert ev.mugge_bench(j, "build")["score"] == 61.5
    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    assert ev.mugge_bench(j, "other") is None


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
