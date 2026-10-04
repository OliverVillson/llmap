import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from common.taskspec import TaskSpec
from stages.data import check_answer, extract_code, parse_array, parse_code_inputs

ROOT = Path(__file__).resolve().parents[1]


def run_data(job):
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    return subprocess.run([sys.executable, "pipeline.py", "--job", str(job), "--only", "data"],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.fixture
def job(tmp_path):
    j = tmp_path / "job"
    j.mkdir()
    shutil.copy(ROOT / "examples/support-tickets.taskspec.json", j / "taskspec.json")
    (j / "config.json").write_text(json.dumps({"n_generate": 200, "n_heldout": 20}))
    return j


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_data_outputs(job):
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train, held = read(job / "data/train.jsonl"), read(job / "data/heldout.jsonl")
    spec = json.loads((job / "taskspec.json").read_text())
    assert len(held) == 20
    assert 200 <= len(train) <= 200 + 2 * len(spec["seed_examples"])
    for r in train:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        json.loads(r["messages"][2]["content"])  # broken teacher answers were dropped
    for r in held:
        assert set(r) == {"input", "reference"}
        json.loads(r["reference"])
    # held-out inputs never appear in train
    assert not {r["input"] for r in held} & {r["messages"][1]["content"] for r in train}
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["dropped"] and stats["train"] == len(train)
    assert (job / "data/calib.txt").read_text().strip()


def test_data_resumes_from_cached_inputs(job):
    assert run_data(job).returncode == 0
    n_inputs = len(read(job / "work/data_inputs.jsonl"))
    (job / ".done/data").unlink()
    p = run_data(job)
    assert p.returncode == 0
    assert "round 1" not in p.stdout  # inputs came from the resume cache
    assert len(read(job / "work/data_inputs.jsonl")) == n_inputs


@pytest.mark.parametrize("text,finished,ok", [
    ('{"a": 1}', True, True),
    ('```json\n{"a": 1}\n```', True, True),
    ('<think>hmm</think>\n{"a": 1}', True, True),
    ('Sure! Here it is: {"a": 1}', True, True),
    ('{"a": 1', True, False),
    ('{"a": 1}', False, False),
    ("", True, False),
])
def test_check_answer(text, finished, ok):
    ans, why = check_answer(text, finished, want_json=True)
    assert (ans is not None) == ok, why
    if ok:
        json.loads(ans)


def test_parse_array():
    assert parse_array('Here:\n```json\n["first input here", "second input here"]\n```') == [
        "first input here", "second input here"]
    assert parse_array('[{"input": "an input as an object"}, "a plain string input"]') == [
        "an input as an object", "a plain string input"]
    assert parse_array("no list at all") == []
    assert parse_array("[broken") == []


def test_failing_stage_reports_cause_and_logs(job):
    (job / "taskspec.json").write_text((job / "taskspec.json").read_text())
    (job / "config.json").write_text(json.dumps({"n_generate": 200, "n_heldout": 5000}))
    p = run_data(job)
    assert p.returncode == 1
    from common.progress import parse
    err = [e for e in map(parse, p.stdout.splitlines()) if e and e["status"] == "error"][0]
    assert "RuntimeError" in err["msg"] and "usable" in err["msg"]
    assert "Traceback" in (job / "logs/data.log").read_text()


def test_gemini_written_heldout(job, monkeypatch):
    """With testgen on, the held-out inputs are Gemini's, answered by the teacher, and never trained on."""
    import random

    from stages import data, testgen
    from stages._util import Job

    j = Job(job)
    fake = [f"gemini test input number {i} with enough words" for i in range(25)]
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setattr(data, "DRY_RUN", False)
    monkeypatch.setattr(data, "Teacher", lambda path, max_len=None: data.DryRunTeacher(j.spec, random.Random(0)))
    monkeypatch.setattr(testgen, "held_out_inputs", lambda spec, cfg, n, seen, *a, **k: fake[:n])
    data.run_stage(j)
    held, train = read(job / "data/heldout.jsonl"), read(job / "data/train.jsonl")
    assert {r["input"] for r in held} <= set(fake) and len(held) >= 15
    assert not set(fake) & {r["messages"][1]["content"] for r in train}
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["heldout_source"] == "gemini-3.8-flash" and "heldout_dropped" in stats

    # a rerun reuses the cached Gemini inputs instead of asking again
    (job / ".done/data").unlink()
    monkeypatch.setattr(testgen, "held_out_inputs", lambda *a, **k: pytest.fail("Gemini called again"))
    data.run_stage(j)
    assert [r["input"] for r in read(job / "work/data_testgen.jsonl")] == fake[:20]
    assert {r["input"] for r in read(job / "data/heldout.jsonl")} <= set(fake[:20])


def test_answer_max_tokens_env(monkeypatch):
    import importlib

    from stages import data

    monkeypatch.setenv("LOBBOT_DATA_ANSWER_MAX_TOKENS", "4096")
    assert importlib.reload(data).ANSWER_MAX_TOKENS == 4096
    monkeypatch.delenv("LOBBOT_DATA_ANSWER_MAX_TOKENS")
    assert importlib.reload(data).ANSWER_MAX_TOKENS == 1536


# --------------------------------------------------------------------------- code tasks

CODE_SPECS = {"python": "python-utils", "c": "c-strings"}


def code_job(tmp_path, lang):
    if lang == "c" and not shutil.which("cc"):
        pytest.skip("cc not installed")
    j = tmp_path / f"job-{lang}"
    j.mkdir()
    shutil.copy(ROOT / f"examples/{CODE_SPECS[lang]}.code.taskspec.json", j / "taskspec.json")
    (j / "config.json").write_text(json.dumps({"n_generate": 120, "n_heldout": 10}))
    return j


def test_old_taskspecs_still_load():
    spec = TaskSpec.load(ROOT / "examples/support-tickets.taskspec.json")
    assert spec.task_type == "text" and spec.language is None and not spec.is_code
    assert TaskSpec.from_dict(json.loads(spec.to_json())) == spec


@pytest.mark.parametrize("name", CODE_SPECS.values())
def test_code_taskspec_validates(name):
    spec = TaskSpec.load(ROOT / f"examples/{name}.code.taskspec.json")
    assert spec.is_code and spec.language in {"python", "c"}
    assert all(e.tests for e in spec.seed_examples)


@pytest.mark.parametrize("change,msg", [
    (lambda d: d.pop("language"), "language"),
    (lambda d: d.update(language="rust"), "language"),
    (lambda d: d["seed_examples"][1].update(tests=" "), "tests"),
    (lambda d: d.update(task_type="essay"), "task_type"),
    (lambda d: d.update(task_type="text"), "language is only valid"),
])
def test_bad_code_taskspec(change, msg):
    d = json.loads((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    change(d)
    with pytest.raises(ValueError, match=msg):
        TaskSpec.from_dict(d)


@pytest.mark.parametrize("lang", CODE_SPECS)
def test_code_data_keeps_only_passing_answers(tmp_path, lang):
    from stages import sandbox

    job = code_job(tmp_path, lang)
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train, held = read(job / "data/train.jsonl"), read(job / "data/heldout.jsonl")
    assert len(held) == 10 and len(train) >= 120
    tests = {json.loads(r)["input"]: json.loads(r)["tests"]
             for r in (job / "work/data_inputs.jsonl").read_text().splitlines()}
    for e in TaskSpec.load(job / "taskspec.json").seed_examples:
        tests[e.input] = e.tests
    rows = [(r["messages"][1]["content"], r["messages"][2]["content"]) for r in train[:40]]
    assert all(r.passed for r in sandbox.run_many([(lang, a, tests[i]) for i, a in rows]))
    for r in held:
        assert set(r) == {"input", "reference", "tests", "language", "system"} and r["language"] == lang
        assert r["tests"] == tests[r["input"]]
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["task_type"] == "code" and 0 < stats["sandbox_pass_rate"] < 1
    assert stats["dropped"].get("compile_error") and stats["answered"] < stats["sandbox_runs"]
    assert "answers passed their tests" in p.stdout


def test_code_data_rejects_seeds_that_fail_their_tests(tmp_path):
    job = code_job(tmp_path, "python")
    spec = json.loads((job / "taskspec.json").read_text())
    spec["seed_examples"][2]["output"] = "def is_balanced(s):\n    return True"
    (job / "taskspec.json").write_text(json.dumps(spec))
    p = run_data(job)
    assert p.returncode != 0 and "fail their own tests; seed 2" in p.stdout + p.stderr


def test_extract_code():
    assert extract_code("Here you go:\n```python\ndef f():\n    return 1\n```\nDone.") == "def f():\n    return 1"
    assert extract_code("<think>x</think>int f(void) { return 1; }") == "int f(void) { return 1; }"
    ans, why = check_answer("```c\nint f(void);\n```", True, want_json=False, want_code=True)
    assert ans == "int f(void);" and why == ""


def test_parse_code_inputs():
    text = '```json\n[{"input": "Write f(x) returning x", "tests": "assert f(1) == 1"}, ' \
           '{"input": "no tests here at all", "tests": ""}, "just a string input"]\n```'
    assert parse_code_inputs(text) == [("Write f(x) returning x", "assert f(1) == 1")]
