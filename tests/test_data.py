import json
import os
import shutil
import subprocess
import sys
from collections import Counter
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
    monkeypatch.setattr(data, "Teacher", lambda path, max_len=None, top_k=20: data.DryRunTeacher(j.spec, random.Random(0)))
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


def test_thinking_data_keeps_the_thinking(tmp_path):
    """data_thinking: every train row carries the teacher's thinking, the human seeds
    (which have none) stay out, and answers whose thinking never closed are dropped."""
    job = code_job(tmp_path, "python")
    cfg = json.loads((job / "config.json").read_text())
    (job / "config.json").write_text(json.dumps({**cfg, "data_thinking": True}))
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train = read(job / "data/train.jsonl")
    assert train and all(r["messages"][2]["reasoning_content"] == "Let me work this out first." for r in train)
    assert "</think>" not in "".join(r["messages"][2]["content"] for r in train)
    seeds = {e.input for e in TaskSpec.load(job / "taskspec.json").seed_examples}
    assert not seeds & {r["messages"][1]["content"] for r in train}
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["dropped"].get("no_thinking_end") and stats["thinking_chars_median"] > 0
    assert "<think>\nLet me work this out first.\n</think>" in (job / "data/calib.txt").read_text()


class CharTok:
    """A character-level tokenizer with a Qwen3.6-style chat template: thinking on
    opens <think> in the generation prompt, thinking off closes an empty one."""
    chat_template = "qwen-ish"
    eos_token = "<|im_end|>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _target(ex):
    return "".join(chr(i) for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100)


def _prompt(ex):
    return "".join(chr(i) for i, l in zip(ex["input_ids"], ex["labels"]) if l == -100)


def test_tokenize_trains_thinking_turns_on_the_thinking():
    from stages.taskdata import tokenize_example

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add"},
            {"role": "assistant", "content": "x = 1", "reasoning_content": "plan it"}]
    ex = tokenize_example(CharTok(), msgs, 10_000)
    assert _prompt(ex).endswith("<|im_start|>assistant\n<think>\n")
    assert _target(ex) == "plan it\n</think>\n\nx = 1<|im_end|>"

    off = tokenize_example(CharTok(), [*msgs[:2], {"role": "assistant", "content": "x = 1"}], 10_000)
    assert _prompt(off).endswith("<think>\n\n</think>\n\n") and _target(off) == "x = 1<|im_end|>"


class GemmaCharTok(CharTok):
    """Gemma 4's chat format: thinking on puts <|think|> in the system turn and leaves
    the thought channel to the model; thinking off closes an empty one; a model turn's
    reasoning_content is rendered as its thought channel."""
    chat_template = "gemma-ish <|turn>"
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        empty = "" if enable_thinking else "<|channel>thought\n<channel|>"
        text = "<bos><|turn>system\n" + ("<|think|>\n" if enable_thinking else "") + msgs[0]["content"] + "<turn|>\n"
        for m in msgs[1:]:
            if m["role"] == "user":
                text += f"<|turn>user\n{m['content']}<turn|>\n"
            else:
                thought = m.get("reasoning_content")
                channel = f"<|channel>thought\n{thought}\n<channel|>" if thought else empty
                text += f"<|turn>model\n{channel}{m['content']}<turn|>\n"
        return text + ("<|turn>model\n" + empty if add_generation_prompt else "")


def test_tokenize_trains_gemma4_thinking_in_its_thought_channel():
    from stages.taskdata import tokenize_example

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "add"},
            {"role": "assistant", "content": "x = 1", "reasoning_content": "plan it"}]
    ex = tokenize_example(GemmaCharTok(), msgs, 10_000)
    assert _prompt(ex).endswith("<|think|>\nsys<turn|>\n<|turn>user\nadd<turn|>\n<|turn>model\n")
    assert _target(ex) == "<|channel>thought\nplan it\n<channel|>x = 1<turn|>"
    # the row is the conversation as Gemma's template renders it with the thinking
    assert _prompt(ex) + _target(ex) + "\n" == GemmaCharTok().apply_chat_template(msgs, enable_thinking=True)

    off = tokenize_example(GemmaCharTok(), [*msgs[:2], {"role": "assistant", "content": "x = 1"}], 10_000)
    assert _prompt(off).endswith("<|turn>model\n<|channel>thought\n<channel|>") and _target(off) == "x = 1<turn|>"


def test_split_think_reads_qwen_tags_and_gemma4_thought_channel():
    from stages.data import clean, split_think
    from stages.taskdata import GEMMA4_THINK, QWEN_THINK, think_tags

    assert split_think("<think>\nplan it\n</think>\n\nx = 1") == ("plan it", "\n\nx = 1")
    assert split_think("plan it\n</think>\n\nx = 1") == ("plan it", "\n\nx = 1")  # template opened it
    assert split_think("<|channel>thought\nplan it\n<channel|>x = 1") == ("plan it", "x = 1")
    assert split_think("<|channel>thought\nplan it, never closed") == ("", "<|channel>thought\nplan it, never closed")
    assert clean("<|channel>thought\nplan it<channel|>x = 1") == "x = 1"
    assert think_tags(GemmaCharTok.chat_template) == GEMMA4_THINK and think_tags(None) == QWEN_THINK
    # a prompt is told by its last turn, whatever its messages quote
    assert think_tags("<|im_start|>user\nwhat is <|turn>?<|im_end|>\n<|im_start|>assistant\n") == QWEN_THINK


# --------------------------------------------------------------------------- contest rows

ADD_STDIN = "a, b = map(int, input().split())\nprint(a + b)"
ADD_FUNC = "class Solution:\n    def add(self, a, b):\n        return a + b"


def lcb_problem(name, date, functional):
    """A tiny LiveCodeBench row in scripts/fetch_codebench.py's shape: add two integers,
    read from stdin or passed to Solution.add (the problems DryRunTeacher can solve)."""
    row = {"id": f"atcoder/{name.replace(' ', '_')}", "language": "python", "stub": "", "entry": "", "date": date,
           "prompt": f"{name}: read A and B and print A + B.",
           "tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "10 -3\n", "output": "7\n"}]}
    if functional:
        row.update(id=f"leetcode/{name.replace(' ', '-')}",
                   tests={"func": "add", "cases": [["1\n2", "3"], ["-4\n9", "5"]]},
                   prompt=f"{name}: return a + b.\n\nUse this starter code:\n```python\nclass Solution:\n"
                          "    def add(self, a: int, b: int) -> int:\n        \n```")
    return row


def write_lcb(d, old=12):
    """old problems released before 2024-10-01; recent ones on or after it (some on or
    after lcb_since 2025-01-01), and one undated, which the eval also scores."""
    d.mkdir(exist_ok=True)
    rows = [lcb_problem(f"Old problem {i}", f"2024-0{1 + i % 9}-15", i % 2) for i in range(old)]
    rows += [lcb_problem(f"Recent problem {i}", date, i % 2)
             for i, date in enumerate(["2024-10-01", "2024-12-31", "2025-01-01", "2025-06-01"])]
    rows.append(lcb_problem("Recent problem undated", None, 0))
    (d / "livecodebench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_contest_problems_date_guard(tmp_path):
    """Only problems released before data_contest_before; undated ones never; a cut-off after
    the eval's lcb_since is refused."""
    from stages import data
    from stages._util import Config

    cfg = Config(code_eval_dir=str(write_lcb(tmp_path)), lcb_since="2025-01-01")
    got = data.contest_problems(cfg)
    assert len(got) == 12 and all(p["prompt"].startswith("Old problem") for p in got)
    cfg.data_contest_before = "2025-01-01"  # the latest allowed: everything before the eval's problems
    assert sorted({p["date"] for p in data.contest_problems(cfg)})[-2:] == ["2024-10-01", "2024-12-31"]
    cfg.data_contest_before = "2025-01-02"
    with pytest.raises(ValueError, match="after lcb_since"):
        data.contest_problems(cfg)


def test_contest_cutoff_after_the_eval_stops_the_stage_early(tmp_path, monkeypatch):
    from stages import data
    from stages._util import Job

    job = code_job(tmp_path, "python")
    (job / "config.json").write_text(json.dumps({
        "n_generate": 120, "n_heldout": 10, "code_eval_dir": str(write_lcb(tmp_path / "bench")),
        "data_contest_rows": 5, "data_contest_before": "2025-03-01", "lcb_since": "2025-01-01"}))
    monkeypatch.setattr(data, "DRY_RUN", False)
    monkeypatch.setattr(data, "Teacher", lambda *a, **k: pytest.fail("teacher loaded before the date check"))
    with pytest.raises(ValueError, match="after lcb_since"):
        data.run_stage(Job(job))


class ScriptedTeacher:
    """Asked the same problem five times, answers: wrong code, right code cut off, right
    code whose thinking never closed, right code after long thinking, right code after short
    thinking. Plain answers wrap the code in prose; tickets answer with the file."""

    def __init__(self):
        self.asked: dict[str, int] = {}

    def chat(self, convs, temperature, max_tokens, thinking=False):
        from stages import harness
        from stages.data import Gen

        out = []
        for conv in convs:
            prompt = conv[-1]["content"]
            k = self.asked[prompt] = self.asked.get(prompt, -1) + 1
            right = ADD_FUNC if "def add(self" in prompt else ADD_STDIN
            code, think, finished = [(right.replace("+", "-"), "short", True), (right, "s", False), (right, None, True),
                                     (right, "long " * 40, True), (right, "short", True)][k]
            body = (harness.render_files({"prog.py": code}, "adds them") if prompt.startswith("Ticket: ")
                    else f"Sure:\n```python\n{code}\n```\nThis adds them.")
            out.append(Gen(body if think is None else f"{think}\n</think>\n\n{body}", finished))
        return out


def test_contest_rows_keep_the_shortest_passing_answer(tmp_path):
    import random

    from stages import codebench, data, harness
    from stages._util import Config

    cfg = Config(code_eval_dir=str(write_lcb(tmp_path)), lcb_since="2025-01-01", data_thinking=True,
                 data_contest_rows=8, data_contest_samples=5, data_contest_harness_share=0.5)
    stats: dict = {}
    rows = data.contest_rows(cfg, ScriptedTeacher(), random.Random(0), data.contest_problems(cfg), stats)
    assert len(rows) == 8  # 12 asked (1.5x), all solved, stopped at data_contest_rows
    tickets = [r for r in rows if r["messages"][0]["content"] == harness.SYSTEM]
    assert 0 < len(tickets) < 8
    for r in rows:
        sys_msg, user, ans = r["messages"]
        right = ADD_FUNC if "def add(self" in user["content"] else ADD_STDIN
        assert ans["reasoning_content"] == "short"  # the shortest passing answer, not the first
        if r in tickets:
            assert user["content"].startswith("Ticket: livecodebench-")
            assert ans["content"] == harness.render_files({"prog.py": right}, "adds them")
        else:
            assert sys_msg["content"] == codebench.SYSTEM and "Old problem" in user["content"]
            assert ans["content"] == f"```python\n{right}\n```"  # the code only, without the prose
    assert stats == {"contest_available": 12, "contest_problems": 12, "contest_answers": 60, "contest_passed_any": 12,
                     "contest_rows": 8, "contest_ticket_rows": len(tickets), "contest_thinking_chars_median": 5,
                     "contest_dropped": {"test_failed": 12, "truncated": 12, "no_thinking_end": 12}}


def test_dry_run_adds_contest_rows(tmp_path):
    """Contest rows land in train and calib.txt with the teacher's thinking and code that
    passes; recent problems appear nowhere and contest problems are never held out."""
    from stages import codebench, harness, sandbox

    job = code_job(tmp_path, "python")
    bench = write_lcb(tmp_path / "bench", old=20)
    (job / "config.json").write_text(json.dumps({
        "n_generate": 120, "n_heldout": 10, "data_thinking": True, "code_eval_dir": str(bench),
        "lcb_since": "2025-01-01", "data_contest_rows": 10, "data_contest_harness_share": 0.5}))
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train = read(job / "data/train.jsonl")
    contest = [r for r in train if "Old problem" in r["messages"][1]["content"]]
    stats = json.loads((job / "data/stats.json").read_text())
    assert len(contest) == 10 == stats["contest_rows"] and stats["train"] == len(train)
    assert stats["contest_problems"] == 15 and stats["contest_answers"] == 30 and stats["contest_dropped"]
    assert stats["contest_thinking_chars_median"] == len("Let me work this out first.")
    calib, held = (job / "data/calib.txt").read_text(), (job / "data/heldout.jsonl").read_text()
    assert "Recent problem" not in json.dumps(train) + held + calib and "Old problem" not in held
    assert "Old problem" in calib and "[data] contest: 10 rows" in p.stdout

    probs = codebench.load_suite("livecodebench", bench)
    items, forms = [], set()
    for r in contest:
        user, ans = r["messages"][1]["content"], r["messages"][2]
        assert ans["reasoning_content"] == "Let me work this out first." and "</think>" not in ans["content"]
        prob = next(q for q in probs if q["prompt"].rstrip() in user)
        if user.startswith("Ticket: "):
            files, note = harness.parse_files(ans["content"], ["prog.py"])
            assert list(files) == ["prog.py"] and ans["content"] == harness.render_files(files, note)
            code = files["prog.py"]
        else:
            assert ans["content"].startswith("```python\n") and ans["content"].endswith("\n```")
            code = codebench.extract_code(ans["content"])
        forms.add(user.startswith("Ticket: "))
        items.append(("python", *codebench.assemble(prob, ans["content"], code), codebench.time_limit(prob)))
    assert forms == {True, False}
    assert all(res.passed for res in sandbox.run_many(items))


def aider_rows(n, chat=(), langs=("python",)):
    """n extra rows as scripts/polyglot.py rows writes them: a chat, an exercise and the answer."""
    return [{"messages": [*chat, {"role": "user", "content": f"aider exercise {i}"},
                          {"role": "assistant", "content": f"edit {i}", "reasoning_content": f"plan {i}"}],
             "meta": {"source": "aider", "language": langs[i % len(langs)], "exercise": f"ex{i}"}}
            for i in range(n)]


def test_dry_run_adds_extra_rows(tmp_path):
    """data_extra_rows: ready chat rows (whole aider chats here) go into train and calib.txt as
    they are, marked as extra beside their own meta, never into heldout, and are counted; a
    relative path is the job's. A file with a row that does not end in an answer stops the stage
    before any teacher work."""
    from stages.taskdata import is_extra, load_examples

    job = code_job(tmp_path, "python")
    chat = [{"role": "system", "content": "Act as an expert software developer."},
            {"role": "user", "content": "Change get_factorial() to use math.factorial"},
            {"role": "assistant", "content": "mathweb/flask/app.py ..."}]
    extra = aider_rows(5, chat)
    del extra[0]["meta"]  # a rows file from before rows carried meta
    (job / "aider-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in extra))
    cfg = json.loads((job / "config.json").read_text())
    (job / "config.json").write_text(json.dumps({**cfg, "data_thinking": True, "data_extra_rows": "aider-rows.jsonl"}))
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train = read(job / "data/train.jsonl")
    marked = [{**r, "meta": {**r.get("meta", {}), "extra": True}} for r in extra]
    assert sorted(json.dumps(r) for r in train if len(r["messages"]) == 5) == sorted(json.dumps(r) for r in marked)
    assert [r for r in train if "meta" in r] == [r for r in train if len(r["messages"]) == 5]
    examples = load_examples(job / "data/train.jsonl")  # what heal and calibration read
    assert sum(map(is_extra, examples)) == 5 and all(set(ex) <= {"messages", "meta"} for ex in examples)
    assert "aider exercise" not in (job / "data/heldout.jsonl").read_text()
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["extra_rows"] == 5 and stats["extra_weight"] == 1 and stats["train"] == len(train)
    assert stats["thinking_chars_median"] > 0
    assert "aider exercise 3\n<think>\nplan 3\n</think>\n\nedit 3" in (job / "data/calib.txt").read_text()

    (job / "aider-rows.jsonl").write_text(json.dumps({"messages": chat[:2]}) + "\n")
    (job / ".done/data").unlink()
    p = run_data(job)
    assert p.returncode != 0 and "line 1: not a chat row ending in an assistant message" in p.stdout + p.stderr


def test_dry_run_weights_extra_rows(tmp_path):
    """data_extra_weight 3: every extra row is in train three times (the copies after the shuffled
    rows), in calib.txt once, never held out; calibration (calib_extra_share) picks each once."""
    from stages.taskdata import calib_examples, is_extra

    job = code_job(tmp_path, "python")
    extra = aider_rows(12, langs=("go", "python", "rust"))
    (job / "aider-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in extra))
    cfg = json.loads((job / "config.json").read_text())
    (job / "config.json").write_text(json.dumps({**cfg, "data_extra_rows": "aider-rows.jsonl", "data_extra_weight": 3}))
    p = run_data(job)
    assert p.returncode == 0, p.stdout + p.stderr
    train = read(job / "data/train.jsonl")
    n_once = len(train) - 2 * len(extra)
    copies = Counter(json.dumps(r) for r in train if "meta" in r)
    assert len(copies) == 12 and set(copies.values()) == {3}
    assert all("meta" in r for r in train[n_once:]) and sum("meta" in r for r in train[:n_once]) == 12
    assert "aider exercise" not in (job / "data/heldout.jsonl").read_text()
    calib = (job / "data/calib.txt").read_text()
    assert all(calib.count(f"aider exercise {i}\n") == 1 for i in range(12))
    stats = json.loads((job / "data/stats.json").read_text())
    assert stats["extra_rows"] == 12 and stats["extra_weight"] == 3 and stats["train"] == len(train)

    picked = calib_examples(job / "data/train.jsonl", 20, 0.45)
    ex = [e for e in picked if is_extra(e)]
    assert len(picked) == 20 and len(ex) == 9 and len({json.dumps(e["messages"]) for e in ex}) == 9
    assert Counter(e["meta"]["language"] for e in ex) == {"go": 3, "python": 3, "rust": 3}

    (job / "config.json").write_text(json.dumps({**cfg, "data_extra_rows": "aider-rows.jsonl", "data_extra_weight": 0}))
    (job / ".done/data").unlink()
    p = run_data(job)
    assert p.returncode != 0 and "data_extra_weight must be 1 or more" in p.stdout + p.stderr
