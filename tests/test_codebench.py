import base64
import json
import os
import pickle
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from stages import codebench as cb
from stages import sandbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_codebench as fc  # noqa: E402

needs = lambda *tools: pytest.mark.skipif(any(not shutil.which(t) for t in tools), reason=f"needs {', '.join(tools)}")


def run(p, answer):
    code, tests = cb.assemble(p, answer)
    return sandbox.run_tests(p["language"], code, tests)


def test_pass_at_k():
    assert cb.pass_at_k(1, 1, 1) == 1.0 and cb.pass_at_k(1, 0, 1) == 0.0
    assert cb.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert cb.pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)  # C(7,5)/C(10,5)
    assert cb.pass_at_k(4, 2, 3) == 1.0  # fewer failures than k
    with pytest.raises(ValueError):
        cb.pass_at_k(1, 1, 5)


def test_extract_code():
    assert cb.extract_code("Here:\n```python\nx = 1\n```\nand\n```\ny = 22222\n```") == "y = 22222"
    assert cb.extract_code("<think>hm</think>```js\nf()\n```") == "f()"
    assert cb.extract_code("def f(): pass") == "def f(): pass"


@needs("cc")
def test_c_set_reference_solutions_pass_and_stubs_fail():
    rows = cb.load_suite("c-set", "/nonexistent")
    assert len(rows) >= 10 and all(r["language"] == "c" and r["entry"] for r in rows)
    ok = sandbox.run_many([("c", r["canonical"], r["tests"]) for r in rows])
    assert [r.reason for r in ok if not r.passed] == []
    # The canonical answer as a model reply, fenced, without the stub's types or includes.
    for r in rows:
        body = r["canonical"].split("\n", r["canonical"].count("#include"))[-1]
        assert run(r, f"```c\n{body}\n```").passed, r["id"]
    stub_only = sandbox.run_many([("c", r["stub"], r["tests"]) for r in rows])
    assert not any(r.passed for r in stub_only)


MPE_PY = {"suite": "multipl-e-py", "language": "python", "entry": "add",
          "prompt": "from typing import List\n\ndef add(xs: List[int]) -> int:\n    \"\"\"Sum.\"\"\"\n",
          "tests": "def check(candidate):\n    assert candidate([1, 2]) == 3\n\ncheck(add)\n"}
MPE_PY["stub"] = MPE_PY["prompt"]


@needs("python3")
def test_python_whole_function_body_only_and_main_block():
    assert run(MPE_PY, "```python\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n```").passed  # import comes from the stub
    assert run(MPE_PY, "    return sum(xs)\n").passed
    assert run(MPE_PY, "```python\ndef add(xs):\n    return sum(xs)\n\nif __name__ == '__main__':\n    input()\n```").passed
    assert not run(MPE_PY, "```python\ndef add(xs):\n    return 0\n```").passed


@needs("node")
def test_javascript_body_only_gets_closed():
    p = {"suite": "multipl-e-js", "language": "javascript", "entry": "add",
         "prompt": "function add(a, b){\n", "stub": "function add(a, b){\n",
         "tests": "const assert = require('node:assert');\nassert.deepEqual(add(1, 2), 3);\n"}
    assert run(p, "  return a + b;\n").passed
    assert run(p, "```javascript\nfunction add(a, b) { return a + b; }\n```").passed


@needs("g++")
def test_cpp_includes_and_main_are_handled():
    stub = "#include<assert.h>\n#include<bits/stdc++.h>\nlong add(long a, long b) {\n"
    p = {"suite": "multipl-e-cpp", "language": "cpp", "entry": "add", "prompt": stub, "stub": stub,
         "tests": "int main() {\n    auto candidate = add;\n    assert(candidate(1, 2) == 3);\n}\n"}
    code, _ = cb.assemble(p, "```cpp\nlong add(long a, long b) { return a + b; }\nint main() { return 0; }\n```")
    assert code.startswith("#include<assert.h>\n#include<bits/stdc++.h>\n") and "main" not in code
    code, _ = cb.assemble(p, "    return a + b;")
    assert code.rstrip().endswith("}") and code.count("{") == code.count("}")
    assert run(p, "```cpp\nlong add(long a, long b) { return a + b; }\nint main() { return 0; }\n```").passed
    assert run(p, "    return a + b;").passed
    assert not run(p, "    return a - b;").passed


@needs("python3")
def test_stdin_problems_run_through_a_harness():
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "Add two numbers.",
         "tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5\n", "output": "10"}]}
    assert "stdin" in cb.build_prompt(p)
    good = "```python\nimport sys\n\ndef main():\n    a, b = map(int, sys.stdin.buffer.read().split())\n    print(a + b)\n\nif __name__ == '__main__':\n    main()\n```"
    assert run(p, good).passed
    assert run(p, "```python\na, b = map(int, input().split())\nprint(a + b)\nexit(0)\n```").passed
    bad = run(p, "```python\nprint(3)\n```")
    assert not bad.passed and "case 1" in bad.output


@needs("python3")
def test_livecodebench_functional_tests():
    tests = fc.functional_tests("twoSum", [{"input": "[2, 7, 11]\n9", "output": "[0, 1]"}])
    p = {"suite": "livecodebench", "language": "python", "stub": "", "entry": "", "prompt": "x", "tests": tests}
    ans = ("```python\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
           "        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen:\n"
           "                return [seen[target - x], i]\n            seen[x] = i\n```")
    assert run(p, ans).passed  # List comes from the LiveCodeBench prelude
    assert not run(p, ans.replace("[seen[target - x], i]", "[i]")).passed


def test_decode_cases_refuses_pickled_objects():
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
    assert fc.decode_cases(blob) == cases
    assert fc.decode_cases(json.dumps(cases)) == cases
    evil = base64.b64encode(zlib.compress(pickle.dumps(os.getcwd))).decode()
    with pytest.raises(pickle.UnpicklingError):
        fc.decode_cases(evil)


def test_multipl_e_row_drops_the_closing_brace_tests_start_with():
    r = fc.multipl_e_row("cpp", {"name": "HumanEval_0_add", "prompt": "long add() {\n",
                                 "tests": "}\nint main() {\n    auto candidate = add;\n}\n"})
    assert r["entry"] == "add" and r["tests"].startswith("\nint main")


def test_lcb_row_shapes():
    base = {"question_content": "Q", "platform": "atcoder", "question_id": "abc1", "contest_date": "2026-06-01T00:00:00",
            "starter_code": "", "metadata": "{}", "private_test_cases": "[]"}
    r = fc.lcb_row({**base, "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "stdin"}])}, 50)
    assert r["tests"] == [{"input": "1", "output": "1"}] and r["date"] == "2026-06-01"
    r = fc.lcb_row({**base, "starter_code": "class Solution:\n    def f(self, x):", "metadata": '{"func_name": "f"}',
                    "public_test_cases": json.dumps([{"input": "1", "output": "1", "testtype": "functional"}])}, 50)
    assert isinstance(r["tests"], str) and "starter code" in r["prompt"]


def test_load_suite_filters_by_date_and_limit(tmp_path):
    rows = [{"id": str(i), "language": "py", "prompt": "", "tests": "", "date": d}
            for i, d in enumerate(["2025-12-01", "2026-05-01", "2026-07-01"])]
    (tmp_path / "livecodebench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = cb.load_suite("livecodebench", tmp_path, since="2026-05-01")
    assert [r["id"] for r in got] == ["1", "2"] and got[0]["language"] == "python"
    assert len(cb.load_suite("livecodebench", tmp_path, limit=1)) == 1
    with pytest.raises(FileNotFoundError):
        cb.load_suite("multipl-e-js", tmp_path)


def test_heldout_rows_from_code_taskspecs():
    held = [{"input": "write f", "reference": "x", "tests": "assert f()", "language": "javascript"},
            {"input": "text task", "reference": "y"}]
    probs = cb.heldout_problems(held)
    assert len(probs) == 1 and probs[0]["language"] == "javascript" and cb.build_prompt(probs[0]) == "write f"


def test_summarize_skips_suites_without_a_toolchain_and_shares_of_ref():
    R = sandbox.Result
    probs = [{"suite": "a", "language": "c"}, {"suite": "a", "language": "c"}, {"suite": "b", "language": "cpp"}]
    res = [[R(True, ""), R(False, "test_failed")], [R(False, "timeout"), R(False, "timeout")],
           [R(False, "unsupported_language"), R(False, "unsupported_language")]]
    s = cb.summarize(probs, res, [1, 2])
    assert s["a"]["pass@1"] == 0.25 and s["a"]["pass@2"] == 0.5 and s["a"]["status"]["timeout"] == 2
    assert "pass@1" not in s["b"] and s["b"]["skipped"] == "unsupported_language"
    assert cb.mean_pass1(s) == 0.25
    share = cb.share_of_ref({"a": {"pass@1": 0.4}, "c": {"pass@1": 0.9}, "d": {"pass@1": 0.5}},
                            {"a": {"pass@1": 0.5}, "c": {"pass@1": 0.9}})
    assert share == {"suites": {"a": 0.8, "c": 1.0}, "mean": 0.9, "min": 0.8}
    ref = {"winner": "lobbot-moe", "candidates": [{"name": "ref", "code": {"suites": {"a": {"pass@1": 1}}}}]}
    assert cb.ref_suites(ref) == {"a": {"pass@1": 1}}


@pytest.mark.parametrize("variant", ["ref", "fp8", "r25q4", "r50mix", "r50mix-gen"])
def test_code10x_configs_load_with_job(tmp_path, variant):
    from stages._util import Job

    (tmp_path / "config.json").write_text((ROOT / "configs/code10x" / f"{variant}.json").read_text())
    cfg = Job(tmp_path).config
    assert cfg.teacher == "Qwen/Qwen3.6-35B-A3B" and cfg.code_eval_suites
    assert all(s == "heldout" or cb.suite_path(s, cfg.code_eval_dir).name == f"{s}.jsonl" for s in cfg.code_eval_suites)
    if variant in ("ref", "fp8"):
        assert list(cfg.eval_candidates) == [variant]
    else:
        assert "heldout" in cfg.code_eval_suites and not cfg.dense_fallback
        assert (cfg.reap_calib == "general") == (variant == "r50mix-gen") and (cfg.reap_calib_path != "") == (variant == "r50mix-gen")
    assert (variant == "ref") == (cfg.code_eval_ref == "")


def test_code10x_dry_run_reports_share_of_ref(tmp_path):
    """setup writes all five jobs; eval in dry run gives pass@1 per suite, each
    variant as a share of ref, and both the job report and the cross-variant
    table print it."""
    from lobbot.report import render

    jobs = tmp_path / "jobs"
    env = {**os.environ, "LOBBOT_DRY_RUN": "1", "LOBBOT_JOBS": str(jobs)}
    sh = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True, text=True)
    p = sh("scripts/code10x.py", "setup", "--taskspec", "examples/python-utils.code.taskspec.json")
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((jobs / "code10x-r25q4/taskspec.json").read_text())["target"]["max_size_gb"] == 16.0
    for v in ("ref", "r50mix"):
        cfg = json.loads((jobs / f"code10x-{v}/config.json").read_text())
        cfg["code_eval_dir"] = str(tmp_path / "bench")  # dry run: suites need not exist
        cfg["code_eval_suites"] = ["c-set"]
        (jobs / f"code10x-{v}/config.json").write_text(json.dumps(cfg))
    p = sh("pipeline.py", "--job", str(jobs / "code10x-ref"), "--only", "eval")
    assert p.returncode == 0, p.stdout + p.stderr
    p = sh("pipeline.py", "--job", str(jobs / "code10x-r50mix"))
    assert p.returncode == 0, p.stdout + p.stderr

    rep = json.loads((jobs / "code10x-r50mix/out/eval.json").read_text())
    assert rep["score_method"] == "pass@1"
    moe = next(c for c in rep["candidates"] if c["name"] == "lobbot-moe")
    assert moe["code"]["suites"]["c-set"]["pass@1"] == pytest.approx(0.8)
    assert moe["code"]["share_of_ref"]["mean"] == pytest.approx(1.0)

    text = "\n".join(render("code10x-r50mix", {"eval": rep, "taskspec": {}, "config": {}}, color=False, width=100))
    assert "Code tests (pass@1, share of ref)" in text and "80% (100% of ref)" in text

    p = sh("scripts/code10x.py", "report")
    assert "r50mix" in p.stdout and "pass (≥90%" in p.stdout and "not run" in p.stdout, p.stdout + p.stderr


@needs("cc")
def test_code_eval_runs_answers_and_writes_per_answer_rows(tmp_path, monkeypatch):
    """eval.code_eval against a fake server: perfect answers on one problem,
    stubs on the other; sampled pass@k when code_eval_samples > 1."""
    from stages import eval as ev
    from stages._util import Job

    job = Job(tmp_path)
    job.config.code_eval_samples, job.config.code_eval_k = 2, [1, 2]
    probs = cb.load_suite("c-set", "/nonexistent", limit=2)
    calls = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False):
        calls.append((system, temperature, seed))
        good = "```c\n" + probs[0]["canonical"] + "\n```"
        return [good if seed == 0 else "```c\nint nope;\n```", "no code here"], 50.0

    monkeypatch.setattr(ev, "generate", fake_generate)
    out = ev.code_eval(job, "cand", probs, "task system")
    s = out["suites"]["c-set"]
    assert s["n"] == 2 and s["pass@1"] == 0.25 and s["pass@2"] == 0.5
    assert out["tok_s_vm"] == 50.0 and out["samples_per_problem"] == 2
    assert {c[0] for c in calls} == {cb.SYSTEM} and {c[2] for c in calls} == {0, 1}
    rows = [json.loads(l) for l in (tmp_path / "work/code_eval/cand.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and sum(r["passed"] for r in rows) == 1


def test_code_eval_thinking_samples_and_asks_for_thinking(tmp_path, monkeypatch):
    """Thinking loops under greedy decoding, so a single-sample thinking eval
    samples at 0.6 with a fixed seed and passes thinking through."""
    from stages import eval as ev
    from stages._util import Job

    job = Job(tmp_path)
    job.config.code_eval_thinking, job.config.code_eval_max_tokens = True, 24576
    probs = cb.load_suite("c-set", "/nonexistent", limit=1)
    calls = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False):
        calls.append((max_tokens, temperature, seed, thinking))
        return [""], None

    monkeypatch.setattr(ev, "generate", fake_generate)
    ev.code_eval(job, "cand", probs, "task system")
    assert calls == [(24576, 0.6, 0, True)]


def test_generate_sends_thinking_flag(monkeypatch):
    import httpx
    from stages import eval as ev

    sent = []

    class R:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "```py\nx = 1\n```"}}]}

    monkeypatch.setattr(httpx, "post", lambda url, timeout, json: sent.append((timeout, json)) or R())
    ev.generate("sys", ["a"], 24576, 0.6, 0, thinking=True)
    ev.generate("sys", ["b"], 2048)
    (t1, on), (t2, off) = sent
    assert on["chat_template_kwargs"] == {"enable_thinking": True} and on["top_k"] == 20 and t1 >= 24576 / 4
    assert off["chat_template_kwargs"] == {"enable_thinking": False} and "top_k" not in off and t2 == 600


def test_eval_context_fits_code_answer_cap(monkeypatch):
    from stages import eval as ev
    from stages._util import Config

    monkeypatch.delenv("LOBBOT_EVAL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LOBBOT_EVAL_CTX", raising=False)
    cfg = Config(code_eval_suites=["livecodebench"], code_eval_max_tokens=24576, data_answer_max_tokens=2048)
    assert ev.limits(cfg) == (2048, 24576 + 4096)
