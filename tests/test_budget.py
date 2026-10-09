import json
from pathlib import Path

from stages import budget, sandbox
from stages import eval as ev
from stages._util import Config, Job, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_classify_does_what_a_capped_run_would():
    c = budget.classify
    assert c({"tokens": 900}, 600, 1000) == "same"  # fit
    assert c({"tokens": 3000}, 1200, 1000) == "cut"  # thinking reached the cap
    assert c({"tokens": 1100}, 800, 1000) == "truncated"  # thinking ended, answer ran past
    assert c({"tokens": 900, "forced": "stop"}, 900, 1000) == "same"  # stopped inside thinking first
    assert c({"tokens": 5000, "forced": "length"}, 5000, 1000) == "cut"
    assert c({"forced": "no_reasoning", "tokens": 5000}, 0, 1000) == "same"


def test_rescore_cuts_thinking_and_retests(tmp_path, monkeypatch):
    """Words stand in for tokens. Four answers at a 10-token budget: one fit, one
    thinking cut (re-answered on its first 10 words), one answer truncated, one
    stopped inside thinking early."""
    job = Job(tmp_path)
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    job.config = Config(code_eval_thinking=True, code_eval_budgets=[10], code_eval_max_tokens=100)
    probs = [{"suite": "livecodebench", "id": f"p{i}", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]} for i in range(4)]
    rows = [
        {"suite": "livecodebench", "id": "p0", "tokens": 8, "reasoning": "a b c", "answer": "good", "passed": True},
        {"suite": "livecodebench", "id": "p1", "tokens": 40, "reasoning": " ".join(["w"] * 30), "answer": "good",
         "passed": True},
        {"suite": "livecodebench", "id": "p2", "tokens": 14, "reasoning": "a b c d e f", "answer": "good x y z w",
         "passed": True},
        {"suite": "livecodebench", "id": "p3", "tokens": 5, "reasoning": "a b", "answer": "", "forced": "stop",
         "passed": False, "reason": "test_failed"},
    ]
    seen = {}
    monkeypatch.setattr(budget, "tokenize", lambda text: text.split())
    monkeypatch.setattr(budget, "detokenize", lambda toks: " ".join(toks))

    def force(messages, reasoning, why, temperature, sampling):
        seen["cut"] = (reasoning, why, temperature)
        return "bad", 3

    monkeypatch.setattr(ev, "force_answer", force)
    monkeypatch.setattr(budget.codebench, "assemble", lambda p, a, code=None: (a, ""))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: [
        sandbox.Result(code.startswith("good"), "" if code.startswith("good") else "test_failed") for _, code, _ in items])
    out = budget.rescore(job, "m", rows, probs, [10])["10"]
    assert out["how"] == {"same": 2, "cut": 1, "truncated": 1} and out["errors"] == 0
    assert seen["cut"] == (" ".join(["w"] * 10), "length", 0.6)
    got = {r["id"]: r for r in read_jsonl(tmp_path / "work/code_eval/m-10.jsonl")}
    assert got["p2"]["answer"] == "good x" and got["p2"]["passed"]  # 10 - 6 thinking - 2 closing words
    assert not got["p1"]["passed"] and got["p0"]["passed"] and not got["p3"]["passed"]
    assert out["suites"]["livecodebench"]["pass@1"] == 0.5

    # A re-run keeps the answers redone before and asks the server for none of them.
    def no_server(*a, **k):
        raise AssertionError("asked again")

    monkeypatch.setattr(ev, "force_answer", no_server)
    assert budget.rescore(job, "m", rows, probs, [10])["10"]["suites"]["livecodebench"]["pass@1"] == 0.5


def test_rescore_retries_once_then_scores_a_lost_answer_as_failed(tmp_path, monkeypatch):
    job = Job(tmp_path)
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    job.config = Config(code_eval_thinking=True, code_eval_budgets=[10], code_eval_max_tokens=100)
    probs = [{"suite": "livecodebench", "id": f"p{i}", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]} for i in range(2)]
    rows = [{"suite": "livecodebench", "id": f"p{i}", "tokens": 40, "reasoning": " ".join(["w"] * 30), "answer": "good",
             "passed": True} for i in range(2)]
    monkeypatch.setattr(budget, "tokenize", lambda text: text.split())
    monkeypatch.setattr(budget, "detokenize", lambda toks: " ".join(toks))
    calls = []

    def force(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setattr(ev, "force_answer", force)
    monkeypatch.setattr(budget.codebench, "assemble", lambda p, a, code=None: (a, ""))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: [
        sandbox.Result(code.startswith("good"), "" if code.startswith("good") else "test_failed") for _, code, _ in items])
    out = budget.rescore(job, "m", rows, probs, [10])["10"]
    assert out["errors"] == 2 and len(calls) == 4  # two tries each
    assert out["suites"]["livecodebench"]["pass@1"] == 0.0
    assert not (tmp_path / "work/code_eval/m-10.redo.jsonl").exists()  # nothing kept, so a re-run tries again


def test_budget_stage_adds_scores_to_eval_json(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "DRY_RUN", True)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    (tmp_path / "config.json").write_text(json.dumps({"code_eval_budgets": [16384, 32768], "code_eval_max_tokens": 65536,
                                                      "eval_candidates": {"m": str(gguf)}}))
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    (tmp_path / "out").mkdir()
    (tmp_path / "out/eval.json").write_text(json.dumps({"candidates": [{"name": "m", "code": {"mean_pass@1": 0.8}}]}))
    budget.run_stage(Job(tmp_path))
    b = json.loads((tmp_path / "out/eval.json").read_text())["candidates"][0]["code"]["budgets"]
    assert set(b) == {"16384", "32768", "65536"} and b["65536"]["mean_pass@1"] == 0.8
    assert b["16384"]["mean_pass@1"] < b["32768"]["mean_pass@1"] < 0.8


def test_kv_cache_type_reaches_llama_server(tmp_path):
    args = ev.server_args(Config(eval_kv_type="q8_0"), str(tmp_path / "m.gguf"))
    assert args[-6:] == ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]
    assert "-ctk" not in ev.server_args(Config(), str(tmp_path / "m.gguf"))
