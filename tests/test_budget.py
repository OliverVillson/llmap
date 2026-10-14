import json
import sys
import types
from pathlib import Path

import pytest

from stages import budget, sandbox
from stages import eval as ev
from stages._util import Config, Job, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def fake_tests(monkeypatch, items_seen=None):
    """Answers starting with "good" pass; the items are what codebench.job() makes."""
    monkeypatch.setattr(budget.codebench, "job", lambda p, a, code=None: (p["language"], a, "", None, {"cases.json": "[]"}))

    def run_many(items, timeout=0):
        if items_seen is not None:
            items_seen.extend(items)
        return [sandbox.Result(code.startswith("good"), "" if code.startswith("good") else "test_failed")
                for _, code, *_ in items]

    monkeypatch.setattr(sandbox, "run_many", run_many)


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
    fake_tests(monkeypatch)
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


def test_rescore_retries_once_then_stops_rather_than_score_a_lost_answer(tmp_path, monkeypatch):
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
    fake_tests(monkeypatch)
    with pytest.raises(RuntimeError, match="2 answers failed twice"):
        budget.rescore(job, "m", rows, probs, [10])
    assert len(calls) == 4  # two tries each
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
    assert b["65536"]["all_samples_mean_pass@1"] == 0.8  # a dry run has no answers to score again
    assert b["16384"]["mean_pass@1"] < b["32768"]["mean_pass@1"] < 0.8


def test_the_long_budget_is_scored_on_the_answers_the_short_ones_are(tmp_path, monkeypatch):
    """The re-score covers sample 0 of each problem, so the long run's entry beside it
    is those answers' own results, and the gap between the two is the budget's effect
    alone. The eval's mean over every sample stays as all_samples_mean_pass@1. Here
    sample 0 passes both problems and sample 1 neither: 1.0 on the same answers, 0.5
    over all of them."""
    monkeypatch.setattr(budget, "DRY_RUN", False)
    (tmp_path / "config.json").write_text(json.dumps({"code_eval_budgets": [10], "code_eval_max_tokens": 100,
                                                      "eval_candidates": {"m": str(tmp_path / "ckpt")}}))
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    (tmp_path / "out").mkdir()
    (tmp_path / "out/eval.json").write_text(json.dumps({"candidates": [
        {"name": "m", "code": {"mean_pass@1": 0.5, "suites": {"livecodebench": {"n": 2, "pass@1": 0.5}}}}]}))
    probs = [{"suite": "livecodebench", "id": f"p{i}", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]} for i in range(2)]
    rows = [{"suite": "livecodebench", "id": f"p{i}", "sample": s, "passed": s == 0, "reason": "" if s == 0 else "test_failed",
             "forced": "length" if (i, s) == (1, 0) else None} for i in range(2) for s in range(2)]
    (tmp_path / "work/code_eval").mkdir(parents=True)
    (tmp_path / "work/code_eval/m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(ev, "code_problems", lambda job, held: probs)
    monkeypatch.setattr(ev, "candidates_to_eval", lambda job: {"m": {"path": str(tmp_path / "ckpt")}})
    monkeypatch.setattr(ev, "serve", lambda *a, **k: object())
    monkeypatch.setattr(ev, "stop", lambda proc: None)
    seen = []
    monkeypatch.setattr(budget, "rescore", lambda job, name, rows, problems, budgets, release=None:
                        seen.extend(rows) or {"10": {"mean_pass@1": 0.5}})
    budget.run_stage(Job(tmp_path))
    b = json.loads((tmp_path / "out/eval.json").read_text())["candidates"][0]["code"]["budgets"]
    assert [(r["id"], r["sample"]) for r in seen] == [("p0", 0), ("p1", 0)]
    assert b["100"]["mean_pass@1"] == 1.0 and b["100"]["all_samples_mean_pass@1"] == 0.5
    assert b["100"]["suites"]["livecodebench"]["n"] == 2 and b["100"]["how"] == {"hit the cap": 1}
    assert b["10"]["mean_pass@1"] == 0.5


def test_kv_cache_type_reaches_llama_server(tmp_path):
    args = ev.server_args(Config(eval_kv_type="q8_0"), str(tmp_path / "m.gguf"))
    assert args[-6:] == ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on"]
    assert "-ctk" not in ev.server_args(Config(), str(tmp_path / "m.gguf"))


def test_repair_lora_reaches_llama_server(tmp_path):
    args = ev.server_args(Config(eval_lora="/x/lora.gguf"), str(tmp_path / "m.gguf"))
    assert args[args.index("--lora") + 1] == "/x/lora.gguf"
    assert "--lora" not in ev.server_args(Config(), str(tmp_path / "m.gguf"))


def test_rescore_on_vllm_cuts_with_the_candidates_own_tokenizer(tmp_path, monkeypatch):
    """Served by vLLM, the thinking is cut in the checkpoint's own HF tokenizer's tokens
    (vLLM has no /tokenize for this here), the forced answer draws with sample 0's seed
    (code_eval_seed_offset), and the sandbox gets codebench.job()'s items with their files."""
    job = Job(tmp_path)
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    job.config = Config(code_eval_thinking=True, code_eval_budgets=[4], code_eval_max_tokens=100,
                        code_eval_seed_offset=1000)
    probs = [{"suite": "livecodebench", "id": "p0", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]}]
    rows = [{"suite": "livecodebench", "id": "p0", "tokens": 40, "reasoning": "a b c d e f g", "answer": "bad",
             "passed": False, "reason": "test_failed"}]
    loads = []

    class Tok:  # a word per token
        words = {}

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [self.words.setdefault(w, len(self.words)) for w in text.split()]

        def decode(self, ids):
            back = {v: k for k, v in self.words.items()}
            return " ".join(back[i] for i in ids)

    auto = types.SimpleNamespace(from_pretrained=lambda d: loads.append(d) or Tok())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=auto))
    monkeypatch.setattr(ev, "_tokenizers", {})
    monkeypatch.setattr(ev, "VLLM", {"name": "m-budgets", "dir": str(tmp_path / "ckpt")})
    monkeypatch.setattr(ev, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked the server")))
    seen = {}

    def force(messages, reasoning, why, temperature, sampling):
        seen.update(reasoning=reasoning, seed=sampling["seed"], min_p=sampling["min_p"],
                    penalty=sampling["repetition_penalty"])
        return "good", 2

    monkeypatch.setattr(ev, "force_answer", force)
    items = []
    fake_tests(monkeypatch, items)
    out = budget.rescore(job, "m", rows, probs, [4])["4"]
    assert loads == [str(tmp_path / "ckpt")] and seen == {"reasoning": "a b c d", "seed": 1000, "min_p": 0.0,
                                                          "penalty": 1.0}
    assert out["how"] == {"cut": 1} and out["suites"]["livecodebench"]["pass@1"] == 1.0
    assert items == [("python", "good", "", None, {"cases.json": "[]"})]


def test_rescore_stops_when_the_server_dies(tmp_path, monkeypatch):
    """A dead server is not an answer that failed: the step stops, and nothing is kept
    that a re-run would then skip."""
    job = Job(tmp_path)
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    job.config = Config(code_eval_thinking=True, code_eval_budgets=[10], code_eval_max_tokens=100)
    probs = [{"suite": "livecodebench", "id": "p0", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]}]
    rows = [{"suite": "livecodebench", "id": "p0", "tokens": 40, "reasoning": " ".join(["w"] * 30), "answer": "good"}]
    monkeypatch.setattr(budget, "tokenize", lambda text: text.split())
    monkeypatch.setattr(budget, "detokenize", lambda toks: " ".join(toks))
    calls = []

    def force(*args, **kwargs):
        calls.append(1)
        raise ev.ServerDead("refused")

    monkeypatch.setattr(ev, "force_answer", force)
    fake_tests(monkeypatch)
    with pytest.raises(ev.ServerDead):
        budget.rescore(job, "m", rows, probs, [10])
    assert len(calls) == 1 and not (tmp_path / "work/code_eval/m-10.redo.jsonl").exists()


def test_redone_answers_are_reused_only_for_the_answer_they_came_from_and_tested_after_release(tmp_path, monkeypatch):
    """The redo file is keyed by the long run's answer, so a new eval of the same problems
    redoes them; the sandbox runs once every budget is redone and the server released."""
    job = Job(tmp_path)
    (tmp_path / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    job.config = Config(code_eval_thinking=True, code_eval_budgets=[10, 20], code_eval_max_tokens=100)
    probs = [{"suite": "livecodebench", "id": "p0", "language": "python", "prompt": "q", "stub": "", "entry": "",
              "tests": [{"input": "", "output": ""}]}]
    rows = [{"suite": "livecodebench", "id": "p0", "tokens": 40, "reasoning": " ".join(["w"] * 30), "answer": "good"}]
    monkeypatch.setattr(budget, "tokenize", lambda text: text.split())
    monkeypatch.setattr(budget, "detokenize", lambda toks: " ".join(toks))
    order = []

    def force(messages, reasoning, why, temperature, sampling):
        order.append(("force", len(reasoning.split())))
        return "good", 1

    monkeypatch.setattr(ev, "force_answer", force)
    monkeypatch.setattr(budget.codebench, "job", lambda p, a, code=None: (p["language"], a, "", None, {}))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: order.append("sandbox") or
                        [sandbox.Result(True, "") for _ in items])
    out = budget.rescore(job, "m", rows, probs, [10, 20], release=lambda: order.append("release"))
    assert order == [("force", 10), ("force", 20), "release", "sandbox", "sandbox"]
    assert out["10"]["how"] == {"cut": 1} and out["20"]["how"] == {"cut": 1}
    order.clear()
    budget.rescore(job, "m", rows, probs, [10, 20])
    assert order == ["sandbox", "sandbox"]  # the same long run: nothing redone
    order.clear()
    budget.rescore(job, "m", [{**rows[0], "reasoning": " ".join(["v"] * 30)}], probs, [10, 20])
    assert order[:2] == [("force", 10), ("force", 20)]  # a new long run: redone
