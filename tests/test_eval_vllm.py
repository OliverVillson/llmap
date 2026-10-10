"""The eval stage's vLLM backend, on fakes: no GPU, vllm or transformers needed."""

import json
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

from stages import eval as ev
from stages import sandbox
from stages._util import Config, Job

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def llama_by_default(monkeypatch):
    """serve() rebinds ev.VLLM; this puts the default back after each test."""
    monkeypatch.setattr(ev, "VLLM", {})


def checkpoint(path: Path, **config) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    return path


class R:
    def __init__(self, body): self.body = body
    def raise_for_status(self): pass
    def json(self): return self.body


class FakeProc:
    def __init__(self, cmd, stdout=None, stderr=None, env=None):
        self.cmd, self.env, self.stopped = cmd, env, False
        FakeProc.started.append(self)

    def poll(self): return None
    def terminate(self): self.stopped = True
    def wait(self, timeout=None): return 0
    def kill(self): self.stopped = True


def fake_servers(monkeypatch) -> list:
    FakeProc.started = []
    monkeypatch.setattr(ev.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: type("H", (), {"status_code": 200})())
    return FakeProc.started


def test_vllm_args(tmp_path, monkeypatch):
    monkeypatch.delenv("LOBBOT_EVAL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LOBBOT_EVAL_CTX", raising=False)
    flat = str(checkpoint(tmp_path / "flat", num_experts_per_tok=8))
    cfg = Config(code_eval_suites=["livecodebench"], code_eval_max_tokens=32768, code_eval_thinking=True)
    assert ev.vllm_args(cfg, flat, "cand") == [
        "--host", "127.0.0.1", "--port", str(ev.PORT), "--served-model-name", "cand",
        "--max-model-len", str(32768 + ev.ANSWER_TOKENS + 4096), "--max-num-seqs", str(ev.VLLM_SEQS),
        "--gpu-memory-utilization", str(ev.VLLM_GPU_UTIL), "--enable-prefix-caching", "--reasoning-parser", "qwen3"]
    cfg.eval_reasoning_parser, cfg.eval_experts_used = "", 6
    cfg.eval_vllm_args = ["--speculative-config", '{"method": "mtp"}']
    args = ev.vllm_args(cfg, flat, "cand")
    assert "--reasoning-parser" not in args and args[-2:] == cfg.eval_vllm_args
    assert json.loads(args[args.index("--hf-overrides") + 1]) == {"num_experts_per_tok": 6}
    # Qwen3.5/3.6 checkpoints keep the text model's settings under text_config.
    nested = str(checkpoint(tmp_path / "nested", text_config={"num_experts_per_tok": 8}))
    args = ev.vllm_args(cfg, nested, "cand")
    assert json.loads(args[args.index("--hf-overrides") + 1]) == {"text_config": {"num_experts_per_tok": 6}}
    # Gemma 4 calls it top_k_experts: the shipped multimodal config and a pruned text-only one
    cfg.eval_reasoning_parser = "gemma4"
    shipped = checkpoint(tmp_path / "gemma", text_config={"top_k_experts": 8})
    pruned = checkpoint(tmp_path / "gemma-text", top_k_experts=8)
    for path, want in ((shipped, {"text_config": {"top_k_experts": 6}}), (pruned, {"top_k_experts": 6})):
        args = ev.vllm_args(cfg, str(path), "cand")
        assert json.loads(args[args.index("--hf-overrides") + 1]) == want
        assert args[args.index("--reasoning-parser") + 1] == "gemma4"


def test_vllm_command_runs_from_the_vllm_venv(tmp_path, monkeypatch):
    """The eval runs in the training venv: vllm comes from LOBBOT_VLLM_PY's venv, with
    its bin first on PATH and no PYTHONPATH; PATH's vllm only without it."""
    ckpt = str(checkpoint(tmp_path / "ckpt"))
    bin_dir = tmp_path / "venv-vllm" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").touch()
    (bin_dir / "vllm").touch()
    monkeypatch.setenv("LOBBOT_VLLM_PY", str(bin_dir / "python"))
    monkeypatch.setenv("PYTHONPATH", "/venv-train/site-packages")
    cmd, env = ev.vllm_command(ckpt)
    assert cmd == [str(bin_dir / "vllm"), "serve", ckpt]
    assert env["PATH"].split(":")[0] == str(bin_dir) and "PYTHONPATH" not in env
    (bin_dir / "vllm").unlink()
    cmd, _ = ev.vllm_command(ckpt)
    assert cmd == [str(bin_dir / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", ckpt]
    monkeypatch.delenv("LOBBOT_VLLM_PY")
    monkeypatch.setattr(ev.shutil, "which", lambda name: f"/opt/bin/{name}")
    assert ev.vllm_command(ckpt)[0] == ["/opt/bin/vllm", "serve", ckpt]


def test_serve_sends_checkpoint_dirs_to_vllm_and_ggufs_to_llama_server(tmp_path, monkeypatch):
    job = Job(tmp_path / "job")
    gguf = tmp_path / "q4.gguf"
    gguf.write_bytes(b"GGUF")
    ckpt = str(checkpoint(tmp_path / "nvfp4"))
    monkeypatch.delenv("LOBBOT_VLLM_PY", raising=False)
    started = fake_servers(monkeypatch)

    ev.serve(job, ckpt, "nvfp4")
    assert started[-1].cmd[1:3] == ["serve", ckpt] and "nvfp4" in started[-1].cmd and started[-1].env is not None
    assert ev.VLLM == {"name": "nvfp4", "dir": ckpt} and ev.workers() == ev.VLLM_SEQS
    assert (job.root / "work/vllm-nvfp4.log").exists()

    ev.serve(job, str(gguf), "q4")
    assert started[-1].cmd[0].endswith("llama-server") and started[-1].cmd[1:3] == ["-m", str(gguf)]
    assert started[-1].env is None and ev.VLLM == {} and ev.workers() == ev.SLOTS
    assert (job.root / "work/llama-server-q4.log").exists()


def test_split_reasoning_reads_vllm_fields_and_cut_thinking():
    from stages.eval import split_reasoning

    assert split_reasoning({"content": "x = 1", "reasoning": "hmm"}) == ("hmm", "x = 1")  # newer vLLM
    assert split_reasoning({"content": "x = 1", "reasoning_content": "hmm", "reasoning": None}) == ("hmm", "x = 1")
    # The qwen3 parser leaves a thinking the prompt opened, and the cap cut, in content.
    assert split_reasoning({"content": "plan it"}, cut=True) == ("plan it", "")
    assert split_reasoning({"content": "plan it"}) == ("", "plan it")
    assert split_reasoning({"content": "</think>\n\nx = 1"}, cut=True) == ("", "x = 1")


def test_generate_on_vllm_names_the_model_and_keeps_seqs_in_flight(monkeypatch):
    """Both answers must be in flight at once (VLLM_SEQS of them, not SLOTS); the
    reasoning comes in either field and the speed from tokens over wall time."""
    monkeypatch.setattr(ev, "VLLM", {"name": "cand", "dir": "/ckpt"})
    monkeypatch.setattr(ev, "SLOTS", 1)
    monkeypatch.setattr(ev, "VLLM_SEQS", 2)
    both = threading.Barrier(2, timeout=10)
    sent = []

    def post(url, timeout, json):
        sent.append((url, json))
        both.wait()
        field = "reasoning" if json["messages"][1]["content"] == "a" else "reasoning_content"
        return R({"choices": [{"finish_reason": "stop", "message": {"content": "x = 1", field: "hmm"}}],
                  "usage": {"completion_tokens": 50}})

    monkeypatch.setattr(httpx, "post", post)
    infos = []
    answers, tps = ev.generate("sys", ["a", "b"], 4096, 0.6, 0, thinking=True, infos=infos)
    assert answers == ["x = 1", "x = 1"] and [i["reasoning_chars"] for i in infos] == [3, 3]
    assert all(url.endswith("/v1/chat/completions") and j["model"] == "cand" for url, j in sent)
    assert tps and tps > 0


def test_force_answer_on_vllm_uses_the_checkpoint_tokenizer(tmp_path, monkeypatch):
    """A thinking without an answer is closed on top of the template the served dir's
    own tokenizer renders (loaded once) and answered through /v1/completions."""
    loads = []

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            assert not tokenize and add_generation_prompt and enable_thinking
            return f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"

    auto = types.SimpleNamespace(from_pretrained=lambda d: loads.append(d) or Tok())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=auto))
    monkeypatch.setattr(ev, "_tokenizers", {})
    monkeypatch.setattr(ev, "VLLM", {"name": "cand", "dir": str(tmp_path)})
    chat = {"a": ("length", {"content": "plan it"}),  # the cap cut it; the parser left it in content
            "b": ("stop", {"content": None, "reasoning": "plan it"})}
    sent = []

    def post(url, timeout, json):
        sent.append((url, json))
        if url.endswith("/v1/chat/completions"):
            finish, message = chat[json["messages"][1]["content"]]
            return R({"choices": [{"finish_reason": finish, "message": message}], "usage": {"completion_tokens": 900}})
        return R({"choices": [{"text": "```py\nx = 1\n```"}], "usage": {"completion_tokens": 12}})

    monkeypatch.setattr(httpx, "post", post)
    infos = []
    answers, _ = ev.generate("sys", ["a", "b"], 24576, 0.6, 0, thinking=True, infos=infos)
    assert answers == ["```py\nx = 1\n```"] * 2 and loads == [str(tmp_path)]
    assert [(i["forced"], i["tokens"], i["forced_tokens"]) for i in infos] == [("length", 900, 12), ("stop", 900, 12)]
    assert {url.rsplit("/", 1)[1] for url, _ in sent} == {"completions"}  # nothing from llama-server's API
    forced = {j["prompt"]: j for url, j in sent if url.endswith("/v1/completions")}
    opened = "<|im_end|>\n<|im_start|>assistant\n<think>\nplan it"
    assert sorted(forced) == [f"<|im_start|>user\na{opened}{ev.EARLY_STOP}\n</think>\n\n",
                              f"<|im_start|>user\nb{opened}\n</think>\n\n"]
    assert all(j["model"] == "cand" and j["max_tokens"] == ev.ANSWER_TOKENS and j["temperature"] == 0.6
               and j["seed"] == 0 and j["top_k"] == 20 and not j["add_special_tokens"] for j in forced.values())


def test_force_answer_reopens_gemma4_thought_channel(tmp_path, monkeypatch):
    """Gemma 4's template leaves the thought channel to the model, so a forced answer
    opens it in the prompt, puts back the thinking vLLM's gemma4 parser split off,
    and closes it with <channel|>."""
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return (f"<bos><|turn>system\n<|think|>\n{messages[0]['content']}<turn|>\n"
                    f"<|turn>user\n{messages[1]['content']}<turn|>\n<|turn>model\n")

    monkeypatch.setattr(ev, "_tokenizers", {str(tmp_path): Tok()})
    monkeypatch.setattr(ev, "VLLM", {"name": "cand", "dir": str(tmp_path)})
    sent = []

    def post(url, timeout, json):
        sent.append(json)
        return R({"choices": [{"text": "x = 1"}], "usage": {"completion_tokens": 3}})

    monkeypatch.setattr(httpx, "post", post)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "a"}]
    assert ev.force_answer(msgs, "plan it", "length", 1.0, {}) == ("x = 1", 3)
    assert sent[0]["prompt"] == ("<bos><|turn>system\n<|think|>\nsys<turn|>\n<|turn>user\na<turn|>\n<|turn>model\n"
                                 f"<|channel>thought\nplan it{ev.EARLY_STOP}\n<channel|>")


def fake_tests(monkeypatch):
    """Answers starting with "good" pass, without a compiler."""
    monkeypatch.setattr(ev.codebench, "assemble", lambda p, a, code=None: (a, ""))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: [
        sandbox.Result(code.startswith("good"), "" if code.startswith("good") else "test_failed")
        for _, code, *_ in items])


PROBS = [{"suite": suite, "id": f"p{j}", "language": "python", "prompt": f"q{j}", "stub": "", "entry": "",
          "tests": [{"input": "", "output": ""}]} for j, suite in enumerate(["livecodebench", "multipl-e-py"] * 2)]


def test_sampled_passes_run_at_once_and_give_the_rows_of_one_after_another(tmp_path, monkeypatch):
    """Three sampled passes are in generate() together (two suite groups each), and
    pass 0 comes in last; rows, their order and the scores match a run that answers
    the passes one after another."""
    job = Job(tmp_path)
    job.config.code_eval_samples, job.config.code_eval_k, job.config.code_eval_thinking = 3, [1, 3], True
    fake_tests(monkeypatch)

    def run(together: bool) -> tuple[dict, list[dict]]:
        meet = threading.Barrier(3, timeout=10)

        def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False,
                          on_answer=None, infos=None):
            if together:
                meet.wait()  # every pass is in at once, or this breaks
                time.sleep(0.05 * (3 - seed))
            ok = [(seed + int(p[1])) % 2 == 0 for p in prompts]
            infos.extend({"finish": "stop" if k else "length", "tokens": 100 * (seed + 1),
                          "forced": None if k else "length", **({} if k else {"forced_tokens": 10})} for k in ok)
            for _ in prompts:
                on_answer()
            return [f"good {seed}" if k else "bad" for k in ok], 40.0

        monkeypatch.setattr(ev, "generate", fake_generate)
        out = ev.code_eval(job, "cand", PROBS, "task system")
        rows = [json.loads(l) for l in (tmp_path / "work/code_eval/cand.jsonl").read_text().splitlines()]
        return out, rows

    out, rows = run(together=True)

    class OneAfterAnother:
        def __init__(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def map(self, f, xs): return map(f, xs)

    monkeypatch.setattr(ev, "ThreadPoolExecutor", OneAfterAnother)
    seq_out, seq_rows = run(together=False)
    assert rows == seq_rows and out["suites"] == seq_out["suites"]
    assert [(r["sample"], r["id"]) for r in rows] == [(i, p["id"]) for i in range(3) for p in PROBS]
    assert [r["answer"] for r in rows[:4]] == ["good 0", "bad", "good 0", "bad"]
    assert out["tokens_mean"] == 200 and out["hit_cap_share"] == 0.5 and out["forced_share"] == 0.5
    assert out["tok_s_total"] > 0 and out["answer_s"] > 0 and out["tok_s_vm"] == 40.0


def test_sampled_lcb_answers_are_checked_on_the_examples(tmp_path, monkeypatch):
    """Each sampled answer that fails is run on the examples alone; pick@examples hands
    in the first sample that passes them, and rows record it."""
    job = Job(tmp_path)
    job.config.code_eval_samples, job.config.code_eval_thinking = 3, True
    cases = [{"input": "1", "output": "1"}, {"input": "2", "output": "2"}]
    probs = [{"suite": "livecodebench", "id": i, "language": "python", "prompt": f"q{i}", "stub": "", "entry": "",
              "tests": cases, "public": 1} for i in "ab"]
    plan = {"qa": ["bad", "examples", "good"], "qb": ["bad", "good", "bad"]}  # "examples" passes only the first case
    monkeypatch.setattr(ev.codebench, "assemble", lambda p, a, code=None: (a, json.dumps(p["tests"])))
    monkeypatch.setattr(sandbox, "run_many", lambda items, timeout=0: [
        sandbox.Result(ok, "" if ok else "test_failed")
        for ok in (code == "good" or code == "examples" and len(json.loads(tests)) == 1 for _, code, tests, *_ in items)])

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False,
                      on_answer=None, infos=None):
        infos.extend({} for _ in prompts)
        for _ in prompts:
            on_answer()
        return [next(v[seed] for k, v in plan.items() if p.startswith(k)) for p in prompts], None

    monkeypatch.setattr(ev, "generate", fake_generate)
    out = ev.code_eval(job, "cand", probs, "task system")
    lcb = out["suites"]["livecodebench"]
    assert lcb["pass@1"] == round(1 / 3, 4) and lcb["pick@examples"] == 0.5  # a hands in "examples", b "good"
    rows = [json.loads(l) for l in (tmp_path / "work/code_eval/cand.jsonl").read_text().splitlines()]
    assert [(r["id"], r["examples_passed"]) for r in rows] == [
        ("a", False), ("b", False), ("a", True), ("b", True), ("a", True), ("b", False)]


def test_answer_stats():
    rows = [{"tokens": 100, "finish": "stop", "forced": None},
            {"tokens": 300, "finish": "length", "forced": "length", "forced_tokens": 50},
            {"tokens": 200, "finish": "stop", "forced": None},
            {"finish": "error", "forced": None, "error": "timeout"}]
    assert ev.answer_stats(rows, 10.0) == {"tokens_mean": 200, "tokens_median": 200, "hit_cap_share": 0.25,
                                           "forced_share": 0.25, "tok_s_total": 65.0, "answer_s": 10.0}
    assert ev.answer_stats([], 0.0)["tok_s_total"] is None


def test_watch_gpu_records_mean_busy_and_peak_memory(monkeypatch):
    outputs = ["40, 1000\n60, 3000\n", "[N/A], [N/A]\n"] + ["40, 500\n60, 500\n"] * 1000
    calls = []

    def run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=outputs[len(calls) - 1])

    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(ev.subprocess, "run", run)
    end = ev.watch_gpu(every=0.01)
    deadline = time.monotonic() + 5
    while len(calls) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert end() == {"busy_pct": 50.0, "mem_peak_mib": 4000}
    assert calls[0] == ["/usr/bin/nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]

    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    calls.clear()
    end = ev.watch_gpu(every=0.01)
    time.sleep(0.05)
    assert end() == {} and calls == []


def test_eval_stage_serves_a_mixed_candidate_list(tmp_path, monkeypatch):
    """A GGUF and a checkpoint dir in one job: each gets its own server, generate()
    talks to the right one, and eval.json has the dir's safetensors size, the code
    stats and the GPU record for both."""
    root = tmp_path / "job"
    root.mkdir()
    (root / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "livecodebench.jsonl").write_text("".join(
        json.dumps({**p, "suite": "livecodebench", "date": "2026-06-01"}) + "\n" for p in PROBS[:2]))
    gguf = tmp_path / "q4.gguf"
    gguf.write_bytes(b"GGUF")
    ckpt = checkpoint(tmp_path / "nvfp4", num_experts_per_tok=8)
    with open(ckpt / "model.safetensors", "wb") as f:
        f.truncate(30_000_000)
    (root / "config.json").write_text(json.dumps({
        "eval_candidates": {"q4": str(gguf), "nvfp4": str(ckpt)}, "code_eval_suites": ["livecodebench"],
        "code_eval_dir": str(bench)}))
    monkeypatch.delenv("LOBBOT_VLLM_PY", raising=False)
    started = fake_servers(monkeypatch)
    fake_tests(monkeypatch)
    served = []

    def fake_generate(system, prompts, max_tokens, temperature=0.0, seed=None, thinking=False,
                      on_answer=None, infos=None):
        served.append(dict(ev.VLLM))
        infos.extend({"finish": "stop", "tokens": 10, "forced": None} for _ in prompts)
        return ["good"] * len(prompts), None

    monkeypatch.setattr(ev, "generate", fake_generate)
    ev.run_stage(Job(root))
    assert served == [{}, {"name": "nvfp4", "dir": str(ckpt)}]
    assert [p.cmd[1:3] for p in started] == [["-m", str(gguf)], ["serve", str(ckpt)]] and all(p.stopped for p in started)
    rep = json.loads((root / "out/eval.json").read_text())
    q4, nvfp4 = rep["candidates"]
    assert nvfp4["size_gb"] == 0.03 and rep["score_method"] == "pass@1"
    for c in (q4, nvfp4):
        assert c["code"]["mean_pass@1"] == 1.0 and c["code"]["tokens_median"] == 10 and c["gpu"] == {}
