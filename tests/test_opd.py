import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
httpx = pytest.importorskip("httpx")

from stages import codebench, opd  # noqa: E402
from stages.eval import EARLY_STOP  # noqa: E402

# A Granite-style chat template: role markers as special tokens, a default system
# message when none is given (what render() must override with the eval's), and the
# thinking opened in the prompt when enable_thinking is on.
TEMPLATE = (
    "{%- if messages[0]['role'] != 'system' %}<|start_of_role|>system<|end_of_role|>"
    "You are a helpful assistant.<|end_of_text|>\n{% endif %}"
    "{%- for m in messages %}<|start_of_role|>{{ m['role'] }}<|end_of_role|>{{ m['content'] }}<|end_of_text|>\n"
    "{% endfor %}"
    "{%- if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>"
    "{%- if enable_thinking is defined and enable_thinking %}<think>\n{% endif %}{% endif %}")
CORPUS = [codebench.SYSTEM, EARLY_STOP, "Write a complete Python program that reads the input from stdin and writes "
          "the answer to stdout. Reply with the program in one ```python code block.",
          "Read an integer n and print twice its value. The first line holds n.",
          "```python\nprint(int(input()) * 2)\n```", "Let me think about the input format and the edge cases first.",
          "def solve(a, b):\n    return a + b\n", "Use this starter code: class Solution: def add(self, a, b)"]


def chat_tokenizer(d):
    """A real byte-level BPE tokenizer (trained on a few lines) with Granite-style role
    tokens, <think> and </think> as added tokens and TEMPLATE, saved to d."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(CORPUS * 20, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end_of_text|>", pad_token="<|end_of_text|>")
    tok.add_tokens(["<think>", "</think>"])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(d)
    return transformers.AutoTokenizer.from_pretrained(d)


STDIN = {"id": "cc/1", "language": "python", "prompt": "Read an integer n and print twice its value.", "stub": "",
         "entry": "", "tests": [{"input": "3\n", "output": "6\n"}, {"input": "5\n", "output": "10\n"}]}
FUNCTIONAL = {"id": "apps/1", "language": "python", "suite": "apps", "stub": "", "entry": "",
              "prompt": "Add two numbers.\n\nUse this starter code:\n```python\nclass Solution:\n"
                        "    def add(self, a: int, b: int) -> int:\n```",
              "tests": {"func": "add", "cases": [{"input": [1, 2], "output": 3}]}}


class FakeVLLM:
    """vLLM's OpenAI server on localhost, as far as gen and opd use it: /health,
    /tokenize (chat messages or a prompt) and /detokenize with a real tokenizer, and
    /v1/completions answered by complete(body) -> [(token ids, finish reason), ...].
    fail(path, body) -> an HTTP status injects an error; every request is recorded."""

    def __init__(self, tok, complete=None, fail=lambda path, body: None):
        self.tok, self.complete, self.fail, self.requests = tok, complete, fail, []
        self.healthy, self.lock = True, threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send(200 if fake.healthy else 503, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append((self.path, body))
                with fake.lock:
                    code = fake.fail(self.path, body)
                if code:
                    return self.send(code, {"error": "injected"})
                with fake.lock:
                    if self.path == "/tokenize" and "messages" in body:
                        text = fake.tok.apply_chat_template(body["messages"], tokenize=False,
                                                            add_generation_prompt=body["add_generation_prompt"],
                                                            **body.get("chat_template_kwargs", {}))
                        return self.send(200, {"tokens": fake.tok(text, add_special_tokens=False)["input_ids"]})
                    if self.path == "/tokenize":
                        return self.send(200, {"tokens": fake.tok(body["prompt"], add_special_tokens=False)["input_ids"]})
                    if self.path == "/detokenize":
                        return self.send(200, {"prompt": fake.tok.decode(body["tokens"])})
                choices = fake.complete(body)
                self.send(200, {"choices": [{"index": i, "token_ids": t, "finish_reason": f}
                                            for i, (t, f) in enumerate(choices)]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def bodies(self, path):
        return [b for p, b in self.requests if p == path]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(opd, "BACKOFF", 0.0)
    monkeypatch.setattr(opd, "HEALTH_WAIT", 0.0)


def test_render_is_the_eval_prompt_as_token_ids(tmp_path):
    tok = chat_tokenizer(tmp_path / "tok")
    for p in (STDIN, FUNCTIONAL):
        ids = opd.render(tok, p)
        msgs = [{"role": "system", "content": codebench.SYSTEM}, {"role": "user", "content": codebench.build_prompt(p)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        assert ids == tok(text, add_special_tokens=False)["input_ids"]
        assert len(ids) > 20 and all(type(t) is int for t in ids)
        back = tok.decode(ids)
        assert back.startswith(f"<|start_of_role|>system<|end_of_role|>{codebench.SYSTEM}")
        assert codebench.build_prompt(p) in back and back.endswith("<think>\n") and "helpful assistant" not in back
    # what the old render did: under transformers 5 this is the BatchEncoding's keys
    naive = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=True))
    if not all(type(t) is int for t in naive):
        assert naive == ["input_ids", "attention_mask"]


def test_render_refuses_anything_but_token_ids(tmp_path):
    tok = chat_tokenizer(tmp_path / "tok")
    tok.chat_template = "{{ '' }}"
    with pytest.raises(ValueError, match="not token ids"):
        opd.render(tok, STDIN)


def test_render_matches_the_servers_template_and_sample_sends_the_eval_sampling(tmp_path):
    tok = chat_tokenizer(tmp_path / "tok")
    fake = FakeVLLM(tok, complete=lambda body: [([5, 6, 7], "stop")] * body["n"])
    try:
        ids = opd.render(tok, STDIN)
        assert opd.render_remote(fake.url, opd.STUDENT, STDIN) == ids
        assert opd.sample(fake.url, opd.STUDENT, ids, 2, 100) == [{"tokens": [5, 6, 7], "finish": "stop"}] * 2
        body = fake.bodies("/v1/completions")[-1]
        assert {k: body[k] for k in opd.SAMPLING} == {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
                                                      "repetition_penalty": 1.0}
        assert body["prompt"] == ids and body["n"] == 2 and body["max_tokens"] == 100
        with pytest.raises(ValueError, match="token ids"):  # the dict keys never reach the server
            opd.sample(fake.url, opd.STUDENT, ["input_ids", "attention_mask"], 2, 100)
        fake.complete = lambda body: [([5], "stop")] * 2 * body["n"]  # a batch of text prompts
        with pytest.raises(ValueError, match="asked for 2 answers, got 4"):
            opd.sample(fake.url, opd.STUDENT, ids, 2, 100)
    finally:
        fake.stop()


def test_post_retries_server_errors_but_not_bad_requests(tmp_path, fast_retries):
    tok = chat_tokenizer(tmp_path / "tok")
    errors = [500, 503]
    fake = FakeVLLM(tok, complete=lambda body: [([9], "stop")],
                    fail=lambda path, body: errors.pop(0) if errors else None)
    try:
        assert opd.sample(fake.url, "m", list(range(30)), 1, 5) == [{"tokens": [9], "finish": "stop"}]
        assert len(fake.bodies("/v1/completions")) == 3
        fake.fail = lambda path, body: 400
        with pytest.raises(httpx.HTTPStatusError):
            opd.sample(fake.url, "m", list(range(30)), 1, 5)
        assert len(fake.bodies("/v1/completions")) == 4  # no retry
        fake.fail = lambda path, body: 500
        with pytest.raises(RuntimeError, match="failed 4 times"):
            opd.sample(fake.url, "m", list(range(30)), 1, 5)
    finally:
        fake.stop()


def test_a_dead_server_raises_instead_of_retrying(tmp_path, fast_retries):
    tok = chat_tokenizer(tmp_path / "tok")
    fake = FakeVLLM(tok, complete=lambda body: [([9], "stop")])
    url = fake.url
    fake.stop()
    with pytest.raises(opd.ServerDead):  # nothing listens any more
        opd.sample(url, "m", list(range(30)), 1, 5)
    fake = FakeVLLM(tok, complete=lambda body: [([9], "stop")], fail=lambda path, body: 500)
    gone = subprocess.Popen(["true"])
    gone.wait()
    try:
        opd.watch(fake.url, gone.pid)  # /health still answers, but the process is gone
        with pytest.raises(opd.ServerDead):
            opd.sample(fake.url, "m", list(range(30)), 1, 5)
        assert len(fake.bodies("/v1/completions")) == 1
    finally:
        opd._pids.pop(fake.url, None)
        fake.stop()


def test_continue_forced_adds_what_eval_force_answer_adds(tmp_path, monkeypatch):
    from stages import eval as ev
    assert opd.ANSWER_TOKENS == ev.ANSWER_TOKENS
    monkeypatch.setattr(opd, "_forced_ids", {})
    tok = chat_tokenizer(tmp_path / "tok")
    answer = tok.encode("```python\nprint(int(input()) * 2)\n```", add_special_tokens=False) + [tok.eos_token_id]
    fake = FakeVLLM(tok, complete=lambda body: [(answer, "stop")])
    try:
        prompt = opd.render(tok, STDIN)
        think = tok.encode("Let me think about the input format", add_special_tokens=False)
        added = opd.continue_forced(fake.url, opd.STUDENT, prompt, think)
        close = tok.encode(EARLY_STOP + "\n</think>\n\n", add_special_tokens=False)
        assert added == close + answer
        body = fake.bodies("/v1/completions")[-1]
        assert body["prompt"] == prompt + think + close and body["max_tokens"] == opd.ANSWER_TOKENS
        assert {k: body[k] for k in opd.SAMPLING} == opd.SAMPLING
        assert tok.decode(think + added).startswith("Let me think about the input format" + EARLY_STOP + "\n</think>")
        # a thinking that stopped by itself is closed without the early-stop line
        stopped = opd.continue_forced(fake.url, opd.STUDENT, prompt, think, why="stop")
        assert stopped == tok.encode("\n</think>\n\n", add_special_tokens=False) + answer
    finally:
        fake.stop()


def test_forced_reason_and_open_thinking_follow_the_eval(tmp_path):
    tok = chat_tokenizer(tmp_path / "tok")
    eos = tok.eos_token_id

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    done = enc("Let me think</think>\n\n```python\nprint(1)\n```") + [eos]
    assert opd.forced_reason(tok, done, "stop") is None
    assert opd.forced_reason(tok, enc("Let me think</think>\n\n```python\nprint("), "length") is None  # cut: counted
    cut = enc("Let me think about the input")
    assert opd.forced_reason(tok, cut, "length") == "length" and opd.open_thinking(tok, cut) == cut
    ended = enc("Let me think about the input") + [eos]
    assert opd.forced_reason(tok, ended, "stop") == "stop" and opd.open_thinking(tok, ended) == ended[:-1]
    empty = enc("Let me think</think>\n\n") + [eos]  # closed, but no answer: the eval forces one
    assert opd.forced_reason(tok, empty, "stop") == "stop"
    assert tok.decode(opd.open_thinking(tok, empty)) == "Let me think"
    twice = enc("a</think> b</think>")
    assert tok.decode(opd.open_thinking(tok, twice)) == "a</think> b"
    assert opd.answer_text(tok, done) == "\n\n```python\nprint(1)\n```" and opd.answer_text(tok, cut) is None


def test_forced_contexts_are_token_for_token_what_eval_force_answer_sends(tmp_path, monkeypatch):
    """eval.force_answer sends the template, the thinking as split_reasoning strips it,
    the early-stop line (when the budget ran out) and the closing tag, as one text.
    The training text's forced context must be those very ids: a thinking that closed
    Granite-style ("\\n</think>\\n") with no answer, and one the cap cut on a newline,
    keep no whitespace before the close. The thinking's other ids stay as sampled."""
    from stages import eval as ev
    from stages.taskdata import think_tags
    monkeypatch.setattr(opd, "_forced_ids", {})
    tok = chat_tokenizer(tmp_path / "tok")
    enc = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
    answer = enc("```python\nprint(int(input()) * 2)\n```") + [tok.eos_token_id]
    fake = FakeVLLM(tok, complete=lambda body: [(answer, "stop")])
    template = tok.apply_chat_template(opd.messages(STDIN), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    try:
        prompt = opd.render(tok, STDIN)
        for sampled, finish in ((enc("Let me think about the input format.\n</think>\n") + [tok.eos_token_id], "stop"),
                                (enc("Let me think about the input format.\n\n"), "length")):
            why = opd.forced_reason(tok, sampled, finish)
            think = opd.open_thinking(tok, sampled)
            assert think == sampled[:len(think)] and tok.decode(think) == "Let me think about the input format."
            close = opd.continue_forced(fake.url, opd.STUDENT, prompt, think, why=why)[:-len(answer)]
            reasoning, _ = ev.split_reasoning({"content": tok.decode(sampled, skip_special_tokens=True)},
                                              cut=finish == "length")
            sent = template + reasoning + (ev.EARLY_STOP if why == "length" else "") + think_tags(template)[1]
            assert prompt + think + close == enc(sent), why
    finally:
        fake.stop()


def test_passes_grades_with_the_eval_harness(tmp_path):
    if not hasattr(codebench, "job"):
        pytest.skip("codebench.job (the shared grading interface) is not in this tree yet")
    tok = chat_tokenizer(tmp_path / "tok")

    def enc(text):
        return tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]

    right = enc("Let me think</think>\n\n```python\nprint(int(input()) * 2)\n```")
    wrong = enc("Let me think</think>\n\n```python\nprint(int(input()) * 3)\n```")
    unclosed = enc("```python\nprint(int(input()) * 2)\n```")
    assert opd.passes(tok, [(STDIN, right), (STDIN, wrong), (STDIN, unclosed)]) == [True, False, False]


def test_problems_mix_in_deduplicated_functional_problems(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    lcb = "Given an array of integers nums return the number of pairs whose sum is divisible by k exactly once"
    with open(bench / "codecontests.jsonl", "w") as f:
        for i in range(40):
            f.write(json.dumps({**STDIN, "id": f"cc/{i}", "prompt": f"Problem {i}: read n and print 2n."}) + "\n")
    with open(bench / "apps.jsonl", "w") as f:
        for i in range(30):
            f.write(json.dumps({**FUNCTIONAL, "id": f"apps/{i}", "prompt": f"Function problem {i}."}) + "\n")
        f.write(json.dumps({**FUNCTIONAL, "id": "apps/copy", "prompt": lcb}) + "\n")
    (bench / "livecodebench.jsonl").write_text(json.dumps({**FUNCTIONAL, "id": "lcb/1", "prompt": lcb}) + "\n")
    probs = opd.problems(str(bench), seed=3)
    ids = [p["id"] for p in probs]
    assert len(ids) == len(set(ids)) == 50 and "apps/copy" not in ids
    assert sum(i.startswith("apps/") for i in ids) == 10  # 20% of the pool
    assert ids == [p["id"] for p in opd.problems(str(bench), seed=3)] != [p["id"] for p in opd.problems(str(bench))]
    (bench / "apps.jsonl").unlink()
    assert len(opd.problems(str(bench))) == 40


def test_token_logprobs_match_the_plain_gather_and_its_gradient():
    torch.manual_seed(0)
    logits = torch.randn(37, 50, requires_grad=True)
    tgt = torch.randint(0, 50, (37,))
    ref = torch.log_softmax(logits, -1).gather(-1, tgt[:, None]).squeeze(-1)
    (g_ref,) = torch.autograd.grad(ref.sum(), logits)
    got = opd.token_logprobs(logits, tgt, chunk=8)
    (g,) = torch.autograd.grad(got.sum(), logits)
    assert torch.allclose(got, ref, atol=1e-6) and torch.allclose(g, g_ref, atol=1e-6)


def test_loss_pushes_toward_the_teacher_and_clips():
    s = torch.tensor([-2.0, -1.0, -3.0], requires_grad=True)
    t = torch.tensor([-1.0, -1.0, -30.0])
    loss, kl = opd.opd_loss(s, t, clip=5.0)
    (g,) = torch.autograd.grad(loss, s)
    # descent raises log p where the teacher likes the token more, keeps it where they agree,
    # and lowers it where the teacher finds it unlikely, by at most the clip
    assert g.tolist() == [-1.0, 0.0, 5.0]
    assert kl == pytest.approx((-2 + 1) + 0 + (-3 + 30))


def test_reads_vllm_token_ids_and_prompt_logprobs():
    assert opd.choice_tokens({"token_ids": [5, 6]}) == [5, 6]
    assert opd.choice_tokens({"logprobs": {"tokens": ["token_id:7", "token_id:8"]}}) == [7, 8]
    with pytest.raises(ValueError):
        opd.choice_tokens({"logprobs": {"tokens": ["hello"]}})
    entries = [None, {"3": {"logprob": -0.5}}, {"4": -0.25, "9": -2.0}, {"9": {"logprob": -1.0}, "5": {"logprob": -3.0}}]
    assert opd.prompt_token_logprobs(entries, [1, 3, 4, 5], 2) == [-0.25, -3.0]
    with pytest.raises(ValueError):
        opd.prompt_token_logprobs(entries, [1, 3, 4, 6], 3)


def tiny_student(d):
    """chat_tokenizer plus a tiny Granite model over its vocab, saved to d."""
    from transformers import GraniteConfig, GraniteForCausalLM

    tok = chat_tokenizer(d)
    torch.manual_seed(0)
    cfg = GraniteConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    GraniteForCausalLM(cfg).save_pretrained(d)
    return tok


def test_rounds_train_toward_the_teacher_and_reload_the_adapter(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    base = tmp_path / "student"
    tok = tiny_student(base)
    answer = tok.encode("Let me think</think> print", add_special_tokens=False)
    loaded = []
    monkeypatch.setattr(opd, "sample", lambda url, model, prompt, n, *a: [{"tokens": answer, "finish": "stop"}] * n)
    # the teacher is sure of every token the student wrote
    monkeypatch.setattr(opd, "teacher_logprobs", lambda url, model, prompt, ans: [0.0] * len(ans))
    monkeypatch.setattr(opd, "load_adapter", lambda url, name, path, old: loaded.append(
        (name, old, (path / "adapter_config.json").exists())))
    monkeypatch.setattr(opd, "passes", lambda tok, items: [True] * len(items))

    probs = [STDIN] * 4
    out = opd.train_opd(str(base), probs, tmp_path / "work" / "opd", "s", "t", rounds=4,
                        problems_per_round=2, samples=2, lr=2e-2, r=4, alpha=8)

    assert loaded == [("opd1", None, True), ("opd2", "opd1", True), ("opd3", "opd2", True), ("opd4", "opd3", True)]
    rows = [json.loads(l) for l in (tmp_path / "work" / "opd-log.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == [1, 2, 3, 4] and rows[0]["answers"] == 4 and rows[0]["pass"] == 1.0
    assert abs(rows[-1]["kl_per_token"]) < abs(rows[0]["kl_per_token"])  # the student moved toward the teacher
    assert (out / "config.json").exists() and not (tmp_path / "work" / "opd.adapters").exists()
    merged = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in n for n, _ in merged.named_parameters())
