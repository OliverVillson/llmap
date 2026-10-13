import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("httpx")
F = torch.nn.functional

sys.path.insert(0, str(Path(__file__).parent))
from test_opd import STDIN, FakeVLLM, chat_tokenizer  # noqa: E402

from stages import opd, sft, ternary  # noqa: E402
from stages import traces as T  # noqa: E402
from stages.eval import EARLY_STOP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def row(pid, prompt, tokens, **kw):
    return {"id": pid, "prompt": prompt, "tokens": tokens, "finish": "stop", "passed": None, "forced": None, **kw}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------- reading and the held-out split ----------

def test_load_keeps_ids_and_refuses_anything_but_token_ids(tmp_path):
    p = tmp_path / "t.jsonl"
    write(p, [row("cc/1", [1, 2, 3], [4, 5], passed=True, forced="length"), row(7, [1], [2])])
    rows = T.load(p)
    assert rows[0] == row("cc/1", [1, 2, 3], [4, 5], passed=True, forced="length") and rows[1]["id"] == "7"
    for bad, match in ((row("cc/1", ["input_ids", "attention_mask"], [5, 6]), "prompt is not a list of token ids"),
                       (row("cc/1", [1, 2], [3.0]), "tokens is not"), (row("cc/1", [1, 2], [True]), "tokens is not"),
                       ({"prompt": [1], "tokens": [2]}, "no problem id")):
        write(p, [row("cc/0", [1], [2]), bad])
        with pytest.raises(ValueError, match=match):
            T.load(p)


def test_split_is_by_problem_stable_across_processes_and_about_held_frac():
    rows = [row(f"cc/{i // 2}", [1], [2]) for i in range(10000)]
    train, held = T.split(rows)
    held_ids = sorted({r["id"] for r in held})
    assert not {r["id"] for r in train} & set(held_ids)  # both samples of a problem on one side
    assert all(sum(r["id"] == i for r in held) == 2 for i in held_ids[:5])
    assert 0.014 < len(held_ids) / 5000 < 0.026
    code = ("import json, sys; sys.path.insert(0, '.'); from stages import traces as T; "
            "rows = [{'id': f'cc/{i // 2}'} for i in range(10000)]; "
            "print(json.dumps(sorted({r['id'] for r in T.split(rows)[1]})))")
    for hash_seed in ("1", "2"):  # Python's own str hash differs per process; crc32 does not
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONHASHSEED": hash_seed}).stdout
        assert json.loads(out) == held_ids
    assert {r["id"] for r in T.split(rows, seed=1)[1]} != set(held_ids)
    few = [row(f"cc/{i}", [1], [2]) for i in range(3)]
    assert len(T.split(few)[1]) == 1  # too few problems for 2%: still one held out


# ---------- batches ----------

def rows_of(lengths, prompt=4):
    return [row(f"cc/{i}", list(range(100, 100 + prompt)), list(range(n))) for i, n in enumerate(lengths)]


def test_batches_start_each_row_at_its_prompt_with_masks_and_weights():
    rows = rows_of([3, 10, 30, 7, 1, 50, 12, 2, 9, 40, 25, 6]) + [row("cc/long", list(range(60)), [1])]
    got = list(T.batches(rows, 20, 3, prompt_weight=0.1, seed=0))
    seen = []
    for ids, mask, w in got:
        assert ids.shape == mask.shape and w.shape == (ids.shape[0], ids.shape[1] - 1)
        for b in range(ids.shape[0]):
            n = int(mask[b].sum())
            assert mask[b, :n].all() and not mask[b, n:].any() and not ids[b, n:].any()
            seq = ids[b, :n].tolist()
            assert seq[:4] == [100, 101, 102, 103]  # the prompt first: never mid-answer
            seen.append(seq)
            assert torch.allclose(w[b, :3], torch.tensor(0.1)) and (w[b, 3:n - 1] == 1).all() and not w[b, n - 1:].any()
    expect = [(r["prompt"] + r["tokens"])[:20] for r in rows[:-1]]  # the 60-token prompt has no answer in 20
    assert sorted(seen) == sorted(expect)
    assert max(ids.shape[1] for ids, _, _ in got) == 20  # cropped to the window


def test_batches_bucket_by_length_and_follow_the_seed():
    import random
    rng = random.Random(0)
    lengths = [rng.randint(5, 400) for _ in range(256)]
    rows = rows_of(lengths)

    def padding(groups):
        return sum(ids.numel() - int(mask.sum()) for ids, mask, _ in groups)

    ours = padding(T.batches(rows, 512, 8, seed=1))
    order = list(range(len(rows)))
    random.Random(1).shuffle(order)
    plain = sum(max(lengths[i] + 4 for i in order[k:k + 8]) * 8 - sum(lengths[i] + 4 for i in order[k:k + 8])
                for k in range(0, len(order), 8))
    assert ours < plain / 5
    a, b, c = (list(T.batches(rows, 512, 8, seed=s)) for s in (1, 1, 2))
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))
    assert [x[0].shape for x in a] != [x[0].shape for x in c]


def test_batches_map_ids_to_a_trimmed_vocab_and_drop_lost_targets():
    rows = [row("cc/1", [5, 6], [7, 9, 8, 7])]
    old_to_new = {"5": 0, "6": 1, "7": 2, "8": 3}  # 9 was dropped
    for vocab_map in (old_to_new, {"kept": [5, 6, 7, 8], "old_to_new": old_to_new},
                      torch.tensor([-1, -1, -1, -1, -1, 0, 1, 2, 3])):
        ((ids, mask, w),) = T.batches(rows, 16, 1, vocab_map=vocab_map, prompt_weight=0.5)
        assert ids.tolist() == [[0, 1, 2, 0, 3, 2]] and mask.all()
        assert w.tolist() == [[0.5, 1.0, 0.0, 1.0, 1.0]]  # the target 9 weighs nothing


# ---------- kd_kl over trace batches ----------

def per_position_kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def test_kd_kl_weights_reverse_and_a_trimmed_teacher():
    torch.manual_seed(0)
    s = torch.randn(23, 30, requires_grad=True)
    t = torch.randn(23, 50)
    w = torch.rand(23)
    w[[3, 7]] = 0
    index = torch.randperm(50)[:30]
    ls, lt_kept = F.log_softmax(s, -1), F.log_softmax(t[:, index], -1)
    cases = [({"weights": w}, (per_position_kl(F.log_softmax(t[:, :30], -1), ls) * w).sum() / w.sum(), t[:, :30]),
             ({"reverse": True}, per_position_kl(ls, F.log_softmax(t[:, :30], -1)).mean(), t[:, :30]),
             ({"index": index, "weights": w}, (per_position_kl(lt_kept, ls) * w).sum() / w.sum(), t),
             ({"index": index, "reverse": True}, per_position_kl(ls, lt_kept).mean(), t)]
    for kw, ref, teacher in cases:
        (g_ref,) = torch.autograd.grad(ref, s, retain_graph=True)
        got = sft.kd_kl(s, teacher, chunk=7, **kw)
        (g,) = torch.autograd.grad(got, s)
        assert torch.allclose(got, ref, atol=1e-5) and torch.allclose(g, g_ref, atol=1e-6), kw
    loss, dropped = sft.kd_kl(s, t, index=index, return_dropped=True, chunk=5)
    assert torch.allclose(dropped, 1 - F.softmax(t, -1)[:, index].sum(-1), atol=1e-6) and not dropped.requires_grad
    _, none = sft.kd_kl(s, t[:, :30], return_dropped=True)
    assert not none.any()
    with pytest.raises(ValueError, match="kept teacher ids"):
        sft.kd_kl(s, t, index=index[:10])


# ---------- validation and the progress sidecar ----------

def test_validate_and_count_sidecar(tmp_path):
    rows = [row(f"cc/{i // 2}", [1], list(range(600 + i)), finish="length" if i % 4 == 0 else "stop",
                passed=i % 2 == 0, forced="length" if i % 4 == 0 else None) for i in range(8)]
    rep = T.validate(rows, 2)
    assert rep["problems"] == 4 and rep["cut"] == rep["forced"] == 0.25 and rep["passed"] == 0.5
    with pytest.raises(ValueError, match="do not have 2 rows"):
        T.validate(rows[:-1], 2)
    with pytest.raises(ValueError, match="median answer"):
        T.validate(rows, 2, min_median=1000)
    out = tmp_path / "traces.jsonl"
    assert T.read_count(out) is None
    T.write_count(out, tokens=5, target=10)
    assert T.count_path(out).name == "traces.count" and T.read_count(out) == {"tokens": 5, "target": 10}
    T.count_path(out).write_text('{"tokens": 5')
    assert T.read_count(out) is None


# ---------- gen against a fake vLLM ----------

THINK = " ".join(["Let me think about the input format and the edge cases first."] * 6)


def bench(d, n, broken=()):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "codecontests.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({**STDIN, "id": f"cc/{i}", "prompt": f"Problem {i}: read an integer n and print twice "
                                f"its value.{' BROKEN' if i in broken else ''}"}) + "\n")
    return d


class Served:
    """A tiny model dir with chat_tokenizer, and a FakeVLLM that answers every problem
    with one finished answer (right) and one thinking cut by the budget, whose forced
    answer is wrong. Prompts with BROKEN in them get HTTP 500."""

    def __init__(self, d):
        self.dir = d / "model"
        self.tok = chat_tokenizer(self.dir)
        enc = lambda text: self.tok.encode(text, add_special_tokens=False)  # noqa: E731
        eos = self.tok.eos_token_id
        self.right = enc("```python\nprint(int(input()) * 2)\n```") + [eos]
        self.wrong = enc("```python\nprint(int(input()) * 3)\n```") + [eos]
        self.closed = enc(THINK + "</think>\n\n") + self.right
        self.cut = enc(THINK)

        def complete(body):
            if body["n"] == 1:  # the forced answer
                return [(self.wrong, "stop")]
            return [(self.closed, "stop"), (self.cut, "length")]

        def fail(path, body):
            if path == "/v1/completions" and "BROKEN" in self.tok.decode(body["prompt"]):
                return 500

        self.fake = FakeVLLM(self.tok, complete, fail)

    def gen(self, problems, out, tokens, **kw):
        return ternary.gen(str(self.dir), self.fake.url, str(problems), out, tokens, samples=2, max_tokens=300,
                           workers=3, max_model_len=300 + opd.ANSWER_TOKENS + ternary.FORCED_MARGIN + 1000,
                           min_median=50, **kw)

    def asked(self):
        """The problem ids sampled so far, one per sampling request."""
        by_prompt = {tuple(opd.render(self.tok, p)): p["id"] for p in opd.problems(str(self.problems_dir))}
        return [by_prompt[tuple(b["prompt"])] for b in self.fake.bodies("/v1/completions") if b["n"] == 2]


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(opd, "BACKOFF", 0.0)
    monkeypatch.setattr(opd, "HEALTH_WAIT", 0.0)
    monkeypatch.setattr(opd, "_forced_ids", {})  # one fake server per test, maybe on a port reused
    s = Served(tmp_path)
    yield s
    s.fake.stop()


def test_gen_writes_eval_shaped_labelled_traces_up_to_the_target(tmp_path, served):
    served.problems_dir = bench(tmp_path / "bench", 30)
    out = tmp_path / "w" / "traces.jsonl"
    n = served.gen(served.problems_dir, out, 1500)
    rows = T.load(out)
    probs = {p["id"]: p for p in opd.problems(str(served.problems_dir))}
    assert n == sum(len(r["tokens"]) for r in rows) >= 1500
    per_problem = max(sum(len(r["tokens"]) for r in rows if r["id"] == i) for i in {r["id"] for r in rows})
    assert n < 1500 + 4 * per_problem  # no new problem once written + in flight reach the target
    for done, cut in zip(rows[::2], rows[1::2]):
        assert done["id"] == cut["id"] and done["prompt"] == cut["prompt"] == opd.render(served.tok, probs[done["id"]])
        assert done["tokens"] == served.closed and (done["finish"], done["forced"], done["passed"]) == ("stop", None, True)
        close = served.tok.encode(EARLY_STOP + "\n</think>\n\n", add_special_tokens=False)
        assert cut["tokens"] == served.cut + close + served.wrong
        assert (cut["finish"], cut["forced"], cut["passed"]) == ("length", "length", False)
    sampled = [b for b in served.fake.bodies("/v1/completions") if b["n"] == 2]
    assert len(sampled) == len(rows) // 2 and all(b["max_tokens"] == 300 for b in sampled)
    assert all({k: b[k] for k in opd.SAMPLING} == opd.SAMPLING for b in served.fake.bodies("/v1/completions"))
    assert served.fake.bodies("/tokenize")[0]["messages"][0]["role"] == "system"  # the render check
    count = T.read_count(out)
    assert count["tokens"] == n and count["done"] and count["report"]["forced"] == 0.5 and count["report"]["passed"] == 0.5


def test_gen_resumes_by_problem_id_and_repairs_a_cut_file(tmp_path, served):
    served.problems_dir = bench(tmp_path / "bench", 30)
    out = tmp_path / "w" / "traces.jsonl"
    served.gen(served.problems_dir, out, 800)
    first = T.load(out)
    done_ids = {r["id"] for r in first}
    lone = {**first[0], "id": "cc/29"} if "cc/29" not in done_ids else {**first[0], "id": "cc/28"}
    with out.open("a") as f:  # one sample of a problem, then a line cut by a stop mid-write
        f.write(json.dumps(lone) + "\n" + '{"id": "cc/27", "prompt": [1, 2')
    asked_before = len(served.asked())
    served.gen(served.problems_dir, out, 2500)
    rows = T.load(out)
    assert rows[: len(first)] == first
    assert all(sum(r["id"] == i for r in rows) == 2 for i in {r["id"] for r in rows})
    assert not set(served.asked()[asked_before:]) & done_ids  # nothing asked twice
    assert T.read_count(out)["tokens"] == sum(len(r["tokens"]) for r in rows) >= 2500


def test_gen_skips_a_failing_problem_and_stops_when_too_many_fail(tmp_path, served):
    order = [p["id"] for p in opd.problems(str(bench(tmp_path / "probe", 30)))]
    first = int(order[0].split("/")[1])
    served.problems_dir = bench(tmp_path / "bench", 30, broken={first})
    out = tmp_path / "w" / "traces.jsonl"
    served.gen(served.problems_dir, out, 1500)
    assert f"cc/{first}" not in {r["id"] for r in T.load(out)} and T.read_count(out)["failed"] == 1
    assert served.asked().count(f"cc/{first}") == opd.RETRIES + 1
    served.problems_dir = bench(tmp_path / "bench2", 30, broken=set(range(30)))
    with pytest.raises(RuntimeError, match="problems failed"):
        served.gen(served.problems_dir, tmp_path / "w2" / "traces.jsonl", 1500)


def test_gen_stops_on_a_dead_server_and_when_the_pool_runs_out(tmp_path, served):
    served.problems_dir = bench(tmp_path / "bench", 4)
    out = tmp_path / "w" / "traces.jsonl"
    with pytest.raises(RuntimeError, match="pool ran out"):
        served.gen(served.problems_dir, out, 1e6)
    assert len(T.load(out)) == 8 and not T.read_count(out)["done"]

    def dead(path, body):
        if path == "/v1/completions":
            served.fake.healthy = False
            return 500

    served.fake.fail = dead
    with pytest.raises(opd.ServerDead):
        served.gen(bench(tmp_path / "bench2", 30), tmp_path / "w2" / "traces.jsonl", 1500)


def test_gen_refuses_prompts_the_server_would_render_differently(tmp_path, served):
    served.problems_dir = bench(tmp_path / "bench", 4)
    served.fake.tok = chat_tokenizer(tmp_path / "other")
    served.fake.tok.chat_template = served.fake.tok.chat_template.replace("<|end_of_text|>\n", "<|end_of_text|>")
    with pytest.raises(RuntimeError, match="disagree"):
        served.gen(served.problems_dir, tmp_path / "w" / "traces.jsonl", 1000)
    assert not served.fake.bodies("/v1/completions")


def test_gen_validates_before_it_returns(tmp_path, served):
    served.problems_dir = bench(tmp_path / "bench", 30)
    out = tmp_path / "w" / "traces.jsonl"
    with pytest.raises(ValueError, match="median answer"):
        ternary.gen(str(served.dir), served.fake.url, str(served.problems_dir), out, 600, samples=2, max_tokens=300,
                    workers=3, max_model_len=6000, min_median=10_000)
    assert not T.read_count(out)["done"]


def test_gen_command_bounds_the_server_and_stops_its_group(tmp_path):
    cmd = ternary.gen_command("/m", "/b", tmp_path / "w" / "traces.jsonl", 8e6, seqs=160)
    assert "--kv-cache-dtype fp8" in cmd and "--max-num-seqs 160" in cmd and "--max-model-len 22528" in cmd
    assert "--workers 80 " in cmd and "--server-pid $gen" in cmd and "--tokens 8000000" in cmd
    assert "setsid " in cmd and "kill -TERM -- -$gen" in cmd and "seq 240" in cmd
    subprocess.run(["bash", "-n", "-c", cmd], check=True)
