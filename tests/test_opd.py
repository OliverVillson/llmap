import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stages import opd  # noqa: E402


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
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast

    words = ["<unk>", "<|end_of_text|>", "user", "assistant", "<think>", "</think>"] + [f"w{i}" for i in range(58)]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end_of_text|>", pad_token="<|end_of_text|>")
    tok.chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant <think>"
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = GraniteConfig(vocab_size=len(words), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    GraniteForCausalLM(cfg).save_pretrained(d)
    return tok


def test_rounds_train_toward_the_teacher_and_reload_the_adapter(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    base = tmp_path / "student"
    tok = tiny_student(base)
    answer = tok.encode("w1 w2 w3 </think> w4", add_special_tokens=False)
    loaded = []
    monkeypatch.setattr(opd, "render", lambda tok, p: tok.encode("user w9 assistant <think>", add_special_tokens=False))
    monkeypatch.setattr(opd, "sample", lambda url, model, prompt, n, *a: [{"tokens": answer, "finish": "stop"}] * n)
    # the teacher is sure of every token the student wrote
    monkeypatch.setattr(opd, "teacher_logprobs", lambda url, model, prompt, ans: [0.0] * len(ans))
    monkeypatch.setattr(opd, "load_adapter", lambda url, name, path, old: loaded.append(
        (name, old, (path / "adapter_config.json").exists())))
    monkeypatch.setattr(opd, "passes", lambda tok, items: [True] * len(items))

    probs = [{"prompt": "x", "language": "python", "tests": []}] * 4
    out = opd.train_opd(str(base), probs, tmp_path / "work" / "opd", "s", "t", rounds=4,
                        problems_per_round=2, samples=2, lr=2e-2, r=4, alpha=8)

    assert loaded == [("opd1", None, True), ("opd2", "opd1", True), ("opd3", "opd2", True), ("opd4", "opd3", True)]
    rows = [json.loads(l) for l in (tmp_path / "work" / "opd-log.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == [1, 2, 3, 4] and rows[0]["answers"] == 4 and rows[0]["pass"] == 1.0
    assert abs(rows[-1]["kl_per_token"]) < abs(rows[0]["kl_per_token"])  # the student moved toward the teacher
    assert (out / "config.json").exists() and not (tmp_path / "work" / "opd.adapters").exists()
    merged = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in n for n, _ in merged.named_parameters())
