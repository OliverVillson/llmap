import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stages import ternary as T  # noqa: E402


def tiny(d, seed=0):
    from transformers import GraniteConfig, GraniteForCausalLM
    torch.manual_seed(seed)
    cfg = GraniteConfig(vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125)
    m = GraniteForCausalLM(cfg)
    m.save_pretrained(d)
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel({f"w{i}": i for i in range(64)}, unk_token="w0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="w1").save_pretrained(d)
    return m


def test_snap_is_ternary_per_block_with_a_bf16_mean_scale_and_passes_gradients():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    q = T.snap(w)
    b, qb = w.reshape(8, 2, 256), q.reshape(8, 2, 256)
    s = b.abs().mean(-1, keepdim=True).to(torch.bfloat16).float()
    assert torch.equal(qb, s * (b / s).round().clamp(-1, 1))
    assert torch.all(qb.abs().amax(-1) == s.squeeze(-1))  # what TQ1_0 stores as the block scale
    lin = torch.nn.Linear(512, 8, bias=False)
    lin.__class__ = T.TernaryLinear
    x = torch.randn(3, 512)
    lin(x).sum().backward()
    assert torch.allclose(lin.weight.grad, x.sum(0).expand(8, -1))  # straight through


def test_captured_layer_inputs_replay_the_model():
    from transformers import GraniteConfig, GraniteForCausalLM
    torch.manual_seed(0)
    m = GraniteForCausalLM(GraniteConfig(vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, residual_multiplier=0.5,
                                         embedding_multiplier=2.0)).eval()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        h, kw = T.layer_inputs(m, ids)
        for layer in T.decoder_layers(m):
            h = T.run_layer(layer, h, kw)
        mine = m.lm_head(m.model.norm(h)) / m.config.logits_scaling
        assert torch.allclose(mine, m(input_ids=ids).logits, atol=1e-4)


def traces(path, n=120, length=60, seed=0):
    """n rows in gen's format, two samples per problem."""
    rng = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for i in range(n):
            toks = torch.randint(0, 64, (length,), generator=rng).tolist()
            f.write(json.dumps({"id": f"cc/{i // 2}", "prompt": toks[:10], "tokens": toks[10:], "finish": "stop",
                                "passed": None, "forced": None}) + "\n")


def test_block_then_e2e_move_the_ternary_model_toward_the_teacher(tmp_path):
    base = tmp_path / "base"
    tiny(base)
    traces(tmp_path / "t.jsonl")
    common = ["--traces", str(tmp_path / "t.jsonl"), "--length", "32"]
    import sys
    from unittest import mock
    with mock.patch.object(sys, "argv", ["x", "block", "--base", str(base), "--windows", "16", "--epochs", "3",
                                         "--lr", "1e-3", "--batch", "4", "--out", str(tmp_path / "block"), *common]):
        T.main()
    rep = json.loads((tmp_path / "block" / "block.json").read_text())
    assert all(l["rel_err_trained"] < l["rel_err_snap"] for l in rep["layers"])
    with mock.patch.object(sys, "argv", ["x", "kl", "--student", str(base), "--teacher", str(base), "--snap",
                                         "--out", str(tmp_path / "snap.json"), *common]):
        T.main()
    snap_kl = json.loads((tmp_path / "snap.json").read_text())["kl"]
    with mock.patch.object(sys, "argv", ["x", "e2e", "--student", str(tmp_path / "block" / "latent"),
                                         "--teacher", str(base), "--tokens", "4096", "--lr", "1e-4", "--batch", "4",
                                         "--log-every", "16", "--out", str(tmp_path / "e2e"), *common]):
        T.main()
    e2e = json.loads((tmp_path / "e2e" / "e2e.json").read_text())
    kls = [h["kl"] for h in e2e["kl"]]
    assert kls[0] < snap_kl  # the block phase already beats a plain snap
    assert kls[-1] < kls[0]  # and the whole-model pass improves on it
    T.check_ternary(tmp_path / "e2e" / "snapped")
    assert 0 < e2e["size_gb_vocab24k"] < 1


def test_check_ternary_rejects_a_plain_model(tmp_path):
    tiny(tmp_path / "m")
    with pytest.raises(ValueError):
        T.check_ternary(tmp_path / "m")


def test_attention_stays_full_precision_and_is_sized_at_its_export_types(tmp_path):
    m = tiny(tmp_path / "m")
    n = T.ternary_model(m)
    layer = T.decoder_layers(m)[0]
    assert n == 2 * 3  # gate, up, down per layer
    assert type(layer.self_attn.q_proj) is torch.nn.Linear and type(layer.mlp.down_proj) is T.TernaryLinear
    attn = sum(p.numel() * T.ATTENTION[k.split(".")[-2]][1] for k, p in m.named_parameters() if "self_attn" in k) / 8e9
    mlp = sum(p.numel() for k, p in m.named_parameters() if ".mlp." in k) * T.TQ1_0_BPW / 8e9
    other = sum(p.numel() for k, p in m.named_parameters() if "norm" in k) * 32 / 8e9
    emb = 2 * 64 * 256 * 4.5 / 8e9
    assert abs(T.size_gb(m) - (attn + mlp + other + emb)) < 1e-9
