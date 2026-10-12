import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
pytest.importorskip("peft")

from stages import qheal  # noqa: E402
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_ternary import tiny, traces  # noqa: E402


def write_gguf(model, path, noise=0.0):
    """A llama-style GGUF of model's weights in Q8_0, q/k permuted as convert_hf_to_gguf.py does,
    optionally with noise added first (a stand-in for a coarser quantization)."""
    w = gguf.GGUFWriter(str(path), "granite")
    rng = np.random.default_rng(0)
    inv = {v: k for k, v in qheal._GGUF.items()}
    cfg = model.config
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        a = p.detach().float().numpy()
        a = a + noise * a.std() * rng.standard_normal(a.shape).astype(np.float32)
        if name == "model.embed_tokens.weight":
            g = "token_embd.weight"
        elif name == "lm_head.weight":
            g = "output.weight"
        else:
            parts = name.split(".")
            kind = ".".join(parts[3:5])
            g = f"blk.{parts[2]}.{inv[kind]}.weight"
            if parts[4] in ("q_proj", "k_proj"):
                a = qheal.permute(a, cfg.num_attention_heads if parts[4] == "q_proj" else cfg.num_key_value_heads)
        q = gguf.quants.quantize(a.astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(g, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_permute_round_trips():
    a = np.arange(8 * 6 * 3, dtype=np.float32).reshape(48, 3)
    assert np.array_equal(qheal.unpermute(qheal.permute(a, 4), 4), a)
    assert qheal.hf_name("blk.3.attn_output.weight") == "model.layers.3.self_attn.o_proj.weight"
    assert qheal.hf_name("blk.3.attn_norm.weight") is None


def test_loads_the_gguf_weights_in_hf_order(tmp_path):
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf")
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    errs = qheal.load_quantized(m, tmp_path / "q.gguf", gguf)
    assert set(errs) >= {"q_proj", "k_proj", "down_proj", "embed_tokens", "lm_head"}
    assert max(errs.values()) < 1e-3  # Q8_0, un-permuted correctly
    q = m.model.layers[0].self_attn.q_proj.weight
    assert not torch.equal(q, ref["model.layers.0.self_attn.q_proj.weight"])  # really replaced


def test_a_mismatched_gguf_is_refused(tmp_path):
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf")  # another model's weights
    with pytest.raises(ValueError):
        qheal.load_quantized(tiny(tmp_path / "m2", seed=1), tmp_path / "q.gguf", gguf)


def test_the_adapter_closes_the_gap_to_the_original(tmp_path):
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf", noise=0.3)
    traces(tmp_path / "t.jsonl", n=200)
    rep = qheal.heal(str(tmp_path / "m"), tmp_path / "q.gguf", tmp_path / "t.jsonl", tmp_path / "out", gguf,
                     r=8, lr=3e-3, tokens=6000, length=32, batch=4, held_windows=8, log_every=10)
    kls = [h["kl"] for h in rep["kl"]]
    assert kls[-1] < 0.8 * kls[0]
    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert json.loads((tmp_path / "out" / "qheal.json").read_text())["lora_r"] == 8
