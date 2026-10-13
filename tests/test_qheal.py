import inspect
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
pytest.importorskip("peft")

from stages import horizon, qheal  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from test_ternary import tiny, traces  # noqa: E402


def write_gguf(model, path, noise=0.0, drop=(), extra=()):
    """A llama-style GGUF of model's weights in Q8_0, q/k permuted as convert_hf_to_gguf.py does,
    optionally with noise added first (a stand-in for a coarser quantization), without the
    GGUF tensors in `drop` and with 2-D tensors named in `extra` added."""
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
        if g in drop:
            continue
        q = gguf.quants.quantize(a.astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(g, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    for g in extra:
        w.add_tensor(g, np.zeros((4, 32), dtype=np.float32))
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


def test_a_missing_tensor_raises_and_an_unmapped_one_is_logged(tmp_path, capsys):
    """A tensor kind the name map misses would keep the original's weights and score the
    quantized model too high: refused, naming it. A 2-D GGUF weight with no destination
    (a fused attn_qkv, say) is logged."""
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "miss.gguf", drop=("blk.1.ffn_down.weight", "output.weight"))
    with pytest.raises(ValueError, match=r"2 parameters .*lm_head\.weight.*model\.layers\.1\.mlp\.down_proj\.weight"):
        qheal.load_quantized(m, tmp_path / "miss.gguf", gguf)
    write_gguf(m, tmp_path / "extra.gguf", extra=("blk.0.attn_qkv.weight",))
    qheal.load_quantized(m, tmp_path / "extra.gguf", gguf)
    assert "blk.0.attn_qkv.weight" in capsys.readouterr().out


def test_parallel_read_is_the_sequential_read(tmp_path):
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf", noise=0.1)
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    e1 = qheal.load_quantized(m, tmp_path / "q.gguf", gguf, workers=0)
    a = {n: p.detach().clone() for n, p in m.named_parameters()}
    m2 = tiny(tmp_path / "m2")
    e2 = qheal.load_quantized(m2, tmp_path / "q.gguf", gguf, workers=2, ref=ref)
    assert e1 == e2
    assert all(torch.equal(a[n], p) for n, p in m2.named_parameters())


def test_export_writes_the_gguf_weights(tmp_path):
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf", noise=0.1)
    with mock.patch.object(sys, "argv", ["x", "export", "--hf", str(tmp_path / "m"), "--gguf", str(tmp_path / "q.gguf"),
                                         "--llama-cpp", str(tmp_path / "no-llama-cpp"), "--out", str(tmp_path / "hf")]):
        qheal.main()
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    saved = load_file(str(tmp_path / "hf" / "model.safetensors"))
    assert {t.dtype for t in saved.values()} == {torch.bfloat16}  # bf16 on the CPU too, not fp32 (twice the size)
    back = AutoModelForCausalLM.from_pretrained(tmp_path / "hf", dtype=torch.bfloat16)
    m2 = qheal.load_causal_lm(str(tmp_path / "m"))
    qheal.load_quantized(m2, tmp_path / "q.gguf", gguf)
    assert all(torch.equal(a, b.to(torch.bfloat16)) for a, b in zip(back.state_dict().values(), m2.state_dict().values()))


def test_kl_ladder_runs_the_teacher_once_and_reads_each_gguf_once(tmp_path, monkeypatch):
    m = tiny(tmp_path / "m")
    for name, noise in (("a", 0.05), ("b", 0.4)):
        write_gguf(m, tmp_path / f"{name}.gguf", noise=noise)
    traces(tmp_path / "t.jsonl", n=400)
    loads, reads = [], []
    real_load, real_reader = qheal.load_causal_lm, gguf.GGUFReader
    monkeypatch.setattr(qheal, "load_causal_lm", lambda p: loads.append(("student", p)) or real_load(p))
    monkeypatch.setattr(horizon, "load_causal_lm", lambda p: loads.append(("teacher", p)) or real_load(p))
    monkeypatch.setattr(gguf, "GGUFReader", lambda p, *a, **k: reads.append(Path(p).name) or real_reader(p, *a, **k))
    with mock.patch.object(sys, "argv", ["x", "kl", "--hf", str(tmp_path / "m"), "--gguf", str(tmp_path / "b.gguf"),
                                         str(tmp_path / "a.gguf"), "--traces", str(tmp_path / "t.jsonl"),
                                         "--llama-cpp", str(tmp_path / "no-llama-cpp"), "--length", "32",
                                         "--tokens", "200", "--workers", "0", "--out", str(tmp_path / "kl.json")]):
        qheal.main()
    assert [k for k, _ in loads] == ["teacher", "student"]
    assert sorted(reads) == ["a.gguf", "b.gguf"]
    rows = {Path(r["gguf"]).stem: r for r in json.loads((tmp_path / "kl.json").read_text())}
    assert 0 < rows["a"]["kl"] < rows["b"]["kl"]
    assert rows["a"]["quant_rel_err"]["q_proj"] < 0.01  # against the original, not the previous GGUF
    assert rows["a"]["bytes"] == (tmp_path / "a.gguf").stat().st_size


def test_heal_is_kept_for_reference_but_nothing_ships_it(tmp_path):
    """The LoRA repair still trains (reference), but there is no merge, no LoRA GGUF and
    no CLI for it."""
    m = tiny(tmp_path / "m")
    write_gguf(m, tmp_path / "q.gguf", noise=0.3)
    traces(tmp_path / "t.jsonl", n=200)
    rep = qheal.heal(str(tmp_path / "m"), tmp_path / "q.gguf", tmp_path / "t.jsonl", tmp_path / "out", gguf,
                     r=8, lr=3e-3, tokens=6000, length=32, batch=4, held_windows=8, log_every=10)
    kls = [h["kl"] for h in rep["kl"]]
    assert kls[-1] < 0.8 * kls[0]
    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert not hasattr(qheal, "to_gguf")
    assert "adapter" not in inspect.signature(qheal.export).parameters
    for argv in (["x", "--hf", "a", "--gguf", "b", "--llama-cpp", "c", "--out", "d"],
                 ["x", "export", "--hf", "a", "--gguf", "b", "--llama-cpp", "c", "--out", "d", "--adapter", "e"],
                 ["x", "heal", "--hf", "a"]):
        with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
            qheal.main()
