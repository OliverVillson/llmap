"""stages/kquant.py: K-quant bytes <-> integers <-> weights, checked bit for bit against
gguf-py and against real llama-quantize output."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

from stages import kquant as K  # noqa: E402
from stages._util import Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QT = {"q2_k": gguf.GGMLQuantizationType.Q2_K, "q3_k": gguf.GGMLQuantizationType.Q3_K,
      "q4_k": gguf.GGMLQuantizationType.Q4_K}
D_AT = {"q2_k": (80, 82), "q3_k": (108,), "q4_k": (0, 2)}  # byte offsets of d (and dmin) in a block


def bits(t) -> np.ndarray:
    return np.asarray(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t, dtype=np.float32).view(np.uint32)


def random_tensor(type, rows, cols, seed=0) -> np.ndarray:
    """Random bytes of a type's tensor, every bit pattern possible except non-finite d/dmin;
    a few blocks get d = 0, -0, a subnormal, the fp16 maximum and a negative d."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=K.bytes_for(type, rows * cols), dtype=np.uint8)
    blocks = raw.reshape(-1, K.BLOCK_BYTES[type])
    special = np.array([0.0, -0.0, 6e-8, 65504.0, -0.75], dtype=np.float16)
    for o in D_AT[type]:
        v = blocks[:, o:o + 2].copy().view(np.float16)
        v[~np.isfinite(v)] = np.float16(0.5)
        v[: len(special), 0] = special
        blocks[:, o:o + 2] = v.view(np.uint8)
    return raw.reshape(rows, -1)


def test_block_sizes_are_ggmls():
    for t in K.TYPES:
        assert gguf.GGML_QUANT_SIZES[QT[t]] == (K.QK_K, K.BLOCK_BYTES[t])
    assert K.BLOCK_BYTES == {"q2_k": 84, "q3_k": 110, "q4_k": 144}
    assert K.bytes_for("q2_k", 2560 * 8192) == 2560 * 8192 // 256 * 84
    assert K.bytes_for("q4_k", 256) == 144
    with pytest.raises(ValueError, match="256"):
        K.bytes_for("q3_k", 1000)
    with pytest.raises(ValueError):
        K.bytes_for("q5_k", 256)


@pytest.mark.parametrize("type", K.TYPES)
def test_any_bytes_unpack_and_pack_back_exactly_and_dequantize_as_gguf_py(type):
    raw = random_tensor(type, 37, 512)
    kq = K.from_gguf(raw, type, (37, 512))
    assert kq.shape == (37, 512) and kq.codes.dtype == torch.int8 and kq.d.dtype == torch.float16
    g = K.grid(type)
    assert kq.scales.shape == (37, 512 // g.sub) and kq.d.shape == (37, 2)
    assert (kq.mins is None) == (type == "q3_k")
    assert int(kq.codes.min()) == g.q[0] and int(kq.codes.max()) == g.q[1]
    assert int(kq.scales.min()) == g.sc[0] and int(kq.scales.max()) == g.sc[1]
    assert K.pack(kq) == raw.tobytes()
    assert K.pack(K.from_gguf(raw.tobytes(), type, (37, 512))) == raw.tobytes()  # bytes in, too
    ref = gguf.quants.dequantize(raw, QT[type])
    assert np.array_equal(bits(K.dequant(kq)), bits(ref))  # bit for bit, signed zeros included


def test_block_layout():
    """q2_k: scales (sc | m << 4), then four codes per byte (element 32j + l in bits 2j of
    byte l), d at byte 80, dmin at 82. q4_k: d first. q3_k: q + 4 split over qs and hmask."""
    codes = torch.zeros(1, 256, dtype=torch.int8)
    codes[0, 32], codes[0, 200] = 3, 1
    sc = torch.arange(16, dtype=torch.int8).reshape(1, 16)
    m = torch.full((1, 16), 2, dtype=torch.int8)
    b = K.pack(K.KQ("q2_k", codes, sc, m, torch.tensor([[1.5]], dtype=torch.float16),
                    torch.tensor([[0.25]], dtype=torch.float16)))
    assert len(b) == 84 and b[0] == 0x20 and b[15] == 0x2F
    assert b[16] == 3 << 2  # element 32: byte 0 of qs, bits 2-3
    assert b[16 + 32 + 8] == 1 << 4  # element 200 = 128 + 64 + 8: second half, byte 8, bits 4-5
    assert np.frombuffer(b[80:84], np.float16).tolist() == [1.5, 0.25]
    q4 = K.quantize(torch.randn(1, 256), "q4_k")
    b4 = K.pack(K.KQ("q4_k", q4.codes, q4.scales, q4.mins, torch.tensor([[1.5]], dtype=torch.float16),
                     torch.tensor([[0.25]], dtype=torch.float16)))
    assert len(b4) == 144 and np.frombuffer(b4[:4], np.float16).tolist() == [1.5, 0.25]
    c3 = torch.full((1, 256), -4, dtype=torch.int8)
    c3[0, 0] = 3  # q + 4 = 7: low bits 3 in qs, high bit set in hmask
    b3 = K.pack(K.KQ("q3_k", c3, torch.zeros(1, 16, dtype=torch.int8), None,
                     torch.ones(1, 1, dtype=torch.float16), None))
    assert b3[0] == 1 and b3[32] == 3 and set(b3[1:32]) == {0}
    assert b3[96:108] == bytes([0] * 8 + [0xAA] * 4)  # every sc + 32 = 0b100000


def test_pack_refuses_what_the_format_cannot_hold():
    kq = K.quantize(torch.randn(4, 256), "q2_k")
    bad = K.KQ("q2_k", kq.codes.clone(), kq.scales, kq.mins, kq.d, kq.dmin)
    bad.codes[0, 0] = 4
    with pytest.raises(ValueError, match="codes must be integers in 0..3"):
        K.pack(bad)
    with pytest.raises(ValueError, match="float16"):
        K.pack(K.KQ("q2_k", kq.codes, kq.scales, kq.mins, kq.d.float(), kq.dmin))
    q3 = K.quantize(torch.randn(4, 256), "q3_k")
    with pytest.raises(ValueError, match="-32..31"):
        K.pack(K.KQ("q3_k", q3.codes, q3.scales + 40, None, q3.d, None))
    with pytest.raises(ValueError, match="mins should be None"):
        K.pack(K.KQ("q3_k", q3.codes, q3.scales, q3.scales, q3.d, None))
    with pytest.raises(ValueError, match="bytes is not a q4_k tensor"):
        K.from_gguf(b"\0" * 144, "q4_k", (2, 256))


def weights(seed=0, rows=24, cols=512):
    """Gaussian rows with the awkward cases in: an all-zero sub-block (scale 0), an
    all-positive one (min clamped to 0) and a constant one."""
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 0.02
    w[0, :32] = 0
    w[1, 32:64] = w[1, 32:64].abs() + 0.01
    w[2, 64:96] = -0.03
    return w


@pytest.mark.parametrize("type", K.TYPES)
def test_forward_is_the_dequantization_of_the_export_bit_for_bit(type):
    qw = K.QuantWeight(weights(), type)
    assert qw.nudged == 0
    assert np.array_equal(bits(qw()), bits(K.dequant(qw.export())))
    assert np.array_equal(bits(qw()), bits(gguf.quants.dequantize(np.frombuffer(K.pack(qw.export()), np.uint8)
                                                                    .reshape(24, -1), QT[type])))
    err = float(((qw().detach() - weights()) ** 2).sum() / (weights() ** 2).sum())
    assert err < {"q2_k": 0.2, "q3_k": 0.05, "q4_k": 0.02}[type]
    # anywhere training takes the parameters, including off their grids and past the fp16 range
    torch.manual_seed(1)
    with torch.no_grad():
        qw.weight.add_(torch.randn_like(qw.weight) * 0.01)
        qw.d.mul_(1 + 0.3 * torch.randn_like(qw.d))
        qw.d[0, 0] = 1e6
        qw.scales.add_(torch.randn_like(qw.scales) * 3)
        if qw.mins is not None:
            qw.mins.add_(torch.randn_like(qw.mins) * 3)
    e = qw.export()
    assert np.array_equal(bits(qw()), bits(K.dequant(e)))
    assert e.d[0, 0].item() == 65504.0 and K.pack(e)  # packs: everything on its grid


@pytest.mark.parametrize("type", K.TYPES)
def test_frozen_codes_stay_put_and_reload_from_a_checkpoint(type):
    qw = K.QuantWeight(weights(), type)
    qw.freeze_codes()
    before = qw.export().codes.clone()
    assert qw.frozen and not qw.weight.requires_grad
    with torch.no_grad():
        qw.weight.add_(1.0)  # would move every code
        qw.d.mul_(1.01)
    assert torch.equal(qw.export().codes, before)
    assert np.array_equal(bits(qw()), bits(K.dequant(qw.export())))
    qw().sum().backward()
    assert qw.weight.grad is None and qw.d.grad is not None and torch.isfinite(qw.d.grad).all()
    fresh = K.QuantWeight(weights(), type)
    fresh.load_state_dict(qw.state_dict())
    assert fresh.frozen and torch.equal(fresh(), qw()) and not fresh.weight.requires_grad


def reference(qw):
    """The same fake quantization through plain autograd: rounding as x + (round(x) - x).detach()."""
    g = qw.grid
    k = K.QK_K // g.sub

    def ste(x, lo, hi):
        x = x.clamp(lo, hi)
        return x + (x.round() - x).detach()

    d = qw.d + (K._fp16(qw.d) - qw.d).detach()
    dl = K._sub_scale(d, ste(qw.scales, *g.sc), k)
    ml = None
    if g.m is not None:
        dm = qw.dmin + (K._fp16(qw.dmin) - qw.dmin).detach()
        ml = K._sub_scale(dm, ste(qw.mins, *g.m), k)
    q = ste(K._target(qw.weight, dl, ml, g), *g.q)
    return K._values(dl, ml, q.flatten(-2), g.sub)


@pytest.mark.parametrize("type", K.TYPES)
def test_gradients_are_the_straight_through_ones(type):
    torch.manual_seed(0)
    w = torch.randn(16, 512) * 0.02
    qw = K.QuantWeight(w, type)
    with torch.no_grad():  # scales and mins off their integers, inside their ranges
        qw.scales.copy_(qw.scales.clamp(qw.grid.sc[0] + 1, qw.grid.sc[1] - 1) + 0.2)
        if qw.mins is not None:
            qw.mins.copy_(qw.mins.clamp(1, qw.grid.m[1] - 1) - 0.2)
    G = torch.randn(16, 512)
    (qw() * G).sum().backward()
    mine = [p.grad.clone() for p in (qw.weight, *qw.quant_parameters())]
    for p in qw.parameters():
        p.grad = None
    out = reference(qw)
    assert torch.equal(out, qw().detach())
    (out * G).sum().backward()
    for a, p in zip(mine, (qw.weight, *qw.quant_parameters())):
        assert torch.allclose(a, p.grad, rtol=0, atol=1e-5 * float(p.grad.abs().max()))


@pytest.mark.parametrize("type", K.TYPES)
def test_starts_exactly_at_the_given_integers(type):
    """Codes that rounding the latent weight would not give (here: a weight off from what
    was quantized) move the latent weight onto the given code's value."""
    w = weights()
    kq = K.quantize(w, type)
    noisy = w + torch.randn_like(w) * 0.004
    qw = K.QuantWeight(noisy, type, kq)
    assert qw.nudged > 0
    assert np.array_equal(bits(qw()), bits(K.dequant(kq)))
    e = qw.export()
    assert torch.equal(e.scales, kq.scales) and torch.equal(e.d, kq.d)
    assert K.QuantWeight(w, type, kq).nudged == 0


def test_take_rows_moves_whole_rows():
    kq = K.quantize(torch.randn(8, 512), "q4_k")
    idx = np.array([3, 1, 2, 0, 7, 6, 5, 4])
    assert torch.equal(K.dequant(K.take_rows(kq, idx)), K.dequant(kq)[idx])
    rows = np.frombuffer(K.pack(kq), np.uint8).reshape(8, -1)
    assert K.pack(K.take_rows(kq, idx)) == rows[idx].tobytes()


# ---------- real llama-quantize output ----------

def llama_cpp_dir() -> Path:
    for c in (Config().llama_cpp, ROOT.parent / "llama.cpp"):
        if (Path(c) / "build" / "bin" / "llama-quantize").exists() and (Path(c) / "convert_hf_to_gguf.py").exists():
            return Path(c)
    pytest.skip("llama.cpp's llama-quantize is not built")


def can_convert():
    """convert_hf_to_gguf imports sentencepiece for every llama-family model."""
    if importlib.util.find_spec("sentencepiece") is None:
        pytest.skip("sentencepiece is not installed, so convert_hf_to_gguf.py cannot convert a Granite model")


def tiny_granite(d, seed=0, vocab=320, init=0.02):
    """A GraniteForCausalLM with rows of 256 and 512 (whole K-quant blocks), q/k heads that
    differ (4 and 2, so q and k are permuted differently) and a llama-style byte-fallback
    BPE tokenizer, which convert_hf_to_gguf.py reads without a known pre-tokenizer hash."""
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast

    letters = "abcdefghijklmnopqrstuvwxyz"
    v = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        v[f"<0x{b:02X}>"] = len(v)
    for t in ["▁", *letters]:
        v[t] = len(v)
    merges = ([("▁", c) for c in letters] + [("a", c) for c in letters[1:]])[: vocab - len(v)]
    for a, b in merges:
        v[a + b] = len(v)
    assert len(v) == vocab
    tk = Tokenizer(models.BPE(vocab=v, merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tk.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tk.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                    decoders.Strip(" ", 1, 0)])
    torch.manual_seed(seed)
    cfg = GraniteConfig(vocab_size=vocab, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125,
                        logits_scaling=2.0, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
                        initializer_range=init)
    m = GraniteForCausalLM(cfg)
    m.save_pretrained(d)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tok.save_pretrained(d)
    return m


# llama-quantize's --tensor-type patterns are unanchored regexes and the first match wins:
# anchor each one, with --pure so no k-quant mixture rule applies besides them.
PLAN = {"attn_k": "q4_k", "attn_v": "q4_k", "ffn_down": "q3_k", "token_embd": "q2_k", "output": "q4_k"}
TYPE_FLAGS = ["--pure", "--token-embedding-type", "q2_k", "--output-tensor-type", "q4_k"]
for _kind in ("attn_k", "attn_v", "ffn_down"):
    TYPE_FLAGS += ["--tensor-type", rf"^blk\.[0-9]+\.{_kind}\.weight$={PLAN[_kind]}"]


def planned_type(gguf_name: str) -> str:
    kind = gguf_name.split(".")[-2] if gguf_name.startswith("blk.") else gguf_name.split(".")[0]
    return PLAN.get(kind, "q2_k")


def llama_quantize(hf, out_dir: Path, llama_cpp: Path, imatrix: bool) -> Path:
    """hf converted at f32 (the exact weights) and quantized by llama-quantize: q2_k with
    k, v and the head at q4_k and ffn_down at q3_k; with imatrix, importance-weighted."""
    f32 = out_dir / "f32.gguf"
    if not f32.exists():
        subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf), "--outtype", "f32",
                        "--outfile", str(f32)], check=True, capture_output=True)
    bin_ = llama_cpp / "build" / "bin"
    flags = list(TYPE_FLAGS)
    if imatrix:
        rng = np.random.default_rng(0)
        text = out_dir / "calib.txt"
        text.write_text(" ".join("".join(rng.choice(list("abcdefghij"), rng.integers(1, 6))) for _ in range(3000)))
        subprocess.run([str(bin_ / "llama-imatrix"), "-m", str(f32), "-f", str(text), "-c", "128", "--chunks", "8",
                        "-o", str(out_dir / "imatrix.gguf"), "-t", "2"], check=True, capture_output=True)
        flags = ["--imatrix", str(out_dir / "imatrix.gguf")] + flags
    out = out_dir / f"q2k{'-imatrix' if imatrix else ''}.gguf"
    subprocess.run([str(bin_ / "llama-quantize"), *flags, str(f32), str(out), "q2_k", "2"], check=True,
                   capture_output=True)
    return out


def qk_index(n_rows, n_head):
    """convert_hf_to_gguf's q/k row permutation: GGUF row r is HF row idx[r]."""
    return np.arange(n_rows).reshape(n_head, 2, n_rows // n_head // 2).swapaxes(1, 2).reshape(-1)


@pytest.mark.parametrize("imatrix", [False, True])
def test_llama_quantize_output_round_trips_and_starts_qat_on_its_codes(tmp_path, imatrix):
    llama_cpp = llama_cpp_dir()
    can_convert()
    if imatrix and not (llama_cpp / "build" / "bin" / "llama-imatrix").exists():
        pytest.skip("llama-imatrix is not built")
    m = tiny_granite(tmp_path / "hf")
    q = llama_quantize(tmp_path / "hf", tmp_path, llama_cpp, imatrix)
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GRANITE, 2)
    hf = {names.get_name(n, try_suffixes=(".weight",)): (n, p.detach()) for n, p in m.named_parameters()}
    seen = set()
    for t in gguf.GGUFReader(str(q)).tensors:
        typ = t.tensor_type.name.lower()
        if len(t.shape) < 2:
            continue
        assert typ == planned_type(t.name), t.name
        seen.add(typ)
        shape = tuple(int(x) for x in reversed(t.shape.tolist()))
        kq = K.from_gguf(t.data, typ, shape)
        assert K.pack(kq) == t.data.tobytes(), t.name
        assert np.array_equal(bits(K.dequant(kq)), bits(gguf.quants.dequantize(t.data, t.tensor_type))), t.name
        kind = t.name.split(".")[-2]
        if kind in ("attn_q", "attn_k"):  # the GGUF's rows are in rotary order; QAT works in HF order
            kq = K.take_rows(kq, np.argsort(qk_index(shape[0], 4 if kind == "attn_q" else 2)))
        qw = K.QuantWeight(hf[t.name][1], typ, kq)
        # QuantWeight's rounding is llama-quantize's last step: the same codes, nothing moved
        assert qw.nudged == 0, t.name
        assert torch.equal(qw.export().codes, kq.codes), t.name
        assert np.array_equal(bits(qw()), bits(K.dequant(kq))), t.name
    assert seen == {"q2_k", "q3_k", "q4_k"}
