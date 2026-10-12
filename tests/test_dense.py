import re

import pytest

from stages import bits, dense
from stages._util import Config

L = 64


def test_w18s_gives_the_feed_forward_r50w95s_bits_on_qwen38():
    """Qwen3.8-27B's shapes: ~9.8B weights outside the feed-forward blocks at q8_0 is
    ~10.4 GB, and the 17.1B feed-forward weights at ~3.35 bits bring it to ~17.6 GB."""
    t = dense.synthetic_tensors()
    assert dense.ffn_params(t, L) / 1e9 == pytest.approx(17.11, abs=0.01)
    assert dense.static_gb(t, L) == pytest.approx(10.39, abs=0.05)
    plan, est, ffn_bits = dense.allocate_w(t, L, None)
    assert est == pytest.approx(17.56, abs=0.1) and 3.25 <= ffn_bits <= dense.FFN_BITS
    for l in plan:
        gu, d = bits.LADDER.index(l.gate_up), bits.LADDER.index(l.down)
        assert gu <= d <= gu + 2


def test_w18s_puts_bits_where_the_imatrix_energy_is():
    t = dense.synthetic_tensors()
    energy = {k: [1.0] * (L - 8) + [50.0] * 8 for k in ("gate", "up", "down")}
    plan, _, ffn_bits = dense.allocate_w(t, L, energy)
    layer_bits = [2 * bits.BPW[l.gate_up] + bits.BPW[l.down] for l in plan]
    assert min(layer_bits[-8:]) > max(layer_bits[:8]) and ffn_bits <= dense.FFN_BITS


def test_quantize_flags_name_the_right_tensors():
    plan = [bits.LayerBits(1, "q2_k", "q3_k"), bits.LayerBits(10, "q3_k", "q4_k")]
    args = dense.w_args(plan)
    types = dict(a.split("=") for a in args[5::2])
    assert args[:4] == ["--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"]

    def match(name):  # llama-quantize: the first pattern that regex_search-es the name
        return next((t for pat, t in types.items() if re.search(pat, name)), "base")

    assert match("blk.1.ffn_up.weight") == "q2_k" and match("blk.1.ffn_down.weight") == "q3_k"
    assert match("blk.10.ffn_gate.weight") == "q3_k" and match("blk.10.ffn_down.weight") == "q4_k"
    assert match("blk.1.attn_qkv.weight") == "base"
    q4 = dict(a.split("=") for a in dense.Q4_ARGS[5::2])
    attn = next(p for p, t in q4.items() if t == "q5_k")
    assert re.search(attn, "blk.3.attn_q.weight") and re.search(attn, "blk.3.attn_output.weight")
    assert not re.search(attn, "blk.0.attn_qkv.weight") and not re.search(attn, "blk.0.attn_gate.weight")


def test_gpu_layers_leaves_the_rest_in_ram():
    t = dense.synthetic_tensors()
    assert dense.gpu_layers(t, 180.0) == 999  # a B200: the output head goes on the GPU too
    assert dense.gpu_layers(t, 58.0) == L  # every layer fits, but not with the head and embeddings
    n = dense.gpu_layers(t, 45.0)
    assert 30 < n < L  # ~0.76 GB per BF16 layer, 6 GB kept free
    assert dense.gpu_layers(t, 4.0) == 0


def test_calibration_never_uses_livecodebench(tmp_path):
    for s in ("multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "livecodebench"):
        (tmp_path / f"{s}.jsonl").write_text("".join(
            f'{{"id": "{s}/{i}", "language": "py", "prompt": "p{i}", "tests": ""}}\n' for i in range(40)))
    probs = dense.calib_problems(str(tmp_path))
    assert not any(p["suite"] == "livecodebench" for p in probs)
    py = [p["id"] for p in probs if p["suite"] == "multipl-e-py"]
    assert len(py) == 10 and py[0] == "multipl-e-py/0" and py[-1] == "multipl-e-py/36"
    text = dense.chat_text("<|im_start|>assistant\n<think>\n", " think ", "```py\nx = 1\n```\n")
    assert text == "<|im_start|>assistant\n<think>\nthink\n</think>\n\n```py\nx = 1\n```<|im_end|>\n"


def test_imatrix_energy_reads_dense_feed_forward_tensors(tmp_path):
    """sum_j in_sum2[j] * ||W[:, j]||^2 per layer and projection, from real GGUF files."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    from stages import quantize

    w = {k: np.arange(1, 13, dtype=np.float32).reshape(3, 4) * (i + 1) for i, k in enumerate(("gate", "up", "down"))}
    m = gguf.GGUFWriter(str(tmp_path / "m.gguf"), "qwen35")
    for k, x in w.items():
        m.add_tensor(f"blk.0.ffn_{k}.weight", x)
    m.add_tensor("blk.0.attn_qkv.weight", np.ones((3, 4), dtype=np.float32))
    m.write_header_to_file(); m.write_kv_data_to_file(); m.write_tensors_to_file(); m.close()
    s2 = np.array([1.0, 2.0, 0.0, 1.0], dtype=np.float32)
    im = gguf.GGUFWriter(str(tmp_path / "im.gguf"), "imatrix")
    for k in w:
        im.add_tensor(f"blk.0.ffn_{k}.weight.in_sum2", s2.reshape(1, 4))
    im.write_header_to_file(); im.write_kv_data_to_file(); im.write_tensors_to_file(); im.close()

    class J:
        config = Config()

    quantize.gguf_lib = lambda job: gguf  # the pip package instead of llama.cpp's checkout
    e = dense.imatrix_energy(J(), tmp_path / "m.gguf", tmp_path / "im.gguf", 1)
    for k, x in w.items():
        assert e[k][0] == pytest.approx(float(s2 @ (x ** 2).sum(axis=0)))


def granite_3b_tensors():
    """Granite 4.2 3B's tensors from its config: 40 layers, hidden 2560, FFN 8192,
    40 query heads and 8 KV heads of 64, untied 100352-token embeddings."""
    T = dense.Tensor
    h, f, v = 2560, 8192, 100352
    out = [T("token_embd.weight", v * h, 2, (v, h)), T("output.weight", v * h, 2, (v, h)), T("output_norm.weight", h, 1)]
    for l in range(40):
        b = f"blk.{l}."
        for name, o, i in (("ffn_gate", f, h), ("ffn_up", f, h), ("ffn_down", h, f), ("attn_q", h, h),
                           ("attn_k", 512, h), ("attn_v", 512, h), ("attn_output", h, h)):
            out.append(T(b + name + ".weight", o * i, 2, (o, i)))
        out.append(T(b + "attn_norm.weight", h, 1))
    return out


@pytest.mark.parametrize("recipe, lo, hi", [("m35", 1.8, 2.0), ("m28", 1.45, 1.6)])
def test_small_model_recipes_shrink_attention_and_embeddings_too(recipe, lo, hi):
    tensors = granite_3b_tensors()
    target, static = dense.MIXED[recipe]
    plan, est, ffn_bits = dense.allocate_w(tensors, 40, None, target, static)
    assert lo < est < hi and abs(ffn_bits - target) < 0.3
    # q8_0 outside the feed-forward, as in w18s, would cost a 3B over half a GB
    assert dense.allocate_w(tensors, 40, None, target)[1] - est > 0.4
    args = dense.w_args(plan, static)
    assert args[:4] == ["--output-tensor-type", static["output"], "--token-embedding-type", static["token_embd"]]
    assert rf"attn_(q|k|v|output)\.weight={static['attn']}" in args


def test_m22_goes_below_the_ladder_with_iq2():
    args, est, ffn_bits = dense.fixed_plan(granite_3b_tensors(), 40, dense.FIXED["m22"])
    assert 1.1 < est < 1.3 and 2.1 < ffn_bits < 2.3
    assert rf"ffn_(gate|up)\.weight=iq2_xxs" in args and rf"ffn_down\.weight=iq2_s" in args
    assert rf"attn_(q|k|v|output)\.weight=iq3_xxs" in args and "m22" in dense.RECIPES


def test_bh15_leaves_room_for_the_repair_adapter_under_1_5_gb():
    args, est, ffn_bits = dense.fixed_plan(granite_3b_tensors(), 40, dense.FIXED["bh15"])
    assert 1.40 < est < 1.47 and 2.6 < ffn_bits < 2.8
    assert rf"attn_(q|k|v|output)\.weight=iq4_xs" in args and "bh15" in dense.RECIPES


def test_end_of_turn_reads_the_template(monkeypatch):
    import httpx

    class R:
        def __init__(self, text):
            self.text = text

        def json(self):
            return {"prompt": self.text}

    granite = ("<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
               "END-OF-ANSWER-MARK<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>again<|end_of_text|>\n")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(granite))
    assert dense.end_of_turn() == "<|end_of_text|>\n"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert dense.end_of_turn() == dense.END_OF_TURN
