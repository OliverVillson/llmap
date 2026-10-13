"""stages/gguf_build.py on a tiny GraniteForCausalLM: HF -> convert_hf_to_gguf.py ->
trained K-quant integers -> GGUF -> gguf-py, llama.cpp and the vLLM checkpoint all see
exactly those integers."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

from stages import gguf_build as GB  # noqa: E402
from stages import horizon  # noqa: E402
from stages import kquant as K  # noqa: E402
from stages import qheal  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_kquant import bits, can_convert, llama_cpp_dir, llama_quantize, planned_type, tiny_granite  # noqa: E402


def test_qk_rows_is_the_converters_permutation():
    for rows, heads in ((256, 4), (128, 2), (2560, 20)):
        w = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        assert np.array_equal(w[GB.qk_rows(rows, heads)], qheal.permute(w, heads))
        assert np.array_equal(qheal.unpermute(w[GB.qk_rows(rows, heads)], heads), w)


def test_row_order_tells_as_is_from_permuted_and_refuses_anything_else(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.standard_normal((256, 256)).astype(np.float32)
    k = rng.standard_normal((128, 256)).astype(np.float32)
    w = gguf.GGUFWriter(str(tmp_path / "t.gguf"), "granite")
    w.add_tensor("blk.0.attn_q.weight", q)  # as is
    w.add_tensor("blk.0.attn_k.weight", k[GB.qk_rows(128, 2)])  # permuted
    w.add_tensor("blk.0.attn_v.weight", k[::-1].copy())  # neither
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    t = {x.name: x for x in gguf.GGUFReader(str(tmp_path / "t.gguf")).tensors}
    heads = {"attn_q": 4, "attn_k": 2}
    assert GB._row_order(gguf, t["blk.0.attn_q.weight"], q, "blk.0.attn_q.weight", heads) is None
    assert np.array_equal(GB._row_order(gguf, t["blk.0.attn_k.weight"], k, "blk.0.attn_k.weight", heads),
                          GB.qk_rows(128, 2))
    with pytest.raises(ValueError, match="not its HF weight"):
        GB._row_order(gguf, t["blk.0.attn_v.weight"], k, "blk.0.attn_v.weight", heads)


# ---------- the real converter ----------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tiny Granite, its weights trained a little in their K-quant grids (q2_k, with
    k, v and the head at q4_k and ffn_down at q3_k, as test_kquant's llama-quantize plan),
    and the converter's output for it, cached for the tests that only check refusals."""
    llama_cpp = llama_cpp_dir()
    can_convert()
    root = tmp_path_factory.mktemp("tiny")
    m = tiny_granite(root / "hf", init=0.1)  # big enough weights that logits are peaked, so the KL is telling
    torch.manual_seed(2)
    kqs = {}
    for name, p in m.named_parameters():
        if p.dim() < 2:
            continue
        gname = gguf_name(name)
        qw = K.QuantWeight(p.detach(), planned_type(gname))
        with torch.no_grad():  # stand-in for training: move the latent weights and scales
            qw.weight.add_(torch.randn_like(qw.weight) * float(qw.weight.std()) * 0.3)
            qw.scales.add_(torch.randn_like(qw.scales))
        qw.project_()
        kqs[name] = qw.export()
    converted = GB.convert(root / "hf", root / "converted.gguf", llama_cpp)
    return {"root": root, "hf": root / "hf", "model": m, "kqs": kqs, "llama_cpp": llama_cpp, "converted": converted}


def gguf_name(hf: str) -> str:
    return gguf.get_tensor_name_map(gguf.MODEL_ARCH.GRANITE, 2).get_name(hf, try_suffixes=(".weight",))


def packed(kqs):
    return {n: (kq.type, K.pack(kq)) for n, kq in kqs.items()}


@pytest.fixture()
def fast_convert(tiny, monkeypatch):
    """convert() copies the cached conversion instead of running the converter again."""
    def copy(hf_dir, out, llama_cpp, outtype="bf16"):
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tiny["converted"], out)
        return out
    monkeypatch.setattr(GB, "convert", copy)


def test_built_gguf_holds_the_packed_integers_for_gguf_py_llama_cpp_and_vllm(tiny, tmp_path):
    kqs, hf = tiny["kqs"], tiny["hf"]
    out = tmp_path / "packed.gguf"
    rep = GB.build(hf, packed(kqs), out, tiny["llama_cpp"])
    assert rep["bytes"] == out.stat().st_size and rep["gb"] == pytest.approx(rep["bytes"] / 1e9)
    assert not list(tmp_path.glob(".*.build"))  # the conversion is cleaned up
    reader = gguf.GGUFReader(str(out))
    assert reader.get_field("general.file_type").contents() == int(gguf.LlamaFileType.MOSTLY_Q2_K)
    tensors = {t.name: t for t in reader.tensors}
    assert set(tensors) == set(rep["tensors"])
    n_q = 0
    for name, kq in kqs.items():
        gname = gguf_name(name)
        t = tensors[gname]
        assert rep["tensors"][gname] == {"hf": name, "type": kq.type, "bytes": K.bytes_for(kq.type, kq.codes.numel())}
        assert t.tensor_type.name.lower() == kq.type
        kind = gname.split(".")[-2]
        rows = None if kind not in ("attn_q", "attn_k") else GB.qk_rows(kq.shape[0], 4 if kind == "attn_q" else 2)
        want = kq if rows is None else K.take_rows(kq, rows)  # whole packed rows, in llama.cpp's rotary order
        n_q += rows is not None
        assert t.data.tobytes() == K.pack(want), gname
        assert np.array_equal(bits(gguf.quants.dequantize(t.data, t.tensor_type)), bits(K.dequant(want))), gname
    assert n_q == 4
    norms = [t for t in reader.tensors if len(t.shape) == 1]
    assert norms and all(t.tensor_type == gguf.GGMLQuantizationType.F32 for t in norms)

    # llama.cpp runs it, and its logits are those of the same weights stored as f32 (so q/k
    # rows land where llama.cpp's rotary embedding expects them)
    ref = tmp_path / "dequant-f32.gguf"
    GB.build(hf, {n: ("f32", K.dequant(kq).numpy().tobytes()) for n, kq in kqs.items()}, ref, tiny["llama_cpp"])
    kld = llama_kld(tiny["llama_cpp"], ref, out, tmp_path)
    if kld is not None:  # not 0: llama.cpp's K-quant matmuls round the activations to 8 bits (Q8_K)
        assert kld < 0.01  # measured 6e-4, against 0.6 with one layer's q rows misplaced
        broken = dict(packed(kqs))
        q0 = "model.layers.0.self_attn.q_proj.weight"
        broken[q0] = ("q2_k", K.pack(K.take_rows(kqs[q0], np.argsort(GB.qk_rows(256, 4)))))  # rows in the wrong order
        GB.build(hf, broken, tmp_path / "broken.gguf", tiny["llama_cpp"])
        assert llama_kld(tiny["llama_cpp"], ref, tmp_path / "broken.gguf", tmp_path) > max(0.1, 100 * kld)

    # vLLM's checkpoint: the file's dequantization at bf16, norms included; logits equal the
    # model with kquant.dequant's weights
    from transformers import AutoModelForCausalLM
    res = GB.vllm_checkpoint(out, hf, tmp_path / "vllm")
    assert set(res["rel_err"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                   "embed_tokens", "lm_head"}
    served = AutoModelForCausalLM.from_pretrained(str(tmp_path / "vllm"), dtype=torch.bfloat16).eval()
    assert next(served.parameters()).dtype == torch.bfloat16
    want = AutoModelForCausalLM.from_pretrained(str(hf), dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in want.named_parameters():
            if name in kqs:
                p.copy_(K.dequant(kqs[name]).to(torch.bfloat16))
    sd = dict(served.named_parameters())
    for name, p in want.named_parameters():
        assert torch.equal(sd[name], p), name
    ids = torch.randint(0, 320, (2, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(served(ids).logits, want(ids).logits)
    assert (tmp_path / "vllm" / "tokenizer.json").exists()


def llama_kld(llama_cpp: Path, base: Path, model: Path, tmp: Path) -> float | None:
    """Mean KL divergence of model's next-token distributions from base's (llama-perplexity),
    or None without the binary."""
    exe = llama_cpp / "build" / "bin" / "llama-perplexity"
    if not exe.exists():
        return None
    rng = np.random.default_rng(1)
    text = tmp / "kld.txt"
    text.write_text(" ".join("".join(rng.choice(list("abcdefghij"), rng.integers(1, 6))) for _ in range(2000)))
    logits = tmp / f"{base.stem}.logits"
    if not logits.exists():
        subprocess.run([str(exe), "-m", str(base), "-f", str(text), "-c", "64", "--chunks", "4", "-t", "2",
                        "--kl-divergence-base", str(logits)], check=True, capture_output=True)
    # horizon's runner: the pinned llama-perplexity can lose its summary under load, and is run again
    return horizon._kld_run([str(exe), "-m", str(model), "-c", "64", "--chunks", "4", "-t", "2",
                             "--kl-divergence-base", str(logits), "--kl-divergence"])["kld"]


def test_llama_quantize_output_read_back_and_rebuilt_is_the_same_bytes(tiny, tmp_path):
    """read_kq hands QAT llama-quantize's integers in HF row order (a QuantWeight on the HF
    weights starts on exactly those codes), and build puts them back byte for byte."""
    q = llama_quantize(tiny["hf"], tmp_path, tiny["llama_cpp"], imatrix=False)
    kqs = GB.read_kq(q, tiny["hf"], gguf)
    params = dict(tiny["model"].named_parameters())
    assert set(kqs) == {n for n, p in params.items() if p.dim() == 2}
    for name, kq in kqs.items():
        assert K.QuantWeight(params[name].detach(), kq.type, kq).nudged == 0, name
    rep = GB.build(tiny["hf"], packed(kqs), tmp_path / "rebuilt.gguf", tiny["llama_cpp"])
    orig = {t.name: t for t in gguf.GGUFReader(str(q)).tensors}
    for t in gguf.GGUFReader(str(tmp_path / "rebuilt.gguf")).tensors:
        assert t.tensor_type == orig[t.name].tensor_type, t.name
        assert t.data.tobytes() == orig[t.name].data.tobytes(), t.name
    assert abs(rep["bytes"] - q.stat().st_size) < 0.01 * q.stat().st_size


def test_build_refuses_a_partial_or_wrong_replacement(tiny, tmp_path, fast_convert):
    hf, lc = tiny["hf"], tiny["llama_cpp"]
    good = packed(tiny["kqs"])
    out = tmp_path / "x.gguf"
    up = "model.layers.1.mlp.up_proj.weight"
    with pytest.raises(ValueError, match=r"1 matrices of the converted GGUF have no packed bytes.*blk\.1\.ffn_up"):
        GB.build(hf, {k: v for k, v in good.items() if k != up}, out, lc)
    with pytest.raises(ValueError, match="maps to no GGUF name"):
        GB.build(hf, {**good, "model.mystery.weight": good[up]}, out, lc)
    with pytest.raises(ValueError, match=r"bytes, but q2_k of shape \(512, 256\)"):
        GB.build(hf, {**good, up: ("q2_k", good[up][1][:-84])}, out, lc)
    with pytest.raises(ValueError, match=r"bytes, but q4_k"):
        GB.build(hf, {**good, up: ("q4_k", good[up][1])}, out, lc)
    with pytest.raises(ValueError, match="not a ggml type"):
        GB.build(hf, {**good, up: ("q2_kk", good[up][1])}, out, lc)
    assert not out.exists() and not list(tmp_path.glob(".*"))
    rep = GB.build(hf, good, out, lc)
    assert GB.build(hf, good, out, lc, expect_gb=rep["gb"] * 1.004)["bytes"] == rep["bytes"]
    before = out.stat().st_mtime_ns
    with pytest.raises(ValueError, match=r"more than 0\.5% off"):
        GB.build(hf, good, out, lc, expect_gb=rep["gb"] * 1.02)
    assert out.stat().st_mtime_ns == before  # a file that fails its checks replaces nothing


def test_vllm_checkpoint_refuses_a_gguf_built_from_other_weights(tiny, tmp_path, fast_convert):
    out = tmp_path / "x.gguf"
    GB.build(tiny["hf"], packed(tiny["kqs"]), out, tiny["llama_cpp"])
    other = tmp_path / "other"
    shutil.copytree(tiny["hf"], other)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(other))
    with torch.no_grad():
        m.model.layers[0].input_layernorm.weight.mul_(1.5)  # trained norms that never made it into the GGUF
    m.save_pretrained(other)
    with pytest.raises(ValueError, match="attn_norm.weight .* is not model.layers.0.input_layernorm.weight"):
        GB.vllm_checkpoint(out, other, tmp_path / "vllm")
    assert not (tmp_path / "vllm").exists()
