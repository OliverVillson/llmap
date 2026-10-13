import json
import re
import subprocess
from pathlib import Path

import pytest

from stages import bits, dense, quantize
from stages._util import Config, Job

L = 64
ROOT = Path(__file__).resolve().parents[1]


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
    # the map holds the same bytes as the estimate
    tmap = dense.ffn_map(t, L, plan)
    dense.check_map(t, tmap)
    assert dense.planned_bytes(t, tmap) / 1e9 == pytest.approx(est, abs=0.01)


def test_w18s_puts_bits_where_the_imatrix_energy_is():
    t = dense.synthetic_tensors()
    energy = {k: [1.0] * (L - 8) + [50.0] * 8 for k in ("gate", "up", "down")}
    plan, _, ffn_bits = dense.allocate_w(t, L, energy)
    layer_bits = [2 * bits.BPW[l.gate_up] + bits.BPW[l.down] for l in plan]
    assert min(layer_bits[-8:]) > max(layer_bits[:8]) and ffn_bits <= dense.FFN_BITS


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
        out += [T(b + "attn_norm.weight", h, 1), T(b + "ffn_norm.weight", h, 1)]
    return out


def chosen(args, base, name):
    """llama-quantize's type for a tensor (llama_tensor_get_type at the pinned commit): the
    head and embedding flags first, then the FIRST --tensor-type whose pattern
    regex_search-es the name, else with --pure the base type."""
    flags = dict(zip(args, args[1:]))
    if name == "output.weight" and "--output-tensor-type" in flags:
        return flags["--output-tensor-type"]
    if name == "token_embd.weight" and "--token-embedding-type" in flags:
        return flags["--token-embedding-type"]
    pats = [a.split("=") for prev, a in zip(args, args[1:]) if prev == "--tensor-type"]
    return next((ty for pat, ty in pats if re.search(pat, name)), base.lower())


def test_every_tensor_gets_one_exact_flag():
    """Each map entry becomes one anchored, escaped flag that matches its tensor and no
    other, so llama-quantize's first-match rule can't pick another flag's type."""
    t = granite_3b_tensors()
    plan = [bits.LayerBits(l, "q2_k", "q3_k") for l in range(40)]
    plan[1], plan[10] = bits.LayerBits(1, "q4_k", "q5_k"), bits.LayerBits(10, "q3_k", "q6_k")
    static = dense.MIXED["m28"][1]
    tmap = dense.ffn_map(t, 40, plan, static)
    dense.check_map(t, tmap)
    args, base = dense.map_args(tmap)
    assert args[:5] == ["--pure", "--output-tensor-type", "q5_k", "--token-embedding-type", "q3_k"]
    pats = [a.split("=")[0] for prev, a in zip(args, args[1:]) if prev == "--tensor-type"]
    assert len(pats) == len(tmap) - 2 and r"^blk\.1\.ffn_up\.weight$" in pats
    for x in t:
        if x.name in tmap:
            assert chosen(args, base, x.name) == tmap[x.name], x.name
            assert sum(bool(re.search(p, x.name)) for p in pats) == (0 if x.name in ("output.weight", "token_embd.weight") else 1)
    assert tmap["blk.1.ffn_up.weight"] == "q4_k" and tmap["blk.10.ffn_up.weight"] == "q3_k"
    assert tmap["blk.10.ffn_down.weight"] == "q6_k" and tmap["blk.11.ffn_down.weight"] == "q3_k"
    assert tmap["blk.0.attn_k.weight"] == "q4_k" and "blk.0.attn_norm.weight" not in tmap
    # what the exact form avoids: the old unanchored patterns also caught other tensors
    assert re.search(r"output\.weight", "blk.0.attn_output.weight") and re.search(r"blk\.1", "blk.10.ffn_up.weight")
    assert base == "Q4_K"  # the type most tensors have (4 attention tensors a layer)


def test_q4_keeps_the_q4_k_m_choices_for_the_rest_of_qwen38():
    t = dense.synthetic_tensors()
    q4 = dense.by_kind(t, dense.FIXED["q4"])
    dense.check_map(t, q4)
    assert q4["blk.0.attn_qkv.weight"] == "q5_k" and q4["blk.0.ssm_out.weight"] == "q4_k"
    assert q4["blk.0.attn_gate.weight"] == "q4_k" and q4["blk.0.ssm_ba.weight"] == "q4_k"
    assert q4["blk.3.attn_q.weight"] == q4["blk.3.attn_output.weight"] == "q5_k"
    assert q4["blk.63.ffn_down.weight"] == q4["blk.5.ffn_gate.weight"] == "q4_k"
    assert q4["output.weight"] == "q6_k" and q4["token_embd.weight"] == "q4_k"
    assert not any(k.endswith(("norm.weight", "ssm_a", ".bias")) for k in q4)
    assert dense.map_args(q4)[1] == "Q4_K"
    q8 = dense.by_kind(t, dense.FIXED["q8"])
    assert set(q8.values()) == {"q8_0"} and set(q8) == set(q4)


@pytest.mark.parametrize("recipe, lo, hi", [("m35", 1.8, 2.0), ("m28", 1.45, 1.6)])
def test_small_model_recipes_shrink_attention_and_embeddings_too(recipe, lo, hi):
    tensors = granite_3b_tensors()
    target, static = dense.MIXED[recipe]
    plan, est, ffn_bits = dense.allocate_w(tensors, 40, None, target, static)
    assert lo < est < hi and abs(ffn_bits - target) < 0.3
    # q8_0 outside the feed-forward, as in w18s, would cost a 3B over half a GB
    assert dense.allocate_w(tensors, 40, None, target)[1] - est > 0.4
    tmap = dense.ffn_map(tensors, 40, plan, static)
    assert tmap["output.weight"] == static["output"] and tmap["token_embd.weight"] == static["token_embd"]
    assert {tmap[f"blk.7.attn_{p}.weight"] for p in ("q", "k", "v", "output")} == {static["attn"]}
    assert dense.planned_bytes(tensors, tmap) / 1e9 == pytest.approx(est, abs=0.01)


def test_m22_goes_below_the_ladder_with_iq2():
    t = granite_3b_tensors()
    tmap = dense.by_kind(t, dense.FIXED["m22"])
    dense.check_map(t, tmap)
    ffn = dense.ffn_summary(t, 40, tmap)
    assert 1.1 < dense.planned_bytes(t, tmap) / 1e9 < 1.3 and 2.1 < ffn["ffn_bits"] < 2.3
    assert tmap["blk.0.ffn_up.weight"] == "iq2_xxs" and tmap["blk.39.ffn_down.weight"] == "iq2_s"
    assert tmap["blk.0.attn_v.weight"] == "iq3_xxs" and "m22" in dense.RECIPES


def test_block_sizes_are_ggml_s():
    gguf = pytest.importorskip("gguf")
    for ty, (block, size) in dense.BLOCK.items():
        assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[ty.upper()]] == (block, size), ty


def test_check_map_refuses_what_llama_quantize_would_not_follow():
    T = dense.Tensor
    t = [T("token_embd.weight", 512 * 256, 2, (512, 256)), T("blk.0.ffn_down.weight", 256 * 320, 2, (256, 320)),
         T("blk.0.ffn_up.weight", 320 * 256, 2, (320, 256)), T("blk.0.attn_norm.weight", 256, 1)]
    good = {"token_embd.weight": "q3_k", "blk.0.ffn_down.weight": "q8_0", "blk.0.ffn_up.weight": "iq2_s"}
    dense.check_map(t, good)
    for bad, why in [({**good, "blk.0.ffn_down.weight": "q4_k"}, "rows of 320 don't divide into q4_k blocks"),
                     ({k: v for k, v in good.items() if k != "blk.0.ffn_up.weight"}, "blk.0.ffn_up.weight: no type"),
                     ({**good, "blk.1.ffn_up.weight": "q4_k"}, "blk.1.ffn_up.weight: not in the model"),
                     ({**good, "blk.0.attn_norm.weight": "q8_0"}, "leaves it as it is"),
                     ({**good, "token_embd.weight": "q3_km"}, "unknown type"),
                     ({**good, "blk.0.FFN_up.weight": "q4_k"}, "literally")]:
        with pytest.raises(ValueError, match=why):
            dense.check_map(t, bad)


def write_gguf(gguf, path, tensors, arch="llama"):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(1)
    for name, x in tensors.items():
        w.add_tensor(name, x)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_verify_reads_every_tensor_back(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    monkeypatch.setattr(quantize, "gguf_lib", lambda job: gguf)  # the pip package instead of llama.cpp's checkout
    write_gguf(gguf, tmp_path / "m.gguf", {"blk.0.ffn_up.weight": np.ones((4, 256), dtype=np.float16),
                                           "blk.0.ffn_down.weight": np.ones((256, 4), dtype=np.float16),
                                           "blk.0.ffn_norm.weight": np.ones(256, dtype=np.float32)})
    job = Job(tmp_path / "job")
    tensors, _ = dense.read_tensors(job, tmp_path / "m.gguf")
    assert {x.name: x.type for x in tensors}["blk.0.ffn_norm.weight"] == "f32"
    size = (tmp_path / "m.gguf").stat().st_size
    tmap = {"blk.0.ffn_up.weight": "f16", "blk.0.ffn_down.weight": "f16"}
    assert dense.verify(job, tmp_path / "m.gguf", tensors, tmap, size)["blk.0.ffn_norm.weight"] == "f32"
    with pytest.raises(RuntimeError, match="blk.0.ffn_up.weight: planned q4_k, written f16"):
        dense.verify(job, tmp_path / "m.gguf", tensors, {**tmap, "blk.0.ffn_up.weight": "q4_k"}, size)
    with pytest.raises(RuntimeError, match="blk.1.ffn_up.weight: planned f16, written nothing"):
        dense.verify(job, tmp_path / "m.gguf", tensors + [dense.Tensor("blk.1.ffn_up.weight", 1024, 2, (4, 256), "f16")],
                     tmap, size)
    with pytest.raises(RuntimeError, match="more than 3% off"):
        dense.verify(job, tmp_path / "m.gguf", tensors, tmap, int(size / 0.96))
    assert dense.overhead_bytes(tmp_path / "m.gguf", tensors) == size - (2 * 1024 * 2 + 256 * 4)


def test_calibration_never_uses_livecodebench(tmp_path):
    for s in ("multipl-e-py", "multipl-e-js", "multipl-e-ts", "multipl-e-cpp", "livecodebench"):
        (tmp_path / f"{s}.jsonl").write_text("".join(
            f'{{"id": "{s}/{i}", "language": "py", "prompt": "p{i}", "tests": ""}}\n' for i in range(40)))
    probs = dense.calib_problems(str(tmp_path))
    assert not any(p["suite"] == "livecodebench" for p in probs)
    py = [p["id"] for p in probs if p["suite"] == "multipl-e-py"]
    assert len(py) == 10 and py[0] == "multipl-e-py/0" and py[-1] == "multipl-e-py/36"
    text = dense.chat_text("<|start_of_role|>assistant<|end_of_role|><think>\n", " think ", "```py\nx = 1\n```\n",
                           "<|end_of_text|>\n")
    assert text == "<|start_of_role|>assistant<|end_of_role|><think>\nthink\n</think>\n\n```py\nx = 1\n```<|end_of_text|>\n"


def test_imatrix_energy_reads_dense_feed_forward_tensors(tmp_path, monkeypatch):
    """sum_j in_sum2[j] * ||W[:, j]||^2 per layer and projection, from real GGUF files."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")

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

    monkeypatch.setattr(quantize, "gguf_lib", lambda job: gguf)  # the pip package instead of llama.cpp's checkout
    e = dense.imatrix_energy(J(), tmp_path / "m.gguf", tmp_path / "im.gguf", 1)
    for k, x in w.items():
        assert e[k][0] == pytest.approx(float(s2 @ (x ** 2).sum(axis=0)))


def test_end_of_turn_reads_the_template_or_fails(monkeypatch):
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
    # no guessed marker (it used to fall back to Qwen's <|im_end|>, which Granite doesn't have)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="apply-template failed"):
        dense.end_of_turn()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R("user: hi\nassistant: END-OF-ANSWER-MARK\nuser: again\n"))
    with pytest.raises(RuntimeError, match="no end-of-turn marker"):
        dense.end_of_turn()


# ---------- calibration text from the traces ----------

CORPUS = ["Read an integer n and print twice its value.", "Let me think about the input format first.",
          "```python\nprint(int(input()) * 2)\n```", "def solve(a, b):\n    return a + b\n", "</think>"]


def trace_tokenizer(d):
    """A real byte-level BPE tokenizer with Granite's role tokens as special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(CORPUS * 20, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end_of_text|>").save_pretrained(d)
    return AutoTokenizer.from_pretrained(d)


def write_traces(path, tok, problems, samples=2, answer="Let me think about the input format first. " * 4):
    """gen's rows: samples answers per problem, written together."""
    rows = []
    for i in range(problems):
        prompt = tok.encode(f"<|start_of_role|>user<|end_of_role|>Problem {i}: read n and print twice its value."
                            "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", add_special_tokens=False)
        for s in range(samples):
            toks = tok.encode(f"{answer}</think> answer {i}.{s}<|end_of_text|>", add_special_tokens=False)
            rows.append({"id": f"cc/{i}", "prompt": prompt, "tokens": toks, "finish": "stop", "passed": s == 0,
                         "forced": None})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_trace_calibration_is_the_training_split_one_sample_each(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    from stages import traces

    monkeypatch.setattr(dense, "IMATRIX_CTX", 64)
    tok = trace_tokenizer(tmp_path / "tok")
    rows = write_traces(tmp_path / "traces.jsonl", tok, 40)
    job = Job(tmp_path / "job")
    out = dense.trace_calibration(job, tmp_path / "traces.jsonl", tmp_path / "tok", held_frac=0.25)
    text = out.read_text()
    held = {f"cc/{i}" for i in range(40) if traces.is_held(f"cc/{i}", 0.25)}
    assert 3 <= len(held) <= 20
    for i in range(40):
        inside = f"Problem {i}: " in text
        assert inside == (f"cc/{i}" not in held), i
        assert (f"answer {i}.0" in text) == inside and f"answer {i}.1" not in text  # the first sample only
    assert text.count("<|start_of_role|>") == 2 * (40 - len(held)) and "<|end_of_text|>" in text  # specials kept
    meta = json.loads(job.path("work", "calib.json").read_text())
    firsts = [rows[2 * i] for i in range(40) if f"cc/{i}" not in held]
    assert meta["problems"] == meta["train_problems"] == 40 - len(held) and meta["held_out_problems"] == len(held)
    assert meta["tokens"] == sum(len(r["prompt"]) + len(r["tokens"]) for r in firsts)
    # problems are shuffled by seed, not in file order
    order = [int(m) for m in re.findall(r"Problem (\d+):", text)]
    assert order != sorted(order)

    # the same traces: nothing to do, the tokenizer is never loaded
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: 1 / 0)
    assert dense.trace_calibration(job, tmp_path / "traces.jsonl", tmp_path / "tok", held_frac=0.25).read_text() == text
    monkeypatch.undo()
    monkeypatch.setattr(dense, "IMATRIX_CTX", 64)

    # other traces: written again, and stopped at the first problem that reaches max_tokens
    rows = write_traces(tmp_path / "traces.jsonl", tok, 40, answer="Read an integer n and print twice its value. " * 4)
    longest = max(len(r["prompt"]) + len(r["tokens"]) for r in rows)
    dense.trace_calibration(job, tmp_path / "traces.jsonl", tmp_path / "tok", max_tokens=4 * longest)
    meta = json.loads(job.path("work", "calib.json").read_text())
    assert 4 * longest <= meta["tokens"] < 5 * longest and out.read_text().count("Problem ") == meta["problems"] >= 4

    # experiment 05's llama-server text is never overwritten or reused
    old = Job(tmp_path / "exp05")
    old.path("work", "calib-chat.txt").write_text("llama-server answers")
    with pytest.raises(RuntimeError, match="its own job dir"):
        dense.trace_calibration(old, tmp_path / "traces.jsonl", tmp_path / "tok")
    assert old.path("work", "calib-chat.txt").read_text() == "llama-server answers"
    # too little text for two chunks
    with pytest.raises(ValueError, match="two chunks"):
        dense.trace_calibration(Job(tmp_path / "small"), tmp_path / "traces.jsonl", tmp_path / "tok", max_tokens=10)


# ---------- the real llama.cpp tools on a tiny model ----------

@pytest.fixture(scope="module")
def llama_cpp():
    for c in (Config().llama_cpp, ROOT.parent / "llama.cpp"):
        b = Path(c) / "build" / "bin"
        if all((b / x).exists() for x in ("llama-quantize", "llama-imatrix")):
            return Path(c)
    pytest.skip("llama.cpp's llama-quantize and llama-imatrix are not built")


def tiny_llama(path, layers=11, hidden=256, ffn=512, heads=4, kv=2):
    """A llama-architecture GGUF with F16 weights, rows of 256 and 512 (whole K-quant
    blocks), 11 layers (so blk.1 and blk.10 both exist) and a byte vocab, so
    llama-imatrix can read any text."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    vocab = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(256)
    w.add_embedding_length(hidden)
    w.add_block_count(layers)
    w.add_feed_forward_length(ffn)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_rope_dimension_count(hidden // heads)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(1)
    w.add_tokenizer_model("llama")
    w.add_token_list(vocab)
    w.add_token_scores([0.0] * len(vocab))
    w.add_token_types([2, 3, 3] + [6] * 256)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    rng = np.random.default_rng(0)

    def m(o, i):
        return (rng.standard_normal((o, i)) * 0.05).astype(np.float16)

    w.add_tensor("token_embd.weight", m(len(vocab), hidden))
    w.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
    w.add_tensor("output.weight", m(len(vocab), hidden))
    for l in range(layers):
        b = f"blk.{l}."
        w.add_tensor(b + "attn_norm.weight", np.ones(hidden, dtype=np.float32))
        for name, o, i in (("attn_q", hidden, hidden), ("attn_k", kv * hidden // heads, hidden),
                           ("attn_v", kv * hidden // heads, hidden), ("attn_output", hidden, hidden),
                           ("ffn_gate", ffn, hidden), ("ffn_up", ffn, hidden), ("ffn_down", hidden, ffn)):
            w.add_tensor(b + name + ".weight", m(o, i))
        w.add_tensor(b + "ffn_norm.weight", np.ones(hidden, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path


def tiny_job(root, llama_cpp):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"llama_cpp": str(llama_cpp)}))
    return Job(root)


def types_of(path):
    import gguf

    return {t.name: t.tensor_type.name.lower() for t in gguf.GGUFReader(str(path)).tensors}


def test_llama_quantize_takes_the_first_matching_flag(tmp_path, llama_cpp):
    """Why every flag is anchored: with overlapping patterns the earlier one wins, and
    an unanchored blk.1 also takes blk.10."""
    bf16 = tiny_llama(tmp_path / "m.gguf")
    subprocess.run([str(llama_cpp / "build/bin/llama-quantize"), "--pure", "--tensor-type", "ffn_up=q4_k",
                    "--tensor-type", r"blk\.1\.ffn_up=q8_0", "--tensor-type", "blk.1=q5_k",
                    "--tensor-type", r"^blk\.10\.ffn_gate\.weight$=q8_0", str(bf16), str(tmp_path / "q.gguf"), "Q2_K"],
                   check=True, capture_output=True)
    got = types_of(tmp_path / "q.gguf")
    assert got["blk.1.ffn_up.weight"] == "q4_k"  # not the later, more specific q8_0
    assert got["blk.10.ffn_gate.weight"] == "q5_k"  # blk.1 caught it before its exact flag
    assert got["blk.2.ffn_gate.weight"] == "q2_k" and got["output.weight"] == "q2_k"


def test_tiny_llama_from_traces_to_a_checked_gguf(tmp_path, llama_cpp, monkeypatch):
    """calib from traces, imatrix at a long context with the head, quantize with an
    explicit map and read every type back; then each step reuses its result only while
    its inputs match."""
    pytest.importorskip("tokenizers")
    monkeypatch.setattr(dense, "IMATRIX_CTX", 64)
    job = tiny_job(tmp_path / "job", llama_cpp)
    bf16 = tiny_llama(tmp_path / "bf16.gguf")
    tok = trace_tokenizer(tmp_path / "tok")
    write_traces(tmp_path / "traces.jsonl", tok, 30)
    dense.trace_calibration(job, tmp_path / "traces.jsonl", tmp_path / "tok")
    job.path("work", "imatrix.gguf").write_text("an old imatrix with no record of what made it")

    im = dense.importance(job, bf16, ctx=64)
    meta = json.loads(job.path("work", "imatrix.json").read_text())
    calib = json.loads(job.path("work", "calib.json").read_text())
    assert meta["key"]["chunks"] == calib["tokens"] // 64 and meta["chunk_size"] == 64
    assert meta["chunk_count"] >= meta["key"]["chunks"] >= 2
    names = set(types_of(im))
    assert "output.weight.in_sum2" in names and "blk.10.ffn_down.weight.in_sum2" in names  # --process-output
    stamp = im.stat().st_mtime_ns
    assert dense.importance(job, bf16, ctx=64).stat().st_mtime_ns == stamp  # same inputs: reused

    tensors, layers = dense.read_tensors(job, bf16)
    tmap = dense.by_kind(tensors, {"ffn_gate_up": "q2_k", "ffn_down": "q3_k", "attn": "q4_k", "output": "q6_k",
                                   "token_embd": "q3_k"})
    tmap.update({"blk.1.ffn_up.weight": "iq2_xxs", "blk.10.ffn_up.weight": "iq3_xxs", "blk.0.attn_v.weight": "q8_0",
                 "blk.10.attn_output.weight": "iq4_xs", "blk.1.ffn_down.weight": "q5_k"})
    rep = dense.quantize_recipe(job, "map", bf16, im, dict(tmap))
    out = job.path("out", "model.gguf")
    assert types_of(out) == {**{t.name: t.type for t in tensors}, **tmap}
    alloc = json.loads(job.path("work", "allocation.json").read_text())
    assert alloc["tensors"] == rep["tensors"] and alloc["key"]["map_sha256"] == dense.map_sha(tmap)
    assert alloc["key"]["imatrix_sha256"] == dense.sha256_file(im) and len(alloc["key"]["llama_cpp"]) == 40
    assert abs(alloc["size_bytes"] / alloc["est_bytes"] - 1) < 0.01 and job.is_done("quantize")
    assert {w["type"] for w in alloc["bit_widths"] if w["layer"] == 1} == {"q2_k", "iq2_xxs", "q5_k"}

    # same inputs: kept, and marked done again
    job.path(".done", "quantize").unlink()
    stamp = out.stat().st_mtime_ns
    assert dense.quantize_recipe(job, "map", bf16, im, dict(tmap))["tensors"] == rep["tensors"]
    assert out.stat().st_mtime_ns == stamp and job.is_done("quantize")

    # another map: rebuilt
    tmap["blk.10.ffn_up.weight"] = "q4_k"
    dense.quantize_recipe(job, "map", bf16, im, dict(tmap))
    assert types_of(out)["blk.10.ffn_up.weight"] == "q4_k"

    # other traces: new text, so a new imatrix, so a new model
    write_traces(tmp_path / "traces.jsonl", tok, 30, answer="Read an integer n and print twice its value. " * 4)
    dense.trace_calibration(job, tmp_path / "traces.jsonl", tmp_path / "tok")
    sha = dense.sha256_file(im)
    dense.importance(job, bf16, ctx=64)
    assert dense.sha256_file(im) != sha
    stamp = out.stat().st_mtime_ns
    dense.quantize_recipe(job, "map", bf16, im, dict(tmap))
    assert out.stat().st_mtime_ns != stamp
    assert json.loads(job.path("work", "allocation.json").read_text())["key"]["imatrix_sha256"] == dense.sha256_file(im)

    # a map llama-quantize can't follow fails before quantizing, and leaves nothing marked done
    with pytest.raises(ValueError, match="no type planned"):
        dense.quantize_recipe(job, "map", bf16, im, {k: v for k, v in tmap.items() if k != "blk.3.ffn_up.weight"})
    assert not job.is_done("quantize")
