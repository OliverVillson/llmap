"""The vLLM 4-bit path: the size model (stages/w4a16.py, no torch) and the real
quantize stage (stages/quantize_ct.py) on tiny random Qwen3.6 and Gemma 4 MoE models."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common.progress import parse
from stages import quantize, quantize_ct, w4a16
from stages._util import Job

ROOT = Path(__file__).resolve().parents[1]

# Qwen3.6-35B-A3B's text config (30 Gated DeltaNet + 10 full-attention layers).
QWEN36 = dict(
    hidden_size=2048, num_hidden_layers=40, full_attention_interval=4, num_experts=256, num_experts_per_tok=8,
    moe_intermediate_size=512, shared_expert_intermediate_size=512, vocab_size=248320, tie_word_embeddings=False,
    num_attention_heads=16, num_key_value_heads=2, head_dim=256, linear_num_key_heads=16, linear_key_head_dim=128,
    linear_num_value_heads=32, linear_value_head_dim=128, linear_conv_kernel_dim=4,
)


def test_104_experts_fit_under_ten_gb():
    p = w4a16.part_bytes(QWEN36, 104)
    # 104 x 40 x 3 x 2048 x 512 weights at 4.125 bits, plus ~2.9 GB for the rest
    assert p["routed_experts"] / 1e9 == pytest.approx(104 * 40 * 3 * 2048 * 512 * 4.125 / 8 / 1e9, abs=0.001)
    assert sum(p.values()) - p["routed_experts"] == pytest.approx(2.9e9, rel=0.01)
    assert w4a16.size_gb(QWEN36, 104) == pytest.approx(9.65, abs=0.05)
    assert w4a16.size_gb(QWEN36, 104) < 9.8 < w4a16.size_gb(QWEN36, 112)
    assert w4a16.max_experts(QWEN36, 9.8) == 104
    assert w4a16.max_experts(QWEN36, 10.0) == 104
    assert w4a16.max_experts(QWEN36, 9.5) == 96
    assert w4a16.max_experts(QWEN36, 100.0) == 256


def test_size_model_reads_every_config_layout():
    layers = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
    explicit = {k: v for k, v in QWEN36.items() if k != "full_attention_interval"} | {"layer_types": layers}
    multimodal = {"model_type": "qwen3_5_moe", "text_config": explicit}
    assert w4a16.size_gb(explicit, 104) == w4a16.size_gb(QWEN36, 104) == w4a16.size_gb(multimodal, 104)
    pruned = {k: v for k, v in QWEN36.items() if k != "num_experts"} | {"num_local_experts": 104}
    assert w4a16.n_experts(pruned) == 104


def test_bf16_attention_costs_its_bytes():
    fp8, bf16 = w4a16.part_bytes(QWEN36, 104), w4a16.part_bytes(QWEN36, 104, fp8=False)
    for part in ("attention", "deltanet", "lm_head"):
        assert bf16[part] / fp8[part] == pytest.approx(2.0, abs=0.01)
    assert bf16["routed_experts"] == fp8["routed_experts"] and bf16["embeddings"] == fp8["embeddings"]
    assert w4a16.max_experts(QWEN36, 9.8, fp8=False) < 104
    with pytest.raises(ValueError):
        w4a16.max_experts(QWEN36, 2.0)


# Gemma 4 26B-A4B's config.json as released: multimodal, text decoder in text_config.
GEMMA4_26B = {"model_type": "gemma4", "tie_word_embeddings": True, "vision_config": {"hidden_size": 1152},
              "text_config": dict(
    model_type="gemma4_text", hidden_size=2816, num_hidden_layers=30, num_attention_heads=16, num_key_value_heads=8,
    head_dim=256, global_head_dim=512, num_global_key_value_heads=2, attention_k_eq_v=True, enable_moe_block=True,
    num_experts=128, top_k_experts=8, moe_intermediate_size=704, intermediate_size=2112, vocab_size=262144,
    hidden_size_per_layer_input=0, tie_word_embeddings=True,
    layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 5)}


def test_gemma4_experts_take_group_64_and_64_fit_under_9_8_gb():
    # vLLM's int4 MoE kernels take whole groups of the expert width: 704 = 5.5 x 128
    assert w4a16.group_size(GEMMA4_26B) == 64 and w4a16.group_size(QWEN36) == 128
    p = w4a16.part_bytes(GEMMA4_26B, 72)
    # 72 x 30 x 3 x 2816 x 704 weights at 4.25 bits, tied embeddings in BF16 and no lm_head
    assert p["routed_experts"] / 1e9 == pytest.approx(72 * 30 * 3 * 2816 * 704 * 4.25 / 8 / 1e9, abs=0.001)
    assert p["lm_head"] == 0 and p["embeddings"] == 2 * 262144 * 2816
    # attention: 25 sliding layers (q 16x256, k and v 8x256, o) and 5 full ones (q 16x512, k 2x512 doubling as v)
    rows = 25 * (4096 + 2 * 2048 + 2816) + 5 * (8192 + 1024 + 2816)
    weights = 25 * (2 * 4096 + 2 * 2048) * 2816 + 5 * (2 * 8192 + 1024) * 2816
    assert p["attention"] == weights + 2 * rows  # FP8 bytes plus a BF16 scale per output row
    assert p["dense_mlp"] == 30 * (3 * 2112 * 2816 + 2 * (2 * 2112 + 2816))
    assert [round(w4a16.size_gb(GEMMA4_26B, k), 2) for k in (64, 72, 80, 128)] == [9.20, 9.96, 10.72, 15.28]
    assert w4a16.max_experts(GEMMA4_26B, 9.8) == 64 and w4a16.max_experts(GEMMA4_26B, 9.8, step=2) == 70
    assert w4a16.max_experts(GEMMA4_26B, 10.0) == 72 and w4a16.max_experts(GEMMA4_26B, 9.8, fp8=False) == 48
    # tied embeddings are read whole for every token, as the LM head
    assert w4a16.bytes_per_token_gb(GEMMA4_26B, 72) == pytest.approx(
        (sum(p.values()) - p["routed_experts"] * (1 - 8 / 72)) / 1e9)
    assert w4a16.targets(GEMMA4_26B) == (w4a16.GEMMA4_INT4_TARGETS, w4a16.GEMMA4_FP8_TARGETS)
    assert w4a16.targets(QWEN36, fp8=False) == (w4a16.INT4_TARGETS, [])


def test_gemma4_size_model_reads_every_config_layout():
    t = GEMMA4_26B["text_config"]
    # as transformers >= 5.15 saves it (text only): per_layer_config instead of global_*, no layer_types needed
    flat = ("global_head_dim", "num_global_key_value_heads", "layer_types")
    saved = {k: v for k, v in t.items() if k not in flat} | {
        "per_layer_config": {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 2} for i in range(5, 30, 6)}}
    assert w4a16.layer_types(saved) == t["layer_types"]
    assert w4a16.size_gb(saved, 72) == w4a16.size_gb(t, 72) == w4a16.size_gb(GEMMA4_26B, 72)
    assert w4a16.n_experts(GEMMA4_26B) == 128 and w4a16.top_k(GEMMA4_26B) == 8 and w4a16.top_k(QWEN36) == 8


def test_size_check_uses_the_saved_files(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 600_000)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 600_000)
    (tmp_path / "tokenizer.json").write_bytes(b"x" * 900_000)
    assert quantize_ct.safetensors_gb(tmp_path) == pytest.approx(0.0012)
    quantize_ct.check_size(0.0012, 0.002, "m")
    with pytest.raises(RuntimeError, match="over the 0.001 GB target"):
        quantize_ct.check_size(0.0012, 0.001, "m")


def test_oversized_estimate_fails_before_calibrating(tmp_path, monkeypatch):
    job = make_job(tmp_path, max_size_gb=9.5)
    (job / "work/healed/config.json").write_text(json.dumps(QWEN36))  # all 256 experts: ~19.5 GB
    monkeypatch.setattr(quantize_ct, "quantize", lambda *a: pytest.fail("quantized an oversized model"))
    with pytest.raises(RuntimeError, match="256 experts per layer is 19.53 GB, over the 9.5 GB target"):
        quantize.run_stage(Job(job))


def test_oversized_files_fail_after_saving(tmp_path, monkeypatch):
    job = make_job(tmp_path, max_size_gb=1.0)
    small = {**QWEN36, "num_hidden_layers": 4, "num_experts": 8, "vocab_size": 1024}  # ~0.19 GB estimated
    (job / "work/healed/config.json").write_text(json.dumps(small))

    def fake_quantize(_job, _healed, out):  # what the estimate misses shows up in the files
        out.mkdir()
        with open(out / "model.safetensors", "wb") as f:
            f.truncate(1_200_000_000)  # sparse: no disk used

    monkeypatch.setattr(quantize_ct, "quantize", fake_quantize)
    with pytest.raises(RuntimeError, match="is 1.20 GB, over the 1.0 GB target"):
        quantize.run_stage(Job(job))
    assert not (job / "work/allocation.json").exists() and not (job / ".done/quantize").exists()


class CharTok:
    """One id per character, no chat template: a row's prompt and answer, concatenated."""
    chat_template = None
    eos_token = ""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_calibration_set_takes_its_share_of_extra_rows(tmp_path):
    """calib_extra_share 0.5 of quant_calib_samples 4: two distinct extra rows (one per language,
    though each is in train three times) and two task rows."""
    pytest.importorskip("datasets")
    job = make_job(tmp_path, max_size_gb=1.0)
    cfg = json.loads((job / "config.json").read_text())
    (job / "config.json").write_text(json.dumps({**cfg, "calib_extra_share": 0.5}))
    extra = [{"messages": [{"role": "user", "content": f"aider {lang} {i}"}, {"role": "assistant", "content": "edit"}],
              "meta": {"source": "aider", "language": lang, "extra": True}}
             for lang in ("go", "rust") for i in range(3)]
    with open(job / "data/train.jsonl", "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in extra * 3))
    texts = ["".join(map(chr, ids)) for ids in quantize_ct.calibration_set(Job(job), CharTok())["input_ids"]]
    aider = sorted(t.split()[1] for t in texts if t.startswith("aider"))
    assert len(texts) == 4 and aider == ["go", "rust"] and sum(t.startswith("Answer in JSON") for t in texts) == 2


def test_unknown_quant_format_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quant_format": "nvfp4"}))
    with pytest.raises(ValueError, match="quant_format"):
        quantize.run_stage(Job(tmp_path))


# --- the real stage on a tiny model (needs torch, transformers 5 and llm-compressor) ---

TEMPLATE = (  # Qwen3's chat template, reduced: thinking in <think>, opened by the generation prompt
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' and m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{% else %}<think>\n{% endif %}{% endif %}")


def tiny_qwen36(out: Path):
    """Byte-level tokenizer and a random Qwen3.6-shaped text model saved as heal
    saves it: 3 DeltaNet layers and 1 full-attention layer, 8 experts plus a shared
    one, every quantized input dimension a multiple of 128 (int4 group size)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    tok.save_pretrained(out)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=384, hidden_size=256, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, moe_intermediate_size=128, shared_expert_intermediate_size=128, num_experts=8,
        num_experts_per_tok=2, linear_num_key_heads=2, linear_key_head_dim=64, linear_num_value_heads=4,
        linear_value_head_dim=64, max_position_embeddings=2048, tie_word_embeddings=False,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    torch.manual_seed(0)
    Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out)


def make_job(root: Path, max_size_gb: float) -> Path:
    job = root / "job"
    for d in ("data", "work/healed"):
        (job / d).mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "examples/support-tickets.taskspec.json").read_text())
    spec["target"]["max_size_gb"] = max_size_gb
    (job / "taskspec.json").write_text(json.dumps(spec))
    (job / "config.json").write_text(json.dumps({
        "quant_format": "w4a16", "quant_calib_samples": 4, "quant_calib_len": 160, "dense_fallback": False}))
    rows = [{"messages": [{"role": "system", "content": "Answer in JSON."},
                          {"role": "user", "content": f"Ticket {i}: my order #{1000 + i} is late."},
                          {"role": "assistant", "content": '{"category": "other"}',
                           "reasoning_content": f"Order {1000 + i} is late, so this is about delivery."}]}
            for i in range(6)]
    (job / "data/train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return job


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    """The quantize stage run as the pipeline runs it, with quant_format "w4a16"."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.5")
    pytest.importorskip("llmcompressor")
    job = make_job(tmp_path_factory.mktemp("w4a16"), max_size_gb=1.0)
    tiny_qwen36(job / "work/healed")
    p = subprocess.run([sys.executable, "-m", "stages.quantize", "--job", str(job)], cwd=ROOT,
                       env={**os.environ, "LOBBOT_DRY_RUN": ""}, capture_output=True, text=True)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    assert p.returncode == 0 and events[-1]["status"] == "done", p.stdout[-3000:] + p.stderr[-3000:]
    return job


def tensors(d: Path) -> dict:
    """{name: (dtype, shape)} from every safetensors header in d."""
    from safetensors import safe_open

    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            out.update({k: (fh.get_slice(k).get_dtype(), fh.get_slice(k).get_shape()) for k in fh.keys()})
    return out


def test_stage_registers_a_vllm_candidate(quantized):
    alloc = json.loads((quantized / "work/allocation.json").read_text())["candidates"]
    c = alloc["lobbot-moe"]
    out = Path(c["path"])
    assert out == quantized / "work/candidates/lobbot-moe" and (out / "config.json").exists()
    assert c["format"] == "compressed-tensors" and c["num_experts"] == 8
    assert c["size_gb"] == round(quantize_ct.safetensors_gb(out), 2)
    # The size model is exact: the files hold exactly the estimated tensor bytes, plus headers.
    sizes = {"F8_E4M3": 1, "BF16": 2, "I32": 4, "I64": 8}
    data = sum(sizes[t] * math.prod(s) for t, s in tensors(out).values())
    hf = json.loads((quantized / "work/healed/config.json").read_text())
    assert data == sum(w4a16.part_bytes(hf, 8).values())
    from transformers import AutoTokenizer
    assert AutoTokenizer.from_pretrained(out).chat_template == TEMPLATE


def test_quantization_config_has_int4_experts_and_fp8_attention(quantized):
    q = json.loads((quantized / "work/candidates/lobbot-moe/config.json").read_text())["quantization_config"]
    assert q["quant_method"] == "compressed-tensors" and q["format"] == "mixed-precision"
    groups = {tuple(g["targets"]): g for g in q["config_groups"].values()}
    int4, fp8 = groups[tuple(w4a16.INT4_TARGETS)], groups[tuple(w4a16.FP8_TARGETS)]
    w = int4["weights"]
    assert (w["type"], w["num_bits"], w["strategy"], w["group_size"], w["symmetric"]) == ("int", 4, "group", 128, True)
    assert int4["input_activations"] is None and int4["format"] == "pack-quantized"
    w, a = fp8["weights"], fp8["input_activations"]
    assert (w["type"], w["num_bits"], w["strategy"]) == ("float", 8, "channel")
    assert (a["type"], a["num_bits"], a["strategy"], a["dynamic"]) == ("float", 8, "token", True)
    # vLLM fuses in_proj_b + in_proj_a; both halves must be listed as unquantized.
    for i in range(3):
        for m in ("linear_attn.in_proj_b", "linear_attn.in_proj_a", "mlp.gate", "mlp.shared_expert_gate"):
            assert f"model.layers.{i}.{m}" in q["ignore"]


def test_every_module_gets_its_format(quantized):
    t = tensors(quantized / "work/candidates/lobbot-moe")
    for i in range(4):
        pre = f"model.layers.{i}."
        for e in [f"experts.{j}" for j in range(8)] + ["shared_expert"]:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert t[f"{pre}mlp.{e}.{proj}.weight_packed"][0] == "I32"
                assert f"{pre}mlp.{e}.{proj}.weight" not in t
        fp8 = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] if i == 3 else \
              ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
        for m in fp8:
            assert t[f"{pre}{m}.weight"][0] == "F8_E4M3" and t[f"{pre}{m}.weight_scale"][0] == "BF16"
        bf16 = ["mlp.gate", "mlp.shared_expert_gate", "input_layernorm", "post_attention_layernorm"]
        bf16 += ["self_attn.q_norm", "self_attn.k_norm"] if i == 3 else \
                ["linear_attn.in_proj_b", "linear_attn.in_proj_a", "linear_attn.conv1d", "linear_attn.norm"]
        for m in bf16:
            assert t[f"{pre}{m}.weight"][0] == "BF16"
    assert t["lm_head.weight"][0] == "F8_E4M3"
    assert t["model.embed_tokens.weight"][0] == "BF16" and t["model.norm.weight"][0] == "BF16"


def test_weights_stay_the_healed_ones(quantized):
    """BF16 tensors are copied bit for bit; quantized ones dequantize close to the originals."""
    import torch
    from compressed_tensors.compressors import unpack_from_int32
    from safetensors.torch import load_file

    healed = load_file(next((quantized / "work/healed").glob("*.safetensors")))
    q = load_file(next((quantized / "work/candidates/lobbot-moe").glob("*.safetensors")))
    for k in ("model.embed_tokens.weight", "model.layers.0.mlp.gate.weight", "model.layers.0.linear_attn.A_log",
              "model.layers.0.linear_attn.conv1d.weight", "model.layers.3.input_layernorm.weight"):
        assert torch.equal(q[k], healed[k]), k

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    for k in ("model.layers.3.self_attn.q_proj", "model.layers.1.linear_attn.in_proj_qkv", "lm_head"):
        assert rel(q[k + ".weight"].float() * q[k + ".weight_scale"].float(), healed[k + ".weight"]) < 0.05, k
    # int4 by GPTQ on random weights: ~0.15-0.18 here; a wrong layout or scale gives ~1
    for k in ("model.layers.2.mlp.experts.5.down_proj", "model.layers.0.mlp.shared_expert.up_proj"):
        shape = q[k + ".weight_shape"].tolist()
        w = unpack_from_int32(q[k + ".weight_packed"], 4, torch.Size(shape)).float()
        assert rel(w * q[k + ".weight_scale"].float().repeat_interleave(128, dim=1), healed[k + ".weight"]) < 0.3, k


def test_calibration_rows_include_the_thinking(quantized):
    from transformers import AutoTokenizer

    job = Job(quantized)
    tok = AutoTokenizer.from_pretrained(str(quantized / "work/healed"))
    data = quantize_ct.calibration_set(job, tok)
    assert len(data) == 4 and all(len(ids) <= 160 for ids in data["input_ids"])
    text = tok.decode(data["input_ids"][0])
    assert "<|im_start|>assistant\n<think>\nOrder 10" in text and "</think>" in text


# --- the real stage on a tiny Gemma 4 MoE ---

GEMMA_TYPES = ["sliding_attention", "full_attention"]  # 26B-A4B repeats 5 sliding-window layers and 1 full


def tiny_gemma4(out: Path, layout: str):
    """Gemma 4's tokenizer specials and template (tests/test_gemma4_moe.py) and a
    random Gemma 4 MoE: a sliding-window layer and a full-attention one with wider
    heads and K reused as V, each with a dense MLP beside 4 fused experts, top-2,
    tied embeddings. The expert width 192 is no multiple of 128 (26B-A4B's 704 is
    not either), so the experts take int4 group 64. layout "text" is how REAP and
    heal save it (Gemma4ForCausalLM); "multimodal" is how it ships, with a vision
    tower, whose config also names its dtype, so llm-compressor rewrites the saved
    config.json from this one."""
    import torch
    from test_gemma4_moe import build_tokenizer
    from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration

    tok = build_tokenizer(out)
    text = dict(vocab_size=384, hidden_size=256, num_hidden_layers=2, layer_types=GEMMA_TYPES,
                num_attention_heads=2, num_key_value_heads=1, head_dim=128, global_head_dim=256,
                num_global_key_value_heads=1, attention_k_eq_v=True, intermediate_size=128, enable_moe_block=True,
                num_experts=4, top_k_experts=2, moe_intermediate_size=192, hidden_size_per_layer_input=0,
                sliding_window=64, tie_word_embeddings=True, pad_token_id=tok.pad_token_id,
                bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16, position_embedding_size=64)
    cfg = Gemma4Config(text_config=text, vision_config=vision)
    torch.manual_seed(0)
    if layout == "text":
        Gemma4ForCausalLM(cfg.text_config).to(torch.bfloat16).save_pretrained(out)
        return
    Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(out)
    p = out / "config.json"
    hf = json.loads(p.read_text())
    hf["text_config"]["dtype"] = "bfloat16"
    p.write_text(json.dumps(hf))


@pytest.fixture(scope="module", params=["text", "multimodal"])
def quantized_gemma(request, tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.15")
    pytest.importorskip("llmcompressor")
    job = make_job(tmp_path_factory.mktemp(f"w4a16-gemma-{request.param}"), max_size_gb=1.0)
    tiny_gemma4(job / "work/healed", request.param)
    p = subprocess.run([sys.executable, "-m", "stages.quantize", "--job", str(job)], cwd=ROOT,
                       env={**os.environ, "LOBBOT_DRY_RUN": ""}, capture_output=True, text=True)
    events = [e for e in map(parse, p.stdout.splitlines()) if e]
    assert p.returncode == 0 and events[-1]["status"] == "done", p.stdout[-3000:] + p.stderr[-3000:]
    return job


def test_gemma4_stage_saves_the_text_decoder(quantized_gemma):
    from test_gemma4_moe import TEMPLATE
    from transformers import AutoConfig, AutoTokenizer

    c = json.loads((quantized_gemma / "work/allocation.json").read_text())["candidates"]["lobbot-moe"]
    out = Path(c["path"])
    hf = json.loads((quantized_gemma / "work/healed/config.json").read_text())
    assert c["num_experts"] == 4 and w4a16.is_gemma4(hf)
    t = tensors(out)
    sizes = {"F8_E4M3": 1, "BF16": 2, "I32": 4, "I64": 8}
    assert sum(sizes[d] * math.prod(s) for d, s in t.values()) == sum(w4a16.part_bytes(hf, 4).values())
    # Gemma4ForCausalLM's own names: no towers, no multimodal prefix, no lm_head (tied)
    assert not [k for k in t if not k.startswith("model.") or "language_model" in k or "vision" in k]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "gemma4_text" and cfg["architectures"] == ["Gemma4ForCausalLM"]
    assert cfg["quantization_config"]["quant_method"] == "compressed-tensors" and "vision_config" not in cfg
    loaded = AutoConfig.from_pretrained(out)
    assert loaded.model_type == "gemma4_text" and loaded.per_layer_config[1].head_dim == 256
    assert loaded.num_experts == 4 and loaded.top_k_experts == 2
    assert AutoTokenizer.from_pretrained(out).chat_template == TEMPLATE


def test_gemma4_quantization_config(quantized_gemma):
    q = json.loads((quantized_gemma / "work/candidates/lobbot-moe/config.json").read_text())["quantization_config"]
    assert q["format"] == "mixed-precision"
    groups = {tuple(g["targets"]): g for g in q["config_groups"].values()}
    int4, fp8 = groups[tuple(w4a16.GEMMA4_INT4_TARGETS)], groups[tuple(w4a16.GEMMA4_FP8_TARGETS)]
    w = int4["weights"]
    assert (w["type"], w["num_bits"], w["strategy"], w["group_size"], w["symmetric"]) == ("int", 4, "group", 64, True)
    assert int4["format"] == "pack-quantized" and int4["input_activations"] is None
    w, a = fp8["weights"], fp8["input_activations"]
    assert (w["type"], w["num_bits"], w["strategy"]) == ("float", 8, "channel")
    assert (a["type"], a["strategy"], a["dynamic"]) == ("float", "token", True)
    for i in range(2):
        assert f"model.layers.{i}.router.proj" in q["ignore"]
    assert not [m for m in q["ignore"] if "language_model" in m or "vision" in m or "_proj" in m and "router" not in m]


def test_gemma4_every_module_gets_its_format(quantized_gemma):
    t = tensors(quantized_gemma / "work/candidates/lobbot-moe")
    for i, kind in enumerate(GEMMA_TYPES):
        pre, full = f"model.layers.{i}.", kind == "full_attention"
        for j in range(4):
            for proj, (rows, cols) in (("gate_proj", (192, 256)), ("up_proj", (192, 256)), ("down_proj", (256, 192))):
                assert t[f"{pre}experts.{j}.{proj}.weight_packed"] == ("I32", [rows, cols // 8])
                assert t[f"{pre}experts.{j}.{proj}.weight_scale"] == ("BF16", [rows, cols // 64])
        fp8 = [f"self_attn.{p}_proj" for p in ("q", "k", "o") + (() if full else ("v",))]
        fp8 += [f"mlp.{p}_proj" for p in ("gate", "up", "down")]
        for m in fp8:
            assert t[f"{pre}{m}.weight"][0] == "F8_E4M3" and t[f"{pre}{m}.weight_scale"][0] == "BF16"
        assert (f"{pre}self_attn.v_proj.weight" in t) == (not full)  # the full layer reuses K as V
        assert t[f"{pre}self_attn.q_proj.weight"][1] == [2 * (256 if full else 128), 256]
        bf16 = ["router.proj.weight", "router.scale", "router.per_expert_scale", "layer_scalar",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight"]
        bf16 += [f"{n}.weight" for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                         "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                                         "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2")]
        for m in bf16:
            assert t[pre + m][0] == "BF16", pre + m
    assert t["model.embed_tokens.weight"][0] == "BF16" and t["model.norm.weight"][0] == "BF16"


def test_gemma4_weights_stay_the_healed_ones(quantized_gemma):
    import torch
    from compressed_tensors.compressors import unpack_from_int32
    from safetensors.torch import load_file

    healed = load_file(next((quantized_gemma / "work/healed").glob("*.safetensors")))
    healed = {k.replace("model.language_model.", "model."): v for k, v in healed.items()}
    q = load_file(next((quantized_gemma / "work/candidates/lobbot-moe").glob("*.safetensors")))
    for k in ("model.embed_tokens.weight", "model.layers.0.router.proj.weight", "model.layers.0.router.scale",
              "model.layers.1.self_attn.k_norm.weight", "model.layers.1.post_feedforward_layernorm_2.weight"):
        assert torch.equal(q[k], healed[k]), k

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    for k in ("model.layers.1.self_attn.k_proj", "model.layers.0.self_attn.v_proj", "model.layers.1.mlp.down_proj"):
        assert rel(q[k + ".weight"].float() * q[k + ".weight_scale"].float(), healed[k + ".weight"]) < 0.05, k
    gate_up, down = healed["model.layers.1.experts.gate_up_proj"], healed["model.layers.1.experts.down_proj"]
    for k, ref in (("model.layers.1.experts.2.down_proj", down[2]),  # gate rows first in gate_up_proj
                   ("model.layers.1.experts.3.gate_proj", gate_up[3, :192]),
                   ("model.layers.1.experts.3.up_proj", gate_up[3, 192:])):
        w = unpack_from_int32(q[k + ".weight_packed"], 4, torch.Size(q[k + ".weight_shape"].tolist())).float()
        assert rel(w * q[k + ".weight_scale"].float().repeat_interleave(64, dim=1), ref) < 0.3, k


def test_gemma4_calibration_rows_think_in_the_thought_channel(quantized_gemma):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(quantized_gemma / "work/healed"))
    text = tok.decode(quantize_ct.calibration_set(Job(quantized_gemma), tok)["input_ids"][0])
    assert "<|think|>" in text and "<|turn>model\n<|channel>thought\nOrder 10" in text and "<channel|>" in text
