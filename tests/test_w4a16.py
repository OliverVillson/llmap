"""The vLLM 4-bit path: the size model (stages/w4a16.py, no torch) and the real
quantize stage (stages/quantize_ct.py) on a tiny random Qwen3.6-shaped model."""
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
