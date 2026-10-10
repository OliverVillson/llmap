"""The vLLM 4-bit scheme for Qwen3.5/3.6 MoE (Config.quant_format "w4a16"): which
modules get which format, and what a checkpoint with K routed experts weighs.

vLLM's fused MoE kernels go no lower than 4 bits, so:
  int4  routed and shared experts: W4A16, symmetric, group 128, GPTQ (Marlin kernels)
  fp8   full-attention q/k/v/o, DeltaNet in_proj_qkv, in_proj_z and out_proj, lm_head:
        FP8 weights per output channel, activations per token at run time (FP8_DYNAMIC)
  bf16  everything else: embeddings, routers, shared_expert_gate, DeltaNet in_proj_b,
        in_proj_a, conv1d, A_log and dt_bias, all norms

Pure Python: a runner picks the expert count for a size budget before any GPU work
(max_experts), and turns it into reap_sparsity = 1 - K / n_experts.
"""

from __future__ import annotations

GROUP = 128  # int4 weights share one BF16 scale per 128 inputs (4.125 bits per weight)

# llm-compressor targets, matched against module names after its MoE calibration
# replacement has unfused the experts into one Linear per projection. vLLM fuses
# in_proj_qkv + in_proj_z into in_proj_qkvz (Qwen3-Next ships it fused), so both
# halves get the same scheme; in_proj_b + in_proj_a likewise both stay BF16.
INT4_TARGETS = [
    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
    r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
]
FP8_TARGETS = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj)$",
    "lm_head",
]


def _int4(rows: int, cols: int) -> int:
    """Packed int4 weight, a BF16 scale per group (symmetric: no zero point) and
    the int64 weight_shape compressed-tensors stores next to them."""
    return rows * cols // 2 + rows * -(-cols // GROUP) * 2 + 16


def _fp8(rows: int, cols: int) -> int:
    return rows * cols + rows * 2  # one BF16 scale per output channel


def _bf16(rows: int, cols: int = 1) -> int:
    return 2 * rows * cols


def text_config(cfg: dict) -> dict:
    """The text decoder's config, also for the multimodal Qwen3.6 checkpoint."""
    return cfg.get("text_config") or cfg


def layer_types(t: dict) -> list[str]:
    n = t["num_hidden_layers"]
    if t.get("layer_types"):
        return list(t["layer_types"])
    every = t.get("full_attention_interval", 4)
    return ["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)]


def part_bytes(cfg: dict, k: int, fp8: bool = True) -> dict[str, int]:
    """Bytes of each part of the quantized checkpoint with k routed experts per
    layer. cfg is an HF config.json (Qwen3.5/3.6 MoE, multimodal or text only);
    fp8=False is Config.quant_fp8_attention off (those layers BF16)."""
    t = text_config(cfg)
    h, inter, vocab = t["hidden_size"], t["moe_intermediate_size"], t["vocab_size"]
    shared = t.get("shared_expert_intermediate_size") or 0
    types = layer_types(t)
    n_full, n_lin = types.count("full_attention"), types.count("linear_attention")
    n_layers = len(types)
    wide = _fp8 if fp8 else _bf16

    nh, nkv = t["num_attention_heads"], t["num_key_value_heads"]
    hd = t.get("head_dim") or h // nh
    q_out = nh * hd * (2 if t.get("attn_output_gate", True) else 1)  # q plus its output gate
    attn = wide(q_out, h) + 2 * wide(nkv * hd, h) + wide(h, nh * hd)
    key = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    nv, vd = t["linear_num_value_heads"], t["linear_value_head_dim"]
    value = nv * vd
    conv = 2 * key + value
    delta = wide(conv, h) + wide(value, h) + wide(h, value)  # in_proj_qkv, in_proj_z, out_proj
    # in_proj_b and in_proj_a, conv1d, A_log and dt_bias, the gated norm
    delta_bf16 = _bf16(2 * nv, h) + _bf16(conv, t.get("linear_conv_kernel_dim", 4)) + _bf16(2 * nv) + _bf16(vd)

    expert = 2 * _int4(inter, h) + _int4(h, inter)
    return {
        "routed_experts": n_layers * k * expert,
        "shared_expert": n_layers * (2 * _int4(shared, h) + _int4(h, shared)) if shared else 0,
        "attention": n_full * attn,
        "deltanet": n_lin * delta,
        "lm_head": 0 if t.get("tie_word_embeddings", cfg.get("tie_word_embeddings")) else wide(vocab, h),
        "embeddings": _bf16(vocab, h),
        # per layer the router, two norms and shared_expert_gate; DeltaNet's small tensors;
        # q_norm and k_norm; the final norm
        "bf16_other": n_layers * (_bf16(k, h) + _bf16(h) * (3 if shared else 2)) + n_lin * delta_bf16
                      + n_full * 2 * _bf16(hd) + _bf16(h),
    }


def size_gb(cfg: dict, k: int, fp8: bool = True) -> float:
    return sum(part_bytes(cfg, k, fp8).values()) / 1e9


def bytes_per_token_gb(cfg: dict, k: int, fp8: bool = True) -> float:
    """Weights read per generated token: everything but the embeddings, with the
    routed experts at the active share (num_experts_per_tok of k)."""
    p = part_bytes(cfg, k, fp8)
    active = min(k, text_config(cfg)["num_experts_per_tok"]) / k
    return (sum(p.values()) - p["embeddings"] - p["routed_experts"] * (1 - active)) / 1e9


def n_experts(cfg: dict) -> int:
    t = text_config(cfg)
    return t.get("num_experts") or t["num_local_experts"]


def max_experts(cfg: dict, budget_gb: float, fp8: bool = True, step: int = 8) -> int:
    """The largest k, a multiple of step and at most the config's expert count,
    whose checkpoint fits budget_gb (decimal GB)."""
    k = n_experts(cfg) // step * step
    while k >= step and size_gb(cfg, k, fp8) > budget_gb:
        k -= step
    if k < max(step, text_config(cfg)["num_experts_per_tok"]):
        raise ValueError(f"even {step} experts per layer ({size_gb(cfg, step, fp8):.2f} GB) do not fit {budget_gb} GB")
    return k
