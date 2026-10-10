"""The vLLM 4-bit scheme (Config.quant_format "w4a16") for the MoE bases: which
modules get which format, and what a checkpoint with K routed experts weighs.

vLLM's fused MoE kernels go no lower than 4 bits, so the experts are int4 and
the dense layers next to them FP8:
  int4  W4A16, symmetric, GPTQ (Marlin kernels), one BF16 scale per group of
        128 inputs, or 64 where an expert's width is not a multiple of 128
        (vLLM's MoE kernels take whole groups only)
  fp8   FP8 weights per output channel, activations per token at run time
        (FP8_DYNAMIC); BF16 with fp8=False (Config.quant_fp8_attention off)
  bf16  everything else

Qwen3.5/3.6 MoE (Qwen3.6-35B-A3B):
  int4  routed and shared experts, group 128
  fp8   full-attention q/k/v/o, DeltaNet in_proj_qkv, in_proj_z and out_proj, lm_head
  bf16  embeddings, routers, shared_expert_gate, DeltaNet in_proj_b, in_proj_a,
        conv1d, A_log and dt_bias, all norms
Gemma 4 MoE (gemma-4-26B-A4B), the text decoder only: the quantize stage drops
the vision and audio towers, which a coding model never runs.
  int4  routed experts, group 64 (moe_intermediate_size 704 = 5.5 x 128)
  fp8   q/k/v/o of every layer (sliding and full attention), and the dense MLP
        each layer runs next to its experts
  bf16  embeddings (tied: also the LM head), router proj, scale and
        per_expert_scale, norms, layer_scalar

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
# Gemma 4 keeps its experts beside the MLP (layers.N.experts.M after unfusing; vLLM
# names the fused layer layers.N.experts). The dense MLP's names end in mlp.*_proj,
# which no expert's do; vLLM fuses its gate and up into gate_up_proj, q/k/v into
# qkv_proj (k twice on the full-attention layers, which ship no v_proj).
GEMMA4_INT4_TARGETS = [r"re:.*experts\.\d+\.(gate|up|down)_proj$"]
GEMMA4_FP8_TARGETS = [r"re:.*self_attn\.(q|k|v|o)_proj$", r"re:.*mlp\.(gate|up|down)_proj$"]


def _int4(rows: int, cols: int, group: int = GROUP) -> int:
    """Packed int4 weight, a BF16 scale per group (symmetric: no zero point) and
    the int64 weight_shape compressed-tensors stores next to them."""
    return rows * cols // 2 + rows * -(-cols // group) * 2 + 16


def _fp8(rows: int, cols: int) -> int:
    return rows * cols + rows * 2  # one BF16 scale per output channel


def _bf16(rows: int, cols: int = 1) -> int:
    return 2 * rows * cols


def text_config(cfg: dict) -> dict:
    """The text decoder's config, also for the multimodal Qwen3.6 and Gemma 4 checkpoints."""
    return cfg.get("text_config") or cfg


def is_gemma4(cfg: dict) -> bool:
    t = text_config(cfg)
    return any(str(c.get("model_type", "")).startswith("gemma4") for c in (cfg, t)) or "enable_moe_block" in t


def layer_types(t: dict) -> list[str]:
    n = t["num_hidden_layers"]
    if t.get("layer_types"):
        return list(t["layer_types"])
    if is_gemma4(t):  # transformers' default: 5 sliding-window layers, then 1 full; the last one full
        return ["full_attention" if (i + 1) % 6 == 0 or i == n - 1 else "sliding_attention" for i in range(n)]
    every = t.get("full_attention_interval", 4)
    return ["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)]


def n_experts(cfg: dict) -> int:
    t = text_config(cfg)
    return t.get("num_experts") or t["num_local_experts"]


def top_k(cfg: dict) -> int:
    t = text_config(cfg)
    return t.get("top_k_experts") or t["num_experts_per_tok"]


def group_size(cfg: dict) -> int:
    """The int4 group: 128, or the largest of 64 and 32 that divides every
    expert's input width (vLLM refuses groups that cross the intermediate size)."""
    t = text_config(cfg)
    widths = [t["hidden_size"], t["moe_intermediate_size"], t.get("shared_expert_intermediate_size") or GROUP]
    for g in (GROUP, 64, 32):
        if all(w % g == 0 for w in widths):
            return g
    raise ValueError(f"no int4 group size of 128, 64 or 32 divides the expert widths {widths}")


def targets(cfg: dict, fp8: bool = True) -> tuple[list[str], list[str]]:
    """(int4 targets, FP8 targets) for this model; no FP8 targets with fp8=False."""
    int4, f8 = (GEMMA4_INT4_TARGETS, GEMMA4_FP8_TARGETS) if is_gemma4(cfg) else (INT4_TARGETS, FP8_TARGETS)
    return list(int4), list(f8) if fp8 else []


def part_bytes(cfg: dict, k: int, fp8: bool = True) -> dict[str, int]:
    """Bytes of each part of the quantized checkpoint with k routed experts per
    layer. cfg is an HF config.json (Qwen3.5/3.6 MoE or Gemma 4 MoE, multimodal
    or text only); fp8=False is Config.quant_fp8_attention off (those layers BF16)."""
    if is_gemma4(cfg):
        return _gemma4_part_bytes(cfg, k, fp8)
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


def _gemma4_heads(t: dict, i: int, kind: str) -> tuple[int, int]:
    """(head_dim, KV heads) of layer i. Full-attention layers have wider heads and,
    with attention_k_eq_v, their own KV head count: per_layer_config in configs
    transformers >= 5.15 saved, global_* fields in the original release."""
    hd, nkv = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"], t["num_key_value_heads"]
    per_layer = t.get("per_layer_config")
    if per_layer is not None:
        o = next((v for key, v in per_layer.items() if int(key) == i), {})
        return o.get("head_dim", hd), o.get("num_key_value_heads", nkv)
    if kind == "full_attention":
        hd = t.get("global_head_dim") or 512  # transformers' default
        if t.get("attention_k_eq_v"):
            nkv = t.get("num_global_key_value_heads") or nkv
    return hd, nkv


def _gemma4_part_bytes(cfg: dict, k: int, fp8: bool) -> dict[str, int]:
    """Gemma4ForCausalLM's tensors (Gemma4TextModel, transformers 5.17)."""
    t = text_config(cfg)
    h, nh, vocab, mi = t["hidden_size"], t["num_attention_heads"], t["vocab_size"], t["moe_intermediate_size"]
    wide, g = (_fp8 if fp8 else _bf16), group_size(cfg)
    types = layer_types(t)
    n = len(types)
    first_shared = n - (t.get("num_kv_shared_layers") or 0)  # later layers reuse earlier K/V
    ple = t.get("hidden_size_per_layer_input", 256) or 0  # per-layer token embeddings (E2B/E4B); 0 on 26B
    attn = mlp = other = 0
    for i, kind in enumerate(types):
        hd, nkv = _gemma4_heads(t, i, kind)
        attn += wide(nh * hd, h) + wide(h, nh * hd)  # q_proj, o_proj
        other += _bf16(hd)  # q_norm
        if i < first_shared:
            # k_proj and v_proj; with attention_k_eq_v full layers reuse K as V and ship no v_proj
            kv = 1 if t.get("attention_k_eq_v") and kind != "sliding_attention" else 2
            attn += kv * wide(nkv * hd, h)
            other += _bf16(hd)  # k_norm (v_norm has no weight)
        inter = t["intermediate_size"] * (2 if t.get("use_double_wide_mlp") and i >= first_shared > 0 else 1)
        mlp += 2 * wide(inter, h) + wide(h, inter)
        # four norms around attention and MLP, three around the experts, layer_scalar;
        # the router's proj, scale and per_expert_scale (its norm has no weight)
        other += 7 * _bf16(h) + _bf16(1) + _bf16(k, h) + _bf16(h) + _bf16(k)
        if ple:  # per_layer_input_gate, per_layer_projection, post_per_layer_input_norm
            other += 2 * _bf16(ple, h) + _bf16(h)
    other += _bf16(h)  # final norm
    if ple:  # per_layer_model_projection and per_layer_projection_norm
        other += _bf16(n * ple, h) + _bf16(ple)
    tied = t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))
    return {
        "routed_experts": n * k * (2 * _int4(mi, h, g) + _int4(h, mi, g)),
        "attention": attn,
        "dense_mlp": mlp,
        "lm_head": 0 if tied else wide(vocab, h),
        "embeddings": _bf16(vocab, h),
        "per_layer_embeddings": _bf16(t.get("vocab_size_per_layer_input", 262_144), n * ple) if ple else 0,
        "bf16_other": other,
    }


def size_gb(cfg: dict, k: int, fp8: bool = True) -> float:
    return sum(part_bytes(cfg, k, fp8).values()) / 1e9


def bytes_per_token_gb(cfg: dict, k: int, fp8: bool = True) -> float:
    """Weights read per generated token: everything but the embedding lookups, with
    the routed experts at the active share (top-k of k). Tied embeddings are read
    in full, as the LM head."""
    p = part_bytes(cfg, k, fp8)
    active = min(k, top_k(cfg)) / k
    lookups = p.get("per_layer_embeddings", 0) + (p["embeddings"] if p["lm_head"] else 0)
    return (sum(p.values()) - lookups - p["routed_experts"] * (1 - active)) / 1e9


def max_experts(cfg: dict, budget_gb: float, fp8: bool = True, step: int = 8) -> int:
    """The largest k, a multiple of step and at most the config's expert count,
    whose checkpoint fits budget_gb (decimal GB)."""
    k = n_experts(cfg) // step * step
    while k >= step and size_gb(cfg, k, fp8) > budget_gb:
        k -= step
    if k < max(step, top_k(cfg)):
        raise ValueError(f"even {step} experts per layer ({size_gb(cfg, step, fp8):.2f} GB) do not fit {budget_gb} GB")
    return k
