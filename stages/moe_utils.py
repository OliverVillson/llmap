"""Version-tolerant access to the sparse MoE blocks of Qwen3-MoE style models
(Qwen3MoeSparseMoeBlock / Qwen2MoeSparseMoeBlock / Mixtral / Qwen3.5 and
Qwen3.6 Qwen3_5MoeSparseMoeBlock), covering both the transformers 4.x layout
(experts = ModuleList of MLPs, gate = nn.Linear) and the fused 5.x layout
(experts = one module holding [E, ...] stacked weights, gate = a router module
with a [E, H] weight).

Shared experts (Qwen2-MoE, Qwen3.5/3.6: `shared_expert` plus a sigmoid
`shared_expert_gate`) run on every token and are never scored or pruned; only
the routed experts behind `gate` are.

Qwen3.6-35B-A3B ships as a multimodal checkpoint (Qwen3_5MoeForConditionalGeneration).
AutoModelForCausalLM loads only its text decoder (Qwen3_5MoeForCausalLM; the
vision tower and the MTP head are skipped), which is what the pipeline wants.
fix_saved_config makes such a text-only save consistent."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def decoder_layers(model):
    """The text decoder layers, also for multimodal checkpoints such as Gemma 4
    (Gemma4ForConditionalGeneration keeps them under model.language_model)."""
    inner = getattr(model, "model", model)
    for owner in (inner, getattr(inner, "language_model", None), getattr(model, "language_model", None)):
        layers = getattr(owner, "layers", None) if owner is not None else None
        if layers is not None:
            return layers
    raise AttributeError(f"cannot find decoder layers in {type(model).__name__}")


def find_moe_blocks(model) -> list[tuple[int, nn.Module]]:
    """(decoder layer index, moe block) for every sparse layer."""
    layers = decoder_layers(model)
    out = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if mlp is not None and hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            out.append((i, mlp))
    return out


def gate_weight(block) -> torch.Tensor:
    return block.gate.weight  # [E, H] for both nn.Linear and router modules


def num_experts(block) -> int:
    return gate_weight(block).shape[0]


def top_k(block, config) -> int:
    for obj, name in ((block, "top_k"), (block.gate, "top_k"), (config, "num_experts_per_tok")):
        v = getattr(obj, name, None)
        if isinstance(v, int):
            return v
    raise AttributeError("cannot find top_k")


def route(block, config, x: torch.Tensor):
    """The routing the block itself uses. x: [N, H]. Returns
    (weights [N, k], indices [N, k]). Router modules (5.x) are called as is,
    since their renormalisation differs by model (Qwen3.5/3.6 always
    renormalises the top-k, Qwen3-MoE only with norm_topk_prob); a plain
    nn.Linear gate (4.x) gets the HF softmax, top-k and optional renormalisation."""
    gate = block.gate
    if not isinstance(gate, nn.Linear):
        out = gate(x)
        if isinstance(out, tuple) and len(out) == 3:  # (logits, top-k weights, top-k indices)
            return out[1], out[2]
    logits = F.linear(x, gate_weight(block))
    probs = F.softmax(logits, dim=-1, dtype=torch.float)
    w, idx = torch.topk(probs, top_k(block, config), dim=-1)
    if getattr(config, "norm_topk_prob", False):
        w = w / w.sum(dim=-1, keepdim=True)
    return w, idx


def _act(experts, config):
    fn = getattr(experts, "act_fn", None)
    if fn is not None:
        return fn
    from transformers.activations import ACT2FN
    return ACT2FN[config.hidden_act]


def expert_forward(block, config, j: int, x: torch.Tensor) -> torch.Tensor:
    """Output of expert j on tokens x [n, H] -> [n, H] (without gate weight)."""
    experts = block.experts
    if isinstance(experts, nn.ModuleList):
        return experts[j](x)
    act = _act(experts, config)
    if hasattr(experts, "gate_up_proj"):
        gu = experts.gate_up_proj[j]  # [2I, H] (5.x) or [H, 2I] (some variants)
        dn = experts.down_proj[j]
        H = x.shape[-1]
        h = F.linear(x, gu) if gu.shape[-1] == H else x @ gu
        g, u = h.chunk(2, dim=-1)
        h = act(g) * u
        return F.linear(h, dn) if dn.shape[0] == H else h @ dn
    raise NotImplementedError(f"unknown experts layout: {type(experts).__name__}")


@torch.no_grad()
def prune_block(block, keep: list[int]) -> None:
    """Keep only the experts in `keep` (original ids, ascending), in place."""
    keep_t = torch.tensor(keep, dtype=torch.long, device=gate_weight(block).device)
    E = num_experts(block)
    n = len(keep)

    # router
    gate = block.gate
    if isinstance(gate, nn.Linear):
        new = nn.Linear(gate.in_features, n, bias=gate.bias is not None,
                        device=gate.weight.device, dtype=gate.weight.dtype)
        new.weight.copy_(gate.weight[keep_t])
        if gate.bias is not None:
            new.bias.copy_(gate.bias[keep_t])
        block.gate = new
    else:
        for name, p in list(gate.named_parameters(recurse=False)):
            if p.shape[0] == E:
                setattr(gate, name, nn.Parameter(p.data[keep_t].clone(), requires_grad=p.requires_grad))
        for name, b in list(gate.named_buffers(recurse=False)):
            if b.shape and b.shape[0] == E:
                setattr(gate, name, b[keep_t].clone())
    for obj in (gate, block):
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            if isinstance(getattr(obj, attr, None), int):
                setattr(obj, attr, n)

    # experts
    experts = block.experts
    if isinstance(experts, nn.ModuleList):
        block.experts = nn.ModuleList([experts[j] for j in keep])
    else:
        for name, p in list(experts.named_parameters(recurse=False)):
            if p.shape[0] == E:
                setattr(experts, name, nn.Parameter(p.data[keep_t.to(p.device)].clone(),
                                                    requires_grad=p.requires_grad))
        for attr in ("num_experts", "num_local_experts"):
            if isinstance(getattr(experts, attr, None), int):
                setattr(experts, attr, n)


def set_config_experts(config, n: int) -> None:
    for c in (config, getattr(config, "text_config", None)):
        for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
            if c is not None and getattr(c, attr, None) is not None:
                setattr(c, attr, n)


def fix_saved_config(model_dir) -> None:
    """Make a saved config.json match what the pipeline wrote and what the
    quantize stage and llama.cpp read:
    - transformers 5 may save the expert count as num_local_experts; the Qwen
      checkpoints and our quantize stage use num_experts. Write both.
    - Qwen3.5/3.6 configs announce an MTP head (mtp_num_hidden_layers) that a
      text-only load drops; set it to 0 when no mtp.* tensors were saved, so
      converters do not look for them."""
    import json
    from pathlib import Path
    d = Path(model_dir)
    p = d / "config.json"
    cfg = json.loads(p.read_text())
    for c in (cfg, cfg.get("text_config")):
        if not isinstance(c, dict):
            continue
        n = c.get("num_experts") or c.get("num_local_experts")
        if n is not None:
            c["num_experts"] = n
            c["num_local_experts"] = n
        if c.get("mtp_num_hidden_layers") and not _has_tensor(d, "mtp."):
            c["mtp_num_hidden_layers"] = 0
    p.write_text(json.dumps(cfg, indent=2) + "\n")


def gguf_convert_flags(model_dir) -> list[str]:
    """Extra convert_hf_to_gguf.py flags a saved checkpoint needs. llama.cpp
    expects an MTP head on Qwen3.5/3.6/Next models unless told --no-mtp, and a
    text-only load does not keep it."""
    import json
    from pathlib import Path
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    mtype = cfg.get("model_type", "")
    if mtype.startswith(("qwen3_5", "qwen3_next")) and not _has_tensor(model_dir, "mtp."):
        return ["--no-mtp"]
    return []


def _has_tensor(model_dir, prefix: str) -> bool:
    import json
    from pathlib import Path
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        names = json.loads(idx.read_text())["weight_map"]
    else:
        from safetensors import safe_open
        names = []
        for f in d.glob("*.safetensors"):
            with safe_open(f, "pt") as fh:
                names += list(fh.keys())
    return any(k.startswith(prefix) or f".{prefix}" in k for k in names)
