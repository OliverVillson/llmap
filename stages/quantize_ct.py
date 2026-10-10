"""Stage 4, vLLM path (Config.quant_format "w4a16"): the healed model as a
compressed-tensors checkpoint that vLLM serves on a GPU, which llama.cpp cannot
keep busy. llm-compressor quantizes it in one pass; the scheme (stages/w4a16.py):
  routed (and shared) experts W4A16 int4, group 128 (Gemma 4: 64), by GPTQ
                              (Marlin MoE kernels)
  Qwen3.6: attention, DeltaNet's in_proj_qkv, in_proj_z and out_proj, lm_head
  Gemma 4: attention, the dense MLP beside the experts
                              FP8_DYNAMIC, round to nearest (BF16 with
                              quant_fp8_attention off)
  everything else             BF16
Gemma 4 is quantized and saved as its text decoder alone (Gemma4ForCausalLM, as
heal saves it; a released multimodal checkpoint is cut to it), so its vision and
audio towers do not count against the size.
GPTQ calibrates on quant_calib_samples rows of data/train.jsonl (calib_extra_share
of them extra rows, see Config), rendered the way heal trains on them (chat
template, thinking included), and every expert
sees every calibration token (moe_calibrate_all_experts), so experts the router
rarely picks still get a full Hessian. Writes:
  work/candidates/lobbot-moe/   safetensors, config.json with the quantization
                                config, tokenizer and chat template
  work/allocation.json          the candidate's path and size, as for a GGUF
The stage fails before calibrating when the estimated size is over the TaskSpec
max size, and after saving when the files are. The dense student is not
quantized on this path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common.progress import emit
from stages import w4a16
from stages._util import DRY_RUN, Job
from stages.quantize import healed_dir, tok_s

STAGE = "quantize"
NAME = "lobbot-moe"


def calibration_set(job: Job, tok):
    """quant_calib_samples train rows (calib_extra_share of them extra rows) as token
    ids, as heal trains on them."""
    from datasets import Dataset

    from stages.taskdata import calib_examples, tokenize_example

    cfg = job.config
    examples = calib_examples(job.path("data", "train.jsonl"), cfg.quant_calib_samples, cfg.calib_extra_share)
    ids = [tokenize_example(tok, ex["messages"], cfg.quant_calib_len)["input_ids"] for ex in examples]
    return Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})


def recipe(hf: dict, fp8: bool) -> list:
    """FP8 first, so GPTQ fits the experts to the outputs of the FP8 layers they
    will see in vLLM; then W4A16 (int4, symmetric) at the model's group size."""
    from compressed_tensors.quantization import preset_name_to_scheme
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    int4, f8 = w4a16.targets(hf, fp8)
    scheme = preset_name_to_scheme("W4A16", int4)
    weights = scheme.weights.model_copy(update={"group_size": w4a16.group_size(hf)})
    gptq = GPTQModifier(config_groups={"group_0": scheme.model_copy(update={"weights": weights})})
    return ([QuantizationModifier(targets=f8, scheme="FP8_DYNAMIC")] if f8 else []) + [gptq]


def load_text_model(healed: Path):
    """The healed model's text decoder in BF16, on the CPU (oneshot moves one decoder
    layer at a time to the GPU). Heal saves Qwen3.6 and Gemma 4 text only already
    (Gemma4ForCausalLM, see sft.load_causal_lm); a Gemma 4 checkpoint as released is
    multimodal (Gemma4ForConditionalGeneration), and only its text decoder is loaded."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    hf = AutoConfig.from_pretrained(str(healed))
    if hf.model_type != "gemma4":
        return AutoModelForCausalLM.from_pretrained(str(healed), dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(str(healed), config=hf.text_config, dtype=torch.bfloat16,
                                                 key_mapping={r"^model\.language_model\.": "model."})
    model._weight_conversions = None  # else save_pretrained maps the tensors back to model.language_model.*
    return model


def keep_model_config(saved: Path, model) -> None:
    """llm-compressor rewrites config.json from the healed one when it can (for
    vLLM's sake); for Gemma 4 that is the multimodal config, whose towers were not
    saved. Keep the saved model's own config, with the quantization config."""
    p = saved / "config.json"
    cfg = json.loads(p.read_text())
    if cfg.get("model_type") != model.config.model_type:
        own = model.config.to_dict() | {"architectures": [type(model).__name__]}
        p.write_text(json.dumps(own | {"quantization_config": cfg["quantization_config"]}, indent=2) + "\n")


def quantize(job: Job, healed: Path, out: Path) -> None:
    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    from stages import moe_utils as mu

    cfg = job.config
    hf = json.loads((healed / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(healed))
    data = calibration_set(job, tok)
    model = load_text_model(healed)
    emit(STAGE, pct=10, msg=f"GPTQ on {len(data)} rows, {sum(map(len, data['input_ids']))} tokens")
    # processor: else oneshot loads one from the model dir (Gemma 4's wants torchvision)
    oneshot(model=model, processor=tok, dataset=data, recipe=recipe(hf, cfg.quant_fp8_attention),
            max_seq_length=cfg.quant_calib_len, num_calibration_samples=len(data), moe_calibrate_all_experts=True)
    emit(STAGE, pct=90, msg="saving compressed-tensors checkpoint")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, save_compressed=True, max_shard_size="5GB")
    if hf.get("model_type") == "gemma4":
        keep_model_config(tmp, model)
    tok.save_pretrained(tmp)
    mu.fix_saved_config(tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def safetensors_gb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1e9


def check_size(size_gb: float, max_gb: float, what: str) -> None:
    if size_gb > max_gb:
        raise RuntimeError(f"{what} is {size_gb:.2f} GB, over the {max_gb} GB target; keep fewer experts "
                           "(reap_sparsity; stages/w4a16.py max_experts gives the count for a budget)")


def run_stage(job: Job) -> None:
    cfg, spec = job.config, job.spec
    healed = healed_dir(job)
    out = job.path("work", "candidates", NAME)
    out.parent.mkdir(exist_ok=True)
    hf = json.loads((healed / "config.json").read_text())
    k, fp8 = w4a16.n_experts(hf), cfg.quant_fp8_attention

    if DRY_RUN:  # placeholder files: the dry-run reap writes a Qwen3-MoE config without DeltaNet
        out.mkdir(exist_ok=True)
        (out / "config.json").write_text(json.dumps(hf))
        (out / "model.safetensors").write_bytes(b"dry run")
        per_token, parts = None, {}
    else:
        est = w4a16.size_gb(hf, k, fp8)
        emit(STAGE, pct=2, msg=f"{k} experts per layer: {est:.2f} GB estimated, target {spec.target.max_size_gb} GB")
        check_size(est, spec.target.max_size_gb, f"the estimate for {k} experts per layer")
        quantize(job, healed, out)
        check_size(safetensors_gb(out), spec.target.max_size_gb, str(out))
        per_token = w4a16.bytes_per_token_gb(hf, k, fp8)
        parts = {p: round(b / 1e9, 3) for p, b in w4a16.part_bytes(hf, k, fp8).items()}

    size_gb = round(safetensors_gb(out), 2)
    report = {"candidates": {NAME: {
        "path": str(out), "format": "compressed-tensors", "size_gb": size_gb, "num_experts": k,
        "fp8_attention": fp8, "tok_s_est": tok_s(cfg, per_token) if per_token else None,
        "bytes_per_token_gb": round(per_token, 3) if per_token else None, "parts_gb": parts,
    }}}
    job.path("work", "allocation.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {NAME: size_gb})
    emit(STAGE, "done", 100, f"{NAME} {size_gb} GB (compressed-tensors, {k} experts per layer)")
