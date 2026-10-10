"""exp04's smoke test on the GPU: does vLLM serve our 4-bit checkpoint of the base?
Run in the training venv (scripts/exp04.py does), before any long step:

    python scripts/exp04_smoke.py --teacher /mnt/nvme/models/Qwen3.6-35B-A3B --out /mnt/nvme/exp04-smoke --budget-gb 9.8

A copy of the base cut just past its first full-attention layer (Qwen3.6: 3 DeltaNet
layers and 1 full, 4 in all; Gemma 4: 5 sliding-window layers and 1 full, 6), with
the expert count the budget allows and random weights, is saved the way REAP saves
a pruned model, quantized by the real quantize stage (quant_format "w4a16") and
served by vLLM through the eval stage's own serve() and generate(). Random weights
answer nonsense; what counts is that vLLM loads the checkpoint and answers at all.
With the FP8 layers (attention, and DeltaNet or Gemma's dense MLP) first; if vLLM
refuses that, again with them in BF16, which costs bytes and so experts. The
unquantized copy is served too, for the before-quantizing score. Writes
<out>/result.json:
  {"fp8_attention": bool, "experts": K, "size_gb_est": GB of the full-depth model,
   "bf16_loads": bool, "errors": {try: message}}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stages import w4a16  # noqa: E402

TASKSPEC = ROOT / "examples" / "python-utils.code.taskspec.json"
# Expert counts go in steps of 2: at 9.8 GB that keeps 106 of Qwen3.6's 256 (104 in
# steps of 8) and 70 of Gemma 4's 128 (64), about 0.13 and 0.19 GB per 2 experts.
STEP = 2


def tiny_layers(hf: dict) -> int:
    """Layers up to and including the first full-attention one, so the tiny copy has
    every kind of layer the base has: 4 for Qwen3.6, 6 for Gemma 4."""
    return w4a16.layer_types(w4a16.text_config(hf)).index("full_attention") + 1


def build_tiny(teacher: Path, out: Path, k: int) -> Path:
    """The base's architecture cut to tiny_layers layers with k experts, random BF16
    weights, saved as the pipeline saves a pruned model (same class, fix_saved_config)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from stages import moe_utils as mu

    if (out / "config.json").exists():
        return out
    n = tiny_layers(json.loads((teacher / "config.json").read_text()))
    cfg = AutoConfig.from_pretrained(teacher)
    if cfg.model_type == "gemma4":  # REAP keeps Gemma 4's text decoder alone (sft.load_causal_lm)
        cfg = cfg.text_config
    t = getattr(cfg, "text_config", cfg)
    per_layer = t.to_dict().get("per_layer_config")  # Gemma 4: the full layers' head size and KV heads
    t.num_hidden_layers = n
    if getattr(t, "layer_types", None):
        t.layer_types = list(t.layer_types)[:n]
    if per_layer:
        t.per_layer_config = {int(i): o for i, o in per_layer.items() if int(i) < n}
    if getattr(t, "mtp_num_hidden_layers", None):
        t.mtp_num_hidden_layers = 0
    mu.set_config_experts(cfg, k)
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(teacher).save_pretrained(tmp)
    mu.fix_saved_config(tmp)
    tmp.rename(out)
    return out


def calib_rows(n: int = 16) -> list[dict]:
    """Chat rows with thinking, the shape data/train.jsonl has."""
    return [{"messages": [{"role": "user", "content": f"Write f(x) that returns x + {i}."},
                          {"role": "assistant", "reasoning_content": f"Add {i} to x.",
                           "content": f"```python\ndef f(x):\n    return x + {i}\n```"}]} for i in range(n)]


def quantize(tiny: Path, job_dir: Path, fp8: bool) -> Path:
    """The real quantize stage on the tiny model; returns the vLLM checkpoint dir."""
    for sub in ("work", "data", ".done"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(json.dumps({
        "quant_format": "w4a16", "quant_fp8_attention": fp8, "quant_calib_samples": 16, "quant_calib_len": 512,
        "dense_fallback": False}))
    spec = json.loads(TASKSPEC.read_text())
    spec["target"] = {**spec.get("target", {}), "max_size_gb": 100}
    (job_dir / "taskspec.json").write_text(json.dumps(spec))
    (job_dir / "data" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in calib_rows()))
    healed = job_dir / "work" / "healed"
    if not healed.exists():
        healed.symlink_to(tiny)
    (job_dir / ".done" / "quantize").unlink(missing_ok=True)
    subprocess.run([sys.executable, "-m", "stages.quantize", "--job", str(job_dir)], cwd=ROOT, check=True)
    alloc = json.loads((job_dir / "work" / "allocation.json").read_text())
    return Path(next(iter(alloc["candidates"].values()))["path"])


def answers(model_dir: Path, job_dir: Path) -> None:
    """Serves model_dir with vLLM as the eval does and asks for one answer, with thinking
    (which also runs the forced-answer path on the checkpoint's tokenizer)."""
    from stages import eval as ev
    from stages._util import Job

    proc = ev.serve(Job(job_dir), str(model_dir), "smoke")
    try:
        got, _ = ev.generate("You write Python.", ["Write f(x) that returns x + 1."], 64, 0.6, 0, True)
        if len(got) != 1:
            raise RuntimeError(f"expected one answer, got {got!r}")
    finally:
        ev.stop(proc)


def attempt(errors: dict, what: str, fn) -> bool:
    try:
        fn()
        return True
    except Exception as e:  # noqa: BLE001
        errors[what] = f"{type(e).__name__}: {e}"[-2000:]
        traceback.print_exc()
        return False


def decide(hf: dict, budget_gb: float, try_format) -> dict:
    """FP8 DeltaNet and attention if vLLM serves them, else BF16; each with the most
    experts that fit the budget. try_format(fp8, k, errors) -> bool."""
    errors: dict = {}
    for fp8 in (True, False):
        k = w4a16.max_experts(hf, budget_gb, fp8, STEP)
        if try_format(fp8, k, errors):
            return {"fp8_attention": fp8, "experts": k, "size_gb_est": round(w4a16.size_gb(hf, k, fp8), 2),
                    "errors": errors}
    raise SystemExit("vLLM served neither format; see the errors above. Send Claude this log.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budget-gb", type=float, default=9.8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hf = json.loads((args.teacher / "config.json").read_text())

    def try_format(fp8: bool, k: int, errors: dict) -> bool:
        tag = f"{'fp8' if fp8 else 'bf16'}-k{k}"
        print(f"== {tag}: {k} experts, {w4a16.size_gb(hf, k, fp8):.2f} GB at full depth", flush=True)
        tiny = build_tiny(args.teacher, args.out / f"tiny-k{k}", k)
        job_dir = args.out / f"job-{tag}"
        holder: dict = {}
        return (attempt(errors, f"{tag} quantize", lambda: holder.setdefault("dir", quantize(tiny, job_dir, fp8)))
                and attempt(errors, f"{tag} vLLM", lambda: answers(holder["dir"], job_dir)))

    result = decide(hf, args.budget_gb, try_format)
    k = result["experts"]
    result["bf16_loads"] = attempt(result["errors"], "bf16 checkpoint vLLM",
                                   lambda: answers(args.out / f"tiny-k{k}", args.out / f"job-{'fp8' if result['fp8_attention'] else 'bf16'}-k{k}"))
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
