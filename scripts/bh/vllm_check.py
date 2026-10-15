#!/usr/bin/env python3
"""The teacher's boot dry-run, before any paid hour: in the pinned vLLM (the same venv or
image the B200 will run), without a GPU, check that each model of the ladder would load.

    $LOBBOT_VLLM_PY scripts/bh/vllm_check.py --pin vllm==X --out boot.json MODEL [MODEL ...]

Per model: its config.json and quantization config are fetched (no weights), the
architecture is one this vLLM registers, the quantization method (modelopt NVFP4, FP8,
...) is one it implements, and the chat template renders a write call with
enable_thinking=false and leaves no thinking block open. vLLM's version must equal the
pin. Exits 1 when the first model (the teacher the run plans on) fails; a failing
fallback is reported only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def fetch(model: str, name: str) -> dict | None:
    p = Path(model) / name
    if p.exists():
        return json.loads(p.read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(model, name)).read_text())
    except Exception:  # noqa: BLE001  (absent file)
        return None


def quant_method(cfg: dict, hq: dict | None) -> str | None:
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if qc:
        return qc.get("quant_method")
    if hq:  # ModelOpt's hf_quant_config.json
        return "modelopt_fp4" if "NVFP4" in json.dumps(hq) else "modelopt"
    return None


def check(model: str) -> dict:
    from stages.harness import SYSTEM
    out = {"model": model, "problems": []}
    cfg = fetch(model, "config.json")
    if cfg is None:
        out["problems"].append("no config.json")
        return out
    archs = cfg.get("architectures") or []
    out["architectures"] = archs
    try:
        from vllm.model_executor.models import ModelRegistry
        supported = set(ModelRegistry.get_supported_archs())
        if not any(a in supported for a in archs):
            out["problems"].append(f"architecture {archs} is not registered in this vLLM")
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"vLLM model registry: {type(e).__name__}: {e}")
    qm = quant_method(cfg, fetch(model, "hf_quant_config.json"))
    out["quant_method"] = qm
    if qm:
        try:
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
            if qm not in QUANTIZATION_METHODS and not (qm.startswith("modelopt") and "modelopt_fp4" in QUANTIZATION_METHODS):
                out["problems"].append(f"quantization {qm!r} is not one of this vLLM's")
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"vLLM quantization methods: {type(e).__name__}: {e}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Ticket: t"}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tail = text.rsplit("<think>", 1)[-1] if "<think>" in text else ""
        out["template_tail"] = text[-80:]
        if "<think>" in text and "</think>" not in tail:
            out["problems"].append("enable_thinking=false leaves a <think> block open")
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"chat template: {type(e).__name__}: {e}")
    out["ok"] = not out["problems"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--pin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import vllm
    rep = {"vllm": vllm.__version__, "pin": a.pin, "models": [check(m) for m in a.models]}
    want = a.pin.split("==", 1)[1] if "==" in a.pin else None
    rep["pin_ok"] = want is None or vllm.__version__ == want
    rep["ok"] = rep["pin_ok"] and rep["models"][0]["ok"]
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({"ok": rep["ok"], "vllm": rep["vllm"], "models": {m["model"]: m["problems"] for m in rep["models"]}}))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
