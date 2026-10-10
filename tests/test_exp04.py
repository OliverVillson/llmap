"""exp04's runner and smoke test logic, without a GPU: the job configs are valid,
the base is picked by the rule, and the smoke test falls back to BF16 DeltaNet."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Qwen3.6-35B-A3B's text config (the numbers stages/w4a16.py sizes from)
QWEN36 = {"text_config": {
    "hidden_size": 2048, "num_hidden_layers": 40, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    "num_experts": 256, "num_experts_per_tok": 8, "vocab_size": 248320, "tie_word_embeddings": False,
    "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "attn_output_gate": True,
    "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10}}


@pytest.fixture
def exp04(tmp_path, monkeypatch):
    monkeypatch.setenv("NVME", str(tmp_path))
    monkeypatch.setenv("LOBBOT_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("EXP04_LOGS", str(tmp_path / "logs"))
    monkeypatch.delenv("EXP04_BASE", raising=False)
    sys.modules.pop("exp04", None)
    return importlib.import_module("exp04")


def score(m, name, lcb):
    d = m.job(name) / "out"
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(json.dumps({"candidates": [{"code": {"suites": {"livecodebench": {"pass@1": lcb}}}}]}))


def test_job_configs_are_valid(exp04):
    from stages._util import Job

    m = exp04
    (m.SMOKE).mkdir(parents=True)
    (m.SMOKE / "result.json").write_text(json.dumps({"fp8_attention": True, "experts": 104, "size_gb_est": 9.6}))
    score(m, "ref", 0.70)
    score(m, "ref-ornith", 0.69)
    for d in (m.make_job("r59-t", m.r59_config()), m.eval_only("ref", {"ref": "/x"}, code_eval_ref=""),
              m.eval_only("r59-t-suites", {"r59-t": "/x"}, "eval-suites")):
        job = Job(d)
        assert job.spec.target.max_size_gb == m.MAX_GB and job.config.code_eval_thinking
    cfg = Job(m.job("r59-t")).config
    assert cfg.teacher == "ornith-ai/Ornith-1.5-35B" and cfg.code_eval_ref == "exp04-ref-ornith"
    assert cfg.reap_sparsity == round(1 - 104 / 256, 6) and cfg.quant_format == "w4a16" and cfg.quant_fp8_attention
    assert cfg.code_eval_samples == 4 and cfg.lcb_since > cfg.data_contest_before


@pytest.mark.parametrize("qwen, ornith, picked", [(0.70, 0.69, "ornith"), (0.70, 0.67, "qwen"), (0.66, 0.72, "ornith")])
def test_ornith_is_built_unless_it_trails_by_more_than_two_points(exp04, qwen, ornith, picked):
    score(exp04, "ref", qwen)
    score(exp04, "ref-ornith", ornith)
    assert exp04.chosen() == picked
    score(exp04, "ref-ornith", 0.0)  # decided once: a later score does not flip it
    assert exp04.chosen() == picked


def test_smoke_falls_back_to_bf16_deltanet_with_fewer_experts():
    import exp04_smoke as smoke

    tried = []

    def fp8_refused(fp8, k, errors):
        tried.append((fp8, k))
        if fp8:
            errors["fp8"] = "refused"
        return not fp8

    out = smoke.decide(QWEN36, 9.8, fp8_refused)
    (k_fp8, k_bf16) = (k for _, k in tried)
    assert out["fp8_attention"] is False and out["experts"] == k_bf16 < k_fp8 == 104
    assert out["size_gb_est"] <= 9.8 and out["errors"] == {"fp8": "refused"}
    assert smoke.decide(QWEN36, 9.8, lambda fp8, k, e: True)["experts"] == 104
    with pytest.raises(SystemExit):
        smoke.decide(QWEN36, 9.8, lambda fp8, k, e: False)
