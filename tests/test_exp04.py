"""exp04's runner and smoke test logic, without a GPU: the build configs are valid,
the builds and the bar follow the rule, and the smoke test falls back to BF16 DeltaNet."""

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
GEMMA4 = {"text_config": {"num_experts": 128, "top_k_experts": 8}}


def load(tmp_path, monkeypatch, bases=None):
    monkeypatch.setenv("NVME", str(tmp_path))
    monkeypatch.setenv("LOBBOT_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setenv("EXP04_LOGS", str(tmp_path / "logs"))
    if bases:
        monkeypatch.setenv("EXP04_BASES", bases)
    else:
        monkeypatch.delenv("EXP04_BASES", raising=False)
    sys.modules.pop("exp04", None)
    m = importlib.import_module("exp04")
    for key, cfg in (("qwen", QWEN36), ("ornith", QWEN36), ("gemma", GEMMA4)):
        m.base_dir(key).mkdir(parents=True, exist_ok=True)
        (m.base_dir(key) / "config.json").write_text(json.dumps(cfg))
    for shape, experts, fp8 in (("qwen", 104, True), ("gemma", 72, True)):
        (m.SMOKE / shape).mkdir(parents=True, exist_ok=True)
        (m.SMOKE / shape / "result.json").write_text(json.dumps({"fp8_attention": fp8, "experts": experts,
                                                                 "size_gb_est": 9.6, "bf16_loads": True}))
    return m


@pytest.fixture
def exp04(tmp_path, monkeypatch):
    return load(tmp_path, monkeypatch)


def score(m, name, rate, **extra):
    m.POLY.mkdir(parents=True, exist_ok=True)
    (m.POLY / f"{name}.json").write_text(json.dumps({"pass_rate_2": rate, "pass_rate_1": rate / 2, **extra}))


def test_build_configs_are_valid(exp04):
    from stages._util import Job

    m = exp04
    for key, rate in (("qwen", 0.60), ("ornith", 0.62), ("gemma", 0.55)):
        score(m, f"{key}-full", rate)
    assert m.builds() == ["ornith", "gemma"]
    for key, kept, total in (("ornith", 104, 256), ("gemma", 72, 128)):
        d = m.make_job(m.build_name(key), m.build_config(key))
        job = Job(d)
        assert job.spec.target.max_size_gb == m.MAX_GB
        cfg = job.config
        assert cfg.teacher == m.BASES[key] and cfg.reap_sparsity == round(1 - kept / total, 6)
        assert cfg.quant_format == "w4a16" and cfg.quant_fp8_attention and cfg.code_eval_thinking
        assert cfg.eval_reasoning_parser == m.PARSER[key] and cfg.data_extra_rows == str(m.rows_file(key))
        assert cfg.code_eval_samples == 1 and cfg.lcb_since > cfg.data_contest_before
        assert (cfg.data_thinking_temperature, cfg.thinking_top_k) == ((1.0, 64) if key == "gemma" else (0.6, 20))
        assert cfg.code_eval_thinking_temperature == cfg.data_thinking_temperature


@pytest.mark.parametrize("qwen, ornith, picked", [(0.60, 0.59, "ornith"), (0.60, 0.57, "qwen"), (0.55, 0.62, "ornith")])
def test_ornith_is_built_unless_it_trails_by_more_than_two_points(exp04, qwen, ornith, picked):
    m = exp04
    score(m, "qwen-full", qwen)
    score(m, "ornith-full", ornith)
    score(m, "gemma-full", 0.50)
    assert m.builds() == [picked, "gemma"]
    assert m.bar() == round(0.9 * max(qwen, ornith), 4)
    score(m, "ornith-full", 0.0)  # decided once: a later score does not flip it
    assert m.builds() == [picked, "gemma"]


def test_builds_wait_for_every_full_score(exp04):
    score(exp04, "qwen-full", 0.6)
    with pytest.raises(RuntimeError):
        exp04.builds()


@pytest.mark.parametrize("bases, built, n_steps", [("qwen,gemma", ["qwen", "gemma"], 12),
                                                    ("ornith", ["ornith"], 9), ("gemma", ["gemma"], 9)])
def test_a_subset_of_bases(tmp_path, monkeypatch, bases, built, n_steps):
    m = load(tmp_path, monkeypatch, bases)
    for key in m.WANTED:
        score(m, f"{key}-full", 0.6)
    assert m.builds() == built and len(m.steps()) == n_steps
    assert m.build_done(1)() is (len(built) == 1)  # an unused second build slot counts as done


def test_full_plan_and_summary(exp04, capsys):
    m = exp04
    assert len(m.steps()) == 13 and m.run_plan(m.steps(), m.summary, m.LOGS, ["x", "--list"]) == 0
    assert "Gemma 4 26B-A4B full, Polyglot" in capsys.readouterr().out
    assert m.summary() == "No Aider Polyglot results yet."
    for key, rate in (("qwen", 0.60), ("ornith", 0.62), ("gemma", 0.55)):
        score(m, f"{key}-full", rate, per_language={"rust": {"pass_rate_2": rate}})
    m.builds()
    score(m, "ornith-w4", 0.58, gpu={"busy_pct": 91.0})
    assert m.best_build() == "ornith"
    out = m.summary()
    assert "ornith-w4" in out and "55.8%" in out and "meets the bar" in out and "91%" in out


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
