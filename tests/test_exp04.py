"""exp04's runner and smoke test logic, without a GPU: the build configs are valid,
the builds and the bar follow the rule, the guards stop broken runs, builds are paired
with the best full model, and the smoke test falls back to BF16 DeltaNet."""

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
# Gemma 4 26B-A4B's config.json as released (multimodal; the size model reads text_config)
GEMMA4_26B = {"model_type": "gemma4", "text_config": {
    "model_type": "gemma4_text", "hidden_size": 2816, "num_hidden_layers": 30, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 2,
    "attention_k_eq_v": True, "enable_moe_block": True, "num_experts": 128, "top_k_experts": 8,
    "moe_intermediate_size": 704, "intermediate_size": 2112, "vocab_size": 262144, "hidden_size_per_layer_input": 0,
    "tie_word_embeddings": True, "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5}}


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
    for shape, experts, fp8 in (("qwen", 106, True), ("gemma", 70, True)):
        (m.SMOKE / shape).mkdir(parents=True, exist_ok=True)
        (m.SMOKE / shape / "result.json").write_text(json.dumps({"fp8_attention": fp8, "experts": experts,
                                                                 "size_gb_est": 9.6, "bf16_loads": True}))
    return m


@pytest.fixture
def exp04(tmp_path, monkeypatch):
    return load(tmp_path, monkeypatch)


def score(m, name, rate, **extra):
    m.POLY.mkdir(parents=True, exist_ok=True)
    (m.POLY / f"{name}.json").write_text(json.dumps({"pass_rate_2": rate, "pass_rate_1": rate / 2,
                                                     "percent_cases_well_formed": 0.95, "n": 225,
                                                     "test_timeouts": 2, **extra}))


def cases(passed: int, start: int = 0, n: int = 200) -> dict:
    """Per-exercise outcomes: `passed` exercises from `start` on pass by the second attempt."""
    return {f"python/ex{i:03d}": [False, start <= i < start + passed] for i in range(n)}


def test_build_configs_are_valid(exp04):
    from stages._util import Job

    m = exp04
    for key, rate in (("qwen", 0.60), ("ornith", 0.62), ("gemma", 0.57)):
        score(m, f"{key}-full", rate)
    assert m.builds() == ["ornith", "gemma"]
    for entry, kept, total in (("ornith", 106, 256), ("gemma", 70, 128), ("ornith@12", 136, 256)):
        key = m.base_of(entry)
        d = m.make_job(m.build_name(entry), m.build_config(entry), m.BIG_MAX_GB if m.big(entry) else m.MAX_GB)
        job = Job(d)
        assert job.spec.target.max_size_gb == (11.95 if entry == "ornith@12" else m.MAX_GB)
        cfg = job.config
        assert cfg.teacher == m.BASES[key] and cfg.reap_sparsity == round(1 - kept / total, 6)
        assert cfg.heal_max_minutes == 90
        assert cfg.data_extra_weight == 2 and cfg.calib_extra_share == 0.5
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
    score(m, "gemma-full", 0.58)
    assert m.builds() == [picked, "gemma"]
    assert m.bar() == round(0.9 * max(qwen, ornith), 4)
    score(m, "ornith-full", 0.0)  # decided once: a later score does not flip it
    assert m.builds() == [picked, "gemma"]


def test_gemma_under_the_bar_gives_its_slot_to_a_12_gb_build(exp04):
    """A build can't beat its own full model, so a Gemma 4 already under the bar can't pass:
    its slot builds the chosen base at ~12 GB, which shows what the 10 GB limit costs."""
    m = exp04
    for key, rate in (("qwen", 0.60), ("ornith", 0.62), ("gemma", 0.50)):
        score(m, f"{key}-full", rate)
    assert m.builds() == ["ornith", "ornith@12"]
    assert "Gemma 4 26B-A4B scored 50.0%" in m.read(m.CHOICE)["swapped"]
    assert m.build_name("ornith@12") == "ornith-w4-12g" and m.label("ornith@12") == "Ornith-1.5 at 12 GB"
    assert m.rows_file(m.base_of("ornith@12")) == m.rows_file("ornith")  # one transcripts run serves both
    score(m, "ornith-w4", 0.55)
    score(m, "ornith-w4-12g", 0.60)
    assert m.best_build() == "ornith"  # the side scores stay on the 10 GB build


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


@pytest.mark.parametrize("gemma", [{"rate": 0.04}, {"rate": 0.5, "percent_cases_well_formed": 0.3}])
def test_a_broken_full_model_run_stops_the_builds(tmp_path, monkeypatch, gemma):
    m = load(tmp_path, monkeypatch)
    score(m, "qwen-full", 0.6)
    score(m, "ornith-full", 0.6)
    rate = gemma.pop("rate")
    score(m, "gemma-full", rate, **gemma)
    with pytest.raises(RuntimeError, match="Gemma 4 26B-A4B"):
        m.builds()
    assert not m.CHOICE.exists()
    monkeypatch.setenv("EXP04_ACCEPT", "1")
    m = load(tmp_path, monkeypatch)
    assert m.builds() == ["ornith", "ornith@12"]  # and gemma, under the bar, gives up its slot


def test_test_timeouts_mean_the_vm_not_the_model(exp04):
    m = exp04
    score(m, "qwen-full", 0.6, test_timeouts=30)
    assert "30 test timeouts in 225 exercises" in m.suspect("qwen-full")
    score(m, "qwen-full", 0.6, test_timeouts=3)
    assert m.suspect("qwen-full") == ""


@pytest.mark.parametrize("passed, start, verdict", [(195, 0, "met"), (170, 30, "too close to call"),
                                                    (140, 30, "missed")])
def test_builds_are_paired_with_the_best_full_model(exp04, passed, start, verdict):
    """The full model passes 180 of 200, so the bar is 162; pairing exercise by exercise
    gives the interval that says whether a build's score is clear of it. A build that
    passes other exercises than the full model is less certain than one that passes a
    subset of them."""
    m = exp04
    score(m, "ornith-full", 0.9, cases=cases(180))
    score(m, "ornith-w4", passed / 200, cases=cases(passed, start))
    p = m.paired("ornith-full", "ornith-w4")
    assert p["n"] == 200 and p["diff"] == pytest.approx((passed - 162) / 200)
    assert m.verdict(p) == verdict
    assert m.paired("ornith-full", "gemma-w4") is None  # no per-exercise results, no pairing


def test_preflight_stops_a_broken_template_early(exp04, monkeypatch):
    m = exp04
    ran = []
    monkeypatch.setattr(m, "sh", lambda step, cmd: ran.append(cmd))
    score(m, "gemma-preflight", 0.0, percent_cases_well_formed=1 / 6, n=6)
    with pytest.raises(RuntimeError, match="pre-flight: 17% of 6 exercises"):
        m.preflight(None, "gemma")
    assert ran == []  # the earlier pre-flight is reused, not run again
    (m.POLY / "gemma-preflight.json").unlink()
    with pytest.raises(RuntimeError, match="wrote no results"):  # sh is faked: asked for, nothing written
        m.preflight(None, "gemma")
    assert "--sample-per-language 1" in ran[0] and "--reasoning-parser gemma4" in ran[0]


def test_mugge_eval_is_an_eval_only_job_in_mugges_format(exp04, monkeypatch):
    from stages._util import Job

    m = exp04
    ran = []
    monkeypatch.setattr(m, "sh", lambda step, cmd: ran.append(cmd))
    m.mugge_eval(None, m.base_dir("gemma"), "gemma-full", "gemma")
    cfg = Job(m.job("gemma-full-mugge")).config
    assert cfg.code_eval_format == "harness" and cfg.code_eval_fix and cfg.code_eval_thinking
    assert cfg.eval_candidates == {"gemma-full": str(m.base_dir("gemma"))} and cfg.thinking_top_k == 64
    assert ran == [f"python pipeline.py --job {m.job('gemma-full-mugge')} --only eval"]
    m.upload_results()  # no evroc here: does nothing, and never fails the step


def test_a_failed_build_does_not_block_the_other(exp04, monkeypatch):
    m = exp04
    for key in ("qwen", "ornith", "gemma"):
        score(m, f"{key}-full", 0.6)
    m.rows_file("ornith").parent.mkdir(parents=True, exist_ok=True)
    m.rows_file("ornith").write_text('{"messages": []}\n' * 10)  # too few rows: logging broke
    monkeypatch.setattr(m, "sh", lambda step, cmd: pytest.fail(f"ran {cmd}"))
    with pytest.raises(m.Skip, match="Ornith-1.5 failed .* 10 training rows"):
        m.run_build(0)(None)
    assert not m.build_done(0)()


def test_full_plan_and_summary(exp04, capsys):
    m = exp04
    assert len(m.steps()) == 13 and m.run_plan(m.steps(), m.summary, m.LOGS, ["x", "--list"]) == 0
    assert "Gemma 4 26B-A4B full, Polyglot" in capsys.readouterr().out
    assert m.summary() == "No Aider Polyglot results yet."
    langs = lambda r: {"python": {"n": 30, "pass_rate_2": r}, "javascript": {"n": 30, "pass_rate_2": r - 0.1},  # noqa: E731
                       "rust": {"n": 30, "pass_rate_2": r}}
    for key, rate in (("qwen", 0.60), ("ornith", 0.62), ("gemma", 0.59)):
        score(m, f"{key}-full", rate, per_language=langs(rate), cases=cases(int(rate * 200)))
    assert m.builds() == ["ornith", "gemma"]
    score(m, "ornith-w4", 0.58, gpu={"busy_pct": 91.0}, per_language=langs(0.58), cases=cases(116))
    assert m.best_build() == "ornith"
    for name, fix in (("ornith-full", 0.80), ("ornith-w4", 0.72)):
        d = m.job(f"{name}-mugge") / "out"
        d.mkdir(parents=True)
        (d / "eval.json").write_text(json.dumps({"candidates": [{"code": {"mean_pass@1": fix - 0.1, "mean_fix@1": fix}}]}))
    out = m.summary()
    assert "ornith-w4" in out and "55.8%" in out and "meets the score bar" in out and "91%" in out
    assert "Paired with ornith-full on 200 exercises: +2.2% over the bar" in out and "too close to call" in out
    assert "53.0%" in out  # Mugge's languages: python and javascript weighted by count, rust left out
    assert "Ornith-1.5: 72.0% after one fix round vs 80.0% for the full Ornith-1.5 (90%)" in out


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
    assert out["fp8_attention"] is False and out["experts"] == k_bf16 == 78 < k_fp8 == 106
    assert out["size_gb_est"] <= 9.8 and out["errors"] == {"fp8": "refused"}
    assert smoke.decide(QWEN36, 9.8, lambda fp8, k, e: True)["experts"] == 106
    with pytest.raises(SystemExit):
        smoke.decide(QWEN36, 9.8, lambda fp8, k, e: False)


def test_smoke_sizes_gemma4():
    """Gemma 4's experts take int4 group 64 (704 is no multiple of 128), its tied
    embeddings stay BF16: 70 experts fit 9.8 GB with FP8 attention and dense MLP."""
    import exp04_smoke as smoke

    assert smoke.decide(GEMMA4_26B, 9.8, lambda fp8, k, e: True) == {
        "fp8_attention": True, "experts": 70, "size_gb_est": 9.77, "errors": {}}
    bf16 = smoke.decide(GEMMA4_26B, 9.8, lambda fp8, k, e: not fp8)
    assert bf16["fp8_attention"] is False and bf16["experts"] == 52 and bf16["size_gb_est"] <= 9.8
    # the tiny copy ends just past the first full-attention layer
    assert smoke.tiny_layers(GEMMA4_26B) == 6 and smoke.tiny_layers(QWEN36) == 4


def test_smoke_tiny_gemma4_is_saved_as_reap_saves_it(tmp_path):
    """Released multimodal, cut to 6 layers and k experts, saved text only
    (Gemma4ForCausalLM) with the full-attention layer's wider heads."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.15")
    import exp04_smoke as smoke
    from test_gemma4_moe import build_tokenizer
    from transformers import AutoModelForCausalLM, Gemma4Config, Gemma4ForConditionalGeneration

    teacher = tmp_path / "teacher"
    build_tokenizer(teacher)
    text = dict(vocab_size=384, hidden_size=64, intermediate_size=48, num_hidden_layers=12, num_attention_heads=2,
                num_key_value_heads=1, head_dim=32, global_head_dim=64, num_global_key_value_heads=1,
                attention_k_eq_v=True, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=8,
                top_k_experts=2, moe_intermediate_size=32)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16)
    Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision)).save_pretrained(teacher)
    tiny = smoke.build_tiny(teacher, tmp_path / "tiny", 4)
    cfg = json.loads((tiny / "config.json").read_text())
    assert cfg["model_type"] == "gemma4_text" and cfg["architectures"] == ["Gemma4ForCausalLM"]
    assert cfg["num_hidden_layers"] == 6 and cfg["layer_types"][-1] == "full_attention" and cfg["num_experts"] == 4
    model = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32)
    assert model.model.layers[5].self_attn.head_dim == 64 and model.model.layers[5].self_attn.v_proj is None
    assert model.model.layers[0].experts.gate_up_proj.shape[0] == 4


def test_failed_selftest_reruns_setup(exp04):
    """selftest.json is written even when the reference solutions fail; setup isn't done until both pass."""
    for out, ok in ((exp04.SELFTEST, True), (Path(f"{exp04.SELFTEST}-train"), False)):
        out.mkdir(parents=True, exist_ok=True)
        (out / "selftest.json").write_text(json.dumps({"ok": ok}))
    assert exp04.selftest_ok(exp04.SELFTEST)
    assert not exp04.selftest_ok(Path(f"{exp04.SELFTEST}-train"))
    assert not exp04.selftest_ok(exp04.LOGS / "missing")
