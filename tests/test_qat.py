"""stages/qat.py on tiny Granite models: weights in their K-quant grids, Block-AP and
end-to-end KD lower the KL, the loop's guard, time cap and resume, the bit-exact export,
and the on-policy rounds against a fake vLLM with real sandbox labels."""
import contextlib
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).parent))
from test_kquant import can_convert, llama_cpp_dir, llama_quantize, tiny_granite  # noqa: E402
from test_opd import STDIN, FakeVLLM, chat_tokenizer  # noqa: E402

from stages import kquant, opd, qat, sft, traces  # noqa: E402
from stages.eval import EARLY_STOP  # noqa: E402

# test_kquant's llama-quantize plan, by HF name: q2_k, with k, v and the head at q4_k and
# ffn_down at q3_k (all three types, and the embedding and head in their own)
KINDS = {"k_proj": "q4_k", "v_proj": "q4_k", "down_proj": "q3_k", "lm_head": "q4_k"}


def plan(model) -> dict[str, str]:
    return {n: KINDS.get(n.split(".")[-2], "q2_k") for n, p in model.named_parameters() if p.dim() == 2}


def rows_of(n, seed=0, vocab=range(3, 320), prompt=16, answer=(40, 112)):
    rng = random.Random(seed)
    vocab = list(vocab)
    return [{"id": f"cc/{i}", "prompt": [rng.choice(vocab) for _ in range(prompt)],
             "tokens": [rng.choice(vocab) for _ in range(rng.randrange(*answer))], "finish": "stop", "passed": True}
            for i in range(n)]


def kl(student, teacher, ids) -> float:
    with torch.no_grad():
        t = teacher(input_ids=ids).logits.float().log_softmax(-1)
        s = student(input_ids=ids).logits.float().log_softmax(-1)
    return float((t.exp() * (t - s)).sum(-1).mean())


@pytest.fixture(autouse=True)
def few_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


# ---------- the type map and the quantized modules ----------

def test_type_map_reads_horizons_gguf_names_and_refuses_what_qat_cannot_train(tmp_path):
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"gb": 1.2, "groups": {"g": "q2_k"}, "types": {
        "blk.0.attn_q.weight": "Q2_K", "blk.1.ffn_down.weight": "q3_k", "token_embd.weight": "q2_k",
        "output.weight": "q4_k", "blk.0.attn_norm.weight": "f32", "output_norm.weight": "f32"}}))
    assert qat.load_type_map(p) == {"model.layers.0.self_attn.q_proj.weight": "q2_k",
                                    "model.layers.1.mlp.down_proj.weight": "q3_k",
                                    "model.embed_tokens.weight": "q2_k", "lm_head.weight": "q4_k"}
    assert qat.load_type_map({"map": {"lm_head.weight": "q4_k"}}) == {"lm_head.weight": "q4_k"}
    with pytest.raises(ValueError, match="q5_k; QAT trains exactly into"):
        qat.load_type_map({"blk.0.attn_q.weight": "q5_k"})
    with pytest.raises(ValueError, match="no layer, embedding or head weight"):
        qat.load_type_map({"blk.0.attn_gate_x.weight": "q2_k"})
    with pytest.raises(ValueError, match="no quantized weight"):
        qat.load_type_map({"output_norm.weight": "f32"})


def test_quant_model_puts_every_weight_in_its_grid_and_the_logits_are_the_fake_quant(tmp_path):
    m = tiny_granite(tmp_path / "hf", init=0.1)
    tmap = plan(m)
    with pytest.raises(ValueError, match="does not cover the model: 1 weights without a type"):
        qat.quant_model(copy.deepcopy(m), dict(list(tmap.items())[1:]))
    q = copy.deepcopy(m)
    mods = qat.quant_model(q, tmap)
    assert set(mods) == set(tmap) == set(qat.quant_modules(q))
    assert isinstance(q.get_input_embeddings(), qat.QuantEmbedding)
    assert isinstance(q.get_output_embeddings(), qat.QuantLinear)
    assert Counter(m.qw.type for m in mods.values()) == Counter(tmap.values())
    # the forward pass is a plain model holding each weight's dequantized integers
    plain = copy.deepcopy(m)
    params = dict(plain.named_parameters())
    with torch.no_grad():
        for name, mod in mods.items():
            params[name].copy_(kquant.dequant(mod.qw.export()))
    ids = torch.randint(0, 320, (2, 40), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(q(input_ids=ids).logits, plain(input_ids=ids).logits)
    norms = sum(p.numel() for n, p in m.named_parameters() if p.dim() == 1)
    assert qat.size_bytes(q, tmap) == sum(kquant.bytes_for(tmap[n], params[n].numel()) for n in tmap) + 4 * norms

    # a head tied to the embedding becomes its own tensor at its own type
    t = copy.deepcopy(m)
    t.lm_head.weight = t.model.embed_tokens.weight
    t.config.tie_word_embeddings = True
    assert qat.tied(t)
    with pytest.raises(ValueError, match="tied to model.embed_tokens.weight and the map has no type for it"):
        qat.quant_model(copy.deepcopy(t), {k: v for k, v in tmap.items() if k != "lm_head.weight"})
    qat.quant_model(t, tmap)
    assert not t.config.tie_word_embeddings and not qat.tied(t)
    assert (t.lm_head.qw.type, t.model.embed_tokens.qw.type) == ("q4_k", "q2_k")
    assert torch.equal(t.lm_head.qw.weight, t.model.embed_tokens.qw.weight)
    assert t.lm_head.qw.weight is not t.model.embed_tokens.qw.weight


def test_thaw_lets_frozen_codes_move_again_without_changing_the_forward_pass():
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 16, bias=False)
    qw = kquant.QuantWeight(lin.weight.detach(), "q2_k")
    mod = torch.nn.Sequential(qat.QuantLinear(qw))
    opt = torch.optim.AdamW(qat.param_groups(mod), lr=1e-2)
    qat.freeze_codes(mod)
    for _ in range(5):  # the scales move while the codes are frozen: the latent weights fall behind
        mod(torch.randn(4, 512)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
        qw.project_()
    before = qw().detach().clone()
    assert qat.thaw_codes(mod) > 0 and not qw.frozen and qw.weight.requires_grad
    assert torch.equal(qw(), before)


# ---------- Block-AP ----------

def test_block_ap_lowers_every_layers_error_and_the_kl_and_resumes_by_layer(tmp_path):
    m = tiny_granite(tmp_path / "hf", init=0.02)
    m.config.use_cache = False
    teacher = copy.deepcopy(m).eval()
    tmap = plan(m)
    rows = rows_of(40, answer=(100, 112))
    wins = qat.block_windows(rows[:32], 64)
    held = torch.tensor([(r["prompt"] + r["tokens"])[:96] for r in rows[32:]])
    ptq = copy.deepcopy(m)
    qat.quant_model(ptq, tmap)
    b = copy.deepcopy(m)
    rep = qat.block_ap(b, wins, tmap, batch=2, epochs=2, lr=1e-3, work=tmp_path / "layers")
    assert [r["layer"] for r in rep] == [0, 1] and all(r["steps"] == 32 and r["skipped"] == 0 for r in rep)
    assert all(r["rel_err_trained"] < 0.9 * r["rel_err_ptq"] for r in rep), rep
    assert kl(b, teacher, held) < kl(ptq, teacher, held)
    assert set(qat.quant_modules(b)) == set(tmap)
    for name in ("model.embed_tokens.weight", "lm_head.weight"):  # at their PTQ init, untrained
        assert torch.equal(qat.quant_modules(b)[name].qw(), qat.quant_modules(ptq)[name].qw())
    assert not any(q.frozen for q in qat.quant_weights(b))  # trained further, not shipped

    again = copy.deepcopy(m)  # a rerun loads every finished layer instead of training it
    assert qat.block_ap(again, wins, tmap, batch=2, epochs=2, lr=1e-3, work=tmp_path / "layers") == rep
    sd = b.state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in again.state_dict().items())


def test_block_ap_holds_two_hidden_state_caches_not_three(tmp_path, monkeypatch):
    """fit_windows budgets two caches: the bf16 layers' hidden states and the quantized
    layers'. Each advances a layer in place, a batch at a time. Rebuilt as a new list
    beside the old one, the quantized cache was held twice at the end of every layer:
    three caches (64 GB at 2,048 windows of 2,048 tokens and 2,560 hidden, against the
    43 GB budgeted). Every cached batch is followed with a weakref and counted as each
    layer output is made."""
    import weakref
    m = tiny_granite(tmp_path / "hf", init=0.02)
    m.config.use_cache = False
    wins = qat.block_windows(rows_of(8, answer=(100, 112)), 64)
    batch, cached, alive = 2, [], []
    real_inputs, real_run = qat.layer_inputs, qat.run_layer

    def inputs(model, ids):
        h, kw = real_inputs(model, ids)
        cached.append(weakref.ref(h))
        return h, kw

    def run(layer, h, kw):  # without gradients: a cache update (or a held-out error, freed at once)
        if torch.is_grad_enabled():
            return real_run(layer, h, kw)
        alive.append(sum(r() is not None for r in cached))
        out = real_run(layer, h, kw)
        cached.append(weakref.ref(out))
        return out

    monkeypatch.setattr(qat, "layer_inputs", inputs)
    monkeypatch.setattr(qat, "run_layer", run)
    qat.block_ap(m, wins, plan(m), batch=batch, epochs=1, lr=1e-3)
    n = len(wins) // batch  # batches per cache
    assert len(wins) == 8 and alive and max(alive) == 2 * n, alive


def test_block_ap_keeps_as_many_windows_as_the_cache_device_has_room_for(monkeypatch):
    """2048 windows x 2048 tokens x 2560 hidden in bf16, twice, is ~43 GB of host RAM: on a VM
    with less free, Block-AP keeps what fits (after a reserve) and says so, and refuses when
    too few would be left."""
    per = 2 * 2048 * 2560 * 2  # bytes of one window in the two caches
    full = qat.fit_windows(2048, 2048, 2560, 2, "cpu", free=100_000_000_000)
    assert full["windows"] == 2048 and full["cache_gb"] == round(2048 * per / 1e9, 2) and full["free_gb"] == 100.0
    short = qat.fit_windows(2048, 2048, 2560, 2, "cpu", free=30_000_000_000)
    assert short["windows"] == int((30e9 - qat.CACHE_RESERVE_GB["cpu"] * 1e9) // per) < 2048
    assert short["cache_gb"] <= 30 - qat.CACHE_RESERVE_GB["cpu"]
    with pytest.raises(RuntimeError, match="free memory, or pass --cache-device"):
        qat.fit_windows(2048, 2048, 2560, 2, "cpu", free=10_000_000_000)
    assert qat.fit_windows(8, 512, 2560, 2, "cpu", free=9_000_000_000)["windows"] == 8  # a small ask is not refused
    monkeypatch.setattr(qat, "free_bytes", lambda device: None)  # unknown: the request stands
    assert qat.fit_windows(2048, 2048, 2560, 2, "cpu")["windows"] == 2048
    monkeypatch.undo()
    assert qat.free_bytes("cpu") > 0  # MemAvailable of /proc/meminfo


def test_block_windows_start_at_the_prompt_and_drop_ids_the_trim_dropped():
    rows = rows_of(6, vocab=range(3, 300), answer=(60, 61)) + [{"id": "short", "prompt": [5] * 4, "tokens": [6] * 4}]
    rows[0]["tokens"][0] = 310
    w = qat.block_windows(rows, 64)
    assert w.shape == (6, 64) and {tuple(x[:16].tolist()) for x in w} == {tuple(r["prompt"]) for r in rows[:6]}
    kept = [i for i in range(320) if i != 310]
    w = qat.block_windows(rows, 64, vocab_map={"kept": kept, "old_to_new": {o: k for k, o in enumerate(kept)}})
    assert len(w) == 5 and int(w.max()) < len(kept)


def test_block_windows_take_further_windows_of_the_same_trace_up_to_the_count_asked():
    """Too few rows for one window each: the long rows give their next `length` tokens,
    then the next, until the count asked; each window one row's own, none overlapping,
    the rows' first windows (from the prompt) all in. Fewer than asked is reported."""
    rows = rows_of(6, vocab=range(3, 300), answer=(200, 201)) + [{"id": "short", "prompt": [5] * 4, "tokens": [6] * 4}]
    seq = {r["id"]: r["prompt"] + r["tokens"] for r in rows}  # 216 tokens: windows at 0, 64 and 128
    rep = {}
    w = qat.block_windows(rows, 64, 10, report=rep)
    assert w.shape == (10, 64) and rep == {"asked": 10, "rows_long_enough": 6, "windows": 10, "by_offset": [6, 4]}
    where = []
    for x in w.tolist():
        at = [(i, k) for i, s in seq.items() for k in range(0, len(s) - 63, 64) if s[k:k + 64] == x]
        assert len(at) == 1, at  # one row's own tokens, at a multiple of the length
        where.append(at[0])
    assert len(set(where)) == 10  # no window twice
    assert {i for i, k in where if k == 0} == set(seq) - {"short"} and {k for _, k in where} == {0, 64}
    rep = {}
    assert len(qat.block_windows(rows, 64, 100, report=rep)) == 18 and rep["by_offset"] == [6, 6, 6]
    assert rep["windows"] < rep["asked"]  # the shortfall cmd_block writes into block.json
    # what fits in the first windows is what one window per row gave before
    assert torch.equal(qat.block_windows(rows, 64, 4), qat.block_windows(rows, 64)[:4])
    # a later window holding a dropped id is left out too
    rows[0]["tokens"][100] = 310
    kept = [i for i in range(320) if i != 310]
    rep = {}
    qat.block_windows(rows, 64, 100, vocab_map={"kept": kept, "old_to_new": {o: k for k, o in enumerate(kept)}},
                      report=rep)
    assert rep["by_offset"] == [6, 5, 6]


# ---------- the loop: guard, time cap, resume ----------

def tiny_quant(tmp_path, seed=0):
    m = tiny_granite(tmp_path / "hf", seed=seed, init=0.1)
    m.config.use_cache = False
    teacher = copy.deepcopy(m).eval()
    qat.quant_model(m, plan(m))
    return m, teacher


def kd_step(student, teacher):
    def step(b):
        ids, mask, w = b
        loss, _, _ = qat.kd_loss(student, teacher, ids, mask, w)
        return loss, int(mask.sum())
    return step


def data_of(rows, length=96, batch=4):
    n = sum(1 for _ in traces.batches(rows, length, batch))
    return (lambda e: traces.batches(rows, length, batch, seed=e)), n


def test_guard_skips_non_finite_steps_and_raises_on_a_streak_and_nothing_non_finite_is_saved(tmp_path):
    student, teacher = tiny_quant(tmp_path)
    data, n = data_of(rows_of(24))
    good = kd_step(student, teacher)
    calls = []

    def flaky(b):
        calls.append(1)
        loss, k = good(b)
        if len(calls) in (2, 3):  # a non-finite loss
            return loss * float("nan"), k
        if len(calls) == 5:  # a finite loss with a non-finite gradient
            p = student.model.norm.weight
            return loss + (p - p.detach()).abs().sqrt().sum(), k
        return loss, k

    rep = qat.fit(student, data, n, 2 * n, flaky, lr=1e-3, out=tmp_path / "run", ckpt_every=4, freeze_steps=0)
    assert rep["skipped"] == 3 and rep["steps"] == 2 * n and rep["tokens_applied"] < rep["tokens"]
    qat.check_finite(student)

    def bad(b):
        loss, k = good(b)
        return loss * float("inf"), k

    with pytest.raises(sft.NonFiniteTraining, match="8 non-finite steps"):
        qat.fit(student, data, n, 100, bad, lr=1e-3, freeze_steps=0)
    qat.check_finite(student)  # no update was applied
    with torch.no_grad():
        student.model.layers[0].mlp.up_proj.qw.d[0] = float("nan")
    with pytest.raises(sft.NonFiniteTraining, match="not finite"):
        qat.save_state(student, tmp_path / "state", {})
    assert not (tmp_path / "state").exists() and not (tmp_path / "state.tmp").exists()


def test_the_time_cap_is_checked_first_and_anneals_the_rate_to_zero(tmp_path, monkeypatch):
    student, teacher = tiny_quant(tmp_path)
    data, n = data_of(rows_of(24))
    lrs = []
    real = qat._set_lr
    monkeypatch.setattr(qat, "_set_lr", lambda opt, lr: (lrs.append(lr), real(opt, lr)))
    now = [0.0]
    monkeypatch.setattr(qat.time, "monotonic", lambda: now[0])
    good = kd_step(student, teacher)

    def slow(b):
        now[0] += 6.0  # every step takes 6 s of training; the cap is 60 s
        return good(b)

    rep = qat.fit(student, data, n, 1000, slow, lr=1e-3, warm=2, max_minutes=1.0, freeze_steps=3)
    assert rep["cap_fired"] and rep["steps"] == 10 and rep["minutes"] == 1.0 and rep["frozen"]
    assert lrs[0] == pytest.approx(0.5e-3) and lrs[1] == pytest.approx(1e-3) and lrs[-1] < 0.05 * 1e-3
    assert all(a > b for a, b in zip(lrs[1:], lrs[2:]))  # annealed by the clock, not by the 1000 steps


def test_resume_keeps_the_schedule_the_optimizer_and_the_training_seconds(tmp_path, monkeypatch):
    rows = rows_of(24)
    data, n = data_of(rows)
    total = 3 * n
    now = [0.0]
    monkeypatch.setattr(qat.time, "monotonic", lambda: now[0])
    lrs = []
    real = qat._set_lr
    monkeypatch.setattr(qat, "_set_lr", lambda opt, lr: (lrs.append(lr), real(opt, lr)))

    def run(out, stop_at=None):
        student, teacher = tiny_quant(tmp_path)
        good = kd_step(student, teacher)
        steps = [0]

        def step(b):
            steps[0] += 1
            if steps[0] == stop_at:
                raise KeyboardInterrupt
            now[0] += 1.0
            return good(b)

        held = list(traces.batches(rows[:4], 96, 2))
        rep = qat.fit(student, data, n, total, step, lr=1e-3, warm=2, out=out, ckpt_every=5, freeze_steps=4,
                      eval_fn=lambda: {"kl": qat.mean_kl(student, teacher, held)}, eval_every=6)
        return student, rep

    whole, rep = run(tmp_path / "whole")
    straight = list(lrs)
    lrs.clear()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", stop_at=13)  # after the checkpoint at step 10
    saved = torch.load(tmp_path / "cut" / "ckpt.pt")["state"]
    assert saved["step"] == 10 and saved["train_s"] == pytest.approx(10.0)
    lrs.clear()
    resumed, rep2 = run(tmp_path / "cut")
    assert lrs == straight[10:]  # the same rates from where it stopped
    sd = whole.state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in resumed.state_dict().items())
    assert rep2["steps"] == rep["steps"] == total and rep2["minutes"] == rep["minutes"] == round(total / 60, 2)
    assert [h["step"] for h in rep2["held"]] == [h["step"] for h in rep["held"]]
    assert rep2["held"][-1]["kl"] == rep["held"][-1]["kl"] < rep["held"][0]["kl"]
    assert rep2["frozen"] and all(q.frozen for q in qat.quant_weights(resumed))


def test_codes_freeze_only_in_a_run_long_enough_and_the_report_has_the_rate_and_peak():
    """In a run shorter than 2 x freeze_steps the freeze would come at step 1, and a rate
    measured on it would leave out the latent weights' gradients: the codes stay free.
    The report gives tokens per second after the first step (whose optimizer state is
    allocated then) and the GPU peak (None without a GPU)."""
    torch.manual_seed(0)
    lin = torch.nn.Sequential(qat.QuantLinear(kquant.QuantWeight(torch.randn(8, 256), "q2_k")))
    x = torch.randn(4, 256)

    def step_loss(b):
        return lin(b).pow(2).mean(), 4

    def run(total, freeze):
        qat.thaw_codes(lin)
        return qat.fit(lin, lambda e: [x] * total, total, total, step_loss, lr=1e-3, warm=1, freeze_steps=freeze)

    rep = run(20, 50)
    assert not rep["frozen"] and rep["steps"] == 20 and rep["freeze_steps"] == 50
    assert not any(q.frozen for q in qat.quant_weights(lin))
    assert run(19, 10)["frozen"] is False and run(20, 10)["frozen"] is True
    assert rep["tok_s"] > 0 and "peak_gb" in rep and (rep["peak_gb"] is None) == (not torch.cuda.is_available())


# ---------- end-to-end KD ----------

def test_e2e_kd_on_a_trimmed_student_lowers_the_held_out_kl(tmp_path):
    from transformers import GraniteForCausalLM
    teacher = tiny_granite(tmp_path / "hf", init=0.1).eval()
    teacher.config.use_cache = False
    kept = [i for i in range(320) if i % 16 != 5][:300]
    vocab = {"kept": kept, "old_to_new": {str(o): k for k, o in enumerate(kept)}}
    cfg = copy.deepcopy(teacher.config)
    cfg.vocab_size = len(kept)
    student = GraniteForCausalLM(cfg)
    idx = torch.tensor(kept)
    student.load_state_dict({k: v[idx] if k in ("model.embed_tokens.weight", "lm_head.weight") else v
                             for k, v in teacher.state_dict().items()})
    qat.quant_model(student, plan(student))
    rows = rows_of(48, vocab=kept, answer=(60, 100))  # in the teacher's (original) ids, as traces are
    train, held = rows[:40], rows[40:]
    rep = qat.e2e_kd(student, teacher, train, held, tmp_path / "e2e", vocab=vocab, length=96, batch=4,
                     tokens=2 * sum(min(96, 16 + len(r["tokens"])) for r in train), lr=1e-3, eval_every=10,
                     held_n=4, held_length=96, freeze_steps=3)
    assert rep["steps"] == rep["total"] == 20 and rep["epochs"] == 2.0 and not rep["cap_fired"]
    assert rep["tokens"] == rep["tokens_applied"] > 0 and rep["frozen"] and rep["skipped"] == 0
    kls = [h["kl"] for h in rep["held"]]
    assert [h["step"] for h in rep["held"]] == [0, 10, 20] and kls[-1] < 0.85 * kls[0], kls
    assert rep["memory_gb"]["total"] > 0
    with pytest.raises(ValueError, match="keeps 299 ids, the student has 300"):
        qat.e2e_kd(student, teacher, train, held, tmp_path / "e2e2", vocab={"kept": kept[:-1]}, length=96)


def test_the_b200_estimate_fits_full_weight_qat_and_the_l40s_does_not():
    mem = qat.e2e_memory()  # untrimmed, 16k tokens
    assert qat.L40S_GB < mem["weights_grads_adam"] < mem["total"] < 0.7 * qat.B200_GB
    assert mem["total"] == pytest.approx(sum(v for k, v in mem.items() if k != "total"), abs=0.1)
    assert qat.e2e_memory(tokens=8192)["total"] < mem["total"] < qat.e2e_memory(tokens=qat.MAX_MODEL_LEN)["total"]
    assert qat.e2e_memory(tokens=qat.MAX_MODEL_LEN)["total"] < 0.75 * qat.B200_GB  # a whole on-policy rollout


# ---------- export ----------

def test_export_packs_the_trained_weights_and_gguf_py_reads_them_back_bit_for_bit(tmp_path):
    llama_cpp = llama_cpp_dir()
    can_convert()
    gguf = pytest.importorskip("gguf")
    m = tiny_granite(tmp_path / "hf", init=0.1)
    tmap = plan(m)
    mods = qat.quant_model(m, tmap)
    torch.manual_seed(1)
    with torch.no_grad():  # stand-in for training: latent weights, scales and a norm move
        for mod in mods.values():
            mod.qw.weight.add_(torch.randn_like(mod.qw.weight) * float(mod.qw.weight.std()) * 0.3)
            mod.qw.scales.add_(torch.randn_like(mod.qw.scales))
            mod.qw.project_()
        m.model.norm.weight.mul_(1.0 + 0.01 * torch.randn_like(m.model.norm.weight))
    qat.freeze_codes(m.model.layers[1])  # some weights shipped with frozen codes
    state = qat.save_state(m, tmp_path / "run" / "state", {"base": str(tmp_path / "hf"), "type_map": tmap})
    model, meta = qat.load_state(tmp_path / "run")
    ids = torch.randint(0, 320, (2, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, m(input_ids=ids).logits)  # the state round trip
    assert meta["type_map"] == tmap and state == tmp_path / "run" / "state"

    out = tmp_path / "out" / "model.gguf"
    rep = qat.export(model, tmap, tmp_path / "hf", out, llama_cpp, vllm_dir=tmp_path / "out" / "vllm")
    assert rep["exact"] == dict(Counter(tmap.values())) and rep["bytes"] == out.stat().st_size
    assert rep["tensor_bytes_estimate"] < rep["bytes"] < rep["tensor_bytes_estimate"] + 2_000_000
    tensors = {t.name: t for t in gguf.GGUFReader(str(out)).tensors}
    trained_norm = model.model.norm.weight.detach().float().numpy()
    assert tensors["output_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert (tensors["output_norm.weight"].data == trained_norm).all()  # fp32, as trained
    # the vLLM checkpoint, written from the file's dequantization, is the trained model at bf16
    from transformers import AutoModelForCausalLM
    served = AutoModelForCausalLM.from_pretrained(str(tmp_path / "out" / "vllm"), dtype=torch.bfloat16)
    plain = qat.plain_state_dict(model)
    for name, p in served.named_parameters():
        assert torch.equal(p, plain[name].to(torch.bfloat16)), name
    # a weight that moves after packing is caught
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.qw.d.mul_(1.5)
    with pytest.raises(ValueError, match="blk.0.ffn_up.weight: the packed weights are not the trained ones"):
        qat.check_exact(model, out, tmap)


def test_qat_starts_exactly_at_llama_quantizes_file_and_block_ap_improves_on_it(tmp_path):
    llama_cpp = llama_cpp_dir()
    can_convert()
    gguf = pytest.importorskip("gguf")
    from transformers import AutoModelForCausalLM

    from stages.qheal import hf_name
    m = tiny_granite(tmp_path / "hf", init=0.02)
    m.config.use_cache = False
    q2k = llama_quantize(tmp_path / "hf", tmp_path, llama_cpp, False)
    tmap = {hf_name(t.name): t.tensor_type.name.lower() for t in gguf.GGUFReader(str(q2k)).tensors
            if len(t.shape) == 2}
    assert tmap == plan(m)
    codes = qat.gguf_codes([q2k], tmp_path / "hf", m, tmap)
    ptq = copy.deepcopy(m)
    qat.quant_model(ptq, tmap, codes)
    assert all(mod.qw.nudged == 0 for mod in qat.quant_modules(ptq).values())
    assert qat.check_exact(ptq, q2k, tmap) == dict(Counter(tmap.values()))  # the PTQ file, bit for bit

    teacher = AutoModelForCausalLM.from_pretrained(str(tmp_path / "hf"), dtype=torch.float32).eval()
    rows = rows_of(40, answer=(100, 112))
    held = torch.tensor([(r["prompt"] + r["tokens"])[:96] for r in rows[32:]])
    rep = qat.block_ap(m, qat.block_windows(rows[:32], 64), tmap, codes, batch=2, epochs=2, lr=1e-3)
    assert all(r["rel_err_trained"] < r["rel_err_ptq"] and r["nudged"] == 0 for r in rep), rep
    assert kl(m, teacher, held) < kl(ptq, teacher, held)
    with pytest.raises(ValueError, match="in no GGUF at their planned type"):
        qat.gguf_codes([q2k], tmp_path / "hf", m, {**tmap, "lm_head.weight": "q2_k"})


def test_cli_block_then_e2e_then_export(tmp_path, monkeypatch):
    llama_cpp = llama_cpp_dir()
    can_convert()
    gguf = pytest.importorskip("gguf")
    hf = tmp_path / "hf"
    tiny_granite(hf, init=0.02)
    q2k = llama_quantize(hf, tmp_path, llama_cpp, False)
    types = {t.name: t.tensor_type.name.lower() for t in gguf.GGUFReader(str(q2k)).tensors}
    assert "f32" in types.values()  # horizon's map lists the norms too
    (tmp_path / "map.json").write_text(json.dumps({"gb": 0.001, "types": types}))
    tr = tmp_path / "traces.jsonl"
    tr.write_text("".join(json.dumps(r) + "\n" for r in rows_of(40, answer=(60, 100))))

    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["qat", *map(str, args)])
        qat.main()

    cli("block", "--base", hf, "--map", tmp_path / "map.json", "--gguf", q2k, "--traces", tr, "--out", tmp_path / "block",
        "--windows", 16, "--length", 64, "--batch", 4, "--epochs", 1, "--lr", 1e-3)
    block = json.loads((tmp_path / "block" / "block.json").read_text())
    assert len(block["layers"]) == 2 and block["windows"] == 16 and not (tmp_path / "block" / "layers").exists()
    assert block["cache"]["windows"] == 16 and block["cache"]["device"] == "cpu"  # sized from MemAvailable
    assert block["asked"] == 16 and block["warnings"] == [] and block["windows_by_offset"] == [16]
    gpu = torch.cuda.is_available()
    assert "peak_gb" in block and (block["peak_gb"] is None) == (not gpu)
    # more windows than the rows have: the long rows give a second and third, and the shortfall is written
    cli("block", "--base", hf, "--map", tmp_path / "map.json", "--gguf", q2k, "--traces", tr, "--out", tmp_path / "more",
        "--windows", 500, "--length", 32, "--batch", 8, "--epochs", 1, "--lr", 1e-3)
    more = json.loads((tmp_path / "more" / "block.json").read_text())
    rows = more["rows_long_enough"]
    assert more["asked"] == 500 and rows == len(traces.split(traces.load(tr))[0])
    assert more["windows"] == sum(more["windows_by_offset"]) < 500 and more["windows_by_offset"][:2] == [rows, rows]
    assert len(more["warnings"]) == 1 and f"{more['windows']} windows of the 500 asked" in more["warnings"][0]
    cli("e2e", "--state", tmp_path / "block", "--teacher", hf, "--traces", tr, "--out", tmp_path / "e2e", "--tokens", 4000,
        "--length", 96, "--batch", 4, "--lr", 1e-3, "--eval-every", 5, "--held-rows", 2, "--held-length", 96,
        "--freeze-steps", 2)
    e2e = json.loads((tmp_path / "e2e" / "e2e.json").read_text())
    assert e2e["steps"] == e2e["total"] > 0 and e2e["tokens"] >= 4000 and not e2e["cap_fired"]
    assert e2e["freeze_steps"] == 2 and e2e["frozen"] and (e2e["peak_gb"] is None) == (not gpu)
    assert not (tmp_path / "e2e" / "ckpt.pt").exists()
    meta = json.loads((tmp_path / "e2e" / "state" / qat.META).read_text())
    assert meta["phase"] == "e2e" and meta["type_map"] == qat.load_type_map(tmp_path / "map.json")
    cli("export", "--state", tmp_path / "e2e", "--out", tmp_path / "final", "--llama-cpp", llama_cpp)
    ex = json.loads((tmp_path / "final" / "export.json").read_text())
    assert ex["exact"] == dict(Counter(meta["type_map"].values())) and ex["phase"] == "e2e"
    assert (tmp_path / "final" / "vllm" / "config.json").exists()


# ---------- on-policy: rules ----------

def metrics(kl_, per):
    return {"kl": kl_, "per_problem": {f"p{i}": dict(zip(("pass", "loop", "cap"), v)) for i, v in enumerate(per)}}


def test_keep_rules():
    best = metrics(0.20, [(0.5, 0.0, 0.5), (1.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.0)])
    kept, rules = qat.keep_round(metrics(0.18, [(0.5, 0.0, 0.5), (1.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.0)]),
                                 best)
    assert kept and rules["kl"]["drop"] == pytest.approx(0.1) and rules["problems"]["n"] == 4
    kept, rules = qat.keep_round(metrics(0.195, [(0.5, 0.0, 0.5)] * 4), best)  # 2.5% < 5%
    assert not kept and not rules["kl"]["ok"]
    # loops up on every problem: a rise with no spread is beyond 2 SE
    kept, rules = qat.keep_round(metrics(0.1, [(0.5, 0.5, 0.5), (1.0, 0.5, 0.0), (0.0, 1.0, 0.5), (0.5, 0.5, 0.0)]),
                                 best)
    assert not kept and not rules["loop"]["ok"] and rules["cap"]["ok"] and rules["pass"]["ok"]
    # pass@1 down on one problem of four, with spread: within 2 SE
    kept, rules = qat.keep_round(metrics(0.1, [(0.5, 0.0, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.0)]),
                                 best)
    assert kept and rules["pass"]["diff"] == pytest.approx(-0.125)
    # ... and down on all of them is not
    kept, rules = qat.keep_round(metrics(0.1, [(0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.5), (0.0, 0.0, 0.0)]),
                                 best)
    assert not kept and not rules["pass"]["ok"]
    kept, rules = qat.keep_round({"kl": 0.1, "per_problem": {"other": {"pass": 1, "loop": 0, "cap": 0}}}, best)
    assert not kept and not rules["problems"]["ok"]


def test_loops_are_thinking_reports_and_cut_rollout_files_are_repaired(tmp_path):
    assert qat.looped("Let me check this again. " * 80)
    assert not qat.looped(" ".join(f"step {i}: x{i * 7919 % 1000} -> y{i * 104729 % 997}" for i in range(80)))
    assert not qat.looped("short " * 50)
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps({"id": i, "n": k}) + "\n" for i, k in (("a", 0), ("a", 1), ("b", 0)))
                 + '{"id": "c", "n')
    assert [r["id"] for r in qat._complete_rows(p, 2)] == ["a", "a"]
    assert [json.loads(x)["id"] for x in p.read_text().splitlines()] == ["a", "a"]


def test_fresh_rollouts_stop_at_the_token_cap(rounds, tmp_path):
    """--rollout-tokens-cap: once a session's fresh rollouts hold that many answer tokens
    no further problem is asked (the ones running finish and are kept); a rerun of the
    session asks nothing more, and without the cap it asks the rest."""
    import time
    r = rounds()
    slow = FakeVLLM(r.tok, lambda body: time.sleep(0.2) or r.fake.complete(body))  # one problem at a time
    try:
        pool = {p["id"]: p for p in opd.problems(str(r.bench))}
        probs = [pool[i] for i in r.fresh]
        out = tmp_path / "fresh.jsonl"
        kw = {"max_tokens": 300, "max_model_len": qat._config(r.work).length, "workers": 1}
        rows = qat.sample_rollouts(slow.url, r.tok, probs, 2, out, token_cap=1, **kw)
        ids = {x["id"] for x in rows}
        assert 1 <= len(ids) <= 2 < len(probs) and len(rows) == 2 * len(ids)  # the first, and the one running
        assert [json.loads(x)["id"] for x in out.read_text().splitlines()] == [x["id"] for x in rows]
        asked = len(slow.bodies("/v1/completions"))
        assert qat.sample_rollouts(slow.url, r.tok, probs, 2, out, token_cap=1, **kw) == rows
        assert len(slow.bodies("/v1/completions")) == asked
        assert {x["id"] for x in qat.sample_rollouts(slow.url, r.tok, probs, 2, out, **kw)} == set(r.fresh)
    finally:
        slow.stop()


def test_onpolicy_session_budgets_follow_the_latest_call(tmp_path, monkeypatch):
    """--max-minutes (a round's training minutes, the rate annealed by time) and
    --rollout-tokens-cap are a session's budgets: a call that gives them sets them from
    its session on, in the config.json the trainer and the sampler read; a call
    without them keeps the last; the run's other settings stay as it started."""
    work = tmp_path / "w"
    seen = []

    def one(w):
        cfg, st = qat._config(w), qat.read_state(w)
        seen.append((st["session"], cfg.max_minutes, cfg.rollout_tokens_cap))
        st.update(trained=st["trained"] + 1, session=st["session"] + 1)
        qat.write_state(w, st)
        return st

    monkeypatch.setattr(qat, "onpolicy_round", one)

    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["qat", "onpolicy", "--work", str(work), *map(str, args)])
        qat.main()

    cli("--state", tmp_path / "start", "--teacher", "t", "--problems", "p", "--rounds", 4, "--until-session", 1,
        "--max-minutes", 30, "--rollout-tokens-cap", 8_000_000)
    cli("--until-session", 2)
    cli("--until-session", 3, "--max-minutes", 20.5, "--rollout-tokens-cap", 0)
    assert seen == [(1, 30.0, 8_000_000), (2, 30.0, 8_000_000), (3, 20.5, 0)]
    cfg = qat._config(work)
    assert (cfg.teacher, cfg.problems, cfg.rounds, cfg.freeze_steps) == ("t", "p", 4, qat.FREEZE_STEPS)


def test_cross_entropy_only_on_passing_answers_the_student_wrote():
    rows = [{"id": "a", "prompt": [1, 2], "tokens": [3, 4, 5], "passed": True, "forced": None},
            {"id": "b", "prompt": [1, 2], "tokens": [3, 4, 6, 7], "passed": True, "forced": "length"},
            {"id": "c", "prompt": [1, 2], "tokens": [3, 4, 6, 7, 8], "passed": False, "forced": None}]
    got = {tuple(ids[0].tolist()): ce for ids, mask, w, ce in qat.rollout_batches(rows, 64, 1, seed=0)}
    assert got == {(1, 2, 3, 4, 5): True, (1, 2, 3, 4, 6, 7): False, (1, 2, 3, 4, 6, 7, 8): False}
    for ids, mask, w, ce in qat.rollout_batches(rows, 64, 1, seed=0):
        assert w[0, 0] == 0 and (w[0, 1:] == 1).all()  # nothing on the prompt, everything after it


def test_gpu_free_without_a_gpu_and_with_one_still_held(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "none"))
    qat.gpu_free(wait_s=0)  # no nvidia-smi: nothing to wait for
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text('#!/bin/sh\ncase "$1" in --query-gpu*) echo 30000;; *) echo "pid, used_memory"; '
                   'echo "4242, 30000 MiB";; esac\n')
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{smi.parent}:/usr/bin:/bin")
    with pytest.raises(RuntimeError, match="(?s)GPU not free.*30000 MiB used.*4242"):
        qat.gpu_free(wait_s=0, poll=0)


# ---------- on-policy: rounds against a fake server ----------

THINK = " ".join(["Let me think about the input format and the edge cases first."] * 6)


def chat_granite(d, init=0.1):
    """A tiny Granite with chat_tokenizer's vocab, its head tied to the embedding (QAT
    unties it), saved to d."""
    from transformers import GraniteConfig, GraniteForCausalLM
    tok = chat_tokenizer(d)
    torch.manual_seed(0)
    cfg = GraniteConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125,
                        logits_scaling=2.0, tie_word_embeddings=True, eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id, initializer_range=init)
    m = GraniteForCausalLM(cfg)
    m.save_pretrained(d)
    return m, tok


def onpolicy_bench(d, n_held=2, n_fresh=8):
    """codecontests.jsonl with n_held problems traces.split holds out and n_fresh it does not."""
    ids = [f"cc/{i}" for i in range(5000)]
    held = [i for i in ids if traces.is_held(i)][:n_held]
    fresh = [i for i in ids if not traces.is_held(i)][:n_fresh]
    d.mkdir(parents=True)
    with open(d / "codecontests.jsonl", "w") as f:
        for i in held + fresh:
            f.write(json.dumps({**STDIN, "id": i, "prompt": f"Problem {i}: read an integer n and print twice its value."})
                    + "\n")
    return d, held, fresh


class Rounds:
    """A start state, its work dir and a fake vLLM that answers every problem with one
    right answer (its thinking closed) and one thinking cut by the budget, whose forced
    answer is wrong."""

    def __init__(self, d, **cfg):
        self.base = d / "base"
        model, self.tok = chat_granite(self.base)
        tmap = {n: "q2_k" for n, p in model.named_parameters() if p.dim() == 2}
        tmap["lm_head.weight"] = "q4_k"
        qat.quant_model(model, tmap)
        self.start = qat.save_state(model, d / "e2e" / "state", {"base": str(self.base), "type_map": tmap,
                                                                 "phase": "e2e"})
        self.bench, self.held, self.fresh = onpolicy_bench(d / "bench")
        enc = lambda text: self.tok.encode(text, add_special_tokens=False)  # noqa: E731
        eos = self.tok.eos_token_id
        self.right = enc("```python\nprint(int(input()) * 2)\n```") + [eos]
        self.wrong = enc("```python\nprint(int(input()) * 3)\n```") + [eos]
        self.closed, self.cut = enc(THINK + "</think>\n\n") + self.right, enc(THINK)

        def complete(body):
            if body["n"] == 1:  # the forced answer
                return [(self.wrong, "stop")]
            return [(self.closed, "stop"), (self.cut, "length")] * (body["n"] // 2)

        self.fake = FakeVLLM(self.tok, complete)
        self.work = d / "work"
        self.served, self.trains, self.fail_train = [], 0, False
        qat.init_onpolicy(self.work, d / "e2e", qat.OnPolicy(
            teacher=str(self.base), problems=str(self.bench), **{
                "fresh": 3, "rollouts": 2, "held": 2, "held_rollouts": 2, "max_tokens": 300, "epochs": 2,
                "length": 300 + opd.ANSWER_TOKENS + qat.FORCED_MARGIN + 1000, "lr": 1e-3, "seqs": 8, **cfg}))

    @contextlib.contextmanager
    def serve(self, model_dir, *, port, seqs, log_path):
        self.served.append(Path(model_dir))
        yield self.fake.url

    def run(self, step, work, target=None):
        if step != "train":
            return qat.run_step(step, work, target)  # its own process, as on the GPU
        self.trains += 1
        if self.fail_train:
            self.fail_train = False
            raise RuntimeError("the trainer stopped")
        qat.onpolicy_train(work)

    def round(self):
        return qat.onpolicy_round(self.work, serve=self.serve, run=self.run)

    def asked(self):
        return len(self.fake.bodies("/v1/completions"))


@pytest.fixture
def rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(opd, "BACKOFF", 0.0)
    monkeypatch.setattr(opd, "HEALTH_WAIT", 0.0)
    monkeypatch.setattr(opd, "_forced_ids", {})
    made = []

    def make(**cfg):
        made.append(Rounds(tmp_path, **cfg))
        return made[-1]

    yield make
    for r in made:
        r.fake.stop()


def test_onpolicy_rounds_sample_label_judge_train_and_resume(rounds):
    r = rounds(rounds=2, freeze_steps=3)  # a 12-step round freezes its codes only with freeze_steps <= 6
    start = r.start
    r.fail_train = True
    with pytest.raises(RuntimeError, match="the trainer stopped"):
        r.round()
    rd = r.work / "round1"
    asked = r.asked()
    assert r.served == [start / "serve"] and (start / "serve" / "model.safetensors").exists()  # written by its CLI
    held = [json.loads(x) for x in (rd / "held.labelled.jsonl").read_text().splitlines()]
    fresh = [json.loads(x) for x in (rd / "rollouts.labelled.jsonl").read_text().splitlines()]
    assert sorted({x["id"] for x in held}) == sorted(r.held) and len(held) == 4
    assert {x["id"] for x in fresh} <= set(r.fresh) and len(fresh) == 6
    for x in held + fresh:  # eval-shaped: the forced answer after the cut thinking, labelled by the sandbox
        assert qat.thinking_text(r.tok, x["tokens"]) == THINK
        if x["finish"] == "stop":
            assert x["tokens"] == r.closed and x["passed"] and x["forced"] is None
        else:
            close = r.tok.encode(EARLY_STOP + "\n</think>\n\n", add_special_tokens=False)
            assert x["tokens"] == r.cut + close + r.wrong and not x["passed"] and x["forced"] == "length"
    m = json.loads((rd / "held.metrics.json").read_text())
    assert (m["pass"], m["cap"], m["loop"], m["problems"]) == (0.5, 0.5, 0.0, 2)
    done = json.loads((rd / "rollouts.done").read_text())
    assert done == {"served": str(start), "fresh_problems": 3, "fresh_sampled": 3, "rollout_tokens_cap": 0,
                    "fresh_tokens": sum(len(x["tokens"]) for x in fresh), "capped": False}

    st = r.round()  # the rerun resumes with the trainer: nothing sampled again
    assert r.asked() == asked and r.trains == 2 and r.served == [start / "serve"]
    assert st["trained"] == 1 and st["session"] == 2 and st["best"] == str(start) and st["best_held"]["kl"] > 0
    cand = Path(st["candidate"])
    assert (cand / "serve" / "config.json").exists() and json.loads((cand / qat.META).read_text())["phase"] == "onpolicy1"
    train = json.loads((rd / "train" / "train.json").read_text())
    assert train["steps"] == 2 * 6 and train["passing"] == train["ce_rows"] == 3 and train["frozen"]
    assert train["skipped"] == 0 and "peak_gb" in train and (train["peak_gb"] is None) == (not torch.cuda.is_available())
    assert not (rd / "train" / "ckpt.pt").exists()

    st = r.round()  # session 2 judges round 1 on the same held-out problems, then trains round 2
    assert r.served[-1] == cand / "serve"
    assert json.loads((r.work / "held.json").read_text()) == sorted(r.held, key=traces.bucket)
    f1 = set(json.loads((r.work / "round1" / "problems.json").read_text()))
    f2 = set(json.loads((r.work / "round2" / "problems.json").read_text()))
    assert len(f1) == len(f2) == 3 and not f1 & f2 and not (f1 | f2) & set(r.held)
    judged = st["rounds"][0]
    assert judged["round"] == 1 and judged["kept"], judged
    assert judged["rules"]["kl"]["drop"] >= qat.KEEP_KL_DROP and st["best"] == str(cand)
    assert st["trained"] == 2 and st["session"] == 3 and not st["done"]

    st = r.round()  # session 3: the rounds are used up, so held-out problems only, then the verdict
    assert not (r.work / "round3" / "problems.json").exists() and len(r.served) == 3
    assert st["done"] and len(st["rounds"]) == 2 and st["final"] == st["best"]
    c2 = str(r.served[-1].parent)
    assert st["final"] == (c2 if st["rounds"][1]["kept"] else str(cand))
    assert r.round() == st  # done: nothing more


def test_onpolicy_until_session_runs_one_session_per_call(tmp_path, monkeypatch):
    """--until-session N stops once session N is done, so a caller can upload after each
    round; a rerun continues, and only the call that ends the rounds writes onpolicy.json."""
    work = tmp_path / "w"
    qat.init_onpolicy(work, tmp_path / "start", qat.OnPolicy(teacher="t", problems="p", rounds=2))
    ran = []

    def one(w):  # a session: trains a round, or (rounds used up) only judges and ends the run
        st = qat.read_state(w)
        ran.append(st["session"])
        if st["trained"] >= 2:
            st.update(done=True, final=st["best"])
        else:
            st.update(trained=st["trained"] + 1, session=st["session"] + 1)
        qat.write_state(w, st)
        return st

    monkeypatch.setattr(qat, "onpolicy_round", one)

    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["qat", "onpolicy", "--work", str(work), *map(str, args)])
        qat.main()

    for k in (1, 2):
        cli("--until-session", k)
        assert ran == list(range(1, k + 1)) and not (work / "onpolicy.json").exists()
    cli("--until-session", 2)  # already past session 2: nothing runs
    assert ran == [1, 2]
    cli("--until-session", 3)
    assert ran == [1, 2, 3] and json.loads((work / "onpolicy.json").read_text())["final"] == str(tmp_path / "start")
    cli()  # done: nothing more
    assert ran == [1, 2, 3]


def test_onpolicy_stops_after_two_rounds_in_a_row_not_kept(rounds, monkeypatch):
    real = qat.keep_round
    monkeypatch.setattr(qat, "keep_round", lambda new, best: real(new, best, kl_drop=10.0))  # nothing is kept
    r = rounds(rounds=4, epochs=1)
    st = r.round()
    c1 = st["candidate"]
    st = r.round()  # round 1 not kept: the start model trains again, on round 1's rollouts
    assert st["streak"] == 1 and st["best"] == str(r.start) and st["candidate"] != c1 and not st["done"]
    assert json.loads((r.work / "round2" / "judge.json").read_text())["served"] == c1
    st = r.round()
    assert st["done"] and st["streak"] == 2 and st["final"] == str(r.start) and st["trained"] == 2
    assert [x["kept"] for x in st["rounds"]] == [False, False] and len(r.served) == 3
