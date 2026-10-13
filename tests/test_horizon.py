"""stages/horizon.py: the exact byte table, the knapsack, the teacher cache and the sweep.

The end-to-end test runs the real tools on a tiny Granite: convert_hf_to_gguf.py,
llama-quantize per type and for the chosen map, gguf-py's dequantization, the sweep,
verify and llama-perplexity. It skips when llama.cpp is not built."""
import copy
import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

from stages import horizon, qheal, traces  # noqa: E402
from stages._util import Config  # noqa: E402
from stages.sft import kd_kl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("the", "a", "cat", "tree", "zebra", "quick", "jumps", "over", "lazy", "dog", "print", "sum", "for", "in")


# ---------- a tiny Granite with a llama-style byte-fallback tokenizer ----------

def tiny_granite(d: Path, layers: int = 2, seed: int = 0):
    """GraniteForCausalLM with K-quant-friendly rows (256, 512) and a BPE tokenizer with
    byte fallback, which convert_hf_to_gguf.py writes as a llama (SPM) vocab that
    llama-perplexity can tokenize text with."""
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast

    d.mkdir(parents=True, exist_ok=True)
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = len(vocab)
    for ch in "▁abcdefghijklmnopqrstuvwxyz":
        vocab[ch] = len(vocab)
    merges = [("▁", "t"), ("▁", "a"), ("▁", "c"), ("▁", "d")]
    for a, b in merges:
        vocab[a + b] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tk.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tk.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                    decoders.Strip(" ", 1, 0)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tok.save_pretrained(d)
    torch.manual_seed(seed)
    cfg = GraniteConfig(vocab_size=len(vocab), hidden_size=256, intermediate_size=512, num_hidden_layers=layers,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125,
                        logits_scaling=1.0, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2)
    m = GraniteForCausalLM(cfg)
    m.save_pretrained(d)
    return m, tok


def word_traces(path: Path, tok, problems: int = 600, samples: int = 2, seed: int = 0) -> None:
    """Trace rows in gen's format whose token ids are real text of the tiny tokenizer."""
    rng = random.Random(seed)
    with open(path, "w") as f:
        for i in range(problems):
            for _ in range(samples):
                prompt = tok(" ".join(rng.choice(WORDS) for _ in range(4)))["input_ids"]
                answer = tok(" ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 20))))["input_ids"]
                f.write(json.dumps({"id": f"cc/{i}", "prompt": prompt, "tokens": answer, "finish": "stop",
                                    "passed": None, "forced": None}) + "\n")


def find_llama_cpp() -> Path | None:
    for c in (os.environ.get("LOBBOT_LLAMA_CPP", ""), Config().llama_cpp, ROOT.parent / "llama.cpp"):
        if c and all((Path(c) / "build" / "bin" / x).exists() for x in ("llama-quantize", "llama-perplexity")):
            return Path(c)
    return None


def convert(hf: Path, out: Path, llama_cpp: Path, tmp: Path) -> None:
    """convert_hf_to_gguf.py to BF16. Without the sentencepiece package the converter
    stops at its import before it would look for (and not find) tokenizer.model, so a
    stand-in module lets it go on to the tokenizer.json vocab."""
    env = dict(os.environ)
    if importlib.util.find_spec("sentencepiece") is None:
        stub = tmp / "stub" / "sentencepiece"
        stub.mkdir(parents=True, exist_ok=True)
        (stub / "__init__.py").write_text("class SentencePieceProcessor:\n    pass\n")
        env["PYTHONPATH"] = os.pathsep.join([str(stub.parent), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf), "--outtype", "bf16",
                        "--outfile", str(out)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        pytest.skip(f"convert_hf_to_gguf.py failed here: {r.stderr[-500:]}")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The whole step on a tiny Granite, budget halfway between the floor and all-Q4_K."""
    llama_cpp = find_llama_cpp()
    if llama_cpp is None:
        pytest.skip("llama.cpp's llama-quantize and llama-perplexity are not built")
    root = tmp_path_factory.mktemp("horizon")
    hf, out = root / "hf", root / "out"
    _, tok = tiny_granite(hf)
    bf16 = root / "bf16.gguf"
    convert(hf, bf16, llama_cpp, root)
    word_traces(root / "traces.jsonl", tok)
    # the budget needs the fixed bytes, which need the PTQ files; run() reuses them
    shapes, align = horizon.gguf_info(bf16, gguf)
    table = horizon.byte_table(shapes, horizon.TYPES, align)
    files = horizon.ptq(bf16, out / "ptq", horizon.TYPES, list(shapes), llama_cpp, gguf, table=table)
    fixed = horizon.fixed_bytes(files, table)
    gb = horizon.group_bytes(table, horizon.groups(shapes))
    floor = fixed + sum(v[horizon.HEAD_MIN if g == "output" else "q2_k"] for g, v in gb.items())
    top = fixed + sum(v["q4_k"] for v in gb.values())
    budget = (floor + top) // 2
    summary = horizon.run(str(hf), bf16, root / "traces.jsonl", out, llama_cpp, budget,
                          sweep_window=(48, 400), check_window=(48, 400), ladder_window=(64, 4), workers=0)
    return {"root": root, "hf": hf, "out": out, "bf16": bf16, "llama_cpp": llama_cpp, "summary": summary,
            "budget": budget, "floor": floor, "top": top, "fixed": fixed, "table": table, "files": files}


# ---------- the byte table against llama-quantize ----------

def test_byte_table_is_exact_against_llama_quantize(world):
    table, fixed = world["table"], world["fixed"]
    for t, p in world["files"].items():
        r = gguf.GGUFReader(str(p))
        for x in r.tensors:
            if horizon.quantizable(x.name, len(x.shape)):
                assert x.tensor_type.name.lower() == t
                assert -(-int(x.n_bytes) // 32) * 32 == table[x.name][t], x.name
        assert p.stat().st_size == fixed + sum(row[t] for row in table.values())
    # the mixed map's file: every type as planned and the size to the byte
    m = json.loads((world["out"] / "map.json").read_text())
    path = Path(world["summary"]["gguf"])
    assert path.stat().st_size == m["bytes"] == horizon.predict_bytes(m["types"], table, fixed)
    got = {x.name: x.tensor_type.name.lower() for x in gguf.GGUFReader(str(path)).tensors
           if horizon.quantizable(x.name, len(x.shape))}
    assert got == m["types"]
    assert len(set(got.values())) > 1  # a real mix, not one type


def test_sweep_measures_every_swap_and_upgrades_help(world):
    lines = [json.loads(x) for x in (world["out"] / "sweep.jsonl").read_text().splitlines()]
    head, rows = lines[0], lines[1:]
    base = head["header"]["base"]
    assert base["output"] == "q4_k" and base["token_embd"] == "q2_k" and base["blk.1.k_v"] == "q2_k"
    want = {(g, t) for g, ts in head["header"]["choices"].items() for t in ts if t != base[g]}
    assert {(r["group"], r["type"]) for r in rows} == want
    assert len(rows) == 9 * 2 + 2  # 2 layers x 4 groups + embedding, 2 alternatives each; the head's two cuts
    up = [r for r in rows if r["group"] != "output" and r["type"] == "q4_k"]
    assert sum(r["gain"] > 0 for r in up) >= 0.75 * len(up) and sum(r["gain"] for r in up) > 0
    cut = [r for r in rows if r["group"] == "output"]
    assert all(r["gain"] < 0 for r in cut)  # cutting the head below Q4_K costs KL
    assert all(r["se"] >= 0 for r in rows)


def test_the_map_is_within_budget_and_verified(world):
    s, m = world["summary"], json.loads((world["out"] / "map.json").read_text())
    assert world["floor"] <= m["bytes"] <= world["budget"]
    assert m["groups"]["output"] == "q4_k"  # never below head_min
    assert m["exact"] and m["pred_kl"] <= m["base_kl"]
    v = s["verify"]
    assert v["identical_tensors"] == len(m["types"])  # the swept weights are the shipped ones
    assert s["true_kl"] < s["floor_kl"]  # the bytes bought KL
    assert s["check_kl"] > 0 and abs(s["true_kl"] - s["pred_kl"]) < 0.5 * (s["floor_kl"] - s["pred_kl"]) + 1e-3
    assert set(m["hf_types"]) == {qheal.hf_name(n) for n in m["types"]}
    assert m["curve"][0]["pred_kl"] == pytest.approx(m["base_kl"])
    assert all(a["pred_kl"] >= b["pred_kl"] - 1e-12 for a, b in zip(m["curve"], m["curve"][1:]))


def test_ladder_orders_the_types(world):
    lad = json.loads((world["out"] / "ladder.json").read_text())
    by = {Path(r["gguf"]).stem: r for r in lad}
    assert set(by) >= {"q2_k", "q3_k", "q4_k"} and len(lad) == 4
    assert by["q4_k"]["kld"] < by["q2_k"]["kld"]
    assert all(0 <= r["top1"] <= 1 and r["kld_se"] >= 0 and r["ctx"] == 64 for r in lad)
    assert [r["bytes"] for r in lad] == sorted(r["bytes"] for r in lad)


def test_a_kld_run_that_lost_its_summary_runs_again(monkeypatch):
    """llama-perplexity's log is asynchronous and not flushed at exit: a run that exits 0
    can end before its summary. It runs again; a run that fails, or keeps losing it, raises."""
    full = "   4     289.7799 ±    7.9754\nMean    KLD:  -0.000010 ±   0.000002\nSame top p: 98.500 ± 0.300 %\n"
    outs = iter(["   1     301.7 ±   17.7\n", full])
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, next(outs), "")
    monkeypatch.setattr(horizon.subprocess, "run", run)
    r = horizon._kld_run(["exe", "-m", "x.gguf", "--kl-divergence"])
    assert len(calls) == 2 and r["kld"] == pytest.approx(-1e-5) and r["top1"] == pytest.approx(0.985)
    monkeypatch.setattr(horizon.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "1 2\n", ""))
    with pytest.raises(ValueError, match="no KLD summary"):
        horizon._kld_run(["exe", "-m", "x.gguf"], tries=2)
    monkeypatch.setattr(horizon.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="x.gguf exited with 1"):
        horizon._kld_run(["exe", "-m", "x.gguf"])


def test_knapsack_cli_replans_and_a_rerun_reuses_everything(world, monkeypatch, capsys):
    from unittest import mock
    out = world["out"]
    with mock.patch.object(sys, "argv", ["x", "knapsack", "--out", str(out), "--budget-gb", str(world["floor"] / 1e9)]):
        horizon.main()
    m = json.loads((out / "map.json").read_text())
    base = json.loads((out / "sweep.jsonl").read_text().splitlines()[0])["header"]["base"]
    assert m["groups"] == base and m["bytes"] == world["floor"]
    with pytest.raises(ValueError, match="over the budget"):
        horizon.plan(out, world["floor"] - 1)
    # a second run measures nothing again: no swap forward passes, no quantizing
    calls = []
    monkeypatch.setattr(horizon, "quantize", lambda *a, **k: calls.append(a))
    real = horizon.kl_positions
    monkeypatch.setattr(horizon, "kl_positions", lambda *a, **k: calls.append("kl") or real(*a, **k))
    horizon.run(str(world["hf"]), world["bf16"], world["root"] / "traces.jsonl", out, world["llama_cpp"],
                world["budget"], sweep_window=(48, 400), check_window=(48, 400), do_ladder=False, workers=0)
    assert calls == ["kl", "kl", "kl"]  # the floor once, then verify's two KLs; no swap, no llama-quantize
    # a file made from other inputs (another imatrix, binary or BF16) is made again
    key = out / "ptq" / "q3_k.key.json"
    key.write_text(json.dumps({**json.loads(key.read_text()), "imatrix": {"path": "other.gguf"}}))
    calls.clear()
    shapes, _ = horizon.gguf_info(world["bf16"], gguf)
    horizon.ptq(world["bf16"], out / "ptq", horizon.TYPES, list(shapes), world["llama_cpp"], gguf)
    assert len(calls) == 1 and calls[0][1] == out / "ptq" / "q3_k.gguf"


# ---------- the knapsack ----------

def brute(table, budget):
    gs = list(table)
    best = None
    for combo in itertools.product(*(table[g].items() for g in gs)):
        b = sum(v[0] for _, v in combo)
        if b <= budget:
            gain = sum(v[1] for _, v in combo)
            best = gain if best is None else max(best, gain)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_knapsack_is_optimal_on_small_cases(seed):
    rng = random.Random(seed)
    unit = rng.choice([1, 7, 32, 1000])
    table = {}
    for g in range(rng.randint(1, 6)):
        n = rng.randint(1, 4)
        costs = sorted(rng.sample(range(1, 40), n))
        table[f"g{g}"] = {f"t{k}": (c * unit, rng.uniform(-1, 3) if k else 0.0) for k, c in enumerate(costs)}
    floor = sum(min(v[0] for v in opts.values()) for opts in table.values())
    top = sum(max(v[0] for v in opts.values()) for opts in table.values())
    for budget in (floor, (floor + top) // 2, floor + rng.randint(0, top - floor), top, top + 5):
        m = horizon.knapsack(table, budget)
        assert set(m) == set(table)
        assert sum(table[g][t][0] for g, t in m.items()) <= budget
        assert sum(table[g][t][1] for g, t in m.items()) == pytest.approx(brute(table, budget))


def test_knapsack_coarse_units_stay_in_budget_and_near_optimal():
    rng = random.Random(1)
    table = {f"g{g}": {f"t{k}": (rng.randint(1, 10 ** 6) * 13 + k, rng.uniform(0, 1)) for k in range(3)}
             for g in range(5)}
    floor = sum(min(v[0] for v in o.values()) for o in table.values())
    budget = floor + 2 * 10 ** 6
    r = horizon._knapsack(table, budget, max_states=500)
    assert not r["exact"]
    m = r["map"]
    assert sum(table[g][t][0] for g, t in m.items()) <= budget
    assert sum(table[g][t][1] for g, t in m.items()) >= brute(table, budget) - 0.5


def test_knapsack_refuses_a_budget_below_the_floor_and_prefers_fewer_bytes():
    table = {"a": {"x": (10, 0.0), "y": (20, 1.0)}, "b": {"x": (10, 0.0), "y": (30, 0.0)}}
    with pytest.raises(ValueError, match="over the budget"):
        horizon.knapsack(table, 19)
    assert horizon.knapsack(table, 100) == {"a": "y", "b": "x"}  # b's upgrade gains nothing


def test_knapsack_table_keeps_the_head_at_its_minimum():
    gb = {"blk.0.down": {"q2_k": 84, "q3_k": 110, "q4_k": 144}, "output": {"q2_k": 84, "q3_k": 110, "q4_k": 144}}
    rows = [{"group": "blk.0.down", "type": "q3_k", "gain": 0.1}, {"group": "blk.0.down", "type": "q4_k", "gain": 0.2},
            {"group": "output", "type": "q2_k", "gain": -0.5}, {"group": "output", "type": "q3_k", "gain": -0.1}]
    base = {"blk.0.down": "q2_k", "output": "q4_k"}
    t = horizon.knapsack_table(gb, rows, base)
    assert t["output"] == {"q4_k": (144, 0.0)} and t["blk.0.down"]["q4_k"] == (144, 0.2)
    assert set(horizon.knapsack_table(gb, rows, base, head_min="")["output"]) == {"q2_k", "q3_k", "q4_k"}


# ---------- groups, flags, bytes ----------

def test_groups_flags_and_byte_table():
    names = ["token_embd.weight", "output.weight"] + [f"blk.{l}.{k}.weight" for l in (0, 10) for k in
                                                      ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
                                                       "ffn_up", "ffn_down")]
    g = horizon.groups(names)
    assert list(g)[:5] == ["token_embd", "blk.0.gate_up", "blk.0.down", "blk.0.q_o", "blk.0.k_v"]
    assert list(g)[-1] == "output" and g["blk.10.k_v"] == ["blk.10.attn_k.weight", "blk.10.attn_v.weight"]
    with pytest.raises(ValueError, match="attn_qkv"):
        horizon.groups(names + ["blk.0.attn_qkv.weight"])
    f = horizon.flags({"blk.1.ffn_up.weight": "q3_k"})
    assert f == ["--pure", "--tensor-type", r"^blk\.1\.ffn_up\.weight$=q3_k"]
    f = horizon.flags({"output.weight": "q4_k", "token_embd.weight": "q2_k", "blk.0.attn_output.weight": "q3_k"})
    assert f[:5] == ["--pure", "--output-tensor-type", "q4_k", "--token-embedding-type", "q2_k"]
    assert r"^output\.weight$=q4_k" in f and r"^blk\.0\.attn_output\.weight$=q3_k" in f
    t = horizon.byte_table({"a": (8192, 2560), "b": (49152, 2560), "c": (7, 300)}, ("q2_k", "q3_k", "q4_k"))
    assert t["a"] == {"q2_k": 8192 * 10 * 84, "q3_k": 8192 * 10 * 110, "q4_k": 8192 * 10 * 144}
    assert t["b"]["q2_k"] == 49152 * 10 * 84
    assert t["c"] == {}  # 300 is not whole 256-blocks: llama-quantize would fall back
    t = horizon.byte_table({"d": (3, 256)}, ("q3_k",))
    assert t["d"]["q3_k"] == 352  # 330 bytes padded to the 32-byte alignment


def test_check_names_every_difference(tmp_path):
    w = gguf.GGUFWriter(str(tmp_path / "x.gguf"), "llama")
    a = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)
    w.add_tensor("blk.0.ffn_up.weight", gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(256, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    p = tmp_path / "x.gguf"
    assert horizon.check(p, {"blk.0.ffn_up.weight": "q8_0"}, gguf)["types"] == {"q8_0": 1}
    with pytest.raises(ValueError, match="planned q2_k, written q8_0"):
        horizon.check(p, {"blk.0.ffn_up.weight": "q2_k"}, gguf)
    with pytest.raises(ValueError, match="blk.0.ffn_down.weight: planned q2_k, written None"):
        horizon.check(p, {"blk.0.ffn_up.weight": "q8_0", "blk.0.ffn_down.weight": "q2_k"}, gguf)
    with pytest.raises(ValueError, match="byte table says"):
        horizon.check(p, {"blk.0.ffn_up.weight": "q8_0"}, gguf, expect_bytes=p.stat().st_size + 32)


def test_bf16_bits_round_as_torch_does():
    a = np.concatenate([np.random.default_rng(0).standard_normal(100000).astype(np.float32) * 10 ** k
                        for k in (-6, 0, 6)] + [np.array([0.0, -0.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8], np.float32)])
    want = torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(horizon.bf16_bits(a), want)


def test_sweep_and_check_windows_are_different_held_out_problems(tmp_path):
    id_traces(tmp_path / "tr.jsonl", problems=1000)
    rows = traces.load(tmp_path / "tr.jsonl")
    held = {r["id"] for r in traces.split(rows)[1]}
    a = {r["id"] for r in horizon.held_rows(tmp_path / "tr.jsonl", "sweep")}
    b = {r["id"] for r in horizon.held_rows(tmp_path / "tr.jsonl", "check")}
    assert a and b and not a & b and a | b == held
    assert 0.25 < len(a) / len(held) < 0.75


def test_peak_memory_fits_an_l40s():
    trimmed = horizon.peak_gb(3.40e9, 3.66e9, 49152, teacher_vocab=100352)
    full = horizon.peak_gb(3.66e9, 3.66e9, 100352)
    assert max(trimmed.values()) < 15 and max(full.values()) < 20  # of 48 GB


# ---------- teacher cache and the sweep without llama.cpp ----------

def tiny_hf(d, vocab=64, seed=0):
    from transformers import GraniteConfig, GraniteForCausalLM
    torch.manual_seed(seed)
    cfg = GraniteConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125,
                        tie_word_embeddings=False)
    m = GraniteForCausalLM(cfg)
    m.save_pretrained(d)
    return m


def id_traces(path, problems=200, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for i in range(problems):
            for _ in range(2):
                t = torch.randint(0, vocab, (30,), generator=g).tolist()
                f.write(json.dumps({"id": f"p{i}", "prompt": t[:6], "tokens": t[6:], "finish": "stop",
                                    "passed": None, "forced": None}) + "\n")


def test_teacher_cache_over_the_kept_vocab_is_kd_kl_with_an_index(tmp_path):
    """A trimmed student against the untrimmed teacher: the cached log-probs give the same
    KL as sft.kd_kl(index=kept) on the teacher's own logits, dropped targets weigh 0."""
    from transformers import GraniteForCausalLM
    teacher = tiny_hf(tmp_path / "t")
    id_traces(tmp_path / "tr.jsonl")
    kept = sorted(random.Random(0).sample(range(64), 40))
    (tmp_path / "vi.json").write_text(json.dumps({"kept": kept, "old_to_new": {str(o): n for n, o in enumerate(kept)}}))
    held = horizon.held_out(tmp_path / "tr.jsonl", str(tmp_path / "t"), length=24, tokens=120,
                            vocab_index=str(tmp_path / "vi.json"), role="sweep", path=tmp_path / "c.pt")
    assert held.positions >= 120 and held.logp[0].dtype == torch.float16 and held.logp[0].shape[-1] == 40
    # the trimmed student: the teacher's rows of the kept ids, plus noise
    cfg = copy.deepcopy(teacher.config)
    cfg.vocab_size = 40
    student = GraniteForCausalLM(cfg)
    sd = {k: v.clone() for k, v in teacher.state_dict().items()}
    sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"][kept]
    sd["lm_head.weight"] = sd["lm_head.weight"][kept] + 0.05 * torch.randn(40, 64)
    student.load_state_dict(sd)
    student.eval()
    kl, w = horizon.kl_positions(student, held)
    # the reference, batch by batch from the raw teacher logits
    rows = horizon.held_rows(tmp_path / "tr.jsonl", "sweep")
    num = den = 0.0
    used, dropped = 0, 0
    vmap = json.loads((tmp_path / "vi.json").read_text())
    with torch.no_grad():
        for tb, sb in zip(traces.batches(rows, 24, 1, prompt_weight=0.1),
                          traces.batches(rows, 24, 1, vocab_map=vmap, prompt_weight=0.1)):
            if used == len(held.logp):
                break
            keep = sb[2] > 0
            if not keep.any():
                continue
            gone = ~torch.isin(tb[0][:, 1:], torch.tensor(kept)) & (tb[1][:, 1:] > 0)
            assert not keep[gone].any()  # a target the trimmed vocab lacks weighs 0
            dropped += int(gone.sum())
            t = teacher(input_ids=tb[0], attention_mask=tb[1]).logits[:, :-1][keep]
            s = student(input_ids=sb[0], attention_mask=sb[1]).logits[:, :-1][keep]
            num += float(kd_kl(s, t, index=torch.tensor(kept), weights=sb[2][keep])) * float(sb[2][keep].sum())
            den += float(sb[2][keep].sum())
            used += 1
    assert dropped > 0
    assert horizon.mean_kl(kl, w) == pytest.approx(num / den, rel=2e-3)  # fp16 cache
    assert float(w.sum()) == pytest.approx(den)
    again = horizon.held_out(tmp_path / "tr.jsonl", str(tmp_path / "t"), length=24, tokens=120,
                             vocab_index=str(tmp_path / "vi.json"), role="sweep", path=tmp_path / "c.pt")
    assert all(torch.equal(a, b) for a, b in zip(again.logp, held.logp))  # read back from the cache


class FakeCache:
    """Per type: the original weights plus noise of the type's size, one noise direction
    per tensor so a coarser type is strictly further off (for the sweep alone)."""

    def __init__(self, model, scale):
        self.w = {}
        g = torch.Generator().manual_seed(0)
        for n, p in model.named_parameters():
            if p.dim() == 2:
                noise = p.detach().std() * torch.randn(p.shape, generator=g)
                for t, s in scale.items():
                    self.w[(t, n)] = p.detach().clone() + s * noise

    def get(self, t, gguf_name):
        return self.w[(t, qheal.hf_name(gguf_name))]


def test_sweep_swaps_one_group_restores_it_and_resumes(tmp_path, monkeypatch):
    m = tiny_hf(tmp_path / "m").eval()
    id_traces(tmp_path / "tr.jsonl")
    held = horizon.held_out(tmp_path / "tr.jsonl", str(tmp_path / "m"), length=24, tokens=200, batch=8,
                            role="sweep")
    names = ["token_embd.weight", "output.weight"] + [f"blk.{l}.{k}.weight" for l in range(2) for k in
                                                      ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
                                                       "ffn_up", "ffn_down")]
    grp = horizon.groups(names)
    cache = FakeCache(m, {"q2_k": 0.6, "q3_k": 0.3, "q4_k": 0.1})
    choices, base = horizon.plan_choices(grp, horizon.TYPES)
    out = tmp_path / "sweep.jsonl"
    res = horizon.sweep(m, cache, held, grp, choices, base, out)
    assert len(res["rows"]) == 9 * 2 + 2
    gain = {(r["group"], r["type"]): r["gain"] for r in res["rows"]}
    for g in grp:
        if g != "output":
            assert gain[(g, "q4_k")] > gain[(g, "q3_k")] > 0
    assert gain[("output", "q2_k")] < gain[("output", "q3_k")] < 0
    # every group is back at its floor type
    for g, ns in grp.items():
        for n in ns:
            assert torch.equal(dict(m.named_parameters())[qheal.hf_name(n)], cache.get(base[g], n))
    # a restart measures only the floor again
    calls = []
    real = horizon.kl_positions
    monkeypatch.setattr(horizon, "kl_positions", lambda *a, **k: calls.append(1) or real(*a, **k))
    again = horizon.sweep(m, cache, held, grp, choices, base, out)
    assert len(calls) == 1 and again["rows"] == res["rows"]
    # an unfinished sweep is not planned from
    part = tmp_path / "part"
    part.mkdir()
    (part / "sweep.jsonl").write_text("\n".join(out.read_text().splitlines()[:-3]) + "\n")
    (part / "bytes.json").write_text("{}")
    with pytest.raises(ValueError, match="3 of 20 swaps not measured"):
        horizon.plan(part, 10 ** 9)
    # the knapsack on these gains, through plan(): the same bytes for every tensor here
    info = {"fixed": 1000, "groups": grp, "tensors": {n: {"q2_k": 84, "q3_k": 110, "q4_k": 144} for n in names}}
    info["group_bytes"] = horizon.group_bytes(info["tensors"], grp)
    (tmp_path / "bytes.json").write_text(json.dumps(info))
    m2 = horizon.plan(tmp_path, 1000 + sum(min(v.values()) for v in info["group_bytes"].values()) - 84 + 144 + 52)
    assert m2["groups"]["output"] == "q4_k" and sum(t != base[g] for g, t in m2["groups"].items()) >= 1
    assert m2["bytes"] <= m2["budget_bytes"]
