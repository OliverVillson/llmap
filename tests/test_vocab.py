import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from stages import codebench, opd, traces, vocab  # noqa: E402
from stages.sft import kd_kl  # noqa: E402

LLAMA_CPP = Path(os.environ.get("LOBBOT_LLAMA_CPP", "/home/claude/llama.cpp"))
# The pre-tokenizer of a tiktoken-style BPE (Granite's 100k vocab is one): cl100k's
# regex, then byte-level mapping without its own regex.
CL100K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
          r"|\s*[\r\n]+|\s+(?!\S)|\s+")
SPECIAL = ["<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>", "<|pad|>"]
TEMPLATE = (
    "{%- for m in messages %}<|start_of_role|>{{ m['role'] }}<|end_of_role|>{{ m['content'] }}<|end_of_text|>\n"
    "{% endfor %}"
    "{%- if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>"
    "{%- if enable_thinking is defined and enable_thinking %}<think>\n{% endif %}{% endif %}")
CODE = ["def solve(a, b):\n    return a + b\n", "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n",
        "for i in range(10):\n    total += i * i\nprint(total)\n", codebench.SYSTEM,
        "Read an integer n and print twice its value. The first line holds n.",
        "Let me think about the input format and the edge cases first. I'll check 12345 and 678.",
        "Wait, that's wrong: we need the sum, not the product. So the answer is 15."]
# Text the code never uses, so a trim drops its tokens.
RARE = ["The zeppelin flew over the aardvark colony at dawn.", "Привет, мир! Это тест.", "naïve café über straße",
        "quokka quokka quokka wombat wombat"]


def tiktoken_merges(tk):
    """The trained tokenizer with its merges rebuilt the way transformers converts a
    tiktoken vocab (Granite's 100k vocab came that way): every split of every token into
    two tokens is a merge at the token's rank. Ranks then stop rising along a merge
    path, which is where a naive BPE replica goes wrong."""
    from tokenizers import Tokenizer
    tj = json.loads(tk.to_str())
    v = tj["model"]["vocab"]
    merges = []
    for t, r in sorted(v.items(), key=lambda kv: kv[1]):
        merges += sorted(((t[:k], t[k:], r) for k in range(1, len(t)) if t[:k] in v and t[k:] in v),
                         key=lambda m: (v[m[0]], v[m[1]]))
    tj["model"]["merges"] = [[a, b] for a, b, _ in sorted(merges, key=lambda m: m[2])]
    return Tokenizer.from_str(json.dumps(tj))


def toy_tokenizer(d, holes=3, style="trained"):
    """A byte-level BPE with Granite's layout: cl100k-style pre-tokenizer, special
    tokens and <think>/</think> added after the trained vocab (high ids), a Granite-style
    chat template, and `holes` unused rows in the model's vocab_size. style "tiktoken"
    swaps in tiktoken-converted merges."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(CL100K), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)
    tk.train_from_iterator((CODE + RARE) * 30, trainers.BpeTrainer(
        vocab_size=700, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    if style == "tiktoken":
        tk = tiktoken_merges(tk)
    tk.add_special_tokens(SPECIAL)
    tk.add_tokens(["<think>", "</think>"])
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    tk.save(str(d / "tokenizer.json"))
    ids = {a: tk.token_to_id(a) for a in SPECIAL + ["<think>", "</think>"]}
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2Tokenizer", "bos_token": "<|end_of_text|>", "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>", "unk_token": "<|end_of_text|>", "add_bos_token": False, "chat_template": TEMPLATE,
        "clean_up_tokenization_spaces": False,
        "added_tokens_decoder": {str(i): {"content": a, "special": a in SPECIAL, "lstrip": False, "rstrip": False,
                                          "normalized": False, "single_word": False} for a, i in ids.items()}}))
    return tk.get_vocab_size() + holes, ids


def toy(d, holes=3, tie=False, seed=0, style="trained"):
    """The toy tokenizer plus a tiny GraniteForCausalLM (tests/test_ternary.py's shape)
    whose config and generation_config name the special ids, saved to d."""
    from transformers import GraniteConfig, GraniteForCausalLM
    n, ids = toy_tokenizer(d, holes, style)
    torch.manual_seed(seed)
    cfg = GraniteConfig(vocab_size=n, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                        residual_multiplier=0.5, embedding_multiplier=2.0, attention_multiplier=0.125,
                        logits_scaling=2.0, tie_word_embeddings=tie, bos_token_id=ids["<|end_of_text|>"],
                        eos_token_id=ids["<|end_of_text|>"], pad_token_id=ids["<|pad|>"])
    m = GraniteForCausalLM(cfg).eval()
    m.save_pretrained(d)
    gc = json.loads((Path(d) / "generation_config.json").read_text())
    gc["eos_token_id"] = [ids["<|end_of_text|>"], ids["<|end_of_role|>"]]
    (Path(d) / "generation_config.json").write_text(json.dumps(gc))
    return m, ids


def tok_of(d):
    return transformers.AutoTokenizer.from_pretrained(str(d))


def code_kept(d, extra=()):
    tok = tok_of(d)
    ids = [i for t in CODE + list(extra) for i in tok.encode(t, add_special_tokens=False)]
    return vocab.choose(vocab.tok_json(d), vocab._bincount(ids, len(tok) + 8))


def samples(seed=0, n=300):
    """Pieces of the corpora, spliced: some use only code tokens, some do not."""
    rng = random.Random(seed)
    src = "\n".join(CODE + RARE + vocab.EDGE_TEXTS)
    out = []
    for _ in range(n):
        a = rng.randrange(len(src))
        out.append(src[a:a + rng.randrange(1, 120)] + (rng.choice(CODE)[:rng.randrange(40)] if rng.random() < 0.3 else ""))
    return out


def rows_of(tok, n=6, answer=None):
    """Trace rows: a rendered prompt and an answer of token ids, two samples a problem."""
    out = []
    for k in range(n):
        p = {"id": f"cc/{k}", "language": "python", "prompt": RARE[k % len(RARE)] if k % 3 == 2 else CODE[k % 3],
             "stub": "", "entry": "", "tests": []}
        prompt = opd.render(tok, p)
        for s in range(2):
            text = answer or (CODE[(k + s) % 3] + " " + CODE[5] + "</think>\n\n```python\n" + CODE[1] + "```")
            out.append({"id": p["id"], "prompt": prompt, "tokens": tok.encode(text, add_special_tokens=False),
                        "finish": "stop", "passed": True, "forced": None})
    return out


# ---------- choosing and the merge-ancestor closure ----------

@pytest.mark.parametrize("style", ["trained", "tiktoken"])
def test_bpe_replica_merges_words_exactly_as_tokenizers_does(tmp_path, style):
    from tokenizers import Tokenizer
    toy_tokenizer(tmp_path, style=style)
    raw = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    bpe = vocab.BPE(vocab.tok_json(tmp_path))
    words = {w for text in samples(2, 400) + CODE + RARE + vocab.EDGE_TEXTS
             for w, _ in raw.pre_tokenizer.pre_tokenize_str(text)}
    words |= set(bpe.vocab)  # every token's own bytes too
    for w in words:
        assert bpe.word(w)[0] == [t.value for t in raw.model.tokenize(w)], w
    if style == "tiktoken":  # merges whose half is a newer token than their result do occur
        assert any(max(bpe.vocab[a], bpe.vocab[b]) > bpe.vocab[a + b] for a, b in bpe.merges)


def test_bpe_replica_follows_tokenizers_when_a_merge_makes_a_better_pair():
    # "bc"+"b" ranks before "b"+"c": after the first "b"+"c" merge, tokenizers takes the
    # new pair next; merging every "b"+"c" first would give ["bc", "bc"]
    from tokenizers import Tokenizer, models, pre_tokenizers
    v = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    v.update(bc=len(v), bcb=len(v) + 1)
    tk = Tokenizer(models.BPE(vocab=v, merges=[("bc", "b"), ("b", "c")]))
    bpe = vocab.BPE(json.loads(tk.to_str()))
    assert [t.value for t in tk.model.tokenize("bcbc")] == ["bcb", "c"] == bpe.word("bcbc")[0]
    assert bpe.path(v["bcb"]) == {v["bc"], v["bcb"]} and bpe.closure([v["bcb"]]) == {v["bc"], v["bcb"]}


def test_choose_keeps_bytes_specials_counts_and_every_merge_path(tmp_path):
    toy_tokenizer(tmp_path)
    tj = vocab.tok_json(tmp_path)
    bpe = vocab.BPE(tj)
    kept = code_kept(tmp_path)
    ks = set(kept)
    assert kept == sorted(kept) and len(bpe.bytes) == 256 and bpe.bytes <= ks
    assert set(bpe.added) <= ks  # specials and <think>/</think>
    tok = tok_of(tmp_path)
    for t in CODE:
        assert set(tok.encode(t, add_special_tokens=False)) <= ks
    assert len(kept) < len(bpe.defined)  # the rare words' tokens are gone
    assert all(bpe.path(i) <= ks for i in kept)
    # the path is what BPE does: each kept token's own bytes tokenize to itself
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    for i in kept:
        t = bpe.token.get(i)
        if t and len(t) > 1:
            assert [x.value for x in raw.model.tokenize(t)] == [t]
            assert i in bpe.path(i)
    # probabilities keep ids the counts miss; size fills to the exact number with paths closed
    probs = torch.zeros(len(tok) + 8, dtype=torch.float64)
    rare = tok.encode(" zeppelin", add_special_tokens=False)[-1]
    probs[rare] = 1e-5
    base = vocab.choose(tj, None, probs)
    assert rare in base and rare not in ks and bpe.closure(base) == set(base)
    sized = vocab.choose(tj, None, probs, size=len(base) + 20)
    assert len(sized) == len(base) + 20 and set(base) <= set(sized) and bpe.closure(sized) == set(sized)
    with pytest.raises(ValueError, match="more than size"):
        vocab.choose(tj, vocab._bincount([], 8), size=100)


@pytest.mark.parametrize("style", ["trained", "tiktoken"])
def test_trimmed_tokenizer_matches_on_kept_text_and_round_trips_the_rest(tmp_path, style):
    src, out = tmp_path / "src", tmp_path / "out"
    toy(src, style=style)
    kept = code_kept(src)
    vocab.trim(src, out, kept, weights=False)
    a, b = tok_of(src), tok_of(out)
    assert len(b) == len(kept)
    o2n = {o: n for n, o in enumerate(kept)}
    same = changed = 0
    for text in samples() + CODE + RARE + vocab.EDGE_TEXTS:
        x, y = a.encode(text, add_special_tokens=False), b.encode(text, add_special_tokens=False)
        if all(i in o2n for i in x):
            assert y == [o2n[i] for i in x], text  # kept-vocab text: identical tokens
            same += 1
        else:
            changed += 1
        assert b.decode(y) == a.decode(x) == text  # everything round-trips
    assert same > 50 and changed > 20
    # a trim that breaks a merge path is refused
    bpe = vocab.BPE(vocab.tok_json(src))
    longest = max((i for i in kept if bpe.path(i) - {i}), key=lambda i: len(bpe.token[i]))
    hole = sorted(bpe.path(longest) - {longest})[0]
    with pytest.raises(ValueError, match="merge ancestry"):
        vocab.trim(src, tmp_path / "bad", [i for i in kept if i != hole], weights=False)
    with pytest.raises(ValueError, match="special/added"):
        vocab.trim(src, tmp_path / "bad", [i for i in kept if i not in bpe.added], weights=False)


def test_trim_remaps_special_ids_configs_and_keeps_the_chat_template(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _, ids = toy(src)
    kept = code_kept(src)
    index = vocab.trim(src, out, kept)
    o2n = {o: n for n, o in enumerate(kept)}
    assert index["kept"] == kept and {int(k): v for k, v in index["old_to_new"].items()} == o2n
    assert json.loads((out / vocab.INDEX).read_text())["kept"] == kept
    cfg, gc = json.loads((out / "config.json").read_text()), json.loads((out / "generation_config.json").read_text())
    eot = ids["<|end_of_text|>"]
    assert o2n[eot] != eot  # the specials sit after the trained vocab, so their ids move
    assert cfg["vocab_size"] == len(kept) and cfg["eos_token_id"] == cfg["bos_token_id"] == o2n[eot]
    assert cfg["pad_token_id"] == o2n[ids["<|pad|>"]]
    assert gc["eos_token_id"] == [o2n[eot], o2n[ids["<|end_of_role|>"]]]
    tc = json.loads((out / "tokenizer_config.json").read_text())
    assert {v["content"]: int(k) for k, v in tc["added_tokens_decoder"].items()} == {a: o2n[i] for a, i in ids.items()}
    a, b = tok_of(src), tok_of(out)
    assert b.eos_token_id == o2n[a.eos_token_id] and b.pad_token_id == o2n[a.pad_token_id]
    rep = vocab.special_check(a, b, o2n, src, out)
    assert rep["pass"], rep
    for m in vocab.MARKERS:
        assert rep["tokens"][m]["ok"] and rep["tokens"][m]["new"] == o2n[ids[m]]
    assert rep["template_0"] == {"same_text": True, "same_ids": True, "tokens": rep["template_0"]["tokens"]}
    assert rep["template_1"]["same_ids"]
    # the index plugs into the trainers: traces.batches maps ids, kd_kl slices the teacher
    rows = rows_of(a, n=2)
    (x, _, w), = traces.batches(rows[:1], 4096, 1, vocab_map=index)
    want = [o2n.get(i, 0) for i in (rows[0]["prompt"] + rows[0]["tokens"])]
    assert x[0].tolist() == want
    t_logits = torch.randn(5, len(a) + 3)
    loss, dropped = kd_kl(t_logits[:, kept], t_logits, index=vocab.kept_tensor(index), return_dropped=True)
    assert loss.abs() < 1e-6 and (dropped > 0).all()
    # a second trim into the same place replaces it; a foreign directory is refused
    vocab.trim(src, out, kept, weights=False)
    (tmp_path / "foreign").mkdir()
    with pytest.raises(FileExistsError):
        vocab.trim(src, tmp_path / "foreign", kept)


def test_post_processor_and_suppress_lists_are_remapped():
    o2n = {5: 0, 9: 1, 7: 2}
    pp = {"type": "Sequence", "processors": [
        {"type": "ByteLevel"},
        {"type": "TemplateProcessing", "special_tokens": {"<s>": {"id": "<s>", "ids": [9], "tokens": ["<s>"]}}},
        {"type": "RobertaProcessing", "sep": ["</s>", 7], "cls": ["<s>", 9]}]}
    new = vocab._remap_post(pp, o2n)
    assert new["processors"][1]["special_tokens"]["<s>"]["ids"] == [1]
    assert new["processors"][2]["sep"] == ["</s>", 2] and new["processors"][2]["cls"] == ["<s>", 1]
    cfg = vocab._remap_config({"eos_token_id": [5, 7], "suppress_tokens": [5, 3], "vocab_size": 10,
                               "text_config": {"pad_token_id": 9}}, o2n, "x")
    assert cfg == {"eos_token_id": [0, 2], "suppress_tokens": [0], "vocab_size": 10, "text_config": {"pad_token_id": 1}}
    with pytest.raises(ValueError, match="not a kept id"):
        vocab._remap_config({"bos_token_id": 3}, o2n, "x")


@pytest.mark.parametrize("tie", [False, True])
def test_trimmed_model_logits_equal_the_originals_on_kept_ids(tmp_path, tie):
    from transformers import GraniteForCausalLM
    src, out = tmp_path / "src", tmp_path / "out"
    m, _ = toy(src, tie=tie)
    kept = code_kept(src)
    index = vocab.trim(src, out, kept)
    assert sorted(index["weights"]) == (["model.embed_tokens.weight"] if tie else
                                        ["lm_head.weight", "model.embed_tokens.weight"])
    t = GraniteForCausalLM.from_pretrained(out).eval()
    assert t.config.vocab_size == len(kept) and t.get_input_embeddings().weight.shape[0] == len(kept)
    a, b = tok_of(src), tok_of(out)
    o2n = {o: n for n, o in enumerate(kept)}
    text = a.apply_chat_template([{"role": "user", "content": CODE[0]}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=True) + CODE[1]
    x = a.encode(text, add_special_tokens=False)
    y = b.encode(text, add_special_tokens=False)
    assert y == [o2n[i] for i in x]
    with torch.no_grad():
        lo = m(torch.tensor([x])).logits[0]
        lt = t(torch.tensor([y])).logits[0]
    assert torch.allclose(lt, lo[:, kept], atol=1e-5, rtol=0)
    # gate's trimmed NLL (the original's logits at the kept ids) is the trimmed model's
    nll = vocab._nll(m, [kept[j] for j in y], torch.tensor(kept), y)
    direct = float(torch.nn.functional.cross_entropy(lt[:-1], torch.tensor(y[1:]), reduction="sum"))
    assert abs(nll - direct) < 1e-3


# ---------- usage counts and the model gates ----------

def write_bench(d, tok):
    d.mkdir()
    lcb = [{"id": "lcb/old", "language": "python", "prompt": "Read the zeppelin count and print it doubled.",
            "date": "2024-10-03", "tests": [{"input": "1\n", "output": "2\n"}]},
           {"id": "lcb/new", "language": "python",
            "prompt": "The aardvark colony at dawn: print how many quokka and wombat pairs there are.",
            "date": "2025-02-10", "tests": {"func": "f", "cases": []}}]
    cc = [{"id": "cc/a", "language": "python", "prompt": "Print the sum of two integers given on one line.",
           "tests": [{"input": "1 2\n", "output": "3\n"}]},
          # shares a 13-word run with a LiveCodeBench statement: never counted
          {"id": "cc/copy", "language": "python",
           "prompt": "Note: The aardvark colony at dawn: print how many quokka and wombat pairs there are. Yes.",
           "tests": [{"input": "1\n", "output": "1\n"}]}]
    (d / "livecodebench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lcb))
    (d / "codecontests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cc))
    return d


def test_count_uses_traces_and_statements_but_never_the_test_window(tmp_path):
    toy(tmp_path / "m")
    tok = tok_of(tmp_path / "m")
    bench = write_bench(tmp_path / "bench", tok)
    probs = vocab.statements(bench)
    assert sorted(p["id"] for p in probs) == ["cc/a", "lcb/old"]
    with pytest.raises(ValueError, match="test window"):
        vocab.statements(bench, before="2025-03-01")
    rows = rows_of(tok, n=1)
    c = vocab.count(tok, len(tok) + 3, rows=rows, problems=probs)
    assert c.sum() == sum(len(r["prompt"]) + len(r["tokens"]) for r in rows) + sum(len(opd.render(tok, p)) for p in probs)
    zep = tok.encode(" zeppelin", add_special_tokens=False)
    assert len(zep) == 1 and c[zep[0]] > 0  # from the 2024 statement
    for word in (" aardvark", " quokka", " wombat"):  # only in the 2025 statement and its copy
        w = tok.encode(word, add_special_tokens=False)
        assert len(w) == 1 and c[w[0]] == 0, word
    window = vocab.window_prompts(tok, bench)
    assert len(window) == 1 and "aardvark" in window[0] and window[0].endswith("<think>\n")


def test_probe_is_the_mean_answer_distribution(tmp_path):
    m, _ = toy(tmp_path, holes=0)
    tok = tok_of(tmp_path)
    rows = rows_of(tok, n=2)
    p = vocab.probe(m, rows, length=4096, windows=2)
    assert p.shape == (len(tok),) and abs(float(p.sum()) - 1) < 1e-6
    want, n = torch.zeros(len(tok), dtype=torch.float64), 0
    with torch.no_grad():
        for r in rows[:2]:
            ids = r["prompt"] + r["tokens"]
            lg = m(torch.tensor([ids])).logits[0, len(r["prompt"]) - 1:-1]
            want += torch.softmax(lg.double(), -1).sum(0)
            n += len(lg)
    assert torch.allclose(p, want / n, atol=1e-9)


def test_gate_passes_a_lossless_trim_and_fails_one_that_drops_likely_ids(tmp_path):
    m, _ = toy(tmp_path / "m", holes=0)
    src = tmp_path / "m"
    tok = tok_of(src)
    rows = rows_of(tok, n=4)
    full = sorted(vocab.BPE(vocab.tok_json(src)).defined)  # every id: renumbered only
    vocab.trim(src, tmp_path / "full", full, weights=False)
    g = vocab.gate(m, tok, tok_of(tmp_path / "full"), full, rows, length=4096, windows=4)
    assert g["pass"] and g["dropped_mass"]["mean"] == 0 and abs(g["nll_per_byte"]["rise"]) < 1e-9
    assert g["nll_per_byte"]["identical_windows"] == 1
    kept = code_kept(src)
    vocab.trim(src, tmp_path / "code", kept, weights=False)
    g = vocab.gate(m, tok, tok_of(tmp_path / "code"), kept, rows, length=4096, windows=4)
    # a random model spreads its mass, so the dropped share is about the dropped rows' share
    share = 1 - len(kept) / len(tok)
    assert not g["pass"] and not g["dropped_mass"]["pass"] and g["dropped_mass"]["mean"] > share / 4
    assert g["dropped_mass"]["positions"] == sum(len(r["tokens"]) for r in vocab.stable(rows)[:4])
    assert g["nll_per_byte"]["identical_windows"] < 1  # the rare prompts retokenize
    assert vocab.gate(m, tok, tok, kept, [], windows=4)["pass"] is False


# ---------- llama.cpp ----------

def converter_copy(tmp_path, full=False):
    """A private copy of the pinned converter (never the shared checkout), with its
    gguf-py and build linked."""
    if not (LLAMA_CPP / "conversion" / "base.py").exists():
        pytest.skip(f"no llama.cpp converter at {LLAMA_CPP}")
    lc = tmp_path / "llama.cpp"
    shutil.copytree(LLAMA_CPP / "conversion", lc / "conversion", ignore=shutil.ignore_patterns("__pycache__"))
    if full:
        shutil.copy2(LLAMA_CPP / "convert_hf_to_gguf.py", lc / "convert_hf_to_gguf.py")
        (lc / "gguf-py").symlink_to(LLAMA_CPP / "gguf-py")
        (lc / "build").symlink_to(LLAMA_CPP / "build")
    return lc


def test_register_maps_the_trimmed_hash_to_the_originals_pre_tokenizer(tmp_path):
    lc = converter_copy(tmp_path)
    src, out = tmp_path / "src", tmp_path / "out"
    toy(src)
    chktxt, table = vocab.pre_table(lc)
    assert "dbrx" in table.values() and len(chktxt) > 100
    with pytest.raises(ValueError, match="does not know"):
        vocab.pre_name(lc, src)
    # the toy stands in for a known tokenizer: map its hash as the update script would
    h0 = vocab.register(lc, src, pre="dbrx")
    assert vocab.pre_name(lc, src) == ("dbrx", h0)
    index = vocab.trim(src, out, code_kept(src), weights=False, llama_cpp=lc)
    assert index["pre"] == "dbrx" and index["orig_chkhsh"] == h0 and index["chkhsh"] not in (None, h0)
    assert vocab.load_index(out)["chkhsh"] == index["chkhsh"]
    assert vocab.pre_table(lc)[1][index["chkhsh"]] == "dbrx"
    body = (lc / "conversion" / "base.py").read_text()
    assert vocab.register(lc, out) == index["chkhsh"]  # idempotent
    assert (lc / "conversion" / "base.py").read_text() == body
    assert body.count(index["chkhsh"]) == 1 and body.index(index["chkhsh"]) < body.index("        if res is None:")
    with pytest.raises(ValueError, match="already mapped"):
        vocab.register(lc, out, pre="gpt-2")
    assert vocab.register(lc, src) is None  # an untrimmed checkpoint is left alone
    assert 'res = "dbrx"' in body and vocab._converter_file(lc) == lc / "conversion" / "base.py"


@pytest.fixture
def llama(tmp_path, monkeypatch):
    lc = converter_copy(tmp_path, full=True)
    if vocab.tokenizer_binary(lc) is None:
        pytest.skip("no llama-tokenize binary")
    if importlib.util.find_spec("sentencepiece") is None:  # GraniteModel tries it first; the VM has it
        stub = tmp_path / "stub"
        stub.mkdir()
        (stub / "sentencepiece.py").write_text("class SentencePieceProcessor:\n    pass\n")
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(stub), os.environ.get("PYTHONPATH")])))
    return lc


def test_llama_cpp_converts_the_trimmed_model_and_tokenizes_like_hf(tmp_path, llama):
    gguf = pytest.importorskip("gguf")
    src, out = tmp_path / "src", tmp_path / "out"
    _, ids = toy(src)
    vocab.register(llama, src, pre="dbrx")
    kept = code_kept(src)
    vocab.trim(src, out, kept, llama_cpp=llama)
    o2n = {o: n for n, o in enumerate(kept)}
    g = tmp_path / "out.gguf"
    r = subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(out), "--outfile", str(g),
                        "--outtype", "f32"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = gguf.GGUFReader(str(g))
    shapes = {t.name: list(t.shape) for t in rd.tensors}
    assert shapes["token_embd.weight"] == shapes["output.weight"] == [256, len(kept)]
    assert rd.fields["tokenizer.ggml.pre"].contents() == "dbrx"
    assert len(rd.fields["tokenizer.ggml.tokens"].contents()) == len(kept)
    assert rd.fields["tokenizer.ggml.eos_token_id"].contents() == o2n[ids["<|end_of_text|>"]]
    texts = vocab.EDGE_TEXTS + CODE + RARE + samples(1, 60) + [vocab.pre_table(llama)[0]]
    for d in (src, out):  # the original as a baseline, then the trim
        rep = vocab.tokenize_check(llama, d, texts)
        assert rep["pass"] and rep["texts"] == len(texts), rep
    assert vocab.tokenize_check(llama, out, texts[:5], gguf=g)["pass"]


def test_llama_cpp_tokenizes_a_trimmed_tiktoken_style_vocab_like_hf(tmp_path, llama):
    src, out = tmp_path / "src", tmp_path / "out"
    toy(src, style="tiktoken")
    vocab.register(llama, src, pre="dbrx")
    vocab.trim(src, out, code_kept(src), weights=False, llama_cpp=llama)
    texts = vocab.EDGE_TEXTS + CODE + RARE + samples(3, 60)
    for d in (src, out):
        rep = vocab.tokenize_check(llama, d, texts)
        assert rep["pass"], rep


def test_run_trims_when_the_gates_pass_and_falls_back_when_they_fail(tmp_path, llama):
    src = tmp_path / "m"
    m, _ = toy(src, holes=0)
    tok = tok_of(src)
    bench = write_bench(tmp_path / "bench", tok)
    tr = tmp_path / "traces.jsonl"
    tr.write_text("".join(json.dumps(r) + "\n" for r in rows_of(tok, n=8)))
    vocab.register(llama, src, pre="dbrx")
    # a random model gives every id ~1/V > MIN_PROB, so all ids stay: a lossless renumbering
    rep = vocab.run(src, tmp_path / "trim", tr, bench, llama, held_frac=0.5, length=4096, probe_windows=4,
                    gate_windows=4, n_texts=40, model=m)
    assert rep["trim"] and rep["rows"] == len(tok), rep
    a = rep["attempts"][0]
    assert a["model"]["pass"] and a["llama_tokenize"]["pass"] and a["special_tokens"]["pass"]
    # edge texts, chktxt, the gate windows, the test-window prompt, then training-answer pieces
    assert 20 < a["llama_tokenize"]["texts"] <= 40 and a["identical"]["n_window_prompts"] == 1
    assert json.loads((tmp_path / "trim.report.json").read_text())["hf_dir"] == str(tmp_path / "trim")
    assert vocab.load_index(tmp_path / "trim")["pre"] == "dbrx"
    assert (tmp_path / "trim" / "model.safetensors").exists() and not list(tmp_path.glob("trim.v*"))
    # with only counted ids kept the dropped mass is far over the gate, at both sizes
    rep = vocab.run(src, tmp_path / "small", tr, bench, llama, held_frac=0.5, length=4096, probe_windows=4,
                    gate_windows=4, n_texts=20, model=m, min_prob=1.1, sizes=(None, len(tok) - 40))
    assert not rep["trim"] and len(rep["attempts"]) == 2 and not (tmp_path / "small").exists()
    assert all(not x["model"]["dropped_mass"]["pass"] and x["llama_tokenize"]["pass"] for x in rep["attempts"])
    assert rep["attempts"][1]["vocab"]["new"] == len(tok) - 40


def test_smoke_trims_a_dummy_vocab_and_compares_llama_with_hf(tmp_path, llama):
    src = tmp_path / "m"
    toy(src)
    vocab.register(llama, src, pre="dbrx")
    rep = vocab.smoke(src, llama, tmp_path)
    assert rep["pass"], rep
    assert rep["rows"] < len(tok_of(src)) and rep["trimmed"]["texts"] == len(vocab.EDGE_TEXTS) + 1
