"""Vocab trim: keep the token ids the model reads and writes in code and thinking, cut
the other rows of the embedding and the head, and gate the cut on what the original
model would do before anything is built on it.

Untrimmed, the two 100k-row tables are about a sixth of a ~1.29 GB file at the plan's
formats, and most of their rows are tokens of scripts and languages code never uses.
A trim can still lose points silently in two ways: it drops a token the model would
sample (a sampling change no check of how text tokenizes can see), or it changes how
text tokenizes. So ids are chosen from what the model reads and writes, and the gates
measure the original model:

  kept   the 256 byte tokens; every special and added token; every id in the traces
         (training split plus the selection half of the held-out problems); every id
         of the rendered CodeContests/APPS prompts and of LiveCodeBench statements
         released before 2025-01-01; every id whose mean probability under the original
         is >= 1e-6 at the answer positions of probe windows (selection half first);
         then closed under merge ancestry (below).
  gates  on the other half of the held-out problems, never used to choose:
         - the original's probability mass on dropped ids per answer position: mean
           <= 1e-4 and 99.9th percentile <= 1e-2;
         - NLL per byte of the same text, each model on its own tokenization: rises
           <= 0.3%;
         - llama-tokenize == HF on 500 texts (hard);
         - the chat template's special tokens survive (same text, same ids remapped);
         - the share of identically tokenized prompts is reported only.

LiveCodeBench's test window (2025-01-01 on) never chooses ids: its prompts are only
tokenized by the checks (identical share, llama-tokenize == HF).

Merge-ancestor closure. Byte-level BPE builds a token by merging pairs in rank order,
so a kept token is useless unless every token on its merge path is kept too: for each
kept token, BPE is run on its own bytes (in tokenizers' exact order) and every
intermediate token is kept. A merge survives when both halves and the result are
kept, in its original order. Then any text whose original tokenization uses only kept
tokens tokenizes to exactly the same tokens: every merge BPE applied to it survives,
and the merges that are gone were never applied, so BPE takes the same steps. Other
text falls back to smaller kept tokens and always round-trips, since all 256 bytes are
kept.

New ids follow old-id order. vocab_index.json (written beside the trimmed checkpoint)
holds {"kept": old ids in new order, "old_to_new": {old: new}}, the form
traces.batches' vocab_map and sft.kd_kl's index take, plus the llama.cpp
pre-tokenizer name and hashes.

llama.cpp. convert_hf_to_gguf.py names the pre-tokenizer (the regex llama.cpp splits
text with) by hashing the ids of a fixed test string (get_vocab_base_pre's chkhsh)
and refuses hashes it does not know. Renumbered ids give a new hash although the
pre-tokenizer is unchanged, so register() adds the trimmed tokenizer's hash to the
pinned converter, mapped to the name the converter gives the original tokenizer. The
same regex then splits text in llama.cpp as before, and its BPE runs on the trimmed
merges. The edit is three lines inside get_vocab_base_pre, idempotent, and the
checkout's HEAD is unchanged (setup_vm.sh accepts local changes at the pinned commit).
NVMe is wiped when the VM stops, so every session that converts a trimmed checkpoint
runs `register` again after setting up llama.cpp; it reads the name from the
checkpoint's vocab_index.json and is a no-op for an untrimmed one.

Tokenizers are always loaded from a checkpoint directory with its config.json:
transformers 5 maps Granite's model types to the backend that keeps tokenizer.json's
pre-tokenizer, while a bare tokenizer directory naming GPT2Tokenizer gets GPT-2's
regex rebuilt in its place.

    python -m stages.vocab run --hf <original> --out <trimmed dir> --traces traces.jsonl \\
        --codebench <dir> --llama-cpp <checkout>      # S1.6; exits 3 when no size passes
    python -m stages.vocab register --llama-cpp <checkout> --hf <trimmed dir>
    python -m stages.vocab smoke --hf <original> --llama-cpp <checkout> --out <dir>   # S1.2(f)
    python -m stages.vocab tokcheck --hf <trimmed dir> --gguf model.gguf --llama-cpp <checkout>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

TEST_SINCE = "2025-01-01"  # LiveCodeBench's test window starts here; nothing from it chooses ids
TEST_UNTIL = "2025-04-30"
MIN_PROB = 1e-6  # mean probability at answer positions that keeps an id
DROP_MEAN, DROP_P999 = 1e-4, 1e-2  # the original's mass on dropped ids per answer position
NLL_RISE = 0.003  # NLL per byte may rise by 0.3%
N_TEXTS = 500  # llama-tokenize == HF on this many texts
TOKENIZE_TRIES = 3  # llama-tokenize runs per text when its id line is missing (see llama_tokenize)
FALLBACK = (None, 65536)  # the measured set, then 65,536 rows; after that no trim
PROBE_WINDOWS, GATE_WINDOWS, WINDOW = 32, 16, 8192
CHUNK = 1024  # positions per softmax chunk: 1024 x 100k fp32 is 0.4 GB
INDEX = "vocab_index.json"
VOCAB_TENSORS = ("embed_tokens.weight", "lm_head.weight", "lm_head.bias")
# The chat template's markers; check() confirms each one the original has as a single
# token is still a single token with its remapped id.
MARKERS = ("<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>", "<think>", "</think>")
# Fixed texts for llama-tokenize == HF, on top of the data: the cases where
# pre-tokenizer regexes differ (digit runs, case of contractions, whitespace before
# newlines, scripts, emoji).
EDGE_TEXTS = [
    "1 12 123 1234 12345 123456 1234567 3.14159 1e-9 0x1F 1_000_000",
    "I'm I'M we'll WE'LL they're THEY'RE it's IT'S don't DON'T you'd YOU'D",
    "a  b   c    d\t\te\n\n\nf \n g  \n\n  h\r\nend   ",
    "def f(x):\n    if x:\n        return [i ** 2 for i in range(x)]\n\treturn None\n\n\n",
    "Привет, мир! Γειά σου κόσμε. 你好，世界。こんにちは。안녕하세요. مرحبا بالعالم",
    "emoji 🚀🦙 ✅ 😶‍🌫️ ⚡️ — “quotes” ‘single’ … naïve café",
    "<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n<think>\nok\n</think>\n\nanswer",
    "```python\nimport sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n```",
    "    \n        \n" + "x" * 300 + " " * 40 + "!!!???..." + "=" * 64,
]


def log(msg: str) -> None:
    print(f"[vocab] {msg}", flush=True)


# ---------- the tokenizer as data ----------

def tok_json(hf_dir: str | Path) -> dict:
    tj = json.loads((Path(hf_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    if tj.get("model", {}).get("type") != "BPE":
        raise ValueError(f"{hf_dir}/tokenizer.json is not a BPE tokenizer; only byte-level BPE can be trimmed here")
    return tj


def _pairs(merges) -> list[tuple[str, str]]:
    return [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in merges]


class BPE:
    """The original tokenizer's vocab and merges, for closure and remapping."""

    def __init__(self, tj: dict):
        from tokenizers import pre_tokenizers
        self.tj = tj
        self.vocab: dict[str, int] = tj["model"]["vocab"]
        self.token = {i: t for t, i in self.vocab.items()}
        self.added = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
        # pair -> (rank, result); a repeated pair keeps its last rank, as tokenizers' map does
        self.merges = {(a, b): (r, a + b) for r, (a, b) in enumerate(_pairs(tj["model"].get("merges", [])))}
        missing = [c for c in pre_tokenizers.ByteLevel.alphabet() if c not in self.vocab]
        if missing:
            raise ValueError(f"not a byte-level BPE: {len(missing)} of the 256 byte tokens are missing")
        self.bytes = {self.vocab[c] for c in pre_tokenizers.ByteLevel.alphabet()}
        self.defined = set(self.token) | set(self.added)
        self._paths: dict[int, set[int]] = {}

    def word(self, text: str) -> tuple[list[str], list[str]]:
        """(tokens, tokens created on the way) of BPE on one pre-tokenized word, in
        tokenizers' order (Word::merge_all): a heap of (rank, position), stale entries
        skipped. With a tiktoken-converted merge list (a token's every split is a merge
        at the token's rank) a merge can make a pair of lower rank than itself, so
        "merge every occurrence of the best pair" is not the same thing."""
        sym = list(text)
        nxt = list(range(1, len(sym))) + [-1]
        prv = list(range(-1, len(sym) - 1))
        heap = [(m[0], i, m[1]) for i in range(len(sym) - 1) if (m := self.merges.get((sym[i], sym[i + 1])))]
        heapq.heapify(heap)
        made = []
        while heap:
            _, pos, new = heapq.heappop(heap)
            if not sym[pos] or nxt[pos] < 0:
                continue
            right = nxt[pos]
            m = self.merges.get((sym[pos], sym[right]))
            if m is None or m[1] != new:
                continue
            sym[pos], sym[right] = new, ""
            made.append(new)
            nxt[pos] = nxt[right]
            if nxt[pos] >= 0:
                prv[nxt[pos]] = pos
            if prv[pos] >= 0 and (m := self.merges.get((sym[prv[pos]], new))):
                heapq.heappush(heap, (m[0], prv[pos], m[1]))
            if nxt[pos] >= 0 and (m := self.merges.get((new, sym[nxt[pos]]))):
                heapq.heappush(heap, (m[0], pos, m[1]))
        return [t for t in sym if t], made

    def path(self, i: int) -> set[int]:
        """The ids BPE creates on the way from token i's bytes to i (i included when BPE
        reaches it). A text's token is built from its own bytes the same way wherever
        it occurs, so these are exactly the tokens it needs."""
        if i not in self._paths:
            t = self.token.get(i)
            self._paths[i] = {self.vocab[x] for x in self.word(t)[1]} if t and len(t) > 1 else set()
        return self._paths[i]

    def closure(self, ids) -> set[int]:
        """ids plus every token on the merge path of each, to a fixed point."""
        keep, todo = set(ids), list(ids)
        while todo:
            for j in self.path(todo.pop()):
                if j not in keep:
                    keep.add(j)
                    todo.append(j)
        return keep


# ---------- choosing ids ----------

def _bincount(ids, n: int) -> torch.Tensor:
    if not ids:
        return torch.zeros(n, dtype=torch.long)
    return torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=n)[:n]


def statements(bench_dir: str | Path, before: str = TEST_SINCE) -> list[dict]:
    """Problems whose text may choose ids: LiveCodeBench rows released before `before`
    (undated rows are left out), and CodeContests and APPS rows that share no 13-word
    run with any LiveCodeBench statement (data.py's dedupe, as the fetch applies it).
    Files are read line by line and only what build_prompt needs is kept, since
    livecodebench.jsonl holds every test case."""
    from stages import codebench
    from stages.data import _shingles
    if before > TEST_SINCE:
        raise ValueError(f"before={before}: LiveCodeBench's test window ({TEST_SINCE} on) must not choose ids")

    def light(p):
        return {"id": str(p.get("id")), "language": codebench.normalize_language(p["language"]), "prompt": p["prompt"],
                "stub": p.get("stub", ""), "entry": p.get("entry", ""), "raw_prompt": p.get("raw_prompt", False),
                "tests": [] if isinstance(p.get("tests"), list) else {}}

    out, seen = [], set()
    lcb = codebench.suite_path("livecodebench", bench_dir)
    if lcb.exists():
        with open(lcb, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    seen |= _shingles(p["prompt"])
                    if p.get("date") and str(p["date"])[:10] < before:
                        out.append(light(p))
    n_lcb = len(out)
    for name in ("codecontests", "apps"):
        path = codebench.suite_path(name, bench_dir)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                out += [light(p) for line in f if line.strip()
                        for p in [json.loads(line)] if not (_shingles(p["prompt"]) & seen)]
    log(f"statements: {n_lcb} LiveCodeBench before {before}, {len(out) - n_lcb} CodeContests/APPS")
    return out


def count(tok, vocab_size: int, *, rows=(), problems=()) -> torch.Tensor:
    """Usage counts [vocab_size]: every id of the trace rows (prompt and answer, as
    sampled) and of each problem rendered as the eval asks it (opd.render)."""
    from stages import opd
    c = torch.zeros(vocab_size, dtype=torch.long)
    for r in rows:
        c += _bincount(r["prompt"] + r["tokens"], vocab_size)
    for p in problems:
        c += _bincount(opd.render(tok, p), vocab_size)
    return c


def stable(rows: list[dict], salt: str = "w") -> list[dict]:
    """rows in an order fixed by a salted hash of their problem id (traces.bucket; samples
    of a problem stay together, in file order)."""
    from stages import traces
    return sorted(rows, key=lambda r: (traces.bucket(r["id"], salt), r["id"]))


def _windows(rows: list[dict], length: int, n: int):
    """(ids [T], answer weights [T-1]) for the first n rows, each from its prompt,
    cropped to length."""
    from stages import traces
    for ids, _, w in traces.batches(rows[:n], length, 1, prompt_weight=0.0):
        yield ids[0], w[0]


def _logits(model, ids) -> torch.Tensor:
    dev = next(model.parameters()).device
    return model(input_ids=torch.as_tensor(ids, dtype=torch.long, device=dev)[None]).logits[0]


@torch.no_grad()
def probe(model, rows: list[dict], length: int = WINDOW, windows: int = PROBE_WINDOWS) -> torch.Tensor:
    """Mean probability of every id (float64 [model vocab]) over the answer positions
    of the first `windows` rows: the positions the model samples at."""
    dev = next(model.parameters()).device
    total, n = None, 0
    for ids, w in _windows(rows, length, windows):
        logits = _logits(model, ids)[:-1][w.to(torch.bool).to(dev)]
        if total is None:
            total = torch.zeros(logits.shape[-1], dtype=torch.float64, device=logits.device)
        for i in range(0, len(logits), CHUNK):
            total += torch.softmax(logits[i:i + CHUNK].float(), -1).sum(0, dtype=torch.float64)
        n += len(logits)
    if not n:
        raise ValueError("probe: no answer positions in the rows")
    return (total / n).cpu()


def choose(tj: dict, counts=None, probs=None, *, min_count: int = 1, min_prob: float = MIN_PROB,
           size: int | None = None) -> list[int]:
    """The kept old ids, sorted: bytes, special/added tokens, ids counted >= min_count
    times, ids with mean probability >= min_prob, closed under merge ancestry. With
    size, the set is then filled to exactly `size` ids (when the closure allows) with
    the most probable, then most counted, other ids, each with its merge path."""
    bpe = BPE(tj)

    def at(t, i):
        return float(t[i]) if t is not None and i < len(t) else 0.0

    must = bpe.bytes | set(bpe.added)
    must |= {i for i in bpe.defined if at(counts, i) >= min_count or (probs is not None and at(probs, i) >= min_prob)}
    keep = bpe.closure(must)
    if size:
        if len(keep) > size:
            raise ValueError(f"{len(keep)} ids are required, more than size={size}")
        for i in sorted(bpe.defined - keep, key=lambda i: (-at(probs, i), -at(counts, i), i)):
            if len(keep) >= size:
                break
            add = ({i} | bpe.path(i)) - keep
            if len(keep) + len(add) <= size:
                keep |= bpe.closure(add)
    return sorted(keep)


# ---------- writing the trimmed checkpoint ----------

def _remap(v, old_to_new: dict[int, int], what: str):
    if v is None:
        return None
    if isinstance(v, list):
        return [_remap(x, old_to_new, what) for x in v]
    if v not in old_to_new:
        raise ValueError(f"{what} = {v} is not a kept id")
    return old_to_new[v]


def _remap_config(cfg: dict, old_to_new: dict[int, int], what: str) -> dict:
    """Every *_token_id(s) field, at the top and in text_config; suppress lists drop
    ids that are gone."""
    out = dict(cfg)
    for k, v in cfg.items():
        if k.endswith(("_token_id", "_token_ids")) and isinstance(v, (int, list)) and not isinstance(v, bool):
            out[k] = _remap(v, old_to_new, f"{what} {k}")
        elif k in ("suppress_tokens", "begin_suppress_tokens") and isinstance(v, list):
            out[k] = [old_to_new[i] for i in v if i in old_to_new]
        elif k == "text_config" and isinstance(v, dict):
            out[k] = _remap_config(v, old_to_new, f"{what} text_config")
    return out


def _remap_post(pp, old_to_new: dict[int, int]):
    if not isinstance(pp, dict):
        return pp
    pp = dict(pp)
    if pp.get("type") == "Sequence":
        pp["processors"] = [_remap_post(p, old_to_new) for p in pp.get("processors", [])]
    if isinstance(pp.get("special_tokens"), dict):  # TemplateProcessing
        pp["special_tokens"] = {k: {**v, "ids": _remap(v["ids"], old_to_new, f"post_processor {k}")}
                                for k, v in pp["special_tokens"].items()}
    for k in ("sep", "cls"):  # Roberta/BertProcessing: [token, id]
        if isinstance(pp.get(k), list) and len(pp[k]) == 2:
            pp[k] = [pp[k][0], _remap(pp[k][1], old_to_new, f"post_processor {k}")]
    return pp


def trimmed_tokenizer(tj: dict, kept: list[int]) -> dict:
    """tokenizer.json with only the kept ids, renumbered in kept order; merges whose
    halves and result are all kept, in their order; added tokens and post-processor
    ids remapped. Everything else (pre-tokenizer, decoder, flags) is the original's."""
    old_to_new = {o: n for n, o in enumerate(kept)}
    m = tj["model"]
    vocab = dict(sorted(((t, old_to_new[i]) for t, i in m["vocab"].items() if i in old_to_new), key=lambda x: x[1]))
    merges = [x for x, (a, b) in zip(m.get("merges", []), _pairs(m.get("merges", [])))
              if a in vocab and b in vocab and a + b in vocab]
    out = dict(tj)
    out["model"] = {**m, "vocab": vocab, "merges": merges}
    out["added_tokens"] = [{**a, "id": _remap(a["id"], old_to_new, f"added token {a['content']!r}")}
                           for a in tj.get("added_tokens", [])]
    out["post_processor"] = _remap_post(tj.get("post_processor"), old_to_new)
    return out


def _check_kept(bpe: BPE, kept: list[int]) -> None:
    ks = set(kept)
    if len(ks) != len(kept):
        raise ValueError("kept has repeated ids")
    if not ks <= bpe.defined:
        raise ValueError(f"kept has ids the tokenizer does not define, e.g. {sorted(ks - bpe.defined)[:5]}")
    lost = (bpe.bytes | set(bpe.added)) - ks
    if lost:
        raise ValueError(f"kept lacks byte or special/added tokens, e.g. {sorted(lost)[:5]}")
    open_ = bpe.closure(ks) - ks
    if open_:
        raise ValueError(f"kept is not closed under merge ancestry: {len(open_)} merge-path ids missing "
                         f"(e.g. {sorted(open_)[:5]}); use choose() or BPE.closure()")


def _slice_weights(src: Path, out: Path, kept: list[int], old_v: int, tied: bool) -> list[str]:
    """Copies every safetensors shard, with the vocab rows of the embedding and head
    (and its bias) cut to kept, in kept order; dtypes and shard layout are unchanged."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    index = src / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        shards = sorted(set(idx["weight_map"].values()))
    elif (src / "model.safetensors").exists():
        idx, shards = None, ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no safetensors weights in {src} (only safetensors checkpoints can be trimmed)")
    rows = torch.tensor(kept, dtype=torch.long)
    sliced, total = [], 0
    for shard in shards:
        tensors = {}
        with safe_open(src / shard, "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith(VOCAB_TENSORS):
                    if t.shape[0] != old_v:
                        raise ValueError(f"{name} has {t.shape[0]} rows, config.vocab_size is {old_v}")
                    t = t.index_select(0, rows).contiguous()
                    sliced.append(name)
                elif t.dim() and t.shape[0] == old_v:
                    raise ValueError(f"{name} has vocab-sized rows ({old_v}) and is not a known vocab tensor")
                tensors[name] = t
                total += t.numel() * t.element_size()
        save_file(tensors, str(out / shard), metadata=meta)
    if not any(n.endswith("embed_tokens.weight") for n in sliced) or \
            (not tied and not any(n.endswith("lm_head.weight") for n in sliced)):
        raise ValueError(f"expected the embedding{'' if tied else ' and the head'} among the cut tensors, got {sliced}")
    if idx is not None:
        idx = {**idx, "metadata": {**idx.get("metadata", {}), "total_size": total}}
        (out / index.name).write_text(json.dumps(idx, indent=2) + "\n")
    return sliced


def trim(hf_dir: str | Path, out_dir: str | Path, kept: list[int], *, weights: bool = True,
         llama_cpp: str | Path | None = None, pre: str | None = None) -> dict:
    """Writes the trimmed checkpoint to out_dir: tokenizer files, config.json
    (vocab_size, *_token_id), generation_config.json, the embedding and head rows of
    kept (weights=False writes the tokenizer and configs only, enough for tokenizer
    checks and a vocab-only GGUF) and vocab_index.json. With llama_cpp, the original's
    pre-tokenizer name is looked up in that checkout's converter (or `pre` given) and
    the trimmed hash registered there. Returns the index."""
    src, out = Path(hf_dir), Path(out_dir)
    kept = [int(i) for i in kept]
    tj = tok_json(src)
    bpe = BPE(tj)
    _check_kept(bpe, kept)
    old_to_new = {o: n for n, o in enumerate(kept)}
    cfg = json.loads((src / "config.json").read_text())
    old_v = int(cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size"))
    if out.exists():
        if not (out / INDEX).exists():
            raise FileExistsError(f"{out} exists and is not a trimmed checkpoint; not overwriting it")
        shutil.rmtree(out)
    part = out.with_name(out.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    weight_files = {"model.safetensors.index.json"}
    for f in src.iterdir():
        if f.is_file() and f.suffix not in (".safetensors", ".bin", ".pt", ".pth", ".gguf") and f.name not in weight_files:
            shutil.copy2(f, part / f.name)
    (part / "tokenizer.json").write_text(json.dumps(trimmed_tokenizer(tj, kept), ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    tc = part / "tokenizer_config.json"
    if tc.exists():
        c = json.loads(tc.read_text(encoding="utf-8"))
        if isinstance(c.get("added_tokens_decoder"), dict):
            c["added_tokens_decoder"] = {str(_remap(int(k), old_to_new, f"added_tokens_decoder {v.get('content')!r}")): v
                                         for k, v in c["added_tokens_decoder"].items()}
        tc.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    vj = part / "vocab.json"  # slow-tokenizer files, if the repo has them, must agree with tokenizer.json
    if vj.exists():
        v = json.loads(vj.read_text(encoding="utf-8"))
        vj.write_text(json.dumps({t: old_to_new[i] for t, i in v.items() if i in old_to_new}, ensure_ascii=False),
                      encoding="utf-8")
    mt = part / "merges.txt"
    if mt.exists():
        new = trimmed_tokenizer(tj, kept)["model"]
        lines = [" ".join(p) for p in _pairs(new["merges"])]
        mt.write_text("#version: 0.2\n" + "\n".join(lines) + "\n", encoding="utf-8")
    cfg = _remap_config(cfg, old_to_new, "config.json")
    if "vocab_size" in cfg or "text_config" not in cfg:
        cfg["vocab_size"] = len(kept)
    else:
        cfg["text_config"]["vocab_size"] = len(kept)
    (part / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    gc = part / "generation_config.json"
    if gc.exists():
        gc.write_text(json.dumps(_remap_config(json.loads(gc.read_text()), old_to_new, "generation_config.json"),
                                 indent=2) + "\n")
    sliced = _slice_weights(src, part, kept, old_v, bool(cfg.get("tie_word_embeddings"))) if weights else []
    index = {"kept": kept, "old_to_new": {str(o): n for o, n in old_to_new.items()}, "old_vocab_size": old_v,
             "new_vocab_size": len(kept), "source": str(src), "weights": sliced,
             "tokenizer_sha256": hashlib.sha256((src / "tokenizer.json").read_bytes()).hexdigest(),
             "pre": pre, "orig_chkhsh": None, "chkhsh": None}
    if llama_cpp is not None:
        chktxt, table = pre_table(llama_cpp)
        index["orig_chkhsh"] = chkhsh(src, chktxt)
        index["pre"] = pre or table.get(index["orig_chkhsh"])
        if not index["pre"]:
            raise ValueError(f"the converter in {llama_cpp} does not know the tokenizer in {src} (chkhsh "
                             f"{index['orig_chkhsh']}); converting the original fails too; pass its pre-tokenizer name")
    (part / INDEX).write_text(json.dumps(index) + "\n")
    os.replace(part, out)
    if llama_cpp is not None:
        index["chkhsh"] = register(llama_cpp, out)
        (out / INDEX).write_text(json.dumps(index) + "\n")
    log(f"trim: {old_v} -> {len(kept)} rows, {len(sliced)} tensors cut, written to {out}")
    return index


def load_index(path: str | Path) -> dict:
    """vocab_index.json from its path or the trimmed checkpoint's directory."""
    p = Path(path)
    return json.loads((p / INDEX if p.is_dir() else p).read_text())


def kept_tensor(index: dict) -> torch.Tensor:
    """The kept old ids in new order, the `index` sft.kd_kl takes."""
    return torch.tensor(index["kept"], dtype=torch.long)


# ---------- llama.cpp: the converter's pre-tokenizer table ----------

def _converter_file(llama_cpp: str | Path) -> Path:
    """The file holding get_vocab_base_pre in this checkout (conversion/base.py at the
    pinned commit, convert_hf_to_gguf.py in older ones)."""
    for rel in ("conversion/base.py", "convert_hf_to_gguf.py"):
        p = Path(llama_cpp) / rel
        if p.exists() and "def get_vocab_base_pre(" in p.read_text(encoding="utf-8"):
            return p
    raise FileNotFoundError(f"no get_vocab_base_pre in {llama_cpp} (conversion/base.py or convert_hf_to_gguf.py)")


def _pre_function(src: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "get_vocab_base_pre":
            return node
    raise ValueError("get_vocab_base_pre not found")


def pre_table(llama_cpp: str | Path) -> tuple[str, dict[str, str]]:
    """(chktxt, {chkhsh: pre-tokenizer name}) as the converter's get_vocab_base_pre has
    them, read from its source (no import). The checks run in order and the last match
    wins, as in the converter."""
    fn = _pre_function(_converter_file(llama_cpp).read_text(encoding="utf-8"))
    chktxt, table = None, {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "chktxt" for t in node.targets):
            chktxt = ast.literal_eval(node.value)
    for node in fn.body:
        t = node.test if isinstance(node, ast.If) else None
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "chkhsh"
                and isinstance(t.ops[0], ast.Eq) and isinstance(t.comparators[0], ast.Constant)):
            for st in node.body:
                if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id == "res":
                    table[t.comparators[0].value] = ast.literal_eval(st.value)
    if not isinstance(chktxt, str) or not table:
        raise ValueError("could not read chktxt and the hash table from get_vocab_base_pre")
    return chktxt, table


def chkhsh(hf_dir: str | Path, chktxt: str) -> str:
    """The converter's hash of a tokenizer: sha256 of str() of its ids for chktxt, the
    tokenizer loaded as the converter loads it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    return hashlib.sha256(str(tok.encode(chktxt)).encode()).hexdigest()


def pre_name(llama_cpp: str | Path, hf_dir: str | Path) -> tuple[str, str]:
    """(pre-tokenizer name, chkhsh) the checkout's converter gives this tokenizer."""
    chktxt, table = pre_table(llama_cpp)
    h = chkhsh(hf_dir, chktxt)
    if h not in table:
        raise ValueError(f"the converter in {llama_cpp} does not know the tokenizer in {hf_dir} (chkhsh {h}); "
                         "converting the original fails too, so pass the pre-tokenizer name explicitly")
    return table[h], h


def register(llama_cpp: str | Path, hf_dir: str | Path, pre: str | None = None) -> str | None:
    """Maps the tokenizer in hf_dir to its pre-tokenizer name in the checkout's
    converter, by adding `if chkhsh == <hash>: res = <name>` just before the
    converter's `if res is None:`. The name is `pre`, else vocab_index.json's; an
    untrimmed hf_dir (no index) is left alone and None returned. Idempotent; raises if
    the hash is already mapped to another name, or is unknown and no name is given.
    Returns the hash."""
    if pre is None:
        p = Path(hf_dir) / INDEX
        if not p.exists():
            return None
        pre = load_index(p).get("pre")
    path = _converter_file(llama_cpp)
    src = path.read_text(encoding="utf-8")
    chktxt, table = pre_table(llama_cpp)
    h = chkhsh(hf_dir, chktxt)
    if h in table:
        if pre and table[h] != pre:
            raise ValueError(f"chkhsh {h} is already mapped to {table[h]!r} in {path}, not {pre!r}")
        return h
    if not pre:
        raise ValueError(f"{Path(hf_dir) / INDEX} names no pre-tokenizer and {path} does not know chkhsh {h}; "
                         "trim with llama_cpp or pass pre")
    fn = _pre_function(src)
    end = next((n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "res"
                and isinstance(n.test.ops[0], ast.Is)), None)
    if end is None:
        raise ValueError(f"no `if res is None:` in get_vocab_base_pre of {path}; this converter needs a new rule here")
    lines = src.splitlines(keepends=True)
    line = lines[end.lineno - 1]
    ind = line[:len(line) - len(line.lstrip())]
    block = (f'{ind}if chkhsh == "{h}":\n'
             f"{ind}    # ref: a trimmed copy of a {pre!r} vocab (registered by llmap stages/vocab.py)\n"
             f'{ind}    res = "{pre}"\n')
    new = "".join(lines[:end.lineno - 1]) + block + "".join(lines[end.lineno - 1:])
    compile(new, str(path), "exec")
    tmp = path.with_name(path.name + ".vocab-tmp")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, path)
    if pre_table(llama_cpp)[1].get(h) != pre:
        raise RuntimeError(f"registering {h} in {path} did not take")
    log(f"register: chkhsh {h[:12]}.. -> {pre!r} in {path}")
    return h


def convert_vocab(llama_cpp: str | Path, hf_dir: str | Path, out: str | Path) -> Path:
    """A vocab-only GGUF of hf_dir by the checkout's converter (what llama-tokenize needs),
    a trimmed hf_dir's tokenizer registered with it first."""
    register(llama_cpp, hf_dir)
    out = Path(out)
    r = subprocess.run([sys.executable, str(Path(llama_cpp) / "convert_hf_to_gguf.py"), str(hf_dir), "--vocab-only",
                        "--outfile", str(out)], capture_output=True, text=True)
    if r.returncode or not out.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py --vocab-only {hf_dir} failed:\n{r.stderr[-3000:]}")
    return out


def tokenizer_binary(llama_cpp: str | Path) -> Path | None:
    p = Path(llama_cpp) / "build" / "bin" / "llama-tokenize"
    return p if p.exists() else None


def llama_tokenize(binary: str | Path, gguf: str | Path, texts: list[str], workers: int = 8,
                   tries: int = TOKENIZE_TRIES) -> list[list[int]]:
    """llama-tokenize's ids for each text: special tokens parsed, no BOS, no escapes,
    as HF's encode(text, add_special_tokens=False). A run whose output has no whole
    "[...]" id line is run again (the pinned llama.cpp's tools can exit 0 with their
    output cut short under load, see horizon._kld_run), then raises."""
    def one(text):
        for i in range(tries):
            r = subprocess.run([str(binary), "-m", str(gguf), "--stdin", "--ids", "--no-bos", "--no-escape",
                                "--log-disable"], input=text.encode("utf-8"), capture_output=True, timeout=600)
            if r.returncode:
                raise RuntimeError(f"llama-tokenize failed ({r.returncode}): "
                                   f"{r.stderr.decode(errors='replace')[-500:]}")
            lines = [ln.strip() for ln in r.stdout.decode("utf-8", "replace").splitlines()]
            lines = [ln for ln in lines if ln.startswith("[") and ln.endswith("]")]
            if lines:
                return json.loads(lines[-1])
            log(f"llama-tokenize printed no whole id line for a {len(text)}-character text; running it again")
        raise RuntimeError(f"llama-tokenize printed no whole id line in {tries} runs: "
                           f"{r.stdout.decode(errors='replace')[-300:]!r} {r.stderr.decode(errors='replace')[-300:]}")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, texts))


def tokenize_check(llama_cpp: str | Path, hf_dir: str | Path, texts: list[str], *, gguf: str | Path | None = None,
                   work: str | Path | None = None) -> dict:
    """llama-tokenize == HF on every text, for hf_dir's tokenizer and its GGUF (a
    vocab-only one is converted when gguf is None). Skipped, not passed, when the
    binary is missing."""
    from transformers import AutoTokenizer
    binary = tokenizer_binary(llama_cpp)
    if binary is None:
        return {"pass": False, "skipped": f"no llama-tokenize in {llama_cpp}/build/bin"}
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        g = Path(gguf) if gguf else convert_vocab(llama_cpp, hf_dir, Path(tmp) / "vocab.gguf")
        tok = AutoTokenizer.from_pretrained(str(hf_dir))
        hf = [tok.encode(t, add_special_tokens=False) for t in texts]
        ll = llama_tokenize(binary, g, texts, workers=min(8, os.cpu_count() or 1))
    bad = [i for i in range(len(texts)) if hf[i] != ll[i]]
    ex = []
    for i in bad[:3]:
        k = next((j for j, (a, b) in enumerate(zip(hf[i], ll[i])) if a != b), min(len(hf[i]), len(ll[i])))
        ex.append({"text": texts[i][:200], "at": k, "hf": hf[i][max(0, k - 3):k + 5], "llama": ll[i][max(0, k - 3):k + 5]})
    return {"texts": len(texts), "mismatches": len(bad), "examples": ex, "pass": not bad and bool(texts)}


# ---------- gates ----------

@torch.no_grad()
def _nll(model, ids: list[int], index: torch.Tensor | None = None, targets: list[int] | None = None) -> float:
    """Summed NLL of ids[1:] given the prefix; with index the softmax is over those
    logits only (the trimmed model) and targets are positions in index."""
    logits = _logits(model, ids)[:-1]
    tgt = torch.tensor((targets or ids)[1:], device=logits.device)
    total = 0.0
    for i in range(0, len(logits), CHUNK):
        lg = logits[i:i + CHUNK]
        lg = (lg.index_select(-1, index) if index is not None else lg).float()
        total += float((torch.logsumexp(lg, -1) - lg.gather(-1, tgt[i:i + CHUNK, None])[:, 0]).sum())
    return total


@torch.no_grad()
def gate(model, tok, tok_trim, kept: list[int], rows: list[dict], *, length: int = WINDOW,
         windows: int = GATE_WINDOWS) -> dict:
    """The model gates on up to `windows` held-out rows (the gate half: never used to
    choose). Dropped mass: the original's probability on ids not kept, at each answer
    position of the row as sampled. NLL per byte: each row's window decoded to text,
    the original on its own tokenization, the trimmed model (the original's logits at
    the kept ids, which is what cutting the rows gives) on the trimmed tokenization."""
    import numpy as np
    dev = next(model.parameters()).device
    index = torch.tensor(kept, dtype=torch.long, device=dev)
    old_to_new = {o: n for n, o in enumerate(kept)}
    masses, nll_o, nll_t, nbytes, same, n = [], 0.0, 0.0, 0, 0, 0
    for ids, w in _windows(stable(rows), length, windows):
        logits = _logits(model, ids)[:-1][w.to(torch.bool).to(dev)]
        drop = torch.ones(logits.shape[-1], dtype=torch.bool, device=dev)
        drop[index[index < logits.shape[-1]]] = False
        for i in range(0, len(logits), CHUNK):
            lg = logits[i:i + CHUNK].float()
            masses.append((torch.logsumexp(lg[:, drop], -1) - torch.logsumexp(lg, -1)).exp().double().cpu()
                          if drop.any() else torch.zeros(len(lg), dtype=torch.float64))
        text = tok.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        a = tok.encode(text, add_special_tokens=False)
        b = tok_trim.encode(text, add_special_tokens=False)
        nll_o += _nll(model, a)
        nll_t += _nll(model, [kept[j] for j in b], index, b)
        nbytes += len(text.encode("utf-8"))
        same += b == [old_to_new.get(i, -1) for i in a]
        n += 1
    if not n:
        return {"pass": False, "error": "no gate windows (no held-out trace rows with an answer)"}
    m = torch.cat(masses).numpy() if masses else np.zeros(0)
    dm = {"mean": float(m.mean()) if len(m) else 0.0, "p999": float(np.percentile(m, 99.9)) if len(m) else 0.0,
          "max": float(m.max()) if len(m) else 0.0, "positions": int(len(m)),
          "limits": {"mean": DROP_MEAN, "p999": DROP_P999}}
    dm["pass"] = bool(len(m)) and dm["mean"] <= DROP_MEAN and dm["p999"] <= DROP_P999
    rise = nll_t / nll_o - 1 if nll_o else 0.0
    nb = {"original": nll_o / nbytes, "trimmed": nll_t / nbytes, "rise": rise, "limit": NLL_RISE, "bytes": nbytes,
          "windows": n, "identical_windows": same / n, "pass": rise <= NLL_RISE}
    return {"dropped_mass": dm, "nll_per_byte": nb, "pass": dm["pass"] and nb["pass"]}


def _chats() -> list[tuple[list[dict], bool]]:
    """(messages, add_generation_prompt): an eval prompt, and a finished turn with
    thinking and an answer."""
    from stages.codebench import SYSTEM
    return [([{"role": "system", "content": SYSTEM},
              {"role": "user", "content": "Read two integers and print their sum."}], True),
            ([{"role": "user", "content": "Add 2 and 3."},
              {"role": "assistant", "content": "<think>\n2 + 3 = 5.\n</think>\n\n```python\nprint(5)\n```"}], False)]


def _render(tok, msgs, generation: bool) -> str:
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation, enable_thinking=True)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation)


def special_check(tok, tok_trim, old_to_new: dict[int, int], hf_dir: str | Path, trimmed_dir: str | Path) -> dict:
    """The chat template survives: the same text for a prompt and for a finished turn
    with thinking, its ids the original's remapped; each added token and each MARKERS
    entry the original has as one token is one token with the remapped id; config and
    generation_config special ids name the same tokens."""
    out, ok = {}, True
    for k, (msgs, gen) in enumerate(_chats()):
        try:
            a, b = _render(tok, msgs, gen), _render(tok_trim, msgs, gen)
        except Exception as e:  # a template that cannot render this is a finding, not a crash
            out[f"template_{k}"] = {"error": repr(e)[:300]}
            ok = False
            continue
        ia, ib = tok.encode(a, add_special_tokens=False), tok_trim.encode(b, add_special_tokens=False)
        same = a == b and ib == [old_to_new.get(i, -1) for i in ia]
        out[f"template_{k}"] = {"same_text": a == b, "same_ids": same, "tokens": len(ia)}
        ok &= same
    tokens = {}
    added = {t.content: i for i, t in tok.added_tokens_decoder.items()}
    for s in sorted(set(added) | set(MARKERS)):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            if s in MARKERS:
                tokens[s] = "not a single token in the original"
            continue
        new = tok_trim.encode(s, add_special_tokens=False)
        good = new == [old_to_new.get(ids[0], -1)]
        tokens[s] = {"old": ids[0], "new": new[0] if len(new) == 1 else new, "ok": good}
        ok &= good
    out["tokens"] = tokens
    for name in ("config.json", "generation_config.json"):
        pa, pb = Path(hf_dir) / name, Path(trimmed_dir) / name
        if not (pa.exists() and pb.exists()):
            continue
        ca, cb = json.loads(pa.read_text()), json.loads(pb.read_text())
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            va, vb = ca.get(k), cb.get(k)
            if va is None:
                continue
            la, lb = (va if isinstance(va, list) else [va]), (vb if isinstance(vb, list) else [vb])
            good = len(la) == len(lb) and all(tok.convert_ids_to_tokens(x) == tok_trim.convert_ids_to_tokens(y)
                                               for x, y in zip(la, lb))
            out[f"{name}:{k}"] = {"old": va, "new": vb, "ok": good}
            ok &= good
    out["pass"] = bool(ok)
    return out


def window_prompts(tok, bench_dir: str | Path, since: str = TEST_SINCE, until: str = TEST_UNTIL) -> list[str]:
    """LiveCodeBench test-window prompts as the eval renders them, for the
    tokenization checks only."""
    from stages import codebench, opd
    path = codebench.suite_path("livecodebench", bench_dir)
    out = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p.get("date") and since <= str(p["date"])[:10] <= until:
                        p = {**p, "language": codebench.normalize_language(p["language"]),
                             "tests": [] if isinstance(p.get("tests"), list) else {}}
                        out.append(_render(tok, opd.messages(p), True))
    return out


def identical_share(tok, tok_trim, old_to_new: dict[int, int], texts: list[str]) -> float | None:
    if not texts:
        return None
    same = sum(tok_trim.encode(t, add_special_tokens=False)
               == [old_to_new.get(i, -1) for i in tok.encode(t, add_special_tokens=False)] for t in texts)
    return same / len(texts)


def check_texts(tok, *, gate_rows=(), train_rows=(), prompts=(), length: int = WINDOW, windows: int = GATE_WINDOWS,
                n: int = N_TEXTS, chktxt: str | None = None) -> list[str]:
    """Texts for llama-tokenize == HF: EDGE_TEXTS and the converter's chktxt, the gate
    windows' text, the given prompts (test-window prompts belong here), then
    512-token pieces of training answers, up to n."""
    texts = list(EDGE_TEXTS) + ([chktxt] if chktxt else [])
    texts += [tok.decode(ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
              for ids, _ in _windows(stable(list(gate_rows)), length, windows)]
    texts += list(prompts)
    for r in stable(list(train_rows), "t"):
        if len(texts) >= n:
            break
        for k in range(0, min(len(r["tokens"]), 4096), 512):
            texts.append(tok.decode(r["tokens"][k:k + 512], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return texts[:n]


def check(hf_dir: str | Path, trimmed_dir: str | Path, *, model=None, gate_rows=(), train_rows=(),
          window_texts=(), llama_cpp: str | Path | None = None, length: int = WINDOW, windows: int = GATE_WINDOWS,
          n_texts: int = N_TEXTS) -> dict:
    """The gate report for a trimmed checkpoint (or a tokenizer-only trim): model gates
    when a model and gate rows are given, the special tokens, llama-tokenize == HF when
    llama_cpp is given, and identical-tokenization shares (reported only). pass needs
    every gate that ran to pass and the model and llama gates to have run."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    tok_trim = AutoTokenizer.from_pretrained(str(trimmed_dir))
    index = load_index(trimmed_dir)
    kept = index["kept"]
    old_to_new = {o: n for n, o in enumerate(kept)}
    rep = {"vocab": {"old": index["old_vocab_size"], "new": len(kept)}}
    rep["special_tokens"] = special_check(tok, tok_trim, old_to_new, hf_dir, trimmed_dir)
    if model is not None and gate_rows:
        rep["model"] = gate(model, tok, tok_trim, kept, list(gate_rows), length=length, windows=windows)
    else:
        rep["model"] = {"pass": False, "skipped": "no model or no gate rows"}
    rep["identical"] = {"window_prompts": identical_share(tok, tok_trim, old_to_new, list(window_texts)),
                        "n_window_prompts": len(window_texts), "reported_only": True}
    if llama_cpp is not None:
        chktxt = pre_table(llama_cpp)[0]
        texts = check_texts(tok, gate_rows=gate_rows, train_rows=train_rows, prompts=window_texts, length=length,
                            windows=windows, n=n_texts, chktxt=chktxt)
        rep["llama_tokenize"] = tokenize_check(llama_cpp, trimmed_dir, texts)
    else:
        rep["llama_tokenize"] = {"pass": False, "skipped": "no llama.cpp checkout given"}
    rep["pass"] = all(rep[k]["pass"] for k in ("special_tokens", "model", "llama_tokenize"))
    return rep


# ---------- the S1.6 step ----------

def _halves(held: list[dict]) -> tuple[list[dict], list[dict]]:
    """(selection, gate) halves of the held-out problems, by traces.half (not a second
    crc32, which would put nearly every held-out problem on one side)."""
    from stages import traces
    sel = [r for r in held if traces.half(r["id"]) == 0]
    gat = [r for r in held if traces.half(r["id"]) == 1]
    if not gat and sel:  # too few problems: the gate gets one
        last = max({r["id"] for r in sel}, key=lambda i: traces.bucket(i, "gate"))
        gat, sel = [r for r in sel if r["id"] == last], [r for r in sel if r["id"] != last]
    return sel, gat


def run(hf_dir: str | Path, out_dir: str | Path, traces_path: str | Path, bench_dir: str | Path,
        llama_cpp: str | Path, *, held_frac: float | None = None, length: int = WINDOW,
        probe_windows: int = PROBE_WINDOWS, gate_windows: int = GATE_WINDOWS, n_texts: int = N_TEXTS,
        sizes=FALLBACK, min_prob: float = MIN_PROB, model=None, pre: str | None = None) -> dict:
    """Counts, probe, then for each size in turn (None = the measured set): choose,
    a tokenizer-only trim, register, check; the first that passes is trimmed with
    weights to out_dir. The report goes to <out_dir>.report.json either way;
    report["trim"] is False when no size passed (the plan's last fallback: keep the
    untrimmed vocab), and the CLI then exits 3."""
    from transformers import AutoTokenizer
    from stages import traces
    from stages.sft import load_causal_lm
    out = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    tj = tok_json(hf_dir)
    rows = traces.load(traces_path)
    train, held = traces.split(rows, **({"held_frac": held_frac} if held_frac is not None else {}))
    sel, gat = _halves(held)
    log(f"traces: {len(train)} training rows, {len(sel)} selection and {len(gat)} gate rows (held out)")
    model = model if model is not None else load_causal_lm(str(hf_dir)).eval()
    vocab_size = max(int(model.config.vocab_size), len(tok))
    counts = count(tok, vocab_size, rows=train + sel, problems=statements(bench_dir))
    # held-out selection rows first; training rows top the windows up when it is small
    probs = probe(model, stable(sel, "p") + stable(train, "p"), length, probe_windows)
    window_texts = window_prompts(tok, bench_dir)
    report = {"sources": {"train_rows": len(train), "selection_rows": len(sel), "gate_rows": len(gat),
                          "counted_ids": int((counts > 0).sum()), "probable_ids": int((probs >= min_prob).sum()),
                          "window_prompts": len(window_texts)},
              "attempts": [], "trim": False, "hf_dir": None}
    for size in sizes:
        if size and size >= len(BPE(tj).defined):
            continue
        try:
            kept = choose(tj, counts, probs, min_prob=min_prob, size=size)
        except ValueError as e:  # the measured set is already larger than this size
            report["attempts"].append({"size": size, "skipped": str(e), "pass": False})
            continue
        cand = out.with_name(f"{out.name}.v{len(kept)}")
        trim(hf_dir, cand, kept, weights=False, llama_cpp=llama_cpp, pre=pre)
        rep = check(hf_dir, cand, model=model, gate_rows=gat, train_rows=train, window_texts=window_texts,
                    llama_cpp=llama_cpp, length=length, windows=gate_windows, n_texts=n_texts)
        rep["size"] = size
        report["attempts"].append(rep)
        log(f"check at {len(kept)} rows: {'pass' if rep['pass'] else 'FAIL'} "
            f"(dropped mean {rep['model'].get('dropped_mass', {}).get('mean')}, "
            f"NLL/byte rise {rep['model'].get('nll_per_byte', {}).get('rise')}, "
            f"llama mismatches {rep['llama_tokenize'].get('mismatches')})")
        shutil.rmtree(cand)
        if rep["pass"]:
            report["index"] = trim(hf_dir, out, kept, llama_cpp=llama_cpp, pre=pre)
            report["index"] = {k: v for k, v in report["index"].items() if k not in ("kept", "old_to_new")}
            report.update(trim=True, hf_dir=str(out), rows=len(kept))
            break
    Path(f"{out}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"report: {out}.report.json ({'trimmed to ' + str(report.get('rows')) if report['trim'] else 'no trim'})")
    return report


def smoke(hf_dir: str | Path, llama_cpp: str | Path, work: str | Path) -> dict:
    """S1.2(f): a dummy trim (the ids of EDGE_TEXTS, chktxt and two rendered chats,
    closed), registered, and llama-tokenize == HF on the original and on the trim.
    No model weights or GPU."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    chktxt = pre_table(llama_cpp)[0]
    texts = EDGE_TEXTS + [chktxt]
    ids = [i for t in texts + [_render(tok, m, g) for m, g in _chats()] for i in tok.encode(t, add_special_tokens=False)]
    kept = choose(tok_json(hf_dir), _bincount(ids, max(ids) + 1))
    d = Path(work) / "vocab-smoke"
    trim(hf_dir, d, kept, weights=False, llama_cpp=llama_cpp)
    tok_trim = AutoTokenizer.from_pretrained(str(d))
    rep = {"rows": len(kept), "original": tokenize_check(llama_cpp, hf_dir, texts, work=work),
           "trimmed": tokenize_check(llama_cpp, d, texts, work=work),
           "special_tokens": special_check(tok, tok_trim, {o: n for n, o in enumerate(kept)}, hf_dir, d)}
    rep["pass"] = all(rep[k]["pass"] for k in ("original", "trimmed", "special_tokens"))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="vocab trim: choose, gate, trim, register with llama.cpp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="S1.6: choose, gate and trim, with the 65,536-row and no-trim fallbacks")
    r.add_argument("--hf", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--traces", required=True)
    r.add_argument("--codebench", required=True)
    r.add_argument("--llama-cpp", required=True)
    r.add_argument("--length", type=int, default=WINDOW)
    r.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    r.add_argument("--gate-windows", type=int, default=GATE_WINDOWS)
    r.add_argument("--texts", type=int, default=N_TEXTS)
    r.add_argument("--pre", default=None, help="the original's pre-tokenizer name, when the converter does not know it")
    g = sub.add_parser("register", help="map a trimmed checkpoint's tokenizer hash in the converter (every session)")
    g.add_argument("--llama-cpp", required=True)
    g.add_argument("--hf", required=True)
    g.add_argument("--pre", default=None)
    s = sub.add_parser("smoke", help="S1.2(f): dummy trim, llama-tokenize == HF")
    s.add_argument("--hf", required=True)
    s.add_argument("--llama-cpp", required=True)
    s.add_argument("--out", required=True)
    t = sub.add_parser("tokcheck", help="llama-tokenize == HF for a GGUF and its HF checkpoint")
    t.add_argument("--hf", required=True)
    t.add_argument("--gguf", default=None)
    t.add_argument("--llama-cpp", required=True)
    t.add_argument("--traces", default=None)
    t.add_argument("--codebench", default=None)
    t.add_argument("--texts", type=int, default=N_TEXTS)
    a = ap.parse_args()
    if a.cmd == "run":
        rep = run(a.hf, a.out, a.traces, a.codebench, a.llama_cpp, length=a.length, probe_windows=a.probe_windows,
                  gate_windows=a.gate_windows, n_texts=a.texts, pre=a.pre)
        sys.exit(0 if rep["trim"] else 3)
    if a.cmd == "register":
        print(register(a.llama_cpp, a.hf, a.pre))
        return
    if a.cmd == "smoke":
        rep = smoke(a.hf, a.llama_cpp, a.out)
        print(json.dumps(rep, indent=2))
        sys.exit(0 if rep["pass"] else 1)
    from transformers import AutoTokenizer
    from stages import traces
    tok = AutoTokenizer.from_pretrained(a.hf)
    rows = traces.load(a.traces) if a.traces else []
    train, held = traces.split(rows) if rows else ([], [])
    texts = check_texts(tok, gate_rows=held, train_rows=train,
                        prompts=window_prompts(tok, a.codebench) if a.codebench else [], n=a.texts,
                        chktxt=pre_table(a.llama_cpp)[0])
    rep = tokenize_check(a.llama_cpp, a.hf, texts, gguf=a.gguf)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
