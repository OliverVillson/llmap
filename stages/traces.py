"""Training text: the trace rows gen writes (stages/ternary.py), read back as training
batches that look like what the model sees at eval.

A trace row is one sampled answer to one problem:

    {"id": problem id, "prompt": [token ids], "tokens": [answer token ids],
     "finish": "stop" | "length", "passed": true | false | null, "forced": null | "length" | "stop"}

with `samples` rows per problem, written together. load() refuses anything that is
not token ids: a garbage prompt (e.g. a chat template's dict keys) once turned a whole
run's text into answers to the words "input_ids" and "attention_mask".

Held-out problems are picked by a stable hash of the problem id (zlib.crc32, not
Python's per-process salted hash()), so gen, every trainer and every later session
agree on the split, and all samples of a problem fall on the same side. Every other
draw by problem id (another seed, the halves of the held-out problems) hashes with
sha256 instead: crc32 is linear, so for ids of one length a crc32 of a prefixed id is
the id's own crc32 xor a constant, and a draw made that way follows the split (every
held-out id of a test set fell into the same "half").

batches() replaces packing traces end to end into fixed windows, which put most
window starts mid-answer with no problem statement in context and let one answer
attend to an unrelated one. Each row here starts at its prompt, is cropped to the
window length and right-padded, with an attention mask and per-target weights: 0 on
padding, prompt_weight on prompt tokens, 1 on the answer, and 0 where a trimmed
vocab dropped the target id.

    python -m stages.traces <traces.jsonl> --samples 2   # validate and print the report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch

# Share of problems held out (the KL gates, vocab's and horizon's halves, and qat's
# on-policy held set). The 8M-token text is ~500 problems at ~8k tokens per answer and
# 2 samples, so 0.06 holds out ~30 (2% held ~10, a handful per half); a ~5000-problem
# CodeContests pool gives qat ~300 held ids to take its 128 from. Fixed before the
# first gen: vocab, imatrix, PTQ, Block-AP and qat all depend on the split.
HELD_FRAC = 0.06
MIN_MEDIAN = 500  # answer tokens: a thinking model's answers are thousands long; fewer means broken text
# Rows per bucket before sorting by length: enough to keep padding small, few enough
# that each epoch's seed still mixes different rows into each batch.
BUCKET = 64


# ---------- reading ----------

def _ids(v) -> bool:
    return isinstance(v, list) and all(type(t) is int for t in v)


def load(path: str | Path) -> list[dict]:
    """The rows of a traces file: id, prompt and tokens (lists of ints), finish, passed
    (None when unlabelled) and forced. Raises on a row without a problem id or with
    anything but token ids in its prompt or answer."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") is None:
            raise ValueError(f"{path}:{n}: row has no problem id")
        for key in ("prompt", "tokens"):
            if not _ids(r.get(key)):
                head = r.get(key)[:4] if isinstance(r.get(key), list) else r.get(key)
                raise ValueError(f"{path}:{n}: {key} is not a list of token ids (starts {head!r}); "
                                 "this text is unusable, delete it and its .count and generate it again")
        rows.append({"id": str(r["id"]), "prompt": r["prompt"], "tokens": r["tokens"], "finish": r.get("finish"),
                     "passed": r.get("passed"), "forced": r.get("forced")})
    return rows


# ---------- held-out split ----------

def bucket(pid: str, seed: int | str = 0) -> float:
    """A problem id's place in [0, 1), the same in every process. Seed 0 is the held-out
    split's crc32 of the id; any other seed (an int or a salt string) is a sha256 of seed
    and id, independent of the split and of other seeds (see the module docstring)."""
    if not seed:
        return zlib.crc32(str(pid).encode()) / 2 ** 32
    return int.from_bytes(hashlib.sha256(f"{seed}:{pid}".encode()).digest()[:8], "big") / 2 ** 64


def half(pid: str) -> int:
    """0 or 1 for a problem id, independent of the held-out split. The one rule that
    halves the held-out problems into a side that chooses (vocab's selection rows,
    horizon's sweep) and a side that only checks (vocab's gate, horizon's check)."""
    return hashlib.sha256(str(pid).encode()).digest()[0] & 1


def is_held(pid: str, held_frac: float = HELD_FRAC, seed: int = 0) -> bool:
    return bucket(pid, seed) < held_frac


def split(rows: list[dict], held_frac: float = HELD_FRAC, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """(train, held out) by problem: a problem is held out when its id's bucket is below
    held_frac, with every sample of it. When too few problems for the fraction to pick
    any, the one with the lowest bucket is held out."""
    held_ids = {r["id"] for r in rows if is_held(r["id"], held_frac, seed)}
    ids = {r["id"] for r in rows}
    if not held_ids and len(ids) > 1 and held_frac > 0:
        held_ids = {min(ids, key=lambda i: bucket(i, seed))}
    return [r for r in rows if r["id"] not in held_ids], [r for r in rows if r["id"] in held_ids]


# ---------- batches ----------

def vocab_lut(vocab_map) -> torch.Tensor:
    """old id -> new id as a LongTensor, -1 where dropped, from a tensor of that form, a
    {old: new} mapping (string keys, as JSON has them, are fine) or a whole
    vocab_index.json dict (its "old_to_new")."""
    if isinstance(vocab_map, torch.Tensor):
        return vocab_map.long()
    if isinstance(vocab_map, Mapping) and "old_to_new" in vocab_map:
        vocab_map = vocab_map["old_to_new"]
    pairs = {int(k): int(v) for k, v in vocab_map.items()}
    lut = torch.full((max(pairs) + 1 if pairs else 0,), -1, dtype=torch.long)
    lut[list(pairs)] = torch.tensor(list(pairs.values()), dtype=torch.long)
    return lut


def batches(rows: list[dict], length: int, batch: int, *, vocab_map=None, prompt_weight: float = 0.1,
            seed: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One epoch of (input_ids [B, T], attention_mask [B, T], weights [B, T-1]) over the
    rows, each row once: its prompt then its answer, cropped to `length` and
    right-padded to the longest in its batch. Rows are shuffled by seed, sorted by
    length within buckets of BUCKET batches (little padding), and the batches shuffled.
    weights[b, t] belongs to the target input_ids[b, t + 1]: 0 on padding, prompt_weight
    on prompt tokens, 1 on answer tokens. With vocab_map (vocab_lut's forms) the ids
    are the trimmed model's and a target it dropped weighs 0 (its input becomes id 0;
    vocab.choose counts the training rows and the held-out selection half, so dropped
    targets turn up in the held-out gate half and in other text).
    Rows with no answer token inside the window are left out."""
    lut = None if vocab_map is None else vocab_lut(vocab_map)
    # (row, tokens in its window) for the rows with an answer token inside it
    seqs = [(r, n) for r in rows if (n := min(len(r["prompt"]) + len(r["tokens"]), length)) > len(r["prompt"])]
    order = list(range(len(seqs)))
    rng = random.Random(seed)
    rng.shuffle(order)
    groups = []
    for k in range(0, len(order), BUCKET * batch):
        chunk = sorted(order[k:k + BUCKET * batch], key=lambda i: seqs[i][1])
        groups += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(groups)
    for g in groups:
        T = max(seqs[i][1] for i in g)
        ids = torch.zeros(len(g), T, dtype=torch.long)
        mask = torch.zeros(len(g), T, dtype=torch.long)
        w = torch.zeros(len(g), T - 1)
        for b, i in enumerate(g):
            r, n = seqs[i]
            ids[b, :n] = torch.tensor((r["prompt"] + r["tokens"])[:n])
            mask[b, :n] = 1
            w[b, :n - 1] = 1.0
            w[b, :max(len(r["prompt"]) - 1, 0)] = prompt_weight
        if lut is not None:
            new = torch.where(ids < len(lut), lut[ids.clamp(max=len(lut) - 1)], -1)
            w[(new[:, 1:] < 0)] = 0.0
            ids = new.clamp(min=0) * mask
        yield ids, mask, w


# ---------- checks and progress ----------

def validate(rows: list[dict], samples: int, min_median: int = MIN_MEDIAN) -> dict:
    """Raises unless every problem has exactly `samples` rows and the median answer is
    over min_median tokens; returns the text's report (counts, cut, forced and pass
    shares). load() already refused non-int prompts."""
    if not rows:
        raise ValueError("no trace rows")
    per = Counter(r["id"] for r in rows)
    odd = {k: v for k, v in per.items() if v != samples}
    if odd:
        raise ValueError(f"{len(odd)} problems do not have {samples} rows each, e.g. {next(iter(odd.items()))}")
    lens = [len(r["tokens"]) for r in rows]
    median = statistics.median(lens)
    if median <= min_median:
        raise ValueError(f"median answer is {median:.0f} tokens (<= {min_median}): the text looks broken")
    labelled = [r["passed"] for r in rows if r["passed"] is not None]
    return {"rows": len(rows), "problems": len(per), "answer_tokens": sum(lens),
            "prompt_tokens": sum(len(r["prompt"]) for r in rows), "median_answer": median, "max_answer": max(lens),
            "cut": round(sum(r["finish"] == "length" for r in rows) / len(rows), 4),
            "forced": round(sum(bool(r["forced"]) for r in rows) / len(rows), 4),
            "passed": round(sum(labelled) / len(labelled), 4) if labelled else None,
            "unlabelled": len(rows) - len(labelled)}


def count_path(path: str | Path) -> Path:
    """The progress sidecar beside a traces file (traces.jsonl -> traces.count)."""
    return Path(path).with_suffix(".count")


def write_count(path: str | Path, **fields) -> None:
    """Writes the sidecar atomically: {"tokens": answer tokens written, "target": ..., ...}."""
    p = count_path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(fields) + "\n")
    os.replace(tmp, p)


def read_count(path: str | Path) -> dict | None:
    """The sidecar's fields, or None when it is missing or half written."""
    try:
        return json.loads(count_path(path).read_text())
    except (OSError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="validate a traces file")
    ap.add_argument("traces")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--min-median", type=int, default=MIN_MEDIAN)
    a = ap.parse_args()
    print(json.dumps(validate(load(a.traces), a.samples, a.min_median), indent=2))


if __name__ == "__main__":
    main()
