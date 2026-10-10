"""REAP calibration source selection (reap_calib, calib_extra_share), without torch."""
import json
import random
from collections import Counter

import pytest

from stages._util import Job
from stages.reap import calib_sequences
from stages.taskdata import calib_examples, is_extra, load_examples


class Tok:
    """Whitespace 'tokenizer': one id per word; no chat template."""
    chat_template = None
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(w) for w in text.split()]}


def _job(tmp_path, **cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    job = Job(tmp_path)
    rows = [{"messages": [{"role": "user", "content": f"task {i}"},
                          {"role": "assistant", "content": "answer"}]} for i in range(5)]
    job.path("data", "train.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return job


def test_task_is_default(tmp_path):
    assert len(calib_sequences(_job(tmp_path, reap_calib_samples=3), Tok())) == 3


def test_general_text_file(tmp_path):
    (tmp_path / "gen.txt").write_text("one two three\n\nfour five\n\n\nsix")
    seqs = calib_sequences(_job(tmp_path, reap_calib="general", reap_calib_path="gen.txt", reap_max_seq=2), Tok())
    assert sorted(map(len, seqs)) == [1, 2, 2]


def test_general_jsonl(tmp_path):
    rows = [{"text": "plain words here"}, {"prompt": "q", "response": "a b"}]
    (tmp_path / "gen.jsonl").write_text("\n".join(map(json.dumps, rows)))
    seqs = calib_sequences(_job(tmp_path, reap_calib="general", reap_calib_path=str(tmp_path / "gen.jsonl")), Tok())
    assert len(seqs) == 2


def test_general_needs_path(tmp_path):
    with pytest.raises(ValueError):
        calib_sequences(_job(tmp_path, reap_calib="general"), Tok())


# --- calib_extra_share: calibration rows from data_extra_rows ---

def write_train(path, n_task, langs, weight=1):
    """A train.jsonl as the data stage writes it: n_task task rows, then for each entry of langs
    an extra row in that language (None: no language), marked extra, weight times each."""
    task = [{"messages": [{"role": "user", "content": f"task {i}"}, {"role": "assistant", "content": "answer"}]}
            for i in range(n_task)]
    extra = [{"messages": [{"role": "user", "content": f"aider-exercise-number-{i}"},
                           {"role": "assistant", "content": "edit"}],
              "meta": {"extra": True, **({"language": lang} if lang else {})}} for i, lang in enumerate(langs)]
    path.write_text("".join(json.dumps(r) + "\n" for r in task + extra * weight))
    return path


def test_no_extra_share_takes_the_shuffle_as_before(tmp_path):
    path = write_train(tmp_path / "train.jsonl", 10, ["go"] * 4, weight=3)
    old = load_examples(path)
    random.Random(0).shuffle(old)
    assert calib_examples(path, 8) == calib_examples(path, 8, 0.0) == old[:8]
    assert calib_examples(path, 100) == old  # fewer rows than asked: all of them


def test_extra_share_takes_distinct_rows_from_each_language(tmp_path):
    path = write_train(tmp_path / "train.jsonl", 30, ["go"] * 6 + ["python"] * 6 + ["rust"], weight=3)
    out = calib_examples(path, 10, 0.5)
    extra = [ex for ex in out if is_extra(ex)]
    assert len(out) == 10 and len(extra) == 5 and len({json.dumps(ex["messages"]) for ex in extra}) == 5
    assert Counter(ex["meta"]["language"] for ex in extra) == {"go": 2, "python": 2, "rust": 1}
    flags = [is_extra(ex) for ex in out]
    assert flags != sorted(flags, reverse=True)  # mixed in, not first: REAP's layer importance reads the first rows
    assert calib_examples(path, 10, 0.5) == out  # the same rows every run


def test_extra_share_with_too_few_rows_takes_them_all(tmp_path):
    """Fewer extra rows than the share: all of them (copies once, no language is fine) and task
    rows for the rest; fewer task rows than the rest: the extra rows left fill in."""
    path = write_train(tmp_path / "train.jsonl", 20, [None] * 3, weight=4)
    out = calib_examples(path, 10, 0.8)
    assert len(out) == 10 and sum(map(is_extra, out)) == 3
    path = write_train(tmp_path / "train.jsonl", 2, ["go", "rust"] * 4, weight=2)
    out = calib_examples(path, 6, 0.5)
    assert len(out) == 6 and sum(map(is_extra, out)) == 4 and len({json.dumps(ex["messages"]) for ex in out}) == 6
    assert len(calib_examples(path, 50, 0.5)) == 10  # every distinct row once


def test_extra_share_out_of_range(tmp_path):
    path = write_train(tmp_path / "train.jsonl", 5, ["go"])
    for share in (-0.1, 1.5):
        with pytest.raises(ValueError, match="calib_extra_share"):
            calib_examples(path, 4, share)


def test_task_calibration_takes_the_extra_share(tmp_path):
    job = _job(tmp_path, reap_calib_samples=6, calib_extra_share=0.5)
    write_train(job.path("data", "train.jsonl"), 10, ["go", "python", "rust", "go"], weight=2)
    seqs = calib_sequences(job, Tok())
    assert len(seqs) == 6 and sum(len("aider-exercise-number-0") in s for s in seqs) == 3
