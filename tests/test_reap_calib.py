"""REAP calibration source selection (reap_calib), without torch."""
import json

import pytest

from stages._util import Job
from stages.reap import calib_sequences


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
