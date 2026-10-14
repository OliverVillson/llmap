import gzip
import json
import random
import statistics
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import thinking_report as tr  # noqa: E402

LOOP = "Wait, let me check again. " * 80


def rows(*specs):
    return [{"suite": "livecodebench", "id": f"p{i}", "sample": 0, **s} for i, s in enumerate(specs)]


REF = rows({"passed": True, "tokens": 1000, "finish": "stop"},
           {"passed": True, "tokens": 2000, "finish": "stop"},
           {"passed": True, "tokens": 3000, "finish": "stop"},
           {"passed": False, "tokens": 9000, "finish": "stop", "reason": "test_failed"})
STUDENT = rows({"passed": True, "tokens": 1500, "finish": "stop"},
               {"passed": False, "tokens": 32768, "finish": "length", "forced": "length", "reasoning_tail": LOOP},
               {"passed": False, "tokens": 4000, "finish": "stop", "reason": "timeout"},
               {"passed": True, "tokens": 8000, "finish": "stop"},
               {"passed": True, "tokens": 10, "finish": "stop", "suite": "multipl-e-py"})


def write_jobs(root: Path):
    for name, rs in (("ref-think", REF), ("r50w95s-t", STUDENT)):
        d = root / f"jobs/code10x-{name}/work/code_eval"
        d.mkdir(parents=True)
        (d / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rs)) + "\n")


def test_breakdown_against_ref(tmp_path):
    write_jobs(tmp_path)
    models = tr.load(tmp_path / "jobs", "livecodebench")
    assert set(models) == {"ref-think", "r50w95s-t"}
    assert len(models["r50w95s-t"]) == 4  # the multipl-e row is another suite
    s = tr.stats(models["r50w95s-t"])
    assert s["pass%"] == 50.0 and s["cap%"] == 25.0 and s["forced_passed"] == "0/1"
    assert s["timeout%"] == 25.0 and s["loop%"] == 25.0
    v = tr.compare(models["r50w95s-t"], models["ref-think"])
    assert v["lost"] == 2 and v["gained"] == 1
    assert v["lost_by"] == {"cap hit": 1, "test timeout": 1} and v["lost_looped"] == 1
    assert v["tokens_vs_ref"] == 1.5  # p0: 1500 over 1000


def test_reads_a_tar_like_the_bucket(tmp_path, capsys):
    write_jobs(tmp_path)
    tar = tmp_path / "exp02-jobs.tar"
    with tarfile.open(tar, "w") as t:
        t.add(tmp_path / "jobs", arcname="jobs")
    assert tr.main([str(tar)]) == 0
    out = capsys.readouterr().out
    assert "r50w95s-t vs ref-think on 4 shared answers: lost 2, gained 1" in out


def test_plain_thinking_is_not_a_loop():
    import random
    rng = random.Random(0)
    words = "the array sum prefix bound check index left right mid case loop answer modulo pair count".split()
    text = " ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(300))
    assert not tr.looped({"reasoning_tail": text})
    assert tr.looped({"reasoning_tail": LOOP})


def test_every_sample_counts_and_answers_pair_by_sample(tmp_path):
    """Four samples per problem: the shares are over all 8 answers, and the comparison
    pairs sample i of the model with sample i of the ref."""
    ref = [{"suite": "livecodebench", "id": f"p{i}", "sample": k, "passed": True, "tokens": 1000, "finish": "stop"}
           for i in range(2) for k in range(4)]
    model = [dict(r, passed=r["sample"] != 3, tokens=2000) for r in ref]
    model[0].update(finish="length", answer_cut=True, passed=False)
    for name, rs in (("ref-think", ref), ("q2", model)):
        d = tmp_path / f"jobs/{name}/work/code_eval"
        d.mkdir(parents=True)
        (d / f"{name}.jsonl").write_text("\n".join(map(json.dumps, rs)) + "\n")
    out = tr.report(tmp_path / "jobs")
    q = out["q2"]
    assert q["n"] == 8 and q["pass%"] == 62.5 and q["cap%"] == 12.5 and q["answer_cut%"] == 12.5
    assert q["forced%"] == 0.0 and q["loop_checked"] == 0 and q["loop_on"] is None
    v = q["vs_ref"]
    assert v["shared"] == 8 and v["lost"] == 3 and v["gained"] == 0
    assert v["lost_by"] == {"wrong answer": 2, "answer cut": 1} and v["tokens_vs_ref"] == 2.0
    assert "vs_ref" not in out["ref-think"]


def test_a_reading_that_stopped_before_its_tests_reads_the_partial_file(tmp_path):
    """Only the partial file: its answers count (an errored answer asked again counts
    once, as its retry), with no pass rate. Next to the graded file it is ignored."""
    d = tmp_path / "jobs/eh-q2/work/code_eval"
    d.mkdir(parents=True)
    base = {"suite": "livecodebench", "seed": 0, "run": "k", "prompt_sha": "s", "answer": "x", "reasoning_chars": 10}
    rows = [dict(base, id="p0", sample=0, finish="error", error="timed out", forced=None),
            dict(base, id="p1", sample=0, finish="length", tokens=16384, forced="length", reasoning_tail=LOOP),
            dict(base, id="p0", sample=0, finish="stop", tokens=900, forced=None),
            dict(base, id="p2", sample=0, finish="stop", tokens=800, forced=None, reason="case_timeout")]
    (d / "eh-q2.partial.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n{\"cut by a st")
    s = tr.report(tmp_path / "jobs")["eh-q2"]
    assert s["n"] == 3 and s["pass%"] is None and s["error%"] == 0.0
    assert s["cap%"] == 33.3 and s["forced%"] == 33.3 and s["forced_passed"] == "0/0"
    assert s["loop%"] == 33.3 and s["loop_checked"] == 1 and s["loop_on"] == "text"
    graded = [dict(r, passed=True, reason="passed") for r in rows[1:]]
    (d / "eh-q2.jsonl").write_text("\n".join(map(json.dumps, graded)) + "\n")
    s = tr.report(tmp_path / "jobs")["eh-q2"]
    assert s["pass%"] == 100.0 and s["case_timeout%"] == 0.0


def ids_natural(rng, n):
    """Token ids that do not repeat in long runs: a heavy-tailed draw over a 100k vocab."""
    return [min(100_000, int(rng.paretovariate(0.6))) + rng.randrange(50) for _ in range(n)]


def test_loops_in_token_ids_without_a_tokenizer():
    rng = random.Random(0)
    cycle = ids_natural(rng, 23)
    assert not tr.looped_ids(ids_natural(rng, 5000))
    assert tr.looped_ids(ids_natural(rng, 3000) + cycle * 120)
    # a loop the cap cut, then a forced answer of 3000 ids after it: still found
    assert tr.looped_ids(ids_natural(rng, 2000) + cycle * 60 + ids_natural(rng, 3000))
    assert not tr.looped_ids(cycle * 4)  # under a third of a window: too short to tell


def rollouts(rng):
    loop = ids_natural(rng, 17) * 100
    return [{"id": "a", "prompt": [1, 2], "tokens": ids_natural(rng, 900), "finish": "stop", "passed": True,
             "forced": None},
            {"id": "a", "prompt": [1, 2], "tokens": ids_natural(rng, 300) + loop + ids_natural(rng, 200),
             "finish": "length", "passed": False, "forced": "length"},
            {"id": "b", "prompt": [1, 2], "tokens": ids_natural(rng, 2000), "finish": "length", "passed": False,
             "forced": None},
            {"id": "b", "prompt": [1, 2], "tokens": ids_natural(rng, 700), "finish": "stop", "passed": None,
             "forced": "stop"}]


def test_rollout_files_give_the_same_shares_as_json(tmp_path, capsys):
    """Held-out on-policy rollouts (token ids, no suite): a cap hit after the thinking
    closed is an answer cut, a forced one is not; --out keeps the JSON."""
    rows = rollouts(random.Random(1))
    f = tmp_path / "held-rollouts.jsonl"
    f.write_text("\n".join(map(json.dumps, rows)) + "\n")
    out_file = tmp_path / "W" / "held-thinking.json"
    assert tr.main([str(f), "--json", "--out", str(out_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(out_file.read_text()) == printed
    s = printed["held-rollouts"]
    assert s["n"] == 4 and s["pass%"] == 33.3  # the unlabelled answer is left out
    assert s["cap%"] == 50.0 and s["forced%"] == 50.0 and s["answer_cut%"] == 25.0
    assert s["loop%"] == 25.0 and s["loop_checked"] == 4 and s["loop_on"] == "ids"
    assert s["tokens_median"] == round(statistics.median(len(r["tokens"]) for r in rows))
    gz = tmp_path / "held-rollouts-2.jsonl.gz"
    gz.write_bytes(gzip.compress(f.read_bytes()))
    assert tr.report([f, gz], ref="held-rollouts")["held-rollouts-2"]["vs_ref"]["lost"] == 0


class WordTok:
    """A stand-in tokenizer: id i is word i."""

    def __init__(self, words):
        self.words = words

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.words[i] for i in ids)


def test_with_a_tokenizer_the_thinking_is_read_as_text():
    from stages.eval import EARLY_STOP
    rng = random.Random(0)
    vocab = "the array sum prefix bound check index left right mid case loop answer modulo pair count".split()
    words = ["Wait, let me check again. ", "</think>", EARLY_STOP, "```python\nprint(1)\n```",
             *(f"{rng.choice(vocab)}{rng.randint(0, 999)} " for _ in range(200))]
    tok = WordTok(words)
    plain = list(range(4, 204))
    looping = plain[:20] + [0] * 80 + [2, 1, 3]  # loops, is cut, gets the forced answer
    answer_after = plain + [1] + [3] * 300  # a long repetitive answer is not thinking
    assert tr.thinking({"tokens": looping}, tok).endswith("Wait, let me check again.")
    assert tr.looped({"tokens": looping}, tok) and not tr.looped({"tokens": answer_after}, tok)
    assert tr.loop_check({"tokens": looping}, tok) == "text" and tr.loop_check({"tokens": looping}) == "ids"
    assert tr.stats({("a", 0): {"tokens": looping, "finish": "length", "forced": "length", "passed": False}},
                    tok)["loop%"] == 100.0
