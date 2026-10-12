import io
import json
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
    assert "r50w95s-t vs ref-think on 4 shared problems: lost 2, gained 1" in out


def test_plain_thinking_is_not_a_loop():
    import random
    rng = random.Random(0)
    words = "the array sum prefix bound check index left right mid case loop answer modulo pair count".split()
    text = " ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(300))
    assert not tr.looped({"reasoning_tail": text})
    assert tr.looped({"reasoning_tail": LOOP})
