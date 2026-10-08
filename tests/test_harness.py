import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from stages import harness

ROOT = Path(__file__).resolve().parents[1]
# Shared with mugge/test/files.test.ts, so the trainer and the engine agree on the format.
CASES = json.loads((ROOT / "mugge/test/fixtures/files-cases.json").read_text())


@pytest.mark.parametrize("case", CASES["parse"], ids=[c["why"] for c in CASES["parse"]])
def test_parse_files_matches_the_engine(case):
    assert harness.parse_files(case["answer"]) == (case["files"], case["note"])


def test_render_files_matches_the_engine_and_parses_back():
    for c in CASES["render"]:
        text = harness.render_files(c["files"], c["note"])
        assert text == c["answer"]
        files, note = harness.parse_files(text)
        assert files == {p: x.rstrip("\n") + "\n" for p, x in c["files"].items()} and note == c["note"]


def test_system_prompt_is_the_engines():
    ts = (ROOT / "mugge/src/engine/prompts.ts").read_text()
    block = re.search(r"export const SYSTEM = \[(.*?)\]\.join", ts, re.S).group(1)
    lines = [json.loads(m) if m.startswith('"') else m[1:-1] for m in re.findall(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*\"""", block)]
    assert "\n".join(lines) == harness.SYSTEM


def test_fix_call_shows_the_files_and_the_failure():
    t = harness.Ticket(id="T1", title="Add", context="Write add(a, b).", owns=["prog.py"],
                       acceptance=["python3 prog.py"], current={"prog.py": "def add(a, b):\n    return a - b\n"})
    sys_msg, user = harness.fix_messages(t, "python3 prog.py", "AssertionError")
    assert sys_msg["content"] == harness.SYSTEM
    text = user["content"]
    assert text.startswith("Ticket: T1\nTitle: Add\nRole: implement\n\nWrite add(a, b).\n")
    assert "Files you own (write each in full): prog.py\nDone when these commands pass: python3 prog.py" in text
    assert "## Your files as they are now\n--- prog.py\ndef add(a, b):\n    return a - b\n" in text
    assert text.endswith("## Fix\nThis command failed:\n$ python3 prog.py\nAssertionError\n\nReturn your files fixed so it passes.")
    assert harness.tail("x" * 10, 4) == "…xxxx"


def test_data_stage_adds_write_and_fix_rows(tmp_path):
    """Dry run with data_harness_share and data_fix_rows: harness write rows carry the same
    code as a file, fix rows show a failed draft and its error, and held-out tasks never
    become fix drafts."""
    job = tmp_path / "job"
    job.mkdir()
    (job / "taskspec.json").write_text((ROOT / "examples/python-utils.code.taskspec.json").read_text())
    (job / "config.json").write_text(json.dumps({"data_harness_share": 0.5, "data_fix_rows": 20,
                                                 "n_generate": 150, "n_heldout": 20}))
    env = {**os.environ, "LOBBOT_DRY_RUN": "1"}
    p = subprocess.run([sys.executable, "pipeline.py", "--job", str(job), "--only", "data"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    rows = [json.loads(l) for l in (job / "data/train.jsonl").read_text().splitlines()]
    held = {json.loads(l)["input"] for l in (job / "data/heldout.jsonl").read_text().splitlines()}
    write = [r for r in rows if r["messages"][1]["content"].startswith("Ticket: ") and "## Fix\n" not in r["messages"][1]["content"]]
    fix = [r for r in rows if "## Fix\n" in r["messages"][1]["content"]]
    stats = json.loads((job / "data/stats.json").read_text())
    assert write and len(fix) == 20 == stats["fix_rows"] and stats["harness_write_rows"] == len(write)
    for r in write + fix:
        sys_msg, user, answer = r["messages"]
        assert sys_msg["content"] == harness.SYSTEM
        files, _ = harness.parse_files(answer["content"])
        assert list(files) == ["prog.py"] and files["prog.py"].strip()
    for r in fix:
        user = r["messages"][1]["content"]
        assert "--- prog.py\n" in user and "This command failed:\n$ python3" in user
        assert "/tmp/" not in user  # the sandbox's temp dir is not in the output
        task = user.split("Role: implement\n\n", 1)[1].split("\n\nThe tests are appended", 1)[0]
        assert task not in held
