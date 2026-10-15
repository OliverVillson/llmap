"""scripts/bh/ data path (the Centipede train harness's rows) on the public spike, which is
smoke data only: the engine renders every row, the sandbox check, decontam, harvest's
guards, the free rows, round 2, build's guards, the tokenizer check, and the dry run."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bh import check, data, decontam, harvest, render  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="needs bun (the engine renders every row)")
SPIKE = ROOT / "mugge" / "spike"


@pytest.fixture(scope="module")
def tks():
    return render.tickets(SPIKE)


def ref(t):
    return json.loads(Path(t["dir"], "solutions", f"{t['id']}.json").read_text())["files"]


def answer(files, note=""):
    return render.one({"op": "answer", "files": files, "note": note})["content"]


# ---------- render ----------

def test_render_is_the_engines_wire(tks):
    from stages import harness
    assert render.system_prompt() == harness.SYSTEM
    t = tks["spike/c-hash"]
    w = render.one({"op": "write", "project": t["dir"], "ticket": t["id"]})["messages"]
    f = render.one({"op": "fix", "project": t["dir"], "ticket": t["id"],
                    "failed": {"command": "make x", "output": "  boom \n", "raw": True}})["messages"]
    assert render.kind_of(w[1]["content"]) == "write" and render.kind_of(f[1]["content"]) == "fix"
    assert render.failed_command(f[1]["content"]) == "make x"
    assert f[1]["content"].endswith("$ make x\nboom\n\nReturn your files fixed so it passes.")
    assert f[1]["content"].startswith(w[1]["content"].split(render.PACK_END)[0])


def test_render_errors_are_returned_or_raised(tks):
    bad = {"op": "write", "project": str(SPIKE), "ticket": "nope"}
    assert "no ticket" in render.call([bad], strict=False)[0]["error"]
    with pytest.raises(render.RenderError):
        render.call([bad])


# ---------- check ----------

def test_every_spike_reference_passes():
    rep = check.check_references(SPIKE)
    assert (rep["tickets"], rep["passed"], rep["broken"]) == (15, 15, {})


def test_check_names_the_failing_stage(tks):
    t = tks["spike/c-hash"]
    path = t["owns"][0]
    good = check.Item(t["dir"], t["id"], ref(t), key=t["key"])
    wrong = check.Item(t["dir"], t["id"], {path: "int nope;\n"}, key=t["key"])
    outside = check.Item(t["dir"], t["id"], {"elsewhere.c": "x"}, key=t["key"])
    r = check.check_many([good, wrong, outside], tks, keep=True)
    assert r[0].ok and r[0].worktree is None
    assert (r[1].ok, r[1].stage, r[1].command) == (False, "acceptance", t["acceptance"][0])
    assert "/work/" in r[1].output and "bh-" not in r[1].output  # paths as the container sandbox shows them
    assert Path(r[1].worktree, path).read_text() == "int nope;\n"
    shutil.rmtree(r[1].worktree)
    assert r[2].stage == "refused"


def test_check_has_loopback_but_no_network():
    code, out, _ = check.run("python3 -c \"import socket;s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
                             "socket.create_connection(s.getsockname());print('lo')\"", "/tmp", "python")
    assert (code, out) == (0, "lo")
    code, _, _ = check.run("python3 -c \"import socket;socket.create_connection(('1.1.1.1',53),timeout=3)\"", "/tmp", "python")
    assert code != 0


def test_check_times_out():
    code, _, timed = check.run("sleep 5", "/tmp", "python", timeout=0.5)
    assert (code, timed) == (124, True)


def test_container_argv_is_the_engines():
    js = ("import { ContainerSandbox } from './src/engine/sandbox.ts';"
          "console.log(JSON.stringify(new ContainerSandbox({engine:'docker'}).argv('make t', {cwd:'/w', image:'c'})))")
    out = subprocess.run(["bun", "-e", js], cwd=ROOT / "mugge", capture_output=True, text=True, check=True).stdout
    assert check.container_argv("docker", "make t", "/w", "c") == json.loads(out)


# ---------- decontam ----------

def test_decontam_catches_the_canary_and_bench_text(tks):
    ix = decontam.index([SPIKE])
    t = tks["spike/c-hash"]
    assert ix.why("hello MUGGE-BENCH-XYZ") == "canary"
    assert ix.why(next(iter(ref(t).values()))).startswith("overlap")
    assert ix.why("def add(a, b):\n    return a + b\n" * 3) == ""
    rows = [{"messages": [{"content": "clean"}, {"content": "MUGGE-BENCH-1"}]}, {"messages": [{"content": "fine"}]}]
    kept, dropped = decontam.filter_rows(rows, ix)
    assert len(kept) == 1 and dropped == {"canary": 1}
    # the spike as if it were a practice export: every ticket flagged, the whole project for its canary
    assert set(decontam.contaminated_tickets(SPIKE, ix)) == set(tks)


# ---------- harvest ----------

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The fake 'teacher' through the engine's own loop: 2 seeds, each ticket's first call empty."""
    d = tmp_path_factory.mktemp("run")
    subprocess.run(["bun", "bench/run.ts", "--bench", str(SPIKE), "--out", str(d), "--fake", "--fake-flaky", "--seeds", "2",
                    "--transcripts", "--sandbox", "local", "--filter", "c-", "--concurrency", "4"],
                   cwd=ROOT / "mugge", check=True, capture_output=True)
    return d


def test_transcripts_carry_what_harvest_needs(run):
    calls = [c for t in harvest.read_jsonl(run / "transcripts.jsonl") for c in t["calls"]]
    assert calls and all({"prompt", "answer", "system", "maxTokens", "completionTokens"} <= set(c) for c in calls)


def test_harvest_keeps_passing_calls_as_engine_rows(run, tks):
    rows, st = harvest.harvest(SPIKE, [run])
    assert st.solved == 10 and st.parser == 5  # five tickets × 2 seeds; the five empty first answers
    assert dict(st.rows) == {"write": 5, "fix": 5}
    for r in rows:
        t = tks[r["meta"]["key"]]
        x = render.one({"op": "canonical", "answer": r["messages"][2]["content"], "owns": t["owns"]})
        assert x["content"] == r["messages"][2]["content"] and not x["parsed"]["missing"]
        if r["meta"]["kind"] == "write":
            again = render.one({"op": "write", "project": t["dir"], "ticket": t["id"], "answer": r["messages"][2]["content"]})
            assert again["messages"] == r["messages"]
        else:
            assert "\n\n## Fix\nThis command failed:\n$ " in r["messages"][1]["content"]


def _edit(src: Path, dst: Path, fn) -> Path:
    shutil.copytree(src, dst)
    trs = harvest.read_jsonl(dst / "transcripts.jsonl")
    for t in trs:
        fn(t)
    harvest.write_jsonl(dst / "transcripts.jsonl", trs)
    return dst


def test_harvest_guards(run, tmp_path):
    # cut answers, a foreign engine's prompts and broken tickets never become rows
    cut = _edit(run, tmp_path / "cut", lambda t: t["calls"][-1].update(cut=True))
    rows, st = harvest.harvest(SPIKE, [cut])
    assert rows == [] and st.dropped["cut"] == 10
    forced = _edit(run, tmp_path / "forced", lambda t: t["calls"][-1].update(completionTokens=100, maxTokens=100))
    assert harvest.harvest(SPIKE, [forced])[1].dropped["cut"] == 10
    drift = _edit(run, tmp_path / "drift", lambda t: t["calls"][-1].update(prompt="Ticket: x\n" + t["calls"][-1]["prompt"]))
    rows, st = harvest.harvest(SPIKE, [drift])
    assert rows == [] and st.dropped["prompt_drift"] == 10 and st.warnings
    rows, st = harvest.harvest(SPIKE, [run], broken={"spike/c-hash"}, excluded={"spike/c-short"})
    assert st.dropped == {"broken_reference": 2, "decontam": 2} and len(rows) == 6


def test_harvest_write_from_fix_and_max_fix(run, tmp_path):
    # only the fixed seeds: no first-call pass, so the write row comes from the fix answer
    only_fix = tmp_path / "only-fix"
    shutil.copytree(run, only_fix)
    for name, fixed in (("results.jsonl", lambda r: r["attempts"] > 1), ("transcripts.jsonl", lambda t: len(t["calls"]) > 1)):
        harvest.write_jsonl(only_fix / name, [r for r in harvest.read_jsonl(only_fix / name) if fixed(r)])
    _, st = harvest.harvest(SPIKE, [only_fix])
    assert dict(st.rows) == {"write-from-fix": 5, "fix": 5}
    assert harvest.harvest(SPIKE, [only_fix], write_from_fix=False, max_fix=0)[0] == []


def test_load_runs_wants_transcripts(run, tmp_path):
    d = tmp_path / "no-trans"
    d.mkdir()
    shutil.copy(run / "results.jsonl", d)
    with pytest.raises(FileNotFoundError, match="--transcripts"):
        harvest.load_runs([d])


# ---------- the row-file commands bh_run.py calls ----------

def test_row_file_commands(run, tks, tmp_path):
    # harvest from transcript files; the proxy's calls log marks a passing call cut, so it drops
    trans = harvest.read_jsonl(run / "transcripts.jsonl")
    last = trans[0]["calls"][-1]
    log = tmp_path / "calls.jsonl"
    harvest.write_jsonl(log, [{"key": harvest.call_key(last["prompt"], last["answer"]), "finish_reason": "length", "cut": True}])
    rows_p = tmp_path / "rows.jsonl"
    st = data.harvest_files([run / "transcripts.jsonl"], SPIKE, rows_p, calls_logs=[log])
    assert st["dropped"]["length"] == 1 and st["rows"] == 9 and st["write_rows"] + st["fix_rows"] == 9
    assert json.loads((tmp_path / "rows.jsonl.stats.json").read_text()) == st
    rows = harvest.read_jsonl(rows_p)
    assert {"id", "kind", "source", "project", "ticket", "kit", "messages", "completion_tokens", "cut", "forced"} <= set(rows[0])
    # check: a row whose answer is broken fails
    bad = json.loads(json.dumps(rows[0]))
    t = tks[bad["meta"]["key"]]
    bad["messages"][2]["content"] = answer({p: "broken\n" for p in t["owns"]})
    harvest.write_jsonl(tmp_path / "mixed.jsonl", [*rows, bad, {**rows[0], "meta": {}, "ticket": "nope"}])
    assert data.check_rows(tmp_path / "mixed.jsonl", SPIKE, tmp_path / "checked.jsonl") == \
        {"rows_in": 11, "passed": 9, "failed": 1, "errors": 1}
    # decontam against the spike drops every spike row
    st = data.decontam_rows(rows_p, tmp_path / "clean.jsonl", [str(SPIKE)])
    assert (st["rows_in"], st["kept"]) == (9, 0) and st["dropped_overlap"] == 9
    # render: call specs become engine rows
    tw = tks["spike/c-hash"]
    specs = [{"id": "w1", "kind": "write", "source": "extra", "project": "spike", "ticket": tw["id"], "kit": "k",
              "bench": str(SPIKE), "answer_files": ref(tw), "note": "n"},
             {"id": "f1", "kind": "fix", "source": "extra", "project": "spike", "ticket": tw["id"], "kit": "k",
              "bench": str(SPIKE), "current": {tw["owns"][0]: "int x;\n"},
              "failed": {"command": tw["acceptance"][0], "output": "error: boom"}, "answer_files": ref(tw)}]
    harvest.write_jsonl(tmp_path / "specs.jsonl", specs)
    assert data.render_calls(tmp_path / "specs.jsonl", tmp_path / "rendered.jsonl") == {"calls": 2, "rows": 2, "errors": 0}
    w, f = harvest.read_jsonl(tmp_path / "rendered.jsonl")
    assert w["messages"] == render.one({"op": "write", "project": tw["dir"], "ticket": tw["id"], "answer": answer(ref(tw), "n")})["messages"]
    assert (f["id"], f["kind"], f["kit"]) == ("f1", "fix", "k") and "int x;" in f["messages"][1]["content"]


def test_repairs_command(tmp_path):
    """A student that writes half of each file: its failures become the engine's next fix calls."""
    src = SPIKE
    tks = render.tickets(src)
    refs = {t["id"]: ref(t) for t in tks.values()}

    def half(prompt):
        import re
        tid = re.search(r"^Ticket: (\S+)$", prompt, re.M).group(1)
        return answer({p: "\n".join(c.splitlines()[: len(c.splitlines()) // 2]) + "\n" for p, c in refs[tid].items()})

    url, stop = data._serve(half)
    try:
        subprocess.run(["bun", "bench/run.ts", "--bench", str(src), "--out", str(tmp_path / "s"), "--url", url, "--model", "s",
                        "--seeds", "1", "--transcripts", "--sandbox", "local", "--filter", "c-", "--thinking", "off"],
                       cwd=ROOT / "mugge", check=True, capture_output=True)
    finally:
        stop()
    st = data.repairs_files([tmp_path / "s" / "transcripts.jsonl"], src, tmp_path / "rep.jsonl")
    reqs = harvest.read_jsonl(tmp_path / "rep.jsonl")
    assert st["prompts"] == len(reqs) > 0 and st["failed_tickets"] >= len(reqs)
    q = reqs[0]
    assert {"id", "project", "ticket", "kit", "owns", "messages"} <= set(q)
    assert render.kind_of(q["messages"][1]["content"]) == "fix" and "/work" in q["messages"][1]["content"]


# ---------- free rows ----------

def test_extra_rows_are_held_to_the_wire(tks, tmp_path):
    cfg = data.DataConfig(work=tmp_path, benches=[])
    t = tks["spike/c-hash"]
    path = t["owns"][0]
    good = render.one({"op": "write", "project": t["dir"], "ticket": t["id"], "answer": answer(ref(t))})["messages"]
    loose = [*good[:2], {"role": "assistant", "content": f"Here it is:\n\n**{path}**\n```c\n{ref(t)[path]}```"}]
    wrong = render.one({"op": "write", "project": t["dir"], "ticket": t["id"], "answer": answer({path: "int nope;\n"})})["messages"]
    rows = [
        {"messages": good, "meta": {"project": t["dir"], "ticket": t["id"]}},
        {"messages": loose},
        {"messages": wrong, "meta": {"project": t["dir"], "ticket": t["id"]}},
        {"messages": [{"role": "system", "content": "You review."}, *good[1:]], "meta": {"kind": "review"}},
        {"messages": [{"role": "system", "content": "Other"}, *good[1:]]},
        {"messages": [*good[:2], {"role": "assistant", "content": "no files"}]},
    ]
    src = tmp_path / "extra.jsonl"
    harvest.write_jsonl(src, rows)
    out = harvest.read_jsonl(data.extra_rows(cfg, src))
    st = json.loads(cfg.stats("extra").read_text())
    assert len(out) == 2 and out[0]["meta"]["checked"] is True and out[1]["meta"]["checked"] is False
    assert out[1]["messages"][2]["content"] == good[2]["content"]  # re-rendered canonical
    assert st["recanonicalised"] == 1 and st["sandbox_acceptance"] == 1 and st["review"] == 1
    assert st["system_drift"] == 1 and st["parser"] == 1 and st["unchecked"] == 1
    assert len(harvest.read_jsonl(cfg.rows("extra-review"))) == 1
    strict = data.DataConfig(work=tmp_path / "strict", benches=[], extra_require_check=True)
    assert len(harvest.read_jsonl(data.extra_rows(strict, src))) == 1


# ---------- build and the tokenizer check ----------

def test_build_refuses_smoke_rows_and_dedupes(tmp_path):
    cfg = data.DataConfig(work=tmp_path, benches=[])
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"}], "meta": {"kind": "write"}}
    harvest.write_jsonl(cfg.rows("teacher"), [row, row])
    harvest.write_jsonl(cfg.rows("extra"), [{**row, "meta": {"kind": "fix"}, "messages": [*row["messages"][:2], {"role": "assistant", "content": "b"}]}])
    out = harvest.read_jsonl(data.build(cfg))
    assert len(out) == 2
    harvest.write_jsonl(cfg.rows("extra"), [{**row, "meta": {"smoke": True}}])
    with pytest.raises(RuntimeError, match="smoke"):
        data.build(cfg)


class StubTokenizer:
    """Character-level, with a chat template shaped like Qwen's (thinking off)."""

    def __init__(self, think_in_prompt_only=False):
        self.odd = think_in_prompt_only

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if self.odd else "")
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_tokenizer_roundtrip():
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a.py\n```python\nx = 1\n```"}]}
    assert data.tokenizer_roundtrip(row, StubTokenizer()) == "ok"
    with pytest.raises(RuntimeError, match="prefix"):
        data.tokenizer_roundtrip(row, StubTokenizer(think_in_prompt_only=True))


# ---------- the dry run ----------

def test_smoke_dry_run(tmp_path):
    rep = data.smoke(data.DataConfig(work=tmp_path))
    assert rep["references"] == "15/15"
    assert rep["smoke_guard"].startswith("ok")
    assert rep["by_kind"]["teacher/write"] == 15 and rep["by_kind"]["teacher/fix"] == 15
    assert rep["by_kind"].get("repair/fix", 0) > 0 and rep["by_kind"].get("rft/write", 0) > 0
    n, total = map(int, rep["write_rows_rerendered_identical"].split("/"))
    assert n == total > 0
