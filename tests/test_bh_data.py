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


def test_harvest_skips_loose_and_unclean_calls(run, tmp_path, tks):
    # a passing answer whose path the parser had to guess (Centipede run 1's prog.c) is never a row
    def loosen(t):
        c = t["calls"][-1]
        owns = tks[t["key"]]["owns"]
        if len(owns) == 1:
            c["answer"] = c["answer"].replace(owns[0] + "\n```", "**" + owns[0].rsplit("/", 1)[-1] + "**\n```", 1)
    loose = _edit(run, tmp_path / "loose", loosen)
    rows, st = harvest.harvest(SPIKE, [loose])
    assert st.dropped["loose"] == 10 and rows == [] and st.loose >= 10
    # the engine's per-call log (telemetry.ts CallRecord) marks a call unclean: it drops,
    # and so does every flag the observability PR writes into transcripts
    recs = []
    for t in harvest.read_jsonl(run / "transcripts.jsonl"):
        for i, _ in enumerate(t["calls"], 1):
            recs.append({"v": 1, "ticket": f"{t['key']}#s{t['seed']}", "attempt": i, "kind": "write" if i == 1 else "fix",
                         "cut": False, "forced": None, "drop": None, "loose": 0, "error": None,
                         "clean": not (t["key"] == "spike/c-hash" and i == len(t["calls"]))})
    recs.append({"v": 1, "ticket": "spike/c-hash#s0", "attempt": 9, "kind": "fix", "error": "inference 500", "clean": False})
    log = tmp_path / "calls.jsonl"
    harvest.write_jsonl(log, recs)
    rows, st = harvest.harvest(SPIKE, [run], calls_logs=[log])
    assert st.dropped["unclean"] == 2 and len(rows) == 8
    for flag, val, why in (("forced", "length", "forced"), ("drop", "unlabelled", "parser"), ("loose", True, "loose")):
        d = _edit(run, tmp_path / f"flag-{flag}", lambda t, flag=flag, val=val: t["calls"][-1].update({flag: val}))
        assert harvest.harvest(SPIKE, [d])[1].dropped[why] == 10, flag


def test_harvest_warns_over_20_percent_and_flagged_calls_never_become_rows(run, tmp_path, capsys):
    """A run where more than 20% of answers are cut, forced or loose: the warning names each
    count, and no flagged answer becomes a row (a debug run of the guards)."""
    flags = [{"cut": True}, {"forced": "length"}, {"loose": True}]
    n = [0]

    def mark(t):  # one flag on the last call of three runs in four
        if n[0] % 4 < 3:
            t["calls"][-1].update(flags[n[0] % 4])
        n[0] += 1
    d = _edit(run, tmp_path / "mixed", mark)
    flagged = {(t["key"], t["seed"]) for i, t in enumerate(harvest.read_jsonl(d / "transcripts.jsonl")) if i % 4 < 3}
    rows, st = harvest.harvest(SPIKE, [d])
    assert st.drop_share() > 0.2 and st.warnings
    w = st.warnings[-1]
    assert "cut at the cap or forced" in w and "with a path the parser had to guess" in w
    assert st.cut >= 2 and st.loose >= 1
    assert rows and not flagged & {(r["meta"]["key"], r["meta"]["seed"]) for r in rows}
    assert "[harvest] WARNING" in capsys.readouterr().out


def test_build_drops_canary_and_spike_rows(tmp_path):
    """build decontaminates against MuggeBench: a row carrying the MUGGE-BENCH- canary or a
    spike ticket's text never reaches train.jsonl."""
    cfg = data.DataConfig(work=tmp_path, benches=[SPIKE])
    t = next(iter(render.tickets(SPIKE).values()))
    w = render.one({"op": "write", "project": t["dir"], "ticket": t["id"]})
    sysm = {"role": "system", "content": render.system_prompt()}
    ok = {"messages": [sysm, {"role": "user", "content": "Ticket: kit/x\nWrite a parser for INI files."},
                       {"role": "assistant", "content": "a.py\n```python\nx = 1\n```"}], "meta": {"kind": "write"}}
    canary = {**ok, "messages": [sysm, {"role": "user", "content": "Ticket: kit/y MUGGE-BENCH-0042"}, ok["messages"][2]]}
    spike = {**ok, "messages": [*w["messages"][:2], ok["messages"][2]]}
    harvest.write_jsonl(cfg.rows("teacher"), [ok, canary, spike])
    harvest.write_jsonl(cfg.rows("extra"), [])
    out = harvest.read_jsonl(data.build(cfg))
    assert [r["messages"][1]["content"] for r in out] == [ok["messages"][1]["content"]]


def _telemetry(run_dir: Path, run_id: str, unclean_key: str = "") -> list[dict]:
    """A CallRecord per transcript call of run_dir; unclean_key's last calls are unclean."""
    return [{"v": 1, "run": run_id, "ticket": f"{t['key']}#s{t['seed']}", "attempt": i,
             "kind": "write" if i == 1 else "fix", "clean": not (t["key"] == unclean_key and i == len(t["calls"]))}
            for t in harvest.read_jsonl(run_dir / "transcripts.jsonl") for i, _ in enumerate(t["calls"], 1)]


def test_calls_logs_join_their_own_run(run, tmp_path, capsys):
    # two runs with the same tickets and seeds, each with its own calls.jsonl: each run's
    # flags land on that run's calls, not pooled per ticket#seed (which dropped every row)
    a, b = shutil.copytree(run, tmp_path / "A"), shutil.copytree(run, tmp_path / "B")
    harvest.write_jsonl(a / "calls.jsonl", _telemetry(a, "20261015-000001", "spike/c-hash"))
    harvest.write_jsonl(b / "calls.jsonl", _telemetry(b, "20261015-000002", "spike/c-short"))
    logs = [a / "calls.jsonl", b / "calls.jsonl"]
    loaded = harvest.load_run_dirs([a, b])
    harvest.flag_calls(loaded, harvest.read_calls_logs(logs))
    flagged = {(d.name, r["key"]) for d, r, calls in loaded if any(c.get("clean") is False for c in calls)}
    assert flagged == {("A", "spike/c-hash"), ("B", "spike/c-short")}
    rows, st = harvest.harvest(SPIKE, [a, b], calls_logs=logs)
    assert st.dropped["unclean"] == 4 and dict(st.rows) == {"write": 5, "fix": 5}
    # one run reads its own log just as it reads one outside its dir
    rows, st = harvest.harvest(SPIKE, [a], calls_logs=[a / "calls.jsonl"])
    assert st.dropped["unclean"] == 2 and len(rows) == 8
    # a log outside the run dirs naming two run ids for one ticket#seed: ambiguous, so unclean
    pooled = tmp_path / "pooled.jsonl"
    harvest.write_jsonl(pooled, [*_telemetry(a, "20261015-000001"),
                                 *(r for r in _telemetry(b, "20261015-000002") if r["ticket"] == "spike/c-hash#s0")])
    capsys.readouterr()
    loaded = harvest.load_run_dirs([run])
    harvest.flag_calls(loaded, harvest.read_calls_logs([pooled]))
    flagged = {f"{r['key']}#s{r['seed']}" for _, r, calls in loaded if any(c.get("clean") is False for c in calls)}
    assert flagged == {"spike/c-hash#s0"}
    assert all(c["clean"] is False for _, r, calls in loaded if (r["key"], r["seed"]) == ("spike/c-hash", 0) for c in calls)
    out = capsys.readouterr().out
    assert "WARNING: spike/c-hash#s0" in out and "20261015-000002" in out


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


# ---------- free rows ----------

def test_extra_rows_are_held_to_the_wire(tks, tmp_path):
    cfg = data.DataConfig(work=tmp_path, benches=[])
    t = tks["spike/c-hash"]
    path = t["owns"][0]
    good = render.one({"op": "write", "project": t["dir"], "ticket": t["id"], "answer": answer(ref(t))})["messages"]
    loose = [*good[:2], {"role": "assistant", "content": f"Here it is:\n\n**{path}**\n```c\n{ref(t)[path]}```"}]
    relaid = [*good[:2], {"role": "assistant", "content": f"{path}\n```\n{ref(t)[path]}```\n\n"}]
    wrong = render.one({"op": "write", "project": t["dir"], "ticket": t["id"], "answer": answer({path: "int nope;\n"})})["messages"]
    rows = [
        {"messages": good, "meta": {"project": t["dir"], "ticket": t["id"]}},
        {"messages": loose},
        {"messages": relaid},
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
    assert st["loose"] == 1 and st["recanonicalised"] == 1 and st["sandbox_acceptance"] == 1 and st["review"] == 1
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


def test_build_reports_a_failed_tokenizer_check_and_goes_on(tmp_path):
    """build's round trip is a report (bh_run.py warns on it); sft.py's own round trip against
    the served prompt ids is the gate, so a paid session does not stop here."""
    cfg = data.DataConfig(work=tmp_path, benches=[])
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a.py\n```python\nx = 1\n```"}], "meta": {"kind": "write"}}
    harvest.write_jsonl(cfg.rows("teacher"), [row])
    harvest.write_jsonl(cfg.rows("extra"), [])
    data.build(cfg, tokenizer=StubTokenizer(think_in_prompt_only=True))
    assert json.loads(cfg.stats("train").read_text())["tokenizer_check"].startswith("FAILED: tokenizer round trip")


# ---------- the dry run ----------

def test_smoke_dry_run(tmp_path):
    rep = data.smoke(data.DataConfig(work=tmp_path))
    assert rep["references"] == "15/15"
    assert rep["smoke_guard"].startswith("ok")
    assert rep["by_kind"]["teacher/write"] == 15 and rep["by_kind"]["teacher/fix"] == 15
    assert rep["by_kind"].get("repair/fix", 0) > 0 and rep["by_kind"].get("rft/write", 0) > 0
    n, total = map(int, rep["write_rows_rerendered_identical"].split("/"))
    assert n == total > 0


# ---------- export with a kit limit, and the shared sampling default ----------

FAKE_BHDATA = '''
import json, shutil, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == "export"
out = Path(args[args.index("--out") + 1])
spike = Path(__import__("os").environ["SPIKE"])
for split, n in (("train-bench", 25), ("heldout-bench", 22)):
    for i in range(n):
        dst = out / split / f"kit-{i:02d}"
        dst.mkdir(parents=True)
        for part in ("plan.json", "project", "solutions", "hidden"):
            (shutil.copytree if (spike / part).is_dir() else shutil.copyfile)(spike / part, dst / part)
(out / "extra_rows.jsonl").write_text("")
'''


def test_export_limits_kits(tmp_path, monkeypatch):
    pkg = tmp_path / "bh-src" / "bhdata"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "llmap.py").write_text(FAKE_BHDATA)
    monkeypatch.setenv("BHDATA", str(pkg))
    monkeypatch.setenv("SPIKE", str(SPIKE))
    work = tmp_path / "w"
    assert data.main(["export", "--work", str(work), "--limit-kits", "20"]) == 0
    st = json.loads((work / "data" / "stats" / "export.json").read_text())
    assert st["limited"] == {"train": [20, 5], "heldout": [20, 2]}
    assert len(list((work / "data" / "export" / "train-bench").iterdir())) == 20
    assert len(list((work / "data" / "export-unused" / "train-bench").iterdir())) == 5


def test_answer_uses_the_shared_sampling_default(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from bh import sampling
    bodies = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            out = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                              "usage": {"completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        q = tmp_path / "q.jsonl"
        harvest.write_jsonl(q, [{"id": "a", "messages": [{"role": "user", "content": "hi"}]}])
        out = data.answer_requests(q, tmp_path / "a.jsonl", url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                   model="t", samples=1)
    finally:
        srv.shutdown()
    b = bodies[0]
    assert {k: b[k] for k in sampling.DEFAULTS} == sampling.DEFAULTS == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert harvest.read_jsonl(out)[0]["answers"][0]["cut"] is True
    with pytest.raises(ValueError, match="finite"):
        data.answer_requests(q, tmp_path / "b.jsonl", url="http://x", model="t", top_k=0)


def test_teacher_command_reads_the_proxy_calls_log(run, tmp_path):
    """bh_run.py passes --calls-log to teacher, rft and repair-requests: the proxy's cap is the
    one that cuts, so its log is the only place a cut shows."""
    cfg = data.DataConfig(work=tmp_path)
    for w in ("train", "heldout"):
        cfg.bench(w).mkdir(parents=True)
        shutil.copytree(SPIKE / "project", cfg.bench(w) / "spike" / "project")
        for part in ("plan.json", "solutions", "hidden"):
            (shutil.copytree if (SPIKE / part).is_dir() else shutil.copyfile)(SPIKE / part, cfg.bench(w) / "spike" / part)
        data._dump(cfg.stats(f"check-{w}"), {"broken": {}, "excluded": {}})
    log = tmp_path / "calls.jsonl"
    harvest.write_jsonl(log, [{"key": harvest.call_key(t["calls"][-1]["prompt"], t["calls"][-1]["answer"]), "cut": True}
                              for t in harvest.read_jsonl(run / "transcripts.jsonl")])
    assert data.main(["teacher", "--work", str(tmp_path), "--runs", str(run), "--calls-log", str(log)]) == 0
    assert harvest.read_jsonl(cfg.rows("teacher")) == []
    assert json.loads(cfg.stats("teacher").read_text())["dropped"]["cut"] == 10
    assert data.main(["rft", "--work", str(tmp_path), "--runs", str(run), "--calls-log", str(log)]) == 0
    assert json.loads(cfg.stats("rft").read_text())["dropped"]["cut"] == 10


def test_harvest_takes_the_answer_the_review_put_back(run, tmp_path):
    # with MUGGE_REVIEW=on a review's fix can fail and the passing answer is put back: the
    # transcript's last call is then the failing fix; bench marks the kept one (answerAttempt)
    base_rows, _ = harvest.harvest(SPIKE, [run])

    def undo(t):
        if t["key"] != "spike/c-hash":
            return
        for i, c in enumerate(t["calls"], 1):
            c["attempt"] = i
        t["answerAttempt"] = len(t["calls"])
        t["calls"].append({**t["calls"][-1], "answer": "no files here", "attempt": len(t["calls"]) + 1})
    d = _edit(run, tmp_path / "restored", undo)
    rows, st = harvest.harvest(SPIKE, [d])
    assert sorted(r["id"] for r in rows) == sorted(r["id"] for r in base_rows)
    assert not st.dropped
