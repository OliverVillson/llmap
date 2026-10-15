"""scripts/bh_run.py (the Centipede train harness) without a GPU: every profile dry-run end
to end with the mocks (scripts/bh/mock.py), the s1 -> laptop -> s2 hand-over through one
bucket, the guards, and the pieces that are real code: the recording proxy, the bench
reading, the row guards, the footprint, and (with torch) the SFT trainer on a tiny hybrid
model."""

import gzip
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bh import bench as B  # noqa: E402
from bh import config as C  # noqa: E402
from bh import footprint as F  # noqa: E402
from bh import rows as R  # noqa: E402
from bh.wire import SYSTEM, call_key  # noqa: E402


def bh_run():
    sys.modules.pop("bh_run", None)
    return importlib.import_module("bh_run")


def dry(tmp_path, profile, *more):
    return bh_run().main([profile, "--dry-run", str(tmp_path / "dry"), *more])


def machine(tmp_path, profile):
    return tmp_path / "dry" / profile / "work"


def read(p):
    return json.loads(Path(p).read_text())


# ---------- the plan, dry ----------

def test_g0_end_to_end(tmp_path, capsys):
    assert dry(tmp_path, "g0") == 0
    w = machine(tmp_path, "g0")
    t = read(w / "teacher" / "teacher.json")
    assert t["chosen"] == "qwen3.5-2b"
    # data.py's layout under W/data; the guarded rows the SFT reads in W/rows
    assert (w / "data" / "export" / "train-bench").is_dir() and (w / "data" / "stats" / "check-train.json").exists()
    rows = [json.loads(x) for x in (w / "rows" / "train.jsonl").read_text().splitlines()]
    assert rows and all(r["messages"][0]["content"] == SYSTEM for r in rows)
    assert {r["meta"]["kind"] for r in rows} <= {"write", "fix", "write-from-fix"}
    assert {r["meta"]["source"] for r in rows} == {"teacher", "bhdata-extra"}
    # the extra rows' review row is left out (BH0.1 is worker-only); broken tickets make no rows
    assert all(r["meta"]["kind"] != "review" for r in rows)
    broken = set(read(w / "data" / "stats" / "check-train.json")["broken"])
    assert broken and not broken & {r["meta"]["key"] for r in rows}
    st = read(w / "rows" / "rows.json")
    assert st["build"]["tokenizer_check"] == "ok" and st["harvest"]["answers"] > 0
    # every bench call went through the proxy with the config's sampling
    proxy = read(w / "answers" / "shard-000" / "proxy" / "proxy.json")
    assert proxy["settings"]["top_k"] == 20 and proxy["settings"]["thinking"] is False
    assert proxy["counts"]["calls"] > 0 and proxy["counts"]["top_k_set"] == 0
    # the G0 profile: LoRA SFT, tiny caps, the 0.8B ships at Q8_0
    cfg = read(w / "config.json")
    assert cfg["sft_mode"] == "lora" and cfg["max_call_tokens"] == 1024 and cfg["limit_kits"] == 20
    ship = read(w / "footprint" / "sft" / "footprint.json")["ship"]
    assert ship["name"] == "BH0.1-1.0" and ship["type"] == "Q8_0"
    assert (w / "ship" / "sft" / "BH0.1-1.0.gguf").exists()
    assert read(w / "score" / "sft" / "score.json")["verdict"] in ("MEETS", "MISSES", "TOO CLOSE")
    assert (w / "sft" / "round2" / "model" / "sft.json").exists()
    out = capsys.readouterr().out
    assert "Finished" in out and "Stopped at" not in out


def test_sessions_hand_over_through_the_bucket(tmp_path):
    """s1 on one machine, the laptop on another, s2 on a third: each finds what it needs in
    the one bucket."""
    assert dry(tmp_path, "s1") == 0
    s1 = machine(tmp_path, "s1")
    sft = read(s1 / "sft" / "sft" / "model" / "sft.json")
    assert sft["kl"] > 0  # the Qwen path: logit KL from the stored top-20
    assert (s1 / "rows" / "topk.jsonl.gz").exists()
    with gzip.open(s1 / "rows" / "topk.jsonl.gz", "rt") as f:
        top = [json.loads(x) for x in f]
    assert top and len(top[0]["top"][0]) == 20
    st = read(s1 / "rows" / "rows.json")
    assert 0 < st["topk_rows"] <= st["teacher_rows"]  # write-from-fix rows have no logged call of their own
    assert dry(tmp_path, "laptop") == 0
    lap = machine(tmp_path, "laptop")
    assert (lap / "sft" / "sft" / "model" / "sft.json").exists()  # restored, not trained here
    ship = read(lap / "footprint" / "sft" / "footprint.json")["ship"]
    assert ship["type"] == "Q6_K" and ship["footprint"] <= 2e9  # Q8_0 does not fit 2 GB
    assert (lap / "round2" / "samples" / "transcripts.jsonl").exists()
    assert not (lap / "score").exists()  # scoring is the eval laptop's
    assert dry(tmp_path, "eval") == 0  # another machine: the ship comes from the bucket
    ev = machine(tmp_path, "eval")
    assert read(ev / "score" / "sft" / "score.json")["verdict"] in ("MEETS", "MISSES", "TOO CLOSE")
    assert (ev / "ship" / "sft" / f"{ship['name']}.gguf").exists()
    assert dry(tmp_path, "s2") == 0
    s2 = machine(tmp_path, "s2")
    r2 = read(s2 / "round2" / "round2.json")
    assert r2["own_repair"]["rows"] > 0 and r2["sources"].get("student-rft", 0) > 0
    assert {"teacher", "own-repair", "student-rft"} <= set(r2["sources"])
    # data.py answer (real code) went through the proxy: finite top_k, thinking off, the cap
    proxy = read(s2 / "round2" / "repairs" / "proxy" / "proxy.json")
    assert proxy["counts"]["calls"] > 0 and proxy["settings"]["top_p"] == 0.8
    assert not (s2 / "teacher" / "g2").exists()  # G2 is session 1's; session 2 reuses the choice
    assert (s2 / "sft" / "round2" / "model" / "sft.json").exists()
    assert dry(tmp_path, "laptop", "--set", "student=round2") == 0
    assert read(lap / "footprint" / "round2" / "footprint.json")["ship"]["name"].startswith("BH0.1.1-")


def test_a_role_served_by_another_machine(tmp_path):
    """Two laptops: the eval laptop's llama-server serves the student for the first one's
    round-2 sampling; no server starts here and every call goes there through the proxy."""
    from bh import mock as M
    other = M.FakeUpstream()
    try:
        backends = json.dumps([{"name": "laptop2", "url": f"{other.url}/v1", "roles": ["student"], "model": "bh"}])
        assert dry(tmp_path, "s1") == 0
        assert dry(tmp_path, "laptop", "--set", f"backends={backends}") == 0
    finally:
        other.close()
    lap = machine(tmp_path, "laptop")
    assert other.requests  # laptop2 answered the student's calls
    st = read(lap / "round2" / "samples" / "stats.json")
    assert st["proxy"]["calls"] == len(other.requests)
    with pytest.raises(ValueError, match="served by both"):
        C.load("laptop", {"BH_BACKENDS": json.dumps([{"name": "a", "url": "http://a/v1", "roles": ["eval"]},
                                                      {"name": "b", "url": "http://b/v1", "roles": ["eval"]}])})


def test_minicpm_path(tmp_path):
    """MiniCPM5-2B: no logit KL (another tokenizer), min_p 0 on every call, ships at a 4-bit quant."""
    assert dry(tmp_path, "s1", "--set", "base=minicpm5-2b") == 0
    w = machine(tmp_path, "s1")
    assert read(w / "sft" / "sft" / "model" / "sft.json")["kl"] == 0.0
    assert not (w / "rows" / "topk.jsonl.gz").exists()
    smoke_proxy = read(w / "smoke" / "sft" / "proxy" / "proxy.json")
    assert smoke_proxy["settings"]["min_p"] == 0.0
    assert dry(tmp_path, "laptop", "--set", "base=minicpm5-2b") == 0
    assert read(machine(tmp_path, "laptop") / "footprint" / "sft" / "footprint.json")["ship"]["type"] == "Q4_K_S"


def test_g2_falls_back_to_the_next_teacher(tmp_path):
    assert dry(tmp_path, "s1", "--set", 'extra={"dry_g2_fail": ["Qwen3.8-27B"]}') == 0
    t = read(machine(tmp_path, "s1") / "teacher" / "teacher.json")
    assert [x["teacher"] for x in t["tried"]] == ["qwen3.8-27b-nvfp4", "qwen3.6-35b-a3b-nvfp4"]
    assert t["chosen"] == "qwen3.6-35b-a3b-nvfp4" and not t["tried"][0]["pass"]


def test_no_teacher_passes_stops_the_run(tmp_path, capsys):
    assert dry(tmp_path, "s1", "--set", 'extra={"dry_g2_fail": ["Qwen"]}') == 1
    assert "no teacher of the ladder passed G2" in capsys.readouterr().out


def test_smoke_falls_back_to_llama_server(tmp_path):
    assert dry(tmp_path, "s1", "--set", 'extra={"dry_vllm_student_fail": true}') == 0
    assert read(machine(tmp_path, "s1") / "smoke" / "sft" / "smoke.json")["served"] == "llama-server"


def test_optional_steps_never_fail_the_job(tmp_path, capsys):
    """A failing vocab cut, tokenizer check or smoke bench is reported; the run goes on."""
    assert dry(tmp_path, "g0", "--set", 'extra={"dry_fail": ["vocab-run", "tokcheck"]}') == 0
    assert dry(tmp_path, "s1", "--set", 'extra={"dry_fail": ["bench"]}') == 1  # G2 itself cannot run: fatal
    out = capsys.readouterr().out
    assert "no teacher of the ladder passed G2" in out


def test_broken_smoke_blocks_export(tmp_path, capsys):
    assert dry(tmp_path, "s1") == 0
    sm = machine(tmp_path, "s1") / "smoke" / "sft" / "smoke.json"
    d = read(sm)
    d["verdict"] = "BROKEN: no ticket solved and most answers wrote no file; do not export"
    sm.write_text(json.dumps(d))
    bucket = tmp_path / "dry" / "bucket" / "blackhole" / "bh01" / "smoke-sft.json"
    bucket.write_text(json.dumps(d))
    assert bh_run().main(["laptop", "export", "--dry-run", str(tmp_path / "dry")]) == 1
    assert "--force exports anyway" in capsys.readouterr().out
    assert bh_run().main(["laptop", "export", "--dry-run", str(tmp_path / "dry"), "--force"]) == 0


def test_resume_skips_done_stages(tmp_path, capsys):
    assert dry(tmp_path, "g0") == 0
    capsys.readouterr()
    assert dry(tmp_path, "g0") == 0
    out = capsys.readouterr().out
    for s in ("data", "SFT", "export", "footprint"):
        assert any(line.startswith("✓") and s in line and "done before" in line for line in out.splitlines()), s


def test_b200_refuses_laptop_stages_and_unpinned_versions(tmp_path, capsys):
    r = bh_run()
    assert r.main(["s1", "export", "--dry-run", str(tmp_path / "d")]) == 0
    assert "does not run on a b200" in capsys.readouterr().out
    assert r.main(["s1", "setup", "--dry-run", str(tmp_path / "d2"), "--set", "vllm=vllm>=0.31"]) == 1
    assert "refusing a paid session without exact pins" in capsys.readouterr().out


def test_session_clock_skips_what_cannot_finish(tmp_path, capsys):
    """A session with no time left skips its time-capped steps instead of overrunning."""
    assert dry(tmp_path, "s1", "--set", "session_minutes=1") == 1
    out = capsys.readouterr().out
    assert "leave no time for it" in out


def test_round2_ships_beside_round1_and_eval_scores_it(tmp_path):
    """Round 2 can ship under round 1's name (same version and footprint): each model has its
    own ship folder and bucket key, so the eval laptop scores the round-2 GGUF, not round 1's."""
    for prof in (["s1"], ["laptop"], ["eval"], ["s2"], ["laptop", "--set", "student=round2"],
                 ["eval", "--set", "student=round2"]):
        assert dry(tmp_path, *prof) == 0, prof
    bucket = tmp_path / "dry" / "bucket" / "blackhole" / "bh01" / "ship"
    names = {p.relative_to(bucket).as_posix() for p in bucket.rglob("*.gguf")}
    r1 = read(machine(tmp_path, "laptop") / "footprint" / "sft" / "footprint.json")["ship"]["name"]
    r2 = read(machine(tmp_path, "laptop") / "footprint" / "round2" / "footprint.json")["ship"]["name"]
    assert names == {f"sft/{r1}.gguf", f"round2/{r2}.gguf"}
    assert r1.startswith("BH0.1-") and r2.startswith("BH0.1.1-") and r2[len("BH0.1.1"):] == r1[len("BH0.1"):]
    ev = machine(tmp_path, "eval")
    assert "/ship/round2/" in read(ev / "score" / "round2" / "score.json")["gguf"]
    assert "/ship/sft/" in read(ev / "score" / "sft" / "score.json")["gguf"]


def test_eval_waits_for_the_ship_from_the_other_laptop(tmp_path):
    """laptop and eval at the same time on one bucket: eval starts first, waits, then scores
    the GGUF the laptop's footprint uploads."""
    import subprocess
    import time
    d = tmp_path / "dry"
    assert dry(tmp_path, "s1") == 0
    cmd = [sys.executable, str(ROOT / "scripts" / "bh_run.py")]
    ev = subprocess.Popen(cmd + ["eval", "--dry-run", str(d)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2)
    assert ev.poll() is None  # still waiting: nothing shipped yet
    assert subprocess.run(cmd + ["laptop", "--dry-run", str(d)], capture_output=True).returncode == 0
    out = ev.communicate(timeout=120)[0]
    assert ev.returncode == 0, out
    assert "waiting for footprint-sft.json from another machine" in out and "SCORE BH0.1-" in out


def test_a_failed_handover_upload_is_retried_in_order(tmp_path, monkeypatch, capsys):
    """The ship's GGUF fails to upload: footprint-sft.json (which tells eval it is there) is
    held back too, both stay pending, and the next run sends them in order."""
    from bh import exec as E
    assert dry(tmp_path, "s1") == 0
    real, sent = E.Bucket.put, []

    def flaky(self, src, name):
        if name.startswith("ship/"):
            return False
        sent.append(name)
        return real(self, src, name)

    monkeypatch.setattr(E.Bucket, "put", flaky)
    assert dry(tmp_path, "laptop") == 0  # the run finishes; the upload failure is reported
    out = capsys.readouterr().out
    assert "hand-over file is still not in the bucket" in out
    assert "footprint-sft.json" not in sent
    bucket = tmp_path / "dry" / "bucket" / "blackhole" / "bh01"
    assert not (bucket / "footprint-sft.json").exists()
    assert bh_run().main(["laptop", "status", "--dry-run", str(tmp_path / "dry")]) == 0
    assert "not in the bucket yet" in capsys.readouterr().out
    monkeypatch.setattr(E.Bucket, "put", real)
    assert dry(tmp_path, "laptop") == 0
    assert (bucket / "footprint-sft.json").exists() and list((bucket / "ship" / "sft").glob("*.gguf"))
    assert read(machine(tmp_path, "laptop") / ".uploaded" / "pending.json")["files"] == []


def test_extra_rows_are_optional(tmp_path, capsys):
    assert dry(tmp_path, "g0", "--set", 'extra={"dry_fail": ["data-extra"]}') == 0
    assert "data.py extra failed" in capsys.readouterr().out
    rows = [json.loads(x) for x in (machine(tmp_path, "g0") / "rows" / "train.jsonl").read_text().splitlines()]
    assert {r["meta"]["source"] for r in rows} == {"teacher"}


def test_round2_rft_restored_model_still_gets_its_smoke(tmp_path):
    import shutil
    for prof in ("s1", "laptop", "s2"):
        assert dry(tmp_path, prof) == 0
    s2 = machine(tmp_path, "s2")
    shutil.rmtree(s2 / "sft" / "round2")
    shutil.rmtree(s2 / "smoke" / "round2")
    (tmp_path / "dry" / "bucket" / "blackhole" / "bh01" / "smoke-round2.json").unlink()
    assert bh_run().main(["s2", "round2-rft", "--dry-run", str(tmp_path / "dry")]) == 0
    assert (s2 / "smoke" / "round2" / "smoke.json").exists()


def test_every_model_call_has_finite_top_k_and_min_p_0(tmp_path, monkeypatch):
    """Every request the dry run's model server sees, from every stage (bench runs through
    the proxy, data.py answer through it too): sampling.py's values, thinking off, the cap."""
    from bh import mock as M
    from bh import sampling as S
    seen = []
    real = M.FakeUpstream.chat
    monkeypatch.setattr(M.FakeUpstream, "chat", lambda self, body: (seen.append(body), real(self, body))[1])
    assert dry(tmp_path, "g0") == 0
    assert len(seen) > 50
    for b in seen:
        assert (b["top_k"], b["top_p"], b["temperature"], b["min_p"]) == (S.TOP_K, S.TOP_P, S.TEMPERATURE, S.MIN_P)
        assert b["chat_template_kwargs"]["enable_thinking"] is False and 0 < b["max_tokens"] <= 1024


def test_watchdog_waits_for_the_first_progress_event_and_steps_stop_at_their_limit(tmp_path, monkeypatch):
    """sft.py reads on the CPU before it loads the model: the idle-GPU watch starts at its
    first progress event, not at launch. A step with a time limit stops there."""
    import time
    from bh import exec as E
    monkeypatch.setattr(E, "WATCH_EVERY", 0.2)
    cfg = C.load("s1", {"NVME": str(tmp_path)}, None, ["gpu_watch_minutes=0.01"])
    ex = E.Exec(cfg, reading=lambda: (0, 100))
    event = '{"stage": "train", "status": "running", "pct": 1}'
    t = time.monotonic()
    with pytest.raises(E.GpuIdle):  # idle after the event: stopped
        ex.run(None, E.Cmd(f"sleep 1.5; echo '{event}'; sleep 30", "sft", gpu=True, watch_from_event=True))
    assert 1.5 < time.monotonic() - t < 10
    ex.run(None, E.Cmd("sleep 2", "sft", gpu=True, watch_from_event=True))  # no event yet: never judged
    with pytest.raises(E.StepTimeout):
        ex.run(None, E.Cmd("sleep 30", "bench", timeout_min=0.02))


def test_every_dry_command_is_valid_bash_and_has_a_mock(tmp_path, monkeypatch):
    """The dry run fails on a command without a mock or with bad syntax; this pins that
    every stage's commands were exercised in G0."""
    r = bh_run()
    cfg = C.load("g0", {"NVME": str(tmp_path / "n"), "BH_WORK": str(tmp_path / "w"), "BH_LOGS": str(tmp_path / "l"),
                        "BH_BUCKET": str(tmp_path / "b")})
    from bh import mock as M
    from bh.exec import Exec
    up = M.FakeUpstream()
    mk = M.Mock(cfg, up)
    ex = Exec(cfg, dry=True, mock=mk, reading=lambda: (90, 1))
    run = r.Run(cfg, ex, url=up.url)
    mk.run = run
    try:
        from runner import run_plan
        assert run_plan(run.steps(list(r.PLAN["g0"])), run.summary, Path(cfg.logs), ["run"]) == 0
    finally:
        up.close()
    kinds = {c.kind for c in ex.commands}
    assert {"vllm-check", "data-export", "data-check", "bench", "data-teacher", "data-extra", "data-build", "sft",
            "gguf-convert", "imatrix", "quantize", "footprint", "data-repair-requests", "data-answer", "data-repair",
            "data-rft", "serve-vllm", "serve-llama"} <= kinds
    # data.py's own command line takes every data command the harness sends (its functions
    # stubbed: this checks the arguments, not the work)
    from bh import data as D
    got = []
    for fn in ("export", "check_benches", "teacher_rows", "extra_rows", "repair_requests", "answer_requests", "repair",
               "rft_rows", "build"):
        monkeypatch.setattr(D, fn, lambda *a, fn=fn, **k: got.append((fn, a, k)))
    seen = set()
    for c in ex.commands:
        if c.kind.startswith("data-"):
            words = M.shell_words(c)
            assert words[:2] == ["python", "scripts/bh/data.py"]
            seen.add(words[2])
            assert D.main(words[2:]) == 0
    assert seen == {"export", "check", "teacher", "extra", "build", "repair-requests", "answer", "repair", "rft"}
    sft = next(c for c in ex.commands if c.kind == "sft")
    assert "--mode lora" in sft.sh and "--topk" in sft.sh and "--probe" in sft.sh
    bench = next(c for c in ex.commands if c.kind == "bench")
    assert "--thinking off" in bench.sh and "--top-k 20" in bench.sh and bench.env["MUGGE_TIMEOUT_S"] == "3600"
    # the teacher server: 16K-or-less slots, prefix caching, top-20 log-probs
    serve = next(c for c in ex.commands if c.kind == "serve-vllm")
    assert "--enable-prefix-caching" in serve.sh and "--max-logprobs 20" in serve.sh


def test_command_line_spans_status_and_usage(tmp_path, capsys):
    r = bh_run()
    assert r.stage_names([], "s1") == list(r.PLAN["s1"])
    assert r.stage_names(["sft..smoke"], "s1") == ["sft", "smoke"]
    assert r.stage_names(["teacher-answers.."], "s1") == ["teacher-answers", "sft", "smoke"]
    assert r.stage_names(["..data", "round2"], "s2") == ["setup", "data", "round2-sample", "round2-repair", "round2-rft"]
    with pytest.raises(SystemExit, match="not in the s1 plan"):
        r.stage_names(["export.."], "s1")
    with pytest.raises(SystemExit) as e:
        r.main(["--help"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("usage: bh_run.py PROFILE")
    assert len(r.USAGE.splitlines()) <= 24  # one screen
    with pytest.raises(SystemExit) as e:
        r.main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit, match="unknown profile"):
        r.main(["all"])
    # a span on its own, then status: the finished stages show as done
    assert r.main(["g0", "..data", "--dry-run", str(tmp_path / "dry")]) == 0
    capsys.readouterr()
    assert r.main(["g0", "status", "--dry-run", str(tmp_path / "dry")]) == 0
    out = capsys.readouterr().out
    done = [line for line in out.splitlines() if line.endswith("[done]")]
    assert len(done) == 1 and "data:" in done[0] and "BH0.1 train harness, profile g0" in out  # setup re-checks
    assert "last run (" in out and "finished" in out


# ---------- config ----------

def test_config_profiles_and_overrides(tmp_path):
    c = C.load("s1", {"NVME": "/n"}, None, ["samples=4", "teachers=qwen3.6-35b-a3b-fp8", "thinking=false"])
    assert c.samples == 4 and c.teachers == ("qwen3.6-35b-a3b-fp8",) and c.session_minutes == 200.0
    assert c.kd and c.base_model.teacher_tokenizer
    with pytest.raises(ValueError, match="top_k"):
        C.load("s1", {}, None, ["top_k=0"])
    with pytest.raises(ValueError, match="unknown"):
        C.load("s1", {}, None, ["nope=1"])
    assert C.pins_problems(C.load("s1", {})) == []
    assert C.pins_problems(C.load("s1", {}, None, ["vllm=vllm"]))
    assert not C.load("s1", {}, None, ["base=minicpm5-2b"]).kd


# ---------- the proxy ----------

def test_proxy_forces_sampling_and_logs_cut_calls(tmp_path):
    from bh import mock as M
    from bh.proxy import recording
    import urllib.request
    up = M.FakeUpstream()
    try:
        with recording(up.url, tmp_path / "p", top_k=20, top_p=0.8, temperature=0.7, thinking=False, max_tokens=4096,
                       min_p=0.0, logprobs=20) as (url, rec):
            for i in range(64):  # u61 is one the fake server cuts at the cap
                body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"u{i}"}],
                        "max_tokens": 60000, "top_k": 0, "chat_template_kwargs": {"enable_thinking": True}}
                req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                             headers={"content-type": "application/json"})
                reply = json.loads(urllib.request.urlopen(req).read())
                assert "logprobs" not in reply["choices"][0]  # the engine sees a plain answer
    finally:
        up.close()
    sent = up.requests[0]
    assert sent["top_k"] == 20 and sent["min_p"] == 0.0 and sent["max_tokens"] == 4096
    assert sent["chat_template_kwargs"] == {"enable_thinking": False} and sent["top_logprobs"] == 20
    calls = [json.loads(x) for x in (tmp_path / "p" / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 64 and rec.counts["top_k_set"] == 64 and rec.counts["thinking_set"] == 64
    assert any(c["cut"] for c in calls) and all(c["cut"] == (c["finish"] == "length") for c in calls)
    top = [json.loads(x) for x in (tmp_path / "p" / "topk.jsonl").read_text().splitlines()]
    answer = M.answer_for("u0", None)
    assert top[0]["key"] == call_key("u0", answer) and len(top[0]["ids"]) == len(answer)


# ---------- bench reading ----------

def test_interval_and_verdict():
    rs = [{"key": f"t{i}", "solved": i < 36} for i in range(57)]  # 36 of 57: 63%
    point, low, high = B.interval(rs)
    assert point == 63.2 and 50 < low < point < high
    assert B.verdict(low, high, 50) == "MEETS"
    assert B.verdict(30, 49, 50) == "MISSES" and B.verdict(40, 60, 50) == "TOO CLOSE"
    # seeds of one ticket average before the interval
    two = [{"key": "a", "solved": True}, {"key": "a", "solved": False}, {"key": "b", "solved": True}]
    assert B.interval(two)[0] == 75.0


# ---------- row guards ----------

def row(answer="a.py\n```python\nx = 1\n```", user="Ticket: t", **kw):
    return {"messages": [
        {"role": "system", "content": kw.pop("system", SYSTEM)}, {"role": "user", "content": user},
        {"role": "assistant", "content": answer}], "meta": {"kind": kw.pop("kind", "write"), "source": "teacher", **kw}}


def test_row_guards():
    assert R.drop_reason(row()) is None
    assert R.drop_reason(row(cut=True)) == "cut"
    assert R.drop_reason(row(forced=True)) == "forced"
    assert R.drop_reason(row("<think>hm</think>a.py\n```python\nx\n```")) == "thinking"
    assert R.drop_reason(row("no files here")) == "parse"
    assert R.drop_reason(row(system="You are helpful.")) == "system"
    assert R.drop_reason(row(kind="review")) == "kind"
    assert R.drop_reason(row(kind="write-from-fix")) is None
    assert R.drop_reason(row("a.py\n```python\n# MUGGE-BENCH-123\n```")) == "canary"
    keep, dropped = R.filter_rows([row(), row(), row(cut=True, user="Ticket: u")])
    assert len(keep) == 1 and dropped == {"dupe": 1, "cut": 1}


def test_drop_warning_over_20_percent():
    # harvest's counts (data.py stats/teacher.json)
    assert R.drop_warning({"answers": 900, "cut_or_forced": 0, "parser_drops": 670}, 0.2).startswith("670 of 900")
    assert R.drop_warning({"answers": 100, "cut_or_forced": 5, "parser_drops": 5}, 0.2) is None


# ---------- footprint ----------

LOG = """load_tensors:        CUDA0 model buffer size =  1234.50 MiB
load_tensors:   CPU_Mapped model buffer size =   300.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =    96.00 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    18.00 MiB
llama_context:      CUDA0 compute buffer size =   300.25 MiB
llama_context:  CUDA_Host compute buffer size =    40.00 MiB
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
"""


def test_footprint_buffers_and_name():
    b = F.buffers(LOG)
    assert set(b) == {"CUDA0 model", "CPU_Mapped model", "CUDA0 KV", "CUDA0 RS", "CUDA0 compute", "CUDA_Host compute",
                      "CUDA_Host output"}
    total = sum(b.values())
    assert abs(total / 2 ** 20 - 1989.33) < 0.01
    assert F.name("0.1", 1_210_000_000) == "BH0.1-1.3"  # never claims less than measured
    assert F.name("0.1", 1_300_000_000) == "BH0.1-1.3"
    assert F.name("0.2", 1_999_999_999) == "BH0.2-2.0"


# ---------- the trainer, with torch (skipped without it) ----------

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def tiny_model(d: Path, hybrid: bool = True):
    """A tiny model dir with a byte-level BPE tokenizer and a Qwen-style template: Qwen3-Next
    (Gated DeltaNet in 3 of 4 layers, like Qwen3.5) when hybrid, else Qwen2."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = [SYSTEM, "Ticket: t\nFiles you own", "a.py\n```python\nx = 1\n```", "def f():\n    return 2\n"] * 50
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|im_start|>", "<|im_end|>",
                                                                                         "<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    if hybrid:
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        cfg = Qwen3NextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=64, num_hidden_layers=4,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4,
                              linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                              num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                              shared_expert_intermediate_size=32, mlp_only_layers=[0, 1, 2, 3], full_attention_interval=4)
        model = Qwen3NextForCausalLM(cfg)
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=64,
                                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return tok


def sft_rows(tmp_path, tok, n=6):
    rows = [row(f"src/m{i}.py\n```python\ndef f():\n    return {i}\n```", id=f"r{i}") for i in range(n)]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    topk = tmp_path / "topk.jsonl"
    with open(topk, "w") as f:
        for r in rows[: n // 2]:
            ans = r["messages"][2]["content"]
            ids = tok(ans, add_special_tokens=False)["input_ids"]
            f.write(json.dumps({"key": call_key(r["messages"][1]["content"], ans), "ids": ids,
                                "top": [[i, (i + 1) % len(tok)] + list(range(18)) for i in ids],
                                "lp": [[-0.1, -2.5] + [-9.0] * 18 for _ in ids]}) + "\n")
    return p, topk


def run_sft(argv):
    sys.modules.pop("bh.sft", None)
    from bh import sft
    return sft.train(sft.parser().parse_args(argv))


def test_sft_full_with_topk_kl(tmp_path):
    base = tmp_path / "base"
    tok = tiny_model(base, hybrid=False)
    rows, topk = sft_rows(tmp_path, tok)
    rep = run_sft(["--base", str(base), "--rows", str(rows), "--out", str(tmp_path / "out"), "--mode", "full",
                   "--epochs", "1", "--lr", "1e-3", "--batch-tokens", "200", "--topk", str(topk), "--kd-weight", "0.5",
                   "--max-len", "512"])
    assert rep["examples"]["kd_rows"] == 3 and rep["kl"] > 0 and rep["ce"] > 0 and rep["roundtrip"]["ok"]
    assert (tmp_path / "out" / "config.json").exists() and (tmp_path / "out" / "sft.json").exists()
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(tmp_path / "out")  # a plain HF checkpoint


def test_sft_lora_reaches_every_deltanet_layer(tmp_path):
    base = tmp_path / "base"
    tok = tiny_model(base, hybrid=True)
    rows, _ = sft_rows(tmp_path, tok)
    rep = run_sft(["--base", str(base), "--rows", str(rows), "--out", str(tmp_path / "out"), "--mode", "lora",
                   "--lora-r", "4", "--epochs", "1", "--lr", "1e-3", "--batch-tokens", "400", "--max-len", "512"])
    assert {"in_proj_qkvz", "in_proj_ba", "out_proj", "q_proj"} <= set(rep["lora"]["targets"])
    assert rep["lora"]["layers"] == 4 and rep["arch"] == "Qwen3NextForCausalLM"


def test_tokenize_row_masks_the_prompt_as_served(tmp_path):
    tok = tiny_model(tmp_path / "m", hybrid=False)
    r = row("a.py\n```python\nx = 1\n```")
    t = R.tokenize_row(tok, r["messages"])
    prompt = tok.decode(t["prompt"])
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # thinking off, as the server renders
    assert tok.decode(t["answer"]) == r["messages"][2]["content"] + "<|im_end|>"
    assert R.roundtrip(tok, r)["ok"]
    with pytest.raises(ValueError, match="differ from the server's"):
        R.roundtrip(tok, r, served_prompt_ids=t["prompt"][:-1] + [0])
