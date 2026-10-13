"""The experiment runner without a VM: the bar keeps an eval's own progress once its
events arrive, and the upload hook runs after every finished step and once at the end
of the run, however the run ends."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runner  # noqa: E402
from runner import Skip, Step, clock_fill, on_line, run_plan  # noqa: E402


def event(stage, status="running", **kw):
    return json.dumps({"stage": stage, "status": status, **kw}) + "\n"


def running(minutes, ago_s, **kw):
    s = Step("bh15 scored", minutes, None, **kw)
    s.state, s.part = "running", "" if "" in minutes else next(iter(minutes))
    s.started = s.part_started = time.time() - ago_s
    return s


def test_eval_events_are_not_overwritten_by_the_clock():
    """An export (no events) then the eval: the clock moves the export against its own
    guess, and once the eval reports 600 of 728 answers the bar keeps that, with the
    export's minutes out of the time left."""
    s = running({"": 10, "eval": 100}, ago_s=300)
    clock_fill([s])
    assert s.frac == pytest.approx(0.5, abs=0.01)  # 5 of the export's 10 minutes, not 5 of 110
    on_line(s, event("eval", answered=600, total=728))
    assert s.part == "eval" and "" in s.finished
    frac = 0.1 + 0.85 * 600 / 728
    assert s.frac == pytest.approx(frac)
    clock_fill([s])  # the old draw() set this to the clock's 5 / 110 = 5%
    assert s.frac == pytest.approx(frac)
    got, left = runner.Run([s]).fraction_and_left()
    assert got == pytest.approx((10 + 100 * frac) / 110)
    assert left == pytest.approx(100 * 60 * (1 - frac), rel=1e-3)  # the eval's rest only


def test_a_finished_or_skipped_first_stage_also_ends_the_eventless_part():
    s = running({"": 10, "eval": 100}, ago_s=60)
    on_line(s, event("eval", "skipped"))
    assert s.finished == {"", "eval"} and s.part == ""
    s.frac = 0.0
    clock_fill([s])
    assert s.frac == 0.0
    assert runner.Run([s]).fraction_and_left()[0] == pytest.approx(1.0)


def test_steps_with_only_an_eventless_part_move_as_before():
    s = running({"": 20}, ago_s=600)
    on_line(s, event("eval", answered=1, total=2))  # a stage the step has no guess for
    clock_fill([s])
    assert s.part == "" and s.frac == pytest.approx(0.5, abs=0.01)
    s = running({"": 20}, ago_s=600, progress=lambda: 0.3)
    clock_fill([s])
    assert s.frac == 0.3
    s = running({"": 1}, ago_s=3600)
    clock_fill([s])
    assert s.frac == 0.95  # the clock never says done


def plan_of(*runs):
    names = "abcdefgh"
    return [Step(names[i], {"": 1}, r) for i, r in enumerate(runs)]


def ok(step):
    pass


def boom(step):
    raise RuntimeError("out of disk")


def skip(step):
    raise Skip("no login")


def record(calls):
    def after_step(step, final):
        calls.append((step.name if step else None, final, step.state if step else None))
    return after_step


def test_upload_after_every_finished_step_and_once_at_the_end(tmp_path, capsys):
    calls = []
    plan = plan_of(ok, skip, ok)
    plan.insert(0, Step("before", {"": 1}, boom, done=lambda: True))  # done in an earlier run
    assert run_plan(plan, lambda: "SUMMARY", tmp_path / "logs", argv=[], after_step=record(calls)) == 0
    assert calls == [("a", False, "done"), ("c", False, "done"), (None, True, None)]
    assert "SUMMARY" in capsys.readouterr().out


def test_a_failed_step_still_uploads_and_the_summary_comes_after(tmp_path, capsys):
    calls = []

    def summary():
        calls.append("summary")
        return "SUMMARY"

    plan = plan_of(ok, boom, ok)
    assert run_plan(plan, summary, tmp_path / "logs", argv=[], after_step=record(calls)) == 1
    assert calls == [("a", False, "done"), ("b", True, "failed"), "summary"]
    assert plan[2].state == "waiting"
    assert "Stopped at b: out of disk" in capsys.readouterr().out


def test_ctrl_c_uploads_then_returns_130(tmp_path):
    calls = []

    def stop(step):
        raise KeyboardInterrupt

    assert run_plan(plan_of(ok, stop, ok), lambda: "", tmp_path / "logs", argv=[], after_step=record(calls)) == 130
    assert calls == [("a", False, "done"), ("b", True, "stopped")]


def test_anything_else_still_uploads(tmp_path):
    calls = []

    def bad_summary():
        raise ValueError("summary broke")

    with pytest.raises(ValueError):
        run_plan(plan_of(ok), bad_summary, tmp_path / "logs", argv=[], after_step=record(calls))
    assert calls == [("a", False, "done"), (None, True, None)]

    def exits(step):
        raise SystemExit(3)

    calls.clear()
    with pytest.raises(SystemExit):
        run_plan(plan_of(exits), lambda: "", tmp_path / "logs", argv=[], after_step=record(calls))
    assert calls == [("a", True, "stopped")]


def test_a_failing_upload_does_not_stop_the_run(tmp_path, capsys):
    calls = []

    def after_step(step, final):
        calls.append((step.name if step else None, final))
        if not final and step.name == "a":
            raise RuntimeError("bucket unreachable")

    assert run_plan(plan_of(ok, ok), lambda: "", tmp_path / "logs", argv=[], after_step=after_step) == 0
    assert calls == [("a", False), ("b", False), (None, True)]
    out = capsys.readouterr().out
    assert "upload after a failed: bucket unreachable" in out
    assert "Uploads that failed" in out and "after a: bucket unreachable" in out


def test_without_a_hook_nothing_changes(tmp_path, capsys):
    plan = plan_of(ok, boom)
    assert run_plan(plan, lambda: "SUMMARY", tmp_path / "logs", argv=[]) == 1
    out = capsys.readouterr().out
    assert "Uploads that failed" not in out and "SUMMARY" in out
    assert [s.state for s in plan] == ["done", "failed"]
