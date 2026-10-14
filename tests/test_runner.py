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
from runner import Run, Skip, Step, clock_fill, on_line, run_plan  # noqa: E402


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


class Clock:
    """runner's time module, moved by hand: time() is now, and steps advance it."""

    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def sleep(self, s):
        pass

    def strftime(self, fmt):
        return time.strftime(fmt)


def test_steps_found_done_leave_the_speed_alone(tmp_path, monkeypatch, capsys):
    """orchestration#7: a step a restore finished (Done) and one that found its work done
    in seconds count as done but not in the speed: counted, a restored 4 h reading made
    every later guess half."""
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    runs = []
    monkeypatch.setattr(runner, "Run", lambda steps: runs.append(Run(steps)) or runs[-1])

    def restored(step):
        raise runner.Done("done before it started: a restore finished it")

    def quick(step):
        clock.now += 5

    def slow(step):
        clock.now += 40 * 60

    plan = [Step("reading", {"": 240}, restored), Step("check", {"": 30}, quick), Step("train", {"": 20}, slow)]
    assert run_plan(plan, lambda: "", tmp_path / "logs", argv=[]) == 0
    assert [s.state for s in plan] == ["done"] * 3
    assert runs[0].done_est == 20 and runs[0].done_real == pytest.approx(40)
    assert "✓ 1. reading (done before it started: a restore finished it)" in capsys.readouterr().out


def test_waiting_steps_take_their_guess_again_when_a_step_starts(tmp_path, monkeypatch):
    """orchestration#7: a guess an earlier step re-bases (gen-measure's rate) reaches the
    plan that measured it; a guess that cannot be read keeps the old one, and a step that
    is done keeps the minutes it was counted with."""
    measured, seen = {}, {}

    def measure(step):
        measured["rate"] = 2.0

    def look(step):
        seen[step.name] = dict(step.minutes)

    def broken():
        raise FileNotFoundError("no eval.json yet")

    plan = [Step("gen-measure", {"": 130}, measure, refresh=lambda: {"": 999}),
            Step("e2e", {"": 60}, look, refresh=lambda: {"": 60 / measured.get("rate", 1)}),
            Step("onpolicy", {"rollouts": 50, "train": 25}, look, refresh=broken)]
    assert run_plan(plan, lambda: "", tmp_path / "logs", argv=[]) == 0
    assert seen == {"e2e": {"": 30.0}, "onpolicy": {"rollouts": 50, "train": 25}}
    assert plan[0].minutes == {"": 999}  # read before it ran: it started with it
    plan[0].state = "done"
    plan[0].refresh = lambda: {"": 1}
    runner.refresh_minutes(plan)
    assert plan[0].minutes == {"": 999}


def test_a_resumed_eval_shows_its_answers_and_the_rate_since_it_began(monkeypatch):
    """orchestration#7: an eval resumed at 600 of 728 answers. The bar shows 600/728, and
    the time left comes from the answers made since its first event, not from the
    fraction done (which once said ~1 min left for 128 answers of ~1 min each)."""
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    s = Step("reading", {"eval": 300}, None)
    s.state, s.part, s.started, s.part_started = "running", "eval", clock.now, clock.now
    run = runner.Run([s])
    on_line(s, event("eval", answered=600, total=728))
    assert s.counts == [600, clock.now, 600, 728]
    clock.now += 60
    on_line(s, event("eval", answered=601, total=728))
    left_early = run.fraction_and_left()[1]  # under 2 min of rate: the guess's rest
    assert left_early == pytest.approx(300 * 60 * (1 - s.frac), rel=1e-6)
    clock.now += 540
    on_line(s, event("eval", answered=610, total=728))
    assert run.fraction_and_left()[1] == pytest.approx((728 - 610) * 600 / 10)
    assert "· eval 610/728" in run.line()
    clock.now += 60
    on_line(s, event("eval", answered=5, total=728))  # the next candidate in the same stage: a new count
    assert s.counts == [5, clock.now, 5, 728] and "· eval 5/728" in run.line()
    # A new part starts its own count.
    s.minutes = {"eval": 300, "sandbox": 10}
    on_line(s, event("sandbox", answered=1, total=50))
    assert s.counts == [1, clock.now, 1, 50] and "· sandbox 1/50" in run.line()
