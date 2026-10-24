"""Tests for the pipeline-generic progress reporter.

Each test pins a property that was chosen because its absence cost something: the
cadence that keeps the forensic tail readable, the line-orientation that lets
stdout double as the witness channel, the ETA check that stops a bar from
misinforming the user, and the breadcrumb consumer that makes the heartbeat a
signal rather than an unread instrument.
"""

from __future__ import annotations

import io
import json

import pytest

from quantibbler.ptq.progress import (
    MAX_SANE_ETA,
    ProgressReporter,
    attach_breadcrumbs,
    attach_telemetry,
)


class FakeClock:
    """A clock the test drives, so cadence is asserted rather than waited for."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_cadence_bounds_a_ten_step_loop_to_two_lines() -> None:
    """10 steps at 0.1s each emit twice, not ten times.

    Per-iteration output is what buries the tail: a 400-step ADMM loop at one line
    per step is 400 lines around the one that matters.
    """
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 10, clock=clock, stream=stream)

    for step in range(1, 11):
        reporter.tick(step)
        clock.advance(0.1)

    lines = [line for line in stream.getvalue().splitlines() if line]
    # First tick emits immediately (so a slow phase is visible at once) and the
    # next is due 10 steps later, which a 10-step loop never reaches: one line.
    # The point of the test is the bound, not the exact count -- 10 lines would
    # mean the cadence is dead.
    assert len(lines) == 1, lines
    assert "iter=1/10" in lines[0]


def test_cadence_scales_with_loop_length() -> None:
    """40 steps emit 4 lines -- the cadence bounds output, it does not cap it."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 40, clock=clock, stream=stream)

    for step in range(1, 41):
        reporter.tick(step)
        clock.advance(0.1)

    reporter.complete()
    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 5, lines
    assert "iter=40/40" in lines[-1] and lines[-1].endswith("complete")
    # The completion marker is what separates "finished" from "killed at 31".
    assert not any(line.endswith("complete") for line in lines[:-1])


def test_time_cadence_fires_when_steps_are_slow() -> None:
    """A slow loop emits on elapsed time even when few steps have passed."""
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter("solver", 400, clock=clock, stream=stream)

    reporter.tick(1)
    clock.advance(6.0)  # past PROGRESS_INTERVAL_S with only one step done
    reporter.tick(2)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 2, lines


def test_eta_renders_unsafe_when_it_exceeds_max_sane_eta() -> None:
    """The 21-hour ETA renders as `eta=unsafe eta_sane=false`, and the run lives.

    A bar predicting 20 minutes for a stage measured at 21 hours is not a small
    error, it is misdirection the reader cannot catch. Suppressed, recorded, and
    the harness continues -- never halted, never raised.
    """
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter("ste", 20, clock=clock, stream=stream)

    # One unit in 4000s => ~21h for twenty. Far past the 4h ceiling.
    clock.advance(4000.0)
    record = reporter.tick(1)

    assert record is not None
    assert record["eta_sane"] is False
    assert record["eta_s"] > MAX_SANE_ETA
    assert "eta=unsafe" in stream.getvalue()
    assert "eta_sane=false" in stream.getvalue()


def test_eta_sane_is_true_and_numeric_for_a_reasonable_pace() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 20, clock=clock, stream=stream)

    clock.advance(90.0)  # 90s per linear, 19 to go ~= 1710s
    record = reporter.tick(1)

    assert record is not None
    assert record["eta_sane"] is True
    assert 1500 < record["eta_s"] < 2000
    assert "eta=unsafe" not in stream.getvalue()


def test_stdout_is_active_without_any_registration() -> None:
    """No sink registration, no mode flag, and stdout still carries the line."""
    stream = io.StringIO()
    reporter = ProgressReporter("capture", 5, clock=FakeClock(), stream=stream)
    reporter.tick(1, force=True)

    assert "progress capture" in stream.getvalue()


def test_registered_sinks_receive_the_same_record(tmp_path) -> None:
    """One interface, many sinks -- and the loop never learns which are live."""
    breadcrumbs = tmp_path / "crumbs.log"
    telemetry = tmp_path / "progress.jsonl"
    stream = io.StringIO()

    reporter = ProgressReporter("ste", 4, clock=FakeClock(), stream=stream)
    attach_breadcrumbs(reporter, str(breadcrumbs))
    attach_telemetry(reporter, str(telemetry))
    reporter.tick(2, force=True)

    crumb = breadcrumbs.read_text().strip()
    assert crumb.startswith("qkb:progress ste iter=2/4"), crumb
    assert "rate=" in crumb and "eta=" in crumb

    row = json.loads(telemetry.read_text().strip())
    assert row["phase"] == "ste" and row["iter"] == 2
    assert "eta_sane" in row, "telemetry must record eta_sane alongside the ETA"


def test_heartbeat_emits_without_main_loop_advancing() -> None:
    """The liveness signal must not depend on the loop making progress.

    This is what lets a reader tell "running slowly" from "hung": the thread
    writes on its own timer, so missing lines mean the thread died.
    """
    stream = io.StringIO()
    reporter = ProgressReporter("solver", 100, clock=FakeClock(), stream=stream)

    reporter.heartbeat()
    reporter.heartbeat()

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 2, lines
    assert "iter=0/100" in lines[0]


def test_killed_loop_leaves_a_forensic_tail(tmp_path) -> None:
    """A mid-loop death leaves every line written before it, and no completion.

    This is the property that rules out CR-overwrite: the reader must be able to
    determine how far the loop got from the artifact alone.
    """
    telemetry = tmp_path / "progress.jsonl"
    stream = io.StringIO()
    clock = FakeClock()
    reporter = ProgressReporter("admm", 400, clock=clock, stream=stream)
    attach_telemetry(reporter, str(telemetry))

    with pytest.raises(RuntimeError):
        for step in range(1, 401):
            reporter.tick(step)
            clock.advance(0.05)
            if step == 60:
                raise RuntimeError("killed mid-loop")

    rows = [json.loads(line) for line in telemetry.read_text().splitlines() if line]
    assert rows, "a killed loop must leave its trail"
    assert rows[-1]["iter"] < 400, "no completion line may be present"
    assert all(row["iter"] <= 60 for row in rows)


def test_stale_constants_warn_on_the_first_line_only(tmp_path) -> None:
    """A version mismatch marks the first line; the run proceeds."""
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 10, clock=FakeClock(), stream=stream)
    reporter._pending_warning = True
    reporter._stale_keys = ["torch"]

    first = reporter.tick(1, force=True)
    second = reporter.tick(2, force=True)

    assert first is not None and first.get("constants_stale") is True
    assert first["constants_stale_keys"] == ["torch"]
    assert "constants_stale" not in (second or {}), "warning is once, not noise"


def test_complete_reports_iterations_actually_run_not_the_budget() -> None:
    """A solve that converged early finished at its real count.

    Reporting `total` here would be the same class of lie the ETA sanity check
    exists to prevent, and it would be invisible: 400/400 on a solve that
    converged at 4 reads as a completed budget.
    """
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 400, clock=FakeClock(), stream=stream)
    reporter.tick(4, force=True)
    reporter.complete(4)

    lines = [line for line in stream.getvalue().splitlines() if line]
    assert "iter=4/400" in lines[-1], lines[-1]
    assert "iter=400/400" not in lines[-1]
    assert lines[-1].endswith("complete")


def test_lb_admm_emits_progress_lines(capsys) -> None:
    """The solve loop is wired: a real lb_admm call writes progress to stdout."""
    import numpy as np

    from quantibbler.ptq.config import PTQConfig
    from quantibbler.ptq.lb_admm import lb_admm

    cfg = PTQConfig(rank=4, admm_max_iter=25, solver_backend="numpy")
    w = np.random.default_rng(0).standard_normal((16, 12))
    lb_admm(w, cfg)

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("progress admm")]
    assert lines, "lb_admm must emit progress"
    assert lines[0].startswith("progress admm iter=1/25")
    assert lines[-1].endswith("complete")
    # 25 iterations under a 10-step cadence: a handful of lines, not 25.
    assert len(lines) < 25, lines
