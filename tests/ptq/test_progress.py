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
import time

import pytest

from quantibbler.ptq.progress import (
    MAX_SANE_ETA,
    ProgressReporter,
    _rate_text,
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
    the harness continues -- never halted, never raised. The prediction comes
    from the measured constants (`spec.md:98-102`), so the test injects the table
    that makes it unreasonable rather than fabricating a pace on the clock.
    """
    clock = FakeClock()
    stream = io.StringIO()
    rates = {"ste": {"gfx1151": {"seconds_per_unit": 4000.0}}}
    reporter = ProgressReporter("ste", 20, rate_constants=rates, clock=clock, stream=stream)

    # 4000s per unit => ~21h for the nineteen remaining. Past the 4h ceiling.
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
    # The staleness comparison runs against the machine the constants were
    # measured on, so a different commit or torch marks the first line too. This
    # test is about the pace verdict, not the warning, so it clears the flag.
    reporter._pending_warning = False

    clock.advance(7200.0)  # deliberately irrelevant: the ETA is not derived from elapsed pace
    record = reporter.tick(1)

    assert record is not None
    assert record["eta_sane"] is True
    # Nineteen units remaining at the measured 89s/linear -- the constants, not the pace.
    assert record["eta_s"] == pytest.approx(19 * 89.0)
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
    epoch, rest = crumb.split(" ", 1)
    assert epoch.isdigit(), crumb
    assert rest.startswith("qkb:progress ste iter=2/4"), crumb
    assert "rate=" in rest and "eta=" in rest and "eta_sane=" in rest

    row = json.loads(telemetry.read_text().strip())
    assert row["phase"] == "ste" and row["iter"] == 2
    assert "eta_sane" in row, "telemetry must record eta_sane alongside the ETA"


def test_breadcrumb_line_is_epoch_prefixed_for_the_watchdog(tmp_path) -> None:
    """The watchdog keys liveness off a numeric first field, so the crumb carries one.

    Without the leading epoch, `tools/qkb-stall-watchdog.sh` finds no parseable
    line at all and raises `probe_fault` on a perfectly healthy run -- a fail-safe
    detector firing on success. This is task 5.2's verification, in code.
    """
    breadcrumbs = tmp_path / "crumbs.log"
    reporter = ProgressReporter("admm", 10, clock=FakeClock(), stream=io.StringIO())
    attach_breadcrumbs(reporter, str(breadcrumbs))

    reporter.tick(1, force=True)
    crumb = breadcrumbs.read_text().strip()
    first_field = crumb.split(" ", 1)[0]

    assert first_field.isdigit(), crumb
    assert abs(int(time.time()) - int(first_field)) <= 5, crumb
    assert "eta_sane=" in crumb, "the verdict rides the line the watchdog reads"


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


def test_eta_is_derived_from_the_measured_rate_constants() -> None:
    """The prediction comes from the phase's measured rate, not from elapsed pace.

    Two runs with the same counts but wildly different clocks must predict the
    same completion: that is what makes the number a measurement of the stage
    rather than an extrapolation of whatever this run has done so far.
    """
    rates = {"admm": {"gfx1151": {"seconds_per_unit": 10.0}}}
    slow_clock, fast_clock = FakeClock(), FakeClock()

    slow_stream = io.StringIO()
    slow = ProgressReporter("admm", 10, rate_constants=rates, clock=slow_clock, stream=slow_stream)
    fast = ProgressReporter(
        "admm", 10, rate_constants=rates, clock=fast_clock, stream=io.StringIO()
    )
    slow._pending_warning = fast._pending_warning = False

    slow_clock.advance(5000.0)  # same counts, wildly different observed pace
    fast_clock.advance(5.0)
    slow_record = slow.tick(4)
    fast_record = fast.tick(4)

    assert slow_record is not None and fast_record is not None
    assert slow_record["eta_s"] == pytest.approx(6 * 10.0)
    assert fast_record["eta_s"] == pytest.approx(slow_record["eta_s"]), (
        "same counts, same prediction: the ETA must not follow the observed pace"
    )
    assert slow_record["eta_sane"] is True
    assert "eta=60s" in slow_stream.getvalue()


def test_eta_is_unknown_before_any_unit_completes() -> None:
    """Nothing completed means nothing to predict -- `unknown`, never a number.

    A measured rate is available here, and the prediction would be computable;
    the spec still forbids rendering it before a unit lands, because a predicted
    ETA on an empty run reads as knowledge the reporter does not have.
    """
    stream = io.StringIO()
    rates = {"admm": {"gfx1151": {"seconds_per_unit": 10.0}}}
    reporter = ProgressReporter("admm", 10, rate_constants=rates, clock=FakeClock(), stream=stream)

    record = reporter.heartbeat()

    assert record is not None
    assert record["eta_s"] is None
    assert record["eta_sane"] is False
    assert "eta=unknown" in stream.getvalue()
    assert "eta_sane=false" in stream.getvalue()


def test_stale_constants_warning_is_rendered_on_the_first_line() -> None:
    """The warning reaches the line a reader sees, not just the telemetry row.

    A mismatch means the numbers may not describe this machine; a reader on
    stdout has to be able to see that without opening the artifact, and the line
    must not simultaneously claim its ETA was sane.
    """
    stream = io.StringIO()
    reporter = ProgressReporter("admm", 10, clock=FakeClock(), stream=stream)
    reporter._pending_warning = True
    reporter._stale_keys = ["commit", "torch"]

    first = reporter.tick(1, force=True)
    second = reporter.tick(2, force=True)

    assert first is not None
    assert first["eta_sane"] is False, "a stale prediction may not claim sanity"
    lines = [line for line in stream.getvalue().splitlines() if line]
    assert "constants_stale=true keys=commit,torch" in lines[0], lines[0]
    # `unsafe` is the fence's word and only the fence's: this prediction is
    # fence-clean (89s/linear, nine to go), so a false verdict from staleness
    # renders as the number rather than claiming unreasonableness.
    assert "eta=801s" in lines[0], lines[0]
    assert "eta=unsafe" not in lines[0], lines[0]
    assert "constants_stale" not in (second or {}), "warning is once, not noise"
    assert "constants_stale=true" not in lines[1], lines[1]


def test_rate_flips_to_seconds_per_unit_when_the_phase_is_slow() -> None:
    """Below 1 unit/s the pace reports its reciprocal, in the unit the table uses.

    An ADMM solve at 0.011 units/s is both unreadable and incomparable to the
    measured table, which is seconds per linear; the flip is what makes the line
    and the table the same unit. The record keeps units/s regardless -- this is a
    rendering rule, not a change to what the telemetry records.
    """
    stream = io.StringIO()
    clock = FakeClock()
    reporter = ProgressReporter("admm", 100, clock=clock, stream=stream)
    reporter._pending_warning = False

    clock.advance(89.0)  # one unit in 89s => 1/89 units per second
    record = reporter.tick(1)

    assert record is not None
    assert record["rate"] == pytest.approx(1 / 89.0), "the record keeps units/s"
    assert "rate=89.000s/unit" in stream.getvalue(), stream.getvalue()


def test_rate_does_not_flip_at_one_per_second_or_at_zero() -> None:
    """The boundary is `0 < rate < 1`, so 1/s stays and 0/s cannot divide by zero."""
    edge_stream = io.StringIO()
    clock = FakeClock()
    reporter = ProgressReporter("admm", 100, clock=clock, stream=edge_stream)
    reporter._pending_warning = False

    reporter.tick(0, force=True)  # nothing completed: rate 0
    clock.advance(1.0)
    reporter.tick(2, force=True)  # two units in one second: rate 2/s
    clock.advance(0.0)
    reporter.tick(3, force=True)  # zero elapsed since the last tick: still not a crash

    lines = [line for line in edge_stream.getvalue().splitlines() if line]
    assert "rate=0.000/s" in lines[0], lines[0]
    assert "rate=2.000/s" in lines[1], lines[1]

    fast_stream = io.StringIO()
    fast_clock = FakeClock()
    fast = ProgressReporter("admm", 100, clock=fast_clock, stream=fast_stream)
    fast._pending_warning = False
    fast_clock.advance(1.0)
    fast.tick(1, force=True)  # exactly 1 unit per second: no flip
    assert "rate=1.000/s" in fast_stream.getvalue(), fast_stream.getvalue()


def test_breadcrumb_rate_uses_the_same_flip(tmp_path) -> None:
    """Two sinks, one rule -- and the crumb keeps its own unit-free fast shape."""
    crumbs = tmp_path / "crumbs.log"
    clock = FakeClock()
    reporter = ProgressReporter("admm", 100, clock=clock, stream=io.StringIO())
    attach_breadcrumbs(reporter, str(crumbs))

    clock.advance(89.0)
    reporter.tick(1, force=True)
    crumb = crumbs.read_text().strip()
    assert "rate=89.000s/unit" in crumb, crumb

    fast_crumbs = tmp_path / "fast.log"
    fast_clock = FakeClock()
    fast = ProgressReporter("admm", 100, clock=fast_clock, stream=io.StringIO())
    attach_breadcrumbs(fast, str(fast_crumbs))
    fast_clock.advance(0.1)
    fast.tick(1, force=True)
    # Pinned crumb shape is `rate=<r>`: the fast case carries no unit token, so
    # appending one would be a silent format change the docs do not show.
    assert "rate=10.000 " in fast_crumbs.read_text(), fast_crumbs.read_text()


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (-5.0, "-5.000/s"),  # outside the flip bound: unflipped, not clamped
        (0.0, "0.000/s"),  # no reciprocal requested, so no division to guard
        (1e-9, "1000000000.000s/unit"),
        (0.999999, "1.000s/unit"),  # strict `< 1`: the last sub-second inverts
        (1.0, "1.000/s"),  # exactly 1/s is not slow
        (1000.0, "1000.000/s"),
    ],
)
def test_rate_text_boundaries(rate: float, expected: str) -> None:
    """The flip's edges, pinned in the suite instead of argued in a chat log.

    `bug-hunter`'s adversarial sweep produced this table by hand; the rule's edges
    belong in the gate, so they cannot drift back silently. The subnormal case
    (`1e-320` -> the degenerate `infs/unit`) is deliberately absent: the reporter
    derives its rate from units over positive elapsed monotonic seconds, so no
    caller can reach it, and a branch for an unreachable input is ceremony.
    """
    assert _rate_text(rate, unit_suffix=True) == expected
    assert _rate_text(rate, unit_suffix=False) == expected.removesuffix("/s")
