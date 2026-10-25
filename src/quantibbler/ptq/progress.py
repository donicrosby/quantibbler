"""Pipeline-generic progress reporting.

One interface, three sinks, no mode knowledge at the calling site. A long-running
loop calls `reporter.tick(...)` once per iteration and does not know -- and must
not know -- whether stdout, the telemetry CSV, the breadcrumbs file, or some sink
added later is live. Subsystems attach themselves; the loop stays ignorant.

Three properties are load-bearing, each paid for:

* **Line-oriented, never CR-overwrite.** A tqdm-style bar keeps one line and
  overwrites it, so a run that dies at 60% leaves a bar and no history. Every
  update here is one newline-terminated line, flush-per-line and fsynced, so a
  mid-loop death leaves the forensic trail that tells you how far it got. This
  also makes stdout safe: `container.log` is the witness channel, and a bar would
  destroy it while these lines feed it.
* **Cadence-bounded.** Not per-iteration: 400 ADMM steps at one line each buries
  the tail that matters. Every `PROGRESS_INTERVAL_S` seconds or every
  `PROGRESS_INTERVAL_STEPS` iterations, whichever comes first.
* **ETAs that cannot lie.** A predicted completion is checked against measured
  per-phase rate constants before it is rendered. A bar reading "20 minutes" for
  a stage measured at 21 hours is worse than no bar, because it is misdirection
  the reader has no way to catch. Unreasonable ETAs render as `eta=unsafe
  eta_sane=false` and the run continues -- suppressed, never hidden, never fatal.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from .progress_constants import (
    DEFAULT_TARGET,
    RATE_CONSTANTS,
    constants_stale,
    rate_for,
)

PROGRESS_INTERVAL_S = 5.0
PROGRESS_INTERVAL_STEPS = 10
MAX_SANE_ETA = 4 * 3600.0


def _rate_text(rate: float, *, unit_suffix: bool) -> str:
    """Render the observed pace, flipping to seconds-per-unit when it is slow.

    Below one unit per second the units-per-second number stops being readable:
    an ADMM solve at 0.011/s is a figure a reader has to invert in their head to
    compare against anything, and the measured-constants table it is compared
    against is already in seconds per unit. So a slow phase reports its reciprocal
    with the unit named, and both sinks print the same value from this one rule.
    Exactly 1/s and nothing completed (rate 0) stay in units per second -- the flip
    is strictly for `0 < rate < 1`, which also keeps the reciprocal away from zero.
    `unit_suffix` preserves each sink's existing convention: stdout names the unit
    in the fast case too, the breadcrumb line -- pinned as `rate=<r>` -- does not.
    """
    if 0.0 < rate < 1.0:
        return f"{1.0 / rate:.3f}s/unit"
    return f"{rate:.3f}/s" if unit_suffix else f"{rate:.3f}"


class ProgressReporter:
    """Cadence-bounded progress for one loop, routed to every live sink.

    The calling site constructs this with a phase and a total and calls `tick()`.
    It never passes a sink and never asks which sinks are active -- subsystems
    attach with `register_sink()`, so adding a fourth sink later touches one place
    and no loop.
    """

    def __init__(
        self,
        phase: str,
        total: int | None = None,
        *,
        target: str = DEFAULT_TARGET,
        rate_constants: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        stream: Any = None,
        interval_s: float = PROGRESS_INTERVAL_S,
        interval_steps: int = PROGRESS_INTERVAL_STEPS,
        max_sane_eta: float = MAX_SANE_ETA,
    ) -> None:
        self.phase = phase
        self.total = total
        self.target = target
        # The measured per-phase rates the ETA is derived from (`spec.md:98-102`).
        # Injectable so a test can pin the derivation against a known rate instead
        # of whatever this machine has measured.
        self._rate_constants = RATE_CONSTANTS if rate_constants is None else rate_constants
        self._clock = clock
        self._stream = sys.stdout if stream is None else stream
        self._interval_s = interval_s
        self._interval_steps = interval_steps
        self._max_sane_eta = max_sane_eta

        self._start = clock()
        self._last_emit = self._start
        self._last_emit_iter = 0
        self._current = 0
        self._emitted = 0

        self._stale, self._stale_keys = constants_stale()
        # A constant mismatch means the numbers may not describe this machine, so
        # the first line carries the warning and the run proceeds regardless.
        self._pending_warning = self._stale

        # Stdout is always live: it is the witness channel, readable while the run
        # is alive and still there after it dies. Everything else is opt-in from
        # the subsystem that owns it.
        self._sinks: list[Callable[[dict[str, Any]], None]] = [self._stdout_sink]

    # -- sink registration (called by subsystems, never by loops) ------------

    def register_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self._sinks.append(sink)

    def _stdout_sink(self, record: dict[str, Any]) -> None:
        print(self._render(record), file=self._stream, flush=True)

    def _render(self, record: dict[str, Any]) -> str:
        # Three states, not two: a numeric ETA that passed the fence, one the
        # fence suppressed, and one that cannot be computed yet because no unit
        # has completed. The third must not be rendered as a number -- `eta_s` is
        # None there and formatting it raises, which is the crash this branch
        # exists to prevent. `unsafe` is the fence's word and only the fence's: a
        # prediction that is numerically clean but untrusted (stale constants,
        # `spec.md:144-148`) renders as its number with `eta_sane=false`, so the
        # reader can tell "unreasonable" from "fine, but not trustworthy today".
        if record["eta_s"] is None:
            eta = "unknown"
        elif record["eta_s"] > record["max_sane_eta"]:
            eta = "unsafe"
        else:
            eta = f"{record['eta_s']:.0f}s"
        line = (
            f"progress {record['phase']} iter={record['iter']}/{record['total']} "
            f"rate={_rate_text(record['rate'], unit_suffix=True)} eta={eta} "
            f"eta_sane={str(record['eta_sane']).lower()}"
            + (" complete" if record.get("complete") else "")
        )
        # `spec.md:144-148`: the first line carries the staleness warning itself,
        # not just a telemetry field -- a reader on stdout must see that the
        # constants may not describe this machine. The breadcrumb line keeps its
        # format untouched: the watchdog parses it (task 5.2).
        if record.get("constants_stale"):
            keys = ",".join(record.get("constants_stale_keys", []))
            line += f" constants_stale=true keys={keys}"
        return line

    def _emit(self, record: dict[str, Any]) -> None:
        """Fan out to every live sink, swallowing sink failures individually.

        Instrumentation must never be able to kill a fire. An unwritable probe
        path, a closed stdout, a full disk -- each of those is a reason for one
        sink to go quiet, never a reason for a six-hour solve to die. The other
        sinks still receive the record, so a broken breadcrumb path cannot take
        the witness channel down with it.
        """
        for sink in self._sinks:
            try:
                sink(record)
            except Exception:  # noqa: BLE001 - deliberate: see docstring
                continue

    # -- the one call a loop makes ------------------------------------------

    def tick(self, current: int | None = None, *, force: bool = False) -> dict[str, Any] | None:
        """Record progress; emit to every live sink when the cadence says so.

        Returns the emitted record, or None when this tick was suppressed by the
        cadence -- callers may ignore the return value entirely.
        """
        if current is not None:
            self._current = current
        now = self._clock()

        # The first tick always emits. A 10-step cadence on an 89s-per-linear
        # solver would otherwise say nothing for fifteen minutes, and the entire
        # point is that a reader can tell the phase started.
        due_steps = (self._current - self._last_emit_iter) >= self._interval_steps
        due_time = (now - self._last_emit) >= self._interval_s
        if not (force or self._emitted == 0 or due_steps or due_time):
            return None

        record = self._record(now)
        self._emit(record)
        self._last_emit = now
        self._last_emit_iter = self._current
        self._emitted += 1
        return record

    def complete(self, current: int | None = None) -> dict[str, Any] | None:
        """Emit a final line marking the loop finished.

        Without this, a loop that completed looks exactly like one killed at its
        last emitted tick -- the crash-forensics scenario depends on the reader
        being able to tell those apart from the artifact alone, and only a
        completion marker makes that possible.
        """
        # The caller passes the count actually reached. Do NOT substitute
        # `total`: a solve that converged at iteration 4 finished at 4, and
        # writing 400 would be the same class of lie the ETA check exists to
        # prevent.
        if current is not None:
            self._current = current
        record = self._record(self._clock())
        record["complete"] = True
        self._emit(record)
        self._emitted += 1
        return record

    def _record(self, now: float) -> dict[str, Any]:
        elapsed = now - self._start
        done = self._current
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0

        eta_s: float | None = None
        # `spec.md:98-102`: the predicted completion is derived from the measured
        # per-phase rate constants, not from this run's observed pace -- an
        # instrument that extrapolates from whatever pace it has seen is exactly
        # the bar that reads "20 minutes" for a 21-hour stage. A phase or target
        # with no measured entry has no prediction, and nothing completed yet has
        # nothing to predict from; both render `unknown` rather than a number.
        # `eta_sane` means "a numeric ETA was rendered and it passed the check".
        eta_sane = False
        phase_rate = rate_for(self.phase, self.target, self._rate_constants)
        if self.total is not None and done > 0 and phase_rate is not None:
            eta_s = (self.total - done) * phase_rate
            eta_sane = eta_s <= self._max_sane_eta

        record: dict[str, Any] = {
            "phase": self.phase,
            "iter": done,
            "total": self.total if self.total is not None else -1,
            "rate": rate,
            "eta_s": eta_s,
            "eta_sane": eta_sane,
            # The fence travels with the record so a reader -- and the breadcrumb
            # renderer -- can tell a number the fence suppressed from one that is
            # merely untrusted, instead of reading every false verdict as
            # "unreasonable".
            "max_sane_eta": self._max_sane_eta,
            "elapsed_s": elapsed,
        }
        if self._pending_warning:
            record["constants_stale"] = True
            record["constants_stale_keys"] = list(self._stale_keys)
            # `spec.md:144-148`: a constant mismatch is itself a reason the
            # prediction cannot be trusted, so the same line that carries the
            # warning reports `eta_sane=false`. The run still proceeds.
            record["eta_sane"] = False
            self._pending_warning = False
        return record

    # -- heartbeat ----------------------------------------------------------

    def heartbeat(self) -> dict[str, Any]:
        """Emit a line from a timer thread, independent of main-loop progress.

        This is the liveness channel: it writes on its own schedule whether or not
        the loop advanced, so two missing samples mean the thread is dead rather
        than the loop being slow. The phase and counts reflect the last known
        state of the main loop, which is exactly what a reader needs to tell
        "running slowly" from "hung".
        """
        return self.tick(force=True)  # type: ignore[return-value]


def breadcrumb_sink(path: str) -> Callable[[dict[str, Any]], None]:
    """A sink writing the watchdog's breadcrumb format.

    Every line starts with its epoch, because the watchdog's stall predicate
    requires a numeric first field (`awk '$1 ~ /^[0-9]+$/'` in
    `tools/qkb-stall-watchdog.sh`): without it a healthy run trips that script's
    no-parseable-line probe fault instead of reading as live, and the watchdog's
    own stall detector goes dark. The verdict travels with the line (`eta_sane=`)
    so its reader does not have to cross-reference stdout to know whether the ETA
    was trusted -- which is what makes the heartbeat a signal with a consumer
    instead of an instrument nobody reads.
    """

    def sink(record: dict[str, Any]) -> None:
        # The same three states as the stdout renderer: `unsafe` means the fence
        # suppressed the number, not that the verdict was false.
        eta = (
            "unknown"
            if record["eta_s"] is None
            else (
                "unsafe"
                if record["eta_s"] > record["max_sane_eta"]
                else f"{record['eta_s']:.0f}"
            )
        )
        line = (
            f"{int(time.time())} qkb:progress {record['phase']} "
            f"iter={record['iter']}/{record['total']} "
            f"rate={_rate_text(record['rate'], unit_suffix=False)} "
            f"eta={eta} eta_sane={str(record['eta_sane']).lower()}\n"
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    return sink


def jsonl_sink(path: str) -> Callable[[dict[str, Any]], None]:
    """A sink appending one JSON object per line, for the telemetry CSV column."""

    def sink(record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return sink


def attach_telemetry(reporter: ProgressReporter, path: str) -> None:
    """Attach the telemetry sink when a telemetry subsystem is present."""
    reporter.register_sink(jsonl_sink(path))


def attach_breadcrumbs(reporter: ProgressReporter, path: str) -> None:
    """Attach the breadcrumbs sink when a watchdog is armed."""
    reporter.register_sink(breadcrumb_sink(path))
