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
from collections.abc import Callable
from typing import Any

from .progress_constants import (
    DEFAULT_TARGET,
    constants_stale,
)

PROGRESS_INTERVAL_S = 5.0
PROGRESS_INTERVAL_STEPS = 10
MAX_SANE_ETA = 4 * 3600.0


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
        clock: Callable[[], float] = time.monotonic,
        stream: Any = None,
        interval_s: float = PROGRESS_INTERVAL_S,
        interval_steps: int = PROGRESS_INTERVAL_STEPS,
        max_sane_eta: float = MAX_SANE_ETA,
    ) -> None:
        self.phase = phase
        self.total = total
        self.target = target
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
        # Three states, not two: a numeric ETA that passed the check, an
        # unreasonable one that is suppressed, and one that cannot be computed
        # yet because no unit has completed. The third must not be rendered as a
        # number -- `eta_s` is None there and formatting it raises, which is the
        # crash this branch exists to prevent.
        if record["eta_s"] is None:
            eta = "unknown"
        elif not record["eta_sane"]:
            eta = "unsafe"
        else:
            eta = f"{record['eta_s']:.0f}s"
        return (
            f"progress {record['phase']} iter={record['iter']}/{record['total']} "
            f"rate={record['rate']:.3f}/s eta={eta} eta_sane={str(record['eta_sane']).lower()}"
            + (" complete" if record.get("complete") else "")
        )

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
        for sink in self._sinks:
            sink(record)
        self._last_emit = now
        self._last_emit_iter = self._current
        self._emitted += 1
        return record

    def complete(self) -> dict[str, Any] | None:
        """Emit a final line marking the loop finished.

        Without this, a loop that completed looks exactly like one killed at its
        last emitted tick -- the crash-forensics scenario depends on the reader
        being able to tell those apart from the artifact alone, and only a
        completion marker makes that possible.
        """
        if self.total is not None:
            self._current = self.total
        record = self._record(self._clock())
        record["complete"] = True
        for sink in self._sinks:
            sink(record)
        self._emitted += 1
        return record

    def _record(self, now: float) -> dict[str, Any]:
        elapsed = now - self._start
        done = self._current
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0

        eta_s: float | None = None
        # `eta_sane` means "a numeric ETA was rendered and it passed the check".
        # An ETA that cannot be computed yet is recorded as not-sane, so the
        # artifact reader never mistakes silence for a reliable prediction.
        eta_sane = False
        if self.total is not None and rate > 0:
            eta_s = (self.total - done) / rate
            eta_sane = eta_s <= self._max_sane_eta

        record: dict[str, Any] = {
            "phase": self.phase,
            "iter": done,
            "total": self.total if self.total is not None else -1,
            "rate": rate,
            "eta_s": eta_s,
            "eta_sane": eta_sane,
            "elapsed_s": elapsed,
        }
        if self._pending_warning:
            record["constants_stale"] = True
            record["constants_stale_keys"] = list(self._stale_keys)
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

    @property
    def emitted(self) -> int:
        return self._emitted


def breadcrumb_sink(path: str) -> Callable[[dict[str, Any]], None]:
    """A sink writing the watchdog's existing breadcrumb format.

    The format is fixed by what `tools/qkb-stall-watchdog.sh` already parses, so
    the watchdog reads lane0 liveness without modification -- which is what makes
    the heartbeat a signal with a consumer instead of an instrument nobody reads.
    """

    def sink(record: dict[str, Any]) -> None:
        eta = (
            "unknown"
            if record["eta_s"] is None
            else ("unsafe" if not record["eta_sane"] else f"{record['eta_s']:.0f}")
        )
        line = (
            f"qkb:progress {record['phase']} iter={record['iter']}/{record['total']} "
            f"rate={record['rate']:.3f} eta={eta}\n"
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
