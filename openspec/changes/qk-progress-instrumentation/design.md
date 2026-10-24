# qk-progress-instrumentation — design

## Sink routing

| sink | consumer | format | when |
|---|---|---|---|
| Lane 0 telemetry CSV | user, planner (artifact read) | one column `progress_json` per row, 2-second cadence matches existing `MemTelemetry` | always, in lane0 mode |
| PTQ breadcrumbs | watchdog (`qkb-stall`), user | `qkb:progress <phase> iter=<N>/<total> rate=<r> eta=<s>` | ptq mode only |

Both sinks are fed from a single `ProgressReporter` object owned by the
`MemTelemetry` thread (lane0) or the main-loop iteration body (ptq), so the
interface is one call site per long-running loop.

## Heartbeat (lane0 only)

The `MemTelemetry` thread already samples every 2 seconds and knows the current
phase. Progress output piggybacks on that cadence: every sample writes a progress
line with the phase and cumulative counts. Critically, a **progress line IS a
heartbeat** — two seconds of silence is a real stall, not a quiet run, because
the telemetry thread writes on its own timer regardless of what the main loop is
doing.

## Sanity-check constants

Measured per-phase rates (seq_len=2048, batch=1, torch fp32, gfx1151):

| phase | rate (s / unit) | unit |
|---|---|---|
| capture | — | per-block forward, negligible |
| stats_shipped (Arm A) | — | one-shot, negligible |
| stats_ce (Arm B) | — | one-shot, negligible |
| ADMM solve | ~89 | per linear (400-iter solve) |
| TuneFP | ~133 | per target factorize |
| STE | ~154 | per linear sign-optimize |

The ETA for a fire is computed as `Σ(phase_rate × count)` using these constants
plus the harness's known geometry. Before rendering, the computed ETA is checked
against a sanity fence: if the predicted ETA exceeds `MAX_SANE_ETA` (default 4
hours for a 20-solve fire), the bar reports `eta=unsafe` instead of a number —
the user sees "not rendering" rather than a lie.