# qk-progress-instrumentation — proposal

## Why

Two decisive-fire deaths on the target-load path were completely invisible until the
container exited — the fire's only signal between weight-load and completion was
silence. The user asked for "iteration rate and time to complete the current loop,
similar to the PyTorch weight-load bar," and that feature is also the watchdog's
missing liveness channel: today a lane0 hang at minute 20 looks identical to a
running fire until the 60-minute wall net stops it.

## What

Add cadence-bounded, line-oriented progress output to every long-running harness
loop (capture, statistics, solver). One progress interface writes to two sinks:
the lane0 telemetry CSV and the ptq breadcrumbs file, so a single implementation
feeds both the user's ETA bar and the watchdog's heartbeat.

## Progress interface

```
{phase: "stats", iter: 3, total: 20, rate: 1.2, eta_s: 14, elapsed_s: 5}
```

- **Line-oriented, never CR-overwrite.** Every update writes one line. A mid-loop
  death leaves a forensic trail, not a single bar frozen at 60%.
- **Cadence-bounded.** Updates emit every `PROGRESS_INTERVAL_S` seconds (default 5)
  or every `PROGRESS_INTERVAL_STEPS` iterations (default 10), whichever comes
  sooner, never per-iteration — a 400-step ADMM loop produces ~80 lines, not 400.
- **Flushed and fsynced per line**, matching `MemTelemetry`'s existing contract,
  so a killed process doesn't lose the last-written update.
- **ETA sanity-checked.** The predicted completion time is compared against
  measured per-phase rate constants; a bar predicting 20 minutes when the
  measured pace says 21 hours is surfaced as a defect, not rendered silently.