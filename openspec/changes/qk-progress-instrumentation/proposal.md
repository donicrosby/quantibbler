# qk-progress-instrumentation — proposal

## Why

Two fires died on paths that were completely invisible until the container
exited — the only signal between startup and completion was silence. The user
asked for "iteration rate and time to complete the current loop, similar to the
PyTorch weight-load bar." That feature is also the watchdog's missing liveness
channel: today a hang at minute 20 looks identical to a running fire until the
wall net stops it.

## What

A pipeline-generic `ProgressReporter` utility (`src/quantibbler/ptq/progress.py`)
that adds cadence-bounded, line-oriented progress output to every long-running
harness loop — capture, statistics, solver, block recon, TuneFP, STE — across
every pipeline stage and mode. One interface feeds three sinks (stdout for live
monitoring, telemetry CSV for artifact reads, breadcrumbs for the watchdog), and
the calling site never checks which are active.

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
