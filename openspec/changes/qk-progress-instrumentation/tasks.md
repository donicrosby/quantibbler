# qk-progress-instrumentation — tasks

## 1. `ProgressReporter` class

- [ ] 1.1 Create `src/quantibbler/ptq/progress.py` with `ProgressReporter(phase, total, rate_constants, sink)`
- [ ] 1.2 Implement cadence-bounded update: emit when `elapsed >= PROGRESS_INTERVAL_S` OR `iter % PROGRESS_INTERVAL_STEPS == 0`
- [ ] 1.3 Implement ETA computation: `remaining = (total - current) * (elapsed / current)`, sanity-checked against constants
- [ ] 1.4 Implement dual-sink dispatch: `sink="csv"` → telemetry column, `sink="breadcrumbs"` → `qkb:progress ...`
- [ ] 1.5 Heartbeat: `MemTelemetry` thread calls `reporter.tick()` with current phase/iter on every sample

## 2. Progress constants

- [ ] 2.1 Create `src/quantibbler/ptq/progress_constants.py` with the measured rate dict
- [ ] 2.2 Stale-constants detection: compare recorded date/commit/ROCm/PT version at init

## 3. Integration into existing loops

- [ ] 3.1 Wire `ProgressReporter` into `run_arms`'s statistics phase (one reporter per arm)
- [ ] 3.2 Wire `ProgressReporter` into `lb_admm.py`'s solve loop (400 iterations → ~80 lines)
- [ ] 3.3 Wire `ProgressReporter` into `block_recon.py`'s TuneFP and STE stages

## 4. Lane 0 telemetry CSV

- [ ] 4.1 Add `progress_json` column to `MemTelemetry` CSV header
- [ ] 4.2 Write progress line on every telemetry sample during lane0 mode

## 5. PTQ breadcrumbs

- [ ] 5.1 Write `qkb:progress ...` on every cadence tick during ptq mode
- [ ] 5.2 Verify watchdog's existing stall detector parses the line unchanged

## 6. Tests

- [ ] 6.1 Unit: `ProgressReporter` cadence logic (5s / 10 steps, whichever first)
- [ ] 6.2 Unit: ETA computation and sanity check (suppress at 4h)
- [ ] 6.3 Unit: dual-sink dispatch writes correct format to each sink
- [ ] 6.4 Integration: a 10-step mock loop writes ~2 lines, not 10
- [ ] 6.5 Integration: killed process leaves forensic tail (last-written line visible)
- [ ] 6.6 Unit: stale-constants warning emitted on version mismatch