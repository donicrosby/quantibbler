# qk-progress-instrumentation — tasks

## 1. `ProgressReporter` class

- [x] 1.1 Create `src/quantibbler/ptq/progress.py` with `ProgressReporter(phase, total, *, target, rate_constants, clock, stream, interval_s, interval_steps, max_sane_eta)`; sinks attach via `register_sink`, never a constructor argument
- [x] 1.2 Implement cadence-bounded update: emit when `elapsed >= PROGRESS_INTERVAL_S` OR the step count since the last emit `>= PROGRESS_INTERVAL_STEPS`, whichever comes first (the first tick always emits; a step delta, not `iter % N`, so a time-cadence emission cannot re-emit inside the step window)
- [x] 1.3 Implement ETA computation: `fraction_remaining = (total - done) / total; eta_s = fraction_remaining * work_units * rate_for(phase, target)` — checked against the per-phase rate constants and fenced by `MAX_SANE_ETA` (per-loop model, design.md Decision 2)
- [x] 1.4 Implement dual-sink dispatch: `attach_telemetry` → telemetry column, `attach_breadcrumbs` → `qkb:progress ...`
- [ ] 1.5 Heartbeat: `MemTelemetry` thread calls `reporter.tick()` with current phase/iter on every sample — **deferred: the `MemTelemetry` host was retired by e5c6939 (Lane 1 rewrite); `ProgressReporter.heartbeat()` exists but has zero production callers at HEAD. Needs a timer thread that exists.**

## 2. Progress constants

- [x] 2.1 Create `src/quantibbler/ptq/progress_constants.py` with the measured rate dict
- [x] 2.2 Stale-constants detection: compare recorded date/commit/ROCm/PT version at init

## 3. Integration into existing loops

- [ ] 3.1 Wire `ProgressReporter` into `run_arms`'s statistics phase (one reporter per arm) — **moot: `run_arms` and the lane0 harness were retired by e5c6939; the replacement `calibration.collect_stats` is a one-shot pass that does not need a progress reporter.**
- [x] 3.2 Wire `ProgressReporter` into `lb_admm.py`'s solve loop (400 iterations → ~40 lines; ruling: the cadence rule produces ~40 at 0.2s/iter — the spec number was wrong, not the code)
- [x] 3.3 Wire `ProgressReporter` into `block_recon.py`'s TuneFP and STE stages

## 4. Telemetry CSV integration

- [ ] 4.1 Add `progress_json` column to telemetry CSV header — **deferred: the `MemTelemetry` host was retired by e5c6939; the `jsonl_sink` and `attach_telemetry` functions exist in `progress.py` but have zero production callers at HEAD. Wire-up needs a telemetry collector that registers the sink.**
- [ ] 4.2 Write progress line on every telemetry sample during solve mode — **deferred: same host dependency as 4.1.**

## 5. PTQ breadcrumbs

- [ ] 5.1 Write `qkb:progress ...` on every cadence tick during ptq mode — **deferred: the `breadcrumb_sink` and `attach_breadcrumbs` functions exist in `progress.py` but have zero production callers at HEAD. The existing `_note_block_progress` writes stage markers to the breadcrumb file directly (not through ProgressReporter). Wire-up: call `attach_breadcrumbs` at reporter construction time in the three existing loops (admm, tunefp, ste).**
- [x] 5.2 Verify watchdog's existing stall detector parses the line unchanged

## 6. Tests

- [x] 6.1 Unit: `ProgressReporter` cadence logic (5s / 10 steps, whichever first)
- [x] 6.2 Unit: ETA computation and sanity check (suppress at 4h)
- [x] 6.3 Unit: dual-sink dispatch writes correct format to each sink
- [x] 6.4 Integration: a 10-step mock loop writes 1 line, not 10 (the cadence bounds output; the test name still says `to_two_lines` — non-blocking, recorded)
- [x] 6.5 Integration: killed process leaves forensic tail (last-written line visible)
- [x] 6.6 Unit: stale-constants warning emitted on version mismatch
