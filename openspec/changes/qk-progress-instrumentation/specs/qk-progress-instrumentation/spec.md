# qk-progress-instrumentation — spec delta

## ADDED Requirements

### Requirement: progress_line_oriented

Every long-running harness loop SHALL emit progress updates as line-oriented
output — one line per update, newline-terminated, never CR-overwrite. A
mid-loop death SHALL leave a forensic trail of every update written before the
crash, flush-per-line and fsynced, matching `MemTelemetry`'s existing contract.

#### Scenario: progress_output_survives_crash

- **WHEN** a fire is killed mid-loop
- **THEN** the progress output contains every line written before the kill signal arrived
- **AND** the final line is the last successfully written update, not a partially-overwritten bar
- **AND** the absence of a "complete" line combined with the visible iteration count lets the reader determine exactly how far the loop progressed

### Requirement: progress_cadence_bounded

Progress updates SHALL emit at a cadence, not per-iteration. The default cadence
is every `PROGRESS_INTERVAL_S` seconds (default 5) or every
`PROGRESS_INTERVAL_STEPS` iterations (default 10), whichever comes sooner. A
400-step ADMM loop SHALL produce ~80 lines, not 400, and a fast 20-step loop
SHALL produce a line roughly every 5 seconds.

#### Scenario: admm_loop_not_verbose

- **WHEN** an ADMM solve runs for 400 iterations at ~0.2s per iteration
- **THEN** the progress output contains roughly 80 lines, not 400
- **AND** every line carries the current iteration, total, rate, and ETA

### Requirement: progress_sink_routing

One `ProgressReporter` interface SHALL feed three sinks without duplication of
the calling site. No long-running loop SHALL check a mode flag to decide which
sink to write to — the reporter dispatches internally.

#### Scenario: stdout_sink_active_in_all_modes

- **WHEN** any fire runs
- **THEN** the `ProgressReporter` writes to stdout — live-readable,
  line-oriented, into `container.log` — the witness channel the user actually
  watches while the run is alive
- **AND** stdout is active in ALL modes, never conditional on lane0 vs ptq

#### Scenario: lane0_writes_csv_column

- **WHEN** the harness runs in lane0 mode
- **THEN** every progress update appears in the telemetry CSV as a
  `progress_json` column with one JSON object per line
- **AND** the `progress_json` column is always present in the CSV header when
  the progress reporter is active

#### Scenario: ptq_writes_breadcrumbs

- **WHEN** the harness runs in ptq mode
- **THEN** every progress update appears in the breadcrumbs file as
  `qkb:progress <phase> iter=<N>/<total> rate=<r> eta=<s>`
- **AND** the watchdog's existing stall detector parses the breadcrumb lines
  without modification

### Requirement: progress_heartbeat_lane0

The `MemTelemetry` thread SHALL emit a progress line on every sample (every 2
seconds by default) regardless of whether the main loop has advanced. The phase
and cumulative counts SHALL reflect the last known state of the main loop.
Silence for longer than one sample interval SHALL be a reliable stall signal —
the thread writes on its own timer, so two seconds without a progress line means
the telemetry thread itself is dead.

#### Scenario: heartbeat_consumer_is_the_watchdog

- **WHEN** the lane0 harness runs
- **THEN** the heartbeat SHALL write to the ptq breadcrumbs file in addition to
  the telemetry CSV, so the existing watchdog's stall detector — which already
  parses `qkb:progress` lines — reads liveness without modification
- **AND** without this consumer, the signal is available and unread, and a lane0
  hang at minute 20 stays invisible: the `max|s|=0.0` class — an instrument that
  reports clean because nothing reads it

#### Scenario: progress_heartbeat_during_slower_phases

- **WHEN** a solver phase takes 89 seconds per linear
- **THEN** the telemetry CSV contains roughly 45 progress lines during that
  phase (one every 2 seconds)
- **AND** every line carries the phase name `solver` and the latest known
  iteration count
- **AND** the watchdog or reader can distinguish "solver is running slowly"
  from "the process is hung" by the presence or absence of progress lines

### Requirement: progress_eta_sanity_checked

Before rendering an ETA, the `ProgressReporter` SHALL check the predicted
completion time against per-phase rate constants measured on the target
hardware at the protocol geometry, and SHALL record the verdict as a boolean
`eta_sane` field in the telemetry.

#### Scenario: eta_suppressed_when_unreasonable

- **WHEN** the progress bar computes an ETA of 21 hours for a 20-solve fire
  (measured pace ~89s/linear, expected ~30 minutes)
- **THEN** the rendered progress line reads `eta=unsafe eta_sane=false`
- **AND** `MAX_SANE_ETA` defaults to 4 hours for a 20-solve lane0 fire
- **AND** the harness does not halt or raise — the run continues with the
  unreasonable ETA suppressed rather than misinforming the user

#### Scenario: eta_sane_recorded_in_telemetry

- **WHEN** any progress line renders a numeric ETA or suppresses one
- **THEN** the telemetry records `eta_sane` as a boolean alongside it, so the
  artifact reader can distinguish "ETAs were reliable" from "ETAs were
  suppressed" without re-deriving the rate constants

### Requirement: progress_rate_constants_durable_source

The per-phase rate constants used for sanity-checking SHALL live in
`src/quantibbler/ptq/progress_constants.py` as a module-level dict keyed by
phase name, with one value per hardware target. The constants SHALL be
re-measured after any change to the solver backend, the attention implementation,
or the GPU driver version, and the measurement date SHALL be recorded alongside
the constant.

#### Scenario: constants_out_of_date_detected

- **WHEN** the `ProgressReporter` initializes
- **THEN** it compares the recorded measurement date against the current
  environment (commit, ROCm version, PyTorch version)
- **AND** if any of the three differs from the recorded measurement, the first
  progress line SHALL include `eta_sane=false` and a warning that constants may
  be stale — but the run proceeds
