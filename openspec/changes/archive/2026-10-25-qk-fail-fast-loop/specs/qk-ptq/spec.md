## ADDED Requirements

### Requirement: The pipeline SHALL offer a per-block progress probe against which prefix perplexity can be gated

`PTQConfig` SHALL expose a `progress_probe` callable field, default `None`, invoked by the sequential block reconstruction after each block is written back into the host model, with the live model, the 1-based prefix depth, and the block name; the probe may raise to abort the pipeline, and an unset probe SHALL change nothing.

#### Scenario: probe_invoked_each_block

- **WHEN** a `progress_probe` is configured and a multi-block model is quantized
- **THEN** the probe observes the live host model once per block in construction order, with depth counting quantized blocks 1-based, after that block's write-back and before the next block's capture begins

#### Scenario: probe_absent_no_change

- **WHEN** the config leaves `progress_probe` unset
- **THEN** the pipeline path is byte-identical to a no-probe run (no invocation overhead in control flow, no checkpoint fingerprint change from a callable being absent), and existing configs and tests keep working untouched

#### Scenario: probe_abort_propagates

- **WHEN** the probe raises its abort exception
- **THEN** the pipeline halts immediately with the block and depth named, no later block is captured or reconstructed, and the error surfaces from `quantize_model` unchanged

### Requirement: The driver probe SHALL fail the run at the first prefix whose ratio crosses the paper-derived collapse line

The driver-side probe SHALL precompute fp-prefix perplexities at every depth on the head slice of the held-out eval, measure each quantized prefix's perplexity on the same slice, compare through the `qk_report` verdict band, and SHALL abort the run when a prefix ratio exceeds `COLLAPSE_LINE` at any probed depth. The probe is kill-only: no prefix measurement SHALL certify `ok` or `degraded`, which come exclusively from the end-of-run full held-out gate.

#### Scenario: exploded_prefix_aborts_fast

- **WHEN** a reconstructed block pushes the quantized prefix's slice ratio above `COLLAPSE_LINE` (min 40.0, the [LB2]-derived line) at depth 2 or later
- **THEN** the run aborts at that block boundary — before any later block's solver work — naming the block, depth, and both prefix perplexities

#### Scenario: healthy_prefix_passes

- **WHEN** the prefix ratio at every probed depth stays at or below `COLLAPSE_LINE`
- **THEN** no probe abort fires, the pipeline completes normally, and the end-of-run verdict alone classifies the artifact

#### Scenario: probe_abort_leaves_report

- **WHEN** the probe aborts at a block boundary
- **THEN** the driver still writes `quantize_report.json` carrying `status: failed`, the aborting block, the depth, both prefix perplexities, and `artifact_exported: false`, and exits nonzero — a probe abort never leaves the run without durable evidence, mirroring the e2e gate's forensics contract

#### Scenario: report_single_shape_both_paths

- **WHEN** any run completes, aborted or not
- **THEN** the report key set is identical on both paths: every field the completion report carries is present on an abort report (unmeasured end-of-run numerics written as `null`), and `artifact_exported` appears on both (`false` on abort, `true` on completion) — strict consumers never KeyError on either shape

#### Scenario: abort_report_construction_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the abort-report construction — building the report dict from a `PrefixProbeAbort`'s fields plus history, routed through `dump_report` with the failed status — is covered by tests that execute without skips, so the catch→report→status logic has repeatable proof independent of weights and CI

#### Scenario: probe_history_recorded_on_completion

- **WHEN** a run completes without a probe abort
- **THEN** `quantize_report.json` carries the per-depth probe history — `(depth, prefix_ppl, fp_prefix_ppl, ratio)` for every probed block — written through the same `dump_report` path as every other report value, so the first surviving run becomes the probe's own calibration data instead of an unauditable near-miss

#### Scenario: probe_comparator_reuses_gate_band

- **WHEN** the probe's decision logic runs
- **THEN** it is the same torch-free `qk_report.verdict` comparator as the end-of-run gate — the missing/non-finite check runs first, and `COLLAPSE_LINE` is the shared derived constant, not a second copy

#### Scenario: probe_slice_same_eval

- **WHEN** the driver builds the probe slice
- **THEN** the slice is the head of the same held-out evaluation batches, in the same order, with the same seed discipline as the end-of-run measurement — never a separately generated dataset

#### Scenario: fp_prefix_baselines_same_depth

- **WHEN** the driver precomputes prefix baselines
- **THEN** each depth's baseline is the fp model's perplexity over the same slice at the same prefix depth, so the probe ratio isolates quantization damage and not depth effects

### Requirement: The probe SHALL be configurable in interval, bounded in cost, and fixed in threshold

The driver SHALL expose `--probe-interval` (positive integer blocks, default 1) and `--probe-batches` (positive integer eval batches, default 2); the abort threshold SHALL be the derived `COLLAPSE_LINE` with no per-run override, and probes SHALL begin no earlier than depth 2.

#### Scenario: probe_knob_defaults_and_validation

- **WHEN** the driver starts with default arguments
- **THEN** the probe measures every block boundary from depth 2 on a 2-batch slice, and a non-positive `--probe-interval` or `--probe-batches` is rejected at argument parsing

#### Scenario: probe_threshold_not_overrideable

- **WHEN** any run executes
- **THEN** no flag or config value re-pins the probe's abort line above or below the shared `COLLAPSE_LINE` constant

#### Scenario: probe_math_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the probe decision logic (slice-ratio verdict, boundary semantics, NaN first) is covered by tests that execute without skips, while model-eval glue tests live in the torch tier per the repo's tier conftest pattern
