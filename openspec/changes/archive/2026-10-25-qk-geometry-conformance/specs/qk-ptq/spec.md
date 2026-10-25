# qk-ptq delta: qk-geometry-conformance

## ADDED Requirements

### Requirement: Calibration and eval data knobs SHALL count sequences, not packed batches

The `--calibration-seqs` and `--eval-seqs` knobs SHALL denote **sequences**, matching the [NQ] protocol unit (128 calibration sequences, [NQ] Sec 3.4/App. C). The loader SHALL convert sequences to packed batches internally (`n_chunks = N // BATCH`, `BATCH = 8`), fail-closed when N is not divisible by BATCH, and the reported counter SHALL be computed at the same boundary as the chunking — before any tune-window slice that changes the chunk count.

#### Scenario: calibration_knob_counts_sequences

- **WHEN** the driver runs with `--calibration-seqs 128`
- **THEN** the loader produces chunks covering exactly 128 sequences (16 packed batches of 8)
- **AND** the reported `calibration_rows` equals 128, matching the [NQ] protocol, so the report's deviation flag does not fire
- **AND** the counter is computed before the `tune_seq_len != seq_len` slice at the loader/driver boundary (a fix placed after that slice computes the value 4× too large under `TUNE_SEQ_LEN=512`)

#### Scenario: calibration_knob_fails_closed_on_odd_count

- **WHEN** `--calibration-seqs` is not divisible by the packed batch size
- **THEN** the driver exits nonzero with an error naming the knob and the divisibility requirement

#### Scenario: eval_knob_counts_sequences

- **WHEN** the driver derives the maximal local eval window
- **THEN** `--eval-seqs` denotes sequences and the window derivation is expressed in sequences (18 batches of 8 re-expressed as 144 sequences), with the eval gate comparing like units at every comparison

#### Scenario: red_first_unit_test

- **WHEN** the unit tests run against the pre-fix driver
- **THEN** an end-to-end assertion `len(calib) * BATCH == CALIBRATION_SEQS` fails with the current value 1024 (red-first), and passes after the fix with 128

### Requirement: The fail-fast verdict machinery SHALL prove geometry-invariance or re-derive constants per geometry

Every gate constant used by the prefix fail-fast verdict SHALL either be geometry-invariant by construction or be re-derived per geometry. The collapse line, pedestals, and the prefix probe each follow the classification stated in the scenarios below.

#### Scenario: collapse_line_is_paper_anchored

- **WHEN** any run evaluates the depth-2 prefix ratio against the collapse line
- **THEN** the line is exactly the published-anchor derivation asserted in `tests/test_qk_report.py`, with no run-specific calibration and no code movement on geometry change

#### Scenario: pedestal_tracks_geometry

- **WHEN** a run changes calibration or eval geometry
- **THEN** its baseline and fp-prefix pedestals are measured at that run's own geometry and recorded in the run's provenance before any ratio verdict is computed

#### Scenario: prefix_probe_same_depth_same_object

- **WHEN** the prefix probe computes its ratio
- **THEN** the quantized-prefix PPL and the fp-prefix PPL are measured on the same object at the same depth, dividing depth-dependent floor effects out of the ratio
