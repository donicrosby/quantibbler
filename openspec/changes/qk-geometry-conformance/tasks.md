# qk-geometry-conformance — tasks

## 1. Red-first: calibration unit test fails at 1024

- [ ] Add `tests/ptq/test_geometry_conformance.py` with `test_calibration_knob_counts_sequences`: drive the loader path with `--calibration-seqs 128` semantics and assert `len(calib) * BATCH == CALIBRATION_SEQS` at the counter boundary; run and record RED against today's 1024.

## 2. Red-first: eval window unit test

- [ ] Add `test_eval_knob_counts_sequences`: assert the eval-window derivation expresses sequences (18 batches → 144 sequences) and the gate compares like units; run and record RED.

## 3. Fix the calibration knob at the loader/driver boundary

- [ ] Convert `--calibration-seqs` to chunks with `n_chunks = N // BATCH`, fail-closed on non-divisibility, at the same boundary as the counter (before the tune-slice at `:971`). Counter reports the same N. Run the red tests — GREEN.

## 4. Fix the eval knob unit

- [ ] Re-express `--eval-seqs` in sequences; update the maximal-window derivation and any gate text naming batch units. Run the red tests — GREEN.

## 5. Documentation pass

- [ ] Note the unit decision on `qk_report.py`'s `calibration_rows` comparison (already sequence-based; doc-only) and in the driver's help text for both knobs.

## 6. Gates and landing

- [ ] Full suite in the lane with torch/hf extras (quote the environment and count), `openspec validate qk-geometry-conformance --strict`, scoped commit. No fire included — v3 launch line goes to the room for @user's go separately.
