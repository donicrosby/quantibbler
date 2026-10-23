# qk-geometry-conformance — proposal

## Why

`--calibration-seqs` lies about its unit: the knob counts **packed batches of 8 rows**, not sequences. `_tokenize_blocks` builds `picked[i*BATCH:(i+1)*BATCH] for i in range(n_chunks)` with `BATCH = 8`, so `--calibration-seqs 128` loads **1024 sequences — 8× the [NQ] protocol's 128**. Every campaign run (v1, v2, CPU fp64 history) calibrated on 1024 sequences and self-declared the deviation in its own report (`runtime_deviations: calibration_rows 1024 against protocol 128`). The eval axis carries the same semantic bug: `EVAL_SEQS=18` means 18 batches = 144 sequences ("eval coverage 144/145 blocks" is the fingerprint).

The v2 collapse verdict is unaffected (8× more calibration data can only help the solver; 339× sits 8.5× above the paper-anchored line), but no artifact from the current template is protocol-conformant, and the fidelity-verdict machinery keeps grading runs that self-declare deviant before anyone reads PPL.

## What Changes

1. **The knobs come to mean what they say.** `--calibration-seqs N` SHALL load exactly N sequences: the loader converts to packed batches internally (`n_chunks = N // BATCH`, fail-closed on non-divisibility). The counter reports at the same boundary, before any tune-window slice. `EVAL_SEQS` SHALL also count sequences, and its maximal-window derivation re-expresses in sequences (18 batches → 144 sequences).

2. **Red-first tests at both boundaries.** One asserts the calibration unit end-to-end (`len(calib) * BATCH == CALIBRATION_SEQS` at the counter boundary, red against today's 1024); one asserts the eval-window unit in sequences.

3. **Threshold taxonomy (Bug Hunter spec line, proof burdens discharged):** `COLLAPSE_LINE` is citation-only — it derives from [LB2] Sec. 5.2 Table 1 anchors (`ceil(241.96 / 6.10) = 40`, asserted in `tests/test_qk_report.py`) and is geometry-invariant by ratio form (`probe_ratio` divides by the same-depth fp prefix). Zero code movement. Pedestals are per-geometry run inputs already recorded in provenance; v3 re-measures them at true geometry. The depth-2 fp-prefix probe is same-object-same-depth and geometry-consistent by construction.

## Impact

- `tools/quantize_hf_model.py` — loader/driver boundary (knob→chunks conversion, counter boundary, eval-window derivation unit).
- `src/quantibbler/qk_report.py` — deviation-flag comparison unchanged (already compares sequences to 128); doc note on the unit.
- `tests/` — new unit tests for both boundaries, red-first.
- Next faithful fire (v3): geometry-conformant template; pedestal re-measured at 128-sequence geometry.

## Non-goals

- Does NOT change the solver, STE, TuneFP, packing, or any numerics — per-linear ladder reproducibility (identical data, identical numbers) is unaffected only where the data stays identical; v3's numbers are expected to move because the data is now the paper's 128.
- Does NOT re-derive or touch `COLLAPSE_LINE` (paper-anchored, citation-only per the discharged proof burden).
- Does NOT resume or gate the parked Phase 1 convergence lane — separate change.
