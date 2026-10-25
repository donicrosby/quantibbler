# qk-geometry-conformance — tasks

## 1. Red-first: calibration unit test fails at 1024

- [x] Add `tests/ptq/test_geometry_conformance.py` with `test_calibration_knob_counts_sequences`: drive the loader path with `--calibration-seqs 128` semantics and assert `len(calib) * BATCH == CALIBRATION_SEQS` at the counter boundary; run and record RED against the pre-fix revision (before `packed_chunks` existed: `--calibration-seqs 128` printed `calib 128 batches`, i.e. 1024 sequences).

## 2. Red-first: eval window unit test

- [x] Add `test_eval_knob_counts_sequences`: assert the eval-window derivation expresses sequences (18 batches → 144 sequences) and the gate compares like units; run and record RED. RED against the pre-fix driver revision (`84b4615^`, via the test module's `QHFM_DRIVER` path override — same fixture, same command): `1 failed, 4 deselected in 5.30s`, `AttributeError: module 'qhfm_geometry' has no attribute 'eval_window_sequences'`, RC=1; the pre-fix CLI read `--eval-seqs 5 batches of 8 = 40 eval rows`. GREEN on the working tree: file-level `5 passed in 15.63s`, RC=0.

## 3. Fix the calibration knob at the loader/driver boundary

- [x] Convert `--calibration-seqs` to chunks with `n_chunks = N // BATCH` (`packed_chunks()` at the `tokenize_chunks` call site), fail-closed on non-divisibility, at the same boundary as the counter: above the `tune_seq_len != args.seq_len` slice in the driver's `main()`, which is what changes the chunk count. Counter reports the same N. Run the red tests — GREEN.

## 4. Fix the eval knob unit

- [x] Re-express `--eval-seqs` in sequences; update the maximal-window derivation and any gate text naming batch units. Run the red tests — GREEN. The knob re-expression and the gate message landed at `84b4615`; this landing named the derivation at the gate site (`eval_window_sequences(evaluation)`, the one place the record's batch window becomes the knob's sequence value) and pinned it. Gate text now demands `--eval-seqs 144 sequences`; `--help` reads `40 held-out evaluation SEQUENCES, packed 8 rows per batch (5 batches)`.

## 5. Documentation pass

- [x] Note the unit decision on `qk_report.py`'s `calibration_rows` comparison (already sequence-based; doc-only) and in the driver's help text for both knobs. Evidence: `geometry_deviations` carries the "counts sequences ([NQ] Sec 3.4/App. C) … no batch factor belongs here" note beside its `!= 128` comparison; `quantize_hf_model.py --help` reads `128 calibration SEQUENCES … (16 batches)` and `40 held-out evaluation SEQUENCES, packed 8 rows per batch (5 batches)`.

## 6. Gates and landing

- [x] Full suite in the lane with torch/hf extras (quote the environment and count), `openspec validate qk-geometry-conformance --strict`, scoped-commit handoff. No fire included — v3 launch line goes to the room for @user's go separately.
  - **Fire lane** (container `quantibbler-ptq:rocm-official`, `--device /dev/kfd --device /dev/dri`, `PYTHONPATH=/repo/src`; torch 2.14.0+rocm10.1.0, hip 7.16.26385, `torch.cuda.is_available()=True`, 1 device), tracked-only snapshot at rev `84ba3b4` (06:40:17Z): `394 passed, 2 skipped, 3 deselected in 194.15s`, RC=0 — the standing green number. Same lane on the earlier rev `6c09948` snapshot: `391 passed, 2 skipped, 3 deselected in 217.47s`, RC=0. The change's own `tests/ptq/test_geometry_conformance.py` in that lane, run after the last doc edit: `5 passed in 16.07s`, RC=0.
  - **Dev invocation** (`uv run --extra kernels --extra torch --extra hf pytest tests/ptq/ tests/test_ci_referee.py tests/test_qk_report.py`, torch 2.14.1+cpu, no accelerator): `52 failed, 342 passed, 3 deselected in 755.70s` (394 collected), RC=1. 50 carry the documented fail-closed `RuntimeError: solver backend 'torch' requires an accelerator`; the other 2 pass in the container — a wheel-without-accelerator tier, not a code defect.
  - **Path list correction**: this tree has no `tests/ci/` or `tests/qk_report/`; the literal list exits RC=4 (`ERROR: file or directory not found: tests/ci/`). The gate runs the real equivalents `tests/test_ci_referee.py` and `tests/test_qk_report.py`.
  - `openspec validate qk-geometry-conformance --strict` → `Change 'qk-geometry-conformance' is valid`, RC=0 (one change id per invocation).
  - **The live working tree is not a green surface**: sibling lanes edit it while the suite runs (06:31:56Z: `7 failed, 396 passed, 2 skipped, 3 deselected`, every failure in a file outside this change's blast radius — see the dated design entry). The two `tests/ptq/test_lb_admm.py` reds seen between snapshots A/B to a sibling's in-flight `SOLVER_PROBE_ENV` gate in `src/quantibbler/ptq/lb_admm.py` (`2 passed` before, `2 failed` after, same interpreter and args).
  - **Landing is a handoff, not a commit** (leaf never commits; the orchestrator is sole committer). Scoped manifest is a comment on the task, and it is load-bearing: the uncommitted `tools/quantize_hf_model.py` diff MIXES this change's 2 hunks with `qk-launch-attribution`'s 7, so the file must not be staged wholesale.
