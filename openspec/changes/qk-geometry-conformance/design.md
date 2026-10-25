# qk-geometry-conformance — design

## Unit decision (resolved)

The knobs keep their names and their protocol values and come to MEAN sequences. Rationale: the paper's protocol is 128 sequences ([NQ] Sec 3.4/App. C); the CLI surface (`--calibration-seqs 128`) is the one the campaign already uses; changing the unit means changing every recorded launch line. The packed-batch structure (BATCH=8) is an implementation detail of the tokenizer and SHALL live below the knob boundary.

## Fix placement (resolved, Coder requirement)

The knob→chunks conversion lands at the **same boundary as the counter** — before the `tune_seq_len != seq_len` slice in the driver's `main()`, which splits each (8, 2048) batch into four (8, 512) windows and quadruples `len(calib)`. A conversion placed after that line computes a knob value 4× too large under `TUNE_SEQ_LEN=512`, a geometry the campaign actually runs. The existing comment already requires the counter "counted BEFORE the slice"; the fix rides the same line of reasoning.

## Constant taxonomy (Bug Hunter spec line, incorporated)

- **(a) Pedestals** (baseline PPL, fp-prefix PPL): geometry-dependent — re-measured per run at that run's geometry (already provenance-recorded); no code movement.
- **(b) COLLAPSE_LINE = 40.0**: citation-only. Derived `ceil(241.96 / 6.10)` from [LB2] Sec. 5.2 Table 1 (241.96 = documented method-collapse PPL at 0.55 bpp; 6.10 = FP16 reference), asserted in `tests/test_qk_report.py`. Ratio form (`probe_ratio` divides by the same-depth fp prefix) makes it transportable across geometries. Zero code movement.
- **(c) Depth-2 fp-prefix probe**: same-object-same-depth by construction; geometry-consistent automatically.
- **(d) Eval-window gate**: the record keeps its own unit (`full_local_window_batches`) and the gate converts it through one named derivation (`eval_window_sequences`), so it compares like units and re-derives automatically at any geometry.

## Known limitations

- The bpp-axis inference (our ~1 bpp stored form sits above the anchor pair's 0.55 bpp collapse point, so the line is lenient for our operating point) is read off two anchor points — inference, not a paper claim. Recorded here so it is never cited as such.
- v2's 339× verdict (8.5× above the line) survives the G geometry correction: no plausible window-size effect spans that margin.

## Implementation crystallization (2026-10-25)

The change's code was already in the tree when the kanban tickets covering it were
dispatched: it landed as `84b4615` ("ptq: data knobs count sequences, not packed
batches"), 102 commits below the revision the tickets were cut against. The tickets'
RED premise was therefore re-established against the pre-fix revision (`84b4615^`)
rather than the working tree — same CLI, same fixture, two driver revisions:

- pre-fix: `calib 128 batches` → `len(calib) * BATCH == 128 * 8 = 1024` (RED)
- current: `calib 16 batches` → `128` (GREEN)

Implementation-discovered details, recorded so they are not re-litigated:

- The conversion is ONE named helper, `packed_chunks(sequences, knob)`, applied at
  the two data-loading call sites (`tokenize_chunks`, `_tokenize_blocks`): the
  packed-batch structure stays below the knob boundary, and non-divisibility raises
  a `ValueError` naming the knob and `BATCH`.
- The counter is `len(calib) * BATCH`, taken above the `tune_seq_len != args.seq_len`
  slice. Executed check at `--seq-len 32 --tune-seq-len 16`: pre-slice count 128,
  post-slice count 256 — a counter placed after the slice misreports the knob.
- The eval gate compares in the knob's unit (`eval_window_sequences(evaluation)`,
  i.e. `full_local_window_batches * BATCH`) while `evaluation_record` keeps its
  harness-fact key in batches; the two are not the same number and must not be
  conflated.
- Superseded wording: the `quantize_hf_model.py:971` citation under "Fix placement"
  and the `:971` citation in `tasks.md` row 3 are replaced by symbol citations —
  line numbers go stale with every insertion above them.

## Implementation crystallization — eval axis (2026-10-25)

The eval axis's knob re-expression, gate message and help text landed at `84b4615`
with the calibration side; the eval ticket's landing added the seam and the pin:

- **Design-silent decision (recorded):** the gate's unit conversion is a NAMED
  module-level derivation, `eval_window_sequences(evaluation)`, mirroring
  `packed_chunks(sequences, knob)`. Rationale: the gate is where the run record's
  batch window meets the knob's sequence value, and without a seam the only way to
  assert "the gate compares like units" was for the test to restate the arithmetic
  — a pin on a transcription of the code rather than on the code. Rejected
  alternative: leaving the expression inline and asserting the message text through
  a second driver subprocess (a several-minute run for one multiplication, and the
  gate's end-to-end refusal is already pinned in `test_driver_stage_map.py`).
- The test is `test_eval_knob_counts_sequences` (spec scenario name), superseding
  the earlier `test_eval_window_re_expresses_in_sequences`: same 18-batch → 144
  coverage, plus the two units being asserted DISTINCT and the knob's own
  conversion asserted to be the derivation's exact inverse. It feeds the real
  `evaluation_record` at the paper geometry (145 blocks, seq 2048) rather than
  hand-written numbers.
- RED re-establishment: the test module takes its driver path from `QHFM_DRIVER`
  (default: the working tree), so the pre-fix revision (`84b4615^`) is run with the
  SAME fixture and command — observed `AttributeError: module 'qhfm_geometry' has
  no attribute 'eval_window_sequences'`, RC=1 — instead of a paraphrase of it. The
  unit defect is observable at the CLI too: pre-fix `--help` read `--eval-seqs 5
  batches of 8 = 40 eval rows`, post-fix `40 held-out evaluation SEQUENCES, packed
  8 rows per batch (5 batches)`.

## Implementation crystallization — the gate lane (2026-10-25)

The gate is measured on the (interpreter × device) pair the fire runs, not on the dev
venv. `pyproject.toml` scopes the `torch` extra to the CPU wheel, so `uv run pytest`
is structurally red here: its 52 failures are the documented fail-closed
`resolve_device` refusal (`RuntimeError: solver backend 'torch' requires an
accelerator`), which no code in this change can clear. The recorded suite is therefore
taken from `quantibbler-ptq:rocm-official` with `--device /dev/kfd --device /dev/dri`
and `PYTHONPATH=/repo/src` (torch 2.14.0+rocm10.1.0, hip 7.16.26385, one device) over
a tracked-only snapshot: at rev `84ba3b4` `394 passed, 2 skipped, 3 deselected`
(194.15s, RC=0), and at the earlier rev `6c09948` `391 passed, 2 skipped,
3 deselected` (217.47s, RC=0). The dev-invocation count is quoted beside it, with its interpreter, because a
number without its wheel is not evidence.

Two further decisions, recorded so they are not reopened:

- **The task list's path list names directories that do not exist.** This tree has no
  `tests/ci/` and no `tests/qk_report/`, so the literal invocation exits RC=4 before
  collecting anything. The equivalent files are `tests/test_ci_referee.py` and
  `tests/test_qk_report.py`; the gate runs those, and row 6 carries the corrected text
  rather than a path that cannot resolve.
- **A shared checkout is not a measurement surface.** Sibling lanes edit this tree
  while a suite runs. Between two snapshots of the SAME revision `6c09948`,
  `tests/ptq/test_lb_admm.py` went from `2 passed` to `2 failed` because
  `src/quantibbler/ptq/lb_admm.py` gained a sibling lane's in-flight
  `SOLVER_PROBE_ENV` gate that leaves `trace.min_eigs` empty by default; the A/B used
  one interpreter, one arg set and the two snapshots. Green is therefore quoted with
  its revision, snapshot time and interpreter, and the standing measurement for this
  change is the TRACKED tree — untracked sibling files (`test_launch_attribution.py`,
  `test_min_eigs_gating.py`, `test_stage_device_parity.py`) are collected by the path
  list but belong to no landing of this change. The uncommitted
  `tools/quantize_hf_model.py` diff is likewise a mix: this change owns two hunks
  (`+eval_window_sequences`, and the gate site that calls it) and `qk-launch-attribution`
  owns the other seven, so the file cannot be staged wholesale.
