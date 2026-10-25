# qk-nanoquant-alignment-lane2 — tasks

> Commits are gate-chained by the orchestrator after each phase; there is no
> commit task here. Scenario IDs in parentheses are the spec scenarios each
> test references. Tests precede implementation per capability.
>
> **Reconcile 2026-10-25 (coder, t_75564534).** Every box below was re-verified
> by re-running its named command against the landed tree. Tier matters: the
> `PTQConfig.solver_backend` default is `torch`, so these fixtures REFUSE at the
> implicit `resolve_device` on a CPU-only interpreter — all lane-2 numbers below
> come from `quantibbler-ptq:rocm-official` with `/dev/kfd` `/dev/dri`
> (torch 2.14.0+rocm10.1.0, `PYTHONPATH=/repo/src`), never from the dev CPU wheel.
> Two boxes stay UNCHECKED as partial shipments and two are unrun
> measurement/owner work; see the reconcile report on the card.

## Phase 1 — Fixtures and pinning tests (red-first)

- [x] (coder) New `tests/ptq/test_dual_stream.py` — stream-separation fixture
      (`stream_separation_fixture`): 3-block tiny model, corrupt the
      compressed stream at block 2 BEFORE the capture, assert `Y_b` unchanged.
      Command: `pytest tests/ptq/test_dual_stream.py -k stream_sep`.
      Reconcile 2026-10-25: GREEN — `1 passed, 5 deselected` (rocm-official tier).
      Red-first held at `a90f7e8`, where this test's failure mode is the
      pre-`a7999ce` accelerator-path `TypeError` (see design.md crystallization).
- [x] (coder) Same file — lockstep fixture (`dual_stream_advances_in_lockstep`):
      `x_fp` at block b equals a fresh pristine-prefix forward; `x_b` equals
      the quantized-prefix forward. Command:
      `pytest tests/ptq/test_dual_stream.py -k lockstep`.
      Reconcile 2026-10-25: GREEN — `1 passed, 5 deselected` (rocm-official tier).
      Red-first held at `a90f7e8`, same accelerator-path `TypeError` caveat.
- [x] (coder) Same file — teacher-stream fixture (`teacher_target_reads_fp_stream`):
      `Y_b == B^FP_b(x_fp)` and `Y_b != B^FP_b(x_b)` on a fixture whose streams
      diverge. Command:
      `pytest tests/ptq/test_dual_stream.py -k teacher_target_reads_fp_stream`.
      (The row was written `-k teacher_stream`, which matches NOTHING: `-k` is a
      substring match on the test NAME and the scenario id is
      `teacher_target_reads_fp_stream`. The literal selector exits RC=5 with
      `6 deselected`; corrected here.)
      Reconcile 2026-10-25: GREEN — `1 passed, 5 deselected` (rocm-official tier).
- [x] (coder) Same file — target-1 fixture
      (`target1_tunefp_vacuity_gone_by_construction`): at block ≥ 2 target 1
      the TuneFP objective is nonzero; at block 1 the skip fires with the
      identity-prefix reason. Command: `pytest tests/ptq/test_dual_stream.py -k
      target1`.
      Reconcile 2026-10-25: GREEN — `2 passed, 4 deselected` (rocm-official tier).
      Red-first held at `a90f7e8`, same accelerator-path `TypeError` caveat.
- [x] (coder) `tests/ptq/test_tunefp_teacher.py` — guard-detector pins
      (`guard_zero_grad_zero_residual_means_broken_plumbing`,
      `guard_rationale_rewritten`): zero-grad + zero-residual still emits
      `tune_fp_vacuous_teacher` carrying both norms; the docstring/config
      text asserts dual-stream rationale (grep for the rewritten text, no
      "inherent" vacuity claim). Command:
      `pytest tests/ptq/test_tunefp_teacher.py`.
      Reconcile 2026-10-25: GREEN — `8 passed` (rocm-official tier). RED-first
      REPRODUCED literally at `a90f7e8`: `test_guard_rationale_rewritten` →
      `AssertionError: the dual-stream rationale is not present`.
- [x] (coder) `tests/test_qk_report.py` — label fixture
      (`label_requires_fp_stream_teacher_provenance`): compressed-stream or
      unprovenanced captures set the label false; fp-stream provenance passes;
      the dual-stream deviation marker does not flip the label. Command:
      `pytest tests/test_qk_report.py -k provenance`.
      Reconcile 2026-10-25: GREEN — `1 passed, 90 deselected` (rocm-official tier).
      RED-first REPRODUCED literally at `a90f7e8`: `AssertionError: a capture
      with no stream provenance was labelled paper_faithful`.
- [x] (coder) `tests/ptq/test_progress_probe.py` + `tests/ptq/test_checkpoints.py`
      — shape pins (`prefix_probe_and_checkpoint_fingerprints_unchanged_in_shape`):
      probe signature/call-shape assertion; fingerprint payload keys and
      joined config-field set assertion. Command:
      `pytest tests/ptq/test_progress_probe.py tests/ptq/test_checkpoints.py`.
      Reconcile 2026-10-25: GREEN — `11 passed` (rocm-official tier).

## Phase 2 — Dual-stream implementation

- [x] (coder) `block_recon.reconstruct_model` — advance `x_fp` through the
      pristine frozen-reference children in lockstep with `x_running`; the
      per-block teacher capture becomes `Y_b = B^FP_b(x_fp)`; the boundary
      finiteness check covers both streams. Command: Phase 1 fixtures.
      Reconcile 2026-10-25: GREEN — `pytest tests/ptq/test_dual_stream.py
      tests/ptq/test_tunefp_teacher.py` → `14 passed`; the boundary finiteness
      guard names `x_fp` (both streams covered) in `block_recon.py`.
- [ ] (coder) `block_recon.reconstruct_block` / `_tune_fp` — accept the
      fp-stream capture; recorder `capture` event gains `x_b_stream` /
      `y_b_stream` fields; `tune_fp`, `ste`, `tune_fp_vacuous_teacher` events
      carry the stream provenance of the capture they consumed. Command:
      `pytest tests/ptq/test_dual_stream.py tests/ptq/test_tunefp_teacher.py`.
      **UNCHECKED — PARTIAL (reconcile 2026-10-25).** The command is GREEN
      (`14 passed`) and the `capture` event does carry `x_b_stream="compressed"`
      / `y_b_stream="fp"`. The third clause is NOT shipped: the `tune_fp`,
      `ste`, and `tune_fp_vacuous_teacher` event payloads carry `capture_index` /
      `teacher_source` / `teacher_fingerprint` and NO stream field
      (`teacher_source` is `"frozen_reference_copy"` — a teacher-source label,
      not a stream), and the spec scenario `recorder_events_gain_stream_provenance`
      has no test anywhere in `tests/`. Delta for the planner: design.md's
      decided Open Question says to REPEAT the provenance fields on each event
      (self-contained events); the weaker reading — resolve the stream through
      `capture_index` — is satisfied.
- [x] (coder) `_tune_fp` docstring + `config.py` `paper_faithful` comment —
      rewritten rationale (scenario `guard_rationale_rewritten`); no new
      config field. Command: `pytest tests/ptq/test_config.py`.
      Reconcile 2026-10-25: GREEN — `5 passed` (rocm-official tier); the
      `paper_faithful` change in `ec906a7` is comment-only, and
      `test_scn_prefix_probe_and_checkpoint_fingerprints_unchanged_in_shape`
      pins the joined config-field set.
- [ ] (coder) `qk_report.py` — capture stage-map description reworded to
      fp-stream teacher outputs; verdict condition for fp-stream capture
      provenance added; deviation list names compressed-stream captures.
      Command: `pytest tests/test_qk_report.py`.
      **UNCHECKED — PARTIAL (reconcile 2026-10-25).** The command is GREEN
      (`91 passed`) and the verdict condition IS shipped (`capture_is_fp_stream`
      in `fidelity_conditions`). Two clauses are not: (i) the stage-map
      description is not reworded — `_STAGE_LINES`' `capture` row still reads
      `"compressed-prefix inputs X_b and teacher outputs Y_b"`, which design.md's
      "Stream provenance in recorder events" section requires reworded to name
      the fp-stream teacher output; (ii) no deviation list names the offending
      capture — the `capture` row's deviation column is `None` and the only
      denial signal is the verdict condition key. The shipped test asserts the
      label flip only.
- [x] (coder) Full Phase 1 + report run:
      `pytest tests/ptq/test_dual_stream.py tests/ptq/test_tunefp_teacher.py
      tests/test_qk_report.py tests/ptq/test_progress_probe.py
      tests/ptq/test_checkpoints.py`.
      Reconcile 2026-10-25: GREEN — `116 passed` in 7.59s (rocm-official tier;
      116 collected).

## Phase 3 — Attribution measurement (no confound)

- [ ] (coder) Tiny e2e gate, dual-stream vs the last single-stream baseline on
      the same fixture: record max-logit-delta and BPW for both.
      Command: `pytest tests/ptq/test_e2e_gate.py` plus the pinned
      single-stream run — expected: both green, numbers recorded.
      **UNCHECKED — measurement work that never ran (reconcile 2026-10-25).**
      Only the first half is executable: `pytest tests/ptq/test_e2e_gate.py` →
      `3 passed` (rocm-official tier, 12.51s) against the pinned A-8 bounds
      (max logit delta ≤ 1.0, BPW < 1.0 measured from bytes). There is no
      single-stream baseline to compare against — stream choice was deliberately
      NOT made a config knob (design.md "Discarded alternatives") — and no
      dual-stream-vs-single-stream numbers are recorded anywhere in the tree.
      design.md's Open Question ("Does dual-stream move the e2e tiny-model gate
      numbers?") is still open, owner: planner.
- [ ] (planner) Review the measured delta; confirm no gate bound moves
      (open question owner: planner).
      **UNCHECKED — blocked on the box above: there is no measured delta to review.**

## Phase 4 — Validation and gates

- [x] (planner) `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane2
      --strict` — green.
      Reconcile 2026-10-25: run by the reconciler against the landed tree,
      `RC=0` → `Change 'qk-nanoquant-alignment-lane2' is valid`.
- [x] (coder) Full gate suite: `pytest tests/` — the row's literal
      `pytest tests/ptq/ tests/ci/ tests/qk_report/` names two paths that do
      NOT exist (`tests/ci/` and `tests/qk_report/` never did in this repo's
      history; the CI-referee test is the FILE `tests/test_ci_referee.py`), so
      the real suite is `tests/` — plus `cargo test` + `cargo clippy
      --all-targets -D warnings` + `cargo fmt --check` + `maturin build`.
      Suite green required. Record results.
      Reconcile 2026-10-25 (rocm-official tier + `/dev/kfd`): `pytest tests/` →
      `652 passed, 4 skipped, 3 deselected` in 110.71s, `FULLSUITE_RC=0` (re-run
      after the last edit; the 94.93s run before it gave the same counts);
      `cargo test` → `24 passed;
      0 failed`, RC=0; `cargo clippy --all-targets -- -D warnings` RC=0;
      `cargo fmt --check` RC=0; `maturin build` (from
      `crates/quantibbler-kernels/`) RC=0, wheel
      `quantibbler_kernels-0.1.0-cp311-abi3-manylinux_2_35_x86_64.whl`.
