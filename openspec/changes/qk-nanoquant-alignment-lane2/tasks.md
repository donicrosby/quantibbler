# qk-nanoquant-alignment-lane2 — tasks

> Commits are gate-chained by the orchestrator after each phase; there is no
> commit task here. Scenario IDs in parentheses are the spec scenarios each
> test references. Tests precede implementation per capability.

## Phase 1 — Fixtures and pinning tests (red-first)

- [ ] (coder) New `tests/ptq/test_dual_stream.py` — stream-separation fixture
      (`stream_separation_fixture`): 3-block tiny model, corrupt the
      compressed stream at block 2 BEFORE the capture, assert `Y_b` unchanged.
      Command: `pytest tests/ptq/test_dual_stream.py -k stream_sep` — expected
      RED against the current single-stream capture.
- [ ] (coder) Same file — lockstep fixture (`dual_stream_advances_in_lockstep`):
      `x_fp` at block b equals a fresh pristine-prefix forward; `x_b` equals
      the quantized-prefix forward. Command:
      `pytest tests/ptq/test_dual_stream.py -k lockstep` — expected RED.
- [ ] (coder) Same file — teacher-stream fixture (`teacher_target_reads_fp_stream`):
      `Y_b == B^FP_b(x_fp)` and `Y_b != B^FP_b(x_b)` on a fixture whose streams
      diverge. Command: `pytest tests/ptq/test_dual_stream.py -k teacher_stream`
      — expected RED.
- [ ] (coder) Same file — target-1 fixture
      (`target1_tunefp_vacuity_gone_by_construction`): at block ≥ 2 target 1
      the TuneFP objective is nonzero; at block 1 the skip fires with the
      identity-prefix reason. Command: `pytest tests/ptq/test_dual_stream.py -k
      target1` — expected RED.
- [ ] (coder) `tests/ptq/test_tunefp_teacher.py` — guard-detector pins
      (`guard_zero_grad_zero_residual_means_broken_plumbing`,
      `guard_rationale_rewritten`): zero-grad + zero-residual still emits
      `tune_fp_vacuous_teacher` carrying both norms; the docstring/config
      text asserts dual-stream rationale (grep for the rewritten text, no
      "inherent" vacuity claim). Command:
      `pytest tests/ptq/test_tunefp_teacher.py` — rationale grep asserts
      expected RED until Phase 2.
- [ ] (coder) `tests/test_qk_report.py` — label fixture
      (`label_requires_fp_stream_teacher_provenance`): compressed-stream or
      unprovenanced captures set the label false; fp-stream provenance passes;
      the dual-stream deviation marker does not flip the label. Command:
      `pytest tests/test_qk_report.py -k provenance` — expected RED until
      Phase 2.
- [ ] (coder) `tests/ptq/test_progress_probe.py` + `tests/ptq/test_checkpoints.py`
      — shape pins (`prefix_probe_and_checkpoint_fingerprints_unchanged_in_shape`):
      probe signature/call-shape assertion; fingerprint payload keys and
      joined config-field set assertion. Command:
      `pytest tests/ptq/test_progress_probe.py tests/ptq/test_checkpoints.py`
      — expected GREEN now (they pin today's shape; the lane must not move it).

## Phase 2 — Dual-stream implementation

- [ ] (coder) `block_recon.reconstruct_model` — advance `x_fp` through the
      pristine frozen-reference children in lockstep with `x_running`; the
      per-block teacher capture becomes `Y_b = B^FP_b(x_fp)`; the boundary
      finiteness check covers both streams. Command: Phase 1 fixtures —
      expected GREEN.
- [ ] (coder) `block_recon.reconstruct_block` / `_tune_fp` — accept the
      fp-stream capture; recorder `capture` event gains `x_b_stream` /
      `y_b_stream` fields; `tune_fp`, `ste`, `tune_fp_vacuous_teacher` events
      carry the stream provenance of the capture they consumed. Command:
      `pytest tests/ptq/test_dual_stream.py tests/ptq/test_tunefp_teacher.py`
      — expected GREEN.
- [ ] (coder) `_tune_fp` docstring + `config.py` `paper_faithful` comment —
      rewritten rationale (scenario `guard_rationale_rewritten`); no new
      config field. Command: `pytest tests/ptq/test_config.py` — expected
      GREEN (field set unchanged).
- [ ] (coder) `qk_report.py` — capture stage-map description reworded to
      fp-stream teacher outputs; verdict condition for fp-stream capture
      provenance added; deviation list names compressed-stream captures.
      Command: `pytest tests/test_qk_report.py` — expected GREEN.
- [ ] (coder) Full Phase 1 + report run:
      `pytest tests/ptq/test_dual_stream.py tests/ptq/test_tunefp_teacher.py
      tests/test_qk_report.py tests/ptq/test_progress_probe.py
      tests/ptq/test_checkpoints.py` — expected GREEN.

## Phase 3 — Attribution measurement (no confound)

- [ ] (coder) Tiny e2e gate, dual-stream vs the last single-stream baseline on
      the same fixture: record max-logit-delta and BPW for both.
      Command: `pytest tests/ptq/test_e2e_gate.py` plus the pinned
      single-stream run — expected: both green, numbers recorded.
- [ ] (planner) Review the measured delta; confirm no gate bound moves
      (open question owner: planner).

## Phase 4 — Validation and gates

- [ ] (planner) `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane2
      --strict` — green.
- [ ] (coder) Full gate suite: `pytest tests/ptq/ tests/ci/ tests/qk_report/` +
      `cargo test` + `cargo clippy --all-targets -D warnings` + `cargo fmt --check`
      + `maturin build`. Suite green required. Record results.
