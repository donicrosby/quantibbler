# Tasks: qk-fix-tunefp-teacher

- [x] (coder) Implement per the spec delta: teacher capture per [NQ]
  Algorithm 1 (`y_fp` = pristine block output on the compressed-prefix
  stream, cached pre-tuning, detached, no_grad) with the teacher block's
  weights loaded from the pinned pristine safetensors (or a frozen
  reference copy), never the live module state; source fingerprint +
  capture index recorded in the artifact; teacher cache memory-only and
  re-captured on resume; sequential in-block factorization with `_tune_fp`
  fired once per block after the attention projections (q/k/v) are
  factorized and before the remaining linears (o/g/u/d) are, tuning the
  still-FP remainder against the cached teacher ([NQ] Figure 3
  placement; the legacy pre-loop call site is removed);
  informed-assumption marker for the TuneFP loss formula; guard: zero
  first-step gradient — operative call or any call emitted with no
  in-block layer factorized — aborts with `tune_fp_vacuous_teacher`
  carrying BOTH the gradient norm and the initial residual norm (no
  trivial-zero carve-out: under the Figure 3 placement the operative
  call always has factorized attention projections); fidelity-claiming
  runs fail (or record `paper_faithful: false`) on any vacuous event.
  Defaults stay [NQ] App. C.
- [x] (coder) Self-test: a regression test constructs the shipped
  self-referential teacher with an in-block layer already factorized and
  asserts the guard fires with both norms recorded (fail-closed path); a
  second test with the corpus-pinned teacher asserts the objective gradient
  is nonzero once an in-block layer is factorized and the tuned block
  reduces teacher error over T_pre steps; a third asserts no TuneFP call
  site precedes the attention projections' factorization (the legacy
  pre-loop placement is removed and a zero-set invocation aborts as
  vacuous);
  a fourth asserts the teacher weights come from the pristine source
  (fingerprint recorded, not the live module) and that no teacher tensor is
  persisted by the checkpoint writer; a fifth asserts a fidelity-claiming
  run fails on a vacuous event while a diagnostic run continues.
- [x] (coder) Rehearse the stage-trace probe against the fixed bytes (toy
  scope first, then the 7 scoped linears if budgeted): `tune_fp_effect` must
  be nonzero on at least one scoped linear with a real teacher; the abort
  event `tune_fp_vacuous_teacher` must be absent; record the new probe
  sha. Post the rehearsal output.
- [x] (coder) Faithful-run stage map: the rehearsal run record maps every
  executed stage to [NQ] Algorithm 1 lines 0–28 and names any deviation;
  the terminal scale stage is verified against TuneScalesKD semantics
  (line 27: global scale-only KD against the FP teacher, packed binaries
  frozen, T_glob steps) or recorded as a named deviation.
- [ ] (bug-hunter) Review on the landed bytes: teacher captured once before
  tuning from the pristine-pinned block weights on the compressed-prefix
  stream, never recomputed from tuned weights and never read from live
  module state; source fingerprint + capture index in the artifact; cache
  is memory-only and re-captured on resume; the vacuous event carries both
  norms; TuneFP fires once per block after q/k/v factorization per
  [NQ] Figure 3, tunable set = the still-FP remainder, and no TuneFP call
  site precedes in-block factorization; guard is
  fail-closed with no trivial-zero carve-out; fidelity-claiming runs
  invalidate on the vacuous event;
  no call site still passes self-outputs as the teacher;
  artifact marker present. Verify the regression tests run green from clean
  checkout.
- [ ] (sec-eng) Integrity pass: the pristine-source load adds no new network
  or persistence surface beyond the pinned cache; the abort path cannot
  corrupt in-flight state (snapshot-restore discipline preserved);
  provenance fields (source fingerprint, capture index) provable from the
  artifact alone; the two-norm event distinguishes broken-teacher from
  detached-student on the landed bytes.
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-tunefp-teacher --strict`; review the delta against the frozen fork
  timing (this fix is owed regardless of the ladder verdict); commit
  (specs + code in one commit), do not archive until a real run shows a
  nonzero TuneFP effect on record.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." This change and its siblings land before any further diagnostic
runs; the ladder fork fires only after a paper-faithful run exists.

## Reviewer amendments (2026-10-21, sec-eng, folded by planner)

- A1: teacher weights pinned to pristine source, source fingerprint +
  capture index recorded (scenarios `teacher_is_captured_before_tuning`,
  `tune_fp_effect_is_recorded_honestly`).
- A2: teacher cache memory-only in v1, re-captured on resume
  (scenario `teacher_cache_is_memory_only_and_recovered_on_resume`).
- A3: vacuous event carries gradient AND initial residual norm;
  fidelity-claiming runs invalidate on any vacuous event
  (scenario `vacuous_teacher_fails_closed`).
- P1 (planner, 2026-10-21, user-pinned figure): TuneFP placement
  re-ruled to the [NQ] Figure 3 encoding — once per block after Q/K/V
  factorization, tunable set = still-FP remainder; the legacy pre-loop
  placement is the measured defect and is removed; the trivial-zero
  carve-out is dropped (scenarios
  `tunefp_fires_once_per_block_after_attention_factorization`,
  `vacuous_teacher_fails_closed`; figure vendored at
  `docs/papers/assets/block_recon.png`).

## Implementation notes (coder, 2026-10-21)

Landed on `src/quantibbler/ptq/block_recon.py`, `config.py`, `api.py`,
`src/quantibbler/qk_report.py` and `tools/quantize_hf_model.py`. Evidence:
`tests/ptq/test_tunefp_teacher.py` (6 tests), `tests/ptq/test_driver_stage_map.py`
(2 real driver runs), `tests/test_qk_report.py` (4 stage-map/stage-failure tests).

**Rehearsal output (real driver, tiny Qwen3 fixture, `--paper-faithful`):**
exit 0, `status ok`, `ppl_ratio 0.9999975`. TuneFP executed **32 steps on each
of 3 blocks** with `grad_norm` 1.6-2.8e-07, initial residual 1.8-2.2e-04,
residual-after strictly below it, `weight_delta_rel` 0.0027-0.0050,
`tunable = [inner.self_attn.o_proj, inner.mlp.gate_proj, inner.mlp.up_proj,
inner.mlp.down_proj]` — the still-FP remainder, exactly as the Figure 3
placement requires. `vacuous_events == 0`. Teacher provenance on every
capture: `source=frozen_reference_copy`, fingerprint recorded, capture index
recorded. STE per-linear flip counts and scale movement recorded for all
seven scoped linears; zero-crossing linears carry `discrete_flip_pass` or the
`ste_scale_only_declared` marker — none completed silently. The full
Algorithm 1 stage map (lines 2-5, 9-10, 12, 14-17, 19, 20-23, 26-27) is in
`quantize_report.json` with every deviation named.

**Implementation crystallization (decisions the spec left open):**

- **Degenerate block shapes.** A block carrying no attention projection cannot
  honour the figure literally, so the pre-TuneFP group degenerates to the
  first quantizable linear in declaration order; a block whose only linear is
  also the whole remainder records `tune_fp` with
  `skipped = no_still_fp_remainder` instead of firing a pass with an empty
  tunable set. The guard's invariant ("the operative call always has >= 1
  in-block layer factorized") therefore holds for every block shape, and the
  skip is recorded, never silent.
- **Statistics timing.** The block's RobustDiag statistics are now collected
  once, before the factorization loop, on the pristine block — the same bytes
  the previous (TuneFP-is-a-no-op) pipeline measured, and the reading closest
  to Algorithm 1 lines 2-5. Collecting them mid-loop per group would have
  required a second statistics pass with no algorithmic warrant.
- **Residual statistic.** `initial_residual_norm` is the RMS
  (`sqrt(mean squared error)`) of the pre-tune objective on the first step's
  batch, recorded alongside the gradient norm. Gradient-zero + residual-zero
  is the broken-teacher signature (target equals current output);
  gradient-zero + residual-nonzero is the detached-student signature. Both
  are exercised in `test_scn_vacuous_teacher_carries_both_norms_and_invalidates_fidelity`.
- **`paper_faithful` is solver state.** It joins the checkpoint fingerprint
  and is set by `--paper-faithful`, which also switches the stage budgets to
  [NQ] App. C verbatim (T_pre/T_post/T_glob = 8 epochs at 1e-4/1e-5/1e-6).
  Without the flag the driver keeps the gate-shaped budgets (TuneFP at one
  step) and the wrapper prints an advisory saying so.
- **A third report path exists now.** The vacuity guard fails a
  fidelity-claiming run *and* writes a forensically complete report
  (`qk_report.stage_failure_report`, status failed, `paper_faithful: false`,
  both norms). The driver's wiring pin was re-homed accordingly: the identity
  fields moved into a shared module-level `_run_identity()` helper, the pin
  now derives its key set from that helper, asserts the abort payload really
  spreads it, expects exactly three `report` bindings, and carries a second
  handler spec plus two new poison cases for the stage-gate path.
- **`sign_extract_pack` is a new record event**, emitted once per block after
  extraction. `test_scn_block_loop_order` was re-pinned to filter for the
  four Algorithm 1 stage names and assert their order, plus the new event's
  count: the property it guards is stage ORDER, not the record's event count.
- **Doc-accuracy flag (sec-eng).** The flame inventory on Figure 3's left
  panel remains contested between readers (planner counts O/G/U/D, sec-eng
  counts G and U). design.md states only the load-bearing facts; the placement
  ruling depends on neither reading.
