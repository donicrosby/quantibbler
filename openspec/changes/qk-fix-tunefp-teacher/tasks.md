# Tasks: qk-fix-tunefp-teacher

- [ ] (coder) Implement per the spec delta: teacher capture per [NQ]
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
- [ ] (coder) Self-test: a regression test constructs the shipped
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
- [ ] (coder) Rehearse the stage-trace probe against the fixed bytes (toy
  scope first, then the 7 scoped linears if budgeted): `tune_fp_effect` must
  be nonzero on at least one scoped linear with a real teacher; the abort
  event `tune_fp_vacuous_teacher` must be absent; record the new probe
  sha. Post the rehearsal output.
- [ ] (coder) Faithful-run stage map: the rehearsal run record maps every
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
