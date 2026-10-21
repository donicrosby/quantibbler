# Tasks: qk-fix-tunefp-teacher

- [ ] (coder) Implement per the spec delta: teacher capture per [NQ]
  Algorithm 1 (`y_fp` = pristine block output on the compressed-prefix
  stream, cached pre-tuning, detached, no_grad); sequential in-block
  factorization with `_tune_fp` run per layer against the cached teacher;
  informed-assumption marker for the TuneFP loss formula; guard: zero
  first-step gradient with ≥1 in-block layer factorized → abort with
  `tune_fp_vacuous_teacher`; trivial-zero first in-block layer → record
  `tune_fp_trivial_first_layer` and proceed. Defaults stay [NQ] App. C.
- [ ] (coder) Self-test: a regression test constructs the shipped
  self-referential teacher with an in-block layer already factorized and
  asserts the guard fires (fail-closed path); a second test with the
  corpus-pinned teacher asserts the objective gradient is nonzero once an
  in-block layer is factorized and the tuned block reduces teacher error
  over T_pre steps; a third asserts the first-layer trivial-zero case
  records `tune_fp_trivial_first_layer` without aborting.
- [ ] (coder) Rehearse the stage-trace probe against the fixed bytes (toy
  scope first, then the 7 scoped linears if budgeted): `tune_fp_effect` must
  be nonzero after the first in-block layer with a real teacher; the abort
  event `tune_fp_vacuous_teacher` must be absent while
  `tune_fp_trivial_first_layer` is expected on record; record the new probe
  sha. Post the rehearsal output.
- [ ] (bug-hunter) Review on the landed bytes: teacher captured once before
  tuning from the pristine block on the compressed-prefix stream, never
  recomputed from tuned weights; TuneFP runs per sequentially-factorized
  in-block layer; guard is fail-closed and scoped to the ≥1-factorized case;
  the trivial-first-layer event fires without abort; no call site still
  passes self-outputs as the teacher; artifact marker present. Verify the
  regression tests run green from clean checkout.
- [ ] (sec-eng) Integrity pass: the fp-prefix pass adds no new network or
  persistence surface; the abort path cannot corrupt in-flight state
  (snapshot-restore discipline preserved); provenance fields provable from
  the artifact alone.
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-tunefp-teacher --strict`; review the delta against the frozen fork
  timing (this fix is owed regardless of the ladder verdict); commit
  (specs + code in one commit), do not archive until a real run shows a
  nonzero TuneFP effect on record.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." This change and its siblings land before any further diagnostic
runs; the ladder fork fires only after a paper-faithful run exists.
