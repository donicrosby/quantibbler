# qk-ptq delta: qk-fix-tunefp-teacher

## ADDED Requirements

### Requirement: TuneFP SHALL tune against the corpus-pinned cached teacher, per sequentially-factorized in-block layer

Per [NQ] Sec 3.2 Step 1 and Algorithm 1 (carried by the pinned corpus), the
full-precision tuning stage SHALL target `Y_b = B^FP_b(X_b)` — the pristine
FP block's output on the compressed-prefix stream, captured once before any
tuning — and SHALL run per sequentially-factorized in-block layer so the
objective carries the error of previously factorized layers. The TuneFP
internal loss formula is the informed assumption recorded in design.md.

#### Scenario: teacher_is_captured_before_tuning

- **WHEN** block reconstruction tunes fp weights for T_pre steps
- **THEN** the teacher tensor is the pristine block's output on the
  compressed-prefix stream, captured once before any tuning or in-block
  factorization, detached, with gradients disabled, and never recomputed
  from the tuned weights

#### Scenario: in_block_factorization_is_sequential_with_per_layer_tunefp

- **WHEN** a block contains multiple quantizable linears
- **THEN** layers are factorized sequentially, and TuneFP runs per layer
  against the cached teacher with previously-factorized in-block layers
  held quantized

#### Scenario: trivial_first_layer_is_expected

- **WHEN** TuneFP runs for the first in-block layer and the first-step
  gradient norm is zero while no in-block layer is factorized yet
- **THEN** the stage records `tune_fp_trivial_first_layer` and proceeds; no
  abort fires

#### Scenario: vacuous_teacher_fails_closed

- **WHEN** the pre-tune objective gradient norm is zero at the first optimizer
  step while at least one in-block layer is already factorized
- **THEN** the stage aborts with recorder event `tune_fp_vacuous_teacher`
  naming the block, the pipeline continues with pristine weights, and the
  event is visible in the run record

#### Scenario: tune_fp_effect_is_recorded_honestly

- **WHEN** a TuneFP stage runs to completion
- **THEN** the artifact records teacher provenance, the informed-assumption
  marker, and the fp weight/output delta versus pristine, and the delta is
  nonzero on at least one scoped linear after the first in-block layer
  whenever in-block factorization error is nonzero
