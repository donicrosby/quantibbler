# qk-ptq delta: qk-fix-tunefp-teacher

## ADDED Requirements

### Requirement: TuneFP SHALL tune the still-FP remainder against the cached teacher at the Figure 3 placement

Per [NQ] Sec 3.2 Step 1, Algorithm 1, Figure 3: the stage SHALL target
`Y_b = B^FP_b(X_b)` — the pristine FP block's output on the
compressed-prefix stream, captured once pre-tuning — and SHALL fire once
per block after the attention projections (Q/K/V) are factorized, tuning
the still-FP remainder (O/G/U/D). The TuneFP internal loss formula is the
informed assumption recorded in design.md.

#### Scenario: teacher_is_captured_before_tuning

- **WHEN** block reconstruction tunes fp weights for T_pre steps
- **THEN** the teacher tensor is the pristine block's output on the
  compressed-prefix stream, captured once before any tuning or in-block
  factorization, detached, with gradients disabled, and never recomputed
  from the tuned weights
- **AND** the teacher block's weights are the pristine FP weights loaded
  from the pinned model safetensors (or a frozen reference copy), never the
  live module's mutable state
- **AND** the artifact records the teacher source fingerprint (hash of the
  pristine weight source) and the capture index (which block the teacher
  belongs to), so provenance is provable from the artifact alone
- **AND** the same capture feeds TuneLatentSTE per [NQ] Algorithm 1 line 19,
  so Steps 1 and 3 share one teacher

#### Scenario: tunefp_fires_once_per_block_after_attention_factorization

- **WHEN** block reconstruction reaches a block's TuneFP pass
- **THEN** the block's attention projections (Q, K, V) are already
  factorized — Figure 3's left panel shows them green while the tuning
  flames sit on the still-FP remainder — and the tunable set is exactly
  the block's still-FP linears (O, G, U, D), tuned against the cached
  teacher with previously-factorized in-block layers held quantized
- **AND** the pass fires once per block per [NQ] Algorithm 1 line 12, at
  the Figure 3 placement — never at the pre-factorization line-12
  position where the objective is identically zero (the shipped defect
  signature)

#### Scenario: vacuous_teacher_fails_closed

- **WHEN** the pre-tune objective gradient norm is zero at the first optimizer
  step — whether at the operative post-Q/K/V call or at any call emitted
  with no in-block layer factorized (the legacy pre-loop placement, which
  the operative placement SHALL never produce)
- **THEN** the stage aborts with recorder event `tune_fp_vacuous_teacher`
  naming the block, and the event carries BOTH the first-step gradient norm
  and the initial residual norm — gradient-zero with residual-zero is a
  broken teacher (target equals current output), gradient-zero with
  residual-nonzero is a detached student (no gradient path)
- **AND** in a diagnostic run the pipeline continues with pristine weights
  and the event is visible in the run record
- **AND** in a run that claims paper fidelity, any `tune_fp_vacuous_teacher`
  event invalidates the claim: the run SHALL fail or record
  `paper_faithful: false`, never silently continue

#### Scenario: tune_fp_effect_is_recorded_honestly

- **WHEN** a TuneFP stage runs to completion
- **THEN** the artifact records teacher provenance naming the source
  (pristine safetensors or the frozen reference copy, with its fingerprint),
  the informed-assumption marker, and the fp weight/output delta versus
  pristine, and the delta is nonzero on at least one scoped linear — the
  operative call always runs with factorized attention projections, so a
  real effect is required whenever in-block factorization error is nonzero

#### Scenario: teacher_cache_is_memory_only_and_recovered_on_resume

- **WHEN** a TuneFP teacher cache exists for the run
- **THEN** the cache is memory-only in v1 and is never persisted to a
  checkpoint or artifact store
- **AND** on resume the teacher is re-captured from the pinned pristine
  source rather than restored from any cache
- **AND** any future persisted cache form SHALL be fingerprint-bound to the
  pristine source hash and the capture index before it may be trusted
