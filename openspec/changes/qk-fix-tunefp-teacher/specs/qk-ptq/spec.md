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

### Requirement: The paper_faithful label SHALL be a computed verdict, never a forwarded flag

The reported `paper_faithful` field SHALL be computed from the run's own
record — the flag AND zero runtime deviations AND zero mechanism-class
deviations AND evidence that every required stage actually executed — so a
run can never carry the label while its protocol deviates, and a label
backed by an empty or incomplete stage record (which certifies nothing)
SHALL be false.

#### Scenario: label_flips_on_runtime_deviations

- **WHEN** the run identity records any runtime deviation (split tune
  geometry, non-protocol calibration row count, non-finite eval geometry)
- **THEN** the reported label is false regardless of the flag
- **AND** the report records the deviation list alongside the label

#### Scenario: label_flips_on_mechanism_deviations

- **WHEN** a paper-faithful-flagged run's record shows any of: lb_admm
  polish_rounds greater than 0, a STE discrete flip pass fired, a STE
  scale-only re-scope marker, a vacuous-teacher event, a stage failure, a
  solver other than admm, or a required stage (per the stage inventory's
  `required` column) that never executed
- **THEN** the reported label is false
- **AND** each condition is evaluated and recorded, so the label is
  auditable from the artifact alone

#### Scenario: required_stage_execution_reads_the_same_record

- **WHEN** the verdict evaluates required-stage execution
- **THEN** it derives run-level stage counts from the same normalized
  stage record every other condition reads, so an empty stage record
  denies the label on every required stage
- **AND** a stage event carrying a skip marker does not count as
  executed (`tune_fp` with `skipped` set, `tune_scales_kd` with
  `artifact` falsy) — executed means fired, not merely recorded
- **AND** the required set is materialized as a column of the stage
  inventory (`_STAGE_LINES`), never a second list beside the check, so
  the stage map and the verdict cannot drift
- **AND** required-stage conditions are recorded on every report path,
  naming the missing stage, but no non-requested run is denied by them —
  the flag condition already denies non-requested runs

#### Scenario: informed_assumptions_do_not_flip_the_label

- **WHEN** the record carries informed-assumption markers (the TuneFP loss
  formula)
- **THEN** the label is unaffected — those markers are documentation, not
  deviations

### Requirement: Phase-1 statistics SHALL be collected from the pristine model on the raw calibration set

RobustDiag statistics SHALL be collected once, in a Phase-1 pass over the
raw calibration set through the pristine FP model ([NQ] Algorithm 1 lines
2-5), before any block reconstruction; the block loop SHALL consume those
pre-collected per-linear statistics, and the per-block capture no longer
doubles as the statistics source.

#### Scenario: phase1_stats_from_pristine_model

- **WHEN** a run begins
- **THEN** one pre-loop pass runs the calibration rows through the
  pristine FP model, collecting per-linear statistics for every scoped
  linear (chunked/streamed collection at real scale, state O(feature
  width))
- **AND** block reconstruction consumes that pre-collected book — the
  compressed-prefix stream remains the tuning input ([NQ] line 9), never
  the statistics source

#### Scenario: stage_map_reflects_the_hoist

- **WHEN** the run records its stage map
- **THEN** the phase1_statistics entry carries no compressed-prefix-source
  deviation note
- **AND** the capture entry names only the line-9/10 teacher capture

### Requirement: Evaluation evidence SHALL state coverage and table comparability separately from method fidelity

The local PPL quality gate SHALL record its exact WT2 coverage and evaluator
semantics. Because [NQ] names WT2 perplexity and `lm_eval=0.4.9` but the pinned
corpus does not specify exact coverage/windowing, the local fixed-window metric
SHALL NOT claim paper-table comparability or redefine the method-fidelity
verdict.

#### Scenario: evaluation_coverage_is_artifact_provable

- **WHEN** the WT2 test shard is tokenized for baseline and quantized PPL
- **THEN** every report path records evaluator kind, source token count,
  complete blocks available, remainder tokens, selected blocks/tokens,
  scored predictions, complete-block and source-token coverage, selection
  seed, and a fingerprint of the ordered selected block indices
- **AND** the report records `paper_table_comparable: false` for the local
  fixed-window evaluator

#### Scenario: faithful_fire_uses_maximal_local_evaluation_window

- **WHEN** a paper-faithful-flagged run uses the local fixed-window evaluator
- **THEN** the requested eval batches equal the largest complete BATCH-sized
  group derivable from the tokenized test shard; smaller coverage fails
  preflight before the expensive solve
- **AND** for the pinned Qwen3-0.6B tokenizer/test shard measurement this is
  EVAL_SEQS=18: 144 of 145 complete 2,048-token blocks, with the 1,978-token
  remainder and one complete block explicitly recorded as uncovered

#### Scenario: evaluation_limit_does_not_masquerade_as_method_deviation

- **WHEN** the local evaluator uses its maximal complete-batch window
- **THEN** `paper_faithful` remains the computed verdict over quantization
  method and calibration protocol conditions
- **AND** evaluation comparability remains separately false, so neither field
  can be read as the other
