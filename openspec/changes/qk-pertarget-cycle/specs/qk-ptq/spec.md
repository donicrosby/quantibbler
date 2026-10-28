# qk-ptq delta: qk-pertarget-cycle

## MODIFIED Requirements

### Requirement: TuneFP SHALL tune the still-FP remainder against the cached teacher at the pertarget-cycle placement

The TuneFP stage SHALL fire per-target within the factorization loop, ahead of each target's own `_set_linear`. Its tunable set SHALL be the still-FP remainder (`quantizable_linears[idx:]`), with `factorized_in_block = idx`. The first target (`idx == 0`) SHALL be factorized with NO preceding TuneFP — a pass before it is bitwise vacuous (student == pristine teacher on the same `x_b`, residual and gradient identically zero). TuneFP therefore fires N−1 times per block and STE fires N times, where N is the number of quantizable linears in the block.

#### Scenario: teacher_is_captured_before_tuning

- **WHEN** block reconstruction tunes fp weights for T_pre steps
- **THEN** the teacher tensor is the pristine block's output on the compressed-prefix stream, captured once before any tuning or in-block factorization, detached, with gradients disabled, and never recomputed from the tuned weights
- **AND** the teacher block's weights are the pristine FP weights loaded from the pinned model safetensors (or a frozen reference copy), never the live module's mutable state
- **AND** the artifact records the teacher source fingerprint (hash of the pristine weight source) and the capture index (which block the teacher belongs to), so provenance is provable from the artifact alone
- **AND** the same capture feeds TuneLatentSTE per [NQ] Algorithm 1 line 19, so Steps 1 and 3 share one teacher

#### Scenario: tunefp_fires_per_target_within_factorization_loop

- **WHEN** block reconstruction reaches a block's factorization loop
- **THEN** TuneFP fires N-1 times (for N quantizable linears), each fire tuning the still-FP remainder before the target is factorized — matching the pertarget-cycle ruling (a) where `idx >= 1` gates the TuneFP call
- **AND** the first target (`idx == 0`) is factorized with NO preceding TuneFP — a pass before it has nothing factorized in-block, so the student block IS the pristine teacher block on the same input and the objective is identically zero
- **AND** the teacher target `Y_b` is captured from the fp-stream reading (per lane 2), and the same capture feeds TuneLatentSTE per [NQ] Algorithm 1 line 19, so Steps 1 and 3 share one teacher

#### Scenario: vacuous_teacher_fails_closed

- **WHEN** the pre-tune objective gradient norm is zero at the first optimizer step — whether at the operative post-Q/K/V call or at any call emitted with no in-block layer factorized (the legacy pre-loop placement, which the operative placement SHALL never produce)
- **THEN** the stage aborts with recorder event `tune_fp_vacuous_teacher` naming the block, and the event carries BOTH the first-step gradient norm and the initial residual norm — gradient-zero with residual-zero is a broken teacher (target equals current output), gradient-zero with residual-nonzero is a detached student (no gradient path)
- **AND** in a diagnostic run the pipeline continues with pristine weights and the event is visible in the run record
- **AND** in a run that claims paper fidelity, any `tune_fp_vacuous_teacher` event invalidates the claim: the run SHALL fail or record `paper_faithful: false`, never silently continue

#### Scenario: tune_fp_effect_is_recorded_honestly

- **WHEN** a TuneFP stage runs to completion
- **THEN** the artifact records teacher provenance naming the source (pristine safetensors or the frozen reference copy, with its fingerprint), the informed-assumption marker, and the fp weight/output delta versus pristine, and the delta is nonzero on at least one scoped linear — the operative call always runs with factorized attention projections, so a real effect is required whenever in-block factorization error is nonzero

#### Scenario: teacher_target_is_persisted_and_recovered_on_resume

- **WHEN** a TuneFP run checkpoint includes the fp-stream teacher target
- **THEN** the target is persisted as the reserved checkpoint npz key `y_b` (per `checkpoints.py:_YB_KEY`) and recovered on resume via `load_block_yb` to reconstruct the fp-stream prefix exactly
- **AND** a legacy checkpoint predating lane 2 carries no `y_b` key, and the resume path falls back to recomputing the fp stream through the pristine copy — never silently starting from a wrong activation

#### Scenario: single_linear_block_skips_tunefp

- **WHEN** a block contains exactly one quantizable linear
- **THEN** TuneFP fires zero times (N−1 = 0)
- **AND** the block is still eligible for factorization of its sole linear — a block with fewer than two quantizable linears is a legitimate skip, not an error

#### Scenario: still_fp_remainder_is_accurate

- **WHEN** TuneFP fires at position idx
- **THEN** its tunable set is `quantizable_linears[idx:]` (the still-FP remainder)
- **AND** `factorized_in_block` equals `idx`, so the vacuity guard `factorized_in_block <= 0` is unreachable in production (first call always has idx ≥ 1)

### Requirement: The STE refinement SHALL run once per block over the block's whole factorized set

The per-target STE cycle (AMEND ruling 2026-10-28, @user) SHALL replace the block-joint pass. Per [NQ] Sec 3.2 Step 3 prose (:641, "the current target linear layer", singular) and Figure 3 — not the block-scoped Algorithm 1 line 19 pseudocode (:780, one `TuneLatentSTE(b,X_b,Y_b;T_post)` call placed after the per-linear factorization loop closes) — the STE refinement SHALL fire N times per block, once per target linear after its factorization, each fire consuming the full [NQ] App. C factorized-stage budget per linear ([NQ] App. C :3005). The block-scoped pseudocode (:780) is declared a tension with the per-target prose (:641) and figure; the per-target reading is the governing interpretation per the 2026-10-28 AMEND ruling. The fidelity condition records this as `ste_per_target: true` (the AMEND path), `ste_block_joint: false` (the overruled path); the label `paper_faithful` SHALL be qualified by the prose-vs-pseudocode citation when downstream fidelity machinery keys on it. Per-target restoration adds ~6.1 h/run (v6b CPU: 1.02 h joint STE × 7 ≈ 7.1 h total, +6.1 h) / ~4.2 h/run (v7 GPU: 0.70 h joint → 4.9 h total, +4.2 h), measured from `quantize_report.json` STE stage rows. The `block_joint` scope marker retires (row vocabulary kept for old-report readability).

#### Scenario: ste_refines_all_factorized_linears_in_one_block_pass

- **WHEN** a block carrying N factorized linears is reconstructed
- **THEN** TuneLatentSTE is invoked N times for that block, once per target linear after its factorization — not once with all N linears as free parameters
- **AND** each per-linear row carries that linear's own block-output objective measurement

#### Scenario: per_target_ste_refines_one_linear_at_a_time

- **WHEN** a block's reconstruction loop reaches target i after its factorization
- **THEN** TuneLatentSTE fires on that single target linear against the block-output objective, consuming the full App. C factorized-stage budget (epochs × rows/batch steps at lr 1e-5, [NQ] App. C :3005)
- **AND** the stage record carries one per-linear row with that target's individual movement metrics

#### Scenario: per_target_ste_cycle_sequence

- **WHEN** a block carries N quantizable linears
- **THEN** the cycle is: TuneFP on remainder (skipped for idx=0) → factorize target i → STE on target i → harden signs → next target
- **AND** TuneFP fires N−1 times and STE fires N times, each on ONE target linear with the full App. C budget
- **AND** each STE fire's student contains exactly one factorized linear plus the still-FP remainder

#### Scenario: per_target_budget_is_appendix_c_per_linear

- **WHEN** per-target STE fires on a single linear
- **THEN** the stage record shows the App. C budget consumed for that one linear: `epochs × (rows // batch_size)` steps at lr 1e-5
- **AND** the total STE budget across a block of N linears is N× the App. C factorized-stage budget — the paper's budget unit is the linear, not the block

### Requirement: The STE stage SHALL emit one block-scope record carrying per-linear rows

Each per-target STE fire SHALL emit exactly one record with a scope marker declaring the firing per-target and the target linear's identity. The record SHALL carry the stage's per-fire summary (steps, wall time, gradient floor, vacuity, inertness) and one row for the target linear; each row SHALL carry that linear's latent and scale movement, its sign-flip counts, and its dead-column count.

#### Scenario: ste_record_is_one_block_record_with_per_linear_rows

- **WHEN** a block's refinement completes (N per-target STE fires)
- **THEN** N STE records are emitted, each carrying one row for its target linear and a scope marker declaring per-target with the linear's identity
- **AND** each record's `weight_delta_rel`, flip counts, and dead-column count belong to that one linear

#### Scenario: dead_columns_are_recorded_per_linear

- **WHEN** a factorized linear carries a rank index whose outer-product column is zero (either factor's column identically zero)
- **THEN** that linear's row reports the number of such dead rank components

### Requirement: The joint STE pass SHALL weight its loss by the normalized terminal statistic and guard on the resolution floor

The per-target STE pass SHALL weight the block-output loss by the terminal linear's Phase-1 output statistic normalized to unit mean, and SHALL test its first-step gradient norm against the floor derived from the optimizer eps and the parameter dtype's relative resolution, taking the existing vacuous/inert declaration path. Both quantities SHALL arrive resolved from the block's resolve site rather than re-derived inside the stage.

#### Scenario: joint_pass_uses_the_normalized_weight_and_the_floor

- **WHEN** a per-target pass runs on a block whose terminal statistic resolved to a weight
- **THEN** the weight handed to the loss has unit mean, both pre- and post-normalization means are recorded, and the record carries the resolution-derived gradient floor

### Requirement: The joint STE pass SHALL consume the App. C factorized-stage regime once per block

The per-target STE pass SHALL run `epochs × (rows // batch_size)` steps at the factorized stage's learning rate under the stage's cosine schedule ([NQ] App. C), consumed afresh for each target linear. The total budget across a block of N linears is N× that regime.

#### Scenario: joint_budget_is_the_appendix_c_regime_once_per_block

- **WHEN** a per-target STE pass runs on a single linear
- **THEN** the recorded stage budget is `epochs × (rows // batch_size)` and the per-linear row records that budget as its scheduled steps
- **AND** the block's total STE budget across N linears is N× the App. C regime

## ADDED Requirements

### Requirement: The pre-group/remainder split machinery SHALL be deleted

The `split` logic that divided quantizable linears into a pre-group and remainder, and the `_is_attention_projection` helper, SHALL be deleted — they existed only to serve the overruled "one TuneFP pass per block, fired after Q/K/V were factorized" reading.

#### Scenario: split_machinery_absent

- **WHEN** the codebase is searched for the pre-group/remainder split and `_is_attention_projection`
- **THEN** neither exists — both are deleted
- **AND** no caller references either deleted function

### Requirement: TuneLatentSTE SHALL record objective per-fire evidence

`TuneLatentSTE` SHALL record, per fire, `residual_before` and `residual_after` over the FULL capture (not a sampled subset), plus the first-step gradient norm. This replaces the overruled `zero_crossings` metric as the primary evidence of whether the stage did anything.

#### Scenario: objective_evidence_per_fire

- **WHEN** TuneLatentSTE completes a per-target fire
- **THEN** `residual_before` and `residual_after` are recorded over the full capture block's teacher/student outputs
- **AND** the first-step gradient norm is recorded
- **AND** the evidence is persisted in the artifact rows for downstream verification

### Requirement: The driver-level stage-map invariant SHALL be derived from the ruling, not asserted from the overruled shape

The test `test_driver_stage_map.py` SHALL replace the overruled pin (`tune_fp.invocations == capture.invocations`) with a derived invariant: `tune_fp.invocations == factorized_linears − blocks` (18 on the tiny model: `BLOCKS = 3` × 7 quantizable linears/block = 21 factorized linears, 21 − 3 = 18), asserting FOR the per-target ruling and AGAINST "once per block."

#### Scenario: derived_invariant_asserts_per_target

- **WHEN** the driver stage-map test runs on the tiny model (3 blocks, 7 quantizable linears/block, 21 factorized linears)
- **THEN** `tune_fp.invocations` equals `factorized_linears − blocks` (21 − 3 = 18, N−1 per block)
- **AND** the test explicitly asserts that the observed invocations are inconsistent with "once per block" (which would predict 3 TuneFP fires, one per block — the 6× gap between 3 and 18 distinguishes the shapes unambiguously)

### Requirement: A CI report gate SHALL verify stage evidence presence

`tools/ci/report_checks.py` SHALL include a `stage_evidence_findings()` gate that inspects report.json for the presence of objective stage evidence. On runs claiming paper fidelity, the gate SHALL assert a per-cycle net-reduction test against `STEPS_EFFECT_EPS=1e-3`.

#### Scenario: stage_evidence_gate_on_faithful

- **WHEN** a report claims `paper_faithful_flag: true` (the claim that `--paper-faithful` was on the argv), regardless of the rolled-up `paper_faithful` verdict
- **THEN** the gate verifies that STE's per-cycle net reduction exceeds `STEPS_EFFECT_EPS` for at least one linear — a no-op stage on a faithful run is a finding

#### Scenario: stage_evidence_gate_on_non_faithful

- **WHEN** a report does NOT claim `paper_faithful_flag: true` (e.g. epochs=0 non-faithful runs)
- **THEN** the gate performs shape checks only — the non-faithful driver's `stage_pre` sets `epochs=0`, leaving TuneFP at one step of the paper's 256, and asserting the floor there would red-light correct behavior

### Requirement: A host-side stall watchdog SHALL detect frozen pipeline runs

`tools/qkb-stall-watchdog.sh` SHALL monitor the container's stdout for stall conditions. It SHALL be alert-only by default (`WATCHDOG_KILL_AFTER_MIN=0`) and SHALL integrate with `run-ptq-container.sh` so that every container fire carries the watchdog.

#### Scenario: watchdog_alert_only_by_default

- **WHEN** the watchdog is started with default configuration
- **THEN** it emits alerts on stall detection but does NOT kill the container
- **AND** setting `WATCHDOG_KILL_AFTER_MIN > 0` enables auto-kill after the specified minutes of stall

### Requirement: Stage learning rates SHALL be [NQ] App. C absolute values when the stage input is the balanced class

TuneLatentSTE and TuneScalesKD lrs SHALL be [NQ] App. C absolute values (1e-5 factorized, 1e-6 global, App. C :3005–3006) when parameters are balanced. Per-tensor relative anchoring `lr × median|p|` SHALL NOT govern balanced inputs. When the input is NOT balanced, the per-tensor relative form SHALL apply as the else-branch. Scale-class anchoring SHALL move to the stage boundary. Disposition: AMENDS per-tensor-relative-lr-as-inert-stage-cure ruling (2026-10-28, @user).

#### Scenario: factorized_stage_lr_is_absolute_under_balanced_inputs

- **WHEN** TuneLatentSTE initializes its optimizer for balanced latents (median magnitude in [0.01, 1.0]) and balanced s1/s2 scales
- **THEN** each parameter group's lr is the App. C absolute value 1e-5, not `1e-5 × median|p|`
- **AND** the effective step magnitude is comparable across U, V, s1, and s2 — no parameter class is strangled below App. C's calibrated regime

#### Scenario: global_scale_lr_is_absolute_under_balanced_scales

- **WHEN** TuneScalesKD initializes its optimizer for globally balanced scale vectors (typical post-ADMM range — a few decades at most, never the polish-era 13+ decade pathological spread)
- **THEN** each parameter group's lr is the App. C absolute value 1e-6, not `1e-6 × median|p|`
- **AND** the optimizer applies the intended per-step magnitude to every scale entry

#### Scenario: scale_class_anchoring_at_stage_boundary

- **WHEN** a stage's input spans multiple scale classes (e.g. s1 entries ranging 1e-2 to 1e2 after a divergent solve, before any rebalancing)
- **THEN** gamma-rebalancing to unit-median is applied at the stage boundary BEFORE lr derivation — the optimizer sees balanced parameters with absolute App. C lrs
- **AND** the rebalancing factor is recorded in the stage record so downstream consumers can unwind it

#### Scenario: pathological_class_survivor_uses_relative_form

- **WHEN** a stage's input is NOT balanced (median magnitude outside [0.01, 1.0] — the pathological class that motivated the polish-era relative anchor)
- **THEN** the per-tensor relative form `lr × median|p|` applies as the else-branch
- **AND** the stage record carries the class determination (balanced/pathological), the median magnitude, and the effective lr

### Requirement: Eq. 8 scale extraction SHALL be the initialization contract with no global LS refit against W

Magnitude balance SHALL output s1, s2 per [NQ] Eq. 8: row mean-abs of η-balanced latents. No global LS refit SHALL be folded into s1. The shipped scaled-product refit (`lb_admm.py:702–707`: `α = ⟨W, M⟩ / ‖M‖²_F` with `M = (s1⊙Ub)(s2⊙Vb)ᵀ`) SHALL be removed. The artifact sanity gate SHALL gain a per-layer unit-gain leg with the `norm_ratio` field read from `qk_quantized_layers`. This requirement depends on F1 for correct downstream compensation — F1 and F2 SHALL land together.

#### Scenario: scales_are_eq8_row_mean_abs_no_refit

- **WHEN** magnitude_balance processes balanced latents U, V with target=None
- **THEN** the returned s1, s2 equal the per-row mean absolute values of ηU and η⁻¹V respectively
- **AND** no scalar `α` multiplies s1 — the scale product `s1 ⊙ s2` carries exactly the magnitude information of the balanced latents

#### Scenario: artifact_sanity_gate_per_layer_unit_gain

- **WHEN** the artifact sanity gate inspects a quantized layer's packed product norm ratio against the original weight W, reading the `norm_ratio` field from `qk_quantized_layers`
- **THEN** the ratio `‖(s1⊙Ub)(s2⊙Vb)ᵀ‖_F / ‖W‖_F` falls within the calibrated band
- **AND** the band edges are derived from measured min/max across production layers, with explicit arithmetic on both sides and a suite test asserting the honest extremes pass

#### Scenario: global_refit_is_removed_from_magnitude_balance

- **WHEN** the codebase is searched for the LS refit path inside magnitude_balance (the `alpha` computation `α = ⟨W, M⟩ / ‖M‖²_F` with `M = (s1⊙Ub)(s2⊙Vb)ᵀ`)
- **THEN** the refit code does not exist — magnitude_balance scale output is Eq. 8 only
- **AND** any consumer that previously depended on the refit's gain normalization receives the raw Eq. 8 scales and SHALL compensate through the downstream tuning stages (TuneLatentSTE, TuneScalesKD) operating at the correct App. C absolute lrs per F1, landed together with this requirement
