# qk-ptq delta: qk-nanoquant-alignment-lane3

## ADDED Requirements

### Requirement: The continuous update SHALL regress against row/col-normalized regressors with a scale-relative ridge

The LB-ADMM continuous update SHALL build its design matrix from the row/col-normalized regressor X (the fixed partner factor of Eq. 5, [NQ] :2877) and SHALL replace the absolute ridge shift `(ρ+λ)I` of Eq. 37 ([NQ] App. B.4 :2916) with the scale-relative stabilizer `ρ·mean(diag(XᵀX)) + reg`, where `reg` is a config field (default 3e-2). The `QK_RHO_END` env-provenance pattern SHALL extend to the new fields, and the checkpoint fingerprint SHALL pick up every new solver-state field.

#### Scenario: normalized_regressor_design_matrix

- **WHEN** the continuous update solves for U given the fixed partner factor (symmetrically for V)
- **THEN** the design matrix is built from the row/col-normalized proxy (the fixed partner factor serving as the regressor of Eq. 5, [NQ] :2877), so the Gram matrix XᵀX is O(1)-scale with unit-scale diagonal
- **AND** the returned iterate is mapped back to the unnormalized factor basis, so the product UVᵀ remains a valid factorization of the target

#### Scenario: stabilizer_scale_relative_ridge

- **WHEN** the ridge shift is computed at schedule value ρ
- **THEN** the shift is exactly `stabilizer = ρ·mean(diag(XᵀX)) + reg` with `reg` read from a config field whose default is 3e-2
- **AND** the stabilizer depends only on the normalized regressor, never on raw layer scale, so one ρ schedule serves every layer

#### Scenario: reg_env_override_explicit_fail_closed

- **WHEN** `QK_ADMM_REG` is set in the environment
- **THEN** the effective `reg` is the env value with source `"env"`, recorded beside the value in the solver probe rows and the run report exactly like `rho_end_source` (the `rho_end_provenance` pattern, `lb_admm.py:325-341`)
- **AND** an unparseable `QK_ADMM_REG` raises an error naming the variable, never falling back to the default silently

#### Scenario: new_solver_state_joins_fingerprint

- **WHEN** a checkpoint is resumed
- **THEN** two configs differing only in a new solver-state field (`admm_reg`, recalibrated ρ endpoints) produce different checkpoint fingerprints
- **AND** the new fields are absent from `_FINGERPRINT_EXCLUDED`, asserted by a suite test so a future exclusion-list edit cannot silently drop solver state

### Requirement: The ADMM penalty SHALL follow the paper's linear schedule over K=400 with endpoints re-calibrated for the normalized regime

ρ SHALL follow the paper's linear schedule across K=400 factorization steps ([NQ] App. C :3008) with endpoints re-calibrated for the normalized-regressor regime: linear ~0→1. With the scale-relative ridge, a final ρ of ~1 no longer creates the consensus dominance qk-fix-admm-solver capped at `1e-1`; that cap's rationale is scoped to the absolute-ridge regime this change replaces. The tension between the two sibling deltas is named in this change's proposal.

#### Scenario: rho_endpoints_recalibrated_normalized_regime

- **WHEN** LB-ADMM runs with default ADMM parameters in the new regime
- **THEN** ρ follows the paper's linear schedule over K=400 ([NQ] App. C :3008) with endpoints ~0→1
- **AND** at schedule end the ridge `ρ·mean(diag(XᵀX)) + reg` sits at the data term's own scale, so the continuous fit is not crushed (the defect qk-fix-admm-solver measured and capped)
- **AND** well-posedness holds at ρ≈0: the `reg` floor keeps the continuous subproblem SPD per [NQ] App. B.4 Lemma 2

### Requirement: The entry Frobenius normalization SHALL be kept or removed deliberately under the new regime

The entry Frobenius normalization (target divided by its Frobenius norm, `lb_admm.py:376-380`) may become redundant once the regressor is normalized and the ridge is scale-relative. Its keep-or-remove decision SHALL be made from with/without measurements, deliberately, and the decision with its rationale SHALL be recorded in this change's design.md.

#### Scenario: frobenius_normalization_keep_or_remove_recorded

- **WHEN** the new regime is measured with and without the entry Frobenius normalization
- **THEN** the keep-or-remove decision is deliberate and recorded in this change's design.md with the measurements and rationale
- **AND** the default path SHALL NOT silently drop the normalization before that decision is recorded

### Requirement: The ADMM regime SHALL be accepted only on Lane-3 measurements against the Lane-0 baseline

The new regime SHALL be accepted only on Lane-3 measurements against the Lane-0 baseline. Acceptance requires: on production layers, post-ADMM pre-polish cos improves materially over the Lane-0 baseline (threshold set FROM Lane-0 measurements, not invented); the `trace.min_eigs` floor rises; the `solver="svd"` ablation is RE-RUN under the new regime before anyone quotes "ADMM ≈ warm start" again. Lane order is load-bearing: Lane 0 gates 1-3 and Lane 1 lands before Lane 3.

#### Scenario: post_admm_pre_polish_cos_over_lane0_baseline

- **WHEN** the new regime runs on production layers with the solver probe recording per-layer cos
- **THEN** on production layers, post-ADMM pre-polish cos improves materially over the Lane-0 baseline (threshold set FROM Lane-0 measurements, not invented)
- **AND** no threshold constant is authored in code or spec before the Lane-0 measurements land; the pinned threshold is recorded in this change's design.md from those measurements

#### Scenario: min_eigs_floor_rises

- **WHEN** the new regime runs on a fixed production target
- **THEN** the `trace.min_eigs` floor rises
- **AND** the floor is bounded below by `reg` at every iteration (the stabilizer at ρ≈0), and the measured floor exceeds the current regime's floor on the same target

#### Scenario: svd_ablation_rerun_under_new_regime

- **WHEN** the `solver="svd"` comparison is used to claim what ADMM buys over the warm start
- **THEN** the `solver="svd"` ablation is RE-RUN under the new regime before anyone quotes "ADMM ≈ warm start" again
- **AND** old-regime ablation rows are inadmissible for the claim — every comparison row records its regime (ρ endpoints, `reg`, normalization, sources)

#### Scenario: lane_order_gates_acceptance

- **WHEN** Lane-3 acceptance numbers are quoted as evidence
- **THEN** Lane 0 and Lane 1 have landed — solver regime numbers only mean something against the fixed preconditioner
- **AND** numbers quoted before both land are marked provisional, never acceptance evidence
