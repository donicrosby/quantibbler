# qk-ptq delta: qk-tune-loss-scale

## ADDED Requirements

### Requirement: The terminal-statistic tuning loss weight SHALL be normalized to unit mean before use

The TuneFP and TuneLatentSTE losses SHALL be weighted MSE by the terminal linear's robust-diagonal Phase-1 output statistic normalized to unit mean, preserving its relative per-feature structure while restoring its absolute scale to the residual's own scale ([NQ] App. C weighted MSE; the statistic's construction is Algorithm 1 lines 2-5, Step 2-1 Eq. 2). Both stage records SHALL carry the pre-normalization and post-normalization weight means.

#### Scenario: weight_is_normalized_before_the_loss

- **WHEN** the block's terminal statistic resolves to a weight whose mean is not 1
- **THEN** the weight handed to TuneFP and TuneLatentSTE has mean 1 while every pairwise ratio of its entries is unchanged
- **AND** both stage records carry the pre-normalization mean and the post-normalization mean

### Requirement: The vacuous-stage guard SHALL test a resolution-derived gradient floor, not exact zero

A tuning stage whose first-step gradient norm is below a floor derived from the optimizer's eps and the parameter dtype's relative resolution SHALL be recorded vacuous and SHALL take the existing fail-closed path. An identically-zero gradient SHALL remain a special case of that floor. The stage record SHALL carry the floor, the optimizer eps, and the dtype relative resolution the floor was derived from.

#### Scenario: below_floor_gradient_declared_vacuous

- **WHEN** a tuning stage's first-step gradient norm is nonzero but below the resolution-derived floor
- **THEN** the stage record reads `vacuous: true` with a reason naming the floor, and the stage takes the existing fail-closed path
- **AND** the record carries the floor, the optimizer eps, and the dtype relative resolution it used
- **AND** an identically-zero gradient reaches the same vacuous outcome as a special case of that floor

### Requirement: Tuning-stage records SHALL declare inertness, not imply it

A tuning stage that completes its full step budget with parameter movement below the parameter dtype's relative resolution SHALL record `inert: true`. A TuneLatentSTE row whose per-linear movement is below that resolution SHALL record inert on its own per-linear row. A stage record without the declaration SHALL be read as unknown, never as an effective stage.

A movement figure carries no value, never a confident `0.0`, when the stage never ran or an exactly-zero baseline leaves no denominator.

#### Scenario: full_budget_without_movement_is_declared_inert

- **WHEN** a tuning stage runs its full step budget and its parameter movement is below the parameter dtype's relative resolution
- **THEN** its stage record carries `inert: true`
- **AND** a stage whose movement exceeds that resolution carries `inert: false` rather than the key being absent

#### Scenario: movement_figure_undefined_over_all_zero_baseline

- **WHEN** a tuning stage ran over parameters whose entry baseline is exactly zero — an aggregate `weight_delta_rel` over an all-zero weight baseline, a `bias_delta_rel` over an all-zero bias baseline, or a TuneLatentSTE row figure over an all-zero entry parameter
- **THEN** the affected figure carries no value while `steps` attests the stage ran — the relative movement has no denominator, so it is left undefined rather than reported as a confident `0.0`
- **AND** the inertness declaration carries no value for the affected TuneLatentSTE row and for the stage — never a verdict aggregated over a figure that does not exist
