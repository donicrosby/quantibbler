# qk-ptq delta: qk-ste-joint

## ADDED Requirements

### Requirement: The STE refinement SHALL run once per block over the block's whole factorized set

Per block, after every factorization completes and before the stream swap, TuneLatentSTE SHALL run exactly once over ALL of the block's factorized linears: one optimizer holding every latent and scale parameter set, and one loss on the block output against the frozen teacher target `y_b` ([NQ] Eq. 10). The paper's per-target-layer reading is a declared deviation (design.md).

#### Scenario: ste_refines_all_factorized_linears_in_one_block_pass

- **WHEN** a block carrying N factorized linears is reconstructed
- **THEN** TuneLatentSTE is invoked exactly once for that block, with all N linears as free parameters
- **AND** every per-linear row of that block carries the same block-objective measurement, because the pass had one objective

### Requirement: The STE stage SHALL emit one block-scope record carrying per-linear rows

Each block's refinement SHALL emit exactly one STE record. The record SHALL carry the stage's block-scope summary (steps, wall time, gradient floor, vacuity, inertness, and a scope marker) and one row per factorized linear; each row SHALL carry that linear's latent and scale movement, its sign-flip counts, and its dead-column count under the joint pass.

#### Scenario: ste_record_is_one_block_record_with_per_linear_rows

- **WHEN** a block's refinement completes
- **THEN** exactly one STE record is emitted, carrying a row for every factorized linear and a scope marker declaring the firing block-joint
- **AND** no per-linear invocation count is implied by the record

#### Scenario: dead_columns_are_recorded_per_linear

- **WHEN** a factorized linear carries a rank index whose outer-product column is zero (either factor's column identically zero)
- **THEN** that linear's row reports the number of such dead rank components

### Requirement: The joint STE pass SHALL weight its loss by the normalized terminal statistic and guard on the resolution floor

The joint pass SHALL weight the block-output loss by the terminal linear's Phase-1 output statistic normalized to unit mean, and SHALL test its first-step gradient norm against the floor derived from the optimizer eps and the parameter dtype's relative resolution, taking the existing vacuous/inert declaration path. Both quantities SHALL arrive resolved from the block's resolve site rather than re-derived inside the stage.

#### Scenario: joint_pass_uses_the_normalized_weight_and_the_floor

- **WHEN** the joint pass runs on a block whose terminal statistic resolved to a weight
- **THEN** the weight handed to the loss has unit mean, both pre- and post-normalization means are recorded, and the record carries the resolution-derived gradient floor

### Requirement: The joint STE pass SHALL consume the App. C factorized-stage regime once per block

The joint pass SHALL run `epochs x (rows // batch_size)` steps at the factorized stage's learning rate under the stage's cosine schedule ([NQ] App. C), consumed ONCE for the whole block. No per-linear share and no block-pooled allocation SHALL be applied on top of that regime.

#### Scenario: joint_budget_is_the_appendix_c_regime_once_per_block

- **WHEN** the joint pass runs on a block of N factorized linears
- **THEN** the recorded stage budget is `epochs x (rows // batch_size)` and every per-linear row records that same budget as its scheduled steps
