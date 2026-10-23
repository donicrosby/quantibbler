# qk-ptq delta: qk-nanoquant-alignment-lane4

## ADDED Requirements

### Requirement: Block tuning losses SHALL be weighted MSE with the terminal linear's Phase-1 output statistic

The TuneFP and TuneLatentSTE losses SHALL be weighted MSE over the block-output residual against the cached teacher: `mean(w ⊙ (B̂(X) − Y_b)²)` per output feature, with `w` the robust-diagonalized Phase-1 output statistic (`d_out_raw`) of the block's terminal linear (down_proj or arch-equivalent), per [NQ] App. C :3012 (weighted MSE) and [NQ] Algorithm 1 lines 2-5 with Step 2-1 Eq. 2 (statistic construction). The weight SHALL be detached and non-trainable, applied in the stage's compute dtype.

#### Scenario: weighting_reaches_the_loss_poisoned_weight_changes_loss

- **WHEN** fixtures run TuneFP and TuneLatentSTE on a toy block whose terminal linear's Phase-1 output statistic is poisoned at one entry (sentinel magnitude)
- **THEN** the poisoned weight vector changes the loss value relative to the run with the un-poisoned statistic
- **AND** the fixtures show the weighting reaches the loss: a poisoned weight vector changes the loss

### Requirement: Uniform fallback when the canonical layer is absent SHALL be recorded, never silent

When the canonical terminal linear (down_proj or arch-equivalent) is absent from the block, or carries no Phase-1 statistic, the tuning stages SHALL fall back to uniform weights AND SHALL record a `tune_loss_uniform_fallback` event naming the block and the missing canonical layer. A uniform-weighted stage without that event on record SHALL be impossible by construction.

#### Scenario: uniform_fallback_recorded_on_down_proj_less_block

- **WHEN** a down_proj-less toy block (no canonical terminal name among its linears) is tuned with terminal-gradient weighting requested
- **THEN** the uniform fallback fires
- **AND** the fallback is recorded as an event naming the block and the missing canonical layer — never silent

### Requirement: Stage records SHALL carry the weighting mode

The `tune_fp` and `ste` stage records SHALL carry the weighting mode actually used (`terminal_grad` or `uniform`) and the weight source (the terminal linear's name, or a reference to the recorded fallback event). The `block_loss_weight` config knob SHALL be solver state and SHALL join the checkpoint fingerprint.

#### Scenario: stage_records_carry_weighting_mode

- **WHEN** the `tune_fp` and `ste` stage records are emitted
- **THEN** each record carries the weighting mode
- **AND** a fallback run's records name the fallback event reference as the weight source

#### Scenario: loss_weight_knob_joins_checkpoint_fingerprint

- **WHEN** two configurations differ only in the `block_loss_weight` knob
- **THEN** their checkpoint fingerprints differ, so resume refuses across the knob change (solver state)
