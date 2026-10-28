# qk-ptq delta: qk-nanoquant-alignment-lane4

## ADDED Requirements

### Requirement: Block tuning losses SHALL be weighted MSE with the terminal linear's Phase-1 output statistic

The TuneFP and TuneLatentSTE losses SHALL be weighted MSE, elementwise, by the robust-diagonalized Phase-1 output statistic of the block's terminal linear. The weight SHALL be detached, non-trainable, and applied in the stage's compute dtype. The canonical terminal linear SHALL be `down_proj`; arch-equivalent expansion is a future mechanism.

#### Scenario: weighting_reaches_the_loss_poisoned_weight_changes_loss

- **WHEN** fixtures run TuneFP and TuneLatentSTE on a toy block whose terminal linear's Phase-1 output statistic is poisoned at one entry (sentinel magnitude)
- **THEN** the poisoned weight vector changes the loss value relative to the run with the un-poisoned statistic
- **AND** the per-feature weight formula is `w = robust_diag(d_out_raw[terminal], tau=cfg.tau, gamma=cfg.gamma)` per [NQ] App. C :3012 and Algorithm 1 lines 2-5 Eq. 2
- **AND** this lane pins `CANONICAL_TERMINAL_NAMES = ("down_proj",)`; arch-equivalent names expand from this set via a future arch-to-role table

### Requirement: Uniform fallback when the canonical layer is absent SHALL be recorded, never silent

When the canonical terminal linear (`down_proj`) is absent from the block, or carries no Phase-1 statistic, the tuning stages SHALL fall back to uniform weights. A stage whose `loss_weight_source == "uniform_fallback"` SHALL carry the `tune_loss_uniform_fallback` event on record naming the block and the missing canonical layer. A stage whose `loss_weight_source == "uniform"` (explicit ablation) SHALL NOT emit the event.

#### Scenario: uniform_fallback_recorded_on_down_proj_less_block

- **WHEN** a down_proj-less block (no `down_proj` among its discovered linear names, e.g. a gpt2 `mlp.c_proj`) is tuned with terminal-gradient weighting requested
- **THEN** the uniform fallback fires
- **AND** the fallback is recorded as an event naming the block and the missing canonical layer — never silent

### Requirement: Test harnesses SHALL discover linear names from the loaded model, not hardcode Qwen3-specific names

All weighting and resolution fixtures SHALL discover linear layer names from the model's structure (`named_modules()`, `state_dict()` keys, or HF config). No fixture SHALL hardcode a Qwen3-specific layer name. Arch-to-role expansion is a future spec change; this lane's fixtures SHALL parameterize over known-good architectures and SHALL fail loudly on an unknown architecture.

#### Scenario: fixtures_discover_names_not_hardcode_qwen3

- **WHEN** the toy block fixture provides a synthetic block whose module tree carries non-Qwen3 names (`fc1`, `fc2`, `c_proj`)
- **THEN** the canonical-terminal resolution discovers the terminal by searching for `down_proj` among the block's actual names
- **AND** a test parameterized over `[qwen3, llama, gpt2]` passes for qwen3 and llama (weighting path) and passes for gpt2 (fallback path with recorded event)
- **AND** no test file contains a hardcoded Qwen3 layer name literal (`"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`) outside of arch-map declaration rows

### Requirement: Stage records SHALL carry the weighting mode

The `tune_fp` and `ste` stage records SHALL carry the weighting mode actually used (`terminal_grad` or `uniform`) and the weight source (the terminal linear's name, or a reference to the recorded fallback event). The `block_loss_weight` config knob SHALL be solver state and SHALL join the checkpoint fingerprint.

#### Scenario: stage_records_carry_weighting_mode

- **WHEN** the `tune_fp` and `ste` stage records are emitted
- **THEN** each record carries the weighting mode
- **AND** a fallback run's records name the fallback event reference as the weight source

#### Scenario: loss_weight_knob_joins_checkpoint_fingerprint

- **WHEN** two configurations differ only in the `block_loss_weight` knob
- **THEN** their checkpoint fingerprints differ, so resume refuses across the knob change (solver state)
