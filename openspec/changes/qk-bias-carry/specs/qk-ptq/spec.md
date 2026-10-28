# qk-ptq delta: qk-bias-carry

## ADDED Requirements

### Requirement: TuneFP SHALL persist the tuned bias

The TuneFP stage's tunable set SHALL include each remainder linear's bias, and
the factorized replacement built after that pass SHALL take the bias the pass
produced. Snapshotting and restoring a pre-tuning bias at factorization SHALL
NOT occur: the compensation the stage computed is the artifact's bias, and the
packed layer state SHALL carry it into the container.

#### Scenario: tunefp_bias_lands_in_the_artifact

- **WHEN** a block with bias-bearing linears is reconstructed
- **THEN** each factorized linear's packed state carries the bias the module
  held at its factorization — the value the preceding TuneFP pass wrote — and
  the host module's bias after reconstruction equals it, so the host forward
  is the artifact's forward.

### Requirement: The TuneFP record SHALL declare the bias movement

The TuneFP stage record SHALL report `bias_delta_rel` — the tuned linears'
relative bias movement — beside `weight_delta_rel`, which spans all tuned
parameters and cannot alone attest the bias compensation. The record SHALL name
which of three causes left the figure absent, never a confident `0.0`: no
bias-bearing tunables (`bias_tunables` 0); a stage that never ran (`steps` 0);
or an all-zero bias baseline (`steps` non-zero, division undefined).

#### Scenario: tunefp_record_declares_bias_movement

- **WHEN** the stage runs over linears that carry a bias
- **THEN** its record's `bias_delta_rel` equals the relative bias movement
  recomputed from a pre-stage snapshot, and when the tunable linears carry no
  bias the field carries no value.

#### Scenario: tunefp_record_bias_movement_undefined_over_all_zero_baseline

- **WHEN** the stage runs over bias-bearing linears whose selected biases are
  all exactly `0.0` at stage entry
- **THEN** its record's `bias_delta_rel` carries no value while `bias_tunables`
  and `steps` are both non-zero — the relative movement has no denominator, so
  it is left undefined rather than reported as a confident `0.0`.
