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

The TuneFP stage record SHALL report `bias_delta_rel` — the relative movement
of the tuned linears' bias vectors — beside `weight_delta_rel`, which is
computed over all tuned parameters and therefore cannot say on its own whether
the bias compensation is real. A stage with no bias-bearing tunables SHALL
record no value, never a confident zero.

#### Scenario: tunefp_record_declares_bias_movement

- **WHEN** the stage runs over linears that carry a bias
- **THEN** its record's `bias_delta_rel` equals the relative bias movement
  recomputed from a pre-stage snapshot, and when the tunable linears carry no
  bias the field carries no value.
