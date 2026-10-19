## ADDED Requirements

### Requirement: The pipeline SHALL gate each reconstructed layer against the fp weight's magnitude before any refinement or downstream use

Immediately after magnitude balancing yields `(Ub, Vb, s1, s2)` for a linear layer, the pipeline SHALL verify the binary reconstruction's Frobenius norm against the fp weight's within the band factor `PTQConfig.divergence_band_factor` (default 1000.0, a catastrophe bound calibrated to the measured honest range of balance-time ratios), and SHALL require every scale entry to be finite. A violation SHALL abort quantization with a dedicated error naming the layer, the measured ratio, and the band.

#### Scenario: divergence_gate_healthy_passes

- **WHEN** magnitude balancing produces scales whose binary reconstruction's Frobenius norm is within the configured factor of the fp weight's Frobenius norm
- **THEN** the gate passes without raising, and the pipeline proceeds to refinement with the balanced factors

#### Scenario: divergence_gate_diverged_aborts

- **WHEN** a layer's balanced scales reconstruct a binary weight whose Frobenius norm exceeds the fp weight's Frobenius norm by more than the band factor
- **THEN** quantization aborts with the layer's module path, the measured ratio, and the configured band in the error message, raised before Straight-Through refinement and before any later block's calibration inputs are captured, and no qpk1 bytes are exported for the model

#### Scenario: divergence_gate_nan_rejected

- **WHEN** any scale entry produced by balancing is NaN or infinite
- **THEN** the gate rejects the layer with the same abort contract as a magnitude violation

## MODIFIED Requirements

### Requirement: Global calibration SHALL produce robust diagonal preconditioners from activation and gradient statistics

Phase 1 SHALL run the calibration data through the fp teacher once, collecting per-linear-layer input and output activation statistics plus gradient statistics, and SHALL construct diagonal preconditioners `D~_in` and `D~_out` per layer by clipping diagonal entries to a ceiling tau and applying shrinkage `(1−γ)·entry + γ·mean(D)` (Eq. 3, [NQ] Step 2-1). When the configuration pins an explicit tau it SHALL be used verbatim as that absolute cap; otherwise the pipeline SHALL derive the cap per vector as the 95th percentile of that vector's raw diagonal entries, so the ceiling is a finite robust bound that engages on heavy-tailed statistics (a per-vector cap at `max(D)` is a no-op and SHALL NOT be the default behavior). Entries SHALL be nonnegative and finite for any finite calibration data.

#### Scenario: preconditioner_shape_robustness

- **WHEN** calibration statistics are collected for a linear layer with weight shape (d_out, d_in)
- **THEN** `D~_in` has d_in diagonal entries, `D~_out` has d_out entries, all entries are clipped to tau before shrinkage and lie in [0, tau] after, and zero-variance activations yield finite positive entries rather than NaN or zero

#### Scenario: stats_cover_all_layers

- **WHEN** calibration batches flow through a multi-block model
- **THEN** every linear layer slated for quantization has statistics recorded after Phase 1, in module declaration order

#### Scenario: preconditioner_tau_default_q95

- **WHEN** the configuration leaves tau unset and a raw statistic vector is heavy-tailed (maximum more than 10x its 95th percentile)
- **THEN** the derived ceiling equals that vector's 95th percentile, entries above it are clipped before shrinkage, and the resulting preconditioner entries are strictly bounded below the raw maximum

#### Scenario: preconditioner_tau_tail_gate_inert

- **WHEN** the configuration leaves tau unset and a raw statistic vector is moderately tailed (maximum at most 10x its 95th percentile)
- **THEN** no clipping is applied, and the preconditioner equals the no-clip shrinkage path exactly

#### Scenario: preconditioner_tau_explicit_override

- **WHEN** the configuration pins an explicit numeric tau
- **THEN** that value is the cap exactly as before, independent of the per-vector statistics
