# qk-ptq delta: qk-ptq-speedups

## ADDED Requirements

### Requirement: Eigenvalue diagnostics SHALL be gated behind the probe flag

The per-iteration `min_eigs` eigenvalue computation SHALL run only when the solver-probe diagnostic flag is enabled. `trace.n_jitter` — the always-on record of non-SPD events from the Cholesky solve — SHALL remain the production-path SPD signal, so no safety property is lost by the gating. The gating SHALL NOT alter any solver state: the checkpoint fingerprint and the solver's arithmetic SHALL be bit-identical with the flag off.

#### Scenario: production_run_pays_no_diagnostic_tax

- **WHEN** a production run executes the ADMM loop without the probe flag
- **THEN** no per-iteration eigenvalue computation occurs, and the solver's measured wall time drops by the measured per-iteration diagnostic cost (benchmarked: 2× `eigvalsh(512×512)` ≈ 142 ms/iteration at 16 threads vs 62 ms for the 2× matmul + 2× Cholesky it guards)

#### Scenario: diagnostic_run_keeps_the_column

- **WHEN** a diagnostic run enables the probe flag (the Lane 0 harness does)
- **THEN** every probe row still carries `min_eigs`, satisfying `qk-nanoquant-alignment-lane0`'s harness requirement unchanged

#### Scenario: gating_is_behavior_neutral

- **WHEN** the same target factorizes with the flag on and off
- **THEN** post-ADMM cos, n_iter, residuals, and the checkpoint fingerprint are identical — the gating moves only wall time

#### Scenario: spd_signal_preserved

- **WHEN** a Cholesky solve jitters during a production run
- **THEN** `trace.n_jitter` still records the event, so the SPD proof carried by the solve itself is never left unrecorded

### Requirement: Torch-autograd stages SHALL run on the accelerator device

The STE, TuneFP, and TuneScalesKD stages SHALL execute on the accelerator with no algorithm change — same per-stage loss, optimizer, scheduler, and batch sizes — via activation-as-authority device resolution, never globals. TuneScalesKD's disk-backed teacher cache SHALL remain on the host.

#### Scenario: ste_tunefp_gpu_zero_algorithm_change

- **WHEN** a run selects an accelerator device
- **THEN** the STE and TuneFP stages execute on that device with the identical loss, optimizer, scheduler, and batch configuration as the CPU reference
- **AND** per-stage row-time is reported, translating the measured ~133 s/TuneFP-fire and ~154 s/STE-linear costs

#### Scenario: tunescaleskd_gpu_zero_algorithm_change

- **WHEN** a run selects an accelerator device
- **THEN** the TuneScalesKD stage executes its model forward/backward on that device with the identical KL loss, optimizer, scheduler, and batch configuration as the CPU reference
- **AND** the disk-backed teacher cache and post-stage scale readback stay on the host; only the per-step model forward/backward and paged teacher logits move to the device
- **AND** `ub`/`vb` are built on device at stage scope (hoisted above the per-step loop), not rebuilt from NumPy inside `scale_forward`; the device authority is `linear.weight.device`

#### Scenario: kd_gpu_device_parity

- **WHEN** a TuneScalesKD stage run on the accelerator is compared against the CPU reference
- **THEN** the per-step KL trace decreases monotonically on both legs
- **AND** the final scales are finite and stay within the stage's trust region on both legs
- **AND** the frozen-bits invariant holds (packed Ub/Vb unchanged; already tested by `test_scn_kl_scale_calibration`)

#### Scenario: ste_tunefp_fp32_gpu_within_tolerance_class

- **WHEN** an STE or TuneFP fp32 stage run is compared against the fp64 CPU reference on production layers
- **THEN** cosine movement stays within the campaign's ±0.002 tolerance class; a breach names the layer and fails that stage's acceptance
- **AND** TuneScalesKD (f64 scale parameters, native-dtype activations) is excluded from this class; its device equivalence is asserted through KL trace agreement (scenario `kd_gpu_device_parity`)

#### Scenario: device_authority_not_globals

- **WHEN** tensors move to the accelerator
- **THEN** movement routes through the existing activation-as-authority path (`flat.device`, `linear.weight.device`) — no torch global device state, no `torch.set_default_device`
