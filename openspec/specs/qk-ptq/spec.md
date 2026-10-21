# qk-ptq Specification

## Purpose

Define the offline Python post-training quantization pipeline that transforms caller-supplied transformer weights and calibration data into validated qpk1-ready factorized layers without network access.

## Requirements

### Requirement: The PTQ pipeline SHALL expose a single entry point that consumes a caller-supplied model and calibration data

`quantibbler.ptq.quantize_model(model, calibration_data, config)` SHALL accept a caller-constructed PyTorch `nn.Module` (the fp teacher), an iterable of caller-supplied calibration batches, and a `PTQConfig`, and SHALL return a `QuantizedModel` carrying packed binaries and float scales for the quantized layers. The pipeline SHALL perform no network access: the model and calibration data are caller-supplied, per the no-network rule of `openspec/config.yaml`.

#### Scenario: caller_supplied_model_and_data

- **WHEN** a caller constructs a tiny transformer `nn.Module` locally and passes it with an iterable of synthetic token-id batches and a default `PTQConfig` to `quantize_model`
- **THEN** the call returns a `QuantizedModel` without any network access, and no download, fetch, or hub-resolution code path exists in the `quantibbler.ptq` package

#### Scenario: config_defaults_paper_values

- **WHEN** a `PTQConfig` is constructed with no arguments
- **THEN** it exposes rank, robust-diagonal (tau, gamma), ADMM (K, penalty schedule, lam, epsilon), stage step counts with learning rates and batch sizes, seed, and temperature fields, with defaults drawn from [NQ] App. C and the informed assumptions in the qk-ptq design

### Requirement: The pipeline SHALL compute in float64 by default with float32 as documented opt-in only

All solver and accumulation paths (activation statistics, preconditioner construction, ADMM linear solves, magnitude balancing, scale calibration) SHALL compute in float64 unless the caller explicitly sets `compute_dtype = "float32"` in `PTQConfig`. float32 SHALL never be a silent default.

#### Scenario: default_numerics_float64

- **WHEN** the pipeline runs on any layer with a default `PTQConfig`
- **THEN** every intermediate solver array is float64 and the resulting scales are float64

#### Scenario: float32_opt_in

- **WHEN** the caller sets `compute_dtype = "float32"`
- **THEN** solver entry points operate in float32, and the change of behavior is observable only through that explicit config setting

### Requirement: Rust kernel dispatch SHALL match the NumPy reference on float64 probes

When the `quantibbler-kernels` Rust extension is present, its float64 output on probe inputs SHALL match the NumPy reference path — the executable specification — within 1e-12 relative tolerance, with packed bits identical (sign-agreement invariant), per the verification gates of `openspec/config.yaml` and the parity-gated dispatch of the qk-kernels capability.

#### Scenario: rust_matches_reference_probes

- **WHEN** a kernel probe (pack, unpack, or a float64 solve helper) executes with the Rust extension installed
- **THEN** the Rust output matches the NumPy reference within 1e-12 relative tolerance in float64, and packed bits are identical between the two paths

### Requirement: Global calibration SHALL produce robust diagonal preconditioners from activation and gradient statistics

Phase 1 SHALL run the calibration data through the fp teacher once, collecting per-linear-layer input and output activation statistics plus gradient statistics, and SHALL construct diagonal preconditioners `D~_in` and `D~_out` per layer by clipping diagonal entries to a ceiling tau and applying shrinkage `(1−γ)·entry + γ·mean(D)` (Eq. 3, [NQ] Step 2-1). Entries SHALL be nonnegative and finite for any finite calibration data.

#### Scenario: preconditioner_shape_robustness

- **WHEN** calibration statistics are collected for a linear layer with weight shape (d_out, d_in)
- **THEN** `D~_in` has d_in diagonal entries, `D~_out` has d_out entries, all entries are clipped to tau before shrinkage and lie in [0, tau] after, and zero-variance activations yield finite positive entries rather than NaN or zero

#### Scenario: stats_cover_all_layers

- **WHEN** calibration batches flow through a multi-block model
- **THEN** every linear layer slated for quantization has statistics recorded after Phase 1, in module declaration order

### Requirement: Each linear layer SHALL be initialized via LB-ADMM with Hessian-aware coupling and stabilized Cholesky solves

For each linear layer, the pipeline SHALL form the preconditioned target `W~ = D~_out @ W @ D~_in` and SHALL solve the ridge-regularized ADMM problem of Eq. 4 ([NQ] Step 2-2) for continuous factors U, V with structured proxies Z and scaled duals Λ.

#### Scenario: admm_spd_solves

- **WHEN** LB-ADMM runs on a random preconditioned target at any rank r and penalty endpoints
- **THEN** every system matrix `VᵀV + (ρ+λ)I` (and its transpose-side counterpart) is symmetric positive definite, the solve is performed by Cholesky, and the stabilized fallback engages only when positive definiteness is numerically violated

### Requirement: LB-ADMM SHALL iterate continuous updates, SVID proxies, and scaled duals to a stated stopping rule

The continuous update SHALL solve `(VᵀV + (ρ+λ)I)Uᵀ = VᵀW~ᵀ + ρ(Z_U − Λ_U)ᵀ` (Eq. 5) via stabilized Cholesky decomposition, symmetrically for V; system matrices SHALL be symmetric positive definite for ρ > 0, λ ≥ 0 ([NQ] App. B.4, Lemma 2). The proxy update SHALL apply the sign-structure-preserving SVID operator to the consensus variables (Eq. 6), and the dual update SHALL accumulate `Λ ← Λ + U − Z` ([NQ] App. B.3).

#### Scenario: admm_consensus_convergence

- **WHEN** LB-ADMM factorizes a matrix with an exact rank-r structure
- **THEN** the consensus residual `max(||U−Z_U||_F, ||V−Z_V||_F)` decreases across iterations and terminates below epsilon or at the K-iteration cap, with the SVID proxy preserving the sign structure of its input at every proxy update

### Requirement: The ADMM penalty SHALL follow a linear schedule over K iterations with epsilon as an early stop

The penalty parameter SHALL follow a linear schedule across K outer iterations ([NQ] App. C, App. D.4), and ADMM SHALL stop early when the consensus residual falls below the tolerance epsilon; otherwise it SHALL run to the K-iteration cap.

#### Scenario: penalty_schedule_stopping

- **WHEN** LB-ADMM runs with default ADMM parameters on a fixed target
- **THEN** the penalty value increases linearly per iteration up to K, and the solver halts at whichever comes first: residual below epsilon or the K-th iteration

### Requirement: LB-ADMM SHALL conclude with latent magnitude balancing that yields Ub, Vb, s1, s2

Upon ADMM convergence, the pipeline SHALL recover unscaled proxies `Û = D~_out⁻¹ P_U`, `V̂ = D~_in⁻¹ P_V`, compute the equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7), extract scales `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8), and set final latents `U = η Û`, `V = η⁻¹ V̂` (Eq. 9) ([NQ] Step 2-3). Initialization of the layer is complete only when all of (Ub, Vb, s1, s2) are produced.

#### Scenario: magnitude_balancing_scales

- **WHEN** ADMM converges and magnitude balancing executes on the recovered proxies
- **THEN** the balanced factor norms satisfy the η relation of Eq. 7, the extracted scales equal per-row mean absolute values per Eq. 8, and no scale entry is NaN; an all-zero row receives the clamped scale floor rather than an undefined value

### Requirement: Block-wise reconstruction SHALL refine factors against quantized-prefix inputs and teacher targets

Phase 2 SHALL process transformer blocks sequentially. For each block it SHALL capture inputs `X_b` by pushing calibration data through the already-quantized prefix and teacher outputs `Y_b` from the fp block ([NQ] Algorithm 1 lines 9–10), and SHALL tune the block's fp weights against (X_b, Y_b) for T_pre steps before factorization ([NQ] Step 1, error propagation mitigation).

#### Scenario: quantized_prefix_inputs

- **WHEN** block b > 0 is reconstructed in a multi-block model
- **THEN** its captured inputs `X_b` are the outputs of the quantized blocks before b, not of the fp prefix, and `Y_b` is the fp block's output on those inputs

### Requirement: Factorized components SHALL be refined per block with the Straight-Through Estimator

After initialization, each block's continuous latents and scales SHALL be jointly refined with the Straight-Through Estimator against the block-output objective of Eq. 10 ([NQ] Step 3) for T_post steps, using the stage learning rate and batch size recorded in [NQ] App. C unless overridden by config. Upon convergence, binaries SHALL be extracted as `sign(U)`, `sign(V)` and packed.

#### Scenario: ste_refinement_improves_mse

- **WHEN** factorized-component refinement runs on a fixed toy block with straight-through sign gradients
- **THEN** gradients reach U, V, s1, and s2 with the sign function treated as passthrough, and the block output MSE decreases over the T_post steps

#### Scenario: sign_extraction_packing_parity

- **WHEN** refinement converges and binaries are extracted as `sign(U)`, `sign(V)` and packed via the qk-kernels pack path (mapping −1→0, +1→1 per [NQ] Fig. 2 stage c)
- **THEN** unpacking the packed bits reproduces the extracted sign matrices exactly, and the same bits are produced by the NumPy reference and Rust dispatch paths

### Requirement: Global scale calibration SHALL tune only the scale vectors under a KL objective with binaries frozen

Phase 3 SHALL optimize the per-layer scale vectors `{s1, s2}` of all quantized layers to minimize the KL divergence between teacher and quantized logit distributions under temperature T (Eq. 11, [NQ] Sec 3.3), with teacher logits cached and the packed binary weights bit-frozen throughout. No other parameter SHALL receive gradients during this phase.

#### Scenario: kl_scale_calibration_frozen_bits

- **WHEN** global scale calibration runs on a tiny quantized model with cached teacher logits
- **THEN** the KL objective decreases over T_glob steps, the packed bits are bit-identical before and after the phase, and only scale parameters carry gradients

### Requirement: The pipeline SHALL round-trip through the qpk1 container with byte-exact fidelity

A `QuantizedModel` SHALL serialize to the qpk1 container defined by the qk-representation capability and SHALL reload from it such that packed binaries are byte-exact, float scales are preserved bit-for-bit in float64, and a forward pass after reload matches the forward pass before serialization on the same probe inputs. Byte accounting for the artifact SHALL come from the serialized container, per the qk-representation capability.

#### Scenario: qpk1_round_trip_byte_exact

- **WHEN** a quantized tiny model is serialized to qpk1 and reloaded into a fresh object
- **THEN** the packed bits compare byte-exact, the scales compare bit-for-bit in float64, and probe-input forwards before and after the round trip agree to float64 round-off

### Requirement: The end-to-end tiny-model gate SHALL bound logit delta and require effective BPW below 1.0 measured from bytes

The verification gate SHALL build a seeded tiny random transformer block, run quantize → serialize qpk1 → reload → forward on probe inputs, and assert that the maximum absolute logit delta against the fp tiny model is bounded by the threshold recorded in the gate test. This gate mirrors the end-to-end smoke gate of `openspec/config.yaml` and runs as part of the full gate suite.

#### Scenario: e2e_tiny_gate_passes

- **WHEN** the seeded tiny random transformer block is quantized, serialized, reloaded, and forwarded on the gate's probe inputs
- **THEN** the max logit delta versus the fp model is within the recorded bound

### Requirement: The gate SHALL assert byte-derived BPW below 1.0 and round-trip bit parity

The gate SHALL compute effective bits-per-weight as serialized bytes from the qpk1 container via the qk-representation accounting — never from parameter counts — and SHALL require it strictly below 1.0; it SHALL also assert pack/unpack parity (bit identity) across the round trip.

#### Scenario: e2e_bpw_and_parity

- **WHEN** the gate measures the serialized qpk1 artifact of the tiny model
- **THEN** effective BPW computed from serialized bytes is below 1.0 and packed bits are identical across the serialize → reload round trip

#### Scenario: e2e_gate_detects_corruption

- **WHEN** a single packed bit in the serialized qpk1 artifact is deliberately flipped and the gate reruns
- **THEN** the gate fails on the logit-delta or parity assertion, demonstrating the gate can detect serialization corruption

### Requirement: factor-mode fp removes the binary constraint

The PTQ pipeline SHALL support a `factor_mode` of `binary` (default) or `fp`.
In `fp` mode the factorized layers compute with continuous latents — no sign,
no straight-through estimator, no bit-flip polish — while capture, TuneFP,
solver, balance, scale initialization, latent refinement, the progress probe,
and the report run unchanged.

#### Scenario: fp_mode_forward_has_no_sign_anywhere

- **WHEN** a run carries `factor_mode=fp`
- **THEN** the factorized forward computes `diag(s1) @ U @ V^T @ diag(s2)` with
  no `sign` operation in any of the three latent-term sites, in either the
  grad-enabled or the `no_grad` regime, and all three sites produce their terms
  through one shared helper

#### Scenario: helper_bitwise_contract_in_binary_mode

- **WHEN** a binary run computes latent terms through the shared helper
- **THEN** the grad-enabled branch returns the straight-through passthrough
  `sign(u) + u - u.detach()` (so latent gradients are nonzero), the `no_grad`
  branch returns `torch.sign` bitwise (`torch.equal` holds on the f64 fixture
  tier), the regime is selected by an explicit `torch.is_grad_enabled()` test,
  and a helper returning literal sign in the grad-enabled branch is rejected
  by the battery

#### Scenario: helper_reachability_is_observed_twice

- **WHEN** the helper pins run
- **THEN** spies observe BOTH the live STE path and `_set_linear` consulting the
  helper, AND the weight `_set_linear` writes is `torch.equal` to the helper's
  `no_grad` output — a helper satisfied by only one site does not pass

#### Scenario: runtime_gradient_contract_catches_dead_ste

- **WHEN** the binary grad-enabled helper is exercised by a real backward pass
- **THEN** its latent gradient equals the analytic `d/dU = s1` contract, and a
  detached-alias implementation that returns the right values but zeroes that
  gradient fails even if the structural AST pin accepts its shape

#### Scenario: default_binary_behavior_is_byte_identical

- **WHEN** `factor_mode` is unset
- **THEN** sign, STE, polish, and packing run exactly as in the pre-knob
  pipeline

#### Scenario: fp_mode_skips_only_the_discrete_stage

- **WHEN** a run carries `factor_mode=fp`
- **THEN** the bit-flip polish stage is not invoked, balance-stage latents flow
  unchanged into refinement, and every other stage's recorder events still fire
  in the same order

### Requirement: untuned-signs ablation exports sign of the balance latents

The pipeline SHALL support an `ablation` mode `untuned-signs` for binary runs:
the exported bits are `sign()` of the balance-stage latents with zero
refinement — no polish rounds and no TuneLatentSTE steps.

#### Scenario: untuned_signs_runs_zero_refinement

- **WHEN** a binary run carries `ablation=untuned-signs`
- **THEN** the exported bits equal `sign()` of the balance latents exactly, and
  the run log records zero polish invocations and zero STE events

#### Scenario: default_refinement_unchanged

- **WHEN** `ablation` is unset
- **THEN** polish and TuneLatentSTE run with their configured budgets exactly
  as before

### Requirement: layer scope factorizes only the named component

The pipeline SHALL support a `layer_scope` of `all` (default), `self_attn`, or
`mlp`: only linears whose module path contains the scope component are
factorized; every other linear stays full precision in prefix capture,
refinement, and export.

#### Scenario: attention_scope_quantizes_only_attention

- **WHEN** a run carries `layer_scope=self_attn`
- **THEN** only `self_attn` linears are factorized, and every MLP weight is
  byte-identical to its pre-run value

#### Scenario: mlp_scope_quantizes_only_mlp

- **WHEN** a run carries `layer_scope=mlp`
- **THEN** only `mlp` linears are factorized, and every attention weight is
  byte-identical to its pre-run value

#### Scenario: all_scope_is_the_default

- **WHEN** `layer_scope` is unset
- **THEN** the quantizable set is unchanged from the pre-knob pipeline

### Requirement: ablation runs are measurements, never artifacts

A run carrying any non-default knob (`factor_mode=fp`, `ablation=untuned-signs`,
or `layer_scope != all`) SHALL be a measurement: the full gate, probe, and
report run with the knob values recorded in provenance, the verdict derives
from the held-out PPL ratio with the unchanged band, and a `fp`-mode run never
reaches the qpk1 export.

#### Scenario: fp_run_reports_but_never_exports

- **WHEN** a run carries `factor_mode=fp`
- **THEN** `extract_and_pack` — the only site that converts latents to bits —
  raises on fp latents rather than sign-converting them, the run completes with
  host write-back via the helper's fp branch, the report is written with a
  verdict, provenance records the knobs, `artifact_exported` is false,
  `effective_bpw` is null, scale calibration is skipped as the ablation
  boundary, and the driver's export refusal remains as a backstop

#### Scenario: binary_ablation_runs_may_export

- **WHEN** a binary run carries `ablation=untuned-signs` or a scoped
  `layer_scope`
- **THEN** the qpk1 export path is unchanged and provenance records the knobs

#### Scenario: verdict_band_is_mode_independent

- **WHEN** any knob is set
- **THEN** status is derived from the held-out PPL ratio against the same
  `OK_LINE` / `COLLAPSE_LINE` band as default runs

### Requirement: chunked statistics stream feature moments

When `collect_stats` receives `chunk_rows`, its forward hooks SHALL retain only
float64 per-feature sum-of-squares and token counts; retained state SHALL be
independent of the number of rows/chunks. It SHALL NOT retain or concatenate
full activation tensors. Gradient accumulation remains unchanged. The
`chunk_rows=None` float64 reference path SHALL remain byte-identical.

#### Scenario: streamed_stats_match_reference

- **WHEN** the same f64 fixture is collected once unchunked and once through
  the streamed chunk path
- **THEN** every input, output, and gradient-derived statistic matches at
  `rtol=1e-12, atol=1e-12`

#### Scenario: streamed_state_is_bounded

- **WHEN** the streamed path processes arbitrarily many row chunks
- **THEN** each layer side retains one feature-width f64 sum vector and one
  token count, never a list of full activation tensors; a poison that restores
  tensor appends is rejected by the structural pin

#### Scenario: default_reference_path_is_unchanged

- **WHEN** `chunk_rows` is unset
- **THEN** the existing list/concatenate reference computation and its
  byte-identical fixture behavior remain unchanged
