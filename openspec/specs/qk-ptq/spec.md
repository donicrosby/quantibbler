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

### Requirement: Rust kernel dispatch SHALL match declared-dtype NumPy parity fixtures on probe inputs

When the `quantibbler-kernels` Rust extension is present, its output on probe inputs at a declared dtype SHALL match same-declared-dtype NumPy parity fixtures within the per-dtype tolerances declared by the qk-kernels capability (bf16 backward 1e-2 and f32 backward 1e-6 of the row term scale; float64 rel 1e-12), with packed bits identical (sign-agreement invariant), per the verification gates of `openspec/config.yaml`.

#### Scenario: rust_matches_parity_fixtures

- **WHEN** a kernel probe (pack, unpack, or a solve helper) executes with the Rust extension installed at any declared dtype
- **THEN** the Rust output matches the same-dtype NumPy parity fixture within the declared per-dtype tolerance, and packed bits are identical between the two paths

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

The continuous update SHALL solve `(VᵀV + (ρ+λ)I)Uᵀ = VᵀW~ᵀ + ρ(Z_U − Λ_U)ᵀ` (Eq. 5) via stabilized Cholesky decomposition, symmetrically for V ([NQ] App. B.4, Lemma 2). The proxy update SHALL apply the two-sided optimal rank-1 SVID operator to the consensus variables (Eq. 6): `Z = sign(P) ⊙ (g hᵀ)` where (g, h) is the top rank-1 SVD of `sign(P)∘P` ([NQ] :592, "optimal rank-1 approximation that preserves the sign structure"). The dual update SHALL accumulate `Λ ← Λ + U − Z` ([NQ] App. B.3).

#### Scenario: admm_consensus_convergence

- **WHEN** LB-ADMM factorizes a matrix with an exact rank-r structure
- **THEN** the consensus residual `max(||U−Z_U||_F, ||V−Z_V||_F)` decreases across iterations and terminates below epsilon or at the K-iteration cap
- **AND** the SVID proxy preserves the sign structure of its input at every proxy update

#### Scenario: svid_is_two_sided_optimum

- **WHEN** the SVID proxy processes a consensus variable `P = U + Λ` with sign structure `S = sign(P)`
- **THEN** the returned Z is `S ⊙ (g hᵀ)` where (g, h) are the top rank-1 SVD factors of `S∘P`
- **AND** for a column-constant input the two-sided optimum is exactly the column-only solution — the fix is a strict superset, never a regression on the existing test fixtures

#### Scenario: svid_sign_preserved

- **WHEN** SVID processes any input matrix
- **THEN** `sign(Z) == sign(input)` entrywise, with sign(0) = +1 per the A-11 convention
- **AND** dead columns (all-zero input) receive the positive `DEFAULT_SCALE_FLOOR`

### Requirement: The ADMM penalty SHALL follow a linear schedule over K iterations with epsilon as an early stop

The penalty parameter SHALL follow a linear schedule across K outer iterations ([NQ] App. C :3008), and ADMM SHALL stop early when the consensus residual falls below the tolerance epsilon; otherwise it SHALL run to the K-iteration cap.

#### Scenario: penalty_schedule_stopping

- **WHEN** LB-ADMM runs with default ADMM parameters on a fixed target
- **THEN** the penalty value increases linearly per iteration up to K, and the solver halts at whichever comes first: residual below epsilon or the K-th iteration
### Requirement: The rho endpoint SHALL lie in the stated interval in the normalized regime

The rho endpoint SHALL stay within the stated endpoint interval so the consensus term does not overwhelm the unit-scale data term: in the normalized regime (target divided by its Frobenius norm at entry per [NQ] Step 2), `rho_end` SHALL lie in the closed interval `[0.0, 1.0]`, with the schema refusing `rho_end < rho_start`. The paper prescribes the linear shape; the endpoints are an implementation choice, and an endpoint that prevents convergence is a defect.

#### Scenario: rho_end_bounded_for_convergence

- **WHEN** LB-ADMM factorizes a planted rank-32 target at rank-64 factorization (`gamma=0.2`; warm-start cos ≥0.99 — the fixture the convergence gate exercises) with the paper's `tau`/`gamma` preconditioning; real-layer evidence: all 20 recorded `layers.0` probe rows in `artifacts/lane0-fire-r3-20261024/rows.jsonl` are non-monotone (2–66 rises each; one row — arm B `self_attn.o_proj` — does not net-decrease over the first 200), confirming the shape is structural, not a fixture artifact
- **THEN** the consensus residual makes net progress across the first 200 iterations (net decrease from start of the second half, with the first half dominated by the consensus-term pull onto the ±1 proxy before the data term reasserts itself) and the solver terminates at the K-iteration cap — `converged=False` is expected with a ±1-structured proxy and does not indicate solver failure
- **AND** the post-iteration continuous reconstruction satisfies `cos(UVᵀ, W) > 0.5`, proving the ADMM produces a usable initializer (cos loss from the unconstrained truncated-SVD warm start is expected; the SVID proxy pushes latents into the ±1-signed rank-1 subspace, and the paper's pipeline compensates in STE refinement per [NQ] Step 3)
- **AND** `cos(UVᵀ, W) > cos(sign(zu) @ sign(zv)ᵀ, W)` — the continuous reconstruction beats a naive sign-only assignment, proving the ADMM iteration does meaningful work beyond trivial binarization
### Requirement: LB-ADMM SHALL conclude with latent magnitude balancing that yields Ub, Vb, s1, s2

Upon ADMM convergence, the pipeline SHALL recover unscaled proxies `Û = D~_out⁻¹ P_U`, `V̂ = D~_in⁻¹ P_V`, compute the equilibrium factor `η = sqrt(||V̂||_F / ||Û||_F)` (Eq. 7), extract scales `[s1]_i = mean(|η û_i|)`, `[s2]_j = mean(|η⁻¹ v̂_j|)` (Eq. 8), and set final latents `U = η Û`, `V = η⁻¹ V̂` (Eq. 9) ([NQ] Step 2-3). Initialization of the layer is complete only when all of (Ub, Vb, s1, s2) are produced.

#### Scenario: magnitude_balancing_scales

- **WHEN** ADMM converges and magnitude balancing executes on the recovered proxies
- **THEN** the balanced factor norms satisfy the η relation of Eq. 7, the extracted scales equal per-row mean absolute values per Eq. 8, and no scale entry is NaN; an all-zero row receives the clamped scale floor rather than an undefined value
### Requirement: The ADMM exit SHALL rescale to preserve scale equivariance

The ADMM exit SHALL rescale factors from the normalized space back to the target's scale with `factor_scale = sqrt(target_scale)` applied to each of `(U, V, P_U, P_V)` — a single factor of `target_scale` on the product `UVᵀ`, maintaining scale equivariance: scaling the entry target by `c` SHALL scale the returned product by `c`, not `c²`.

#### Scenario: exit_rescale_is_equivariant

- **WHEN** the ADMM entry target is scaled by a nonzero factor c
- **THEN** the returned product `UVᵀ` scales by exactly c — not c² — verified within relative tolerance 1e-5
- **AND** the iteration trajectory in the normalized space is identical regardless of the entry scale
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
`chunk_rows=None` path SHALL remain byte-identical to its committed fixture records at the declared dtype.

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

### Requirement: The pipeline SHALL offer a per-block progress probe against which prefix perplexity can be gated

`PTQConfig` SHALL expose a `progress_probe` callable field, default `None`, invoked by the sequential block reconstruction after each block is written back into the host model, with the live model, the 1-based prefix depth, and the block name; the probe may raise to abort the pipeline, and an unset probe SHALL change nothing.

#### Scenario: probe_invoked_each_block

- **WHEN** a `progress_probe` is configured and a multi-block model is quantized
- **THEN** the probe observes the live host model once per block in construction order, with depth counting quantized blocks 1-based, after that block's write-back and before the next block's capture begins

#### Scenario: probe_absent_no_change

- **WHEN** the config leaves `progress_probe` unset
- **THEN** the pipeline path is byte-identical to a no-probe run (no invocation overhead in control flow, no checkpoint fingerprint change from a callable being absent), and existing configs and tests keep working untouched

#### Scenario: probe_abort_propagates

- **WHEN** the probe raises its abort exception
- **THEN** the pipeline halts immediately with the block and depth named, no later block is captured or reconstructed, and the error surfaces from `quantize_model` unchanged

### Requirement: The driver probe SHALL fail the run at the first prefix whose ratio crosses the paper-derived collapse line

The driver-side probe SHALL precompute fp-prefix perplexities at every depth on the head slice of the held-out eval, measure each quantized prefix's perplexity on the same slice, compare through the `qk_report` verdict band, and SHALL abort the run when a prefix ratio exceeds `COLLAPSE_LINE` at any probed depth. The probe is kill-only: no prefix measurement SHALL certify `ok` or `degraded`, which come exclusively from the end-of-run full held-out gate.

#### Scenario: exploded_prefix_aborts_fast

- **WHEN** a reconstructed block pushes the quantized prefix's slice ratio above `COLLAPSE_LINE` (min 40.0, the [LB2]-derived line) at depth 2 or later
- **THEN** the run aborts at that block boundary — before any later block's solver work — naming the block, depth, and both prefix perplexities

#### Scenario: healthy_prefix_passes

- **WHEN** the prefix ratio at every probed depth stays at or below `COLLAPSE_LINE`
- **THEN** no probe abort fires, the pipeline completes normally, and the end-of-run verdict alone classifies the artifact

#### Scenario: probe_abort_leaves_report

- **WHEN** the probe aborts at a block boundary
- **THEN** the driver still writes `quantize_report.json` carrying `status: failed`, the aborting block, the depth, both prefix perplexities, and `artifact_exported: false`, and exits nonzero — a probe abort never leaves the run without durable evidence, mirroring the e2e gate's forensics contract

#### Scenario: report_single_shape_both_paths

- **WHEN** any run completes, aborted or not
- **THEN** the report key set is identical on both paths: every field the completion report carries is present on an abort report (unmeasured end-of-run numerics written as `null`), and `artifact_exported` appears on both (`false` on abort, `true` on completion) — strict consumers never KeyError on either shape

#### Scenario: abort_report_construction_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the abort-report construction — building the report dict from a `PrefixProbeAbort`'s fields plus history, routed through `dump_report` with the failed status — is covered by tests that execute without skips, so the catch→report→status logic has repeatable proof independent of weights and CI

#### Scenario: probe_history_recorded_on_completion

- **WHEN** a run completes without a probe abort
- **THEN** `quantize_report.json` carries the per-depth probe history — `(depth, prefix_ppl, fp_prefix_ppl, ratio)` for every probed block — written through the same `dump_report` path as every other report value, so the first surviving run becomes the probe's own calibration data instead of an unauditable near-miss

#### Scenario: probe_comparator_reuses_gate_band

- **WHEN** the probe's decision logic runs
- **THEN** it is the same torch-free `qk_report.verdict` comparator as the end-of-run gate — the missing/non-finite check runs first, and `COLLAPSE_LINE` is the shared derived constant, not a second copy

#### Scenario: probe_slice_same_eval

- **WHEN** the driver builds the probe slice
- **THEN** the slice is the head of the same held-out evaluation batches, in the same order, with the same seed discipline as the end-of-run measurement — never a separately generated dataset

#### Scenario: fp_prefix_baselines_same_depth

- **WHEN** the driver precomputes prefix baselines
- **THEN** each depth's baseline is the fp model's perplexity over the same slice at the same prefix depth, so the probe ratio isolates quantization damage and not depth effects

### Requirement: The probe SHALL be configurable in interval, bounded in cost, and fixed in threshold

The driver SHALL expose `--probe-interval` (positive integer blocks, default 1) and `--probe-batches` (positive integer eval batches, default 2); the abort threshold SHALL be the derived `COLLAPSE_LINE` with no per-run override, and probes SHALL begin no earlier than depth 2.

#### Scenario: probe_knob_defaults_and_validation

- **WHEN** the driver starts with default arguments
- **THEN** the probe measures every block boundary from depth 2 on a 2-batch slice, and a non-positive `--probe-interval` or `--probe-batches` is rejected at argument parsing

#### Scenario: probe_threshold_not_overrideable

- **WHEN** any run executes
- **THEN** no flag or config value re-pins the probe's abort line above or below the shared `COLLAPSE_LINE` constant

#### Scenario: probe_math_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the probe decision logic (slice-ratio verdict, boundary semantics, NaN first) is covered by tests that execute without skips, while model-eval glue tests live in the torch tier per the repo's tier conftest pattern

### Requirement: The real-model driver SHALL report an end-to-end perplexity ratio against the same run's bf16 baseline

The real-model driver SHALL compute `ppl_ratio = ppl_quant_native / ppl_baseline_bf16` over one held-out eval partition through one logits path (the host object quantize_model wrote the artifact back into), and SHALL record both perplexities, the ratio, and the byte-derived effective BPW in `quantize_report.json`.

#### Scenario: ratio_single_measurement_path

- **WHEN** the driver measures quantized perplexity after `quantize_model` writes the artifact back into the host
- **THEN** the reported ratio uses exactly that measurement — no re-import or second measurement path — and `ppl_baseline_bf16`, `ppl_quant_native`, and `ppl_ratio` all appear in `quantize_report.json`

#### Scenario: ratio_history_all_refused

- **WHEN** the verdict logic replays the recorded production shapes (ratios 1,156,577.1; 943.49; 5,517.8; 463,742.5; 524,894.8; and NaN)
- **THEN** every shape classifies `failed` — no historical artifact ever receives `ok` or `degraded`

### Requirement: Every driver run SHALL receive a three-value end-to-end quality verdict from a paper-calibrated ratio band

The driver SHALL classify each run `ok`, `degraded`, or `failed` from the end-to-end ratio against two derived band edges, expose the classification in the report `status` field, and recur the same classification into the process exit code.

#### Scenario: verdict_ok_within_functional_envelope

- **WHEN** a run's ratio is at or below `OK_LINE` = `ceil(23.74/6.10)` = 4.0 — the functional envelope at the extreme budget end of [LB2] Sec. 5.2 Table 1 (LittleBit-2 at 0.1 bpp, PPL 23.74, against the FP16 6.10)
- **THEN** the verdict is `ok` and the process exits 0

#### Scenario: verdict_degraded_between_lines

- **WHEN** a run's ratio exceeds `OK_LINE` and is at or below `COLLAPSE_LINE` = `ceil(241.96/6.10)` = 40.0 — the documented method collapse of [LB2] Sec. 5.2 Table 1 (STBLLM at 0.55 bpp, PPL 241.96, against the FP16 6.10)
- **THEN** the verdict is `degraded`, the process exits 0, and the report carries the degraded status rather than a silent pass

#### Scenario: verdict_failed_above_collapse_line

- **WHEN** a run's ratio exceeds `COLLAPSE_LINE`, or the ratio is missing or non-finite
- **THEN** the verdict is `failed` and the process exits nonzero; the missing/non-finite check always precedes any threshold comparison

#### Scenario: verdict_edge_arithmetic_pinned

- **WHEN** the suite runs
- **THEN** a test asserts both edges' derivations exactly — `ceil(241.96/6.10) == 40` and `ceil(23.74/6.10) == 4` — and asserts the boundary semantics (`ratio == 40.0` → `degraded`; `ratio == 40.0001` → `failed`) plus the trap that `nan > 40.0` is False so a threshold comparison alone can never catch NaN

#### Scenario: verdict_helpers_torch_free_tested

- **WHEN** the test suite runs in a lean environment without torch
- **THEN** the verdict and sanitizer tests execute (no skips, no collection errors) because the logic lives outside the torch-importing driver, in the compressor package

### Requirement: quantize_report.json SHALL be provenance-complete and strict-JSON-parseable

The report SHALL name the actual run — top-level `solver` and `scale_dtype` from the argument values, `status` from the verdict — and SHALL emit `regime.artifact` and `config_f64.dtype_note` from those same arguments rather than hardcoded literals, and SHALL write every non-finite float as `null` so the report parses under a NaN-rejecting strict parser.

#### Scenario: report_mirrors_run_args

- **WHEN** a run executes with `--solver svd --scale-dtype float32`
- **THEN** the report records `solver: svd` and `scale_dtype: float32` top-level, and `regime.artifact` names float32 scales — the float16 literal never appears for a float32 run

#### Scenario: report_fp16_opt_in_named

- **WHEN** a run executes with the `--scale-dtype float16` opt-in
- **THEN** `regime.artifact` and `config_f64.dtype_note` name float16 scales, recorded from the arguments rather than from a constant

#### Scenario: report_nonfinite_as_null

- **WHEN** any report value is NaN or infinite — including a numpy scalar float such as `np.float32(nan)` — or writing the full report fails for any reason
- **THEN** non-finite values are written as `null` with `status: failed` and the file parses under a `parse_constant`-rejecting strict JSON parser; if serializing or writing the report itself raises, the driver SHALL still write a minimal `{"status": "failed", "report_error": ...}` report, exit nonzero, and never leave the run without a durable status; a healthy report parses with `status: ok`

### Requirement: The driver process SHALL fail closed on a failed verdict while always exporting the artifact

`main()` SHALL return 0 only for `ok` or `degraded` verdicts and a nonzero exit code for `failed`, and SHALL write the qpk1 export and `quantize_report.json` regardless of verdict, so a failed run still leaves a forensically complete artifact.

#### Scenario: failed_run_exits_nonzero

- **WHEN** a run's verdict is `failed` (including the missing-or-non-finite ratio case)
- **THEN** the process exit code is nonzero, distinguishable from a pass by any CI or orchestrator chain

#### Scenario: healthy_run_exits_zero

- **WHEN** a run's verdict is `ok` or `degraded`
- **THEN** the process exits 0

#### Scenario: export_and_report_persist_on_failure

- **WHEN** the gate fails the run
- **THEN** the qpk1 model directory and `quantize_report.json` are still written, with the report carrying `status: failed` as the durable trace

### Requirement: Checkpoint resume SHALL fail closed on empty or incomplete block archives

Resume SHALL never advance a block with fewer reconstructed layers than the
block is owed. Guards SHALL be driven from the live model's expected
quantizable names, not from the archive's contents.

#### Scenario: empty_archive_is_never_written

- **WHEN** a block completes with zero reconstructed layers
- **THEN** `save_block_checkpoint` raises and no `block-*.npz` file is
  written for that block

#### Scenario: empty_archive_is_never_counted_done

- **WHEN** `completed_blocks` scans a checkpoint directory
- **THEN** archives with zero keys (or no readable quantized-layer names) are
  excluded from the done set

#### Scenario: restore_asserts_expected_equals_present

- **WHEN** a resumed run applies checkpointed layers to a block
- **THEN** the set of layer names reconstructed from the archive is asserted
  equal to the block's expected quantizable names from the live model, and
  any symmetric difference raises an error naming the block and the missing
  names before the prefix advances

### Requirement: Lane 0 SHALL be a measurement-only decisive experiment that lands nothing in the artifact path

The lane SHALL be a decisive preconditioner-divergence experiment built on the existing `lb_admm` + `magnitude_balance` solver pair and the repo's ablation/probe conventions, adding measurement code only. It SHALL NOT modify any shipped pipeline module's algorithm, SHALL NOT reach the qpk1 export path, and SHALL NOT emit any OK/DEGRADED/FAILED verdict band. Any non-default measurement knob SHALL be recorded in provenance.

#### Scenario: measurement_only_no_export

- **WHEN** the lane-0 probe completes on its pinned probe set
- **THEN** no code path on the lane reaches `extract_and_pack`, qpk1 serialization, or any export site, the report records `artifact_exported: false` with no effective-BPW field, and no verdict band appears: "No verdict bands, no export."

#### Scenario: shipped_pipeline_modules_untouched

- **WHEN** the lane executes against the repo
- **THEN** `calibration.py`, `lb_admm.py`, `block_recon.py`, and `config.py` carry no algorithm changes (empty diff on their solver logic), and all new code lives in a new tool module plus its test file
- **AND** progress instrumentation imports in `lb_admm.py` and `block_recon.py` (the `ProgressReporter` belt code from `qk-progress-instrumentation`) are the sole allowed diff — they are additive instrumentation that changes no solver behavior, and their presence is expected state, not a drift

### Requirement: The decisive probe SHALL run a fixed set of real production layers with full identity recorded

The probe set SHALL be a fixed constant pinned before any measurement: real production layers, one entry per (model, block index, layer name). Every report row SHALL record those three identity fields. Both arms SHALL run the identical set in the identical order; per-run ad-hoc set edits SHALL be impossible by construction.

#### Scenario: fixed_probe_set_identity_recorded

- **WHEN** the lane-0 report is written
- **THEN** every probed-layer row records `model`, `block_idx`, and `layer_name` populated from the pinned fixed set, and the set itself is a named constant in the harness with no CLI override

#### Scenario: both_arms_identical_probe_set

- **WHEN** the two arms execute
- **THEN** arm A and arm B consume the same pinned probe set in the same order, so every comparison is within-layer

### Requirement: The intended preconditioner SHALL be constructed the paper-intended way

The intended arm SHALL build `W~ = D~_out W D~_in` ([NQ] B.1 Eq. 26) with diagonal entries folded once per side as square roots of robust second-moment statistics (sqrt fold), the gradient side from a real cross-entropy backward through the head ([NQ] App. C :3011), per-token norm clipping at the robust 99.9th percentile, and the paper's clip-and-shrinkage robustness ([NQ] Eq. 3) applied on top.

#### Scenario: sqrt_fold_once_per_side

- **WHEN** the intended preconditioner is constructed for a probed layer
- **THEN** `W~_intended = D~_out W D~_in` holds exactly, each preconditioner diagonal is applied once per side, the diagonal entries equal the square root of the robust clipped-and-shrunk second-moment statistics, and the magnitude-balance unfold uses the same diagonals

#### Scenario: gradient_side_real_ce_backward

- **WHEN** the intended arm collects gradient statistics
- **THEN** they come from a backward of the cross-entropy loss of the fp teacher's logits against the calibration targets through the head, never a uniform-ones cotangent, and any layer with no reachable gradient records the fallback it used

#### Scenario: per_token_norm_clip_999

- **WHEN** calibration batches contain outlier tokens
- **THEN** per-token vector norms are clipped at the robust 99.9th percentile of the batch's per-token norms before statistics accumulate, and the applied percentile is recorded in the report

### Requirement: Both arms SHALL run the existing lb_admm and magnitude_balance unchanged

Each arm SHALL feed its `W~` variant into the shipped `lb_admm(w_tilde, cfg)` and `magnitude_balance(p_u, p_v, d_out_tilde, d_in_tilde, target=W)` entry points under a production-default `PTQConfig`. The only difference between arms SHALL be the preconditioner construction. Neither arm SHALL invoke polish, STE refinement, or any altered solver parameter.

#### Scenario: same_solver_pair_both_arms

- **WHEN** the two arms run a probed layer
- **THEN** both call the shipped solver entry points with the same production-default `PTQConfig`, and the only per-arm variation is `(D~_in, D~_out, W~)`

#### Scenario: pre_polish_stage_boundary

- **WHEN** a probed layer's measurements are taken
- **THEN** `greedy_polish` and TuneLatentSTE are never invoked on the lane, so every quantity reflects the post-ADMM pre-polish state

### Requirement: Per probed layer each arm SHALL record the four decisive quantities

For every probed layer and both arms the report SHALL record: the post-ADMM pre-polish reconstruction cosine against the original fp weight; the full consensus residual trajectory; the ADMM iteration count `n_iter`; and the `min_eigs` floor over the solve's system matrices. A missing or truncated quantity SHALL fail the run.

#### Scenario: four_quantities_per_layer_per_arm

- **WHEN** a probed layer completes both arms
- **THEN** its rows each carry the post-ADMM pre-polish recon cos, the complete consensus residual trajectory, `n_iter`, and the min_eigs floor; any missing quantity fails the run closed instead of writing a partial row

#### Scenario: recon_cos_against_original_weight

- **WHEN** the recon cos is computed
- **THEN** it is `cos(W, U_bal V_bal^T)` against the ORIGINAL un-preconditioned fp weight at the post-balance pre-polish boundary, matching the solver-probe `cos_raw` convention, and the intended arm's band comparison uses that same quantity

### Requirement: The lane SHALL end in a quantize_report-style JSON and a winner-or-falsified verdict paragraph

The decisive run SHALL write a quantize_report-style JSON (strict-parseable, provenance-complete) and a verdict paragraph that names, per probed layer, the arm owning the strictly higher post-ADMM pre-polish recon cos, or declares the hypothesis falsified. The domination rule and the Lanes 1-3 gate are defined in the design. "No verdict bands, no export."

#### Scenario: winner_named_per_probed_layer

- **WHEN** the decisive run completes
- **THEN** the report is a quantize_report-style JSON plus a verdict paragraph naming the winner per probed layer, and a layer where the intended arm does not strictly win is recorded as such rather than omitted

#### Scenario: falsified_hypothesis_gates_lanes

- **WHEN** the intended preconditioner does not dominate (per the design's domination rule)
- **THEN** the verdict paragraph declares the hypothesis falsified and the report carries the lane gate for Lanes 1-3: STOP and re-analyze before landing anything, with Lane 0 gating Lanes 1-3 because lane order is load-bearing

### Requirement: The CE head SHALL be chunked and statistic-exact

The CE-gradient pass SHALL chunk the head per 256-token slice off a full-context trunk forward under `torch.utils.checkpoint`, sum the chunk losses, apply the mean gauge to the total, and take ONE backward, bitwise identical to the whole-batch path. Per-slice `empty_cache` SHALL bound the allocator.

#### Scenario: chunked_ce_preserves_statistic

- **WHEN** the equivalence fixture runs the legacy whole-batch path and the checkpointed-head path on identical weights and inputs (tiny geometry, one sequence, 128 tokens, CPU)
- **THEN** the accumulated gradient-square vectors match to machine precision — identity, not tolerance
- **AND** the fixture goes red against the naive trunk-rechunk variant AND against per-chunk backward (attention couples rows, so squared partials do not sum)

### Requirement: The trunk forward SHALL gradient-checkpoint every production block

The trunk forward SHALL checkpoint every production block — backward recomputes per block, so the retained activation graph is bounded to one block, the statistic is unchanged bitwise, and the ~30% stats-phase slowdown is accepted. The fixture SHALL run under the attention implementation the fenced rehearsal will use.

#### Scenario: trunk_checkpointing_preserves_statistic

- **WHEN** the equivalence fixture runs the retained-graph path against the checkpointed-trunk path on identical weights and inputs (tiny geometry, CPU, the container's attention implementation)
- **THEN** the gradient-square vectors match bitwise — identity, not tolerance
- **AND** the fixture goes red against any build where a production block is left uncheckpointed

### Requirement: The stats accumulator SHALL count each row exactly once

The accumulation hooks SHALL gate on `torch.is_grad_enabled()` so each row is counted exactly once on the recompute pass, and the trunk checkpoint SHALL be reentrant mode — the discriminator's premise. The equivalence fixture SHALL assert RAW-sum identity against the uncheckpointed path, not just the mean.

#### Scenario: raw_sums_are_the_correct_version

- **WHEN** the equivalence fixture runs the checkpointed-trunk path against the uncheckpointed path
- **THEN** `d_in_raw` and `d_out_raw` match bitwise — raw, not mean — each row counted exactly once, so the raw sums are the naive-magnitude values and no future reader can be surprised by a ×2

#### Scenario: doubling_trap_is_a_red_case

- **WHEN** the fixture exercises an ungated accumulator or a non-reentrant checkpoint (both passes grad-enabled)
- **THEN** the raw sums double relative to the uncheckpointed path and the fixture goes red, so the trap fails loudly instead of surviving as folklore

#### Scenario: whole_head_checkpointed_trunk_fenced

- **WHEN** the harness is asked to run the whole-batch head with checkpointed blocks — the combination whose `d_out_raw` moved ~0.8% while raw sums and rows stayed identical, executed by no shipped path
- **THEN** it fails closed before any statistics pass, the error names the finding and its commit, and the configuration is not launchable until the divergence is root-caused

### Requirement: Harness changes touching a production import SHALL pass the surface-table gate

Any harness change that touches a production import SHALL be accompanied by the production-surface diff table — one row per touchpoint (collect_stats, robust_diag, lb_admm, magnitude_balance), each citing the production call-site line and reading `matches` or `DEFECT`. No fire SHALL launch while any row reads DEFECT.

#### Scenario: surface_table_gate_before_fire

- **WHEN** a Lane 0 fire is staged
- **THEN** the harness change's surface table exists with every production touchpoint row citing its call-site line, all rows read `matches`, and any DEFECT row blocks the launch — instance six gets caught by a grep, not a floor-switch

### Requirement: Checkpoint resume SHALL bind the model identity

The resume fingerprint SHALL bind the identity of the model being quantized: the
quantized-parameter scope's module inventory, each module's `(dout, din)` shape, and
its weight dtype, all derived from the LIVE model at fingerprint-write time. A resume
SHALL refuse a diverged identity, naming the diverged component(s), and SHALL refuse a
directory with no recorded identity. A repository id or path SHALL NOT be the identity.

#### Scenario: model_identity_binds_the_resume

- **WHEN** a checkpoint directory written while quantizing model A is resumed by a run
  quantizing model B under an identical `PTQConfig`
- **THEN** `check_fingerprint` raises before any block archive is applied, and no
  block of B is restored from A's archives

#### Scenario: model_identity_mismatch_names_the_divergence

- **WHEN** the live model's quantized-parameter scope differs from the recorded
  identity in module membership, in a per-module `(dout, din)` shape, or in a weight
  dtype
- **THEN** the refusal names the diverged module(s) and what diverged about them, so
  the mismatch is diagnosable from the error alone

#### Scenario: legacy_fingerprint_without_model_identity_is_refused

- **WHEN** a checkpoint directory's `fingerprint.json` carries no model-identity
  component (it was written before this binding existed)
- **THEN** the resume refuses and says the directory predates the model-identity
  component, so its provenance with respect to the live model cannot be verified

#### Scenario: driver_default_checkpoint_dir_is_per_run

- **WHEN** `tools/quantize_hf_model.py` runs without an explicit `--checkpoint-dir`
- **THEN** it checkpoints into a FRESH per-run directory rather than a shared stable
  path, so two defaulted launches never share block archives; resumption is opt-in by
  naming the directory, and the chosen directory is named in the launch's output

### Requirement: Calibration and eval data knobs SHALL count sequences, not packed batches

The `--calibration-seqs` and `--eval-seqs` knobs SHALL denote **sequences**, matching the [NQ] protocol unit (128 calibration sequences, [NQ] Sec 3.4/App. C). The loader SHALL convert sequences to packed batches internally (`n_chunks = N // BATCH`, `BATCH = 8`), fail-closed when N is not divisible by BATCH, and the reported counter SHALL be computed at the same boundary as the chunking — before any tune-window slice that changes the chunk count.

#### Scenario: calibration_knob_counts_sequences

- **WHEN** the driver runs with `--calibration-seqs 128`
- **THEN** the loader produces chunks covering exactly 128 sequences (16 packed batches of 8)
- **AND** the reported `calibration_rows` equals 128, matching the [NQ] protocol, so the report's deviation flag does not fire
- **AND** the counter is computed before the `tune_seq_len != seq_len` slice at the loader/driver boundary (a fix placed after that slice computes the value 4× too large under `TUNE_SEQ_LEN=512`)

#### Scenario: calibration_knob_fails_closed_on_odd_count

- **WHEN** `--calibration-seqs` is not divisible by the packed batch size
- **THEN** the driver exits nonzero with an error naming the knob and the divisibility requirement

#### Scenario: eval_knob_counts_sequences

- **WHEN** the driver derives the maximal local eval window
- **THEN** `--eval-seqs` denotes sequences and the window derivation is expressed in sequences (18 batches of 8 re-expressed as 144 sequences), with the eval gate comparing like units at every comparison

#### Scenario: red_first_unit_test

- **WHEN** the unit tests run against the pre-fix driver
- **THEN** an end-to-end assertion `len(calib) * BATCH == CALIBRATION_SEQS` fails with the current value 1024 (red-first), and passes after the fix with 128

### Requirement: The fail-fast verdict machinery SHALL prove geometry-invariance or re-derive constants per geometry

Every gate constant used by the prefix fail-fast verdict SHALL either be geometry-invariant by construction or be re-derived per geometry. The collapse line, pedestals, and the prefix probe each follow the classification stated in the scenarios below.

#### Scenario: collapse_line_is_paper_anchored

- **WHEN** any run evaluates the depth-2 prefix ratio against the collapse line
- **THEN** the line is exactly the published-anchor derivation asserted in `tests/test_qk_report.py`, with no run-specific calibration and no code movement on geometry change

#### Scenario: pedestal_tracks_geometry

- **WHEN** a run changes calibration or eval geometry
- **THEN** its baseline and fp-prefix pedestals are measured at that run's own geometry and recorded in the run's provenance before any ratio verdict is computed

#### Scenario: prefix_probe_same_depth_same_object

- **WHEN** the prefix probe computes its ratio
- **THEN** the quantized-prefix PPL and the fp-prefix PPL are measured on the same object at the same depth, dividing depth-dependent floor effects out of the ratio

### Requirement: The pipeline SHALL be held to a paper-recreation bar pinned to [NQ] Table 2 rows for Qwen3-0.6B

An acceptance run SHALL demonstrate, on Qwen3-0.6B over [NQ]'s WikiText-2 sequence-length-2048 protocol, a perplexity at or below `1.15 * row` for one of the three NanoQuant rows of [NQ] Sec. 4.2 Table 2 (27.56 @1.00 bits, 33.79 @0.80 bits, 52.94 @0.55 bits, against the 12.66 baseline), with byte-derived `effective_bpw` at or below the targeted row's bit budget.

#### Scenario: bar_constants_asserted

- **WHEN** the suite runs
- **THEN** a test asserts the four row constants and the tolerance arithmetic (`1.15 * row` for each targeted row) are the values pinned in this delta, and that no runtime path recomputes them

#### Scenario: bar_undefined_below_lowest_row

- **WHEN** an artifact's bit budget is below 0.55 bits
- **THEN** no recreation row exists for it; it is beyond-paper research — never recorded under the word recreation, never granted ledger status on interpolation

#### Scenario: bar_tolerance_provisional_until_measured

- **WHEN** artifacts are still far from any row
- **THEN** the 1.15 tolerance stands as the pre-registered provisional, and the first artifact landing within 2x of a targeted row obligates re-deriving the tolerance from that artifact's measured run-to-run eval variance and re-pinning it as a spec change

#### Scenario: bar_never_recalibrates_the_band

- **WHEN** a band-edge recalibration is ever proposed
- **THEN** moving an edge so a paper-reference artifact (including [NQ]'s own rows, e.g. @0.55 ratio 4.18) reads `ok` is prohibited — the band is never tuned to flatter a reference; it recalibrates only from production healthy-extreme evidence

#### Scenario: bar_not_purchased_with_bits

- **WHEN** an artifact claims a bar row
- **THEN** its serialized-byte `effective_bpw` is at or below that row's bit budget — a lower PPL at a higher bit budget meets no row

#### Scenario: bar_requires_paper_protocol

- **WHEN** an acceptance run executes
- **THEN** it evaluates at sequence length 2048 over the WT2 test partition per [NQ] Sec. 4.2; seq-512 numbers alone are recorded alongside, never substituted for the bar

#### Scenario: bar_recorded_not_gated

- **WHEN** an acceptance run completes
- **THEN** a ledger entry records model id, commit hash, targeted row, eval protocol, measured PPL, ratio, effective_bpw, and the operational-gate verdict — the bar is a documented human-gated acceptance criterion, not a per-run gate verdict

#### Scenario: bar_distinct_from_landing_gate

- **WHEN** any artifact is classified
- **THEN** the `ok/degraded/failed` band remains the only per-run verdict; an `ok` artifact may still be below the bar, and no artifact above the bar receives `ok` on bar evidence alone

### Requirement: The PTQ driver SHALL support independent tune and evaluation sequence lengths

The driver's `--seq-len` SHALL remain the evaluation geometry, and a `--tune-seq-len` flag SHALL bound the per-block reconstruction windows independently. When `--tune-seq-len` is unset it SHALL default to `--seq-len`, preserving single-geometry behavior exactly.

#### Scenario: split_mode_preserves_eval_geometry

- **WHEN** the driver runs with `--seq-len 2048 --tune-seq-len 512`
- **THEN** evaluation PPL runs over 2048-token WT2 test sequences while block reconstruction tunes on 512-token windows, and the report records both lengths

#### Scenario: default_single_geometry_unchanged

- **WHEN** `--tune-seq-len` is unset
- **THEN** tuning windows use `--seq-len` exactly as before the flag existed — existing run behavior is unchanged and no silent geometry change occurs

### Requirement: The ADMM solver SHALL default to the torch backend with NumPy unit-test-only

The solver SHALL default to the torch backend, the only backend that produces artifacts, resolved fail-closed once per run with the `QK_RHO_END` provenance idiom. NumPy SHALL execute ONLY inside unit tests, as the parity reference — never in a fire, harness, or integration path — with NO fallback: a torch request without an accelerator SHALL fail rather than resolve to NumPy. `solver_is_admm` SHALL remain an algorithm condition.

#### Scenario: backend_knob_provenance

- **WHEN** a run selects `SOLVER_BACKEND=torch` (or resolves the default)
- **THEN** launch.env, the checkpoint fingerprint, and every solver-probe row record backend and dtype
- **AND** the default resolves to torch — the accelerated backend
- **AND** a torch backend without an available accelerator device fails closed at startup, on the same pattern as `GPU=1` requiring a rocm-tagged image — a torch backend running on CPU is a slowdown wearing an optimization's name

#### Scenario: numpy_is_unit_test_only

- **WHEN** the NumPy backend executes
- **THEN** it executes inside a unit test and nowhere else — no fire, no harness, no integration path

#### Scenario: admm_algorithm_unchanged

- **WHEN** either backend factorizes a matrix
- **THEN** it executes the same continuous-update, SVID-proxy, dual-update, and magnitude-balance steps, and `solver_is_admm` remains true on either backend

### Requirement: Backend parity SHALL be established by unit tests with torch-only production rows

The torch backend SHALL be accepted on its unit-test parity suite — the only place the NumPy reference executes — together with the campaign's measured tolerance record (fp32 GPU ≡ fp64 CPU within ±0.002 on production layers). Production and probe fires are torch-only, and every probe row SHALL carry row-time so the measured win replaces the estimate.

#### Scenario: torch_only_fire_rows

- **WHEN** the Lane 0 fire executes
- **THEN** all 20 solves run on torch, no NumPy process runs, and every row carries row-time alongside recon cos, residual trajectory, n_iter, and min_eigs

#### Scenario: unit_test_parity_is_the_gate

- **WHEN** the backend's parity suite runs as unit tests
- **THEN** tests compare reconstructed products (sign-ambiguity-safe) within the tolerance class, and a failing parity test blocks acceptance

#### Scenario: tolerance_breach_named

- **WHEN** any parity unit test moves outside the tolerance class
- **THEN** the failing test names the operation and its delta, and acceptance fails instead of passing silently

### Requirement: Checkpoint restore SHALL refuse a mis-shaped state before writing any layer

Checkpoint restore SHALL validate each stored state's arrays against the live
linear's weight and bias shapes before writing any layer. A state whose
Ub/Vb/s1/s2/bias cannot factorize to the live `(dout, din)` weight SHALL raise
naming the block, layer, and stored shapes, leaving the block untouched.

#### Scenario: restore_refuses_mis_shaped_state

- **WHEN** a resumed run applies a checkpointed state whose arrays cannot
  produce the live linear's weight `(dout, din)` — Ub not `(dout, r)`, Vb not
  `(din, r)` with a shared r, s1 not `(dout,)`, s2 not `(din,)`, or a stored
  bias not `(dout,)`
- **THEN** `apply_layer_states` raises a RuntimeError naming the block, the
  layer, the stored shapes, and the live weight shape, refusing to write the
  state

#### Scenario: mis_shaped_refusal_leaves_block_untouched

- **WHEN** a right-key/wrong-shape archive is applied to a multi-layer block
- **THEN** the raise fires before the first `copy_`, and every linear's weight
  in the block is bit-identical to its value before the restore
