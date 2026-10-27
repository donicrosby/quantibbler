## REMOVED Requirements

### Requirement: The pipeline SHALL compute in float64 by default with float32 as documented opt-in only

**Reason**: Superseded by bf16-default ruling. Replaced by ADDED requirement "The pipeline SHALL compute in bf16 by default with float32 and float64 as documented opt-in only."

## ADDED Requirements

### Requirement: The pipeline SHALL compute in bf16 by default with float32 and float64 as documented opt-in only

All solver and accumulation paths SHALL compute in bf16 unless the caller sets `compute_dtype` to `"float32"` or `"float64"`. No stage SHALL silently upcast from bf16; widening to accumulation dtype is an explicit per-site declaration.

#### Scenario: default_numerics_bf16

- **WHEN** the pipeline runs on any layer with a default `PTQConfig`
- **THEN** intermediate solver arrays are bf16, accumulation operations widen to float32 at named sites, and the artifact records `compute_dtype: bf16`

#### Scenario: float32_opt_in

- **WHEN** the caller sets `compute_dtype = "float32"`
- **THEN** solver entry points operate in float32 and accumulation equals float32

#### Scenario: float64_opt_in

- **WHEN** the caller sets `compute_dtype = "float64"` with `--solver-backend numpy`
- **THEN** the pipeline is bit-identical to the pre-change float64 path on the same inputs, and every existing f64 NumPy-backend fixture passes unchanged
- **WHEN** the caller sets `compute_dtype = "float64"` with the torch backend
- **THEN** the reconstruction cosine vs the float64 NumPy reference satisfies ≥1−1e-12 (cosine drop 1−cos ≤1e-12) — the declared float64 cell of `admm_parity_per_dtype`

## MODIFIED Requirements

### Requirement: Each linear layer SHALL be initialized via LB-ADMM with Hessian-aware coupling and stabilized Cholesky solves

For each linear layer, the pipeline SHALL form the preconditioned target at `compute_dtype` and SHALL solve the ridge-regularized ADMM problem of Eq. 4 ([NQ] Step 2-2). Normal equations and Cholesky SHALL execute at the accumulation dtype; latents, scales, and bias SHALL be stored at `compute_dtype`. The solver SHALL execute on the module's device with no host round-trip.

#### Scenario: admm_spd_solves

- **WHEN** LB-ADMM runs on a random preconditioned target at any rank r and penalty endpoints, at any declared compute_dtype
- **THEN** every system matrix `VᵀV + (ρ+λ)I` (and its transpose-side counterpart) is symmetric positive definite, the solve is performed by Cholesky at the accumulation dtype, and the stabilized fallback engages only when positive definiteness is numerically violated

#### Scenario: admm_dtype_generic

- **WHEN** LB-ADMM completes at bf16 compute
- **THEN** the returned latents U, V, scales s1, s2, and bias are bf16 tensors on the module's device

#### Scenario: admm_on_device

- **WHEN** LB-ADMM factorizes a layer on an accelerator device
- **THEN** the solve executes entirely on-device with no `.cpu().numpy()` round-trip, using `torch.linalg.cholesky` and `torch.linalg.solve_triangular` on device tensors

#### Scenario: admm_parity_per_dtype

- **WHEN** LB-ADMM factorizes a fixed target at each compute_dtype against the float64 NumPy reference
- **THEN** the reconstruction cosine similarity `cos(recon, target)` computed over the block-output dimensions satisfies: ≥0.99 for bf16 (cos ≤ 1e-2 drop), ≥0.9999 for float32 (cos ≤ 1e-4 drop), and ≥1−1e-12 for float64 — measured on the torch backend's output before packing (unpacked reconstruction vs float64-NumPy reference output)

### Requirement: Factorized components SHALL be refined per block with the Straight-Through Estimator

After initialization, each block's continuous latents and scales SHALL be jointly refined with the STE against Eq. 10 ([NQ] Step 3) for T_post steps at `compute_dtype`. The STE passthrough SHALL operate at `compute_dtype`. Binaries SHALL be extracted as `sign(U)`, `sign(V)` at `compute_dtype` and packed.

#### Scenario: ste_refinement_improves_mse

- **WHEN** factorized-component refinement runs on a fixed toy block with straight-through sign gradients at any supported compute_dtype
- **THEN** gradients reach U, V, s1, and s2 with the sign function treated as passthrough, and the block output MSE decreases over the T_post steps

#### Scenario: sign_extraction_packing_parity

- **WHEN** refinement converges and binaries are extracted as `sign(U)`, `sign(V)` and packed via the qk-kernels pack path (mapping −1→0, +1→1 per [NQ] Fig. 2 stage c)
- **THEN** unpacking the packed bits reproduces the extracted sign matrices exactly, and the same bits are produced by the NumPy reference and Rust dispatch paths
- **AND** the sign convention at zero is **+1** (`sign(0) = +1`, implemented as `x >= 0` not `torch.sign`) — exact-zero latents (common under bf16 underflow) map to bit-0, preserving the encoding

#### Scenario: sign_zero_plus_one_convention

- **WHEN** a latent tensor under bf16 compute contains exact zeros (from underflow or initialization)
- **THEN** the packed bits encode those positions as +1 (bit-0), and the unpacked sign matrix is identical to the input under the +1-at-zero convention

#### Scenario: ste_dtype_generic

- **WHEN** STE refinement runs at bf16 compute on a production layer
- **THEN** the STE passthrough produces non-zero latent gradients at bf16, the per-linear gradient norm is recorded relative to the float64 reference, and a deviation exceeding 1e-3 relative triggers a `ste_bf16_gradient_deviation` event

## ADDED Requirements

### Requirement: Every pipeline stage SHALL be parameterized by the declared compute_dtype with no hardcoded numeric type

Every tensor creation, cast, and conversion in the PTQ pipeline SHALL use `compute_dtype` — never hardcoded `torch.float64`, `torch.float32`, or `np.float64`. The codebase SHALL be grep-evidenced with zero unjustified hits.

#### Scenario: grep_evidenced_no_hardcoded_dtype

- **WHEN** the implementation is complete
- **THEN** `grep -rnE '\.double\(\)|torch\.float32|torch\.float64|np\.float32|np\.float64|astype\(.*float' src/quantibbler/ptq/ tools/quantize_hf_model.py` returns zero hits that are not justified by one of three named resolution classes: **(a) parametric** — the site uses `compute_dtype` or `accum_dtype` from config; **(b) accumulation-site** — a named accumulation dtype with arithmetic justification in a comment; **(c) parity-oracle** — a deliberate float64 literal in the NumPy parity path (`solver_backend.py`, unit tests, reference fixtures) carrying an explicit `# parity-oracle: float64 by design` comment

#### Scenario: every_hardcoded_site_enumerated

- **WHEN** the pre-change grep runs against the full pattern set
- **THEN** every existing hardcoded-dtype hit is enumerated and each has a planned resolution class (a/b/c above) in the design.md crystallization section

### Requirement: Where bf16 mantissa width is insufficient, each accumulation site SHALL declare an explicit accumulation dtype

Every operation summing more values than bf16's 7-bit mantissa can accommodate SHALL declare a named accumulation dtype with arithmetic justification. The mapping SHALL be a pure function of compute_dtype: float32 under bf16 compute, identity at float32 and float64.

#### Scenario: accumulation_dtype_is_named_per_site

- **WHEN** any bf16 run executes
- **THEN** each accumulation site (normal equations, Cholesky, SVD, running statistics, Frobenius norms) names its accumulation dtype with a comment citing the term-count bound

#### Scenario: accumulation_identity_at_higher_dtypes

- **WHEN** compute_dtype is float32
- **THEN** the accumulation dtype equals compute_dtype
- **WHEN** compute_dtype is float64 with `--solver-backend numpy`
- **THEN** the accumulation dtype equals compute_dtype, and the pipeline is bit-identical to the pre-change float64 path on the same inputs
- **WHEN** compute_dtype is float64 with the torch backend
- **THEN** the accumulation dtype equals compute_dtype, and the reconstruction cosine vs the float64 NumPy reference satisfies ≥1−1e-12 (cosine drop 1−cos ≤1e-12)

### Requirement: The LB-ADMM and SVD solvers SHALL execute on the accelerator device when present

The torch solver backend SHALL execute on the module's device. On an accelerator, the solve SHALL use torch device tensors throughout with no host round-trip. The NumPy backend remains the float64 parity oracle, unit-test-only.

#### Scenario: solver_on_device

- **WHEN** a model's parameters are on an accelerator
- **THEN** the ADMM solve executes on that device, and the two per-layer host↔device crossings of the v6 pipeline are eliminated

#### Scenario: solver_cpu_explicit_only

- **WHEN** `--device cpu` is set
- **THEN** the torch solver backend executes on CPU tensors — no silent fallback from a failed accelerator probe

#### Scenario: numpy_backend_stays_parity_oracle

- **WHEN** `--solver-backend numpy` is set
- **THEN** the float64 reference path executes in a unit test only; the fire path never resolves to NumPy

### Requirement: The pipeline SHALL place every stage on the declared device without host round-trips

Every stage SHALL derive its device from the model's parameters and SHALL keep all tensors on that device: capture, TuneFP, solver, STE, TuneScalesKD, and eval. The host-side stages are sign/extract/pack (latent-to-bit conversion boundary) and qpk1 export (disk write).

#### Scenario: per_stage_device_from_module

- **WHEN** any pipeline stage executes
- **THEN** the device is read from the owning module's parameters — not from a global or environment variable

#### Scenario: device_auditor_catches_mismatch

- **WHEN** `--device-audit` is set and a module's `__call__` receives tensors on a different device
- **THEN** a named error identifies the module, expected device, and actual device

#### Scenario: per_stage_timing_recorded

- **WHEN** any pipeline fire completes
- **THEN** the report records elapsed-wall seconds per stage (capture, TuneFP, solver, STE, sign/extract/pack, TuneScalesKD, eval, export) as a per-layer JSONL row, appended as each stage produces its timing — enabling comparison against the ~8h v6-class baseline

### Requirement: The CPU path SHALL size thread pools to available cores with load-aware backoff

At startup, the pipeline SHALL read CPU affinity and 1-minute loadavg, compute a thread count, set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` before any BLAS import, and apply the same count to torch's own pool via `torch.set_num_threads` after import (torch does not read `TORCH_NUM_THREADS`). A `--max-threads N` cap and `--no-good-guest` opt-out SHALL exist.

#### Scenario: thread_sizing_light_load

- **WHEN** the 1-minute loadavg is below half the available cores
- **THEN** the thread count is `max(2, n_cores - 2)`

#### Scenario: thread_sizing_heavy_load

- **WHEN** the 1-minute loadavg exceeds the available core count
- **THEN** the thread count is `max(1, n_cores // 4)`

#### Scenario: thread_sizing_moderate_load

- **WHEN** the 1-minute loadavg is at or above half the available cores but does not exceed the core count
- **THEN** the thread count is `max(2, n_cores - 4)`

#### Scenario: thread_sizing_small_affinity

- **WHEN** the affinity set contains fewer than 4 cores (e.g. a container with `--cpus 2`)
- **THEN** the light-load formula produces `max(2, n_cores - 2) = 2` and the heavy-load formula produces `max(1, n_cores // 4) = 1` — a computed value of 1 is acceptable when only 2 cores are visible

#### Scenario: max_threads_cap

- **WHEN** `--max-threads 8` is set
- **THEN** the thread count is `min(computed, 8)`

#### Scenario: no_good_guest_disables

- **WHEN** `--no-good-guest` is set
- **THEN** no thread-pool environment variables are set, and the GPU memory cap is skipped

### Requirement: GPU memory allocation SHALL be capped at a default fraction with a user override

When an accelerator is present, the pipeline SHALL call `torch.cuda.set_per_process_memory_fraction(0.85)` at startup. A `--gpu-memory-fraction F` flag SHALL override the cap.

#### Scenario: gpu_memory_cap_applied

- **WHEN** the pipeline starts on a GPU with default settings
- **THEN** at most 85% of GPU memory is available to this process

#### Scenario: gpu_memory_fraction_override

- **WHEN** `--gpu-memory-fraction 0.5` is set
- **THEN** the cap is 0.5

### Requirement: The artifact remainder dtype SHALL default to the pipeline compute dtype without silent upcast

The `remainder_dtype` for export SHALL default to the mapping of compute_dtype (`bf16`→`"bf16"`, `float32`→`"fp32"`, `float64`→`"fp32"`). The `--remainder-dtype` flag SHALL override. The export path SHALL NOT silently upcast to fp64.

#### Scenario: remainder_dtype_defaults_to_compute_dtype

- **WHEN** a bf16 compute run exports without `--remainder-dtype`
- **THEN** remainder sidecars are written in bf16 and the manifest declares `dtype = "bf16"`

#### Scenario: remainder_dtype_override_independent

- **WHEN** a bf16 compute run exports with `--remainder-dtype fp32`
- **THEN** remainder sidecars are fp32, compute and storage dtypes are separately declared in the manifest

#### Scenario: no_fp64_intermediate_in_export

- **WHEN** the export path writes remainder sidecars
- **THEN** no `.to(torch.float64)` or `.astype(np.float64)` intermediate is applied

### Requirement: FactorizedLinear SHALL store all parameters at the declared compute_dtype

`FactorizedLinear.__init__` SHALL store latents, scales, and bias as `nn.Parameter` at `compute_dtype`. The STE passthrough SHALL be algebraically identical at any float dtype.

#### Scenario: latents_at_compute_dtype

- **WHEN** a `FactorizedLinear` is constructed with `compute_dtype=torch.bfloat16`
- **THEN** all five parameter tensors (u_latent, v_latent, s1, s2, bias) have `dtype == torch.bfloat16`

#### Scenario: from_weight_casts_to_compute_dtype

- **WHEN** `FactorizedLinear.from_weight` is called with a bf16 weight tensor
- **THEN** the returned module's latents and scales are bf16 — the SVD init path casts to compute_dtype before storing

### Requirement: The bias path SHALL carry bias tensors at the declared compute_dtype through every stage

Bias tensors SHALL be stored at `compute_dtype`, carried through solver and STE at `compute_dtype`, and exported at the declared remainder dtype. No stage SHALL zero or drop the bias.

#### Scenario: bias_preserved_through_pipeline

- **WHEN** a quantized layer has a non-zero fp32 bias from the teacher
- **THEN** the exported bias is non-zero at the declared remainder dtype — never zeroed by a silent dtype conversion or skipped write-back

#### Scenario: bias_zero_assertion_gate

- **WHEN** the e2e gate quantizes a tiny model with non-zero biases
- **THEN** the exported artifact's bias vectors are all non-zero, and a zero-bias artifact fails the gate even if PPL passes

#### Scenario: bias_dtype_tracks_compute_dtype

- **WHEN** a FactorizedLinear is constructed with bf16 compute_dtype
- **THEN** `bias.dtype == torch.bfloat16`
