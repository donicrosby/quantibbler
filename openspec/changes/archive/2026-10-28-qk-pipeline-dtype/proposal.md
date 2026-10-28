# Proposal: qk-pipeline-dtype

## Why

The v6b artifact (landed a208a6e) is faithfully packed and served, yet the model
is 14.14x off the FP baseline (PL 285.79 vs 20.21, greedy output degenerate).
Bias-carry has been exonerated (t_59815b89: Qwen3-0.6B is bias-free by config,
zero biases are legit-zero math). The pipeline is proven faithful; the prime
suspect class is dtype mixing acros stages. The curent pipeline hardcodes
float64 in the solver and factorized latents, float32 in artifact remainder
sidecars, bf16 in the model weights — and silently casts between them at
boundaries. The user ruling (2026-10-27) is explicit: one dtype through every
stage including STE, default bf16 end-to-end, no hardcoded types anywhere.

Additionally, the solver (LB-ADMM, SVD) runs on CPU/NumPy while the autograd
stages can use a GPU accelerator — leaving the GPU idle during the most
compute-intensive phase. The ruling demands GPU acceleration for every
compute-heavy stage with CPU as fallback, never default. The ~8h v6-class fire
wall is the baseline to beat.

## What Changes

1. **Dtype-agnostic pipeline.** Every compute path in `src/quantibbler/ptq/` is
   parameterized by `compute_dtype`. Default: `bf16`. Selectable: `float32`,
   `float64`. `PTQConfig.compute_dtype` gains `"bf16"` as the default and
   accepted value. The driver's `--compute-dtype` adds `bf16` as the default
   choice. No hardcoded `torch.float64`, `torch.float32`, or `np.float64` in
   any pipeline stage — latents, scales, bias, gradients, and intermediates
   all hold the declared dtype.

2. **Accumulation dtype separation.** Where bf16's 7-bit mantissa is
   insufficient (dot-product accumulators in ADMM normal equations, Cholesky
   factorization, robust-diagonal accumulation), each such site declares an
   explicit accumulation dtype distinct from storage/compute, justified by the
   required precision width. The default accumulation dtype is `float32` under
   bf16 compute. The separation is a named per-site decision, never emergent.

3. **GPU solver.** The LB-ADMM solver and SVD init run on the accelerator
   device (torch backend), not CPU/NumPy. The solver moves to torch tensors on
   the device — eliminating per-block host round-trips for the ~400-iteration
   ADMM loop. The NumPy backend is the unit-test-only reference path at the declared
   dtype (per existing spec).

4. **Per-stage device placement map.** Every stage declares its device:
   capture (accelerator), TuneFP (accelerator), solver (accelerator),
   TuneLatentSTE (accelerator), sign/extract/pack (host — latent-to-bit
   conversion boundary, `x >= 0` convention), TuneScalesKD
   (accelerator), eval (accelerator). CPU is the explicit `--device cpu`
   fallback. The map identifies where the v6 pipeline left the GPU idle
   (solver on CPU, per-block host round-trips at `.cpu().numpy()` boundaries)
   and how the rewrite eliminates them.

5. **Good-guest resource discipline.** CPU thread pools (`OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) sized to `max(1, ncores - 2)` at
   startup with load-aware backoff (read /proc/loadavg; downsize when 1-min
   loadavg exceeds ncores). GPU memory fraction capped at 0.85 with
   `--gpu-memory-fraction` override. No unbounded worker spawns anywhere. The
   loadavg-47–70 production-measurement incident (the anti-pattern where
   hardcoded `OMP_NUM_THREADS=16` left 0 free cores) is recorded in design.md
   as the never-repeat reference.

6. **Artifact dtype alignment.** The artifact remainder dtype defaults to
   `bf16` — the same dtype as the pipeline compute — not `fp32`. The export
   path emits remainder sidecars in the declared remainder dtype without
   silent fp64 or fp32 upcast. `remainder_dtype` stays a separately settable
   knob (`compute bf16 + store fp32` is legal), but its default tracks the
   compute dtype.

7. **Hardcoded-dtype elimination.** Grep-evidenced: every `float64` literal,
   `to(torch.float64)`, `dtype=torch.float64`, `np.float64`, and `np.dtype`
   call site in the PTQ pipeline and driver is either made
   `compute_dtype`-parametric or justified as an accumulation-dtype site with
   an explicit named declaration. The `loadavg-47-70` anti-pattern and the
   `OMP_NUM_THREADS=16` hardcode are replaced by the good-guest sizing rule.
   The config.yaml method-invariant line changes from "solvers compute in
   float64 by default" to "solvers compute in bf16 by default; float32 and
   float64 are selectable non-defaults."

8. **Bias-path dtype.** The parallel bug-hunt (t_59815b89, completed
   2026-10-27) found the v6b artifact's all-zero biases are legit-zero math —
   Qwen3-0.6B is bias-free by config, not a dtype defect. The bias path is
   nonetheless rewritten to dtype-agnostic form as defense-in-depth: bias
   tensors hold the declared `compute_dtype`, not hardcoded float64
   (eliminating `.double()` hardcodes at `_factorized.py:91,92,95`).

## Impact

- **Invariant-touching: yes — the method numerics.** The default compute dtype
  changes from float64 to bf16. The config.yaml method-invariant line and the
  qk-ptq default-numerics requirement are MODIFIED. The solver backend default
  stays torch; the NumPy backend remains the unit-test-only reference path
  at the declared dtype.
- **Affected specs**: `qk-ptq` (1 REMOVED, 2 MODIFIED, 10 ADDED requirements).
  `qk-representation` and `qk-server` are untouched — the container format
  already declares `remainder_dtype`, and the server already reads it.
- **Affected code (implementation)**: `src/quantibbler/ptq/config.py`,
  `_factorized.py`, `lb_admm.py`, `block_recon.py`, `calibration.py`,
  `bit_polish.py`, `solver_backend.py`, `e2e.py`, `config.yaml`,
  `tools/quantize_hf_model.py`, `tools/run-ptq-container.sh`, and tests.
- **Co-touch notes**:
  - `qk-artifact-dtype` (active, 0/18): remainder_dtype declaration exists; this
    change changes the default from fp32 to bf16 and makes it track
    compute_dtype. **Orchestrator ruling (2026-10-27):** pipeline-dtype OWNS
    the `--remainder-dtype` default; qk-artifact-dtype MUST fold this default
    flip into its delta/tasks or sequence after pipeline-dtype with an explicit
    supersede note. The qk-ptq capability is double-claimed — this ruling
    resolves the default collision (hotspot flagged in review).
  - `qk-kernels-dtype-generic` (active, 19/23): kernel dtype genericity consumed by
    the pipeline side's dtype-parameterized tensor ops. The pipeline does not
    reimplement any kernel; it calls the generic entry points.
  - Parallel bug-hunt t_59815b89: **completed** — bias zeros are legit-zero
    math (bias-free Qwen3-0.6B config, not dtype-related). OQ-1 resolved
    as defense-in-depth. Two peripheral findings routed to t_373bdeff — no
    scope expansion for this change.</

## Paper anchor

No method change: the [NQ] algorithm, stages, budgets, and solver structure are
unchanged. The dtype at which the method executes is a deployment choice —
[NQ] Sec 3.1 pins the shared weight representation (binary factors with
full-precision scales), not the compute precision. The bf16 accumulation
justifications derive from IEEE bf16 format properties (7 explicit mantissa
bits → ~2.3 decimal digits, relative ulp 2^-7) and the observed precision
needs of each accumulation site (ADMM normal equations sum over `(din, r)` and
`(dout, r)` terms — float32 accumulation with bf16 storage is justified for
ranks up to ~4096 before the accumulator's ~23-bit mantissa risks
catastrophic cancellation). These are arithmetic derivations from the format
spec, not corpus claims.

## Non-goals

- No method change: no new stages, no budget changes, no geometry changes.
- No performance work beyond device placement: generic kernel instantiation
  for bf16 matmuls is owned by `qk-kernels-dtype-generic`; this change calls
  whatever the face exports at the declared dtype.
- No GPU kernel authoring: the face (`torch` → ROCm/CUDA) already exists.
- No change to the qpk1 layer-stream format (`ScaleDtype`, bitplane layout).
- No re-fire authorization: producing artifacts under the new defaults is a
  separate GO question for @user.
- No sqrt-fold preconditioner changes, no Lane 0/1/2/3 changes.
- Parity verification uses same-declared-dtype NumPy fixtures: each compute_dtype is verified against fixtures at its own dtype, not against a higher-precision oracle.
