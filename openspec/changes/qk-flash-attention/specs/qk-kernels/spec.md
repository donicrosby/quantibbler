# qk-flash-attention — Spec Delta

## ADDED Requirements

### Requirement: Tiled exact-attention NumPy reference (executable specification)

The kernels package SHALL provide a NumPy tiled attention reference
implementing [FA] §3.1 Algorithm 1 (blockwise online softmax with running
statistics m and ℓ and the exp(m_old − m_new) rescaling identity), in
float64, with causal masking and GQA group broadcast matching the pinned
engine conventions (qk-server spec: post-RoPE q/k, post-norm v,
n_heads/n_kv_heads grouping).

#### Scenario: Reference matches the unblocked formula

- **WHEN** the tiled reference runs the fixture battery (seq ∈ {8, 10, 64},
  d ∈ {4, 128}, head counts (2,1) and (8,2), optional mask) against the
  direct formula softmax(QKᵀ/√d + mask)·V computed in float64,
- **THEN** every output agrees with relative error ≤ 1e-12, including ragged
  lengths where seq is not a multiple of the block size.

#### Scenario: Causal mask semantics

- **WHEN** the reference runs with causal masking on,
- **THEN** tokens at positions j > i contribute exactly zero to output row i
  (verified against a −inf-masked dense formula), for every fixture shape.

### Requirement: CubeCL tiled attention GPU face (exact, f32)

`quantibbler-kernels` (opt-in `gpu` feature) SHALL provide a CubeCL kernel
computing the same tiled algorithm in f32: one workgroup per Q block, K/V
blocks iterated with running (m, ℓ) statistics, no N×N materialization in
global memory. Construction SHALL follow the landed fail-closed boundary
pattern (typed error with the validated staging hint), and hardware tests
MUST be double-gated (QUANTIBBLER_GPU_TESTS=1 + feature) per the qk-kernels
hardware-gate requirement.

#### Scenario: GPU parity within the reassociation budget

- **WHEN** hardware tests run on a real device (both gate states verified:
  unset = clean skips, set = executed),
- **THEN** the GPU face matches the NumPy reference (widened from f32) with
  relative error ≤ 1e-5 on the full fixture battery, including ragged
  tails and the masked positions.

#### Scenario: Fail-closed construction without hardware

- **WHEN** the face is constructed on a machine with no usable HIP device,
- **THEN** construction returns a typed error embedding the staging recipe
  (never a silent wrong-compute path), and the CPU reference face remains
  the only active implementation.
