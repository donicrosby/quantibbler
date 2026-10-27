# qk-flash-attention — Spec Delta

## ADDED Requirements

### Requirement: Tiled exact-attention NumPy reference (declared-dtype verification fixture)

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
- **THEN** outputs agree with the direct formula under a two-part gate:
  infinity-norm relative error ≤ 1e-12 over the whole row, AND
  per-element relative error ≤ 1e-12 restricted to entries within 1e-2
  of their row maximum (per-element relative error is undefined near
  softmax cancellation — an exactly-correct entry of magnitude 1e-4
  under a row norm O(1) reads as ~1e-11 relative through no fault of
  the implementation; the restricted clause keeps per-element
  stringency where it is meaningful, and the inf-norm clause covers
  the whole row), including ragged
  lengths where seq is not a multiple of the block size.

#### Scenario: Causal mask semantics

- **WHEN** the reference runs with causal masking on,
- **THEN** tokens at positions j > i contribute exactly zero to output row i
  (verified against a −inf-masked dense formula), for every fixture shape.

#### Scenario: Combined causal and additive mask

- **WHEN** the reference runs with causal masking on AND an additive mask
  that suppresses allowed (lower-triangle) positions,
- **THEN** both constraints apply together — the additive mask takes
  effect on top of causal blocking, never silently disabling either —
  verified against the dense formula carrying the same combined
  −inf masking.

#### Scenario: Fully-masked row yields the dense NaN pattern

- **WHEN** a fixture masks every score in one row (row max −inf, ℓ = 0),
- **THEN** that row's output is NaN in exactly the positions the dense
  `0/0` formula produces them (position-for-position `isnan` match),
  with no invented fallback value — asserted by a dedicated
  NaN-pattern check, NOT routed through the finite-output roundoff
  gate.

### Requirement: CubeCL tiled attention GPU face (exact, f32)

The kernels crate (`gpu` feature) SHALL provide an f32 GPU attention face on
the adopted `cubek-attention` substrate (registry evidence and contract-gap
adapters recorded in design.md), with causal masking enabled and no N×N
score-matrix materialization in global memory. The independent cubek
`cpu-reference` implementation SHALL be a third face in the parity matrix.

#### Scenario: GPU parity within the reassociation budget

- **WHEN** hardware tests run on a real device,
- **THEN** the GPU face matches the NumPy reference (widened from f32) with
  relative error ≤ 1e-5 on the full fixture battery, including ragged
  tails and the masked positions.

### Requirement: Fail-closed construction of the GPU face

Construction of the GPU face SHALL fail closed: with no usable HIP device it
returns a typed error carrying the validated staging hint, never a silent
wrong-compute path.

#### Scenario: Fail-closed construction without hardware

- **WHEN** the face is constructed on a machine with no usable HIP device,
- **THEN** construction returns a typed error embedding the staging recipe
  (never a silent wrong-compute path), and the CPU reference face remains
  the only active implementation.

### Requirement: Double-gated hardware tests

Hardware tests for the face SHALL be double-gated: they execute only when
`QUANTIBBLER_GPU_TESTS=1` is set AND the `gpu` feature is enabled.

#### Scenario: Both gate states are verified

- **WHEN** the hardware suite runs with the gate unset, and separately with
  `QUANTIBBLER_GPU_TESTS=1` set,
- **THEN** the unset state produces clean skips and the set state executes
  the parity battery.
