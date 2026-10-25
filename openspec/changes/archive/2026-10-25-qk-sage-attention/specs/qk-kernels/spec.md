# qk-sage-attention — Spec Delta (qk-kernels)

## ADDED Requirements

### Requirement: Quantized tiled-attention NumPy reference (executable specification)

The kernels package SHALL provide a NumPy reference implementing the [SA]
§4 scheme on top of the tiled reference — γ smoothing per [SA] §4.2 Eq. 6;
granularity and dtypes per [SA] §4.3 — in float64 emulating the kernel's
f32 accumulator semantics, and MUST expose its quantizer outputs (scales,
smoothed K) so probes can assert each stage against the paper's algebra.

#### Scenario: Shift invariance holds exactly through gamma

- **WHEN** the reference runs any fixture with γ applied per K-block,
- **THEN** the dequantized-then-unsmoothed score path equals the exact
  reference scores to f64 round-off (≤ 1e-12 relative): γ is loss-free by
  [SA] §4.2, so observed deviation isolates INT8 quantization error alone.

#### Scenario: Quantized reference stays inside the paper-class budget

- **WHEN** the reference runs the fixture battery plus captured real
  activations,
- **THEN** outputs match the exact tiled reference with Cosine Sim ≥ 0.999
  and Relative L1 ≤ 0.02 ([SA] §4.3 metrics), and per-token Q/K scale
  factors are strictly positive with no zero-range tokens.

### Requirement: INT8 tiled path in the CubeCL attention kernel (opt-in `gpu` feature)

The CubeCL attention kernel SHALL provide an INT8 tile path implementing
the same scheme (per-token Q/K scales, per-block γ, static P̃ scale,
per-channel V, f32 statistics/accumulator) with INT8 dots on device.
Construction and hardware gating follow the exact face's patterns
(fail-closed typed init; QUANTIBBLER_GPU_TESTS=1 + feature, both states
verified).

#### Scenario: Device parity for the quantized path

- **WHEN** hardware tests run on a real device (both gate states verified),
- **THEN** the INT8 kernel path matches the quantized NumPy reference
  within a block-level f32 budget (rel ≤ 1e-4 vs the reference's own
  quantized arithmetic), and quality metrics vs the EXACT face land inside
  the reference's measured envelope on the same fixtures.
