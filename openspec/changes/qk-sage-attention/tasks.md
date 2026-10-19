# qk-sage-attention — Tasks

Dependency: qk-flash-attention lands first (tiled scaffold + parity
harness are the foundation). Tests precede implementation per capability.

## 1. Quantized reference (Python face)

- [ ] 1.1 RED: `tests/test_attention_sage.py` for scenario IDs
      `gamma-shift-invariance` and `quantized-budget` (fixtures + a
      captured-activation fixture hook); observe RED, record it.
- [ ] 1.2 GREEN: `src/quantibbler/kernels/attention_sage.py` ([SA] §4
      scheme; γ per K-block; exposed quantizers); suite green.

## 2. INT8 kernel path (kernels crate, `gpu` feature)

- [ ] 2.1 Probe (cheap, pre-code): compile-run a CubeCL INT8 dot form on
      gfx1151 (i8 tensor vs u32-packed fallback); record the working idiom
      as crystallization. Command: throwaway bench in /opt/data/tmp
      (gemm-bench pattern), NOT repo code.
- [ ] 2.2 RED: `tests/gpu_attention_sage.rs` hardware-gated; observe RED.
- [ ] 2.3 GREEN: INT8 tile path in gpu_attention.rs; both gate states
      green on gfx1151; clippy/fmt green.

## 3. Server face + quality gate

- [ ] 3.1 RED: `tests/sage_dispatch.rs` — `gate-pass-enables`,
      `gate-fail-refuses` (no-hardware runnable),
      `soft-fallback-device-only`; observe RED.
- [ ] 3.2 GREEN: `gpu:sage` face, startup gate on the shipped record,
      refusal typing, fallback chain; default-feature suite green.
- [ ] 3.3 Gate tool: committed script regenerating the record from a model
      dir + activation fixture ([SA] §4.3 metrics verbatim);
      `gate-record-regenerates` scenario green.

## 4. Probes resolving the Open Questions (measurement, never argument)

- [ ] 4.1 INT8 vs fp32 dot microbench at attention shapes on gfx1151 —
      resolves OQ-1 WITH numbers into design.md crystallization.
- [ ] 4.2 Quality gate run on captured r137 activations + quantized-PPL
      delta on the eval set — resolves OQ-2; face ships as
      opt-in-recommended only if both gates pass, else recorded as
      opt-in-experimental with the failing numbers.

## 5. Landing

- [ ] 5.1 Full gate suite (pytest; cargo test kernels+server in both
      feature states; clippy -D warnings; fmt --check; maturin build;
      `openspec validate qk-sage-attention --strict`) and scoped commit
      (spec + code + crystallization together).
