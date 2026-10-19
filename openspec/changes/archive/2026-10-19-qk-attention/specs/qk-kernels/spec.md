## ADDED Requirements

### Requirement: Packed matvec SHALL dispatch to a CubeCL HIP backend behind an opt-in feature

With the `gpu` cargo feature enabled and a HIP device present, the kernels
dispatch SHALL execute `packed_matvec` via CubeCL. With the feature off, or
no device present, behavior SHALL be identical to the current CPU-only
build. Default builds MUST NOT enable the GPU path (server purity and
determinism gates stay green on GPU-less machines), and no HIP library is
required at build time.

#### Scenario: K-3.1 — GPU parity on packed matvec
- **WHEN** `gpu` is enabled on a machine with a HIP device and
  `packed_matvec` runs on random bitplanes and scales
- **THEN** GPU results match the CPU rlib within rel 1e-12 on the f64
  accumulation, and a repeated run is bit-identical

#### Scenario: K-3.2 — GPU-less machine with `gpu` enabled stays CPU-correct
- **WHEN** the feature is enabled but no HIP device is visible
- **THEN** dispatch falls back to CPU and the suite passes unchanged
