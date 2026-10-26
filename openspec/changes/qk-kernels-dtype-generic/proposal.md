# Proposal: qk-kernels-dtype-generic

## Why

The kernel crate's float surface is hardcoded to one concrete type per
face: the CPU rlib's FP-remainder products (`packed_matvec`,
`packed_matmul`) name `f64` in their signatures and bodies, and the CubeCL
device face names `f32` everywhere. Nothing can be instantiated at a second
precision without editing an algorithm body, so a serving path that wants
bf16 activations (half the bytes on the wire and in the working set, the
dtype the artifact chain is moving toward) has no kernel to call. The
program directive: kernels written generic per type, not hardcoded to one.

## What Changes

1. **Scalar dtype abstraction**: one `KernelScalar` trait in
   `crates/quantibbler-kernels` (each impl declares its own accumulator)
   implemented for `f64`, `f32` and `bf16`, plus a fail-closed `Dtype`
   tag. `bf16` comes from the `half` crate (2.7.1 — the live crates.io
   latest at authoring time, already in the workspace lock graph).
2. **Dtype-generic FP-remainder products**: `packed_matvec` /
   `packed_matmul` defined once over `T: KernelScalar`, accumulation order
   unchanged. The `f64` instantiation is the v0.1 one, term for term.
   Bitplane pack/unpack and the solver stay integer/float64.
3. **PyO3 dtype-aware entry points** (`packed_matvec_dtype` /
   `packed_matmul_dtype`) returning dtype-consistent buffers; bf16 crosses
   as `uint16` bit patterns (NumPy has no native bfloat16).
4. **bf16 cross-face fixtures** at a declared, justified tolerance
   (backward 1e-2 on the row term scale; f32 backward 1e-6), exact across
   the three faces, fixtures byte-exact for the packed bytes.
5. **Grep-evidenced no-hardcoded-dtype** on the product surface (the
   definition of done names the evidence).

## Impact

- **Invariant-touching: the kernel surface — additive only.** The v0.1
  `f64` rlib and PyO3 signatures, error messages, accumulation order and
  outputs are byte-preserved; the storage format (qpk1 bitplane layout) is
  untouched. Compatibility impact: none for existing callers (server rlib
  call sites and compressor call sites compile and behave identically).
- **Affected specs**: `qk-kernels` — 2 modified requirements (stale
  float64-only wording), 7 added requirements.
- **Affected code**: `crates/quantibbler-kernels/src/{dtype.rs,
  dispatch.rs, core.rs, lib.rs}`,
  `crates/quantibbler-kernels/Cargo.toml`,
  `crates/quantibbler-kernels/Cargo.lock`,
  `crates/quantibbler-kernels/tests/parity.rs`,
  `crates/quantibbler-kernels/tests/fixtures/dtype_parity.json`,
  `tools/gen_kernel_fixtures.py`, `src/quantibbler/kernels/__init__.py`,
  `tests/test_kernels_parity.py`, `tests/test_kernels_dispatch.py`.

## Paper anchor

[NQ] Sec 3.1 pins the shared weight representation (binary factors with
full-precision scale vectors `s1`, `s2`); the precision of the
activation/remainder operand is a deployment choice, not part of the
representation — which is why the kernel surface carries it as a
parameter. Clean-room law: no implementation source consulted.

## Non-goals

- No performance work of any kind: no SIMD instantiation, no device
  kernel, no measured speed claim (the bench program owns measurement).
- No dtype genericity for the CubeCL device face (`gpu.rs`,
  `gpu_attention.rs`) or the SIMD fast face (`fast.rs`).
- No norm-application or logits-head genericity: those live in
  `quantibbler-server` (config component decomposition) and are owned by
  change `qk-serve-bf16`, which reuses this change's tolerance declaration.
- No storage-format or artifact change: the qpk1 container, serialization
  and the artifact dtype declaration are owned by `qk-artifact-dtype` /
  `qk-artifact-blob`.
- No solver change: `joint_itq` and its splitmix/Gram-Schmidt float64
  numerics stay bit-identical.
- No `f16`; no regeneration of the committed `f64` fixture records.
