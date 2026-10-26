# Design: qk-kernels-dtype-generic

## Context

`crates/quantibbler-kernels` carries three float-bearing surfaces today:

| surface | file | concrete type | gate |
|---|---|---|---|
| CPU rlib FP-remainder products | `src/core.rs` | `f64` | rel 1e-12 vs NumPy, bit-identical accumulation |
| CPU rlib SIMD fast path | `src/fast.rs` (feature `fast-cpu`) | `f64` | backward-error bound vs the scalar face |
| CubeCL device face | `src/gpu.rs`, `src/gpu_attention.rs` (feature `gpu`) | `f32` | rel 1e-12 on hardware only |

All `[features]` are opt-in (`default = []`); the default build compiles
`core.rs` and the PyO3 face alone. Dtype genericity applies to the
FP-remainder products (`core::packed_matvec`, `core::packed_matmul`) and
their PyO3 faces — the crate's only dense-operand float products.

## Decisions

### D1 — A trait bound, not enum dispatch, on the scalar

Chosen: one `KernelScalar` trait; kernels written as
`fn packed_matvec<T: KernelScalar>(...)`; the instantiation is selected at
the boundary. Discarded: an enum over payloads with a `match` in the inner
loop. Reason: the generic instantiation at `f64` compiles to the existing
expression verbatim, so the zero-fp32/f64-behavior-change requirement is
provable by construction rather than by re-measurement; a per-term `match`
changes the exact code shape the committed `f64` fixtures pin. Enum
dispatch is the right shape only where monomorphization is wrong
(code-size-bound callers); this crate has two callers and three types.

### D2 — The accumulator is a declared associated type, not the operand type

Chosen: `KernelScalar::Acc` with `f64 -> f64`, `f32 -> f32`,
`bf16 -> f32`. Discarded: accumulating in the operand dtype for every `T`.
Measured (200 random draws per width, worst absolute deviation divided by the
row's term scale `Σ_k |v_k|`, operand bf16-rounded in both arms; the `f32`
arm's delivered value is its `f32` accumulator rounded once into the bf16 wire
form, which is the deployed shape):

| `n_bits` | accumulate in `f32` | accumulate in `bf16` | ratio |
|---|---|---|---|
| 13 | 2.56e-3 | 4.00e-3 | 1.6x |
| 64 | 8.89e-4 | 6.12e-3 | 6.9x |
| 256 | 5.19e-4 | 6.22e-3 | 12.0x |
| 1024 | 2.78e-4 | 4.48e-3 | 16.1x |

The `bf16` accumulator is a random walk of per-term roundings: it does not
average out, its worst case does not shrink as the row widens (4.0e-3 to 4.5e-3
across 13 to 1024 terms), and it leaves only a 1.6x margin under the declared
1e-2 bound. The `f32` accumulator's error falls by an order of magnitude over
the same widths because only the output rounding survives — bf16 storage with
`f32` accumulation is the standard inference-kernel shape, and the shipped arm
keeps a 4.7x-5.4x margin (crystallization entry, 2026-10-26). The `f64`
instantiation's accumulator is `f64`, so its bytes are the v0.1 bytes.

### D3 — `half` as the bf16 source

Chosen: the `half` crate, pinned `half = "2.7"` (resolves 2.7.1).
Freshness evidence, verified against the live crates.io registry at
authoring time: 2.7.1 IS the latest stable — the requirement sits exactly
on the floor, within one major of latest per the config dependency
freshness rule. The crate is already in the workspace lock graph via the
tokenizers toolchain, is pure Rust with no C dependency, and does not
breach the C/C++ ban. Discarded: hand-rolled bit manipulation against
`u16` — a private rounding convention is exactly what the parity contract
exists to avoid.

### D4 — bf16 crosses the PyO3 boundary as `uint16` bit patterns

Chosen: `uint16` buffers with a `"bfloat16"` dtype tag; Rust converts with
`bf16::from_bits` / `to_bits`; Python documents the wire form as
`tensor.view(torch.uint16)`. NumPy has no native bfloat16, so some
bit-level transport is unavoidable; making it explicit is the fail-closed
choice. Discarded: (a) `float32` buffers with a silent narrowing — hides
the cast the caller asked for and makes the declared dtype a lie;
(b) an `ml_dtypes` dependency — a new Python dependency for transport, and
the product's dependency set is ruled, not grown for convenience.

### D5 — Dtype-aware entry points are added beside the v0.1 ones

Chosen: the v0.1 PyO3 `packed_matvec` / `packed_matmul` keep their exact
signatures and route to the `f64` instantiation; new
`packed_matvec_dtype` / `packed_matmul_dtype` take the tag. The Python
facade adds a `dtype=` keyword routing `float64` to the v0.1 entry point
and the other tags to the dtype-aware ones. Discarded: widening the
existing signatures with an optional `dtype=` keyword — the v0.1 signature
is a shipped contract with committed fixtures and a server caller, and
zero behavior change is cheapest to prove when the old symbol is untouched.

### D6 — The bf16/f32 tolerance is a BACKWARD bound on the term scale

The brief's honest tolerance for bf16 is ~1e-2 relative (8 mantissa
bits). Declared form: **backward** — deviation from the exact `f64` product
of the SAME operand, divided by the row's term scale `Σ_k |v_k|` — with
bf16 at 1e-2 and f32 at 1e-6. A forward-relative bound is governed by the
sum's condition number: measured 4.5e-2 forward against 3.3e-3 backward on
the committed matmul fixture, i.e. a forward criterion would have forced a
tolerance ~14x looser than the arithmetic needs and would stop being
evidence. This is the same criterion the SIMD face's measured bound
already uses; it is reused, not reinvented.

## Tolerance declaration (canonical — reused by qk-serve-bf16)

- **float64**: rel 1e-12 and bit-identical accumulation — the unchanged
  v0.1 gates.
- **f32 products**: backward 1e-6 of the row term scale `Σ_k |v_k|`.
- **bf16 products**: backward 1e-2 of the row term scale.
- **Face-to-face** (rlib vs NumPy reference vs PyO3): EXACT, bit-identical
  — all three declare the same accumulator, so a tolerance there would
  hide a face that quietly picked a different one. The tolerances are
  exercised against the exact `f64` product of the same operand (the
  genuinely different-precision computation each dtype replaces), never
  between two faces.
- **Term scale**: for one output element, the sum of absolute values of
  the dense operand's elements the row contracts with (matvec: the vector;
  matmul: the column).
- **Measured worst at authoring time** (units of the term scale): f32
  7.99e-8 (matvec), 2.81e-8 (matmul); bf16 1.86e-3 (matvec), 2.11e-3
  (matmul) — 12.5x / 35.6x and 5.4x / 4.7x inside the declared bounds.
  Each bound is re-verified and recorded in the gate row at landing.

## Excluded surfaces (named, with the reason)

- **Norm application and the logits head — not in this crate.** They live
  in `crates/quantibbler-server/src/engine/`, which the config component
  decomposition assigns to `quantibbler-server`, not to
  `quantibbler-kernels`. Dtype-generic serving composition is change
  `qk-serve-bf16`; this change deliberately does not move serving code
  into the kernel crate.
- **Scale application — caller-side.** The only in-crate scale arithmetic
  is the `±1` sign multiply, now written in `T`/`T::Acc`. The `s1`/`s2`
  dequantization scales are applied by the callers (the server's engine
  and the compressor's reconstruction); there is no scale kernel here to
  make generic. The artifact-side dtype declaration is owned by
  `qk-artifact-dtype` / `qk-artifact-blob`.
- **SIMD fast path (`fast.rs`) — stays `f64`.** Its whole contract is a
  measured backward-error bound against the `f64` scalar face; a second
  instantiation has no measured bound and performance work is out of
  scope. The generic surface is the core face.
- **CubeCL device face (`gpu.rs`, `gpu_attention.rs`) — stays `f32`.** The
  arithmetic type is fixed by the adopted substrate (cubecl 0.11 /
  cubek-attention 0.3), correcting a device path is a hardware-gated proof
  obligation (`QUANTIBBLER_GPU_TESTS=1`), and no device is authorized in
  this lane. This is why the no-hardcoded-dtype grep is scoped to the
  product surface: the device face IS an f32-specific surface.
- **Solver surface — stays `f64`.** `sign_estimate`, `joint_itq`,
  `splitmix_matrix`, `gram_schmidt` and the two f64 constants are the
  pinned solver numerics; their operands are the `f64` solver latents and
  the `qk-kernels` spec pins float64 there.

## Informed assumptions

1. **The bf16 bound has to be backward, not forward-relative.** bf16's 8
   mantissa bits make operand rounding ≤ 2^-8 per element, so deviation
   scales with the terms `Σ_k |v_k|`, while a forward-relative bound is
   governed by the sum's condition number and is unbounded on a canceling
   row. Measured on the committed matmul fixture (D6). How a wrong
   assumption surfaces: the fixture legs redden.
2. **Accumulating a bf16 product in bf16 is worse than in f32, and by a
   margin.** Measured across widths (D2 table): the ratio grows 4.2x to
   18.5x with row width, and the bf16 accumulator's worst (1.08e-2) would
   not fit the declared 1e-2 at all. A single-width sample cannot
   discriminate the two arms — the first fixture built (13 terms) had them
   coincide at 1.86e-3; the claim is pinned by the width sweep, not one
   draw.
3. **`half`'s `bf16::from_f32` rounds to nearest-even, bit-matching
   torch's cast.** Confirmed on 10,000 samples: zero mismatches. The
   committed bf16 operands are generated by TORCH, not by the package's
   own emulation, so the fixture cannot validate itself.

## Discarded alternatives (whole-change)

- **Make only the PyO3 face generic** (Python passes a dtype string, Rust
  keeps an f64 core). Discarded: satisfies the letter of "dtype-aware
  entry points" while leaving the rlib face — half the crate's consumers —
  hardcoded, which is the exact shape the directive names.
- **Introduce `f16` as well.** Discarded: the brief names no third type,
  and f16's 5-bit exponent range is wrong for accumulator-scale values in
  this pipeline. Adding it later is one impl block.

## Open questions

| question | status | owner |
|---|---|---|
| SIMD `fast-cpu` face gains an f32 instantiation? | open — needs a measured backward-error bound, i.e. a performance lane | planner/@user |
| norm/logits dtype-genericity | routed — server lane, change `qk-serve-bf16` (spec card parented to this one) | planner |
| `sign_estimate` generic over its input dtype? | resolved — no; its operand is the f64 solver latent and the `qk-kernels` spec pins float64 | this change |

## Implementation crystallization (2026-10-25)

Recorded on the implementation branch (green fixtures there) at spec
authoring time; every number is re-verified in the gate row at landing.

- **The f64 surface did not move.** The 24 pre-existing crate tests pass
  unchanged, and the `f64_product_bit_identical` leg re-runs every
  committed f64 product record through the f64 instantiation asserting
  BYTE identity. Regenerating the fixtures left `pack_bits.json`,
  `sign_estimate.json`, `packed_matvec.json`, `packed_matmul.json` and
  `joint_itq.json` byte-identical (`cmp`) — the change does not perturb
  the generator's output.
- **Both tolerance bounds are load-bearing.** Tightening them to 1e-12 /
  1e-6 reddened all four dtype legs; the files were restored
  byte-identical.
- **Face parity is exact, not toleranced.** rlib, NumPy reference and PyO3
  agree bit-for-bit on all four dtype fixtures; the declared tolerances
  are exercised only against the exact f64 product.
- **The no-hardcoded-dtype pin is a symbol-sliced source scan.** It slices
  `packed_matvec`, `packed_matmul`, `accumulate_row` and
  `accumulate_packed_row` out of `core.rs` by their generic signatures —
  never by line number — and asserts no `f64`/`f32`/`bf16`/`half::` token
  inside any of them. Mutation-proven: planting `let _probe: f64 = 0.0;`
  in `accumulate_row` reddens it naming the symbol. The remaining f64
  tokens in `core.rs` are the excluded solver surface, enumerated in the
  gate row's grep evidence.
- **The bf16 operands in the fixtures come from torch**, not from this
  package's emulation, and a probe asserts the two agree bit-for-bit —
  otherwise a rounding-mode bug could be written into the fixture and
  validated by itself.
- **Source compatibility held for the rlib consumer.** `cargo check -p
  quantibbler-server --all-targets` is clean; the generic signature change
  needs no call-site edit in the server.
- **One pre-existing test needed a one-token adaptation**, not a
  relaxation: `packed_matmul_zero_columns_returns_empty` passed `&[]`,
  which leaves `T` uninferred; it is now `&[] as &[f64]` with the
  assertion unchanged.

### Scope amendment (records a delta against the brief's scope sentence)

The brief named "scale application, the FP-remainder matmul paths, norm
application, and the logits head" as the surfaces to make generic. On this
tree, only the FP-remainder products exist in `quantibbler-kernels`; norm
application and the logits head are serving code in
`crates/quantibbler-server/src/engine/` and belong to a different
component per `openspec/config.yaml`. This change deliberately does not
move them — the serving side is change `qk-serve-bf16`, whose spec card
already depends on this one.

## Implementation crystallization (2026-10-26) — reconciled against the reviewed delta

The reviewed delta replaced the draft change dir in the shared checkout; the
implementation was then re-read against it and every gate re-run on the
reconciled tree. Numbers below are from that run, not carried over.

- **The `f64` surface did not move.** The 24 pre-existing crate tests pass
  unchanged; every committed `f64` product record is reproduced BYTE-identically
  through the `f64` instantiation, and regenerating the fixtures leaves
  `pack_bits.json`, `sign_estimate.json`, `packed_matvec.json`,
  `packed_matmul.json` and `joint_itq.json` byte-identical (`cmp`). Crate suite:
  **34 passed / 0 failed** (2 lib unit + 32 parity = the 24 pre-existing + 8
  dtype legs + `parity_f64_error_strings_unchanged`).

- **Per-leg measured worst (task 4.1)**, units of the term scale. The fixture
  legs exercise each declared bound against the exact `f64` product of the same
  operand — never face against face, which is exact:

  | leg | measured worst | declared bound | margin |
  |---|---|---|---|
  | bf16 matvec | 1.859e-3 | 1e-2 | 5.4x |
  | bf16 matmul | 2.106e-3 | 1e-2 | 4.7x |
  | f32 matvec | 7.991e-8 | 1e-6 | 12.5x |
  | f32 matmul | 2.814e-8 | 1e-6 | 35.5x |

- **D2's width sweep, re-measured (task 5.1).** D2's table now carries the
  re-measured columns. The shipped arm (bf16 operand, `f32` accumulator, bf16
  output) reproduces the authoring-time figure at width 13 (2.56e-3) and its
  falling-with-width shape. The discarded arm (accumulate in `bf16`) is NOT
  reproduced at the authoring-time level — re-measured 4.00e-3 / 6.12e-3 /
  6.22e-3 / 4.48e-3 for 13 / 64 / 256 / 1024 terms against 1.08e-2 / 1.03e-2 /
  8.99e-3 / 5.54e-3 — so the earlier claim that the discarded arm "would consume
  the declared 1e-2 bound entirely" does not hold under the arm definition D2
  states; it leaves a 1.6x margin. The decision is unaffected and rests on the
  shape the re-measurement strengthens: the bf16 accumulator's worst case does
  not shrink with row width while the `f32` accumulator's falls an order of
  magnitude, and the ratio grows 1.6x -> 16.1x. Recorded as a doc-vs-measurement
  divergence for the planner rather than reconciled silently.

- **Grep evidence (task 5.2).** Scoped `grep -c -E '\b(f64|f32|bf16)\b'` over
  `src/{dtype.rs, core.rs, lib.rs}` = **dtype.rs 40, core.rs 25, lib.rs 21**;
  not one hit is inside a product kernel body.
  - `dtype.rs` (40) — the three `KernelScalar` impl blocks and their unit
    tests; naming a concrete type is an impl block's whole purpose.
  - `core.rs` (25) — 8 are doc-comment prose, the remainder the pinned solver
    surface (`DEFAULT_TOL`, `TWO_POW_53`, `splitmix_matrix`, `gram_schmidt`,
    `matmul_lr`, `matmul_at_b_lr`, `sign_estimate`, `joint_itq`), which stays
    float64 per the `float64_default_numerics` scenario. `accumulate_row`,
    `accumulate_packed_row`, `packed_matvec` and `packed_matmul` carry zero, and
    the symbol-sliced source scan asserts it by symbol, never by line number.
  - `lib.rs` (21) — the PyO3 boundary: the `dyn_f64`/`dyn_f32`/`dyn_bf16`
    extractors, the two v0.1 `f64` compatibility wrappers, and the tag arms of
    the dtype-aware entry points — exactly the two classes the requirement
    carves out.

- **`half = "2.7"` (task 1.3).** Registry floor re-checked at reconciliation
  time: `cargo search half` -> `half = "2.7.1"`, the current stable, so the pin
  sits on the floor and within one major of latest. Pure Rust, no C
  dependency — the C/C++ ban holds.

- **bf16 fixture provenance (task 3.2).** `gen_kernel_fixtures.py` takes its
  bf16 operands from torch through a named `_require_torch()` that raises a
  hard generation error when torch is absent, so an emulated operand cannot
  silently stand in for the independent encoder.

- **No commit from this leaf (tasks 6.7).** The reconciled tree is the
  deliverable and stays unstaged: landing waits for the pre-landing
  implementation review to return PASS, and the shared checkout is live with
  sibling lanes, so a scoped stage here could not be scoped to this change.
