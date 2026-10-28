# qk-kernels Specification

## Purpose

Define the shared Rust kernels and parity contracts for packed binary-factor operations used by both the Python compressor and the pure-Rust inference server.

## Requirements

### Requirement: Packed bit layout for binary factors

The packed output SHALL be `uint8` bitplanes, one row per input row:
row-major order, little-endian bit order within each byte (element 8j+k →
bit k of byte j), +1 → bit 1, −1 → bit 0 ([NQ] Sec 3.1 Fig. 2: −1 → 0,
+1 → 1, 8-bit blocks); pad bits SHALL be written 0 and ignored on read.
This layout is invariant-touching (storage dominance / serialized bytes)
and MUST NOT change without a format version bump.

#### Scenario: pack_roundtrip_small
- **WHEN** a (5, 11) int8 factor of ±1 values is packed and then unpacked
- **THEN** the unpacked values equal the original on all 55 logical
  positions
- **AND** the packed array has dtype uint8 and shape (5, 2) with all pad
  bits equal to 0

#### Scenario: pack_roundtrip_exact8
- **WHEN** a (4, 8) ±1 factor is packed and unpacked
- **THEN** the packed shape is (4, 1) with no pad bits
- **AND** the round-trip is exact

#### Scenario: pack_roundtrip_padded
- **WHEN** a (3, 9) ±1 factor is packed and unpacked with `n_bits=9`
- **THEN** the round-trip preserves all 9 logical bits per row regardless of
  dirty pad bits in the packed input

#### Scenario: pack_roundtrip_empty
- **WHEN** pack_bits is called on a (2, 0) or (0, 7) array
- **THEN** the result is an empty packed/unpacked array of the correct dtype
- **AND** no error is raised

#### Scenario: pack_layout_bitorder
- **WHEN** rows containing a single +1 at element position k (k = 0..15) are
  packed
- **THEN** byte floor(k/8) of the packed row equals 2^(k mod 8) and all other
  bytes are 0
- **AND** a value of −1 at any position leaves the corresponding bit 0

#### Scenario: pack_rejects_non_binary
- **WHEN** pack_bits receives a value outside {−1, +1} (e.g., 0, 2, 0.5)
- **THEN** a ValueError is raised

### Requirement: Bit unpacking with explicit logical width

The system SHALL provide `unpack_bits(packed, n_bits, dtype)` restoring a
±1 factor (int8 or float64) from packed bitplanes, where `n_bits` is the
caller-supplied logical row length; pad bits beyond `n_bits` SHALL be
dropped, never validated.

#### Scenario: unpack_dtype_selection
- **WHEN** a packed array is unpacked with `dtype="int8"` and with
  `dtype="float64"`
- **THEN** both outputs represent the same ±1 pattern with the requested
  dtype
- **AND** int8 output contains only −1 and +1

### Requirement: Sign estimation with documented zero-handling

The system SHALL provide `sign_estimate(x) -> int8` mapping `x > 0 → +1`,
`x ≤ 0 → −1`, and exact zeros → +1 (documented tie-break; `sign(0) = +1`),
for use as the quantizer in [NQ] Sec 3.2 Step 3 (Eq. 10 finalization) and
the B-update of [LB2] Sec 4.4 (Algorithm 1 line 8).

#### Scenario: sign_zeros
- **WHEN** sign_estimate is applied to an array containing exact zeros
  (including an all-zero array)
- **THEN** every zero maps to +1
- **AND** no output value is 0

#### Scenario: sign_extremes
- **WHEN** sign_estimate is applied to positive and negative finite floats
  of any magnitude (e.g., 1e-300, −1e300)
- **THEN** positives map to +1 and negatives map to −1
- **AND** the output dtype is int8

### Requirement: Packed binary matvec and matmul

The system SHALL provide packed-domain products `packed_matvec(packed,
n_bits, vec)` and `packed_matmul(packed, n_bits, mat)`, computing exactly
the arithmetic of unpack(packed) @ operand without materializing an
expanded binary matrix. The dense-operand dtype SHALL default to float64
with the v0.1 contract (row-element accumulation order, bit-identical
outputs). An explicit dtype tag SHALL select float32 or bfloat16 on the
same generic body. Packed × packed products are out of scope in v0.1.

#### Scenario: matvec_matches_dense_reference

- **WHEN** packed_matvec is applied to a packed (6, 13) factor and a float64
  vector of length 13
- **THEN** the result equals `unpack_bits(packed, 13, float64) @ vec` within
  float64 rel 1e-12
- **AND** the result dtype is float64

#### Scenario: matvec_padded

- **WHEN** the packed factor's last row byte contains pad bits and `n_bits`
  is less than 8 × packed-width
- **THEN** pad bits contribute exactly 0 to every accumulation

#### Scenario: matmul_wide

- **WHEN** packed_matmul is applied to a packed (5, 12) factor and a (12, 3)
  float64 matrix
- **THEN** the result equals the dense unpack-then-multiply product within declared-dtype relative tolerance

### Requirement: Joint-ITQ alternating Procrustes solver

joint_itq(u_hat, v_hat, *, max_iter=50, tol, seed) SHALL implement the
alternating minimization of [LB2] Sec 4.4: Z = [Û; V̂], seeded random
orthogonal R, loop `B ← sign(ZR)`, `R ← ΨΦᵀ` from SVD(BᵀZ) (Algorithm 1
lines 8–10), stopping on `‖R_new − R_old‖_F ≤ tol` or max_iter; float64;
sign zero-tie-break (+1) applies; returns (u_t, v_t, n_iter, converged).
SVD init of Û, V̂ from W (lines 3–5) is out of scope.

#### Scenario: itq_preserves_product
- **WHEN** joint_itq converges on random float64 inputs Û (7, 3), V̂ (5, 3)
  with a fixed seed
- **THEN** `u_t @ v_t.T` equals `u_hat @ v_hat.T` within float64 rel 1e-12
- **AND** u_t and v_t rows are in {−1, +1} per the sign of the rotated
  factors' packing path (sign_estimate-compatible: int8 ±1 via
  sign_estimate(ZR) split)

#### Scenario: itq_converged
- **WHEN** joint_itq runs with default max_iter=50 and tol=1e-12 on
  well-conditioned random inputs with a fixed seed
- **THEN** converged is True and n_iter ≤ 50
- **AND** the final rotation increment norm ≤ tol

#### Scenario: itq_maxiter
- **WHEN** joint_itq runs with tol smaller than achievable in a
  finite-precision fixed point (e.g., tol=0) on random inputs
- **THEN** converged is False and n_iter equals max_iter exactly

#### Scenario: itq_deterministic_seed
- **WHEN** joint_itq is called twice with the same seed
- **THEN** the packed sign factors of (u_t, v_t) are byte-identical
- **AND** float outputs are bit-identical across repeated calls on the same
  backend

### Requirement: Kernel dispatch with graceful fallback and explicit override

Backend resolution SHALL be: explicit `backend=` argument →
`QUANTIBBLER_BACKEND` env var → automatic (Rust when importable, else NumPy
reference). Auto mode SHALL fall back to NumPy silently when
`quantibbler_kernels` is absent; every kernel SHALL exist identically on
both backends; an explicit Rust request with the extension absent SHALL
raise RuntimeError naming the module and the `kernels` extra.

#### Scenario: dispatch_default_auto
- **WHEN** any kernel is called with no backend argument and no env var
- **THEN** it uses the Rust extension when importable, otherwise the NumPy
  reference, and `backend()` reflects the active backend

#### Scenario: dispatch_env_override
- **WHEN** `QUANTIBBLER_BACKEND=numpy-reference` is set and pack_bits is
  called without an explicit backend
- **THEN** the NumPy reference path executes even when the extension is
  importable

#### Scenario: dispatch_explicit_rust_missing_raises
- **WHEN** the extension is absent and a kernel is called with
  `backend="rust"` (or the env var set to "rust")
- **THEN** a RuntimeError is raised naming `quantibbler_kernels` and the
  `kernels` extra

#### Scenario: dispatch_explicit_numpy_always_works
- **WHEN** a kernel is called with `backend="numpy-reference"` regardless of
  extension availability
- **THEN** the NumPy reference path executes

### Requirement: Rust/NumPy parity probe tests named by scenario ID

Parity SHALL be enforced by probe tests named after spec scenario IDs on both sides (pytest across backends in-process; cargo against committed JSON fixtures at each declared dtype), with float outputs matching same-dtype parity fixtures within per-dtype tolerances and packed outputs byte-identical (config Verification gates).
The pytest Rust leg SHALL skip cleanly when the extension is absent.

#### Scenario: parity_float_tolerance
- **WHEN** any probe test compares a kernel output across backends
  (or Rust output against a fixture) at a declared dtype
- **THEN** it asserts the declared per-dtype relative tolerance and fails beyond it

#### Scenario: parity_packed_bitidentical
- **WHEN** any probe test compares a packed output across backends (or Rust
  output against a fixture)
- **THEN** it asserts byte-identical buffers and fails on any differing byte

#### Scenario: parity_extension_absent_skips
- **WHEN** the pytest parity suite runs without `quantibbler_kernels`
  installed
- **THEN** Rust-leg probes skip with an explicit reason
- **AND** the NumPy legs still run and pass

### Requirement: CPU-only pure-Rust kernels without C/C++ dependencies

The kernel extension SHALL be CPU-only and pure Rust: no GPU kernels, no
C/C++/Cython/CUDA-C++ dependency anywhere in the kernel build or test graph
(config Architecture; C/C++ banned). Any linear algebra inside Rust kernels
(e.g., the r×r SVD of `BᵀZ`) SHALL come from pure-Rust crates verified
against the live registry at add-time and pinned within the config
dependency-freshness window.

#### Scenario: no_c_family_dependencies

- **WHEN** the kernel crate's dependency tree is inspected (cargo tree) at
  review time
- **THEN** no C/C++ compiler or C-linked linear-algebra crate (LAPACK/
  BLAS/BLIS bindings) is present
- **AND** the crate builds with no C toolchain invoked

#### Scenario: float64_default_numerics

- **WHEN** any kernel performs floating-point computation
- **THEN** it computes in float64 by default
- **AND** float32 and bfloat16 paths exist only behind an explicit dtype
  tag (the dtype-generic product requirements), never as a silent default
- **AND** the solver surface (joint_itq and its splitmix/Gram-Schmidt
  numerics) stays float64

### Requirement: Dual-face kernel surface from one core implementation

`quantibbler-kernels` SHALL expose ONE kernel implementation with TWO
consumption faces from one crate (config Component decomposition): a plain
Rust rlib API for pure-Rust consumers without a Python interpreter, and the
`quantibbler_kernels` PyO3 module for the compressor's quantized-side
inference instead of torch/NumPy reimplementation. Core logic SHALL live in
one pyo3-free layer; the PyO3 face is a thin binding; no algorithm SHALL be
duplicated.

#### Scenario: rlib_face_usable_without_python
- **WHEN** a pure-Rust binary built without a Python interpreter calls
  `pack_bits` and `packed_matvec` through the rlib face on a committed fixture
- **THEN** the calls compile and return the contracted packed layout and
  float64 results with no Python runtime involvement

#### Scenario: pyo3_face_thin_binding
- **WHEN** a kernel is invoked with identical inputs through the PyO3 extension
  module and through the rlib face
- **THEN** both invocations reach the same core implementation
- **AND** inspection of the PyO3 binding layer finds boundary conversion and
  error mapping only, with no kernel arithmetic outside the core layer

#### Scenario: single_core_implementation
- **WHEN** the crate source tree is inspected
- **THEN** each v0.1 kernel (`pack_bits`, `unpack_bits`, `sign_estimate`,
  `packed_matvec`, `packed_matmul`, `joint_itq`) has exactly one core
  implementation wrapped by both faces
- **AND** no duplicated algorithm body exists across the faces

### Requirement: Face parity between the rlib and PyO3 faces

The same fixture SHALL yield identical results when pushed through the rlib
face and the PyO3 face: packed outputs byte-identical, float64 outputs within
float64 rel 1e-12 (`openspec/config.yaml`, Verification gates). Face-parity
probes SHALL extend — never replace — the Rust-vs-NumPy parity probes, reusing
the committed fixtures so that the NumPy reference, the rlib face, and the
PyO3 face form one parity chain.

#### Scenario: face_parity_packed_bitidentical
- **WHEN** a committed packing fixture is pushed through `pack_bits` via the
  rlib face and via the PyO3 face
- **THEN** the packed uint8 buffers are byte-identical

#### Scenario: face_parity_float_rel_1e-12
- **WHEN** `packed_matvec`, `packed_matmul`, and `joint_itq` run on the same
  fixture through both faces
- **THEN** each pair of face outputs agrees within float64 rel 1e-12

#### Scenario: face_parity_extends_numpy_chain
- **WHEN** the face-parity probes run against the committed NumPy-reference
  fixtures
- **THEN** the rlib face and the PyO3 face each match the NumPy reference
  within the existing gates (packed byte-identical, float64 rel 1e-12)

### Requirement: cdylib and rlib crate-type coexistence

The crate SHALL keep `crate-type = ["cdylib", "rlib"]`, and both faces SHALL
build and test green from one workspace state: a single `cargo build` produces
both artifacts, and the gate suite (`cargo test`, `cargo clippy --all-targets
-- -D warnings`, `cargo fmt --check`, `uv run maturin build`) passes with no
feature-unification conflict or duplicate-symbol error between the faces.

#### Scenario: both_crate_types_build
- **WHEN** `cargo build` runs in `crates/quantibbler-kernels/`
- **THEN** both the rlib and the cdylib artifacts are produced (exit 0)

#### Scenario: workspace_gates_green_single_tree
- **WHEN** the gate suite runs from one checkout state (`cargo test`,
  `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`, and
  `uv run maturin build`)
- **THEN** all gates pass (exit 0 each) with no cross-face build conflict

#### Scenario: rlib_consumer_independent_of_cdylib
- **WHEN** a pure-Rust consumer links the rlib face
- **THEN** it builds without requiring the cdylib artifact or a Python
  installation

### Requirement: Opt-in extension-module feature

The `extension-module` feature SHALL remain opt-in (`default = []`) so
default-feature builds serve the pure-Rust world: `cargo build`, `cargo
test`, and `cargo clippy --all-targets` succeed without an extension-module
build, and pure-Rust rlib consumers link the rlib face with default
features. Only the maturin wheel build enables it (maturin enables it
automatically), producing the cdylib imported by the Python side.

#### Scenario: extension_module_absent_from_default_features
- **WHEN** the `[features]` table of `crates/quantibbler-kernels/Cargo.toml`
  is inspected
- **THEN** `extension-module` is defined and forwarded to
  `pyo3/extension-module`
- **AND** `default` is empty (`default = []`)

#### Scenario: cargo_test_green_with_default_features
- **WHEN** `cargo test` runs in `crates/quantibbler-kernels/` with default
  features, without any extension-module build step first
- **THEN** the suite — core layer, rlib face, and face-parity tests — passes
  (exit 0)

#### Scenario: maturin_enables_extension_module
- **WHEN** the extension is built and installed with maturin
  (`uv run maturin develop`)
- **THEN** the build enables `extension-module` and
  `import quantibbler_kernels` succeeds

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

### Requirement: Quantized tiled-attention NumPy reference (declared-dtype verification fixture)

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

### Requirement: Scalar dtype abstraction with f32 and bf16 first-class

The kernel crate SHALL carry ONE scalar abstraction: a `KernelScalar`
trait implemented for `f64`, `f32` and `bf16`, each impl declaring the
accumulator type its products sum in (`f64 -> f64`, `f32 -> f32`,
`bf16 -> f32`). A `Dtype` tag SHALL resolve a caller-supplied name
fail-closed. `bf16` values and conversions SHALL come from the `half`
crate. Generic kernels SHALL be written against the trait bound, never a
concrete float type.

#### Scenario: dtype_tag_resolution

- **WHEN** the `Dtype` tag is resolved from `"float64"`, `"float32"` and
  `"bfloat16"`
- **THEN** each name resolves to its own variant
- **AND** any other name is refused with an error naming the offending value
  and the three accepted names

#### Scenario: scalar_trait_covers_three_dtypes

- **WHEN** the crate's scalar trait is inspected
- **THEN** it is implemented for exactly `f64`, `f32` and `bf16`
- **AND** each impl declares that dtype's own accumulator (`f64 -> f64`,
  `f32 -> f32`, `bf16 -> f32`)

### Requirement: FP-remainder packed products generic over the dense-operand dtype

`packed_matvec` and `packed_matmul` SHALL be defined once, generic over the
dense-operand scalar `T: KernelScalar`, accumulating in `T`'s declared
accumulator with the accumulation order unchanged (row-element order, left
to right). Both the rlib face and the PyO3 face SHALL reach the one generic
body; no second algorithm body SHALL be introduced.

#### Scenario: product_generic_over_operand_dtype

- **WHEN** the packed products are instantiated at `f32` and at `bf16` on a
  packed factor and a dense operand of that dtype
- **THEN** both instantiations compile and return a result of the operand's
  own dtype
- **AND** both are computed by the same generic body the `f64`
  instantiation uses

#### Scenario: f64_product_bit_identical

- **WHEN** the committed `f64` parity fixtures run through `packed_matvec`
  and `packed_matmul` after the refactor
- **THEN** every float output is bit-identical to the pre-refactor bytes
- **AND** every packed output is byte-identical

### Requirement: Zero behavior change on the v0.1 f64 product surface

The refactor SHALL be structural, not numeric: every existing `f64` rlib and
PyO3 call site SHALL compile unchanged and route to the `f64` instantiation,
their error messages SHALL be unchanged, and no `f64` output SHALL move by
one ULP. No committed fixture record at any declared dtype SHALL be regenerated by this
change.

#### Scenario: f64_surface_signature_and_error_unchanged

- **WHEN** the `f64` product entry points are called with the committed
  fixtures and with malformed shapes
- **THEN** the results are the ones v0.1 produced
- **AND** the error strings are the v0.1 literals: `"vec must have length
  {n_bits}, got {}"`, `"mat length {} is not a multiple of n_bits {n_bits}"`
  and the `validate_packed` messages, probed by the executing tests named in
  tasks 2.1/4.2

### Requirement: bf16 cross-face parity at a declared tolerance

The committed fixtures SHALL carry bf16 variants of the packed products,
run through the rlib, PyO3 and NumPy-reference faces. bf16 deviation from
exact arithmetic over the declared-dtype operands (unrounded products, declared accumulator) SHALL be bounded as a
BACKWARD error — 1e-2 relative to the row's term scale `Σ_k |v_k|` — and
f32 products by the same criterion at 1e-6. Face-to-face agreement SHALL be
exact, and a tolerance SHALL never be loosened to admit an implementation
defect.

#### Scenario: bf16_product_within_declared_tolerance

- **WHEN** the bf16 product fixtures run through the rlib, PyO3 and
  NumPy-reference faces
- **THEN** each face's deviation from exact arithmetic over the declared-dtype
  operands (unrounded products, declared accumulator) is within the declared backward-error bound
- **AND** the f32 products stay within their own declared bound (1e-6 of
  the term scale)
- **AND** the three faces agree with each other bit-exactly
- **AND** the v0.1 `f64` rel 1e-12 gate (Requirement: Packed binary matvec
  and matmul) is unchanged — these backward-error bounds apply to the `f32`
  and `bf16` instantiations only

#### Scenario: bf16_packed_still_bit_identical

- **WHEN** fixture records that differ only in dense-operand dtype are
  generated and run
- **THEN** the packed bitplane inputs and outputs are byte-identical across
  the NumPy reference, rlib and PyO3 faces
- **AND** unpack-then-repack through the integer core is byte-exact — the
  dtype affects the dense operand only

### Requirement: PyO3 dtype-aware entry points with dtype-consistent buffers

The PyO3 face SHALL expose product entry points that take a dtype tag and
return an array of the matching dtype (`float64`, `float32`, `bfloat16`).
A bf16 buffer SHALL cross the boundary as `uint16` bit patterns, documented
as the wire form, because NumPy has no native bfloat16. An unrecognised tag
SHALL raise, naming the value.

#### Scenario: pyo3_dtype_tag_selects_output_dtype

- **WHEN** a product is called through the PyO3 face with an explicit dtype
  tag
- **THEN** for `f64`/`f32`, the returned array's NumPy dtype is the tagged
  dtype; for bfloat16, the returned array's dtype is `uint16` (the declared
  wire form; NumPy has no native bfloat16)
- **AND** its values agree with the rlib face's bit-exactly

#### Scenario: pyo3_unknown_dtype_refused

- **WHEN** a product is called through the PyO3 face with an unrecognised
  dtype name
- **THEN** it raises, naming the offending value, and returns no array

#### Scenario: bf16_wire_form_is_uint16

- **WHEN** a bf16 product is called through the PyO3 face
- **THEN** its input and output buffers are `uint16` bit patterns whose
  values are the bf16 encodings
- **AND** a torch `bfloat16` tensor maps directly through a
  `view(torch.uint16)`

### Requirement: Integer bitplane core untouched by dtype genericity

`pack_bits`, `unpack_bits` and the packed-layout validation SHALL stay
integer-only. Dtype genericity SHALL NOT introduce a float type into
bitplane packing, unpacking, or the packed-bit decoding that feeds the
products, and the committed packed fixtures' bytes SHALL be unchanged by
this change.

#### Scenario: bitplane_core_stays_integer

- **WHEN** the bitplane kernels and the packed-bit decoding are inspected
- **THEN** their signatures and bodies carry only integer types
- **AND** the committed packed fixtures' bytes are unchanged by this change

### Requirement: No hardcoded dtype in the product surface's signatures or dispatch

On the FP-remainder product surface, no kernel signature and no dispatch
site SHALL name a concrete float type outside that type's `KernelScalar`
impl block and the thin `f64` compatibility wrappers. The dtype SHALL be
carried by the trait bound or by the `Dtype` tag; product bodies SHALL name
only `T` and `T::Acc`.

#### Scenario: no_hardcoded_dtype_in_product_surface

- **WHEN** the FP-remainder product kernel bodies in `core.rs` — sliced by
  symbol, never by line number — are searched for a concrete float type
  name
- **THEN** no such name occurs in any of them: the bodies name only `T` and
  `T::Acc`
- **AND** the product signatures carry the `KernelScalar` bound
- **AND** the dtype enumeration lives in the dispatch module, not in a
  kernel

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
