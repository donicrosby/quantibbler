# qk-kernels Specification

## Purpose
TBD - created by archiving change qk-kernels. Update Purpose after archive.

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

The system SHALL provide packed-domain products against dense float64
operands: `packed_matvec(packed, n_bits, vec) -> float64 (n_rows,)` and
`packed_matmul(packed, n_bits, mat) -> float64 (n_rows, k)`, computing
exactly the arithmetic of unpack(packed) @ operand without materializing an
expanded binary matrix as the contracted fast path. Accumulation order SHALL
be fixed (row-element order) in every backend. Packed × packed products are
out of scope in v0.1.

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
- **THEN** the result equals the dense reference within float64 rel 1e-12

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

Parity SHALL be enforced by probe tests named after spec scenario IDs on
both sides (pytest across backends in-process; cargo against committed JSON
fixtures generated by the NumPy reference), with float outputs at float64
rel 1e-12 and packed outputs byte-identical (config Verification gates).
The pytest Rust leg SHALL skip cleanly when the extension is absent.

#### Scenario: parity_float_tolerance
- **WHEN** any probe test compares a float64 kernel output across backends
  (or Rust output against a fixture)
- **THEN** it asserts float64 relative tolerance 1e-12 and fails beyond it

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
- **THEN** it computes in float64
- **AND** no float32 path exists in v0.1 (float32 is a future, documented,
  opt-in accelerator)

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
