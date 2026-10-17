# design.md — qk-kernels

Follows `openspec/config.yaml` (binding): evidence hierarchy §Evidence, parity
gates §Verification gates, C/C++ ban §Architecture.

## 1. Context

The crate `quantibbler-kernels` (pyo3 0.30, abi3-py311, edition 2024, MSRV
1.85) is scaffolded with a single `backend_version()` function.
`src/quantibbler/kernels.py` imports it optionally and reports `backend()`.
The NumPy path is the executable specification; Rust must match on float64
within rel 1e-12 plus bit-identical packed output on every probe test.

Paper grounding for the v0.1 surface:

- **pack_bits / unpack_bits** — [NQ] Sec 3.1, Fig. 2 stage (c): binarized
  factors map −1 → 0, +1 → 1 and are "packed into integer formats (e.g.,
  8-bit blocks) for memory efficiency". Storage dominance is counted from
  serialized bytes (config, Method invariants), so the packing layout is a
  storage-format decision, not an implementation detail.
- **sign_estimate** — the quantizer used in [NQ] Sec 3.2 Step 3 (Eq. 10
  finalization: `U±1 = sign(U)`) and in the B-update `B ← sign(ZR)` of [LB2]
  Sec 4.4 Algorithm 1 line 8.
- **packed_matvec / packed_matmul** — the packed-domain forward products:
  [NQ] Sec 3.3 freezes and packs binary weights before global scale
  calibration, so reconstruction-side products run against packed operands.
- **joint_itq** — [LB2] Sec 4.4 (Eq. 9–10, Algorithm 1): the alternating
  minimization `B ← sign(ZR)`, `ΦΩΨᵀ ← SVD(BᵀZ)`, `R ← ΨΦᵀ` over the joint
  manifold `Z = [Û; V̂] ∈ R^((din+dout)×r)`; converges in ≤ ~50 iterations
  ([LB2] Sec 4.4). Cited from Sec 4.4 of the pinned copy (the task brief said
  3.3; the section numbering of the pinned markdown is authoritative here).

## 2. Decisions

### D1. Packed layout (invariant-touching)

Packed output is `uint8` of shape `(n_rows, ceil(n_bits_per_row / 8))` —
one uint8 bitplane row per input row:

- Row-major element order; within a row, byte `j` holds elements `8j..8j+7`;
  within a byte, **little-endian bit order**: element `8j + k` → bit `k`
  (LSB-first), k = 0..7.
- `+1 → bit 1`, `−1 → bit 0` ([NQ] Sec 3.1 Fig. 2 stage (c)).
- Trailing pad bits (row length not a multiple of 8) are **zero** on write and
  **ignored** on read — never validated, never assumed.
- `unpack_bits` is the exact inverse on non-pad bits and is tolerant of dirty
  pad bits.

Rationale: byte-per-row-of-8 granularity keeps column slicing of the *bit*
axis impossible-free but row operations trivially SIMD-shaped in Rust later,
and little-endian bit order matches the dominant packed-gemm literature and
x86/ARM `PSHUFB`-style LUT decode. The layout is agreed across the qk-* lanes;
this change pins it in spec, and any change requires a format version bump
(proposal: Impact).

### D2. `pack_bits` / `unpack_bits` semantics

- `pack_bits(factor) -> uint8 (n, w)` where `factor` is an array of ±1
  (int8 or float; see D3) with `n` rows. Any dimensionality is flattened
  row-major to `n × m` first; `w = ceil(m/8)`.
- `unpack_bits(packed, dtype) -> array (n, m)` with `dtype ∈ {int8, float64}`;
  the packed width `w` implies `m ∈ {8(w−1)+1 .. 8w}`; the caller passes the
  logical width `m` explicitly (`n_bits`) — pad bits are dropped, so shape
  recovery from bytes alone is ambiguous and we do not pretend otherwise.
- Validation: input values outside {−1, +1} raise `ValueError`; empty arrays
  (m = 0 or n = 0) are supported and round-trip to empty.

### D3. `sign_estimate` and zero-handling

- `sign_estimate(x) -> int8 (same shape)` mapping `x > 0 → +1`,
  `x ≤ 0 → −1` **with the documented exception: exact zeros map to +1**
  (`sign(0) = +1`). Output dtype `int8` so it feeds `pack_bits` without a
  float round-trip.
- Zero → +1 is a documented tie-break, required because `np.sign(0) = 0`
  is not a legal binary value and the papers' `sign(·)` is defined on
  nonzero latents ([LB2] Sec 4.4 uses sign(ZR) with random-rotated, hence
  a.s. nonzero, arguments — the tie-break matters for near-degenerate
  factors). Both NumPy and Rust implement identical zero-handling; parity
  probes include an explicit all-zeros scenario.

### D4. Packed binary matvec / matmul semantics

- `packed_matvec(packed, n_bits, float_vec) -> float64 (n_rows,)`:
  computes `sign_unpacked @ float_vec` without materializing the int8
  matrix — per row, accumulate `dot(row_bits, vec)` with +1/−1 decode
  (Popcount-LUT in Rust; NumPy does the arithmetic equivalent:
  `2*bits − 1` per bit then weighted sum).
- `packed_matmul(packed, n_bits, float_mat) -> float64 (n_rows, k)`:
  column-batched matvec. Both kernels are **packed × dense-float only**;
  packed × packed is a non-goal (proposal).
- `n_bits` (logical row length) is an explicit argument per D2.
- All float arithmetic in float64 (`openspec/config.yaml`, Numerics).

### D5. `joint_itq` semantics

- `joint_itq(u_hat, v_hat, *, max_iter=50, tol=1e-12, seed=None)
  -> (u_t, v_t, n_iter, converged)`:
  [LB2] Sec 4.4 Algorithm 1 steps 6–11 only — given the SVD-derived latents
  Û, V̂ (Algorithm 1 lines 3–5 are the caller's job, keeping this kernel a
  pure Procrustes loop), build `Z = [Û; V̂]`, iterate:
  `B ← sign(ZR)`; `Φ,Ω,Ψᵀ ← SVD(BᵀZ)`; `R ← ΨΦᵀ`; until
  `‖R_new − R_old‖_F ≤ tol` or `n_iter == max_iter`.
- `max_iter` default 50 ([LB2] Sec 4.4: converges within 50 iterations);
  `tol` on the rotation increment (chosen over objective-delta or B-flip
  count — see DA3); `seed` seeds the random orthogonal R₀ (Algorithm 1 line
  6) deterministically so parity probes are reproducible; both backends
  implement identical seeded RNG. Returns `n_iter` and `converged` so callers
  and probes can assert loop behavior, not just the fixed point.
- float64 throughout; inputs are float64 arrays `((din+dout) × r shared
  rank)`, shapes `u_hat (dout, r)`, `v_hat (din, r)` with matching r.
- Convergence criterion: the papers do not state the exact stopping rule
  (Algorithm 1 runs a fixed T); `tol` on the rotation increment is an
  **informed assumption** (A2).

### D6. Dispatch semantics

`src/quantibbler/kernels.py` remains the only dispatch point:

- `backend() -> str` ∈ {`rust`, `numpy-reference`} — unchanged.
- Resolution order per call: explicit `backend=` kwarg (`"rust"` /
  `"numpy-reference"`) → `QUANTIBBLER_BACKEND` env var → auto (Rust when
  importable, else NumPy).
- **Graceful fallback when the extension is absent** (config Architecture:
  the Python wrapper "FALLS BACK to a NumPy reference path when the
  extension is absent"): auto mode silently uses NumPy; every public kernel
  exists with identical semantics on both backends, so user code runs
  extension-less.
- **Explicit override fails loudly**: `backend="rust"` with the extension
  absent (or `QUANTIBBLER_BACKEND=rust` without the extension) raises
  `RuntimeError` naming the missing extension and the install extra
  (`pip`-sentence: install with the `kernels` extra) — silent degradation
  under an explicit performance request is a bug, not a feature.
- Each kernel forwards to `_numpy_<name>` or `_rust_<name>`; Rust wrappers
  convert dtypes at the boundary and re-raise `pyo3` errors as documented
  exception types (ValueError for validation, RuntimeError for state).

### D7. Parity test strategy

- **Scenario IDs**: every probe test is named for its scenario ID
  (`tests/test_kernels_parity.py::test_parity_<scenario>`; `#[test] fn
  parity_<scenario>` on the Rust side) so a failure names the exact contract
  breach. Scenario families: `pack_roundtrip_{small,exact8,padded,empty}`,
  `pack_layout_bitorder` (single 1 at each bit position), `sign_zeros`,
  `sign_extremes`, `matvec_padded`, `matmul_wide`, `itq_converged`,
  `itq_maxiter`, `itq_deterministic_seed`, plus randomized shapes fixed by
  seed.
- **Float tolerance**: float64 outputs compared with rel 1e-12 —
  `openspec/config.yaml` Verification gates ("Rust kernel output vs NumPy
  reference (rel 1e-12, float64)").
- **Bit-exactness**: packed outputs compared byte-identical (the
  "sign-agreement invariant: packed bits identical" gate); validated with
  `assert packed_a.tobytes() == packed_b.tobytes()`.
- **Mechanism**: pytest probes call both backends in-process (via the
  `backend=` kwarg) when the extension is installed and auto-skip the Rust
  leg when it is not, so the suite is green both ways; the Rust-side cargo
  tests compare against **committed JSON fixtures generated by the NumPy
  reference** (golden files under `crates/quantibbler-kernels/tests/fixtures/`),
  keeping the two test suites independent of each other's runtime.
- **Fixture regeneration** is a scripted, committed command
  (`tools/gen_kernel_fixtures.py`, run via `uv run`) so fixture drift is a
  reviewable diff, never a manual copy.

### D8. Rust-side numerics

`joint_itq` needs SVD. No LAPACK linkage (config bans C/C++ anywhere in the
build graph). Pure-Rust options: `nalgebra` (pure Rust, f64 SVD via
jacobi/implicit symmetric algorithms) or a hand-rolled symmetric jacobi for
the `BᵀZ` step (`BᵀZ ∈ R^(r×r)`, r ≤ few hundred — small). **Decision:
nalgebra** (pure Rust; dependency freshness rule applies — verify latest
stable on crates.io at implementation time and pin within one minor/major
window per config). Rust parity for `joint_itq` is asserted on the *outputs*
(`u_t`, `v_t`, `n_iter`, `converged`) against NumPy reference, not on
bit-identical intermediate rotations; float outputs must still meet rel
1e-12, which holds when both sides run the same algorithm in float64 with
deterministic seeded R₀. If rel-1e-12 proves unattainable across SVD
implementations (open question OQ2), the fallback contract is: Rust returns
`converged`/`n_iter` + packed `sign(·)` factors bit-identically, with float
`u_t`/`v_t` held to rel 1e-9 — this relaxation would itself be a
config-gate deviation and requires an explicit config.yaml change first.
Bit-exact kernels (`pack_bits`, `sign_estimate`, packed matvec/matmul
accumulation order) have no such wiggle room: accumulation order is pinned
left-to-right per row in both implementations so float dot products are
bit-identical, not merely 1e-12-close.

### D9. Dual-face surface — one core, two faces (user-directed component split)

The user directed the repo-level component split now recorded in
`openspec/config.yaml` (Component decomposition): a pure-Rust
OpenAI-compatible server (`crates/quantibbler-server`, planned; authored by
the qk-server lane as its own change) plus a Python compressor
(`src/quantibbler/`). The kernels are the shared compute substrate for both,
so the crate must expose ONE kernel implementation with TWO consumption faces:

- **rlib face** — the plain Rust public API of the crate, linked directly by
  the server (config Architecture: the serving component has "No Python
  dependency, no embedded interpreter, ever").
- **PyO3 face** — the `quantibbler_kernels` extension module consumed by the
  compressor for quantized-side inference (parity probes, pipeline gates)
  instead of a torch/NumPy reimplementation (config: the Python side "does not
  reimplement kernels in torch or NumPy"; the NumPy modules in
  `src/quantibbler/kernels/` remain the executable specification and the
  extension-absent fallback, not an inference engine).

Layering: core kernel logic lives in a pyo3-free core layer (plain Rust
signatures, no pyo3 types in its interface); the rlib face IS that layer's
public API; the PyO3 face is thin `#[pyfunction]` wrappers doing boundary
conversion (arrays → slice types per D2–D5) and error mapping (D6 rules:
ValueError for validation, RuntimeError for state). No algorithm code exists
outside the core layer. v0.1 kernel semantics (D1–D8) are unchanged — the
dual face is a packaging contract over the same behavior, verified by face
parity (D11).

### D10. Build graph: crate-type coexistence, opt-in extension-module

`crate-type = ["cdylib", "rlib"]` (already declared) means one `cargo build`
produces both faces' artifacts from one source tree; the gate suite (cargo
test / clippy / fmt, `uv run maturin build`) must stay green in a single
workspace state with no feature-unification conflicts or duplicate symbols.

`extension-module` stays opt-in (`default = []`, as scaffolded).
Default-feature builds are the pure-Rust world: `cargo test`, clippy, and any
rlib consumer build without the extension-module configuration. maturin
enables the feature for the wheel, so the cdylib defers Python symbol
resolution to the importing interpreter. Mechanics worth recording:
extension-module ON removes the libpython link directive from the cdylib
(symbols resolved at import time); default builds keep pyo3's ordinary
linkage, and the v0.1 test design (D11) deliberately introduces no
embedded-Python cargo tests, so default `cargo test` runs have no
interpreter-embedding step. Whether the server can consume the rlib with pyo3
fully absent from its link graph (optional/target-gated pyo3 dependency) is
deferred — see OQ6.

### D11. Face-parity probes

D7's backend parity (NumPy ↔ Rust extension) extends to a three-way chain
over the SAME committed fixtures:

    NumPy reference ↔ rlib face ↔ PyO3 face

- The fixture generator (`tools/gen_kernel_fixtures.py`, D7) additionally
  records PyO3-face outputs (produced through the maturin-built extension)
  next to the NumPy-reference outputs.
- cargo tests push each fixture through the **rlib face** and compare against
  both records: the NumPy reference (existing Rust-vs-NumPy gate) and the
  PyO3-face record (new face-parity gate). Packed outputs byte-identical;
  float64 outputs rel 1e-12 — the config Verification gates, unchanged.
- pytest probes continue to compare NumPy ↔ the extension in-process (the
  extension IS the PyO3 face), so every adjacent pair of the chain is pinned
  by an executable test.

No embedded-Python cargo tests are introduced: the rlib leg is pure Rust and
the PyO3 leg is exercised through the built extension, keeping default
`cargo test` runs free of an interpreter-embedding step (D10).

## 3. Discarded alternatives

- **DA1. Column-major / bit-major packing (bitplane per column).** Rejected:
  row-wise serialization matches the row-major storage of Ub/Vb and keeps
  `unpack` row-slices contiguous; column bitplanes would make the common
  `packed @ x` kernel gather-strided.
- **DA2. Sign convention +1 → bit 0 (OneHot/Complement style used in some
  BNN works).** Rejected: contradicts the agreed lane convention and [NQ]
  Sec 3.1 Fig. 2's explicit −1 → 0, +1 → 1; misalignment here would corrupt
  every serialized artifact.
- **DA3. Convergence by objective value `‖B − ZR‖²_F` delta or B-flip count.**
  Rejected for tol semantics: the objective is flat near the fixed point and
  flip-count stalls early; rotation-increment norm is monotone-decreasing,
  cheap, and directly bounds the fixed-point distance.
- **DA4. Fixed iteration count only (exactly as Algorithm 1's `for t=1..T`).**
  Rejected: a pure fixed count gives callers no convergence evidence;
  `max_iter` + `tol` keeps the paper's bound while making convergence
  observable (`n_iter`, `converged`).
- **DA5. Return binary floats (float64 ±1.0) from `sign_estimate`.** Rejected:
  int8 halves the intermediate footprint and feeds `pack_bits` directly;
  float ±1 remains available at the `unpack_bits(dtype=float64)` boundary.
- **DA6. Expose `sign_estimate` returning 0 for exact zeros (np.sign parity).**
  Rejected: 0 is not a legal binary factor ([NQ] Sec 3.1 defines Ub ∈
  {−1,+1}); zeros would poison pack/unpack and packed matvec decode.
- **DA7. `joint_itq` taking W and doing SVD-init internally (Algorithm 1 lines
  3–5).** Rejected: couples the kernel to the representation lane's SVD
  choices; a pure Procrustes loop is testable, reusable by both the [LB2]
  path and PTQ-side experiments, and keeps the kernel surface minimal.
- **DA8. `nalgebra` vs hand-rolled jacobi — hand-rolled rejected.** A private
  jacobi implementation is ~200 lines and audit-heavy; nalgebra is pure Rust,
  maintained, and satisfies the dependency-freshness rule. (LAPACK/`ndarray`
  + `lax` rejected outright: C dependency, banned.)
- **DA9. Dispatch via a `Backend` enum object / context manager instead of
  string kwarg + env var.** Rejected for v0.1: strings keep the public API
  trivially serializable and testable; a context manager adds statefulness
  (thread-local override) with no v0.1 consumer.
- **DA10. Packed matvec via `unpack` then BLAS matmul in Python.** Rejected as
  the *contracted* fast path: it materializes the full int8 matrix,
  defeating the memory-dominance invariant (storage dominated by packed bits;
  config Method invariants). The NumPy *reference* implements the same
  arithmetic (bit-expand per row, weighted sum) without caching expanded
  matrices; it is the spec, not the performance claim.
- **DA11. A separate kernels crate for the Python face vs the Rust/server
  face (two crates, e.g. `quantibbler-kernels` and a `*-py` binding crate).**
  Rejected: it forks the kernel implementation into two sources of truth.
  Every future semantic fix would have to be applied twice and re-proven for
  parity — the drift risk is exactly what config Component decomposition
  bans ("ONE kernel implementation, TWO consumers", "Kernel behavior is
  identical on both sides by construction"). A thin-binding layer inside the
  same crate (D9) gives the same decoupling between binding and logic with
  zero drift surface.
- **DA12. NumPy-only quantized-side inference in the compressor (no PyO3
  consumption of the kernels).** Rejected: it is a double implementation —
  the Rust kernels (consumed by the server) would be a second, parallel
  implementation of behavior whose executable specification is the NumPy
  reference, with the parity gate as the only thing keeping three
  implementations honest. Config Component decomposition instead makes the
  compressor consume the Rust kernels through the PyO3 face for
  quantized-side inference (parity probes, pipeline gates), so there is one
  implementation plus one reference, not two implementations.

## 4. Open questions

- **OQ1. On-disk serialization framing** (header, version, endianness of
  multi-byte payloads beyond the uint8 bitplanes): **open** — owner:
  qk-representation lane. This change pins the in-memory layout only (D1).
- **OQ2. float64 rel-1e-12 attainability for `joint_itq` across SVD
  implementations**: **open, blocking for the Rust leg of that kernel only** —
  owner: qk-kernels implementer (this lane). Resolution path: measure at
  implementation; if unattainable, the fallback in D8 applies and requires an
  explicit `openspec/config.yaml` gate amendment (config is outside this
  change's write scope).
- **OQ3. Packed × packed binary matmul** (XNOR+popcount style): **open** —
  owner: qk-representation (owns the packed arithmetic surface for inference);
  explicitly a non-goal here (proposal).
- **OQ4. float32 opt-in accelerators** for all kernels: **open** — owner:
  qk-kernels (this lane), deferred until v0.1 profiles justify it.
- **OQ5. Whether LB-ADMM inner algebra ([NQ] Eq. 5–6) becomes a Rust kernel**:
  **open** — owner: qk-ptq lane (owns that solver); this change intentionally
  does not contract it.
- **OQ6. Whether the server's consumption of the rlib face can drop pyo3 from
  its link graph entirely** (pyo3 made an optional, target- or feature-gated
  dependency of `quantibbler-kernels`, so a pure-Rust consumer builds with no
  pyo3 code compiled in at all): **open** — owner: qk-server lane (owns the
  server crate and its dependency profile). Until resolved, the server links
  the rlib with default features per spec; pyo3 is compiled but never
  initialized on the rlib path. Resolution belongs to the serving change, not
  this one.

## 5. Informed assumptions

Each assumption states its rationale and how a wrong assumption would
surface.

- **A1. The r×r Procrustes SVD (`BᵀZ`) is the only linear-algebra primitive
  `joint_itq` needs, and r stays small (≤ ~512) in practice.** Rationale:
  [LB2] Sec 4.4 Algorithm 1 line 9 is the only decomposition inside the loop;
  ranks used by the papers are small ([LB2] Sec 4.1 Proposition 4.1 discusses
  sub-1-bit break-even at low rank). *If wrong* (very large r): performance
  suffers symmetrically on both backends (parity unaffected); surfaces as
  profile hotspots in the [LB] init path, prompting a blocked-SVD kernel
  later.
- **A2. Rotation-increment `‖R_new − R_old‖_F ≤ tol` is an acceptable
  convergence criterion for callers.** Rationale: Algorithm 1 fixes T = 50;
  the increment norm converges geometrically for Procrustes alternation
  ([LB2] Sec 4.4 reports ≤ 50 iterations to convergence), so `tol` defaults
  are chosen to reproduce the paper's practical stopping point (A2 backs
  D5/DA3). *If wrong*: `joint_itq` exits early/late relative to some
  hypothetical paper-exact rule; surfaces as parity probes
  (`itq_converged`, `itq_maxiter`) disagreeing across backends — impossible
  while both implement the same rule — or as downstream init quality drift,
  caught by the PTQ/QAT lane's reconstruction-error gates.
- **A3. Zero-valued latents are rare enough in practice (random-rotated
  factors are a.s. nonzero) that the zero → +1 tie-break never dominates
  learned structure.** Rationale: [LB2] Sec 4.4 motivates rotation precisely
  to push mass away from the "uncertainty zone near zero". *If wrong*: packed
  parity still holds (both sides use the same rule) but reconstruction error
  rises; surfaces in the representation lane's quantization-error metrics,
  not in parity.
- **A4. abi3-py311 wheels plus uv workspace install give every supported
  Python (3.11–3.13) the same extension; backend detection via ImportError is
  reliable.** Rationale: the crate is already scaffolded abi3-py311 and the
  workspace builds it via maturin (AGENTS.md tooling). *If wrong*: probes
  silently skip the Rust leg (pytest skip is loud in CI summaries); surfaces
  as a missing-extension warning in `backend()` probes, actionable by
  re-running `uv sync --all-groups --extra kernels`.
- **A5. Little-endian bit order is the right default even though [NQ] does
  not pin endianness** ("packed into integer formats (e.g., 8-bit blocks)",
  [NQ] Sec 3.1). Rationale: lane-agreed convention (this task's shared
  facts); LSB-first is the natural match for popcount-LUT decode in Rust. *If
  wrong*: any future cross-tool artifact exchange with a MSB-first tool
  breaks silently — mitigated by OQ1 (serialization framing owning a version
  header) and by the bit-order probe (`pack_layout_bitorder`) which pins the
  convention in executable tests.
- **A6. num_cpus-level parallelism is unnecessary for v0.1 kernel parity.**
  Rationale: single-threaded determinism is the simplest way to guarantee
  bit-identical accumulation (D8). *If wrong*: profiles show kernels too
  slow for 70B-scale reconstruction; surfaces as wall-clock regression in the
  PTQ lane, handled by a future parallel-schedule delta (proposal non-goal).
