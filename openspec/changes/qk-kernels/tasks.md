# tasks.md — qk-kernels

Every task is independently verifiable with a command and expected outcome.
Tests precede implementation per capability; test names reference scenario
IDs from `specs/qk-kernels/spec.md`. Commands run from repo root
(`/opt/data/quantibbler`) unless noted; Rust commands run from
`crates/quantibbler-kernels/` as noted.

## 1. Fixtures and probes (NumPy reference first)

- [x] 1.1 Scenario-ID parity test file `tests/test_kernels_parity.py` with
      probe tests named for scenario IDs (see spec: `pack_roundtrip_small`,
      `pack_roundtrip_exact8`, `pack_roundtrip_padded`, `pack_roundtrip_empty`,
      `pack_layout_bitorder`, `sign_zeros`, `sign_extremes`, `matvec_padded`,
      `matmul_wide`, `itq_converged`, `itq_maxiter`,
      `itq_deterministic_seed`), each running the kernel under test through
      both `backend="numpy-reference"` and `backend="rust"` legs. Verify:
      `uv run pytest tests/test_kernels_parity.py -v` — all probes pass with
      the NumPy leg executing and the Rust leg skipped (exit 0).
- [x] 1.2 Bit-order layout probe (`pack_layout_bitorder`) asserts: a row of
      eight +1s followed by rows with a single +1 at element position k
      produce byte j = k // 8 with bit k % 8 set (LSB-first), pad bits zero.
      Verify: `uv run pytest tests/test_kernels_parity.py -k pack_layout_bitorder -v`
      — passes (exit 0).
- [x] 1.3 Fixture generator `tools/gen_kernel_fixtures.py`: emits JSON
      fixtures (inputs, NumPy-reference outputs, scenario ID, dtype) for the
      scenario families into `crates/quantibbler-kernels/tests/fixtures/`.
      Verify: `uv run python tools/gen_kernel_fixtures.py` — writes one JSON
      per scenario family; `git status --porcelain crates/` shows only new
      fixture files (exit 0).
- [x] 1.4 Rust-side probe harness `crates/quantibbler-kernels/tests/parity.rs`
      loading the committed fixtures by scenario ID, asserting float64 rel
      1e-12 and byte-identical packed output. Verify (from
      `crates/quantibbler-kernels/`): `cargo test parity_ -- --nocase` shows
      scenario-named tests compiling and failing (RED) — expected before
      implementation; exit code 101 with "not implemented" panics is the
      expected outcome at this stage.

## 2. Packing kernels (pack_bits / unpack_bits)

- [x] 2.1 Implement NumPy reference `_numpy_pack_bits` /
      `_numpy_unpack_bits` in `src/quantibbler/kernels.py` per spec
      (uint8 bitplanes, row-major, LSB-first, +1→1, −1→0, zero pads,
      `ValueError` off ±1, empty support). Verify:
      `uv run pytest tests/test_kernels_parity.py -k "pack_" -v` — NumPy-leg
      probes pass (exit 0).
- [x] 2.2 Implement Rust `pack_bits` / `unpack_bits` (`#[pyfunction]`) in
      `crates/quantibbler-kernels/src/lib.rs`; register in the pymodule.
      Verify: `uv run maturin develop && uv run pytest tests/test_kernels_parity.py -k "pack_" -v`
      — both legs pass (exit 0).
- [x] 2.3 Rust unit tests for pack layout (bit positions, pad-zero, error on
      off-values) mirroring the scenario IDs. Verify (from
      `crates/quantibbler-kernels/`): `cargo test` — passes (exit 0).

## 3. Sign estimation (sign_estimate)

- [x] 3.1 Implement `_numpy_sign_estimate` (int8 out, >0→+1, ≤0→−1 with
      exact-zero→+1 documented tie-break). Verify:
      `uv run pytest tests/test_kernels_parity.py -k "sign_" -v` — passes
      (exit 0).
- [x] 3.2 Implement Rust `sign_estimate` with identical zero-handling.
      Verify: `uv run maturin develop && uv run pytest tests/test_kernels_parity.py -k "sign_" -v`
      — both legs pass (exit 0).

## 4. Packed binary matvec / matmul

- [x] 4.1 Implement `_numpy_packed_matvec` / `_numpy_packed_matmul`:
      arithmetic-equivalent bit-expand per row and weighted sum, float64
      accumulation in row-element order, no expanded-matrix caching. Verify:
      `uv run pytest tests/test_kernels_parity.py -k "matvec or matmul" -v`
      — passes (exit 0).
- [x] 4.2 Implement Rust `packed_matvec` / `packed_matmul` (Popcount-LUT
      decode, identical left-to-right accumulation order so float results are
      bit-identical). Verify: `uv run maturin develop && uv run pytest tests/test_kernels_parity.py -k "matvec or matmul" -v`
      — both legs pass, and a debug assertion comparing NumPy-leg bytes to
      Rust-leg bytes holds (exit 0).

## 5. Joint-ITQ solver (joint_itq)

- [x] 5.1 Implement `_numpy_joint_itq` per spec: Z = [Û; V̂], seeded
      orthogonal R₀, alternating `B ← sign(ZR)` / `R ← ΨΦᵀ` from
      `svd(BᵀZ)`, stopping on `‖R_new − R_old‖_F ≤ tol` or max_iter;
      returns (u_t, v_t, n_iter, converged), float64. Verify:
      `uv run pytest tests/test_kernels_parity.py -k "itq_" -v` — passes
      (exit 0).
- [x] 5.2 Implement Rust `joint_itq` with pure-Rust SVD (nalgebra,
      dependency floor verified against crates.io per config freshness rule,
      pin within one minor) and seeded RNG bit-matching the NumPy leg's
      R₀. Verify: `uv run maturin develop && uv run pytest tests/test_kernels_parity.py -k "itq_" -v`
      — both legs pass at rel 1e-12 (exit 0). If rel 1e-12 is unattainable
      across SVD implementations, STOP and apply design OQ2's fallback path
      (config amendment first) — do not silently relax the tolerance.
- [x] 5.3 Convergence-behavior probes: `itq_converged` (n_iter < max_iter,
      converged True), `itq_maxiter` (tight tol → converged False, exact
      max_iter), `itq_deterministic_seed` (same seed → byte-identical packed
      sign factors; different seed → different rotation fixed point is
      permitted). Verify: `uv run pytest tests/test_kernels_parity.py -k itq_ -v`
      — passes on both legs (exit 0).

## 6. Dispatch semantics

- [x] 6.1 Extend `src/quantibbler/kernels.py`: per-call `backend=` kwarg on
      every public kernel (`"rust"` | `"numpy-reference"`), resolution order
      kwarg → `QUANTIBBLER_BACKEND` → auto; auto falls back to NumPy without
      the extension; explicit rust with extension absent raises `RuntimeError`
      naming the missing module and the `kernels` extra; every kernel exists
      with identical signatures on both backends. Tests
      `tests/test_kernels_dispatch.py`: `dispatch_default_auto`,
      `dispatch_env_override`, `dispatch_explicit_rust_missing_raises`
      (monkeypatched module-absence), `dispatch_explicit_numpy_always_works`.
      Verify: `uv run pytest tests/test_kernels_dispatch.py -v` — passes
      (exit 0).
- [x] 6.2 Document zero-handling (+1 for exact zeros) and the packed layout
      in the `quantibbler.kernels` module docstring. Verify:
      `uv run python -c "import quantibbler.kernels as k; assert 'zero' in k.__doc__.lower()"`
      — exit 0.

## 7. Fixture parity on the Rust side

- [x] 7.1 Regenerate fixtures against final NumPy reference:
      `uv run python tools/gen_kernel_fixtures.py`. Verify:
      `git status --porcelain crates/quantibbler-kernels/tests/fixtures/`
      — either empty or reviewable diffs only (exit 0).
- [x] 7.2 Rust fixture parity suite green:
      from `crates/quantibbler-kernels/`, `cargo test` — all
      scenario-ID-named parity tests pass at rel 1e-12 / byte-identical
      (exit 0).

## 8. Dual-face architecture (rlib + PyO3 faces from one core)

- [x] 8.1 Dual-face probe tests first (tests precede implementation), named
      for their scenario IDs: `rlib_face_usable_without_python`,
      `pyo3_face_thin_binding`, `single_core_implementation` (structure tests
      over the crate layout) and `face_parity_packed_bitidentical`,
      `face_parity_float_rel_1e-12`, `face_parity_extends_numpy_chain`
      (fixture-driven comparisons of the rlib face against the NumPy-reference
      and PyO3-face fixture records). Verify (from
      `crates/quantibbler-kernels/`): `cargo test` — the new scenario-named
      tests compile and fail (RED: core layer and fixture fields do not exist
      yet); exit 101 or failing tests is the expected outcome at this stage.
- [x] 8.2 Extract the pyo3-free core layer (`src/core/`): plain-Rust
      implementations of all six v0.1 kernels; `#[pyfunction]` wrappers
      reduced to boundary conversion + error mapping (design D9); the rlib
      face is the core's public API. Verify (from
      `crates/quantibbler-kernels/`): `cargo test` — all tests including 8.1's
      go green (exit 0); and `bash -c '! grep -rn pyo3 src/core/'` — no pyo3
      references in the core layer (exit 0).
- [x] 8.3 Record the PyO3 face in fixtures: extend
      `tools/gen_kernel_fixtures.py` to run each scenario through the built
      extension (the PyO3 face) and store its outputs alongside the
      NumPy-reference outputs; regenerate. Verify:
      `uv run maturin develop && uv run python tools/gen_kernel_fixtures.py && git status --porcelain crates/quantibbler-kernels/tests/fixtures/`
      — fixtures updated with PyO3-face records as a reviewable diff (exit 0).
- [x] 8.4 Crate-type coexistence + opt-in extension-module checks. Verify:
      from `crates/quantibbler-kernels/`, `cargo build` — both rlib and cdylib
      artifacts produced (exit 0); `grep -q '^default = \[\]' Cargo.toml` —
      exit 0; from repo root, `uv run maturin develop && uv run python -c "import quantibbler_kernels"`
      — import succeeds (exit 0).
- [x] 8.5 Face-parity suite green end-to-end (both faces, all six kernels).
      Verify (from `crates/quantibbler-kernels/`):
      `cargo test face_parity` — `face_parity_packed_bitidentical`,
      `face_parity_float_rel_1e-12`, `face_parity_extends_numpy_chain` pass at
      byte-identical packed / float64 rel 1e-12 (exit 0).

## 9. Full gate suite (final task)

- [x] 9.1 Run the full gate suite and record results:
      1. `uv run pytest` — PASS/FAIL: ____
      2. `cargo test` (in `crates/quantibbler-kernels/`) — PASS/FAIL: ____
      3. `cargo clippy --all-targets -- -D warnings` (same dir) — PASS/FAIL: ____
      4. `cargo fmt --check` (same dir) — PASS/FAIL: ____
      5. `uv run maturin build` — PASS/FAIL: ____
      Verify: all five PASS recorded above with command versions/dates in the
      change PR description; this change does not land with any gate failing.
