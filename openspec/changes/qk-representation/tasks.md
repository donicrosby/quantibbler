# Tasks: qk-representation

Conventions: tests are written before implementation, per capability, and
reference scenario IDs from `specs/qk-representation/spec.md` (SF-x.y).
Implementation lands on the NumPy reference path in the `quantibbler`
Python package (config Architecture). Commands run from the repo root.

## 1. Tests first — qpk1 container

- [ ] 1.1 Create `tests/test_representation.py` with one test per scenario
      ID, named to reference it:
      `test_sf1_3_known_layer_exact_bytes` (SF-1.1, SF-1.3),
      `test_sf1_2_dtype_tag_selects_precision` (SF-1.2),
      `test_sf1_4_malformed_headers_rejected` (SF-1.4),
      `test_sf2_1_byte_identical_roundtrip` (SF-2.1),
      `test_sf2_2_factors_scales_recovered` (SF-2.2),
      `test_sf3_1_row_major_layout` (SF-3.1),
      `test_sf3_2_sign_mapping` (SF-3.2),
      `test_sf4_1_bpw_from_stream_length` (SF-4.1),
      `test_sf4_2_bpw_matches_formula` (SF-4.2),
      `test_sf4_3_sub1bpw_at_target_rank` (SF-4.3),
      `test_sf5_1_rank_hits_target` (SF-5.1),
      `test_sf5_2_formula_reduces_to_papers` (SF-5.2),
      `test_sf6_1_stream_extent_one_layer` (SF-6.1).
      Command: `uv run pytest tests/test_representation.py --collect-only -q`.
      Expected: 13 tests collected, none errored. (If collection fails on a
      missing module, create the stub `src/quantibbler/qpk1.py` with
      signatures raising `NotImplementedError`, then re-run.)

- [ ] 1.2 Populate the tests with assertions derived from the spec
      scenarios: exact 24-byte header content for a known input; exact
      serialized size `24 + 2*(dout+din) + ceil(dout*r/8) + ceil(din*r/8)`;
      byte-identity round-trips; canonical packing of a hand-computed 3x5
      example; BPW computed from `len()` of the returned bytes; the
      4096x11008 r=100 shape measured at BPW < 1.0.
      Command: `uv run pytest tests/test_representation.py -q`.
      Expected: all fail with `NotImplementedError` (RED), zero passes.

## 2. Implementation — NumPy reference path

- [ ] 2.1 Implement `src/quantibbler/qpk1.py`: constants (`QPK1_MAGIC =
      b"QPK1"`, `REPRESENTATION_VERSION = 1`, dtype tags 1/2),
      `serialize(...)`, `deserialize(...)`, `bpw_from_bytes(...)`,
      `rank_for_target_bpw(...)`, `expected_size(...)`, per design.md
      D1–D9. Command:
      `uv run python -c "import quantibbler.qpk1"`. Expected: exit 0.
- [ ] 2.2 Make the RED tests pass. Command:
      `uv run pytest tests/test_representation.py -q`. Expected: 13 passed
      (GREEN).
- [ ] 2.3 Randomized-shape round-trip sweep (seeded): shapes from 1x1
      through 4096x11008, r from 1 up, float16 and float32, odd and even
      r; assert byte identity, exact factor/scale recovery, and
      `expected_size(...) == len(serialize(...))` per case. Command:
      `uv run pytest tests/test_representation.py -q`. Expected: all pass.

## 3. Spec-compliance spot checks

- [ ] 3.1 Hand-computed packing vector: serialize a fixed (3, 5, 7) case
      with known signs and assert the exact bitplane bytes computed by
      hand per SF-3.1/SF-3.2. Command:
      `uv run pytest tests/test_representation.py::test_sf3_1_row_major_layout tests/test_representation.py::test_sf3_2_sign_mapping -q`.
      Expected: pass.
- [ ] 3.2 Reject corpus: malformed streams (bad magic; version 0 and 2;
      dtype tag 0/3/0xFFFFFFFF; r=0; r > min(dout,din); truncated at every
      header offset 0..23 and mid-payload; one trailing byte; dims at the
      u32 ceiling) each raising the documented error type. Command:
      `uv run pytest tests/test_representation.py -k sf1_4 -q`.
      Expected: pass.
- [ ] 3.3 BPW agreement: for a sweep of (dout, din, r), assert
      `bpw_from_bytes(serialize(...))` equals the D6 analytical `BPW(r)`
      exactly (same integer byte count), and `rank_for_target_bpw` output
      re-serializes to BPW within one rank-step of target (SF-5.1).
      Command: `uv run pytest tests/test_representation.py -k "sf4 or sf5" -q`.
      Expected: pass.

## 4. Gate suite (final)

- [ ] 4.1 Run the full gate suite from config.yaml and record results:
      - `uv run pytest` — record pass/fail counts.
      - `cargo test` (in `crates/quantibbler-kernels/`) — record result.
      - `cargo clippy --all-targets -D warnings` (same dir) — record result.
      - `cargo fmt --check` (same dir) — record result.
      - `uv run maturin build` — record result.
      Append the outcomes (command, exit status, summary line) under a
      `## Gate run` heading added to this file. Command:
      `uv run pytest && (cd crates/quantibbler-kernels && cargo test && cargo clippy --all-targets -D warnings && cargo fmt --check) && uv run maturin build`.
      Expected: every command exits 0; a `## Gate run` block exists in
      tasks.md with per-command outcomes, including the statement that
      BPW measured-from-bytes passed (SF-4.1–SF-4.3).
