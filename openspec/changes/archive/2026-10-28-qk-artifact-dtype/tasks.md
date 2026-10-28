# Tasks: qk-artifact-dtype

Implementation rides the coder card for this change (program brief
t_71e38da2, or the fresh coder card the orchestrator cuts after spec
review — the card body points at this change dir, per the config tasks
rule). Work runs in an isolated worktree; leaves never commit; the
orchestrator stages scoped paths and lands. `openspec validate
qk-artifact-dtype --strict` is re-run after EVERY edit to this dir.

Test names reference scenario IDs (config rule); RED means the test was
observed failing against the pre-change tree before the implementation
landed GREEN.

## 1. Spec integrity

- [ ] 1.1 `npx --yes @fission-ai/openspec validate qk-artifact-dtype
      --strict` → `Change 'qk-artifact-dtype' is valid` (RC=0); every
      ADDED requirement body ≤ 500 chars.
- [ ] 1.2 Deltas exist for the two owning capabilities
      (`specs/qk-server`, `specs/qk-ptq`); design.md carries D1 (the
      extension-vs-bump decision with its byte-layout evidence), D2-D12,
      the discarded alternatives, and the D9 tolerance derivation.

## 2. Rust loader: declared dtype, grammar, agreement

- [ ] 2.1 RED: against the pre-change loader, the new-form refusal battery
      fails as specified — `@bf16` is refused with the fp32-only message
      and a manifest `[sidecar] dtype = "bf16"` is ignored (serde drops
      the unknown key), so the bf16 fixture cannot load.
- [ ] 2.2 GREEN: `RemainderDtype` enum, `[sidecar] dtype` parsing, the
      extended `weight_tying` grammar, the agreement rule, and the
      absence-reads-fp32 legacy rule (D1, D2).
      Tests (crates/quantibbler-qpk1): `loads_declared_bf16_sidecar_rd1_1`,
      `absent_dtype_reads_fp32_rd1_2`,
      `rejects_suffix_field_mismatch_rd1_3`,
      `rejects_unknown_dtype_token_rd1_4`,
      `unset_tying_bf16_sidecar_rd1_5`.
- [ ] 2.3 GREEN: declared-itemsize reading (D5) — bf16 at 2 bytes, fp32 at
      4 — with the exact-length stat gate refusing both mismatch
      directions. Tests: `loads_bf16_fixture_exact_values_rd2_1`,
      `rejects_bf16_declared_wrong_len_rd2_2`,
      `fp32_fixture_unchanged_rd2_3` (the committed fp32 fixture's loaded
      values are byte-identical pre/post change).
      Evidence: `cargo test --locked -p quantibbler-qpk1` → all pass.

## 3. Host writer: `--remainder-dtype` + shared cast helper

- [ ] 3.1 RED: against the pre-change exporter, the bf16-export fixture
      tests fail (no flag exists; no bf16 sidecar is writable).
- [ ] 3.2 GREEN: `--remainder-dtype {fp32,bf16}` (default fp32) plumbed
      like `--scale-dtype`; ONE shared RNE cast helper with the post-cast
      finiteness gate (D4) used by the export path (and later the repack
      utility); bf16 filenames, `[sidecar] dtype`, and `@bf16` suffix all
      written to agree (D6); export self-check at the declared itemsize.
      Tests (tests/ptq): `fp32_default_export_byte_identical_rdw1_1`
      (remainder file bytes equal the pre-change writer's pinned bytes),
      `bf16_export_loads_in_rust_rdw1_2` (cross-cites qk-server RD-2.1),
      `bf16_path_refuses_nonfinite_rdw1_3`,
      `export_selfcheck_itemsize_rdw1_4`.

## 4. Report fidelity fields

- [ ] 4.1 RED: against the pre-change report writer, the bf16 report test
      fails (no `remainder_dtype` field exists).
- [ ] 4.2 GREEN: `provenance()` gains `remainder_dtype` from the run's
      arguments; `regime_artifact` and `dtype_note` name it (D8); never a
      hardcoded literal. Tests:
      `report_names_remainder_dtype_from_args_rdw2_1`,
      `report_fp32_default_not_hardcoded_rdw2_2`.
      Reconcile against `qk-serialization`'s landed report models (the
      field joins whatever pydantic surface that change shipped).

## 5. Repack utility

- [ ] 5.1 RED: against the pre-change tree, the repack tests fail (the
      tool does not exist).
- [ ] 5.2 GREEN: `tools/repack_remainder.py` per D7 — new-directory
      output, byte-identical non-remainder copies, shared cast helper,
      rewritten manifests, `[repack]` provenance, determinism, named
      refusals. Tests (tests/ptq):
      `repack_roundtrip_loads_rdp1_1`, `repack_refuses_already_bf16_rdp1_2`,
      `repack_refuses_nonfinite_rdp1_3`,
      `repack_streams_byte_identical_rdp1_4`,
      `repack_deterministic_rdp1_5`,
      `cast_elementwise_half_ulp_rdp2_1` (the 2^-8|x| + 2^-134 bound).

## 6. Fixtures, parity, decode tolerance

- [ ] 6.1 GREEN: `tools/gen_server_fixtures.py` emits the deterministic
      bf16 twin; committed bytes load in the Rust loader with pinned f64
      values asserted in BOTH faces (D10); Python-written bf16 bytes equal
      the committed fixture bytes (byte-exact cross-face exchange — legal
      here: remainder sidecars are raw IEEE floats, no bitplane packing).
- [ ] 6.2 GREEN: server e2e `bf16_twin_decode_within_tolerance_rd3_1` —
      decode logits vs the fp32 original satisfy the D9 bound
      (2^-4 · max(1, max|logit|) on the fixture) computed from the loaded
      tensors, and the max delta is NONZERO.
      Evidence: `cargo test --locked -p quantibbler-server` → all pass.
- [ ] 6.3 Acceptance repack: produce the loadable bf16 variant of the v6
      artifact (if its directory is stable and available at implementation
      time — orchestrator/@user call) OR of the committed fixture (the
      brief's standing alternative). Logit comparison uses the general D9
      formula computed from that artifact's tensors.

## 7. Full gate suite and landing

- [ ] 7.1 `uv run pytest` (full suite, environment extras recorded in the
      report — quote the count WITH its environment) → green.
- [ ] 7.2 `cargo test` in `crates/quantibbler-kernels`; `cargo test
      --locked` in `crates/quantibbler-qpk1` and `crates/quantibbler-server`;
      `cargo clippy --all-targets -- -D warnings`; `cargo fmt --all --
      --check` → green.
- [ ] 7.3 `uv run maturin build --manifest-path crates/quantibbler-kernels/
      Cargo.toml` → builds; `npx --yes @fission-ai/openspec validate
      qk-artifact-dtype --strict` → valid on the final tree.
- [ ] 7.4 Scoped staging (explicit paths only — sibling lanes are live in
      shared checkouts), no trunk commits by leaves; the landing commit
      carries the gate evidence in its body; a design.md crystallization
      entry folds any implementation-resolved design-silent choices in the
      same commit.
