# Tasks: qk-serialization (foundation)

Change dir: `openspec/changes/qk-serialization/`. Board card: `t_0a8f5936`
(coder). No producer wiring in this change — adoption is `t_b4a1383d`
(report/driver) and `t_f0872f45` (journal/probe).

- [ ] 1.1 (coder) Add `pydantic>=2.14,<3` to `[project].dependencies` after
      verifying the live registry floor, and refresh `uv.lock`.
      Verify: `uv lock` resolves `pydantic 2.14.1` / `pydantic-core 2.50.1`
      from wheels; `uv sync --all-groups --extra torch --extra kernels --extra hf`
      exits 0.
- [ ] 1.2 (coder) Author `src/quantibbler/serialization.py`: one model per
      surface, `schema_version` + `ArtifactSchemaError` refusal, `extra` chosen
      per surface. Verify: `uv run --no-sync python -c "import
      quantibbler.serialization"` exits 0 in a lean (no-torch) environment.
- [ ] 1.3 (coder) Roundtrip + reject tests in `tests/test_serialization.py`,
      named `test_scn_<scenario_id>` against this change's spec delta.
      Verify: `uv run --no-sync pytest tests/test_serialization.py -q` green.
- [ ] 1.4 (coder) Strict-validate this change: `npx --yes
      @fission-ai/openspec validate qk-serialization --strict` → valid, RC=0.
- [ ] 1.5 (coder) Run the full gate suite and record results: `pre-commit run
      --all-files`; `uv run --no-sync pytest`; `cargo test` in
      `crates/quantibbler-kernels/`; `cargo clippy --all-targets -- -D
      warnings`; `cargo fmt --check`; `uv run --no-sync maturin build
      --manifest-path crates/quantibbler-kernels/Cargo.toml`.
