# Tasks: qk-serialization

Change dir: `openspec/changes/qk-serialization/`. Foundation card `t_0a8f5936`
(coder) landed the dependency, the models and their tests; producer adoption is
`t_b4a1383d` (report/driver, section 2 below) and `t_f0872f45` (journal/probe).

Environment note: `UV_PROJECT_ENVIRONMENT=/opt/venv` is exported on this box, so
a bare `uv run` in a worktree re-points the SHARED venv at that worktree for
every co-tenant lane. The rows below quote the worktree's own interpreter
(`.venv/bin/python`, per the worktree law's per-worktree `.venv`); `uv run
--no-sync` is the right form only where the shared venv is the intended one.

- [x] 1.1 (coder) Add `pydantic>=2.14,<3` to `[project].dependencies` after
      verifying the live registry floor, and refresh `uv.lock`.
      Verify: `uv lock` resolves `pydantic 2.14.1` / `pydantic-core 2.50.1`
      from wheels; `uv sync` exits 0. (The `--all-groups --extra torch --extra
      kernels --extra hf` form this row carried is dead: `qk-dependency-floor`
      made torch/transformers/tokenizers/safetensors/pyarrow product
      dependencies, so there is no extra left to name.)
- [x] 1.2 (coder) Author `src/quantibbler/serialization.py`: one model per
      surface, `schema_version` + `ArtifactSchemaError` refusal, `extra` chosen
      per surface. Verify: a torch-free process imports
      `quantibbler.serialization` (`'torch' not in sys.modules`).
- [x] 1.3 (coder) Roundtrip + reject tests in `tests/test_serialization.py`,
      named `test_scn_<scenario_id>` against this change's spec delta.
      Verify: `pytest tests/test_serialization.py -q` green.
- [x] 1.4 (coder) Strict-validate this change: `npx --yes
      @fission-ai/openspec validate qk-serialization --strict` → valid, RC=0.
- [x] 1.5 (coder) Run the full gate suite and record results: `pre-commit run
      --all-files`; the full suite green; `cargo test` in
      `crates/quantibbler-kernels/`; `cargo clippy --all-targets -- -D
      warnings`; `cargo fmt --check`; `maturin build --manifest-path
      crates/quantibbler-kernels/Cargo.toml`.

## Adoption A — report/driver surface (card `t_b4a1383d`)

- [x] 2.1 (coder) `qk_report.dump_report` validates the report through
      `QuantizeReport` before serializing, stamps `schema_version`, keeps the
      writer's own keys/values/order, and falls back to the minimal failed
      report when validation or serialization fails. The model validates rather
      than formats because the report's KEY SET is its own contract
      (`report_single_shape_both_paths`); a model dump would reorder it — see
      `design.md`, "Implementation crystallization" (2026-10-25).
      Verify: `tests/test_qk_report.py::test_scn_report_payload_*` green; a
      completion, abort and stage-failure payload each validate against the
      model.
- [x] 2.2 (coder) Report-consuming tests read their payloads back through the
      model: `tests/test_qk_report.py`, `tests/test_qk_report_abort.py`,
      `tests/test_qk_report_stages.py`.
- [x] 2.3 (coder) Delta the writer's contract into this change's spec delta:
      scenarios `report_payload_validates_against_the_model`,
      `report_payload_keeps_the_writers_keys_and_order`,
      `report_payload_model_rejection_falls_back_to_the_failed_stub`.
      Verify: `npx --yes @fission-ai/openspec validate qk-serialization --strict`
      RC=0.
- [ ] 2.4 (coder) Acceptance-ledger rows — NOT WIRED, and not wireable on this
      card. The ledger has no producer and no reader in the pipeline:
      `docs/acceptance-ledger.jsonl` does not exist, and `qk-nq-recreation-bar`'s
      `ledger-format.md` (archived) rule 2 states nothing in the pipeline writes
      or reads it — a row is authored by a human at gate time, and that note
      defers the ledger's tooling to the first acceptance run, an @user-gated
      fire. The row model and its `qk_bar.BAR_ROWS`-derived vocabulary landed
      with the foundation; `row_meets_bar` keeps its zero-production-caller
      status (it is the ledger's predicate, never wired into `verdict()`).
