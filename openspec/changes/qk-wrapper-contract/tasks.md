# qk-wrapper-contract — tasks

## Phase 1 — Spec authoring

- [x] (planner) Spec delta: 7 requirements extracted verbatim from `qk-nanoquant-alignment-lane0`. Verify: `npx --yes @fission-ai/openspec validate qk-wrapper-contract --strict` green.

## Phase 2 — Gates

- [x] (planner) No code changes to gate — this is an as-built spec only. The implementation's gate suite (`pytest tests/test_run_ptq_wrapper.py`, shellcheck `tools/run-ptq-container.sh`) is covered by the existing test discipline.
