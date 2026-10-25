# Tasks: qk-launch-attribution (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed (tasks 1–3.1)

Code landed: launch identity line (`rev`, resolved `out` dir, `breadcrums-<pid>.log`, configured path honored). Tasks 1.1–1.3 (RED→GREEN, tripwire pin), 2.1 (wrapper interplay unchanged), 3.1 (gate suite) all verified green on 2026-10-25:

- Lean tier: 718 passed, 16 skipped RC=0 (skip discipline holds)
- Rust legs: cargo test 24/0 RC=0, cargo test server 150 passed/10 ignored RC=0, clippy -D warnings RC=0, fmt --check RC=0
- `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` RC=0
- `npx --yes @fission-ai/openspec validate qk-launch-attribution --strict` VALID RC=0
- `pre-commit run --files <changed paths>` all Passed RC=0

## Open

- [ ] 3.2 Full tier on the interpreter × device the fire runs in (`quantibbler-ptq:rocm-official`, `/dev/kfd` + `/dev/dri`). Container launch needs the AGENTS.md team-room announcement. This is a nice-to-have — not blocking.
  - **Note (2026-10-25)**: Dev CPU wheel gives 968 passed, 14 failed (all fail-closed `resolve_device` refusals), 3 skipped. The fire-lane number is a standing record that needs a quiet box.
