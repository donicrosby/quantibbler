# Tasks: qk-flash-attention (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed (sections 1–4)

Tiled scaffold, parity harness, and flash-attention kernel face landed in multiple commits (`22bfd44`, `b8e1a18`, etc). All implementation tasks in sections 1–4 complete.

**Suite state (2026-10-25)**: 8 reds — 4 in `progress.py` (t_993f1df5) + 4 in `qk_report` (t_14977580). All attributed to sibling uncommitted edits. The change's own surface is GREEN in both feature states.

## Open

- [ ] Section 5: Landing. Prerequisites: (a) sibling reds resolved (t_993f1df5, t_14977580), (b) full gate suite green: `uv run pytest` (quote env + count), `cargo test` kernels+server in both feature states, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml`, `npx --yes @fission-ai/openspec validate qk-flash-attention --strict`, (c) scoped commit + push.
  - **Note (2026-10-25)**: `tests/test_ci_referee.py` and `tests/test_qk_report.py` are the real files.
