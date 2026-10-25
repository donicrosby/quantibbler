# Tasks: qk-divergence-guards (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed (tasks 1–4)

All code landed (tau default, divergence gate, thread pin measurement). Tasks 1.1–4.1 verified complete with evidence:

- **1.1–1.2**: Block-0 probe: same species break confirmed (raw-stat tails), r8 is milder manifestation. Recorded in design.md.
- **2.1–2.3**: tau default (q95) implemented in `robust_diag`; RED→GREEN observed; no healthy-layer regression.
- **3.1–3.3**: Post-balance divergence gate in `block_recon.reconstruct_block`; RED→GREEN observed; end-to-end tripwire proof (`test_divergence_tripwire.py`).
- **4.1**: Thread pin: 16T median 12.8s vs 24T 13.8s; KEEP 16.

## Open

- [ ] 5.1 Full r137 rerun through the container driver with the guards live. Expect either a clean artifact (ppl bounded well below baseline ×2, sane scales across all 28 blocks) or a fast named abort. Record the report JSON.
  - **Note (2026-10-25)**: originally killed by @user (t_03c80f1d). This task row stays as a placeholder — the operational fire is not currently scheduled.
- [ ] 5.2 Full gate suite: `uv run pytest` (quote env + count), `cargo test --manifest-path crates/quantibbler-kernels/Cargo.toml --all-targets`, `cargo clippy --manifest-path crates/quantibbler-kernels/Cargo.toml --all-targets -- -D warnings`, `cargo fmt --manifest-path crates/quantibbler-kernels/Cargo.toml --all -- --check`, `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml`. Record results.
  - **Note (2026-10-25)**: `tests/test_ci_referee.py` and `tests/test_qk_report.py` are the real files (not `tests/ci/` or `tests/qk_report/` directories).
