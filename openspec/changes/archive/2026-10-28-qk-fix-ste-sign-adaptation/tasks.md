# Tasks: qk-fix-ste-sign-adaptation (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

ALL tasks complete. Code landed at `059ee9b`: per-step flip-count instrumentation in `_tune_latent_ste` (both factors, per linear, all 7 scoped linears), flip acceptance against block-reconstruction objective, toy block engineered with ≥1 sign crossing, stage-trace probe rehearsed, all reviews (bug-hunter, sec-eng) PASS, orchestrator validate VALID, gate suite recorded (lean 412 passed/16 skipped RC=0; cargo test 24/0 RC=0; clippy/fmt/maturin RC=0; pre-commit RC=0).

## Open

- [ ] Re-fire with the fixed stage on a real run. The flip counts must land in a run on record before this change can be archived.
  - **Note (2026-10-25)**: This is a fire-class condition requiring @user GO. Zero code work remains. The change stays alive as the tracking vehicle for the re-fire. Once a real run records nonzero flip counts, archive immediately.
