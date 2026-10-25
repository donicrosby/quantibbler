# Tasks: qk-fix-tunefp-teacher (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

ALL tasks complete. Code landed at `6c09948`: teacher capture per [NQ] Algorithm 1 (pristine-pinned weights, memory-only cache, source fingerprint + capture index recorded), Figure 3 placement (TuneFP once per block after q/k/v factorization), vacuous-teacher guard (two-norm event, fail-closed, no trivial-zero carve-out), faithful-run stage map, all reviews (bug-hunter, sec-eng) PASS, orchestrator validate VALID.

**Cross-lane finding (sec-eng 2026-10-25)**: lane-2 commit `ec906a7` persists fp-stream teacher output `y_b` into block checkpoints — the A2 "never persisted" wording no longer describes HEAD bytes. Routed to planner for reconciliation.

## Open

- [ ] Re-fire with the fixed TuneFP teacher on a real run. The run record must show a nonzero TuneFP effect before this change can be archived.
  - **Note (2026-10-25)**: This is a fire-class condition requiring @user GO. Zero code work remains. The change stays alive as the tracking vehicle for the re-fire. Once a real run records nonzero TuneFP effect, archive immediately.

## Deferred (planner)

- [ ] TuneFP spec-graph contradiction (orchestrator ruling, harvest comment): route to this change's planner work — author the reconciliation citing the t_416ad06f review resolutions. NOT a blocker for audit execution.
