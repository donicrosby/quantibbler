# Tasks: qk-progress-instrumentation (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed

Core `ProgressReporter` landed (`50e3c73`, `cae79b9`, etc). Tasks 1.1–1.4 (class + cadence + ETA + dual-sink dispatch) [x]. Tasks 2.1–2.2 (progress constants) [x]. Tasks 3.2–3.3 (lb_admm and block_recon integration) [x]. Task 6 (tests: cadence, ETA, dual-sink, mock loop, forensic tail, stale constants) [x].

## Open

- [ ] 1.5 Heartbeat: `MemTelemetry` host was retired by `e5c6939` (Lane 1 rewrite). `ProgressReporter.heartbeat()` exists but has zero production callers at HEAD. Needs a timer thread.
- [ ] 3.1 Wire into `run_arms`'s statistics phase — **moot**: `run_arms` and lane0 harness retired by `e5c6939`; replacement `calibration.collect_stats` is a one-shot pass that does not need a progress reporter. Close with disposition note.
- [ ] 4.1–4.2 Telemetry CSV integration: `MemTelemetry` host retired. The `jsonl_sink` and `attach_telemetry` functions exist but have zero production callers at HEAD. Needs a telemetry collector that registers the sink.
- [ ] 5.1 PTQ breadcrums: `breadcrumb_sink` and `attach_breadcrumbs` exist but have zero production callers at HEAD. Wire-up: call `attach_breadcrumbs` at reporter construction time in the three existing loops (admm, tunefp, ste).

## Notes (2026-10-25)

The deferred items all share one dependency: a runtime host (telemetry collector, breadcrumb caller) that doesn't exist. The core class and its unit tests are complete. When a telemetry collector or breadcrumb host is re-introduced, the wire-up is a one-line call at construction time per loop.
