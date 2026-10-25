# Proposal: qk-nq-recreation-bar

## Why

The user's standard for this effort is explicit: the project should be
able to recreate the NanoQuant paper's results ([NQ]). As sec-eng pointed
out, the `ok <= 4` band is the *operational* landing gate — anchored on
[LB2]'s functional envelope — and is not the finish line. A finish line
defined after results arrive is a landing site chosen after touchdown;
the acceptance criteria must be pinned now, before the climb gets close,
exactly the discipline the band-recalibration obligation already follows.

The bar must also be honest about protocol: [NQ]'s published rows are
WikiText-2 at sequence length 2048 ([NQ] Sec. 4.2, 128 calibration
samples); this repo's driver currently evaluates at 512. A finish line
that silently accepts the seq-512 numbers is finish-line drift.

## What Changes

- A new requirement in the `qk-ptq` capability pins the recreation bar
  against [NQ] Table 2's own textually-present rows for Qwen3-0.6B:
  baseline 12.66; NanoQuant 27.56 @1.00 bits, 33.79 @0.80 bits,
  52.94 @0.55 bits; with tolerance = 1.15x the row value (ALL edges
  arithmetic asserted in a suite test). The artifact SHALL have
  byte-derived `effective_bpw` at or below the row's bit budget — no
  buying PPL with extra bits.
- Protocol reconciliation SHALL be part of the acceptance run: the bar is
  met on [NQ]'s eval protocol (WT2, seq 2048), never on the seq-512
  surrogate. Accepting a shorter-context number requires an
  equivalence argument recorded in the run ledger, not assumed.
- The acceptance run is a documented, human-gated release criterion
  (report + ledger), distinct from the per-run verdict gate: the gate
  says "not dead", the bar says "recreated the paper".

## Impact

- **Affected specs**: `qk-ptq` (one ADDED requirement, delta under
  `changes/qk-nq-recreation-bar/specs/qk-ptq/spec.md`; distinct surface
  from the operational gate and probe changes — no MODIFIED/REMOVED).
- **Affected code**: nothing immediately; the requirement's first
  implementable unit is the suite test asserting the row constants and
  tolerance arithmetic, plus a run-ledger format when an acceptance run
  is first scheduled.
- **Not invariant-touching**: no format, packing, kernels, or server
  change.

## Non-goals

- No change to the operational band or its edges (the gate stays what it
  is; this bar sits above it).
- No attempt to reproduce other paper tables (Table 5 init ablation,
  Table 6 Qwen3-8B stage ablation) — Table 6 rows (init-only collapse
  206.03 vs full pipeline 12.47) are recorded in design.md as diagnostic
  context, not acceptance criteria.
- No benchmark-adjacent evals (zero-shot suites) in this change.

## Correction register

The planner's earlier room statement "ok <= 4 is the NQ-recreation
target in normalized form" was wrong on provenance and is retracted:
the 4.0 edge derives from [LB2] Table 1 (LittleBit-2 @0.1 bpp on
Llama-3-8B), not from [NQ]. This change is the corrected, separately-
owned finish line.
