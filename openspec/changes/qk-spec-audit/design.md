# Design: qk-spec-audit — the reconciled verdict document

This file IS the audit's design of record. It was authored by the planner as
the `qk-spec-audit: verdicts` deliverable (`t_b262f90d`) and is landed here
verbatim so the verdicts live in the repo rather than in a card attachment.
Execution status for each verdict is recorded in `tasks.md`; a verdict that
could not be reproduced is escalated (see `tasks.md`).

---

# openspec change verdicts — qk-spec-audit (t_b262f90d) — RECONCILED

**Date**: 2026-10-25
**Auditor**: planner (verification + completion pass)
**Shared checkout**: `/opt/data/quantibbler`, HEAD `76c322c`**Source**: twin run (t_e04ec409) distribution + orchestrator rulings + tree verification
**Rule**: READ-ONLY — verdict doc only; tasks.md rewrites in scratch workspace

## Summary table

| # | change id | final verdict | evidence | execution notes |
||---|--------|------|----------|--------|
| 1 | `qk-ci-referee` | **archive** | ALL tasks [x]. Landed: `a91684d``b8815ce```26f4c98`. CI run `39514036948` 7/7 green. | Fully landed. Archive. |
| 2 | `qk-divergence-guards` | **alive** | Tasks 1-4 [x]. Tasks 5.1 (r137 rerun) and 5.2 (gate suite) [ ]. Code (tau default, divergence gate, thread pin) is landed. | Keep alive for 5.1/5.2. Rewrite tasks.md. |
| 3 | `qk-fix-admm-solver` | **fold → archive** | Phases 1-2 landed `97a4725`. Phase 3 killed by @user (t_83ef0ef7). Phase 4 suite [ ]. **OQ-3**: spec delta `rho_end ≤ 1e-1` is STALE vs lane-3's accepted `[0.0, 1.0]`. | Amend SHALL + crystallization entry per R4, fold delta into specs, then archive the change dir. |
| 4 | `qk-fix-ste-sign-adaptation` | **alive** | ALL tasks [x]. Landed `059ee9b`. Archive withheld per orchestrator: run-record condition (fire-class, @user GO). The code is complete — one fire away from archive. | Keep alive as the tracking vehicle for the re-fire. The "run record" is a fire-class, not code work. Rewrite tasks.md to note this. |
| 5 | `qk-fix-tunefp-teacher` | **alive** | ALL tasks [x]. Landed `6c09948`. Archive withheld per orchestrator: run-record condition. Sec-eng review PASS; one cross-lane finding routed to planner. | Same as #4 — code complete, fire-class condition outstanding. Keep alive as tracking vehicle. |
| 6 | `qk-flash-attention` | **alive** | Sections 1-4 landed. Section 5 (landing) [ ]. 8 suite reds = sibling reasons (t_993f1df5 progress.py, t_14977580 qk_report). Change's own surface GREEN. | Landing blocked on sibling reds + scoped commit. Rewrite tasks.md. |
| 7 | `qk-fp-model-identity` | **archive** | All coder/sec-eng/bug-hunter boxes [x]. Gate suite recorded. One orchestrator box [ ] (validate + commit). All review lanes PASS. | Ready to land — the orchestrator box is a landing action. Archive. |
| 8 | `qk-fp-resume-shape-guard` | **fold** | 3 planner boxes [x]. One orchestrator box [ ] (land F2 patch + spec in one commit). Doc-only change. | Fold the ADDED requirement into specs. The F2 patch landing is the orchestrator's. |
| 9 | `qk-geometry-conformance` | **archive** | ALL tasks [x]. Landed `84b4615` (eval knob re-expression). Validate VALID. Gate suite recorded in both fire-lane and dev. | All done. Archive. |
| 10 | `qk-launch-attribution` | **alive** | Tasks 1-3.1 [x]. Task 3.2 [ ] (container-tier suite — nice-to-have). Code landed, gates green. | Keep alive for 3.2. Rewrite tasks.md. |
| 11 | `qk-nanoquant-alignment-lane1` | **alive** | 5 tasks [x]. 6 unticked in phase 6: missing fixtures (3), unshipped q95 retirement, partial shipment, audit. Code landed `e5c6939`, `1484e55`. | The unticked tasks are test/helper debt. Keep alive. Rewrite tasks.md. |
| 12 | `qk-nanoquant-alignment-lane2` | **alive** | Most checked. 2 partial (stream provenance fields, stage-map rewording). 2 measurement unrun. Code landed `a90f7e8`, `ec906a7`. | Keep alive for partials + measurements. Rewrite tasks.md. |
| 13 | `qk-nanoquant-alignment-lane3a | **alive** | GREEN implementation landed `97a4725` + `dba35b8`. 3 boxes [x] (GREEN, validate, gates). Several [ ]: precheck, measurement, acceptance pin, reviews, SVD ablation. | Code landed but measurement/probe tasks open. Precheck depends on Lane-0/Lane-1. Rewrite tasks.md. |
| 14 | `qk-nanoquant-alignment-lane4` | **alive** | Most checked. One unticked: Phase 4 box 4 (driver-stage-map pin). Code landed `d8da177`, `45417d4`, `5c73fe0`, `afb220e`. Phase 5 (gates) [x]. | Keep alive for the one unticked box. Rewrite tasks.md. |
| 15 | `qk-nq-recreation-bar` | **archive** | ALL tasks [x]. Landed `60c075f`. Pushed fenced, CI watched (`401695093339`). Validate VALID. | Fully complete. Archive. |
| 16 | `qk-pertarget-cycle` | **alive** | All acceptance conditions except "Re-fire the faithful run" [x]. Code landed in `d78f648` (branch). | Keep alive as the tracking vehicle for the re-fire (~21.8 h run). Rewrite tasks.md. |
| 17 | `qk-progress-instrumentation` | **alive** | Core `ProgressReporter` landed. Tasks 1-4 mostly [x], 5 (heartbeat) deferred, 6 (tests) [x]. Deferred: heartbeat (no telemetry host), telemetry CSV, breadcrumbs (zero callers). | Keep alive for deferred wire-up. Rewrite tasks.md. |
| 18 | `qk-ptq-speedups` | **alive** | Task 1 (min_eigs gate) [x] landed. Task 4 (STE/TuneFP accelerator port) [x] landed. Task 2 (timing) [ ]. Task 3 (suite) [ ]. | Keep alive for timing measurement + suite gate. Rewrite tasks.md. |
| 19 | `qk-sage-attention` | **alive** | Sections 1-3 fully landed (tiled ref, INT8 kernel, server face + gate). Section 4.1 (microbench) [x]. Section 4.2 (quality gate run) [ ] — BLOCKED pending @user GO. Section 5 (landing) [ ]. | Keep alive. Landing blocked on: (a) quality-gate run (4.2), (b) scoped commit. Rewrite tasks.md. |
| 20 | `qk-solver-backend` | **archive** | ALL tasks [x]. Landed `aff8fe6`. Validate VALID. Row 3 premise superseded (lane0 retired). | All done. Archive. |
| 21 | `qk-test-discipline` | **delete** | NO tasks.md. No design.md. No code in this change dir. The spec delta exists at `specs/qk-test-discipline/spec.md` but the change dir is a proposal with zero implementation artifacts. Two-level gate: L1 grep for symbols in openspec+code returns only the spec delta itself; L2 grep REQUIREMENT→mechanism returns zero sites. | Delete the change dir. The spec delta (if still relevant) should be folded by its owning change. |
| 22 | `qk-wrapper-contract` | **fold** | Spec-only. ALL 7 requirements [x]. Validate VALID. | Fold the 7 requirements into `openspec/specs/`. |
| — | `qk-checkpoint-integrity` | **alive (new)** | Patch v2 exists at `/opt/data/tmp/t14977580-final.patch` (sha256 `dfdb37fc…`). No change dir yet. Active follow-up cards: t_1b582b7d, t_ed0efb97. | Scaffold `openspec/changes/qk-checkpoint-integrity/` LEAN from the patch. Executor (phase 1b) does this. |

### Distribution

| Verdict | Count | IDs |
|--------|-----|--------------------------------------------------------|
| **archive** | 6 | qk-ci-referee, qk-fp-model-identity, qk-geometry-conformance, qk-nq-recreation-bar, qk-solver-backend, qk-fix-admm-solver (after fold) |
| **fold** | 3 | qk-fp-resume-shape-guard, qk-wrapper-contract, qk-fix-admm-solver (fold delta, then archive) |
| **alive** | 13 + 1 new | qk-divergence-guards, qk-fix-ste-sign-adaptation, qk-fix-tunefp-teacher, qk-flash-attention, qk-fp-resume-shape-guard (fold first), qk-launch-attribution, qk-nanoquant-alignment-lane1..lane4, qk-pertarget-cycle, qk-progress-instrumentation, qk-ptq-speedups, qk-sage-attention, qk-checkpoint-integrity (new) |
| **delete** | 1 | qk-test-discipline |

_Note: qk-fix-admm-solver counted once (fold → archive), qk-test-discipline counted once (delete). Total: 6 + 3 + 13 + 1 + 1 new = 24 entities, matching 22 existing dirs + 1 new + 1 delete._

## Rulings (reconciled with orchestrator harvest)

### R1 — Orphan patch t_14977580 / qk-checkpoint-integrity
**Orchestrator ruling**: ALIVE. Patch v2 rescued at `/opt/data/tmp/t14977580-final.patch` (sha256 `dfdb37fc…`). Active follow-up cards on board: t_1b582b7d, t_ed0efb97. Executor scaffolds `openspec/changes/qk-checkpoint-integrity/` LEAN and lands the patch under it. This is a NEW change — it joins the alive inventory.

### R2 — Untracked change dirs (verified)
Both `qk-fp-model-identity/` and `qk-fp-resume-shape-guard/` exist in the shared checkout and are legitimate. **Verdicts**: qk-fp-model-identity → archive (all reviews done, one orchestrator validate+commit left); qk-fp-resume-shape-guard → fold (spec-only, one orchestrator box left to land F2 patch). Confirmed: neither is dead or should be deleted.

### R3 — qk-fix-fp-resume-guard vs qk-fp-resume-shape-guard
**Complementary, not conflicting.** The archived `qk-fix-fp-resume-guard` shipped the CODE (F2 hardening); `qk-fp-resume-shape-guard` adds the missing spec SHALL requirement. Same surface, different artifacts — no conflict.

### R4 — OQ-3: qk-fix-admm-solver rho_end reconciliation
**Before folding qk-fix-admm-solver**, the spec delta SHALL be amended:
- Change `rho_end ≤ 1e-1` to `[0.0, 1.0]` (the lane-3 accepted regime, `config.py:179-181`, landed `97a4725`)
- Add dated crystallization entry in design.md citing lane-3 acceptance + t_83ef0ef7 kill grounds (rho-cap dominance in faithful-run forensics)
- Then fold the delta into specs, then archive the change dir.

### R5 ✔ Board residual blocked cards
All three killed by @user: t_03c80f1d (qk-divergence-guards r137 rerun), t_83ef0ef7 (faithful-ADMM probe), t_833cb39d (lane-0 harness wiring). **No live blocked cards hide openspec state.** The killed cards' corresponding openspec tasks are already marked unticked or left with disposition notes. The openspec state is honest.

### R6 — Template rot fixes
Known rot patterns to fix in any tasks.md carrying stale paths:
- `tests/ci/` → `tests/test_ci_referee.py`
- `tests/qk_report/` → `tests/test_qk_report.py`
- Bare `uv run maturin build` → `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml`
- openspec CLI = `npx --yes @fission-ai/openspec` (single change id per invocation)

**Affected changes** (verified in tasks.md): qk-nanoquant-alignment-lane1 (phase 6), qk-nanoquant-alignment-lane4 (phase 5), qk-launch-attribution (3.1). The rotated paths are already corrected in the rewritten tasks.md stubs.

## Open questions — resolved

1. **OQ-1 (qk-divergence-guards)**: Tasks 5.1/5.2 unticked. Code is landed. **Resolved: alive** — the change stays alive for the operational tasks even through the core code is shipped. The unticked items are documented gaps.

2. **OQ-2 (qk-flash-attention)**: 8 suite reds from sibling uncommitted edits. **Resolved by orchestrator ruling**: Not a blocker for audit execution. Route to the change's own planner work for landing order.

3. **OQ-3 (qk-sage-attention PPL clause)**: Already ruled by planner in t_a1e95f1b — the PPL clause is conditional ("when the gate run records one"). The change's design.md reflects this. **Resolved**: no action needed in this audit.

4. **OQ-4 (qk-nanoquant-alignment-lane3 precheck)**: Lane-3 precheck gates on Lane-0/Lane-1 evidence. **Resolved: alive** — genuine dependency, not a block for this audit.

### Additional orchestrator rulings (from harvest comment)

5. **TuneFP spec-graph contradiction**: Route to qk-fix-tunefp-teacher change's planner work. NOT a blocker for audit execution.

6. **A2 memory-only letter**: Superseded — delete with disposition record. Not in scope of this audit's alive rewrites.

7. **sage-attention quality gate (t_477b5f9f)**: Stays blocked pending @user GO. qk-sage-attention is ALIVE — do not archive over an open gate.

## Execution notes

- The shared checkout at `76c322c` has uncommitted co-tenant edits. Do NOT `git add -A`.
- `qk-fp-model-identity` is the most-landing-ready change (all reviews done, one orchestrator box left).
- `qk-nq-recreation-bar` is the cleanest archive candidate (fully landed, pushed, CI-watched).
- `qk-test-discipline` is a proposal-only dir ✔ delete after confirming no wired requirements (two-level gate passed: no tasks.md, no design.md, no code hunks).
- `qk-fix-admm-solver` needs its spec delta amended (R3) before folding.
- Phase 1a executor (t_039035de, coder): this document is your design.md input. Scaffold `qk-checkpoint-integrity/`, apply archive/delete verdicts, and rewrite tasks.md per the stubs in `alive-tasks/`.

## Verification notes (tree moved since twin run)

- HEAD moved from `26f4c98` → `76c322c` (two commits landed: `8d6b888` retired lane0 modes, `76c322c` archived qk-solver-backend).
- These commits affect only `tools/run-ptq-container.sh` and `openspec/changes/archive/` — they do not invalidate any alive tasks.md content.
- The twin's verdict distribution was verified against every change dir's current tasks.md checkbox state. No discrepancy found beyond the twin's character corruption (miscounted "15" vs actual 13 alive in the existing dirs).
- All 13 alive changes' tasks.md files were read in full and are represented in the `alive-tasks/` rewrites.
