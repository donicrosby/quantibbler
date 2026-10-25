# qk-nanoquant-alignment-lane2 — design

## The two streams

`reconstruct_model` (block_recon.py:1758-1837) currently advances ONE
activation stream: `x_running` starts at the fp stem output, each block's
teacher target is `y_b = teacher_block(x_running)`, and after solve + write-back
the stream advances through the now-quantized block. Student and teacher share
one input object.

Lane 2 splits this into two streams advanced in lockstep:

- **fp stream** — `x_fp` starts at the same stem output as `x_running` and
  advances through the pristine frozen-reference children
  (`teacher_children[name]`), i.e. `X_fp = M_<b(X_cal)`. The block-b teacher
  capture becomes `Y_b = B^FP_b(x_fp)`.
- **compressed stream** — `x_running` advances through the quantized blocks
  exactly as today, producing `X_b = M̂_<b(X_cal)` ([NQ] :730). The student
  sees `X_b`.

Cost: the pristine-block forward previously consumed the compressed stream;
now it consumes the fp stream at identical cost, plus one extra
pristine-block forward per block to advance `x_fp`. Memory: one additional
running activation tensor (`x_fp`), bounded by what the loop already holds
(`x_running` plus the frozen reference model).

## Deliberate deviation from the pinned paper

- **Letter ([NQ] :735):** Algorithm 1 line 10 writes `Y_b ← B^FP_b(X_b)`,
  "teacher output for current block b", where `X_b` (:730) is the
  compressed-prefix activation.
- **Deviation:** the lane feeds the teacher the fp-stream activation
  `X_fp = M_<b(X_cal)`: `Y_b = B^FP_b(X_fp)`.
- **Rationale:** the paper's own Step 1 prose names TuneFP's purpose:
  "minimize the error introduced by the quantization of preceding blocks, as
  well as previously factorized layers in the current block" ([NQ] :525), and
  the Figure 3 caption says the phase exists "to adjust full-precision weights
  for accumulated errors" ([NQ] :577). Under the literal single-stream feed
  the teacher target already carries the prefix error, so the preceding-block
  term vanishes from the objective and Step 1 cannot do its named work. The
  dual-stream capture is the reading under which Algorithm 1's Step 1 does
  what the paper says it does — that is the "alignment" the lane is named for.
- **Consequence:** at every block b ≥ 2 the streams diverge, so TuneFP fired
  at a block's first target would have a nonzero objective by construction —
  the single-stream target-1 vacuity disappears. However, the universal
  `idx == 0` skip from pertarget-cycle (ruling (a)) still applies: at the
  first target of ANY block, nothing in-block is factorized yet, so there
  is nothing for TuneFP to compensate. The skip is universal (every block's
  first target), not limited to block 1's identity prefix. The justification
  is "nothing factorized in-block yet," not "teacher equals frozen deepcopy"
  (the stale single-stream rationale that a coder follow-up rewrites).
- **Edge case:** a prefix whose quantization error is exactly zero would still
  produce a zero objective at target 1 and trip the detector — accepted as
  fail-closed behavior (a zero-error prefix has nothing for TuneFP to do;
  stopping loudly beats tuning vacuously).

## The guard stays a detector; its rationale is rewritten

Behavior is byte-unchanged: zero first-step gradient with zero initial
residual emits `tune_fp_vacuous_teacher` (both norms attached), and a
`paper_faithful` run fails closed on the event. The old rationale — "target-1
vacuity is inherent to the single-stream placement" — is replaced: under
dual-stream capture every fired TuneFP call runs with a diverged prefix or
with in-block factorized layers, so zero-grad + zero-residual means broken
plumbing (stream cross-talk, a capture reading live or compressed state, a
detached student). The detector's meaning strengthens; its mechanics do not
move.

## `paper_faithful` re-pinning

The verdict keeps every existing condition (flag AND no runtime deviations AND
no mechanism deviations AND required stages executed) and gains one: the
teacher captures must carry **fp-stream** provenance. A capture recorded with
compressed-stream provenance — the legacy single-stream form — or with missing
provenance is a mechanism deviation recorded in the deviation list and denies
the label. The dual-stream deviation marker itself (fp-stream feed vs the
:735 letter) is documentation; under the re-pinned semantics fp-stream
provenance IS the faithful form. The `paper_faithful` config comment is
rewritten to state this; the flag's fail-closed behavior is unchanged.

## Stream provenance in recorder events

The `capture` event (block_recon.py:1230) already records `x_b`, `y_b`,
`teacher_source`, `teacher_fingerprint`, `capture_index`. Lane 2 adds
per-tensor stream provenance: `x_b_stream = compressed-prefix`,
`y_b_stream = fp-stream` (values for the lane's own capture; legacy values
kept distinguishable for report forensics). `tune_fp`, `ste`, and
`tune_fp_vacuous_teacher` events reference the capture they consume and carry
the same provenance. qk_report's stage-map row for `capture`
("compressed-prefix inputs X_b and teacher outputs Y_b", qk_report.py:549) is
reworded to name the fp-stream teacher output. Event payloads are additive —
existing readers ignore unknown keys.

## No new solver state: probe and fingerprint shapes unchanged

- **Prefix probe:** the dual-stream capture changes nothing the probe sees —
  the live host after each block write-back is still exactly the quantized
  prefix of `len(results)` blocks. Signature `probe(model, depth, name)`,
  per-block call site, and raising semantics are unchanged.
- **Checkpoint fingerprint:** `config_fingerprint` (checkpoints.py:48) hashes
  `asdict(PTQConfig)` plus code identity with `progress_probe` the only
  exclusion. Lane 2 adds NO config field, so the fingerprint's shape — payload
  keys and the joined field set — is unchanged. The VALUES change across the
  lane boundary because the payload embeds `git_describe` (code identity):
  a pre-lane checkpoint is refused resume by the existing stale-state
  protection, which is exactly the discipline's "checkpoint fingerprints
  updated for any new solver state" — there is no new solver state, so there
  is no fingerprint update, only the code-identity refusal.

## Discarded alternatives

- **Keep single-stream, drop the target-1 skip instead:** re-introduces a
  call whose objective is identically zero — tuning nothing — to satisfy a
  counter. Discarded: it manufactures the vacuous event the guard exists to
  catch.
- **Single-stream plus a separate "prefix-error" loss term:** two objectives,
  an unanchored weighting, and extra solver state. The paper describes one
  objective per step; the fp-stream capture expresses prefix error inside the
  single MSE. Discarded.
- **On-demand pristine-prefix forward per block (no running stream):** an
  O(B²) pristine-forward cost. Discarded for the O(B) running stream.
- **Stream choice as a config knob:** forks label semantics per run, adds a
  fingerprint-joined field (shape change), and re-opens the single-stream
  defect under a flag. Discarded; dual-stream is the one capture.
- **Remove the guard:** discarded — zero-grad + zero-residual still means
  broken plumbing, and the lane makes the detector stronger, not obsolete.

## Informed assumptions

1. **The fp stream reproduces the pristine prefix exactly.** Advancing `x_fp`
   through the frozen reference children equals `M_<b(X_cal)`. Rationale: the
   frozen reference copy is the pinned pristine source already used for
   teacher capture (qk-fix-tunefp-teacher A1). Wrong → the stream-separation
   fixture still passes (Y_b independent of the compressed stream) but the
   `dual_stream_advances_in_lockstep` fixture fails its pristine-prefix
   forward comparison.
2. **Block 1 is the only point where the streams coincide.** Rationale:
   nothing is quantized before the first quantized block. Wrong → the
   block-1-target-1 skip would also suppress legitimate non-vacuous calls;
   the target-1 fixture at block 2+ would expose it by construction.
3. **The fp-stream capture is the paper-intent reading, the :735 letter is
   the literal feed.** Rationale: [NQ] :525/:577 define Step 1's purpose in
   prose; a deviation record plus the re-pinned label make the choice
   auditable rather than silent. Wrong → the e2e tiny-model gate comparison
   (dual-stream vs single-stream quality) and the report's own deviation list
   would carry the evidence; the lane is reversible without format impact.
4. **One extra running activation tensor stays within memory budgets.**
   Rationale: the loop already holds `x_running` and the frozen reference
   model; `x_fp` is the same shape as `x_running`. Wrong → gate-run OOMs
   surface immediately in the gate suite.

## Open Questions

- **Block-1-target-1: skip or fire-and-trip?** — Status: decided. The
  universal `idx == 0` skip applies at every block's first target, per
  pertarget-cycle ruling (a). Under dual-stream, the streams diverge at
  block 2+, but at idx==0 no in-block layer is factorized, so TuneFP has
  nothing to compensate — the skip is universal. The earlier "block 1
  only" reading is withdrawn (planner reconcile 2026-10-25).
- **Does dual-stream move the e2e tiny-model gate numbers (max logit delta,
  BPW)?** — Status: open until the Phase 3 re-measure. Owner: planner. The
  gate bounds do NOT move without a measured delta and planner sign-off; the
  attribution run records dual-stream vs single-stream on the same fixture.
- **Should `tune_fp`/`ste` events back-reference the capture they consumed by
  capture index rather than repeating stream fields?** — Status: decided.
  Repeat the provenance fields on each event (self-contained events; the
  recorder's contract is additive key-value pairs, not references).

## Interaction with other lanes

Lane order is load-bearing: 0 gates 1-3; lane 2 is independent of lanes 1 and
3 but lands before any new production run. Merge surface: lane 2 touches
`block_recon.py`'s capture loop and the `paper_faithful` config comment —
lines lane 1 and lane 3 changes may also edit; conflicts are textual, not
semantic. No lane-2 requirement depends on lane 1 or lane 3 artifacts.

## Implementation crystallization

### 2026-10-25 — reconcile of the landed lane (`a90f7e8` Ph1, `ec906a7` Ph2)

Re-verified every task row by re-running its named command against the landed
tree (card `t_75564534`). Four records, none of which changes a requirement:

1. **The lane's fixtures are accelerator-tier artifacts.** `PTQConfig.solver_backend`
   defaults to `"torch"`, so these fixtures reach the implicit `resolve_device`
   refusal on a CPU-only interpreter and cannot be verified from the dev venv
   (`torch 2.14.1+cpu`); every number recorded on the card comes from
   `quantibbler-ptq:rocm-official` with `/dev/kfd` + `/dev/dri` and
   `PYTHONPATH=/repo/src`. The rejection is not overridable by
   `QUANTIBBLER_DEVICE=cpu` here: `lb_admm.magnitude_balance` calls
   `get_backend(backend)` with no device argument, so a CPU-forced diagnostic
   run still stops inside the block solve.

2. **The red-first witness is recorded, with its confound.** The Phase-1 boxes
   say "expected RED"; that was honoured at `a90f7e8` (its commit tally: 5
   held-red, 110 passed). Re-running the five fixture files against a read-only
   export of that commit in today's accelerator configuration gives
   `16 failed, 99 passed`, in which 13 of the 16 are
   `TypeError: can't convert cuda:0 device type tensor to numpy` — the
   pre-`a7999ce` accelerator-path defect, later fixed by the single
   `to_host_np` idiom. The two lane-2 mechanism reds reproduce exactly
   (`guard_rationale_rewritten`, `label_requires_fp_stream_teacher_provenance`);
   the four counting reds are masked by that confound. A red-first witness is
   therefore only re-observable in the configuration it was measured in.

3. **Two rows are partial shipments, and are left UNCHECKED.** The `capture`
   event carries `x_b_stream`/`y_b_stream`, but `tune_fp`, `ste`, and
   `tune_fp_vacuous_teacher` events do not carry any stream field, and the spec
   scenario `recorder_events_gain_stream_provenance` has no test. `qk_report`,
   separately, ships the verdict condition (`capture_is_fp_stream`) but has not
   reworded `_STAGE_LINES`' `capture` description (still "compressed-prefix
   inputs X_b and teacher outputs Y_b") and names no offending capture in a
   deviation list. Both deltas are handed to the planner rather than resolved
   by the reconciler: the decided Open Question ("Repeat the provenance fields
   on each event") is the letter of the unshipped half.

4. **Doc drift found, not repaired here.** The delta's `suite_green` scenario
   and the Phase-4 row both name `pytest tests/ptq/ tests/ci/ tests/qk_report/`;
   `tests/ci/` and `tests/qk_report/` do not exist and never did (the CI-referee
   test is the file `tests/test_ci_referee.py`). The reconcile ran the real
   suite (`pytest tests/` → 652 passed, 4 skipped, 3 deselected) and amends only
   the task row; the scenario text is the requirement of record and is left for
   its owner. Same for the `-k teacher_stream` selector, corrected in the task
   row to the scenario id `teacher_target_reads_fp_stream`. The
   `dual_stream_deviation_marker` scenario's "recorded … in the run record" half
     is likewise unimplemented: nothing in `src/` writes the marker into a record,
     and the shipped test supplies it by hand.

   ### 2026-10-25 — Planner reconcile ruling (t_966fb4bc): universal idx==0 skip

   The lane's design recorded the target-1 skip as the "single precise
   identity-prefix case" (block 1 only). That is wrong against standing code:
   `block_recon.py:1595` applies `if idx >= 1:` universally — at every block's
   first target, regardless of block index.

   **Corrected posture:** Under dual-stream the streams diverge at block 2+,
   so a TuneFP call at the first target of block 2+ WOULD have a nonzero
   objective. But the pertarget-cycle ruling (a) still skips it: at `idx==0`,
   no in-block layer is factorized, so TuneFP has nothing to compensate. The
   skip is "nothing factorized in-block yet" — NOT "teacher equals frozen
   deepcopy" (the stale single-stream rationale that lives in
   `block_recon.py:1490-1496` and the `_tune_fp` docstring, routed to coder
   by the reconcile card). The universal skip is compatible with dual-stream
   because TuneFP compensates in-block factorization error, not prefix error
   — and prefix error is nonzero for blocks ≥ 2, but in-block error at
   `idx==0` is still zero.

   Spec delta amended: scenario `target1_tunefp_vacuity_gone_by_construction`
   now names the universal skip and its rationale. The Open Question
   "Block-1-target-1" is resolved to universal skip. Design.md §Consequence
   rewritten.
