# Design: qk-fix-tunefp-teacher

## Teacher construction (corpus-pinned, [NQ] Algorithm 1)

Verified 2026-10-21 on the pinned corpus bytes
(`docs/papers/nanoquant-2602.06694v3.md`, algorithm block): the figure text
IS carried by the conversion and pins the Step-1 construction:

- `X_b ← M̂_{<b}(X_cal)` — input activation after the already-compressed
  prefix.
- `Y_b ← B^FP_b(X_b)` — teacher output: the pristine FP block applied to
  `X_b`.
- `TuneFP(b, X_b, Y_b; T_pre)` — Step 1.

So `y_fp = block_fp(X_prefix_q)` where `X_prefix_q` is the
compressed-prefix stream, captured once per block BEFORE any tuning or
in-block factorization, under `torch.no_grad()`, `.detach()`ed, gradients
disabled, never recomputed from tuned weights. The tuned objective is
`‖block_tuned(X_prefix_q) − y_fp‖²`.

**Placement (figure-encoded, 2026-10-21).** The user pinned the method
image directly: [NQ] Figure 3, `block_recon.png` — vendored at
`docs/papers/assets/block_recon.png`, sha256
`3b4c5ed8cb12757f6a39cb366329418abcc00c372366762223ae9926ab798c61`,
source `https://arxiv.org/html/2602.06694v3/block_recon.png`. Panel
inventory (verified on the vendored bytes): left panel — Q/K/V GREEN
(already factorized), O/G/U/D YELLOW (still FP) with the tuning flame
markers on the still-FP remainder (exact flame inventory on the remainder
is contested between readers — planner's passes count flames on O/G/U/D,
sec-eng's counts G and U; the placement ruling depends only on Q/K/V being
factorized while the remainder is FP, which both reads agree on); middle
panel — scissors on O only; right panel — O green with a flame on O only;
sequence arrows left→right.
Official caption (pinned HTML): "(1) Error Propagation Mitigation to
adjust full-precision weights for accumulated errors". Together with
the Sec 3.2 Step 1 prose ("previously factorized layers in the current
block") the figure encodes the operative placement: **TuneFP fires ONCE
per block (Algorithm 1 line 12, honored), AFTER the attention
projections Q/K/V are factorized and BEFORE the remaining linears
O/G/U/D are, tuning the still-FP remainder against the cached
teacher.** At that placement the objective is never identically zero —
the compensation source always exists. The literal
line-12-before-everything placement (zero factorized siblings,
objective `‖B^FP_b(X_b) − Y_b‖²` ≡ 0 at init) is the measured defect:
the bit-identical T0 the stage trace recorded, ruled out by the
figure's own encoding. The same captured (X_b, Y_b) feeds TuneLatentSTE
per line 19 — pinned in the sibling change
`qk-fix-ste-sign-adaptation`; Figure 3's right panel (flame on O alone
after its factorization) confirms STE refinement is per-layer, on the
just-factorized layer.

## Why the objective is nonzero (prose is binding)

[NQ] Sec 3.2 Step 1 prose: TuneFP compensates "the error introduced by the
quantization of preceding blocks, as well as previously factorized layers
in the current block." With the Figure 3 placement, at TuneFP time the
attention projections are already factorized, so `block_tuned` (factorized
Q/K/V + tunable FP remainder O/G/U/D) deviates from the pristine teacher
by exactly the error those factorized layers introduced — nonzero
gradient, real compensation. The algorithm's line-12 placement is
schematic; at init there the objective is identically zero, which is
precisely the vacuity the guard below exists to expose (and the figure's
own encoding rules that placement out).

Cost: the teacher capture already exists for the `y_b = None` path
(`reconstruct_block` captures `y_b = block(x_b)` under `no_grad` before
tuning); the fix pins that capture as the teacher source, adds the single
block-level TuneFP pass between the attention projections and the output
projection inside the sequential factorization loop, and keeps one fp
forward per block for the teacher — budgeted, offline, same passes
as existing captures.

## Guard semantics (zero first-step gradient is always the defect)

Under the Figure 3 placement the TuneFP pass always runs with the
attention projections already factorized, so the first-step gradient MUST
be nonzero. A zero first-step gradient — whether at the legacy pre-loop
position (nothing factorized in-block, the shipped bit-identical
signature) or at the operative call (≥1 layer factorized but no
compensation path) — aborts with `tune_fp_vacuous_teacher` naming the
block and carrying BOTH the first-step gradient norm and the initial
residual norm: gradient-zero + residual-zero = broken teacher (target
equals current output — the shipped defect); gradient-zero +
residual-nonzero = detached student (no gradient path). Diagnostic runs
continue with pristine weights and the event on record. There is no
legitimate trivial-zero case under the figure-encoded placement, so the
guard has no carve-out.

## Remaining informed assumption (narrow)

The corpus names the TuneFP call but not its internal loss formula. This
design uses the block-output squared error against the cached teacher
(`‖block_tuned(x) − y_fp‖²`), the direct reading of the prose ("minimize
the error introduced by..."). Marker recorded in code and artifacts as
`tune_fp_loss: informed_assumption` citing the corpus block.
Wrong-assumption surfacing: the vacuous-teacher abort (case 2) plus the
nonzero-effect acceptance criterion.

## Reviewer amendments (2026-10-21, sec-eng integrity pass; folded by planner)

1. **A1 — teacher weights pinned to pristine, provenance recorded.** By
   block N the live model carries earlier blocks written back quantized, so
   the teacher block's weights SHALL load from the pinned pristine
   safetensors (or a frozen reference copy), never the live module's
   mutable state; the artifact records the source fingerprint and the
   capture index. Scope note (planner ruling): the input stream stays the
   compressed-prefix stream per the corpus (`X_b ← M̂_{<b}(X_cal)`); what is
   pristine-pinned is the teacher block's *weights*. The informed-assumption
   marker is only honest if the provenance it hangs on is recorded.
2. **A2 — teacher cache memory-only in v1.** The cached teacher is never
   persisted; on resume it is re-captured from the pinned pristine source.
   Any future persisted form is fingerprint-bound to the pristine source
   hash + capture index. (A stale restored teacher would be the third
   fail-silent surface of this class: ratio falsy-zero, empty-archive
   resume, stale-teacher cache.)
3. **A3 — two-norm event + fidelity invalidation.** `tune_fp_vacuous_teacher`
   carries the first-step gradient norm AND the initial residual norm:
   gradient-zero + residual-zero = broken teacher; gradient-zero +
   residual-nonzero = detached student (construction bug, no gradient path).
   Runs claiming paper fidelity treat any vacuous event as invalidating the
   claim (fail or `paper_faithful: false`); diagnostic runs keep the
   continue-with-pristine semantics of the guard section above.

## Discarded alternatives

- **Teacher = task loss / next-token logits.** Rejected for this change:
  couples the PTQ stage to an eval surface and reopens the probe-detection
  gate's territory; block-local objectives only (v1 scope).
- **Teacher = live module output recomputed per step (the shipped shape).**
  Rejected: self-referential — recomputing the teacher from the tuned
  module each step makes the difference identically zero; the measured
  defect.
- **Teacher on the unquantized FP-prefix stream (the former informed
  assumption).** Rejected: the corpus pins `X_b ← M̂_{<b}(X_cal)` (compressed
  prefix) and `Y_b ← B^FP_b(X_b)`; an unquantized-prefix teacher is a
  different mechanism the paper does not describe at this step.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence: sequential in-block factorization + the Figure 3
TuneFP placement are in scope here (the former v1 non-goal is withdrawn);
the three fix dirs land before any further diagnostic runs; the ladder
fork fires only after a paper-faithful run exists.

## Amendment (2026-10-21, planner, pre-fire: fidelity verdict + Phase-1 statistics)

- **The label becomes a verdict (sec-eng's ruling, folded).** No unwaived
  runtime deviation is compatible with `paper_faithful: true`. The
  reported label is computed from the run's own record: the flag AND
  `runtime_deviations` empty AND no mechanism-class deviation (polish
  rounds > 0, a fired STE discrete flip pass, a scale-only re-scope
  marker, a vacuous-teacher event, a stage failure, or solver != admm).
  Informed-assumption markers (the TuneFP loss formula) are
  documentation and do not flip the label. Each condition is recorded, so
  the label is auditable from the artifact alone. The solver condition
  follows the paper's letter: Algorithm 1 line 16 is LB-ADMM; the
  pipeline's `solver=svd` is the measured production ablation and cannot
  ride the faithful label.

> **Annotation (coder, 2026-10-22 — repo-clock author date; see
> `qk-pertarget-cycle` §9 on the phantom date).** This amendment's
> mechanism list — in particular the **vacuous-teacher event** it names — was
> defined against the **pre-re-rule Figure-3 reading**. Under ruling (a)
> (`qk-pertarget-cycle`) the first TuneFP of a block no longer fires at all, so
> that vacuity case becomes production-unreachable rather than routine. The
> condition **stays** as a fail-closed safety net; the reading it was written
> against does not.

- **Phase-1 statistics (Algorithm 1 lines 2-5).** The landed stage map
  names a real deviation: statistics are collected per block with the
  compressed-prefix stream as input, where the paper collects them once
  from the pristine model over raw X_cal. (Coder's crystallization note
  "Statistics timing" calls the collection "pristine" — the block
  WEIGHTS are pristine at collection time, but the INPUT stream is the
  compressed prefix; the input is the deviation, and the stage-map string
  is the accurate reading.) The hoist: one pre-loop `collect_stats` pass
  (streamed D-7 path) over the pristine model on the raw calibration
  batches; block reconstruction consumes the pre-collected book; capture
  (lines 9-10) is teacher capture only. This also removes one fwd+bwd
  per block from the loop.
- **Eval window — corrected after Bug Hunter review.** The corpus pins
  WT2 perplexity and names `lm_eval=0.4.9` ([NQ] Sec 4.1/App. C), but does
  NOT pin exact test-set coverage or windowing semantics. My earlier
  "full WT2 / same pedestal as the paper tables" sentence is WITHDRAWN.
  Measured with the pinned Qwen tokenizer: the test shard is 298,938
  tokens = 145 complete 2,048-token blocks + a 1,978-token remainder.
  The local evaluator's largest batch-8 request is `EVAL_SEQS=18`: 144
  blocks / 294,912 tokens / 294,768 scored predictions — 99.310% of the
  complete blocks but 98.653% of source tokens. It omits one complete
  block plus the 1,978-token remainder and is therefore NOT full-set or
  paper-table-comparable. Ruling: 18 is the maximal local quality-gate
  window for this fire; the artifact records evaluator kind, all source /
  selected counts, coverage, seed and ordered-selection fingerprint, and
  `paper_table_comparable: false`. The method-fidelity verdict is scoped
  to the quantization method and calibration protocol; evaluation
  comparability is a separate verdict. A faithful-flagged fire using this
  local evaluator must use its maximal complete-batch window or fail
  preflight — 5 batches can no longer ride the proof-scale run.
- **Ladder note (parked, unchanged).** The stage-trace threshold bands
  (T1 etc.) were pinned on svd-path traces; the parked ladder re-derives
  its bands on the faithful (admm) artifact when it un-parks. Bug
  Hunter's independent `real_floor`/`naive` table is solver-independent
  (computed from the pinned safetensors) and stands as-is.

## Implementation crystallization (landing, 2026-10-25)

Both review lanes PASS on the landed bytes (sec-eng t_5855d059: integrity
+ required-stage re-audit verdicts; bug-hunter t_a531ff73: teacher-once +
guard, required-stage poison directions). The landing commits their ticks
and evidence; the findings are ROUTED, not fixed post-review (no RED-first
exists for any of them on this change's surface):

- **Balance-arm bypass is not this change's bytes.** The 52-57F CPU-suite
  class (`magnitude_balance` → `get_backend(None)` → torch default refuses
  implicit CPU) originates in lane-3's backend-default flip (8d1bcc0 /
  97a4725) after this change's CPU-green landing (2bb04fb, 2026-10-21);
  both reviews proved this change's surfaces green under the numpy-default
  probe (teacher/guard/dual-stream 17 passed RC=0 with a sitecustomize
  discriminator; `tests/test_qk_report.py` 91/91 RC=0). Owned by card
  t_5e71e281; the dropped-QUANTIBBLER_DEVICE half landed via t_7190f18b.
- **A2's "memory-only" letter is superseded on HEAD.** Lane-2 commit
  ec906a7 persists the fp-stream teacher target as checkpoint npz key
  `y_b` (checkpoints.py save path; block_recon.py resume restores from
  it), and the A2 self-test greps npz keys for the substring "teacher",
  which cannot see the real key; resume trusts `y_b` without the
  pristine-source-hash + capture-index binding A2 names. Spec-accuracy
  reconcile routed to the planner BEFORE any archive; the deny-only
  resume property on landed bytes is verified and unaffected.
- **The delta's teacher-cycle wording is superseded by the spec graph.**
  pertarget-cycle's universal idx==0 skip (landed block_recon.py:1531,
  pinned by test_dual_stream.py) vs lane-2's single-identity-prefix
  design vs this change's compressed-prefix/once-per-block delta text
  contradict over the same requirement surface; the stale single-stream
  rationale remains at block_recon.py:1490-1496 and in the `_tune_fp`
  docstring. A planner reconcile card gates the archive sweep (linked as
  a parent of t_944374c0) so `openspec/specs/` never folds contradictory
  SHALLs.
- **Eval-geometry label gap (t_f5dc83f7 routing).**
  `geometry_deviations` never flags `eval_len != 2048`, so a split-window
  run can carry `paper_faithful: true` under stage-fidelity semantics;
  the bar protection is the human-gated acceptance ledger. Routed to the
  same planner reconcile: either the label contract states that
  protection or the deviation class is added — a contract decision, not
  a post-review code patch.
- **Archive gate stands.** Per the orchestrator task row: do not archive
  until a real run shows a nonzero TuneFP effect on record — a fire-class
  event needing @user GO (lane authority rule). This change is
  validate-strict, double-reviewed PASS, landed and pushed; the archive
  itself belongs to sweep t_944374c0 after the run record and the planner
  reconcile exist.

## Planner reconcile rulings (2026-10-25, t_966fb4bc)

Three review findings routed here by the landing orchestrator (t_416ad06f,
crystallization entry). Code evidence re-verified against current HEAD
(84ba3b4, post-f9c1544) before ruling. Rulings settle the spec-text
contradictions so `openspec/specs/` never folds contradictory SHALLs.

### F1 — A2 "memory-only" superseded: VERIFIED, spec text corrected

Re-verified each claim against post-f9c1544 HEAD:

- **y_b IS persisted.** `checkpoints.py:265` writes `arrays[_YB_KEY] = _yb`
  under the reserved npz key `y_b` (`checkpoints.py:305`). f9c1544 added the
  `"|" in key` filter to prevent `y_b` from being misread as a layer name;
  it did NOT stop persisting `y_b` — it only fixed the resume-side name
  derivation.
- **Resume restores from it.** `block_recon.py:2082-2084` calls
  `ckpt.load_block_yb(ckpt_dir, i)` and sets `x_fp` from the restored
  tensor. Legacy checkpoints predating lane 2 fall back to recomputing the
  stream.
- **Self-test is blind to the real key.** `test_tunefp_teacher.py:377`
  greps npz keys for substring `"teacher"` — cannot see `"y_b"`. The test
  passes green but certifies nothing about the actual persistence.

**Ruling:** A2's "memory-only in v1" and "never persisted to a checkpoint"
text is factually wrong on HEAD. The corrected posture: the fp-stream
teacher target IS checkpoint-persisted as `y_b` and recovered on resume
(per lane-2's stream contract). The persistence is fingerprint-bound to
the checkpoint's own config+code identity (the existing `config_fingerprint`
mechanism that gates resume); a separate pristine-source-hash +
capture-index binding is not required because `y_b` is the stream
activation (derived from the fp-stream prefix, itself reproducible from
the checkpointed layer states), not a detached teacher cache. The
`memory-only` scenario text in this change's delta is amended to match.

**Spec delta edit:** `teacher_cache_is_memory_only_and_recovered_on_resume`
scenario reworded to `teacher_target_is_persisted_and_recovered_on_resume`.

### F2 — TuneFP-cycle spec graph: universal idx==0 skip is standing code

Three deltas contradict over the same requirement surface:

1. `qk-pertarget-cycle` SHALLs universal `idx == 0` skip — "the first
   target SHALL be factorized with NO preceding TuneFP" — matches the
   landed `if idx >= 1:` at `block_recon.py:1595`.
2. `qk-nanoquant-alignment-lane2` records "the single precise
   identity-prefix case" (block 1 only) — WRONG against standing code.
   The skip applies at EVERY block's first target because at `idx==0` no
   in-block layer is factorized, so TuneFP has nothing to compensate. Under
   dual-stream the teacher reads fp-stream input while the block reads
   compressed-stream input — streams have diverged for blocks ≥ 2 — but
   that divergence is prefix error, not in-block factorization error, and
   TuneFP compensates the latter per the paper's prose ([NQ] :525).
3. `qk-fix-tunefp-teacher` delta text says "once per block" (line 9 of its
   spec requirement) — also stale against the re-ruled per-target cycle.

**Ruling:** The universal `idx == 0` skip from pertarget-cycle is the
standing truth. Lane-2's design.md and spec delta must be corrected: the
skip is universal (every block's first target), not block-1-only. The
once-per-block language in this change's own delta is corrected to match
(the requirement header is kept, but the scenario text referencing
"once per block" and the "compressed-prefix" teacher is reworded).

**Stale code rationale (routed to coder):** `block_recon.py:1490-1496`
("the teacher is its frozen deepcopy") and the `_tune_fp` docstring at
lines 36-42 ("the teacher is its deepcopy") still justify the skip with
the single-stream argument that is false under dual-stream. These are
code-side text defects — routed to coder as a RED-first task (see below).

### F3 — Eval-geometry label gap: deviation class added

`geometry_deviations()` checks `tune_len != eval_len` and
`calibration_rows != 128` but never checks `eval_len != 2048`. A 512-window
run where `tune_len == eval_len == 512` passes clean and can carry
`paper_faithful: true`.

**Ruling:** The computed verdict must be exhaustive — the human-gated
acceptance ledger is for waivers, not for gaps in the label's condition
set. Add a `non_protocol_eval_geometry` deviation class to
`geometry_deviations()` that flags `eval_len != 2048`. The code half is
routed to coder as a RED-first task (separate card).

### Coder follow-up (code-side text fixes + eval-geometry gap)

A single coder task is created for the code-side repairs, RED-first:
1. Rewrite `block_recon.py:1490-1496` rationale to justify the idx==0 skip
   under dual-stream semantics (nothing factorized in-block yet, not
   "teacher is deepcopy").
2. Rewrite `_tune_fp` docstring lines 36-42 to match.
3. Add `non_protocol_eval_geometry` deviation class to
   `geometry_deviations()`: flag `eval_len != 2048` with the paper's
   documented protocol geometry.

See card t_ (created by this reconcile on completion).

### F3 follow-up — integration driver test re-encoding (2026-10-25, t_49f35e27)

The F3 code half (coder t_8c9b9179) added `non_protocol_eval_geometry` to
`geometry_deviations()`, but the F3 card did not name its consequence on the
seq-32 integration driver tests, which assert the old leniency. The coder
correctly refused to re-encode them (a verdict-field assertion is a review
decision) and routed the ruling here. Ruling and re-encoding recorded below.

**Ruling: option (a) — keep the two tests at seq 32 and assert the deviation.**
Option (b) (move the faithful test to seq 2048) is infeasible at fixture
scale, not merely slower:

1. The fixture model is pinned `max_position_embeddings=256`
   (`test_driver_tune_seq_len.py:106`); a seq-2048 run is impossible without
   re-engineering the fixture model.
2. The synthetic corpus (`20 lines × 30 × sentence`, ~12 K tokens) cannot
   supply the protocol's 128 calibration sequences × 2048 tokens
   (262,144) plus a maximal eval window — `tokenize_chunks` refuses
   (`n_blocks < n_chunks * BATCH`), so `--calibration-seqs 128` at seq 2048
   fails the sizer before any solve.
3. seq-2048 ADMM at the App. C 400-iteration schedule is ~64× the seq-32
   compute; the seq-32 faithful test already exceeded 900 s on a loaded box,
   so seq-2048 would be hours.

The premise of `test_paper_faithful_run_records_the_algorithm1_stage_map` —
"a paper-faithful run at fixture scale" — is now impossible BY DESIGN: the
protocol geometry (2048) is unreachable at fixture scale, which is exactly
what the deviation class correctly encodes. This is the design working, not
a regression.

**Re-encoding (the exact new assertions, routed to coder as a spec):**

`test_paper_faithful_run_records_the_algorithm1_stage_map` — the faithful
SURFACE (stage map, per-target cycle, teacher provenance, STE `paper_ste`,
polish off, L137–277) stays unchanged; only the verdict-label assertions
flip to prove the geometry alone denies:

- `paper_faithful` → `is False` (was `is True`).
- the `fidelity_conditions == {...}` literal → `no_runtime_deviations is
  False` and `all(v for k, v in conditions.items() if k !=
  "no_runtime_deviations")` (mirrors the split-geometry test).
- `runtime_deviations == []` → `[d["kind"] for d in
  report["runtime_deviations"]] == ["non_protocol_eval_geometry"]` (exactly
  one, the named kind).
- the SEPARATION block's substring guard `not any("eval" in kind)` is now too
  blunt (`non_protocol_eval_geometry` legitimately contains "eval"): replace
  it with the explicit kind-set assertion above; keep
  `"evaluation" not in fidelity_conditions` as the separation proof.

`test_a_faithful_claim_on_the_svd_ablation_is_denied_by_the_computed_verdict`
— essence (mechanism denial: `solver_is_admm is False`) survives; the
geometry is now an independent, correctly-named denier:

- `runtime_deviations == []` → the deviation list names
  `non_protocol_eval_geometry`; the comment "It is the solver, not the
  geometry" is rewritten to "the solver denial fires end-to-end; the geometry
  deviation is also present and named (seq 32 != 2048)".

`test_a_split_tune_geometry_is_named_as_a_runtime_deviation` — NO change: its
membership assertion (`"tune_geometry_split" in kinds`) and all-else-True
negation already hold with the deviation list grown to three entries.

The verdict True-path end-to-end coverage moves to the proof-scale fire
(seq 2048, real model) — the unit-level True path remains pinned in
`test_qk_report.py` (`test_fidelity_label_is_true_only_when_every_condition_holds`,
`geometry_deviations(2048, 2048, 128) == []`).

**Spec delta:** `label_flips_on_runtime_deviations` scenario's parenthetical
amended to name the fourth class, so the exhaustive deviation set is
greppable from the spec.

## Stage journal (post-fire row, 2026-10-25)

The post-fire row's journal is ``stage-events.jsonl`` beside the solve
checkpoints, written by ``append_stage_event`` (one open/write/close per
event so every event up to a crash is on disk) and read by
``read_stage_log`` / ``resume_stage_events`` behind the live fingerprint
guard. The per-line shape check (``stage_journal_record``) validates
payload keys against ``_STAGE_PAYLOAD_KEYS`` and finiteness
(``_non_finite_path``), so a name-only or saturated line raises rather
than counting as an execution — the payload-blind required-stage verdict
must not be handed a record re-read from disk unvalidated (sec-eng
t_5855d059 named this property).

Two consequences t_a48c5c76 recorded in tasks.md:

- **The fresh-start reset drops the launching run's pre-guard Phase-1
  line.** ``reset_stage_log`` is called where the fingerprint is written,
  which is before the block loop. Phase-1 statistics re-fires on every
  launch (it is not checkpointed), so a resume never needs it from disk.
- **``tune_scales_kd`` now travels through the driver's recorder.**
  Before the journal the scale stage was emitted directly by the wrapper,
  bypassing the recorder that writes the journal — so the record carried a
  ``tune_scales_kd`` event but the journal did not. Now the wrapper emits
  through the same recorder that writes the journal, so the record and the
  journal carry one event set.
