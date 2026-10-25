# Tasks: qk-fix-tunefp-teacher

- [x] (coder) Implement per the spec delta: teacher capture per [NQ]
  Algorithm 1 (`y_fp` = pristine block output on the compressed-prefix
  stream, cached pre-tuning, detached, no_grad) with the teacher block's
  weights loaded from the pinned pristine safetensors (or a frozen
  reference copy), never the live module state; source fingerprint +
  capture index recorded in the artifact; teacher cache memory-only and
  re-captured on resume; sequential in-block factorization with `_tune_fp`
  fired once per block after the attention projections (q/k/v) are
  factorized and before the remaining linears (o/g/u/d) are, tuning the
  still-FP remainder against the cached teacher ([NQ] Figure 3
  placement; the legacy pre-loop call site is removed);
  informed-assumption marker for the TuneFP loss formula; guard: zero
  first-step gradient — operative call or any call emitted with no
  in-block layer factorized — aborts with `tune_fp_vacuous_teacher`
  carrying BOTH the gradient norm and the initial residual norm (no
  trivial-zero carve-out: under the Figure 3 placement the operative
  call always has factorized attention projections); fidelity-claiming
  runs fail (or record `paper_faithful: false`) on any vacuous event.
  Defaults stay [NQ] App. C.
- [x] (coder) Self-test: a regression test constructs the shipped
  self-referential teacher with an in-block layer already factorized and
  asserts the guard fires with both norms recorded (fail-closed path); a
  second test with the corpus-pinned teacher asserts the objective gradient
  is nonzero once an in-block layer is factorized and the tuned block
  reduces teacher error over T_pre steps; a third asserts no TuneFP call
  site precedes the attention projections' factorization (the legacy
  pre-loop placement is removed and a zero-set invocation aborts as
  vacuous);
  a fourth asserts the teacher weights come from the pristine source
  (fingerprint recorded, not the live module) and that no teacher tensor is
  persisted by the checkpoint writer; a fifth asserts a fidelity-claiming
  run fails on a vacuous event while a diagnostic run continues.
- [x] (coder) Rehearse the stage-trace probe against the fixed bytes (toy
  scope first, then the 7 scoped linears if budgeted): `tune_fp_effect` must
  be nonzero on at least one scoped linear with a real teacher; the abort
  event `tune_fp_vacuous_teacher` must be absent; record the new probe
  sha. Post the rehearsal output.
- [x] (coder) Faithful-run stage map: the rehearsal run record maps every
  executed stage to [NQ] Algorithm 1 lines 0–28 and names any deviation;
  the terminal scale stage is verified against TuneScalesKD semantics
  (line 27: global scale-only KD against the FP teacher, packed binaries
  frozen, T_glob steps) or recorded as a named deviation.
- [x] (bug-hunter) Review on the landed bytes: teacher captured once before
  tuning from the pristine-pinned block weights on the compressed-prefix
  stream, never recomputed from tuned weights and never read from live
  module state; source fingerprint + capture index in the artifact; cache
  is memory-only and re-captured on resume; the vacuous event carries both
  norms; TuneFP fires once per block after q/k/v factorization per
  [NQ] Figure 3, tunable set = the still-FP remainder, and no TuneFP call
  site precedes in-block factorization; guard is
  fail-closed with no trivial-zero carve-out; fidelity-claiming runs
  invalidate on the vacuous event;
  no call site still passes self-outputs as the teacher;
  artifact marker present. Verify the regression tests run green from clean
  checkout.
- [x] (sec-eng) Integrity pass: the pristine-source load adds no new network
  or persistence surface beyond the pinned cache; the abort path cannot
  corrupt in-flight state (snapshot-restore discipline preserved);
  provenance fields (source fingerprint, capture index) provable from the
  artifact alone; the two-norm event distinguishes broken-teacher from
  detached-student on the landed bytes.
  *(sec-eng 2026-10-25: PASS — verdict + command evidence on card
  t_5855d059; one cross-lane spec-accuracy finding routed to planner:
  lane-2 commit ec906a7 persists the fp-stream teacher output `y_b` into
  block checkpoints, so the A2 "never persisted" wording and the
  substring-`teacher` npz-key assertion no longer describe HEAD bytes.)*
- [x] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-tunefp-teacher --strict`; review the delta against the frozen fork
  timing (this fix is owed regardless of the ladder verdict); commit
  (specs + code in one commit), do not archive until a real run shows a
  nonzero TuneFP effect on record.
  *(orchestrator 2026-10-25: validate VALID rc=0, re-run after the
  crystallization entry — still VALID. Delta-vs-frozen-fork review: the
  2026-10-21 user directive already pins this fix as owed before any
  further diagnostic run, so no ladder-verdict dependency exists; nothing
  re-sequenced. Review ticks + verification evidence landed in the landing
  commit scoped to this change dir; archive withheld per this row's own
  condition — a real run showing nonzero TuneFP effect on record is
  outstanding (fire class, @user GO). Findings routed, not
  post-review-patched: t_5e71e281 (balance arm; QUANTIBBLER_DEVICE half
  via t_7190f18b), planner reconcile card (A2 letter vs lane-2 `y_b`
  checkpoint + TuneFP-cycle spec graph + eval-geometry label gap),
  post-fire JSONL card.)*

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." This change and its siblings land before any further diagnostic
runs; the ladder fork fires only after a paper-faithful run exists.

## Reviewer amendments (2026-10-21, sec-eng, folded by planner)

- A1: teacher weights pinned to pristine source, source fingerprint +
  capture index recorded (scenarios `teacher_is_captured_before_tuning`,
  `tune_fp_effect_is_recorded_honestly`).
- A2: teacher cache memory-only in v1, re-captured on resume
  (scenario `teacher_cache_is_memory_only_and_recovered_on_resume`).
- A3: vacuous event carries gradient AND initial residual norm;
  fidelity-claiming runs invalidate on any vacuous event
  (scenario `vacuous_teacher_fails_closed`).
- P1 (planner, 2026-10-21, user-pinned figure): TuneFP placement
  re-ruled to the [NQ] Figure 3 encoding — once per block after Q/K/V
  factorization, tunable set = still-FP remainder; the legacy pre-loop
  placement is the measured defect and is removed; the trivial-zero
  carve-out is dropped (scenarios
  `tunefp_fires_once_per_block_after_attention_factorization`,
  `vacuous_teacher_fails_closed`; figure vendored at
  `docs/papers/assets/block_recon.png`).

## Implementation notes (coder, 2026-10-21)

Landed on `src/quantibbler/ptq/block_recon.py`, `config.py`, `api.py`,
`src/quantibbler/qk_report.py` and `tools/quantize_hf_model.py`. Evidence:
`tests/ptq/test_tunefp_teacher.py` (6 tests), `tests/ptq/test_driver_stage_map.py`
(2 real driver runs), `tests/test_qk_report.py` (4 stage-map/stage-failure tests).

**Rehearsal output (real driver, tiny Qwen3 fixture, `--paper-faithful`):**
exit 0, `status ok`, `ppl_ratio 0.9999975`. TuneFP executed **32 steps on each
of 3 blocks** with `grad_norm` 1.6-2.8e-07, initial residual 1.8-2.2e-04,
residual-after strictly below it, `weight_delta_rel` 0.0027-0.0050,
`tunable = [inner.self_attn.o_proj, inner.mlp.gate_proj, inner.mlp.up_proj,
inner.mlp.down_proj]` — the still-FP remainder, exactly as the Figure 3
placement requires. `vacuous_events == 0`. Teacher provenance on every
capture: `source=frozen_reference_copy`, fingerprint recorded, capture index
recorded. STE per-linear flip counts and scale movement recorded for all
seven scoped linears; zero-crossing linears carry `discrete_flip_pass` or the
`ste_scale_only_declared` marker — none completed silently. The full
Algorithm 1 stage map (lines 2-5, 9-10, 12, 14-17, 19, 20-23, 26-27) is in
`quantize_report.json` with every deviation named.

**Implementation crystallization (decisions the spec left open):**

- **Degenerate block shapes.** A block carrying no attention projection cannot
  honour the figure literally, so the pre-TuneFP group degenerates to the
  first quantizable linear in declaration order; a block whose only linear is
  also the whole remainder records `tune_fp` with
  `skipped = no_still_fp_remainder` instead of firing a pass with an empty
  tunable set. The guard's invariant ("the operative call always has >= 1
  in-block layer factorized") therefore holds for every block shape, and the
  skip is recorded, never silent.
- **Statistics timing (SUPERSEDED by the pre-fire hoist below).** The block's
  RobustDiag statistics were collected once, before the factorization loop, on
  the pristine block — the same bytes the previous (TuneFP-is-a-no-op)
  pipeline measured. That was still the wrong input: the pristine block was
  fed the compressed-prefix stream (Algorithm 1 line 9's TUNING input), not
  the raw calibration set. The Phase-1 hoist replaces the whole per-block pass
  with one pre-loop whole-model pass (see the pre-fire notes below).
- **Residual statistic.** `initial_residual_norm` is the RMS
  (`sqrt(mean squared error)`) of the pre-tune objective on the first step's
  batch, recorded alongside the gradient norm. Gradient-zero + residual-zero
  is the broken-teacher signature (target equals current output);
  gradient-zero + residual-nonzero is the detached-student signature. Both
  are exercised in `test_scn_vacuous_teacher_carries_both_norms_and_invalidates_fidelity`.
- **`paper_faithful` is solver state.** It joins the checkpoint fingerprint
  and is set by `--paper-faithful`, which also switches the stage budgets to
  [NQ] App. C verbatim (T_pre/T_post/T_glob = 8 epochs at 1e-4/1e-5/1e-6).
  Without the flag the driver keeps the gate-shaped budgets (TuneFP at one
  step) and the wrapper prints an advisory saying so.
- **A third report path exists now.** The vacuity guard fails a
  fidelity-claiming run *and* writes a forensically complete report
  (`qk_report.stage_failure_report`, status failed, `paper_faithful: false`,
  both norms). The driver's wiring pin was re-homed accordingly: the identity
  fields moved into a shared module-level `_run_identity()` helper, the pin
  now derives its key set from that helper, asserts the abort payload really
  spreads it, expects exactly three `report` bindings, and carries a second
  handler spec plus two new poison cases for the stage-gate path.
- **`sign_extract_pack` is a new record event**, emitted once per block after
  extraction. `test_scn_block_loop_order` was re-pinned to filter for the
  four Algorithm 1 stage names and assert their order, plus the new event's
  count: the property it guards is stage ORDER, not the record's event count.
- **Doc-accuracy flag (sec-eng).** The flame inventory on Figure 3's left
  panel remains contested between readers (planner counts O/G/U/D, sec-eng
  counts G and U). design.md states only the load-bearing facts; the placement
  ruling depends on neither reading.

**Follow-up, 2026-10-21: the geometry the stages ran at is now named too.**
The stage map proves which STAGES executed; it cannot prove they ran at the
paper's protocol. [NQ] states ONE sequence length (2048) for calibration and
evaluation, so a run that slices the calibration rows into shorter tuning
windows fires every stage with App. C budgets while the protocol's geometry is
gone — a complete stage map over a deviant geometry is the same green-label
class as every other defect this week. `qk_report.geometry_deviations()` now
emits `{kind, detail}` entries for a split tune geometry, a non-protocol
calibration row count (the protocol is 128 rows = 16 batches x 8), and a
non-finite eval geometry, and every report path carries them as
`runtime_deviations` (driven from the shared `_run_identity()`, so the abort
and stage-failure reports cannot drift from the completion one). Evidence:
`test_geometry_deviations_name_a_split_tune_geometry_and_row_count` (lean) and
`test_a_split_tune_geometry_is_named_as_a_runtime_deviation` (real driver run
with `--tune-seq-len` shorter than `--seq-len`).

**Preflight of the proof-scale launch (measured, not estimated).** One
decoder-block forward+backward on Qwen3-0.6B, 16 threads, inside the
`quantibbler-ptq:latest` image: **150.6 ms at (1, 2048)**, 30.7 ms at
(1, 512), 131.1 ms at (4, 512), 519.9 ms at (4, 2048). With App. C budgets
(T_pre = T_post = 8 epochs) the block-step counts are 57,344 at bs=1 /
14,336 at bs=4 over 28 blocks in the whole-sequence geometry, and 229,376 /
57,344 in the 512-window geometry — so the **tuning stages alone are ~2.0-2.2
hours** at every combination measured. The faithful run is an hours-long
launch, not a days-long one. (First, un-warmed measurement of the same call
read 666 ms; the 150.6 ms figure is a warmed three-step average, and the
un-warmed number is recorded here so nobody re-cites it as the cost.)

**Measured preflight of the hoist at the protocol's geometry (done, not
estimated).** The whole-model statistics pass retains its chunk's N^2
attention graph, so the chunk size is the memory knob once the pass runs the
model instead of one block. Inside `quantibbler-ptq:latest` at the wrapper's
32g memcg, one 8 x 2048 batch of token ids through Qwen3-0.6B:
`chunk_rows=4` was **OOM-KILLED**; `chunk_rows=1` **completed at an 18.47 GB
cgroup peak**, 42.3 s for the 8 rows / 197 linears (~1.7x margin under the
cap). `_PHASE1_CHUNK_ROWS` is therefore 1, and the constant carries the
measurement. At the fire's 16 batches that is ~11 min of Phase-1 wall on top
of the tuning budget — inside the quoted 13.5-14.5 h. Evidence: the probe is
`/opt/data/tmp/qk-phase1-probe/probe.py` (run with `CHUNK=4` and `CHUNK=1`);
the constant is pinned by `test_phase1_chunking_is_bounded_for_real_models`.

## Pre-fire amendments (2026-10-21, planner rulings)

- [x] (coder) Fidelity verdict: reported `paper_faithful` = flag AND
  `runtime_deviations` empty AND no mechanism deviation (lb_admm
  polish_rounds > 0, ste discrete flip pass fired, ste scale-only
  re-scope marker, vacuous-teacher event, stage failure, solver !=
  admm); each condition evaluated and recorded in the report. Tests per
  the delta's scenarios (lean where possible, no torch).
- [x] (coder) Phase-1 statistics hoist: one pre-loop `collect_stats` pass
  over the pristine model on the raw calibration batches (streamed D-7
  path); `block_recon` consumes the pre-collected book (no per-block
  stats collection); update the `_STAGE_LINES` phase1_statistics
  deviation entry. Update the crystallization note "Statistics timing"
  (the block weights were pristine but the input stream was the
  compressed prefix — the hoist removes the input-side deviation). Tests:
  block loop consumes the Phase-1 book (identity on a fixture);
  stage-map entry carries no compressed-prefix note.
- [x] (coder) Size the maximal complete-batch local evaluation window at
  seq 2048; verify the probe and gate machinery is unchanged on the larger
  held_out (they consume a slice and a ratio). Record the pinned number here.

**Pinned local quality-gate window: `EVAL_SEQS=18` (NOT full WT2).**
Measured on the pinned corpus + tokenizer, by running the driver's own sizer:
WikiText-2's test shard tokenizes to **298,938** tokens = 145 complete
2,048-token blocks + a 1,978-token remainder. `tokenize_chunks` requires
`n_blocks >= n_chunks * BATCH` (8), so **18** batches is the largest that fits:
144 rows = 294,912 tokens and 294,768 scored next-token predictions. This is
99.310% of complete blocks but **98.653% of source tokens**; it omits one full
block plus the remainder, so it is neither full-set nor paper-table-comparable.
The driver refuses 19 (`wikitext yields 145 blocks of 2048; need 152`). Measured
EVAL_SEQS=14 / 18 / 19 in one run: 14 OK, 18 OK, 19 refused. The probe consumes
a 2-batch head slice and the gate consumes the ratio, so both keep their pinned
semantics on the larger held-out set — and the same sizer path is exercised at
seq 32 by the driver fixture tests.

- [x] (coder) Bug Hunter should-fix: add a separate evaluation record on every
  report path with evaluator kind, source tokens, full blocks, remainder,
  selected blocks/tokens, scored predictions, both coverage fractions,
  selection seed and ordered-index fingerprint, plus
  `paper_table_comparable: false`. For paper-faithful-flagged runs on this
  evaluator, fail preflight unless `eval_seqs == full_blocks // BATCH`
  (18 on the pinned Qwen measurement); diagnostics may use smaller windows
  but record them honestly. Keep evaluation comparability OUT of
  `runtime_deviations` and the method-fidelity conjunction — the spec scopes
  those to method/calibration — and expose the two verdicts side by side.
  Tests: 14 fails faithful preflight, 18 passes with exact counts/coverage,
  19 remains the existing overshoot refusal, and completion/abort/stage-failure
  report shapes all carry the same evaluation record.

  **Built.** `qk_report.evaluation_record()` (torch-free) derives every count
  from `(source_tokens, seq_len, selected_blocks, batch)` so no field can be
  passed in inconsistently; `EVAL_EVALUATOR` names the harness. The driver
  splits `tokenize_chunks` into `_tokenize_blocks` returning
  `(batches, selection record)` — the record comes from the SAME seeded pass
  that built the batches, so the fingerprint cannot drift to a second
  selection — and hashes the ordered picked indices (sha256, 16 hex). It lands
  on `_run_identity`, so the abort and stage-failure builders inherit it via
  `dict(identity_fields)`, and the completion literal carries the same object
  beside `fidelity_conditions`.

  **Preflight** fires before the solve: a faithful-flagged run whose
  `eval_seqs != full_local_window_batches` fails with the number to use and
  the coverage it would have published, before any quantization work. It is
  raised as a `ValueError` (like the sizer's overshoot refusal beside it), not
  `SystemExit`: the driver's single exit path belongs to the gate's verdict
  (`test_report_driver_is_wired_to_the_gate` reds a `SystemExit` in `main`),
  and a config refusal must not imply a completed run.

  **Fixture numbers, measured** (tiny Qwen3 test corpus, seq 32): 6,015 tokens
  → 188 blocks → max window 23 batches → 184 blocks / 5,888 tokens / 5,704
  predictions = 97.87% of blocks, 97.50% of tokens. So the fixture exercises the
  *gapped* case (188 % 8 = 4), not a lucky exact fit, and the three faithful
  driver tests now pass `--eval-seqs 23` (computed in-test by mirroring the
  driver's sizer rather than hardcoding, so a corpus change cannot silently
  desynchronise the preflight).

  Evidence: `test_evaluation_record_states_the_pinned_wikitext_coverage`
  (298,938 → 145 blocks / 1,978 remainder / 18 max / 98.653% of tokens, lean),
  `test_full_block_coverage_still_is_not_a_paper_table_row` (comparability is
  `False` even at 100% coverage),
  `test_an_undersampled_eval_window_does_not_enter_the_method_verdict`
  (40 rows at 27.6% leaves `fidelity_verdict` true and
  `geometry_deviations` empty — the separation, pinned),
  `test_non_completion_report_shapes_carry_the_evaluation_record`,
  `test_a_faithful_claim_is_refused_at_preflight_on_an_undersampled_eval_window`
  (refuses `max-1`, names `max`, no report and no qpk1 left behind),
  `test_an_eval_window_past_the_corpus_is_still_refused_by_the_sizer`
  (`max+1` still hits the sizer's overshoot), and the record assertions inside
  `test_paper_faithful_run_records_the_algorithm1_stage_map`.

- [x] (coder) Hardening — required stages (planner ruling, 2026-10-21;
  closes sec-eng's named residual after the empty-record repro): the
  verdict SHALL deny when the stage record is missing a required stage.
  `fidelity_verdict("admm", [], [], requested=True)` returns true with
  every condition green today — a green label certifying nothing. Add a
  fifth `required` column to `_STAGE_LINES` (data, not a parallel
  constant); mark required exactly the stages the faithful-run test
  asserts executed: `phase1_statistics`, `capture`, `tune_fp`, `lb_admm`,
  `ste`, `sign_extract_pack`, `tune_scales_kd`. Derive run-level counts
  from the same `stages` iteration `fidelity_conditions` already runs via
  `_stage_pair` (both record shapes); count an invocation only when
  `info.get("skipped")` is falsy and, for `tune_scales_kd`, only when
  `artifact` is truthy — a skip-marked event is a recorded skip, not an
  execution (sec-eng amendment); record one named condition per required
  stage (`required_<key>_executed`, invocations > 0) in the conditions
  dict. No new parameters, no stage_map dependency, no new report data —
  the stage map already carries the evidence. Tests (lean where
  possible): a requested record with one required stage dropped returns
  false with that stage's condition named; a tune_fp-only-skipped record
  (`skipped: "no_still_fp_remainder"`, no genuine fire) leaves
  `required_tune_fp_executed` false; an fp-mode `tune_scales_kd` event
  (`artifact` falsy) leaves `required_tune_scales_kd_executed` false; a
  non-requested diagnostic record legitimately skipping `tune_scales_kd`
  (fp measurement) is denied by the flag condition, not newly by a
  required-stage condition; the faithful fixture record (all seven
  fired) stays green; update condition-key assertions in
  `test_qk_report.py` and the `fidelity_conditions` docstring.
  Do not touch `qk-ci-referee/` or `qk-launch-attribution/`.

  **Landed (2026-10-21).** `_STAGE_LINES` carries `required` as its fifth
  column (all seven rows marked; the comment names the column). The verdict
  counts executions inside the same `_stage_pair` iteration it already ran,
  skipping skip-marked events and `tune_scales_kd` payloads whose `artifact`
  is falsy, and reports one `required_<key>_executed` per marked row with no
  `requested` gating. Tests added in `tests/test_qk_report.py`: the empty
  record denies on all seven; a dropped required stage names itself; the two
  skip-marker directions; a per-block skip does not false-deny a run-level
  execution; the inventory's column is DATA (a monkeypatched `_STAGE_LINES`
  moves the condition set with it). The ptq-tier faithful run's exact
  condition dict gained the seven required keys, all true on the real
  driver's tiny-fixture record — evidence the tiny fixture fires all seven.

  **Crystallization (a decision the spec left open).** `stage_map`'s own
  `invocations` / `executed` counting is UNCHANGED: it still counts a
  skip-marked event as an invocation, because the stage map describes what
  the driver EMITTED while the spec's "executed means fired, not merely
  recorded" wording binds the VERDICT. The two can therefore disagree on a
  skip-marked event, and the verdict is the conservative side. Recorded so
  that divergence reads as a declared split, not drift; if a future change
  wants the map stricter, that is a stage-map spec delta, not this task.
- [x] (bug-hunter) Verify on landed bytes: both poison directions
  (dropped required stage → false + named; full faithful record → green),
  plus the skip-marker directions — a tune_fp-skipped-only record and an
  fp-mode tune_scales_kd event each leave their condition false; the
  required column is the only inventory of the set, the empty-record
  repro now denies, and no diagnostic path is newly denied.
- [x] (sec-eng) Re-audit the amended enumeration (the verdict you
  audited gains required-stage conditions): fail-closed still holds on
  garbage records, and the new conditions cannot be satisfied vacuously.
  *(sec-eng 2026-10-25: PASS — 40/40 adversarial probes on landed
  `fidelity_verdict` (control green, empty/truncated/garbage/skip-marker/
  vacuous classes all deny-or-raise); named property: the required-stage
  conditions are payload-blind (a name-only record with capture provenance
  reaches True) — unreachable today because `stage_events` is in-process
  only, but it is the exact property the post-fire JSONL fix must not
  expose; evidence on card t_5855d059.)*

### Post-fire follow-ups (2026-10-21)

- [x] (post-fire card) Resume can only deny, never bless: `stage_events`
  is an in-process list and nothing persisted it, so a faithful fire that
  crashes in the tail (post `sign_extract_pack`, pre completion) and
  resumes completed with a session record holding only `phase1` +
  `tune_scales_kd` — five required-stage conditions false, label false on
  a legitimate run (honest false-deny; cannot manufacture a vacuous green
  because resume only ever shrinks the record and the config fingerprint
  already kills cross-protocol resume). Fix: append-per-event JSONL
  beside the checkpoints, fingerprint-bound by the existing
  checkpoint-dir discipline; seed the sink on resume; missing log =
  today's behavior — never synthesize events. Reading guidance before the
  journal existed: a tail-resumed artifact's `paper_faithful: false` with
  missing stages named is the resume artifact's shape, not a protocol
  deviation; the per-block evidence lives in the fingerprinted checkpoints.

  **Landed (2026-10-25): the journal is `stage-events.jsonl` beside the
  checkpoints.** `checkpoints.py` owns the writer (`append_stage_event`,
  one line per event, opened and closed per event so a crash keeps every
  event up to the crash), the reader (`read_stage_log`: `check_fingerprint`
  first, then a per-line payload-shape check against
  `qk_report._STAGE_PAYLOAD_KEYS`), and the fresh-start reset that rides the
  fingerprint write; `api.py` seeds the sink through
  `resume_stage_events` BEFORE the first event is emitted (a directory with
  no completed block inherits nothing); `tools/quantize_hf_model.py`'s
  recorder writes it while checkpointing is enabled, and `tune_scales_kd`
  now goes through that recorder so the record and the journal carry one
  event set. Two properties this row did not name, recorded so they are not
  discovered later: (a) the fresh-start reset also drops the launching run's
  pre-guard Phase-1 line — safe by construction, since Phase-1 statistics is
  not checkpointed and re-fires on every launch, so a resume never needs it
  from disk; (b) a re-read line whose payload is missing raises rather than
  counting as an execution — the required-stage conditions are payload-blind
  (sec-eng t_5855d059), so the journal is the one place a name-only record
  could have reached them. Evidence: `tests/test_qk_stage_journal.py` (lean:
  the shape table vs the inventory, the name-only-record motivation pin, the
  corruption battery), `tests/ptq/test_stage_journal.py` (a tail crash +
  resume whose condition record EQUALS the uninterrupted run's, and the same
  resume with the journal removed denying the five), and
  `tests/ptq/test_driver_stage_journal.py` (three real driver runs).

## Bug-hunter verification (2026-10-25, this change's two review rows)

Both rows verified on the landed bytes (HEAD `afb220e`, worktree
`/opt/data/quantibbler`, branch `qk-pertarget-cycle`). Environment named:
CPU-only host (no accelerator), `UV_PROJECT_ENVIRONMENT=/opt/venv` shared
across lanes, extras re-synced immediately before each run
(`uv sync --all-groups --extra kernels --extra torch --extra hf` RC=0).
`tests/ci/` and `tests/qk_report/` do not exist (lane-2 design item 4
records the drift); the real paths are `tests/ptq/`, `tests/test_qk_report.py`,
`tests/test_ci_referee.py`.

Commands + counts (RC via `echo RC=$?`, never a filtered pipe):

- `uv run pytest tests/ptq/ tests/test_qk_report.py tests/test_ci_referee.py -q`
  → 57 failed / 331 passed, RC=1. Census: ONE class dominates —
  `RuntimeError: solver backend 'torch' requires an accelerator` raised at
  `solver_backend.resolve_device` via `magnitude_balance` (`lb_admm.py:586`).
- Same + `QUANTIBBLER_DEVICE=cpu` → 52 failed / 336 passed (the env reached
  the solve site only). Same + `SOLVER_BACKEND=numpy` → still 52 failed:
  both knobs are deaf at the balance site.
- Same + `/tmp` pytest plugin repointing `DEFAULT_SOLVER_BACKEND='numpy'`
  (matches what these tests themselves configure: `PTQConfig(solver_backend=
  "numpy", ...)`) → 4 failed / 384 passed; the 4 = the plugin's own expected
  footprint (`test_default_backend_is_torch_and_fails_closed_without_accelerator`
  pins `DEFAULT=='torch'`) + 3 subprocess-tier driver tests whose subprocess
  has no plugin.
- Discriminator: same 3 driver tests + `tests/ptq/test_tunefp_teacher.py` +
  `tests/ptq/test_dual_stream.py` with a `sitecustomize.py` repointing the
  default in the subprocess too → **17 passed, RC=0**. The refusal class is
  therefore attributable to the balance site alone; with it patched at the
  invocation level every teacher/guard/dual-stream regression test is green.
- `uv run pytest tests/test_qk_report.py -q` → **91 passed** in 1.92s, RC=0.
- Independent in-memory probe of `fidelity_verdict` on landed bytes: full
  faithful record → True/all-true; empty record → False with all 7 required
  conditions false; each required stage dropped in turn → exactly its own
  condition falls; tune_fp skip-marked only → `required_tune_fp_executed`
  False with all sibling conditions true; fp-mode `tune_scales_kd`
  (`artifact: False`) → `required_tune_scales_kd_executed` False, siblings
  true; genuine fire + later per-block skip → True (no false-deny); vacuous
  event → `no_vacuous_teacher` False; `_STAGE_LINES.required` is the only
  inventory (condition set == derived set).

### Findings (named severity; evidence file:line)

1. **should-fix (coder/lane owner)** — `magnitude_balance` ignores the run's
   configured solver arm: `lb_admm.py:586` calls `get_backend(backend)` with
   the kwarg unset → `DEFAULT_SOLVER_BACKEND='torch'`, and the production
   caller `block_recon.py:1582` passes no `backend`, while the solve site
   (`lb_admm.py:422`) honors cfg + `SOLVER_BACKEND`/`QUANTIBBLER_DEVICE`.
   Consequences: (a) the ptq tier cannot run green on any accelerator-less
   host (the 52-test class above) even when the tests configure the numpy
   oracle — the recorded "two knobs moved the same 5 tests, the rest deaf"
   pattern, reproduced; (b) on accelerator hosts the balance stage runs torch
   even when cfg says numpy, so the recorded `solver_backend` provenance
   misdescribes the balance stage's arm (attribution-doctrine violation).
   Fix direction: thread the resolved backend into the balance call, keeping
   fail-closed device semantics intact.
2. **should-fix (orchestrator, archive gate)** — the A2 "memory-only" letter
   is superseded on HEAD and its self-test is vacuous: `save_block_checkpoint
   (..., y_b=y_b)` persists the teacher target under npz key `y_b`
   (`checkpoints.py:245-260,300`; call site `block_recon.py:2072`), resume
   restores the fp stream from it (`block_recon.py:2016-2018`) without the
   pristine-source-hash + capture-index binding A2 requires (fingerprint
   covers config+version+git only — `checkpoints.py:86-115`; calibration
   data identity absent, so a cross-data resume mixes the two streams
   silently). The A2 self-test greps archive keys for the substring
   `teacher` (`test_tunefp_teacher.py:374-378`) and cannot see the real key.
   Matches sec-eng's independent finding (lane-2 commit `ec906a7`).
3. **should-fix (planner/orchestrator, spec graph)** — active deltas
   contradict over the TuneFP cycle: qk-pertarget-cycle's delta SHALLs the
   universal idx==0 skip (landed: `block_recon.py:1531`, pinned by
   `test_dual_stream.py:361-370`), while qk-nanoquant-alignment-lane2's
   design records the skip as "the single precise identity-prefix case"
   (block 1 only) and its `guard_rationale_rewritten` scenario forbids
   inherent-vacuity text — yet `block_recon.py:1490-1496` still justifies
   the skip with the single-stream "objective identically zero" argument,
   false under the landed dual-stream capture at blocks ≥ 2 (and
   `_tune_fp`'s docstring line "the teacher is its deepcopy" is the
   single-stream reading). This change's own delta text (compressed-prefix
   teacher, once-per-block) is likewise superseded by lane-2 + pertarget
   deltas; reconcile before `openspec archive`, or `openspec/specs/` will
   carry contradictory SHALLs over the same requirement surface.

Verdicts: review row 1 (teacher-once + guard) — PASS on the core contract
(captured once pre-tuning, pristine-pinned weights via frozen reference copy,
never recomputed from tuned weights, single production call site fed by the
pristine forward, fingerprint + capture index in capture/tune_fp/ste records,
marker present, two-norm fail-closed guard with diagnostic-continue and
faithful-raise, regression suite green under the named invocation) with
findings 1-3 above. Review row 2 (required-stage poison directions) — PASS
clean (91/91 + probe).
