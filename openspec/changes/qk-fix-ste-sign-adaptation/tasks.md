# Tasks: qk-fix-ste-sign-adaptation

- [x] (coder) Implement per the spec delta: per-step flip-count
  instrumentation in `_tune_latent_ste` (both factors, per linear), plus
  scale (s1, s2) movement recording, for EVERY factorized linear in the
  block (all seven scoped: attn q/k/v/o, mlp gate/up/down) — never a
  partial O/G/U/D-only path. Refinement is per-layer on the just-factorized
  linear (Figure 3 right panel); the line-19 T_post budget splits
  deterministically across the block's factorized linears (equal split,
  remainder to the earliest in fixed order) and the per-layer allocation is
  recorded. Flip acceptance uses ONE objective: the block-reconstruction
  `‖block_tuned(x_b) − y_fp‖²` with the proposed flip applied to the linear
  under refinement and every other block parameter held fixed — typechecks
  for every linear including GQA k/v — on the shared corpus-pinned
  `(x_b, y_fp)` capture that feeds TuneFP ([NQ] Algorithm 1 lines 9–10,
  19), pristine-pinned, memory-only, provenance (source fingerprint +
  capture index) recorded. Then the mechanism ladder per design.md — (1)
  diagnose and fix why latents never cross zero within [NQ] App. C
  parameter space (lr, T_post, init margins), attempting the paper's own
  STE mechanism first per the user directive; (2)-(4) fallbacks per
  design.md, each recorded in the run record with deviation status.
  Binaries stay ±1 by construction; no qpk1 format change.
- [x] (coder) Self-test: a toy block engineered with latents near zero must
  show ≥1 sign crossing under the diagnosed paper mechanism; a toy block
  with far-from-zero latents must either cross after the adopted mechanism
  or emit the scale-only declaration — no silent zero-flip completion.
  Shape coverage: the toy block MUST include a non-square projection (GQA
  k/v style, out-dim ≠ d_model) and the flip-acceptance objective must
  typecheck and behave on it — the dimensionally-incoherent objective this
  fix replaced passed by coincidence only on square projections.
  Reproducibility test: same seed, same sweep → identical flip sequence.
- [x] (coder) Rehearse the stage-trace probe against the fixed bytes: post
  per-linear flip counts for the 7 scoped linears and the accepted-flip
  deltas; record the new probe sha and the adopted alternative per linear.
- [x] (bug-hunter) Review on the landed bytes: instrumentation counts both
  factors and cannot be bypassed; refinement covers all seven scoped
  linears (attn q/k/v/o, mlp gate/up/down) and scales (s1, s2) are in the
  parameter set with movement recorded; the acceptance objective is the
  block-reconstruction `‖block_tuned(x_b) − y_fp‖²` and typechecks on GQA
  k/v shapes (verify by reading the control flow, not just square-shape
  tests); the objective's `(x_b, y_fp)` is the same captured teacher as
  TuneFP (pristine-pinned, provenance recorded), never recomputed from live
  state; the re-scoping marker is greppable in code, spec, and run record;
  zero-flip completion without a declaration is impossible (prove by
  reading the control flow).
- [x] (sec-eng) Integrity pass: the flip pass adds no new persistence or
  network surface; deterministic order removes nondeterminism from the
  artifact; accepted flips cannot violate the round-trip byte-exact contract
  (bits are exported from the same tensors the pass wrote).
- [x] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-ste-sign-adaptation --strict`; commit (specs + code in one commit);
  do not archive until the fixed stage's flip counts land in a real run on
  record.
  <!-- 2026-10-25 orchestrator (t_a3cd5cee): validate → "Change
  'qk-fix-ste-sign-adaptation' is valid", RC=0. Review ticks (bug-hunter,
  sec-eng) + marker literal named in the delta + crystallization entry land
  in the same commit. Archive WITHHELD per the run-record condition (fire
  class, @user GO). Gates: `uv run pytest -q` 412 passed/16 skipped RC=0
  (lean tier — torch/transformers absent from the shared venv at this
  sync; ptq tier cleanly skipped; full-extras tier = 57F/331P/3D on
  CPU-only hosts, proven cross-lane root cause 8d1bcc0 → card
  t_5e71e281, this change's file 13/13 under the numpy-default probe per
  both review lanes). cargo test 24/0 RC=0; clippy -D warnings RC=0;
  cargo fmt --check RC=0; maturin build RC=0; pre-commit (3 touched
  paths) RC=0. -->

## Implementation notes (coder, 2026-10-21)

Landed on `src/quantibbler/ptq/block_recon.py` and `config.py`. Evidence:
`tests/ptq/test_ste_sign_adaptation.py` (7 tests) plus the per-linear records
in the real driver run recorded in `qk-fix-tunefp-teacher`'s implementation
notes.

**What the code does now.** `_tune_latent_ste` is invoked once per
just-factorized linear with that linear's share of the line-19 T_post budget
(equal split, remainder to the earliest linears in declaration order, the
allocation recorded per linear), and returns per-linear, per-factor sign-flip
counts, the per-step flip trace, the relative latent movement, and the scale
(s1, s2) movement. Flip acceptance in the bounded discrete pass is judged
against the block-reconstruction objective `||block(x_b) - y_b||**2` with the
proposal applied to the linear under refinement and every other block
parameter held at its current value; the sweep order is deterministic
(ascending |latent|, ties by (factor, row, col)).

**Mechanism ladder status (design.md alternatives 1 -> 3 -> 4).** Alternative
1 — the paper's own STE — is implemented and instrumented: on a fixture with
latents at the decision boundary (|latent| ~ 1e-9) the stage flips both
factors within 8 steps. On the rehearsal run every scoped linear finished
with ZERO continuous crossings in 18-19 steps (the measured geometry:
warm-start latents sit far from zero), so the ladder fell through to
alternative 3, the bounded discrete flip pass, which accepted flips on four
of seven linears; the remaining three carry the `ste_scale_only_declared`
marker (alternative 4). No linear completed with zero flips and no
declaration.

**Named deviation.** The discrete flip pass is not in the paper; every record
carries `deviation: paper has no discrete sign-flip pass (design alt. 3)`, and
`ste_flip_pass_budget` defaults to 0 (pass disabled -> the scale-only
declaration path), so enabling it is an explicit, recorded act.

**Known residual (superseded by the 2026-10-21 pre-fire re-ruling).** The
per-linear budget re-ruling (design.md amendment) removes the pooled
allocation entirely: every scoped linear receives the full
`epochs × (n // batch)` steps, so the zero-budget case cannot occur on any
path. The `not_scheduled_zero_budget` declaration and the alloc machinery
are deleted with the re-ruling.

## Pre-fire amendments (2026-10-21, planner rulings)

- [x] (coder) Per-linear full budget: delete the block-pool alloc split;
  each linear's `_tune_latent_ste` call receives `steps=total_ste`
  (`epochs × (n // batch)`); the per-linear schedule record lists the
  full scheduled steps; the `not_scheduled_zero_budget` residual and its
  note are removed. Update the stage-map `ste` deviation string
  (`qk_report._STAGE_LINES`): per-layer scheduling follows Figure 3's
  right panel, each linear consumes the full App. C budget — no split
  language. Test: a 7-linear block at test budgets gives every linear
  the full step count; assert no allocation array survives in the
  record.
- [x] (coder) Polish gate: `polish = factor_mode == "binary" and ablation
  != "untuned-signs" and not cfg.paper_faithful`; the lb_admm event keeps
  recording polish_rounds (0 on the faithful path). Test: paper_faithful
  config -> polish_rounds 0 recorded and the STE stage receives
  continuous latents (fixture whose balanced init is not all ±1);
  non-faithful binary path unchanged (polish_rounds 4).
- [x] (coder) Faithful zero-crossing semantics: under `cfg.paper_faithful`
  skip the discrete-pass/scale-only fallback; record
  `declared: paper_ste` with the measured zero crossings. Test both
  paths: faithful records the measurement with no rescope marker and no
  flip pass fired; diagnostic path keeps the ladder (discrete pass at
  budget>0, scale-only declaration at 0 accepted flips).
- [x] (coder) Rehearse the whole faithful surface on the tiny Qwen3
  fixture with PAPER_FAITHFUL=1 SOLVER=admm: full per-linear budgets,
  polish_rounds 0, declared paper_ste on every linear, empty
  runtime_deviations, label true with all verdict conditions recorded.
  Gates: `uv run pytest` full suite, `openspec validate
  qk-fix-ste-sign-adaptation --strict`, pre-commit on the touched paths.

## Pre-fire implementation notes (coder, 2026-10-21)

Landed on `src/quantibbler/ptq/block_recon.py` (budget + faithful-path
semantics), `src/quantibbler/qk_report.py` (stage-map `ste` deviation text)
and `src/quantibbler/ptq/api.py` (Phase-1 hoist, from the sibling change).

- **The pooled split is gone.** `alloc` and its `divmod` are deleted; every
  scoped linear's `_tune_latent_ste` call takes `steps=total_ste` where
  `total_ste = max(stage_latent.epochs * (n_rows // batch), 1)`. The `ste`
  event carries `per_linear` + `total_steps` and NO `allocation` key; each
  linear's record carries `scheduled_steps` (== `total_steps`) beside
  `steps`. The `not_scheduled_zero_budget` declaration and the zero-budget
  branch are removed with the re-ruling: no linear can complete unscheduled.
- **Polished off, proven by the artifact.** `polish` gained
  `and not cfg.paper_faithful`; the `lb_admm` event still carries the round
  count, and `fidelity_verdict` reads it (a nonzero count denies the label).
  Evidence: `test_scn_paper_faithful_disables_greedy_polish` (greedy_polish
  spied — zero calls on the faithful path, one per scoped linear on the
  production path) and `test_scn_paper_faithful_hands_ste_continuous_latents`
  (the latents reaching STE are NOT all ±1).
- **Faithful zero crossings.** On `cfg.paper_faithful` the stage declares
  `paper_ste` unconditionally — no discrete flip pass fires, no scale-only
  marker appears, and `ste_flip_pass_budget` is inert (asserted with a
  nonzero budget). The diagnostic path keeps the ladder byte-for-byte.
- **Rehearsal (real driver, tiny Qwen3, `--paper-faithful --solver admm
  --max-admm-iter 400 --calibration-seqs 16`):** exit 0, `ppl_ratio
  0.99999985`, `paper_faithful: true` with all eight verdict conditions true
  and `runtime_deviations: []`. Per block: `total_steps 1024` (8 epochs × 128
  rows at batch 1), all seven scoped linears `scheduled_steps 1024`,
  `steps 1024`, `declared paper_ste`, zero crossings measured on all seven;
  `polish_rounds 0`; `lb_admm` ×3; `phase1_statistics` executed once with
  `deviation: null`. Evidence: `test_paper_faithful_run_records_the_algorithm1_stage_map`
  (real driver run) plus `test_a_faithful_claim_on_the_svd_ablation_is_denied_by_the_computed_verdict`
  and `test_a_split_tune_geometry_is_named_as_a_runtime_deviation` (the
  computed verdict denying a claim on each axis).

> **Annotation (coder, 2026-10-22 — author date per the repo clock; see
> `qk-pertarget-cycle/design.md` §9 on the phantom date).** The
> rehearsal above was emitted under the **pre-re-rule Figure-3 reading** (one
> TuneFP pass per block, post-Q/K/V). Its `paper_faithful: true` and its "all
> eight verdict conditions true" describe the verdict machinery *as it then
> stood*, and the reading they certified has since been overruled (ruling (a),
> `qk-pertarget-cycle`). The number is **not retracted** — it honestly records
> what the instruments said at the time — but it is not evidence about the
> per-target cycle, and the branch's own corrected machinery now contradicts
> it. Annotate, don't rewrite: this is what was believed then.

**Seen and NOT waived, for @planner:** the closed mechanism list does not
name `max_admm_iter != 400`, so a faithful-flagged run with a shortened
LB-ADMM schedule (App. C pins 400) still computes `paper_faithful: true`.
Measured, not theorised: `--solver admm --max-admm-iter 2` on the tiny
fixture diverges in block 0 (`||Wq||_F/||W||_F = 5.3e-08`), and the same run
at 400 is clean. The rehearsal therefore pins 400. Adding a seventh
condition would change the spec's enumeration, which belongs to you.

> **Annotation (coder, 2026-10-22 — repo-clock author date; see
> `qk-pertarget-cycle` §9).** "Still computes `paper_faithful: true`" above is a
> statement about the closed mechanism list **under the pre-re-rule Figure-3
> reading**, which is what the verdict was computed against. The *observation*
> stands (`max_admm_iter != 400` is not named by the closed list); the *label*
> it grants no longer describes this branch's machinery. Still open for
> @planner, unchanged.
