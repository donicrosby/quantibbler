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
- [ ] (bug-hunter) Review on the landed bytes: instrumentation counts both
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
- [ ] (sec-eng) Integrity pass: the flip pass adds no new persistence or
  network surface; deterministic order removes nondeterminism from the
  artifact; accepted flips cannot violate the round-trip byte-exact contract
  (bits are exported from the same tensors the pass wrote).
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-ste-sign-adaptation --strict`; commit (specs + code in one commit);
  do not archive until the fixed stage's flip counts land in a real run on
  record.

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

**Known residual (documented, not fixed here).** When `total_steps < number
of factorized linears` some linears receive a 0-step allocation. They are
recorded with `declared = not_scheduled_zero_budget`, which is a declaration
rather than a silent no-op, but the refinement genuinely did not run for
them. At [NQ] App. C budgets (T_post = 8 epochs over the block's rows) the
allocation is never degenerate; the residual only bites at test-scale
budgets. The faithful run's record is the evidence that closes it.
