# Tasks: qk-fix-ste-sign-adaptation

- [ ] (coder) Implement per the spec delta: per-step flip-count
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
- [ ] (coder) Self-test: a toy block engineered with latents near zero must
  show ≥1 sign crossing under the diagnosed paper mechanism; a toy block
  with far-from-zero latents must either cross after the adopted mechanism
  or emit the scale-only declaration — no silent zero-flip completion.
  Shape coverage: the toy block MUST include a non-square projection (GQA
  k/v style, out-dim ≠ d_model) and the flip-acceptance objective must
  typecheck and behave on it — the dimensionally-incoherent objective this
  fix replaced passed by coincidence only on square projections.
  Reproducibility test: same seed, same sweep → identical flip sequence.
- [ ] (coder) Rehearse the stage-trace probe against the fixed bytes: post
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
