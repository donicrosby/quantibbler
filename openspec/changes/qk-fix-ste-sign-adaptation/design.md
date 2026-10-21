# Design: qk-fix-ste-sign-adaptation

## The measured constraint

STE gradients reach `u_latent`/`v_latent` (the passthrough contract holds),
but a sign flip requires a latent to cross zero. The trace shows latents move
5–33% relative without crossing zero on any of the 7 linears in 4,096 steps —
the warm-start latents sit far from the decision boundary, so gradient steps
never reach it. U/V refinement is therefore inert *in the configuration this
pipeline actually runs*, not in theory.

## Alternatives

Per the user directive below, the paper's own mechanism ranks first; the
numbered alternatives are the fallback ladder, ranked closest-to-paper
first:

1. **Paper mechanism — STE on continuous latents, diagnosed.** Instrument
   per-step flip counts, lr/gradient norms, and warm-start latent margins;
   diagnose why latents never cross zero within [NQ] App. C parameter space
   (lr, T_post, init margins) and make crossings happen. This is the "to
   the letter" path and is attempted FIRST per the user directive.
2. **Magnitude decay / temperature toward zero, re-binarize per step.**
   Closest to the STE letter among mechanisms that force crossings; the
   measured geometry says 4,096 steps was not enough without help, so this
   adds an explicit decay schedule rather than unbounded steps.
3. **Bounded discrete sign-flip pass.** After the STE steps, propose flips
   of `Ub[i,j]`/`Vb[i,j]` one entry at a time (budgeted, e.g. one pass per
   entry per linear), recompute the block-reconstruction objective
   `‖block_tuned(x_b) − y_fp‖²` (the proposed flip applied to the linear
   under refinement, every other block parameter held fixed), accept on
   decrease. Cost: budget × block forward — bounded and offline.
   Deterministic sweep order (row-major, seeded tie handling) so runs are
   reproducible. Recorded as an **explicit deviation** if adopted (the
   paper has no discrete flip pass).
4. **Re-scope as scale-only.** Rename the stage, drop `u_latent`/`v_latent`
   from the parameter set, update spec prose to say "scales refined, binaries
   frozen at warm start". Honest, cheap, and defensible — but it abandons the
   paper's joint-refinement letter, so it is the last resort.

## Decision gate (pre-pinned)

The fix is accepted only when, on the traced 7 linears: **(A)** the
instrumentation reports per-linear flip counts, and **(B)** one of the
alternatives achieves ≥1 accepted flip with decreasing block output error
(alternatives 1–3), or the stage is officially re-scoped (alternative 4)
with code, spec, and run record all saying scale-only. A stage that reports
zero flips with no re-scoping fails the gate — exactly the silent no-op
this change exists to kill. Whichever alternative is adopted is named in
the run record; alternatives 3 and 4 additionally record their deviation
from the paper letter.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence for the alternatives ranking: the paper's own mechanism
— STE on continuous latents per [NQ] Eq. 10, with per-step re-binarization
on the latent sign — gets FIRST priority. Order becomes: (1) instrument and
diagnose why latents never cross zero (lr, steps, warm-start margins, all
within [NQ] App. C parameter space) and make the paper mechanism flip
signs; (2) magnitude-decay toward zero with re-binarization (closest to the
STE letter); (3) the discrete flip pass, recorded as an explicit deviation;
(4) re-scope as scale-only. Gate (A)/(B) acceptance criteria are unchanged.

## Amendment (2026-10-21, planner, from user-pinned Algorithm 1)

The user pasted Algorithm 1 into the room (2026-10-21); it matches the
pinned corpus (`docs/papers/nanoquant-2602.06694v3.md` lines 690–825) on
every binding element. It pins what this change's spec left unstated: line
19 `TuneLatentSTE(b, X_b, Y_b; T_post)` consumes the SAME captured teacher
(X_b, Y_b) as line 12 `TuneFP` — captured once per block at lines 9–10.
Folded into the delta: the STE stage's objective uses that shared,
pristine-pinned, memory-only cached capture, and the artifact records the
same teacher provenance as TuneFP. Any construction where TuneLatentSTE
recomputes its own target from live state reproduces the
self-referential-teacher defect this fix class exists to kill. [NQ] Figure
3's right panel (vendored at `docs/papers/assets/block_recon.png`) confirms
the placement: the flame sits on O alone AFTER O's factorization (scissors,
middle panel) — STE refinement is per-layer, on the just-factorized layer,
inside the block loop.

## Amendment (2026-10-21, planner, from sec-eng required changes S1–S2)

sec-eng found two spec defects before build; both folded as rulings:

- **S1 — objective dimensional coherence.** The previous delta pinned
  `‖Wq@x_b − y_b‖` — a layer-local residual (q_proj output, not
  d_model-shaped on GQA k/v) scored against the block-level teacher, which
  typechecks only by coincidence on square projections and scores noise.
  Ruling: ONE objective, block-level `‖block_tuned(x_b) − y_fp‖²` where
  `block_tuned(x_b)` is the block forward with the proposed flip applied to
  the linear under refinement and every other block parameter held at its
  current value. It typechecks for every factorized linear (the block
  output is always d_model-shaped), it is consistent with the tunefp
  design's objective family and Algorithm 1 line 19's block-level
  signature, and it needs no new capture: `(x_b, y_fp)` is the shared
  corpus-pinned capture, provenance recorded as in TuneFP. The layer-local
  alternative (per-layer teacher intermediates) is discarded — it invents a
  capture the algorithm never names.
- **S2 — coverage and parameter set.** Refinement covers EVERY factorized
  linear in the block (the seven scoped linears: attn q/k/v/o, mlp
  gate/up/down), scheduled per-layer on the just-factorized linear per
  Figure 3's right panel; the line-19 T_post budget is split
  deterministically across the block's factorized linears (equal split,
  remainder to the earliest linears in fixed order) and the per-layer
  allocation is recorded. Parameter set ruling: **scales are IN** — the
  joint set is (U, V, s1, s2) per [NQ] Eq. 10 and the Figure 3 caption
  (corpus line 577, "fine-tunes the continuous latent matrices and
  scales"). Flip counts instrument the U/V sign crossings; scale movement
  is recorded per linear alongside. A build that omits scales from the
  parameter set records a named deviation, never silence.

## Discarded

- **Per-step re-binarization WITHOUT the crossing diagnosis.** Re-binarizing
  `sign(u_latent)` per step changes nothing while latents stay same-side; as
  a standalone "fix" it is cosmetic. The diagnosis (alternative 1) is the
  paper path; re-binarization accompanies it, never replaces it.
- **Continuous relaxation (temperature-annealed sigmoid).** Changes the
  forward the kernels contract with; violates "binaries ±1 by construction".

## Amendment (2026-10-21, planner, pre-fire: T_post re-ruling + faithful-path semantics)

Four pre-fire rulings, all read from the corpus on my own bytes. The first
REVOKES the S2 equal-split budget rule; the scheduling ruling (per-layer,
Figure 3 right panel) stands.

- **Budget re-ruling — the refinement unit is the LINEAR; full budget per
  linear.** Evidence the S2 fold predates: Step-3 prose (corpus line 641)
  names the refinement unit "the current target linear layer" (singular);
  Eq. 10's optimization variables are ONE linear's (U, V, s1, s2) against
  the block mapping; App. C (corpus line 3007) attaches the factorized
  stage's 8 epochs to that per-linear refinement — at batch 1, 8 × 128 =
  1024 steps per linear. Algorithm 1's line 19 is a block-level CALL
  because the objective needs the block mapping; the schematic elides the
  internal per-layer loop exactly as it elides every subroutine's
  internals (same elision class as the TuneFP placement ruling). Ruling:
  each scoped linear's TuneLatentSTE consumes the FULL
  `epochs × (n // batch)` steps; the landed block-pool equal split (~146
  steps/linear ≈ 1.14 epochs) under-budgets STE 7× against the paper and
  is REMOVED from the faithful path. The per-layer schedule record
  survives (each linear records its full scheduled steps). Driver diff:
  `steps=total_ste` instead of `steps=alloc[idx]`; the alloc array and
  the zero-budget residual disappear. Bug Hunter's reading was correct;
  my earlier split ruling predates anyone reading the Step-3 prose
  closely.
- **Faithful-path zero-crossing semantics.** The paper's method is: STE 8
  epochs per linear, then sign+pack whatever signs result (Algorithm 1
  lines 20-23). A zero-crossing outcome IS the paper's outcome — a
  measurement, not a re-scope. On the faithful path: no discrete flip
  pass fires and no scale-only declaration fires; the record carries
  `declared: paper_ste` with the measured zero-crossing count. The ladder
  (discrete pass, re-scope marker) remains the diagnostic path's
  fail-honest machinery. Note: the rehearsal's zero-crossing geometry was
  measured on the POLISHED path (latents at exactly ±1); the paper's path
  has never been measured — the faithful run is its first measurement.
- **Polish off on the faithful path.** The greedy bit-flip polish is an
  extra-paper discrete stage (named in the stage map); it refolds latents
  to exactly ±1, making the paper's STE structurally unable to flip
  signs. The paper's Stage 2 ends at the balanced continuous latents (Eq.
  9); Stage 3 refines them. Ruling: `polish = factor_mode == "binary" and
  ablation != "untuned-signs" and not cfg.paper_faithful`; the lb_admm
  event records polish_rounds 0 so the artifact proves it; export signs
  the refined latents at lines 20-23. Production runs keep polish
  (solver quality is a production concern; production-runs.md's
  svd/polish rulings are untouched).
- **Solver ruling (launch line).** Algorithm 1 line 16 = LB-ADMM; App. C
  pins a linear penalty schedule over 400 factorization steps per weight
  matrix — the pipeline's `lb_admm` implements exactly that (linear rho
  schedule over `cfg.admm_max_iter`, default 400). The production line's
  `solver=svd` is the measured ABLATION (lands within ~0.003 cos of ADMM
  on production layers); an ablation verdict about artifact quality is
  not a fidelity ruling. The faithful run fires `SOLVER=admm`. Cost:
  66-83 s per linear × 196 linears ≈ +4 h (in the fire budget below).
  Production path (svd) untouched.

## Fire pins (pre-fire, 2026-10-21)

- **Launch line:** `RANK=512 SOLVER=admm SEQ_LEN=2048 PAPER_FAITHFUL=1
  OUT=/opt/data/quantibbler/artifacts/qwen3-0.6B-r512-admm-paperfaithful
  sh tools/run-ptq-container.sh`. TUNE_SEQ_LEN unset (tune geometry =
  calibration geometry, Algorithm 1 line 9); STE_FLIP_PASS_BUDGET unset
  (0); CALIBRATION_SEQS=16 (128 rows = App. C); max-admm-iter default 400
  (App. C); `EVAL_SEQS=18`, the measured maximal complete-batch window
  of the local evaluator. It is 144/145 complete blocks (294,912 /
  298,938 source tokens = 98.653%), not "full WT2" and not asserted
  paper-table-comparable; the separate evaluation record pins coverage.
- **Budget:** TuneFP ≈1.0 h (7,168 steps × 519.9 ms measured at (4,2048))
  + STE ≈8.4 h (200,704 steps × 150.6 ms measured at (1,2048)) + LB-ADMM
  ≈4 h (66-83 s × 196 linears) + Phase-1 stats and PPL evals — **~14 h
  wall, quoted as 13.5-14.5 h**.
- **Label rule (sec-eng, folded):** no unwaived runtime deviation is
  compatible with `paper_faithful: true`. The reported label is a
  computed verdict (tunefp delta: the flag AND empty runtime_deviations
  AND no mechanism deviation), never a forwarded flag.
  Informed-assumption markers are documentation, not deviations.
- **Contingency:** only a real memcg OOM at the paper's batch points
  (measured preflight: (4,2048) = 519.9 ms inside the container) makes a
  tune-geometry split defensible — and then with the OOM evidence
  attached to the named deviation and the label false.
