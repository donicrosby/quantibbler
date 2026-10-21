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
   entry per linear), recompute the block-local objective `‖Wq@x_b − y_b‖`,
   accept on decrease. Cost: budget × block forward — bounded and offline.
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

## Discarded

- **Per-step re-binarization WITHOUT the crossing diagnosis.** Re-binarizing
  `sign(u_latent)` per step changes nothing while latents stay same-side; as
  a standalone "fix" it is cosmetic. The diagnosis (alternative 1) is the
  paper path; re-binarization accompanies it, never replaces it.
- **Continuous relaxation (temperature-annealed sigmoid).** Changes the
  forward the kernels contract with; violates "binaries ±1 by construction".
