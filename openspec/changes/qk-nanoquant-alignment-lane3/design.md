# qk-nanoquant-alignment-lane3 — design

## Regime mechanics: normalized regressors + scale-relative ridge

The continuous update (Eq. 5, [NQ] :2877) solves
`(VᵀV + (ρ+λ)I)Uᵀ = VᵀW̃ᵀ + ρ(Z_U − Λ_U)ᵀ` — the regressor is the fixed
partner factor V (symmetrically U for the V-update), and the ridge is the
absolute shift `(ρ+λ)I` (Eq. 37, [NQ] App. B.4 :2916). Lemma 2 (:2910)
guarantees SPD, but the conditioning bound κ(H_U) ≤ 1 + ‖V‖₂²/(ρ+λ)
(:2945/:2972) is regressor-scale-dependent: a ridge tuned on one layer's
Gram scale is wrong for another's.

**Reading resolution (planner, recorded so the coder lane does not
re-litigate):** "row/col-normalized proxy" in the spec text denotes the
FIXED PARTNER FACTOR — the regressor of Eq. 5 — not the SVID consensus
proxy Z. Regressing against Z is discarded below (rank-1 Gram); Eq. 5's
regressor is V. The continuous update therefore:

1. Builds the design matrix from X = row/col-normalized V (sequential
   row-then-column rescaling, ≤2 passes, O(din·r)) — the Gram XᵀX is
   O(1)-scale with unit-scale diagonal, independent of raw layer scale.
2. Computes the scale-relative ridge: `stabilizer = ρ·mean(diag(XᵀX)) +
   reg`, `reg` from the new `admm_reg` config field (default 3e-2),
   replacing the absolute `(ρ+λ)I` shift in Eq. 37.
3. Solves `(XᵀX + stabilizer·I)U_Xᵀ = XᵀW̃ᵀ + ρ(Z_U − Λ_U)ᵀ` and maps
   the solution back to the unnormalized factor basis (the documented
   linear map U = U_X·D_col·D_row⁻¹ for row/col rescalings), so the
   returned iterate remains a valid factor of the product UVᵀ and the
   consensus/proxy/dual updates continue in the original basis
   unchanged.

At ρ ≈ 0 the stabilizer is ≥ `reg` > 0, so Lemma 2's SPD guarantee holds
at every schedule point without needing ρ > 0 — the reg floor is what
makes `rho_start ≈ 0` admissible. `trace.min_eigs` continues to record
the solved system's minimum eigenvalue each iteration (`lb_admm.py:411,
415`), now with a floor of `reg` instead of `rho_start + admm_lam`.

## Paper grounding (citations grep-verified against the pinned copy)

- Eq. 5 continuous update: `docs/papers/nanoquant-2602.06694v3.md:2877`
  and :584 — verified verbatim.
- Ridge/regressor form: H = VᵀV + (ρ+λ)I at :2741 and :2916 (Eq. 37);
  "λ is the ridge regularization coefficient" :2745.
- Lemma 2 (B.4 well-posedness): :2910-2921, SPD system matrices.
- Conditioning bound: :2945, :2972 — κ ≤ 1 + ‖V‖₂²/(ρ+λ); this is the
  paper-grounded motivation for regressor normalization.
- Linear schedule: App. C :3008 — "We employed a linear ADMM penalty
  scheduler for 400 factorization steps, for each weight matrix across
  all models." Verified verbatim.
- **ρ endpoint values: NOT in the paper.** Grep for endpoint magnitudes
  found none — the old [1e-2→1e2], the current 1e-1 cap, and the new
  ~0→1 are all implementation choices.
- **λ value: NOT in the paper.** λ appears only symbolically (λ≥0,
  :2798). `reg = 3e-2` is an informed assumption (below).

## ρ endpoints ~0→1 and the named tension with qk-fix-admm-solver

Under the scale-relative ridge, `ρ_end = 1` gives a final stabilizer of
`1·mean(diag(XᵀX)) + 3e-2 ≈ O(1)` — the ridge tracks the data term's own
scale, so the consensus term no longer dominates the continuous fit.
qk-fix-admm-solver's `rho_end ≤ 1e-1` cap was derived for the
ABSOLUTE-ridge regime (ridge added to unnormalized Gram matrices of
arbitrary scale, consensus RHS at up to 100× the O(1) data RHS). Its
rationale dissolves here; its requirement text remains on the books and
is superseded only when this change's acceptance measurements land. The
reconciliation (MODIFIED delta naming the regime split) is OQ-3, owned by
the orchestrator — this change is ADDED-only and does not touch that
requirement.

## Frobenius entry normalization — deliberate keep-or-remove

The entry normalization (`target_scale = ‖W̃‖_F`, `w = w/target_scale`,
`lb_admm.py:376-380`) may become redundant: the regressor normalization
plus the scale-relative ridge already make the solve scale-independent,
and the exit `factor_scale = sqrt(target_scale)` rescale is the only
consumer of the split. **Decision protocol:** the measurement task runs
the production layers with and without the entry normalization under the
new regime (solver probe rows, same targets), compares post-ADMM
pre-polish cos and `trace.min_eigs`; the keep-or-remove decision is
DELIBERATE — recorded here with measurements and rationale — and the
default path SHALL NOT silently drop the normalization before that
record exists. **Status: unresolved. Owner: coder lane (measurement
task).**

## Env provenance extension (`QK_RHO_END` pattern)

`rho_end_provenance` (`lb_admm.py:325-341`) returns (value, "env" |
"config") for `QK_RHO_END`, and the solver probe row carries both
(`block_recon.py:201-202`). The new fields follow the same pattern:
`QK_ADMM_REG` for `admm_reg`, and the recalibrated endpoints stay behind
`QK_RHO_END` (a `QK_RHO_START` knob is added only if the coder needs it
— OQ-4). Every new solver-state field that is an override knob gets:
fail-closed float parsing naming the variable, and (value, source)
recorded in the solver probe rows and the run report so no row can be
mislabelled about which regime produced it.

## Checkpoint fingerprint

`config_fingerprint` (`checkpoints.py:48-72`) hashes
`dataclasses.asdict(cfg)` minus `_FINGERPRINT_EXCLUDED`
(`progress_probe`). New dataclass fields therefore join the fingerprint
automatically. The deliverable is the GUARD, not the wiring: a suite test
asserts the new fields are absent from `_FINGERPRINT_EXCLUDED` and that
two configs differing only in a new field produce different fingerprints
(scenario `new_solver_state_joins_fingerprint`) — so a future
exclusion-list edit cannot silently drop solver state.

## Deliberate deviations (each with rationale; wrong-surfaces in the assumptions section)

1. **Row/col-normalized regressor.** The paper's Eq. 5 uses plain V as
   the regressor; no regressor normalization is stated anywhere in the
   pinned copy. Deviation rationale: the κ bound (:2945/:2972) shows
   conditioning is regressor-scale-dependent; normalization is the
   minimal change that makes one ρ schedule serve every layer. Fidelity
   label: implementation-level numerics — the normal-equation structure
   of Eq. 5 is preserved with X in place of V and the documented
   map-back.
2. **Stabilizer ridge replaces `(ρ+λ)I`.** The paper's shift is
   `(ρ+λ)I` (Eq. 37). The scale-relative form is a deviation in FORM,
   not a numeric conflict (the paper pins no λ value); it subsumes λ
   into `reg`. Rationale: an absolute shift cannot be simultaneously
   meaningful across the repo's measured Gram-scale spread.
3. **ρ endpoints ~0→1 supersede the 1e-1 cap.** Named tension, above;
   reconciliation is OQ-3.
4. **Frobenius entry normalization keep/remove.** Pending measurement;
   whichever way it lands, the decision is recorded in this file before
   the default path changes.

## Discarded alternatives

1. **Keep the absolute ridge and re-cap `rho_end` again.** Discarded: the
   defect is the absolute shift against variable Gram scale; another
   absolute cap is another layer-specific guess, and the paper's
   single-schedule design point is lost.
2. **Regress against the SVID consensus proxy Z.** Discarded: the
   two-sided SVID optimum is rank-1 by construction (Z = S ⊙ (g hᵀ),
   [NQ] :592), so its Gram matrix is rank ≤1 and singular — unusable as
   a design matrix. (This is also the reading resolution above.)
3. **Scale-relative ρ on the target norm (`rho × ‖W̃‖_F`).** Discarded:
   already measured and rejected in qk-fix-admm-solver — trivial
   dominance, "converges" at iteration 1 with product unmoved.
4. **Adaptive/per-layer ρ (L-curve or similar).** Discarded: the paper's
   fixed linear schedule (App. C :3008) is the anchor; per-layer
   adaptation adds a heuristic the paper never describes.

## Informed assumptions

1. **`reg = 3e-2` is a reasonable default.** The paper gives λ only
   symbolically (λ≥0, :2798). Rationale: with XᵀX O(1), 3e-2 is a small
   but nonzero well-posedness floor that keeps Lemma 2's SPD guarantee
   at ρ≈0. Wrong-surfaces: `trace.min_eigs` floor too high (stiff solve,
   residual stalls) or too low (ill-conditioned) — visible directly in
   the trace.
2. **ρ endpoints ~0→1 are the right re-calibration.** Wrong-surfaces:
   consensus dominance returns at schedule end (post-ADMM cos drops vs
   Lane-0) or the proxy pull vanishes (residual never tightens) — both
   visible in `trace.residuals`/`trace.rhos`.
3. **The normalized-regressor solve with map-back preserves the ALS
   fixed point.** Wrong-surfaces: post-ADMM pre-polish cos collapses vs
   the Lane-0 baseline, or the pack stage regresses on the same layers
   — the acceptance measurements catch both.
4. **The Lane-0 baseline is stable enough to threshold on.**
   Wrong-surfaces: the threshold is vacuous (improvement below run
   noise); mitigation is the acceptance scenario itself — the threshold
   is set FROM the Lane-0 measurement distribution, not a single row.

## Open Questions

- **OQ-1 (owner: coder):** is the consensus term `ρ(Z_U − Λ_U)ᵀ`
  expressed in the solved basis or after map-back? Resolution: whichever
  keeps `trace.residuals` meaningful, pinned by a test.
- **OQ-2 (owner: coder):** keep or remove the Frobenius entry
  normalization? Resolved by the measurement task, recorded above.
- **OQ-3 (owner: orchestrator):** spec reconciliation of the
  qk-fix-admm-solver `rho_end ≤ 1e-1` requirement with the new regime
  requirements — a MODIFIED delta authored only after Lane-3 acceptance
  measurements land.
- **OQ-4 (owner: coder):** does `QK_RHO_START` become an env knob
  alongside `QK_RHO_END`? The provenance pattern applies to every new
  field; decide during implementation.

## Lane ordering and the acceptance measurement protocol

Lane 0 gates 1–3; Lane 1 (the fixed preconditioner) lands before Lane 3.
No Lane-3 number is quotable as evidence before both have landed —
numbers quoted earlier are marked provisional. The acceptance
measurements (production layers, solver probe on, new regime) record:
per-layer post-ADMM pre-polish cos (`cos_raw` against the original W),
the `trace.min_eigs` floor, and the regime identity (ρ endpoints, `reg`,
normalization, sources) per row; the threshold is then pinned FROM the
Lane-0 baseline distribution in this file — never authored a priori.
