# qk-nanoquant-alignment-lane3 — proposal

## Why

400 LB-ADMM iterations currently buy almost nothing over the plain SVD
warm start: the code's own comment records that `solver="svd"` through the
same balance+polish lands within 0.003 cos of the full 400-iteration ADMM
on production layers, while ADMM spends 66–83 s/layer at rank 512
(`src/quantibbler/ptq/config.py:163-166`). That measured claim — "ADMM ≈
warm start" — was made under the CURRENT regime, and the current regime
has a known scale defect. The ridge is the absolute shift `(ρ+λ)I` in the
continuous update (Eq. 5, [NQ] :2877; H = VᵀV + (ρ+λ)I, :2741), so the
conditioning bound of [NQ] App. B.4 (Lemma 2 :2910; κ(H_U) ≤ 1 +
‖V‖₂²/(ρ+λ), :2945/:2972) lets layer-to-layer regressor scale dominate
the solve. An absolute ridge tuned once cannot be right for every layer's
Gram scale — that is why the `rho_end` cap of qk-fix-admm-solver was
needed at all.

Lane 3 replaces the regime, not the algorithm. The continuous update
regresses against the row/col-normalized proxy (the fixed partner factor
that serves as the regressor of Eq. 5), so the Gram matrix XᵀX is O(1),
and the ridge becomes scale-relative: `stabilizer = ρ·mean(diag(XᵀX)) +
reg`, with `reg` a config field defaulting to 3e-2. ρ endpoints are
re-calibrated for this regime: linear ~0→1 over K=400. [NQ] App. C :3008
pins the linear schedule over 400 factorization steps; the endpoints are
an implementation choice the paper never pins (grep-verified against the
pinned copy — no ρ endpoint values exist in it).

Named tension (sibling delta, required by lane discipline): the
qk-fix-admm-solver MODIFIED requirement "The ADMM penalty SHALL follow a
linear schedule over K iterations with epsilon as an early stop" states
`rho_end` SHALL NOT exceed `1e-1` in the normalized regime. That cap was
derived for the ABSOLUTE-ridge regime this change replaces: with the
ridge now multiplying the data term's own scale, the consensus-dominance
arithmetic that produced the cap (consensus RHS ρ(z−l) at 100× the O(1)
data RHS) dissolves, and `rho_end ≈ 1` is safe. This change does NOT
MODIFY that requirement — it ADDs the new regime's requirements, scoped
to the regime, and leaves the reconciliation of the two (a MODIFIED delta
authored only after the Lane-3 acceptance measurements land) to the
orchestrator. Recorded as OQ-3 in design.md.

Lane order is load-bearing: Lane 0 gates 1–3, and Lane 1 (the fixed
preconditioner) lands before Lane 3 — solver-regime numbers only mean
something against the fixed preconditioner.

## What Changes

1. **Normalized regressors + scale-relative ridge.** The continuous
   update regresses against the row/col-normalized proxy; the ridge
   becomes `stabilizer = ρ·mean(diag(XᵀX)) + reg` with `reg` a config
   field (default 3e-2). (`src/quantibbler/ptq/lb_admm.py:343-456`, the
   continuous-update block.)
2. **ρ endpoints re-calibrated for the normalized regime** — linear ~0→1
   over K=400 ([NQ] App. C :3008 for the schedule shape).
3. **Frobenius entry normalization: decide with measurements.** The
   entry `target_scale` normalization (`lb_admm.py:376-380`) may become
   redundant under the new regime; it is kept or removed deliberately,
   and the decision is recorded in design.md.
4. **Provenance and fingerprint.** The `QK_RHO_END` env-provenance
   pattern extends to the new fields (`QK_ADMM_REG`), and the checkpoint
   fingerprint picks up every new solver-state field (automatic via the
   asdict-based `config_fingerprint`; guarded by a suite test so the
   exclusion list can never silently drop solver state).
5. **Acceptance, pinned now:** on production layers, post-ADMM pre-polish
   cos improves materially over the Lane-0 baseline (threshold set FROM
   Lane-0 measurements, not invented); the `trace.min_eigs` floor rises;
   the `solver="svd"` ablation is RE-RUN under the new regime before
   anyone quotes "ADMM ≈ warm start" again.

## Impact

- **Affected specs**: `qk-ptq`, ADDED requirements only (delta under
  `changes/qk-nanoquant-alignment-lane3/specs/qk-ptq/spec.md`; no
  MODIFIED/REMOVED — the rho_end-cap reconciliation is a follow-up).
- **Affected code**: `src/quantibbler/ptq/lb_admm.py` (normalized design
  matrix + stabilizer ridge in the continuous update; solver probe rows
  gain the new fields and sources), `src/quantibbler/ptq/config.py`
  (`admm_reg` field, recalibrated `rho_start`/`rho_end` defaults),
  `tests/ptq/` (regime tests). `checkpoints.py` needs no change — the
  asdict fingerprint covers new dataclass fields; the change adds the
  guard test.
- **Not invariant-touching**: no kernels contract, qpk1 container,
  packing, storage format, or server change; no compatibility impact.

## Non-goals

- Does NOT change the ADMM algebra — Eq. 5 normal equations, dual update
  (Eq. 36), SVID proxy (Eq. 6), magnitude balancing (Eq. 7–9) are
  untouched ([NQ] App. B).
- Does NOT MODIFY any existing requirement, including qk-fix-admm-solver's
  `rho_end` cap; reconciliation is a separate follow-up delta after the
  acceptance measurements.
- Does NOT touch the pack stage (the pack-stage bottleneck is a separate
  lane), the `paper_faithful` solver binding (qk-fix-ste-sign-adaptation),
  stage hyperparameters ([NQ] App. C), or the operational band.
- Does NOT author the Lane-0 threshold value — the acceptance threshold is
  set FROM Lane-0 measurements, not invented in this change.
