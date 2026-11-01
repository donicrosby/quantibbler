# qk-fix-admm-factor-scale — proposal

## Why

The ADMM arm's serialized binary factor pair `diag(s1) @ Ub @ Vbᵀ @ diag(s2)` overshoots `‖W‖_F` by **8.6–1.2e7** at low rank (18-probe deterministic table; the earlier larger-geometry run measured the same serialized-ratio species at 7.5e2–2.8e8), driven by heavy cancellation in the continuous factors that the per-row mean-absolute scales s1, s2 do not preserve once the ±1 sign matrices Ub, Vb are substituted. The continuous product `U @ Vᵀ` is healthy (0.939–1.350), so the defect lives in the serialization/scale-attribution path, not the ADMM solve. (The `4.1e3–1.6e9` figure quoted in earlier analysis is a *different* species — the factor-norm-pair product `‖U‖_F·‖V‖_F/‖W‖_F` — and is not the serialized overshoot.)

The landed `magnitude_balance` LS α refit (the `target` branch) was the only mechanism reconciling the serialized product with `W` — removing it (per F2's user-ruled disposition) fires the divergence guard (D2 band `[1e-4, 1e4]`) on the rank-3 fixture class (5 of 18 probes already exceed the `1e4` edge uncorrected). The user ruled: **fix the ADMM serialized factor-scale defect first, then land F2's refit removal on top.**

This change makes the serialized product scale-faithful by renormalizing it to the continuous product's norm the balance stage already holds — with no reference to `W`/`target`, so the correction survives F2's removal of that parameter. The LS refit becomes genuinely redundant rather than load-bearing.

## What Changes

- **`magnitude_balance` in `lb_admm.py`:** Replace the LS α refit (the `target` branch) with a target-free symmetric norm-faithfulness correction: after Eq. 8 scale extraction, compute `M = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` and the continuous-product norm `‖UVᵀ‖_F`, derive `γ = sqrt(‖UVᵀ‖_F / ‖M‖_F)`, and apply `s1 *= γ, s2 *= γ`. The correction is recorded in `BalanceResult.gamma`. Eq. 8 ([NQ] Sec 3.2 Step 2-3) remains the initialization contract; the correction is a post-extraction serialization-faithfulness step the record declares.
- **`BalanceResult` dataclass:** Gains a `gamma: float` field carrying the correction factor. `gamma == 1.0` only in the limiting case where the serialized product is already norm-faithful to the continuous product (e.g. a constructed faithful fixture); measured γ at probed ranks is shape-dependent ≈ 1.01–1.17, not 1.0.
- **`target` parameter:** retained in the signature for this change (call sites unchanged) but no longer consumed by the correction; F2 deletes the now-vestigial parameter as an isolated follow-up.
- **Contract reconciliation:** MODIFY the archived `magnitude_balancing_scales` scenario (scales are Eq. 8 mean-abs *then* corrected by γ) and amend the sibling F2 requirement to read "Eq. 8 initialization + scale-faithful serialization correction; LS refit against W removed".
- **Discrimination tests:** New tests under `tests/ptq/test_admm_factor_scale.py` asserting the corrected serialized product is norm-faithful to the continuous product (`‖M‖_F / ‖UVᵀ‖_F ≈ 1.0` within `1e-6`) at rank 3 on the `_SevenLinearBlock` ADMM fixture, and that the uncorrected Eq. 8 product reds on the overshoot — so a `gamma = 1.0` stub cannot pass.

### Impact

- **Invariant-touching:** The serialized product scale now matches the continuous product's norm at initialization. Existing artifacts with the LS refit will have different s1 magnitudes; that's the intended effect — the refit is being retired.
- **Affected specs:** `qk-ptq` (magnitude balancing requirements, amended), and the `qk-pertarget-cycle` F2 requirement (reconciliation text). `qk-divergence-guards` D2 keeps firing on genuine divergence; this fix removes the false-positive overshoot mode that was triggering it.
- **Affected code:** `src/quantibbler/ptq/lb_admm.py` (magnitude_balance, BalanceResult), `tests/ptq/test_admm_factor_scale.py` (new discrimination test). Call sites passing `target=` are unchanged.
- **Effect on SVD arm:** the correction lives in the balance stage shared by both arms and moves every SVD-layer s1/s2 by γ ≈ 1.28–1.39. **No effect on:** the guard band, fixture seeds, the paper's Eq. 8 formula, the ADMM solve path.

### Non-goals

- Does NOT remove the `target` parameter from magnitude_balance — F2's parameter deletion is a follow-up that sequences after this.
- Does NOT change how s1, s2 are extracted from the balanced latents (Eq. 8 remains verbatim).
- Does NOT modify the SVD solver arm or any code outside the magnitude balance stage.
- Does NOT re-band the divergence guard (band stays `[1e-4, 1e4]` at `divergence_band_factor=10000.0`).
- Does NOT re-anchor fixture seeds, thresholds, or acceptance criteria.
- Does NOT address the §17/Q3 `|s1|/|s2|` relative-split mode — γ is symmetric and therefore invariant to that coordinate; it addresses the orthogonal total-scale cancellation overshoot.
- Does NOT export, fire, or produce any qpk1 artifact.
