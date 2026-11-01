# qk-fix-admm-factor-scale — design

## 0. Branch decision (stated explicitly)

The reviewer sanctioned two directions. This change takes **branch (i) — re-anchor the correction OFF `target`** — because the balance stage already holds the continuous-product norm and the landed `sqrt(target_scale)` exit-rescale proves the solve carries target scale. Re-anchoring makes the correction mechanically immune to F2's removal of the `target` parameter, which is the failure the round-1 review blocked on. Branch (i) is combined with the branch-(ii) reconciliation: the archived `magnitude_balancing_scales` requirement and the sibling F2 requirement are amended so no contradictory `magnitude_balance` contract remains. The @user standing ruling names the end-state verbatim — *"Eq. 8 initialization + scale-faithful serialization correction; LS refit removed"* — and both contracts are brought in line with it.

**Acceptance (the F2 gate):** post-change, LS-refit removal is a semantic no-op — the serialized product stays in the healthy band `[0.5, 2.0]` of `‖W‖_F` without any refit, because the correction consumes no `target` and the serialized product is norm-faithful to the continuous product the stage already holds.

## 1. Root cause

The ADMM solver produces continuous factors U, V whose product `U @ Vᵀ` is healthy: `‖U@Vᵀ‖_F / ‖W‖_F ∈ [0.939, 1.350]` (18-probe deterministic table, ranks 2–8, shapes 8×16/16×8). But the individual factors are enormous — the factor-norm-pair species `‖U‖_F · ‖V‖_F / ‖W‖_F ∈ [3.3e1, 4.8e7]` (the earlier larger-geometry run measured `4.1e3–1.6e9` for the same species) — so the factors achieve a good product only through heavy cancellation.

When magnitude balancing serializes:

- `Ub = sign(U)`, `Vb = sign(V)` — all entries ±1
- `s1_i = mean(|u_i|)`, `s2_j = mean(|v_j|)` — per-row mean-absolute (Eq. 8)

The ±1 sign matrices do not preserve the cancellation structure of the continuous values: entries that nearly cancel in `Σ_k U_{ik}·V_{jk}` become ±1 entries whose sum `Σ_k Ub_{ik}·Vb_{jk}` behaves differently. The per-row s1, s2 capture average magnitude but not the internal row structure that drives cancellation. Result: the serialized product `M = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` has `‖M‖_F / ‖W‖_F ∈ [8.647, 1.158e7]` — **5 of 18 probes above the `1e4` D2 band edge** — while the continuous product it came from is ≈ `‖W‖_F`. (The earlier run measured the same serialized-ratio species at `7.5e2–2.8e8`; both runs agree the species is chaotic and rank-nonmonotonic.)

The overshoot is rank-driven and chaotic across seeds: at dims 8×16 the raw ratio spans `4.3e1–5.8e3` across seeds 0–2 at rank 6 alone, and `2.4e1–4.6e5` at rank 2 — no monotonic behavior. At production-like rank (≥ 64) the overshoot attenuates, but the rank-3 fixture class reliably fires the divergence guard when the refit is absent (the refit-removal attempt failed 35 tests on exactly this class).

The landed LS α refit (the `magnitude_balance` `target` branch) rescales `s1` by `α = ⟨W, M⟩ / ‖M‖²_F`, which simultaneously corrects the norm AND projects onto span(M). The projection component introduces the systematic cos(W,M) shrink (~0.70–0.93× measured) that F2's disposition identifies as a deviation from [NQ] Eq. 8. But the norm-correction component is load-bearing: without it, the serialized product has no mechanism to match `‖W‖_F`. That is why the refit was masking the defect rather than fixing it.

## 2. Chosen fix: target-free symmetric norm-faithfulness correction

After Eq. 8 extraction produces `(Ub, Vb, s1, s2)` and the balanced latents `U = η Û`, `V = η⁻¹ V̂` are in hand:

1. Build the serialized product: `M = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` (float64)
2. Compute the continuous-product norm `‖UVᵀ‖_F` from the balanced latents the stage already holds (equals `‖Û V̂ᵀ‖_F`, since η cancels)
3. Compute the correction `γ = sqrt(‖UVᵀ‖_F / ‖M‖_F)` and apply `s1 *= γ, s2 *= γ`

The symmetric split (both s1 and s2 receive `γ`) preserves the Eq. 7 equilibrium: `‖V̂‖_F / ‖Û‖_F` is unchanged because both vectors are scaled identically. After correction `‖M‖_F = ‖UVᵀ‖_F` exactly, and since `‖UVᵀ‖_F / ‖W‖_F ∈ [0.939, 1.350]`, the serialized product lands in the healthy band `[0.5, 2.0]` — and well inside the `[1e-4, 1e4]` D2 band.

**The correction consumes no `target`.** It is anchored to the continuous-product norm, which the stage computes from its own latents. This is the property that makes the fix survive F2: when the `target` parameter and the α refit are removed, `γ` is still computed and the product still lands scale-faithful. The `target` parameter is retained in this change's signature (the call site `_block_stage.py` still passes `target=w`) only so F2's deletion of the now-unconsumed parameter is a clean, isolated follow-up.

### Relationship to the §17/Q3 scale-imbalance mode

The §17/Q3 `qk-scale-imbalance` mode is the `|s1|/|s2|` **relative-split** coordinate (the ADMM arm's 13–17-decade imbalance). The symmetric γ is exactly invariant to that coordinate: scaling both s1 and s2 by the same γ leaves every `s1_i / s2_j` ratio unchanged, and the product norm is invariant to the redistribution `(s1·c, s2·c⁻¹)` that moves scale between the two factors — not to a global rescale, which changes the norm by c². γ therefore does not address the relative-split mode; it addresses the orthogonal **total-scale cancellation overshoot** — the absolute norm of the serialized product, not the balance between the two factors.

### Degenerate path (reconciled to raise)

If `‖M‖_F ≤ 0` or `‖UVᵀ‖_F ≤ 0` or the ratio is non-finite, the stage **raises `ValueError`** naming the condition. A silent skip is the silent-pass class the repo's gates refuse; the pre-existing all-zero-factor and non-positive-preconditioner guards already raise, and this guard extends the same fail-closed posture to the correction itself.

### Comparison with the LS α refit

| property | LS α refit (removed) | target-free norm correction (this fix) |
|---|---|---|
| source | `α = ⟨W,M⟩ / ‖M‖²_F` (fit against W) | `γ = sqrt(‖UVᵀ‖_F / ‖M‖_F)` (no W) |
| effect on ‖M‖ | `‖W‖_F · cos(W,M)` | `‖UVᵀ‖_F` (exact) |
| direction | projects onto span(M) | preserves M's direction |
| shrink | ~0.70–0.93× systematic | none (1.0) |
| consumes target | yes (dies at F2) | no (survives F2) |
| paper | not present | not present — implementation detail (recorded) |

Both α and γ are absent from the paper's Eq. 8; the distinction is that α is a *fit against W* (the deviation F2 removes), while γ is a *serialization-faithfulness renormalization* anchored to the continuous product (a distinct mechanism that keeps the serialized factors norm-faithful without any reference to W). This distinction is what reconciles the correction with F2's "no global LS refit against W" prohibition.

### Why not fix the sign extraction itself?

The ±1 sign representation is fundamental to the method: the paper stores packed bits, not continuous values. Alternative decompositions (e.g. per-column normalization before sign extraction) change the factorization's internal structure and interact with the ADMM SVID proxy in untested ways. The norm correction is the minimal change that addresses the measured defect without reopening the solver architecture.

## 3. The `gamma` field on BalanceResult

```python
@dataclass
class BalanceResult:
    U: np.ndarray  # continuous latents (unchanged)
    V: np.ndarray
    Ub: np.ndarray  # binary factors (unchanged)
    Vb: np.ndarray
    s1: np.ndarray  # scales, post-correction
    s2: np.ndarray
    eta: float  # Eq. 7 equilibrium (unchanged)
    gamma: float  # NEW: norm-faithfulness correction factor
```

`gamma` travels in the balance result so the correction is auditable: a `gamma` far from 1.0 signals a layer whose Eq. 8 scales did not produce a norm-faithful product (the overshoot mode). `gamma == 1.0` means the serialized product was already norm-faithful to the continuous product — the limiting case, not observed at any probed rank (measured γ ≈ 1.01–1.17 at r ≥ 64: the raw Eq. 8 product is under-normed there). There is no `target`-gating: the correction is always computed, so the field has no "no-target" branch.

## 4. Discrimination test design

**Test class:** `tests/ptq/test_admm_factor_scale.py::TestSerializedProductScale`

**Fixture:** `_SevenLinearBlock` at rank 3, ADMM solver arm. Constructed deterministically (no global seed draw).

**Assertions (per layer, all seven linears):**

1. Compute the continuous product `M_cont = U @ Vᵀ` and the serialized product `M_ser = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` from the returned `BalanceResult`.
2. **Exact-norm property (the γ distinguishing property):** `‖M_ser‖_F / ‖M_cont‖_F ≈ 1.0` within relative tolerance `1e-6`. This is what distinguishes the γ correction (which makes serialization norm-faithful) from the α refit (which leaves `‖M_ser‖_F / ‖M_cont‖_F` at the α-refit ratio ≈ `0.555–0.931`) and from a no-op `gamma = 1.0` stub.
3. **Healthy band:** `0.5 ≤ ‖M_ser‖_F / ‖W‖_F ≤ 2.0` on every linear layer.
4. `gamma` is recorded, finite, and `> 0` — asserted only after the exact-norm ratio (2), so the trunk red is the ratio failure rather than an `AttributeError`.
5. The divergence guard (band `[1e-4, 1e4]`) passes.

**RED-first on the ratio, not on a missing field:** `test_corrected_product_is_norm_faithful` computes the exact-norm ratio `‖M_ser‖_F / ‖M_cont‖_F` from the RETURNED `s1, s2` *before* any `res.gamma` access, and asserts `≈ 1.0` within `1e-6`. On current trunk (α refit, no γ) the returned scales make that ratio the α-refit ratio ≈ `0.555–0.931` (not 1.0), so the red is the ratio failure — not an `AttributeError` on the missing `gamma` field. The separate uncorrected-product test is a defect canary: it forms the Eq. 8 product *before* γ and asserts `‖M_unc‖_F / ‖M_cont‖_F` is far from 1.0 (measured `8.647–1.158e7`), which holds on trunk AND post-fix because the overshoot is a property of the data, not the code — it documents the defect but does not discriminate. The trivial mutant "keep the α refit, add `gamma = 1.0`" FAILS assertion (2), because its `‖M_ser‖_F / ‖M_cont‖_F` is the α-refit ratio ≈ `0.555–0.931` (equivalently `‖M_ser‖_F / ‖W‖_F = cos(W,M) ≈ 0.70–0.93`), outside the `1e-6` tolerance — the test discriminates the fix from the masked status quo.

**Non-goals of the test:**

- Does NOT check the SVD arm (out of scope)
- Does NOT re-anchor the fixture seeds
- Does NOT modify the divergence guard band
- Does NOT test at production rank (rank ≥ 64 does not fire the divergence guard — raw `‖M‖_F / ‖W‖_F ≈ 0.79–0.81 at 512×512 r64`, inside the `[1e-4, 1e4]` band — so this test gates the low-rank regime where the overshoot is measurable)

## 5. Discarded alternatives

### (a) Anchor the correction to `‖W‖_F` (the round-1 shape)
`γ = sqrt(‖W‖_F / ‖M‖_F)` requires `target`/`W`, which F2 removes. **Rejected:** the correction would die exactly at the end-state it exists to enable (round-1 blocker B-1).

### (b) Per-column normalization before sign extraction
Pre-normalize each column of U (and V) before extracting signs, attributing column norms to s2 (and s1). Changes the factorization's internal structure and the paper's Eq. 8 semantics. **Rejected:** too invasive for a scoped fix.

### (c) RMS-based scale extraction (s1_i = sqrt(mean(u_i²)))
Makes `‖diag(s1) @ Ub‖_F = ‖U‖_F` but does not fix the product cancellation — the sign matrices still don't cancel the same way. **Rejected:** addresses a different part of the problem.

### (d) Apply correction to only one scale vector (s1 *= γ², s2 unchanged)
Simpler but breaks the Eq. 7 equilibrium property. **Rejected:** symmetric split preserves both Eq. 7 and Eq. 9.

### (e) Fix inside the ADMM loop (penalize factor norms)
Adds a regularizer to the ADMM objective that discourages large factor norms. Changes the solver's fixed point, retuning needed. **Rejected:** the solver produces good factors (cos_raw ≈ 0.80); the defect is in serialization, not the solve.

## 6. Reconciliation with the Eq. 8 contracts (round-1 blocker B-2)

This change amends two contract texts so no contradictory `magnitude_balance` contract remains:

- **Archived `magnitude_balancing_scales` scenario** (`openspec/specs/qk-ptq/spec.md`): MODIFIED in this change's delta so it records that the scales are *first extracted* per Eq. 8 and *then* corrected by the serialization-faithfulness factor γ. The returned scales are the corrected scales.
- **Sibling F2 requirement** (`qk-pertarget-cycle` delta, "Eq. 8 scale extraction SHALL be the initialization contract"): its scenarios are amended to read *Eq. 8 initialization + scale-faithful serialization correction (anchored to the continuous product, not W); LS refit against W removed* — matching the @user ruling's end-state. γ is explicitly NOT "a global scalar fit against W", so it does not fall under the crosswalk §3.4a prohibition (which concerns the α refit's fit against W).

The paper's Eq. 8 ([NQ] Step 2-3) remains the *initialization* contract verbatim; the correction is a post-extraction serialization-faithfulness step, recorded as an implementation detail.

## 7. Open questions

None. The 18-probe serialized-scale measurement pins the defect coordinates, the correction is target-free and minimal, and the discrimination test gates the outcome on the exact-norm property.

### RESOLVED (2026-10-29): correction formula and anchor
**Decision:** target-free symmetric `γ = sqrt(‖UVᵀ‖_F / ‖M‖_F)` applied to both s1 and s2, anchored to the continuous-product norm (not `W`). Rationale: survives F2's `target` removal; exact norm-faithfulness; no directional shrink; preserves Eq. 7 equilibrium.

### RESOLVED (2026-10-29): gamma field type and degenerate path
**Decision:** `float` on `BalanceResult`, always computed (no `target`-gating). Degenerate input (`‖M‖_F ≤ 0`, `‖UVᵀ‖_F ≤ 0`, or non-finite ratio) raises `ValueError` — never a silent skip.

## 8. Known limitations

- The correction addresses the norm overshoot but not the directional quality of the binary product. If Ub and Vb produce a product M whose direction is far from W, the norm correction makes `‖M‖_F = ‖UVᵀ‖_F` but `cos(W,M)` may still be low. This is the job of downstream tuning stages (STE, KD), not initialization.
- At very low rank (r < 3) on very skinny matrices, the correction may still produce `γ` far from 1.0. The discrimination test gates the rank-3 case because that's where the divergence guard fires; lower-rank fixtures are not part of the test suite.
- The correction requires computing `M = diag(s1) @ Ub @ Vbᵀ @ diag(s2)` in float64, an `O(dout · din · r)` operation — negligible compared to the ADMM solve (400 Cholesky iterations).
