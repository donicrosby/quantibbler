# qk-nanoquant-alignment-lane0 — design

## Decisive-experiment framing

Lane 0 is a measurement, not a fix. It runs the hypothesis — "the preconditioner divergence is the dominant accuracy term; a healthy preconditioner moves post-ADMM pre-polish recon cos out of the measured ~0.02–0.17 band with NO other change" — through the EXISTING `lb_admm` + `magnitude_balance` pair on both W̃ variants, and produces either a winner-per-layer verdict or a falsified hypothesis. The band 0.02–0.17 is a repo measurement (recorded at `block_recon.py` in the greedy-polish rationale: "cos ~0.02-0.17" on the stored artifacts pre-polish); it is not a paper number.

Gate contract (acceptance, verbatim from the lane spec): "a quantize_report-style JSON + verdict paragraph naming the winner per probed layer, or a falsified hypothesis", and "if the intended preconditioner doesn't dominate, STOP and re-analyze before landing anything". Lanes 1–3 SHALL NOT open without this lane's report attached; lane order is load-bearing (0 gates 1–3).

## Paper anchors (verified against the pinned copy)

- **[NQ] Step 2-1 (:533–556), Eq. 2:** Hessian-aware preconditioning; D̃_in, D̃_out diagonal, constructed from activation and gradient statistics; robustness = clipping to τ_max + Ledoit-Wolf shrinkage γ (Eq. 3); γ ≈ 0.2 (Llama/Qwen), ≈ 0.6 (Gemma/Rnj-1). VERIFIED.
- **[NQ] Algorithm 1 lines 2–5 (:690–710):** global calibration collects (z_in, z_out) per linear, `RobustDiag(·; τ, γ)`. VERIFIED.
- **[NQ] B.1 Eq. 26 (:2771–2779):** `W = D~_out W_FP D~_in`. VERIFIED — the fold is one application per side, at the TARGET, with the inverse fold D̃⁻¹ applied at magnitude-balance unfold ([NQ] Step 2-3 :604–635, Eqs. 7–9).
- **[NQ] B.2 Lemma 1 (:2803–2819):** entries nonnegative, clipped to τ_max; ‖D̃‖₂ ≤ τ_max. VERIFIED — the paper's only robustness mechanism is clip + shrinkage.
- **[NQ] App. C :3011:** gradient statistics derived with gradient checkpointing and a memory-efficient cross-entropy (Wijmans et al., 2025). VERIFIED — the gradient side is CE-loss gradients, not a uniform cotangent.

## Citation corrections (pinned bytes vs the lane paraphrase)

1. **"per-token norm clipping (robust 99.9th percentile)" — SOFTENED.** The pinned copy contains no percentile anywhere (grep `percentile|per-token` = 0 hits). The paper's robustness is clip-to-τ_max + shrinkage ([NQ] Eq. 3, B.2 Lemma 1). The per-token 99.9th-percentile norm clip is therefore an INFORMED IMPLEMENTATION of robust τ selection, listed in the Informed Assumptions section; the spec cites [NQ] Eq. 3 for the clip/shrink shape only, not for the percentile.
2. **"sqrt fold" — SOFTENED.** The paper defines the fold shape (Eq. 26) but never states that the diagonal entries are square roots of the second-moment statistics. The sqrt reading follows the K-FAC diagonal the paper cites (Martens & Grosse, 2015) for the Hessian-weighted objective. Flagged as informed interpretation; the spec scenario pins the intended construction, not a paper quote.
3. **"gradient side from a real CE backward through the head" — ANCHORED, no correction.** App. C :3011 supports CE-based gradient statistics. The divergence being measured is the shipped uniform-ones cotangent (`calibration.py:245`) vs this intended construction.
4. **"~0.02–0.17 band" — repo measurement, not paper.** No paper citation is offered for the band; it is the measured pre-polish baseline this lane exists to move.

## Divergence inventory (shipped vs intended)

| Aspect | Shipped (arm A) | Intended (arm B) |
|---|---|---|
| Fold | `W~ = D_out[:,None] * W * D_in[None,:]`, D = raw stats (mean-squares + grad mean-square on input side) | `W~ = D~_out W D~_in`, D entries = sqrt of robust second-moment statistics, once per side |
| Gradient side | uniform-ones cotangent backward (`model_out.backward(torch.ones_like(model_out))`) | real CE backward: fp teacher logits vs calibration targets, through the head |
| Robustness | clip raw diagonal entries at tail-gated q95, then LW shrinkage γ=0.2 | per-token norm clipping at the robust 99.9th percentile, then clip+shrinkage ([NQ] Eq. 3) |
| Solver | `lb_admm` + `magnitude_balance` | IDENTICAL — existing entry points, production-default `PTQConfig` |
| Polish/STE/export | available downstream | never invoked on the lane |

## Measurement protocol

- **Execution vehicle (user directive, 2026-10-23):** the decisive fire runs ON the torch backend once `qk-solver-backend`'s port lands; the NumPy backend runs as the in-fire oracle on the stratified six-cell grid (one entry per family×depth cell), with escalation to all ten entries on any tolerance breach — per the oracle-grid rule owned by `qk-solver-backend`. "Existing lb_admm" holds in the algorithmic sense — the torch port preserves the algebra and is tolerance-class-gated before the decisive run, and the parity acceptance rides the same fire (one execution delivers verdict + parity; target ~5–10 min).

- **Fixed probe set:** pinned constant in the harness; real production layers loaded from the offline HF cache (the `bench_admm_rank_sweep.py` loading pattern); one entry per (model, block idx, layer name); spans attention and MLP projections across ≥2 blocks. The exact list is pinned at task time (Open Question OQ-1) and is immutable per run — no CLI override.
- **Both arms, per probed layer:** arm A builds W̃ from shipped `collect_stats` + `robust_diag`; arm B builds W̃_intended. Each feeds `lb_admm(w_tilde, cfg)` then `magnitude_balance(p_u, p_v, d_out_tilde, d_in_tilde, target=W)` with the SAME production-default `PTQConfig` and the arm's own diagonals.
- **Stage boundary:** post-ADMM = after `lb_admm` + `magnitude_balance`; pre-polish = before `greedy_polish`. The lane never calls polish or TuneLatentSTE.
- **Four quantities per layer per arm** (fail-closed if any is missing):
  1. post-ADMM pre-polish recon cos — `cos(W, U_bal V_balᵀ)` against the ORIGINAL fp weight, the solver-probe `cos_raw` convention;
  2. consensus residual trajectory — the full `trace.residuals` series from `ADMMResult.trace`;
  3. `n_iter` — `ADMMResult.n_iter`;
  4. `min_eigs` floor — `min(trace.min_eigs)` over both U- and V-side SPD probes.
- Arm-B report rows additionally record the applied percentile and the CE-backward fallback used, if any (layers unreachable by the CE gradient — e.g. weights frozen upstream of the backward graph).

## Report, verdict, and the lanes gate

- **JSON:** quantize_report-style — strict-parseable, provenance-complete (model/block/layer identity, image digest where available, compute dtype, solver identity, per-arm rows), written with `qk_report.json_safe` conventions. No status field, no band, no BPW, `artifact_exported: false`.
- **Verdict paragraph:** names, per probed layer, the arm with the STRICTLY higher post-ADMM pre-polish recon cos. A layer where arm B does not strictly win is named as not-won (ties are recorded as ties and count as non-wins).
- **Domination rule** (defined here; the spec scenario defers to it): arm B (intended) dominates iff it wins a strict majority of probed layers AND its per-layer mean recon cos lies above the top of the measured band (0.17). Anything less → hypothesis falsified.
- **Gate:** on a falsified verdict the report carries the lane gate: STOP and re-analyze before landing anything; Lanes 1–3 SHALL NOT open without this report's gate state.

## Discarded alternatives

- **A/B inside `quantize_model` behind a config knob:** rejected — it would edit shipped pipeline code (violates "lands nothing in the artifact path") and a measurement knob masquerading as a production option is the exact hazard qk-root-cause-ablation D-5 walls against.
- **Single-arm run (intended only):** rejected — without arm A on the same layers there is no control and no way to attribute movement to the preconditioner.
- **Synthetic fixtures only:** rejected — the hypothesis is about the measured production band; fixtures cannot reproduce it. The fixed production probe set is the evidence.
- **Full-model run:** rejected — cost; a fixed per-layer probe set at the solver boundary answers the hypothesis at a fraction of the compute, and the band was measured at exactly this boundary.
- **Adding a `preconditioner_mode` knob to `PTQConfig`:** rejected — any config field is solver state that joins the checkpoint fingerprint (D-6); the lane adds NO solver state, so no fingerprint update is owed.
- **Bootstrap the CE backward from an external loss module:** rejected — a new module inside `src/` would be a shipped-code change; the CE backward lives entirely in the tool.

## Informed assumptions

1. **Per-token 99.9th-percentile norm clip is the intended robust τ.** The paper pins only clip+shrinkage shape ([NQ] Eq. 3). Rationale: per-token outliers are the measured failure class (r137 heavy tails); 99.9 is the robust-estimator convention named in the lane spec. *Wrong-assumption surface:* the arm-B run records the applied percentile and the clipped fraction; if arm B does not dominate, the percentile choice is one re-analysis axis, not silently kept.
2. **Sqrt fold.** D̃ entries are square roots of the robust statistics; the fold is applied once per side per Eq. 26, with the inverse fold consumed by the shipped `magnitude_balance` unfold. *Surface:* an inconsistency between fold and unfold shows up immediately as a magnitude-balance error or a `check_layer_magnitude`-style norm blowup in the arm-B rows (the harness re-uses that gate's ratio as a diagnostic, never as a verdict).
3. **CE-backward fallback for gradient-unreachable layers.** Layers whose weight gradient is None under the CE backward (e.g. layers not on the autograd path of the head) fall back to the activation-only statistic, and the row records it. *Surface:* the fallback marker is required in the row; rows without it fail the completeness check.
4. **Probe-set representativeness.** The pinned set stands in for "production layers" (OQ-1). *Surface:* the report records model/block/layer identity per row; a non-representative pinning is visible in the report and correctable without re-running the machinery.

## Open Questions

- **OQ-1 (owner: planner, OPEN at authoring):** the exact pinned probe set entries. Constraint: real production layers in the offline HF cache, attention + MLP, ≥2 blocks. Pinned as the first implementation task before any measurement.
- **OQ-2 (owner: executor, OPEN):** whether the CE backward at paper geometry (128 seqs × 2048) fits the container memory cap without gradient checkpointing; if not, chunked CE backward with the same per-batch statistic definition is the documented fallback (the `collect_stats` chunk precedent), recorded per row.
- **OQ-3 (owner: planner, OPEN):** which side(s) the per-token norm clip applies to (gradient-side, activation-side, both). Default: both, recorded per side; resolved during harness construction and asserted by `per_token_norm_clip_999`.

## Checkpoint fingerprints / solver state

No new solver state: no `PTQConfig` fields, no solver-stage changes, no artifact-format changes. Therefore no checkpoint-fingerprint update is owed by this lane. If a later lane promotes any lane-0 construction into the pipeline, THAT lane updates the fingerprint per D-6.
