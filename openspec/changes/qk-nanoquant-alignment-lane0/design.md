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

## Post-incident record (2026-10-23 box OOM)

The first Lane 0 fire OOM-killed the whole box ~10 min after launch. Diagnosis, fix, and proof protocol, pinned here so the record is self-contained:

- **Mechanism:** the CE-gradient stats pass materialized full-vocabulary logits under autograd, unchunked — per batch: logits (8×2048×151936 bf16 ≈ 5 GB), non-contiguous reshape copy (≈5 GB), fp32 cast (≈10 GB), CE gradient (≈10 GB), plus the retained trunk activation graph (≈9–10 GB) — peak ~35–40 GB/batch. On the 8060S iGPU, "VRAM" is system RAM granted through `/dev/kfd` by the kernel driver — **outside the container's memory cgroup** — so the 32G fence could not hold it, and the ROCm caching allocator never returns freed blocks, so 16 back-to-back batches only climbed.
- **Launch regression:** the fire was launched hand-rolled, outside `run-ptq-container.sh`, the wrapper that carries this box's launch discipline. Lesson applied: every Lane 0 launch rides the wrapper conventions; the wrapper gains a lane0 mode rather than being bypassed.
- **Fix (landed `9aa37dc`):** head chunking per 256 tokens under `torch.utils.checkpoint` off a full-context trunk forward, chunk losses summed, mean gauge on the total, ONE backward — bitwise-identical statistic (fixture: `np.array_equal`, no tolerance; the forbidden per-chunk-backward shape has a must-differ test) and head peak cut ~64×.
- **Proof protocol (amended):** red run on parent bytes (byte-identical to the killer, verified: zero code-line diff) bounded to one batch — artifact is the MEASURED peak from `torch.cuda.max_memory_reserved` (the instrument that sees kfd-granted memory), not an expected kill; green run = full 16-batch stats phase; refire requires the green peak to beat the red peak with solve headroom under 32G. Operational nets: free-RAM baseline beside the artifact, hard wall-clock timeout, no co-tenants mid-run. Watchdog silence is weighted as no evidence for the kfd class.
- **Second death — 2026-10-23 ~22:21Z, and it falsified the arithmetic.** The red rehearsal (one packed batch, parent bytes, 32G fence set, 64G host-free) killed the host a second time before writing any peak. Conclusion, now measured not inferred: the retained **trunk activation graph is ≳35–40 GB**, not the ~9–10 GB the byte arithmetic assumed — the suspected mass is SDPA falling to the math path and materializing per-head attention matrices for backward (28 layers × 8 heads × 2048²), and the head chain's ~25 GB sits on top. A per-batch peak larger than the host cannot be measured on the host: check (a)'s "measured peak" is retired, the incident record IS the artifact, and the red protocol executes never again. Consequence adopted: the stats pass SHALL gradient-checkpoint the trunk blocks too (bounded to one block, statistic bitwise unchanged, ~30% slower) — so the fixed path's peak is bounded by construction BEFORE it is ever fired, and green-1 on an uncheckpointed trunk is ruled out as a third dice roll. Green protocol: rehearsal starts at ONE batch on the checkpointed build, 16-batch endurance only if the measured peak ≤15G, then refire on headroom under 32G.
- **Hook-doubling invariant (superseded by one-count gating):** checkpoint recompute re-runs each block's forward, so the first landed build's accumulation hooks fired twice; the statistic survived only because it was consumed as a RATIO (×2 exact in binary FP). The amendment replaces that folklore with structure: the accumulator gates on `torch.is_grad_enabled()` — first pass (reentrant checkpoint, grad disabled) is skipped, the recompute pass counts, and every row is counted exactly once — so raw `sum_sq` IS the correct magnitude and the equivalence fixture asserts raw-sum identity, not just the mean. Red cases: an ungated accumulator and a non-reentrant checkpoint both double the raw sums and fail loudly.
- **Fenced anomaly (from the `803c3fc` bisect):** whole-batch head + checkpointed blocks diverges `d_out_raw` by ~0.8% while raw sums and rows stay identical — a different gradient, not FP noise, in a configuration no shipped path executes. Fenced per spec: the harness fails closed on the combination until root-caused; bug-hunter's hypothesis (single-backward-owns-whole-trunk hook registration asymmetry, chunked runs eight rounds vs one) is the open lead.
- **Measured rehearsal numbers — green-1c, 2026-10-24, commit `dd9e2e5`, COMPLETED end-to-end, box never stressed:** per-phase device-memory peaks: tokenize 4.4 / model_load 6.1 / capture 7.7 / stats_shipped **7.2** / batch1/1 (intended arm) **8.6GiB**. Overall peak **8.6GiB** against the 15G gate — PASS with room. The retention class that killed three boxes is dead: the shipped arm went 50.1GiB → 7.2GiB. Min device free during the run: 115.4GiB (floor 20G never approached); watchdog.log empty. Gate state: green-16 endurance run authorized by protocol; decisive fire follows it.
- **Green-16 death record — 2026-10-24, commit `1476248`, killed ~3.4 min in at batch2/16 of Arm B:** batch1/16 completed cleanly; the run died mid-batch-2 — the FIRST multi-batch Arm B execution in campaign history (the e2e stub's `max_batches=1` made it unwritten territory by design). Floor-switch exonerated (120.8GiB free at death), no kernel OOM, no trace (`--rm` destroyed it). The two-batch CPU fixture (`b201672`) passed green — the fault is device-only. Capture term peaked 15.67GiB, which is why the gate reads against the statistics-arm peaks with the capture term annotated as harness plumbing. Class ruling committed to spec: no fire executes a batch count the CPU suite has never run — necessary, not sufficient, as this death proves (fixture was green; the real path still died).
- **Container witness landed (`b9af299`, verified 2026-10-24):** wrapper-level `tee` to `$OUT/container.log` — no in-container change, no image rebuild, the log survives `--rm`. The exit-status capture had a real defect found by the two-second missing-image check, not by reading: POSIX `sh` has no `PIPESTATUS`, and under `set -eu` the naive `{ "$@"; echo "$?" > f; }` never writes anything because the failing container kills the subshell before the echo runs — `$?` after the pipeline would be tee's status. The correct form `{ rc=0; "$@" || rc=$?; echo "$rc" > f; }` is errexit-exempt. Lane0 modes fail closed on an absent or empty witness, and an `unknown` rc fails closed too. Independently re-verified by measurement: unpullable rocm-tagged image → wrapper exit 125 (docker's own code, propagated), `container.log` captured the pull failure verbatim, `launch.env` gained `container_exit_code=125` and `container_log=`; the rocm-tag gate and the digest-pin gate both refused the fake image before the docker block — the wrapper's own fail-closed layers working as designed.

## Measurement protocol

- **Execution vehicle (user directive, 2026-10-23):** the decisive fire runs ON the torch backend once `qk-solver-backend`'s port lands — all 20 solves torch-only, no NumPy runs anywhere outside unit tests. "Existing lb_admm" holds in the algorithmic sense — the torch port preserves the algebra and its unit-test parity suite is the acceptance gate before the decisive run (one execution delivers the verdict plus measured row-times; target ~5–10 min).

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
