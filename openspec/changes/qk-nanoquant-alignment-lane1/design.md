# qk-nanoquant-alignment-lane1 — design

Lane 1 of the NanoQuant alignment ladder. Gated on Lane 0 (`qk-nanoquant-alignment-lane0`, the decisive preconditioner experiment — "0 gates 1–3"; this lane implements the preconditioner that verdict selects); Lane 3 gates on this change (1 before 3). Also depends on `qk-geometry-conformance` at the calibration-data boundary. Authoring per the config.yaml evidence hierarchy: pinned paper first, informed assumptions explicitly marked.

## Paper citations — verified against the pinned copy

Grep-verified against `docs/papers/nanoquant-2602.06694v3.md` (the only authority):

- **K-FAC structure of Eq. 2 — VERIFIED.** :536 "The objective minimizes the Hessian-weighted distortion approximated via Kronecker-factored approximate curvature (K-FAC) (Martens and Grosse, 2015)"; Eq. (2) at :540 is `L(Ŵ) ≈ ‖D̃_out (W−Ŵ) D̃_in‖_F²`; :544 "D̃_in and D̃_out are diagonal preconditioners constructed from activation and gradient statistics."
- **Eq. 3 shrinkage — VERIFIED.** :551 `[D̃_(·)]_ii ← (1−γ)[D_(·)]_ii + γ·mean(D_(·))`; :554-555 γ∈[0,1], ~0.2 for Llama/Qwen, ~0.6 for Gemma-3/Rnj-1. Unchanged by this lane: shrinkage stays on the clipped statistics.
- **Real-CE gradient provenance — VERIFIED.** App. C :3011 "To derive the activation- and gradient-based diagonal preconditioners, we utilized gradient checkpointing and used a memory-efficient implementation of the cross-entropy function (Wijmans et al., 2025)." This is the citation that retires the uniform cotangent.
- **Application/recovery form — VERIFIED.** Algorithm 1 line 15 / Eq. 26: `W̃ = D̃_out W D̃_in`; :605 `Û = D̃_out⁻¹ P_U` (Step 2-3). The preconditioner is applied once, and its inverse is what Step 2-3 divides by.
- **`D_in = sqrt(E[x²])`, `D_out = sqrt(E[g²])` — NOT pinned as formulas.** The pinned copy never writes the functional form of the raw D statistics (only "constructed from activation and gradient statistics", :544, and `RobustDiag(z_in, z_out; τ, γ)`, Algorithm 1 :705). The per-side assignment and the sqrt(second-moment) identification are informed assumptions (IA-1, IA-2 below), consistent with the Eq. 2 application form and the :605 recovery. The user's paraphrase was softened accordingly: the fold IS `W̃ = sqrt(d_out) ⊙ W ⊙ sqrt(d_in)`, and that is exactly the paper's `D̃_out W D̃_in` under the identification `D̃ ≡ sqrt(d)`.
- **Per-token norm clipping — NOT pinned.** The pinned copy's only clip is the per-ENTRY diagonal cap at τ_max (Lemma 1, :2807; `RobustDiag(·; τ, γ)` takes τ). No percentile, no token-axis clip appears anywhere in the copy. The 99.9% per-token clip is IA-3, serving the paper's stated motivation ("empirical estimates can be sensitive to outliers", :546).
- **The paper's τ is an explicit parameter.** Lemma 1 assumes "clipped to a maximum value τ_max"; the auto-derived tail-gated q95 cap was our invention (qk-divergence-guards D1) — its retirement moves us closer to the paper's signature, not further.

## Decisions

### D-1 Retire the OQ-2 input-side gradient term (uniform-cotangent stat out of d_in)

`d_in = E[x²]` only. The OQ-2 informed choice ("mirror the output-side construction on the input side", `calibration.py` `_layer_statistic` docstring) is RETIRED. Rationale: App. C :3011 pins CE-derived statistics; the paper pairs "activation and gradient statistics" (:544) with a two-sided K-FAC product whose natural reading is activation energy on the input side, gradient energy on the output side; a `ones_like` cotangent over a bare-block forward is not a task loss. The weight-gradient object the old code accumulated is `(din,)`-shaped per linear — the wrong object for an output-side statistic, which must be per-OUTPUT-feature.

### D-2 Real-loss gradient path

`collect_stats` gains the gradient source as an explicit, documented parameter:

- Whole model with head: forward → memory-efficient CE with per-batch next-token targets derived from the calibration sequences (teacher forcing) → one backward → per-layer `g = ∂L/∂z_out` captured via output-gradient hooks, per-feature mean of squares in float64. Chunked backward accumulates raw f64 grads across chunks and squares once per batch (same definition as the landed `acc` pattern, moved from weight grads to output-activation grads).
- Bare block: caller supplies a head/cotangent factory (module output → scalar loss). No head and no cotangent → fail closed, naming the missing loss source.
- Both paths feed the SAME per-token clip before squaring.

### D-3 Per-token norm clipping before squaring (99.9th percentile, robust max)

For each token's activation vector (input side, per feature) and each token's output-gradient vector (output side, per feature): compute per-token L2 norms over the batch; the ceiling is the 99.9th percentile of those norms; tokens above the ceiling are rescaled to the ceiling (direction preserved); THEN square and accumulate. Both statistics paths, reference and streamed, compute the threshold from the same per-batch norm vectors so parity at 1e-12 holds by construction (the streamed path keeps per-token norms — O(batch rows) scalars — in addition to its O(feature width) accumulators; this is a bounded, documented extension of the D-7 state envelope, not activation retention). The 99.9th percentile is a named module constant `_TOKEN_CLIP_QUANTILE = 0.999`, not a config knob: Phase-1 statistics are not solver state, no checkpoint fingerprint change (audit task verifies).

### D-4 Retire the per-feature tail-gated q95 auto-cap (decision: RETIRE, not re-scope)

`_TAU_TAIL_GATE` and the `tau=None` → derived-q95 branch are removed. With tau unset, no per-feature cap is applied; robustness is carried by the per-token clip (D-3), which bounds every squared contribution at the cap² before the square. Rationale: (a) the paper's RobustDiag takes τ explicitly and Lemma 1 assumes a fixed τ_max — the auto-cap is not paper-prescribed; (b) one clip layer keeps the bound provenance clean (the effective Lemma-1 bound is cap²(1−γ)+γ·mean after shrinkage); (c) the q95 cap was built for the r137 block-0 heavy tail, and that protection now lives on the token axis where the damage enters (new fixture, scenario `heavy_tail_still_finite_positive`). Explicit numeric tau remains verbatim (A-1).

Re-scoping was considered and discarded: a backstop cap "only when the token clip is inactive" adds a second conditional clip layer whose activation condition would itself need pinning — worse provenance, no added protection.

**Cross-change consequence (for the orchestrator):** the active change `qk-divergence-guards` carries a MODIFIED requirement whose scenarios `preconditioner_tau_default_q95`, `preconditioner_tau_tail_gate_inert` (and `test_preconditioner_tau.py`) pin the OLD default. This change's implementation rewrites those tests to pin the retired semantics; the sibling's delta TEXT still asserts the q95 default until the orchestrator reconciles it (see OQ-3). Per the ADDED-only house rule this change does not edit the sibling dir.

### D-5 Sqrt fold at W̃ formation, shared D̃

`reconstruct_block` forms `W̃ = sqrt(d_out) ⊙ W ⊙ sqrt(d_in)` where `d_out/d_in` are the `robust_diag` outputs (clipped, Eq. 3-shrunk, floored). The SAME sqrt'd vectors are the `d_out_tilde/d_in_tilde` handed to `magnitude_balance` (its `Û = D̃⁻¹ P_U` recovery), so fold and recovery are one definition (D-1 discipline). `robust_diag` itself does not sqrt — it stays the paper's `RobustDiag(z; τ, γ)`. Implementation note: extract the fold into a small helper so the sqrt appears at exactly one site and the fixture can assert it directly.

### D-6 Preserved invariants

float64 accumulators everywhere (config.yaml numerics policy); `_TINY` zero-variance floor in both `collect_stats` and `robust_diag`; the `chunk_rows` streamed path streams BOTH legs (AST structural pin `test_d7_pin_streaming_branch_cannot_materialize_activations` must stay green — the implementer re-checks the branch shape after the edit); reference path (`chunk_rows=None`) keeps its landed materializing construction.

## Discarded alternatives

- **Keep the q95 tail-gate alongside the token clip** — two cascaded clip layers with ambiguous bound provenance; the paper has one τ. Discarded (D-4).
- **Per-feature clip after squaring only (status quo)** — outlier energy is already squared in; the mechanism that produced the r137 incident is exactly what is being replaced. Discarded.
- **True-max per-token clip** — not robust: a single extreme token sets the ceiling at its own norm and nothing is clipped. Discarded (IA-3: percentile is the robust max).
- **Weight-gradient statistic on the output side** — wrong shape (weight grads are `(din,)` per linear) and mixes input dims; output-activation gradients are the K-FAC-consistent object and give per-output-feature statistics directly. Discarded.
- **sqrt inside `robust_diag`** — the fold must live at W̃ formation so `magnitude_balance` consumes the identical vector object; `robust_diag` output stays "the robust statistic" so the Eq. 3 semantics remain testable in isolation. Discarded.
- **New config knobs for the quantile or the loss mode** — Phase-1 statistics are computed before any per-block checkpoint and are not solver state; a constant + a documented parameter keep the checkpoint fingerprint surface unchanged. Discarded.

## Retired assumptions (complete log — acceptance criterion)

1. **OQ-2 informed choice** — gradient statistic mirrored into `d_in` (`_layer_statistic`). RETIRED by D-1. Reason: no pinned basis for an input-side gradient term; App. C :3011 pins the real-CE gradient, which belongs on the output side.
2. **Uniform `ones_like` cotangent as the gradient source** (`calibration.py:245`). RETIRED by D-2. Reason: a fabricated loss; superseded by the paper-pinned CE path.
3. **Tail-gated q95 per-feature auto-cap default** (qk-divergence-guards D1, `_TAU_TAIL_GATE`). RETIRED by D-4. Reason: not paper-prescribed (Lemma 1 takes τ explicitly); replaced by the per-token clip.

No other deliberate deviations are touched by this lane; solver-side assumptions (A-1 explicit tau, A-2/A-3/A-4, SVID, Eq. 5 algebra) are unchanged.

## Informed assumptions

- **IA-1: D̃ ≡ sqrt(d) functional form** (d = second moment, D̃ = sqrt(d)). Not written in the pinned copy; consistent with Eq. 2's application form and the :605 recovery. If wrong, the ADMM target's scale regime shifts and `magnitude_balance`'s recovery mismatches — surfaces immediately as solver-probe cos degradation and divergence-gate trips.
- **IA-2: input side = activation E[x²], output side = gradient E[g²].** The copy names both statistics but not their sides. If wrong, preconditioning mis-scales the two factor axes — surfaces as ADMM residual behavior and the recreation-bar PPL comparison.
- **IA-3: 99.9th percentile per-token clip as the robust max.** Level not pinned (only the motivation, :546). Too tight → statistics lose signal, solver-probe cos degrades on healthy layers; too loose → heavy-tail incidents return as divergence-gate trips. Both have existing detectors.
- **IA-4: per-batch threshold computation for parity.** Both paths compute the ceiling from the same per-batch norm vectors. If violated, the 1e-12 parity test goes red — the scenario exists to catch exactly this.
- **IA-5: CE targets = teacher-forced next tokens from the calibration sequences.** App. C :3011 names the CE function, not the target rule. If wrong, gradient statistics reflect the wrong objective — surfaces as a systematic d_out shift vs. expectation on a fixture with analytic gradients.

## Open questions

- **OQ-1 (OWNER: implementer)** — exact CE target rule for whole-model collection (shift-by-one teacher forcing over sequences, ignore positions where the next token is padding). The pinned copy names only the CE function (:3011). Resolve at implementation, record the rule in the run artifact.
- **OQ-2 — RESOLVED/RETIRED in this change (see Retired assumptions #1).**
- **OQ-3 (OWNER: orchestrator)** — `qk-divergence-guards`'s MODIFIED calibration requirement still pins the q95 default (`preconditioner_tau_default_q95`, `preconditioner_tau_tail_gate_inert`). This change retires that policy; the orchestrator must reconcile the sibling's delta text before its archive (amend or archive-order decision). Not edited here per the ADDED-only rule.
- **OQ-4 (OWNER: implementer)** — whether the streamed path's per-token-norm buffer needs a hard cap for adversarial row counts; the batch size bounds it in practice. Default: no cap; revisit only if a real run shows growth.

## Interaction with siblings

- `qk-nanoquant-alignment-lane0` (Lane 0, the gating experiment): its verdict selects whether this preconditioner lands in the artifact path; this lane implements the winning shape. No code overlap — the experiment probes the preconditioner dominance, this lane rewires `collect_stats`/`robust_diag`/W̃ formation.
- `qk-geometry-conformance`: statistics are collected over the SAME calibration data this change's statistics run over; knob-unit fixes land first (sequence counts, not packed batches). No code overlap beyond the data boundary.
- `qk-fix-admm-solver`: consumes W̃ — the solver's scale-normalization and exit rescale are W̃-norm-invariant, so the sqrt fold changes the target's entries but not the solver contract. No edits.
- `qk-nq-recreation-bar` / `qk-e2e-quality-gate`: verdict machinery reads run artifacts; `phase1_statistics` record fields (chunk_rows) are unchanged; new statistics shift numbers, which is the point of the lane and is recorded as a deviation-class change in the run record, not in the verdict machinery.
