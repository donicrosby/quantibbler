# qk-nanoquant-alignment-lane1 — proposal

## Why

Phase-1 statistics deviate from [NQ] Step 2-1 in three compounding ways, all in `collect_stats` / `robust_diag`:

1. **The gradient statistic sits on the wrong side, from the wrong source.** `_layer_statistic` adds a mean-square WEIGHT-gradient term to the INPUT-side statistic `d_in`, and the gradient itself comes from a uniform `ones_like` cotangent backward (`calibration.py:245`) — not a task loss. The paper's preconditioners are "constructed from activation and gradient statistics" ([NQ] Step 2-1 :544) and, crucially, App. C :3011 pins that the paper DERIVES them from a real cross-entropy backward through the model. The uniform cotangent is a fabricated loss; the input-side placement was the OQ-2 informed choice, now retired.

2. **No clipping before the square.** Outlier robustness is applied per-feature AFTER squaring (the tail-gated q95 auto-cap, our invention in qk-divergence-guards), so a single outlier token's energy is already squared into the statistic before anything can bound it. The paper's own outlier motivation ("empirical estimates can be sensitive to outliers", [NQ] Step 2-1 :546) points at the raw data, not the squared features.

3. **The sqrt fold is missing.** The paper applies the preconditioners once at target formation (`W̃ = D̃_out W D̃_in`, Algorithm 1 line 15, Eq. 26). Our D̃ currently carries no sqrt relationship to the raw statistics; the K-FAC-consistent identification D̃ ≡ sqrt(second moment) (Eq. 2's K-FAC structure) is folded nowhere.

Lane order is load-bearing: this change gates on Lane 0 (`qk-nanoquant-alignment-lane0`, the decisive preconditioner experiment — "0 gates 1–3"), implements the preconditioner that Lane 0's verdict selects, and is itself gated by Lane 3 after it (1 lands before 3). It also depends on `qk-geometry-conformance` (calibration knob units) at the data boundary. It does not block on, or modify, any sibling change.

## What Changes

1. **Real-loss gradient path in `collect_stats`.** `d_out = E[g²]` per output feature, where `g = ∂L/∂z_out` for a real loss: CE backward through the model head for whole-model collection, or a documented caller-supplied head/cotangent for bare-block use (fail-closed when neither exists). `d_in = E[x²]` only — the uniform-cotangent gradient term is removed from `d_in`, and the retirement of the OQ-2 informed choice is recorded in design.md.

2. **Per-token norm clipping BEFORE squaring.** Each token's activation vector and output-gradient vector is clipped by L2 norm to the robust 99.9th percentile of per-token norms (a robust max, never the true max), on both the reference and the `chunk_rows` streamed path, from the same per-batch threshold; online accumulation is permitted. Clipped vectors are squared and accumulated in float64.

3. **Per-feature tail-gated q95 clip retired.** `tau=None` means no per-feature cap; an explicit numeric tau stays verbatim as the paper's Lemma 1 ceiling (assumption A-1). Decision + rationale + cross-change consequence (qk-divergence-guards pins the old default) in design.md.

4. **Sqrt fold at W̃ formation.** `W̃ = sqrt(d_out) ⊙ W ⊙ sqrt(d_in)`, with the SAME sqrt'd vectors handed to `magnitude_balance` for the Eq. 9 recovery — one shared definition. `robust_diag` keeps applying Eq. 3 shrinkage to the clipped statistics and does NOT sqrt.

Preserved by construction: accumulation at the pipeline's declared numerics precision, zero-variance floors (`_TINY`), the `chunk_rows` streamed path's O(feature-width) state and no-materialization pin, and the reference/stream parity scenario at 1e-12 — it stays green under the NEW statistics.

## Impact

- `src/quantibbler/ptq/calibration.py` — `collect_stats` real-loss gradient path (+ head/cotangent parameter), per-token clip before squaring, `_layer_statistic` gradient term removed; `robust_diag` q95 auto-cap branch removed.
- `src/quantibbler/ptq/block_recon.py` — W̃ formation applies the sqrt fold; the same sqrt'd vectors flow to `magnitude_balance`.
- `src/quantibbler/ptq/config.py` — tau docstring rewritten (retired default semantics; explicit tau = A-1 verbatim). No new config fields: the clip quantile is a named module constant, so the checkpoint fingerprint needs no new solver state (audit task confirms).
- `tests/ptq/` — new fixtures pin the construction (gradient on the output side, sqrt fold, per-token clip); `test_preconditioner_tau.py` re-pinned to the retired-then-replaced semantics; r137 block-0 heavy-tail fixture asserts finite-positive termination.
- Fidelity direction: this moves the implementation TOWARD the paper (App. C's CE-derived statistics); the retired pieces (uniform cotangent, input-side gradient term, q95 auto-cap) were the deviations.
- NOT invariant-touching: no kernel, container (qpk1), or storage-format change. Solver inputs change (different statistics ⇒ different W̃), so solver-state checkpoint fingerprints remain valid only if the cfg they pin is unchanged — it is (no new fields); audit task verifies.

## Non-goals

- Does NOT touch the LB-ADMM iteration algebra, rho schedule, SVID, magnitude balancing math, TuneFP/STE, polish, or packing — solver and pack stages are unchanged.
- Does NOT modify the qk-divergence-guards or any other sibling change dir (ADDED-only over the shared qk-ptq capability); the q95-retirement consequence on the sibling's delta text is flagged for the orchestrator, not edited here.
- Does NOT change the qpk1 container, kernels, or server.
- Does NOT commit — the orchestrator reviews and lands.
