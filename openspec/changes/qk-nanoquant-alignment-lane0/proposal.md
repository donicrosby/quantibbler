# qk-nanoquant-alignment-lane0 — proposal

## Why

Post-ADMM pre-polish reconstruction cosine sits in a measured ~0.02–0.17 band on production layers (recorded at `block_recon.py` in the greedy-polish rationale), and every stage so far has been tuned around that observation. Before Lanes 1–3 land anything on the assumption that this band is inherent, Lane 0 runs the decisive experiment:

**Hypothesis: the preconditioner divergence is the dominant accuracy term. A healthy preconditioner moves post-ADMM pre-polish recon cos out of the measured ~0.02–0.17 band with NO other change.**

The shipped construction diverges from the paper-intended one in three ways the code itself documents:

1. **Gradient side is a uniform cotangent, not a real CE backward.** `collect_stats` runs `model_out.backward(torch.ones_like(model_out))` (`calibration.py:245`). [NQ] App. C (:3011) derives the activation- and gradient-based diagonals with gradient checkpointing and a memory-efficient cross-entropy implementation (Wijmans et al., 2025) — i.e. gradients of a real CE loss through the head.
2. **No sqrt fold.** The shipped statistics are mean-squares (plus gradient mean-square on the input side, `_layer_statistic`), applied without the square root; [NQ] B.1 Eq. 26 defines `W = D~_out W_FP D~_in` with diagonals constructed from robust statistics ([NQ] Eq. 3, B.2 Lemma 1) — the sqrt-fold reading is flagged as informed interpretation in design.md.
3. **Robustness is a q95 tail-gated clip, not per-token norm clipping.** `robust_diag` caps raw diagonal entries at a tail-gated q95 (change qk-divergence-guards); the intended construction clips per-token norms at the robust 99.9th percentile — an informed assumption, the paper pins only the clip-to-τ_max + Ledoit-Wolf-shrinkage shape ([NQ] Eq. 3, B.2 Lemma 1 :2803–2819).

If the intended preconditioner dominates, Lanes 1–3 have a target. If it does not, the band is not a preconditioner artifact and re-analysis is mandatory before anything lands. Lane order is load-bearing: **0 gates 1–3**.

## What Changes

- New measurement tool `tools/lane0_precond_probe.py`: builds the intended preconditioner (sqrt fold, CE backward through the head, per-token norm clipping at the robust 99.9th percentile), then runs BOTH W̃ variants — shipped `collect_stats`/`robust_diag` (arm A) and the intended construction (arm B) — through the EXISTING `lb_admm` + `magnitude_balance` under a production-default `PTQConfig`.
- Fixed probe set of real production layers, pinned in the harness; every row records model, block index, layer name.
- Per layer, both arms: post-ADMM pre-polish recon cos, consensus residual trajectory, `n_iter`, `min_eigs` floor.
- A quantize_report-style JSON + verdict paragraph naming the winner per probed layer, or a falsified hypothesis. No verdict bands, no export.
- Tests `tests/ptq/test_lane0_probe.py` (scenario-referencing) guarding the harness contracts.

## Impact

- New files only: `tools/lane0_precond_probe.py`, `tests/ptq/test_lane0_probe.py`. Zero edits to `src/quantibbler/ptq/calibration.py` (collect_stats, robust_diag), `ptq/lb_admm.py` (343–456), `ptq/block_recon.py`, or `ptq/config.py`.
- **Not invariant-touching**: no kernels-contract change, no storage-format change; no solver state changes, so no checkpoint-fingerprint update is owed.
- Reuses the repo's ablation/probe conventions: solver-probe-style JSONL rows, `qk_report` provenance/`json_safe` helpers for the final JSON, offline HF-cache weight loading (the `bench_admm_rank_sweep.py` pattern).
- Gate suite: unchanged — the lane adds only new test files and a new tool.

## Non-goals

- Does NOT land the intended preconditioner in the artifact path, even if it wins. Wiring it into `quantize_model` is Lanes 1–3's business, gated by this lane's verdict.
- Does NOT modify `calibration.py`, `lb_admm.py`, `block_recon.py`, or `config.py`.
- Does NOT invoke polish, STE, or scale calibration; does NOT export a qpk1 or emit an OK/DEGRADED/FAILED band.
- Does NOT tune solver parameters, add config knobs, or add solver stages.
- Does NOT commit anything (the orchestrator commits).
