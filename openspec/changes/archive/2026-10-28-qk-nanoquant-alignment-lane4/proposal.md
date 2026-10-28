# qk-nanoquant-alignment-lane4 — proposal

## Why

Both block-tuning losses are plain MSE. TuneFP (`block_recon.py:667`) and TuneLatentSTE (`block_recon.py:871`) minimize `(out - y_b).pow(2).mean()` — uniform over output features. The pinned paper states otherwise: [NQ] App. C :3012 — "During block reconstruction, we employed a weighted MSE function, utilized in previous quantization works (Boža and Macko, 2026; Kim et al., 2025)." The weighting machinery the paper attaches to block reconstruction already exists in our Phase 1: Algorithm 1 lines 2–5 collect per-linear `(z_in, z_out)` statistics and RobustDiag turns them into `(D~_in, D~_out)` ([NQ] :703–705), preconditioners the paper describes as "constructed from activation and gradient statistics" ([NQ] Step 2-1, Eq. 2 :536–540). The statistics are collected (`calibration.py`, `LayerStatsEntry.d_out_raw`) and consumed at ADMM entry — but never applied to the block-level tuning objectives. `TUNE_FP_LOSS_MARKER` still declares `tune_fp_loss: informed_assumption` while a paper-anchored weighted form is sitting unused in the Phase-1 book.

## What Changes

1. **Weighted block-tuning losses.** TuneFP and TuneLatentSTE losses become weighted MSE: the per-feature weight `w` is the block's terminal linear's (down_proj or arch-equivalent) Phase-1 output statistic — `robust_diag(d_out_raw, tau, gamma)`, the same `D~_out` the solver already uses at ADMM entry ([NQ] Algorithm 1 lines 2–5; Eq. 2). Loss form: `mean(w ⊙ (B̂(X) − Y_b)²)` with `w` detached and applied in the stage's compute dtype.

2. **Uniform fallback, recorded — never silent.** When the canonical terminal linear is absent (no `down_proj` and no arch-equivalent terminal name in the block), the stages fall back to uniform weights AND emit a `tune_loss_uniform_fallback` recorder event naming the block and the missing canonical layer. A config knob `block_loss_weight: "terminal_grad" | "uniform"` (default `terminal_grad`) selects the mode explicitly; the absence-triggered fallback is distinct from an explicit `uniform` run and always records the event either way.

3. **Stage records carry the weighting mode.** The `tune_fp` and `ste` records gain `loss_weight` (mode actually used) and `loss_weight_source` (terminal linear name or fallback-event reference). `TUNE_FP_LOSS_MARKER` is updated: the weighted-MSE form is now paper-anchored at [NQ] :3012; only the weight source remains an informed assumption, and the marker says so.

## Impact

- `src/quantibbler/ptq/block_recon.py` — the two loss sites (code anchors :567–700 TuneFP and the STE loop at :1239–1330's stage), record fields, marker text.
- `src/quantibbler/ptq/config.py` — new `block_loss_weight` field with validation. It is solver state and joins the checkpoint fingerprint automatically (`config_fingerprint` hashes `dataclasses.asdict(cfg)` over the full config) — the cross-cutting "fingerprints updated for any new solver state" discipline is discharged by construction and pinned by a test.
- `tests/ptq/` — red-first tests per scenario (see tasks.md).
- Fidelity label: this moves the pipeline CLOSER to the paper (a paper-stated weighted form replaces a self-declared informed assumption); it is not a deviation. Not invariant-touching: no kernels contract, no storage format, no interchange change.

## Non-goals

- Does NOT change the ADMM solver, the Eq. 2 preconditioning at ADMM entry, polish, packing, or Phase 3 (KL scale calibration).
- Does NOT add new statistics collection — the weight rides the existing Phase-1 book; no extra forward/backward passes.
- Does NOT enumerate arch names beyond the alias map; architectures without a recognized terminal name take the recorded uniform fallback.
- Does NOT retune any stage hyperparameter or touch the paper-faithful gate semantics.
- Lane-order note: this change (lane 4) is independent of lane 2 and both must land before any new production run; lane 0 gates 1–3 as before.
