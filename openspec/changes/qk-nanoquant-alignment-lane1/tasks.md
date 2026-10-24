# qk-nanoquant-alignment-lane1 — tasks

Gates on Lane 0 (`qk-nanoquant-alignment-lane0`, the decisive preconditioner experiment). Tests precede implementation per capability; scenario IDs in test names. Do NOT commit — the orchestrator reviews and lands.

## 1. Red-first: output-side gradient fixture

- [ ] Add `tests/ptq/test_calibration.py::test_scn_gradient_on_output_side` (scenario `gradient_statistic_on_output_side`): a fixture whose loss is `0.5·‖z − t‖²` on the layer output with fixed `t`; assert `d_out_raw` equals the analytic per-feature mean of `(z−t)²` and `d_in_raw` equals plain per-feature mean of `x²` (no gradient term). Run `python -m pytest tests/ptq/test_calibration.py -k gradient_on_output_side` and record RED against today's uniform-cotangent input-side construction.

## 2. Red-first: sqrt-fold fixture

- [ ] Add `tests/ptq/test_block_recon.py::test_scn_sqrt_fold_at_formation` (scenario `sqrt_fold_at_formation`): assert the W̃ formation helper yields `sqrt(d_out)[:,None] * W * sqrt(d_in)[None,:]` elementwise against an independent reference. Record RED against today's direct (unsqrt) fold at `block_recon.py:1350`.

## 3. Red-first: per-token clip fixture

- [ ] Add `tests/ptq/test_calibration.py::test_scn_per_token_clip_before_squaring` (scenarios `per_token_clip_before_squaring`, `clip_quantile_is_robust_max`): a batch with one extreme token (norm many × the 99.9th percentile); assert the extreme token contributes exactly cap² to the squared-sum, and the clip threshold is the 99.9th order statistic (not interpolated). Also assert `d_in_raw == E[x²]` — see coupled task 4. Record RED (no clip exists today). **This task and task 4 SHALL land in one commit.**

## 4. Implement the real-loss gradient path — coupled with task 3

- [ ] `src/quantibbler/ptq/calibration.py`: CE backward through the model head for whole-model collection ([NQ] App. C :3011), documented caller-supplied head for bare blocks (`head=`: callable receiving layer output, returning scalar loss; gradient flows via `.backward()` — the harness derives the cotangent), fail-closed when neither exists; `d_out = E[g²]` from output-activation gradients (hook-captured, f64, chunked backward accumulates raw grads and squares once per batch); `d_in = E[x²]` only — remove the gradient term from `_layer_statistic` and update `LayerStatsEntry` docs. **One commit with task 3's fixture and this implementation.** Run task-1 and task-3 tests — GREEN.

## 5. Implement per-token norm clipping before squaring

- [ ] Same file: 99.9th-percentile per-token L2-norm clip on activation vectors and output-gradient vectors, on BOTH the reference and the `chunk_rows` streamed path, from the same per-batch norm vectors; `_TOKEN_CLIP_QUANTILE = 0.999` named module constant; streamed path keeps O(feature width) + O(batch rows) norm state. Run task-3 test — GREEN.

## 6. Retire the tail-gated q95 auto-cap

- [ ] `robust_diag`: remove the `_TAU_TAIL_GATE` branch (tau=None ⇒ no per-feature cap); explicit tau stays verbatim (A-1). Rewrite `tests/ptq/test_preconditioner_tau.py` to pin the retired semantics (scenarios `tau_unset_means_no_per_feature_cap`, `explicit_tau_stays_verbatim`). Update `src/quantibbler/ptq/config.py` tau docstring. Run `python -m pytest tests/ptq/test_preconditioner_tau.py` — GREEN.

## 7. Sqrt fold at W̃ formation

- [ ] `src/quantibbler/ptq/block_recon.py`: extract the W̃ formation into a helper applying `sqrt(d_out) ⊙ W ⊙ sqrt(d_in)`; hand the SAME sqrt'd vectors to `magnitude_balance` as `d_out_tilde/d_in_tilde` (one shared definition). Run task-2 test — GREEN.

## 8. r137 block-0 heavy-tail termination fixture

- [ ] Add `tests/ptq/test_calibration.py::test_scn_heavy_tail_still_finite_positive` (scenario `heavy_tail_still_finite_positive`): block-0-style embedding-scale activations with rare extreme tokens through `collect_stats` (per-token clip) + `robust_diag(tau=None)`; assert the pipeline terminates, every preconditioner entry is finite and positive, and the preconditioned target is finite. — GREEN.

## 9. Parity stays green under the new statistics

- [ ] Run `python -m pytest tests/ptq/test_root_cause_ablation.py -k "streamed"` — the 1e-12 reference/stream parity must hold under the new statistics (scenario `parity_green_under_new_statistics`), and the D-7 AST structural pin (both legs streamed, no `_reduce_tokens` in the streamed branch) must still pass (`streamed_path_still_streams`); update the structural pin only if the branch shape legitimately changed, with a comment naming why.

## 10. Design/retirement and fingerprint audit

- [ ] Verify design.md's Retired assumptions log is complete (OQ-2 input-side gradient term, uniform cotangent, q95 auto-cap) and matches the code; confirm no new config fields and that `checkpoints.config_fingerprint` needs no update (Phase-1 statistics are pre-checkpoint, not solver state); run `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane1 --strict` — VALID.

## 11. Gates and record (last task — no commit)

- [ ] Run the full gate suite: `python -m pytest tests/` (torch/hf extras, quote the environment), `cargo test`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check`, `maturin build`; then `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane1 --strict`. Record results in design.md. Do NOT git commit — the orchestrator reviews the full lane set and commits once.
