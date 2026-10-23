# qk-nanoquant-alignment-lane4 — tasks

## Phase 1 — Config knob + fingerprint pin (tests first)

- [ ] (coder) `tests/ptq/test_config.py` — red-first: add the pin that `block_loss_weight` exists, defaults to `"terminal_grad"`, accepts only `{"terminal_grad", "uniform"}`, and that a config differing ONLY in `block_loss_weight` from another produces a different `config_fingerprint` (scenario `loss_weight_knob_joins_checkpoint_fingerprint`). Command: `pytest tests/ptq/test_config.py -q`. Expected: the new asserts FAIL on the knob-less config.
- [ ] (coder) `src/quantibbler/ptq/config.py` — add `block_loss_weight: Literal["terminal_grad", "uniform"] = "terminal_grad"` with `__post_init__` validation and a docstring noting: paper-anchored weighted MSE ([NQ] App. C :3012), weight source is informed assumption IA-1 (design.md), solver state → joins the checkpoint fingerprint. Command: `pytest tests/ptq/test_config.py -q`. Expected: green including the new pins.

## Phase 2 — Weighted MSE in TuneFP (test first)

- [ ] (coder) `tests/ptq/test_tune_loss_weighting.py` — red-first fixture for scenario `weighting_reaches_the_loss_poisoned_weight_changes_loss` (TuneFP half): a toy block with a terminal linear; run `_tune_fp` twice, identical inputs, second run with one entry of the weight statistic poisoned to a sentinel (e.g. ×1e6); assert the two first-step losses differ — a poisoned weight vector changes the loss. Command: `pytest tests/ptq/test_tune_loss_weighting.py -q -k tune_fp`. Expected: FAIL (loss is plain MSE; poison has no effect).
- [ ] (coder) `src/quantibbler/ptq/block_recon.py` — TuneFP loss site (:667) becomes `(w[None, :] * (out - y_b[idx]).pow(2)).mean()` with `w` = detached `robust_diag` of the terminal linear's `d_out_raw` (uniform when mode is `uniform`; fallback path per Phase 4). Command: `pytest tests/ptq/test_tune_loss_weighting.py -q -k tune_fp`. Expected: green.
- [ ] (coder) Confirm no regression in the existing teacher guard: `pytest tests/ptq/test_tunefp_teacher.py -q`. Expected: green (guard semantics untouched).

## Phase 3 — Weighted MSE in TuneLatentSTE (test first)

- [ ] (coder) `tests/ptq/test_tune_loss_weighting.py` — extend the poisoned-weight fixture to the STE half (scenario `weighting_reaches_the_loss_poisoned_weight_changes_loss`): run `_tune_latent_ste` twice, poisoned statistic on the second run, assert the first-step losses differ. Command: `pytest tests/ptq/test_tune_loss_weighting.py -q -k ste`. Expected: FAIL (plain MSE at :871).
- [ ] (coder) `src/quantibbler/ptq/block_recon.py` — STE loss site (:871) gets the same weighting construction; the weight is applied in the stage's compute dtype, detached, non-trainable. Command: `pytest tests/ptq/test_tune_loss_weighting.py -q -k ste`. Expected: green.
- [ ] (coder) Regression sweep over the touched region: `pytest tests/ptq/test_block_recon.py tests/ptq/test_ste_sign_adaptation.py tests/ptq/test_relative_lr.py -q`. Expected: green.

## Phase 4 — Uniform fallback + stage records (tests first)

- [ ] (coder) `tests/ptq/test_tune_loss_weighting.py` — red-first fixtures for scenarios `uniform_fallback_recorded_on_down_proj_less_block` and `stage_records_carry_weighting_mode`: (a) a toy block whose linears carry NO down_proj/arch-equivalent terminal name, terminal_grad requested → assert the fallback fires AND the recorder captured a `tune_loss_uniform_fallback` event naming the block and the missing canonical layer; (b) assert the returned `tune_fp` and `ste` records carry `loss_weight` and `loss_weight_source`. Command: `pytest tests/ptq/test_tune_loss_weighting.py -q -k "fallback or record"`. Expected: FAIL (no fallback, no fields).
- [ ] (coder) `src/quantibbler/ptq/block_recon.py` — implement the name-based terminal lookup (canonical `down_proj`, arch-equivalent alias map, uniform fallback + `tune_loss_uniform_fallback` event — never silent), and add `loss_weight` / `loss_weight_source` to both stage records (set at record construction so the vacuous-teacher early return also carries them). Update `TUNE_FP_LOSS_MARKER` to name the [NQ] :3012 weighted-MSE form with the weight source flagged as informed assumption. Command: `pytest tests/ptq/test_tune_loss_weighting.py -q`. Expected: green.
- [ ] (coder) `tests/ptq/test_driver_stage_map.py` (or qk_report surface) — pin that a run carrying a fallback event surfaces it in the run record (stage_map/event counts), so the fallback is visible in artifacts, not just in-process. Command: targeted pytest. Expected: green.
- [ ] (planner) `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane4 --strict`. Expected: VALID.

## Phase 5 — Full gate suite

- [ ] (coder) Run the full gate suite and record results: `pytest tests/ptq/ tests/ci/ tests/qk_report/` + `cargo test` + `cargo clippy --all-targets -D warnings` + `cargo fmt --check` + `maturin build`. Expected: all green. Record the results in the change's run notes.
