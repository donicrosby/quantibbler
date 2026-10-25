# qk-nanoquant-alignment-lane4 — design

## Citation verification (paper is the authority)

- **Weighted MSE exists in the paper:** [NQ] App. C :3012 — "During block reconstruction, we employed a weighted MSE function, utilized in previous quantization works (Boža and Macko, 2026; Kim et al., 2025)." Verified verbatim against `docs/papers/nanoquant-2602.06694v3.md`.
- **The statistic exists in the paper:** [NQ] Step 2-1 :536–540 — the block-reconstruction distortion is Hessian-weighted via K-FAC with "diagonal preconditioners constructed from activation and gradient statistics"; [NQ] Algorithm 1 lines 2–5 (:703–705) — Phase 1 collects per-linear `(z_in, z_out)` and RobustDiag produces `(D~_in, D~_out)`.
- **The paper does NOT state which statistic weights the block-reconstruction MSE.** No formula accompanies :3012, and the only `D~_out` usages are Eq. 2/Algorithm 1 (ADMM entry preconditioning). The terminal-linear weight-source choice below is therefore an **informed assumption (IA-1)**, not a paper pin — the paper's "weighted MSE" claim is what it anchors.

## Weight source (IA-1): the terminal linear's Phase-1 output statistic

The block-tuning objectives (TuneFP, TuneLatentSTE) compare block outputs to the cached teacher on the residual stream — `(out - y_b)` is indexed by the block's output features. The per-feature importance of those features is the K-FAC-style output curvature proxy of the layer whose output IS the block output: the terminal linear. In Llama/Qwen/Gemma-3-style decoder blocks that linear is `down_proj` (the MLP's output projection onto the residual stream) — and `down_proj` is the terminal projection in every architecture the paper itself compresses (Llama, Qwen, Gemma 3, Rnj-1). So the weight is:

```
w = robust_diag(d_out_raw[terminal], tau=cfg.tau, gamma=cfg.gamma)   # (dout,) float64
loss = (w[None, :] * (out - y_b[idx]).pow(2)).mean()
```

`w` rides the EXACT statistic the solver already consumes at ADMM entry (`robust_diag(entry.d_out_raw, ...)`, `block_recon.py:1349`) — one statistic path, no new collection, no extra backward passes. `w` is detached and non-trainable; applied in the stage's compute dtype per the numerics policy. Note on naming: `d_in_raw` folds in the mean-square weight-gradient statistic (`calibration.py:119`), `d_out_raw` is the output-side statistic (`calibration.py:290`); Step 2-1's prose covers the preconditioner pair as "activation and gradient statistics", and `d_out_raw` of the terminal linear is the Phase-1 output-side statistic the user's spec names.

**Why not per-target-layer weighting for STE:** the STE objective is computed on the BLOCK output (`_quantized_forward_block(block, facs, x_b[idx])` vs `y_b`, `block_recon.py:870–871`) — it is Eq. 10's block-output objective, not a per-layer objective — so the block-level weight is the consistent choice for both stages.

**How a wrong assumption would surface:** the red-first poisoned-vector fixture (scenario `weighting_reaches_the_loss_poisoned_weight_changes_loss`) fails if the weight does not enter the loss; a fidelity regression vs. the plain-MSE history on the same fixture geometry (measured at the next fire) fails if the weighting HURTS; the run report's stage records name the mode, so a run can never drift weighting silently.

## Canonical layer identification and the uniform fallback

Identification is name-based, never positional guessing:

1. Canonical: the block's quantizable linear with leaf name `down_proj`.
2. Arch-equivalent: resolved through an explicit alias map (v0.1 ships `{"down_proj"}`; a new architecture adds its terminal-projector name to the map — never inferred silently). Every architecture in the paper's tables uses `down_proj`, so the v0.1 map is complete for paper scope.
3. Fallback: no match in the block (or the matched layer carries no Phase-1 statistic — a mis-scoped stats book) → **uniform weights** and a `tune_loss_uniform_fallback` recorder event `{block, missing, capture_index, mode: "uniform"}`. Never silent: a uniform-weighted stage without that event on record is impossible by construction (same discipline as `tune_fp_vacuous_teacher` / `ste_scale_only_declared`).

The down_proj-less toy block of the acceptance fixture is exactly trigger 3: its linears (e.g. `w1`, `w2`) carry no canonical terminal name, so the fallback fires and is recorded.

## Record and fingerprint surface

- `tune_fp` record and `ste` record each gain `loss_weight: "terminal_grad" | "uniform"` and `loss_weight_source: <terminal linear name> | "uniform_fallback"`. The fields are set at record construction, so the vacuous-teacher early-return path also carries them.
- `TUNE_FP_LOSS_MARKER` becomes e.g. `"tune_fp_loss: weighted_mse_nq_3012 (weight source: informed_assumption)"` — greppable in code, spec, and run record, and now honest about the paper-anchored form.
- Checkpoint fingerprint: `block_loss_weight` is a plain `PTQConfig` dataclass field, and `config_fingerprint` hashes `dataclasses.asdict(cfg)` (`checkpoints.py:57`) — the knob joins the fingerprint by construction. A config pin test asserts it (scenario `loss_weight_knob_joins_checkpoint_fingerprint`); a checkpoint dir produced under one mode refuses resume under the other.

## Discarded alternatives

- **Block-level full K-FAC weighting (D_out ⊗ D_in):** Eq. 2's two-sided form is a per-layer preconditioning device; the block tuning losses are output-space residuals and have no input-side index. Discarded: no matrix to hang `D_in` on without inventing one.
- **Weight by the current target layer's `d_out`:** mid-block layers' outputs are not the block output; their statistics do not index `(out - y_b)`. Discarded: wrong feature space.
- **Normalize `w` to mean 1 before the loss:** the paper's shrinkage (Eq. 3) is the only normalization the statistics get; an added mean-1 normalization would deviate the weighting from the statistic the solver consumes elsewhere and hide scale information. Discarded: un-papered extra step.
- **Config flag without a fallback event:** an explicit `uniform` mode alone would let absence-triggered uniform runs pass as deliberate. Discarded: the acceptance requires the fallback to be recorded — never silent.
- **Gradient-statistic recomputation for `d_out_raw` (backward hook on layer outputs):** adds a second statistics definition and a new collection pass; Phase-1's book already exists and matches Algorithm 1 lines 2–5. Discarded: cost and definitional drift for no paper gain.

## Informed assumptions

1. **IA-1 (weight source):** the paper states "weighted MSE" (:3012) but no weight formula; we use the terminal linear's robust-diagonalized Phase-1 output statistic (rationale above: it is the block-output curvature proxy on the paper's own Eq. 2 statistic path, and `down_proj` is the terminal projection in every paper-scope architecture). If wrong, the poisoned-vector fixture and the next fire's fidelity comparison against plain-MSE history surface it.
2. **IA-2 (alias map scope):** the arch-equivalent map ships with `{"down_proj"}` because every architecture in the paper's tables uses that name. If a future architecture uses another terminal name, the behavior is the recorded uniform fallback (loud, measurable), not silent mis-weighting — which is the designed safety property.
3. **IA-3 (mean-of-weighted-squares form):** the weight multiplies the squared residual per feature with the plain mean reduction. The paper gives no reduction detail; the mean keeps loss magnitude comparable to the legacy plain-MSE form, so stage records' residual norms stay interpretable across modes. If wrong (sum-form intended), only the loss scale shifts — the weighting still reaches the gradient, and the poisoned-vector fixture still passes; a fidelity difference would show at the fire.

## Open Questions

- **OQ-1 — does terminal_grad weighting move fidelity verdicts on real runs?** Status: open until the next production fire; owner: planner. Lane discipline (lanes 2 and 4 land before any new production run) means the fire measures this directly; no verdict change is claimed here.

## Implementation crystallization — 2026-10-25 (reconcile pass)

Verified against the lane tip. The working tree is SHARED and was being rewritten
by sibling lane reconcilers during this pass: `src/quantibbler/ptq/block_recon.py`
(mtime 05:10Z) and `calibration.py` (05:11Z) moved after the commits at 04:25/05:05Z.
One combined-tier run therefore read a mid-write state and reported 7 failures; two
later serial runs of the SAME command at the SAME commit were 386 passed / 0 failed.
Treat any single tier number taken from this tree as provisional unless the tree was
pinned.

### Gate run (the Phase-5 row's "run notes")

Every number carries its interpreter × device pair; the dev venv cannot exercise the
accelerated path (`pyproject.toml` scopes the `torch` extra to the `pytorch-cpu` index).

- `quantibbler-ptq:rocm-official`, `--device /dev/kfd --device /dev/dri`, mounted at
  `/repo` with `PYTHONPATH=/repo/src`, torch 2.14.0+rocm10.1.0 (hip 7.16.26385),
  1 device:
  - `python -m pytest tests/ptq -q -p no:cacheprovider` → **226 passed, 2 skipped,
    3 deselected** (0 failed) — the exact figure the lane's landing commit quoted.
  - `python -m pytest tests/ptq tests/test_ci_referee.py tests/test_qk_report.py -q
    -p no:cacheprovider` → **386 passed, 2 skipped, 3 deselected** (0 failed), twice.
  - lane-local: `test_tune_loss_weighting.py test_tunefp_teacher.py test_config.py`
    → 22 passed; regression sweep (`test_block_recon.py test_ste_sign_adaptation.py
    test_relative_lr.py`) → 22 passed; `-k ste` → 1 passed; `-k "fallback or record"`
    → 2 passed; `test_driver_stage_map.py` → 3 passed, 3 deselected.
- dev venv (`uv sync --all-groups --extra kernels --extra torch --extra hf`),
  torch 2.14.1+cpu, 0 devices: `uv run pytest tests/ptq tests/test_ci_referee.py
  tests/test_qk_report.py -q` → **331 passed, 57 failed, 3 deselected**. All 57 are
  the fail-closed `resolve_device` refusal ("solver backend 'torch' requires an
  accelerator"), i.e. the documented CPU-wheel baseline rather than lane-4 behaviour.
  16 of them are this lane's own fixtures routing through `reconstruct_block` →
  `magnitude_balance(p_u, p_v, d_out_t, d_in_t, target=w)` (`block_recon.py:1582`),
  which never receives `cfg.solver_backend` and so resolves the DEFAULT torch backend
  even when the fixture pins `solver_backend="numpy"`. That call-site gap predates
  every lane here (blame `3793fc0`, 2026-10-19) and is untouched by this lane; it is
  the reason a `solver_backend="numpy"` fixture is only green where an accelerator is
  present.
- kernels (`crates/quantibbler-kernels/`): `cargo test` → **24 passed, 0 failed**
  (`tests/parity.rs`; all other targets 0 tests); `cargo clippy --all-targets --
  -D warnings` RC=0; `cargo fmt --all -- --check` RC=0.
- `maturin build` at the repo root FAILS ("Failed to parse Cargo.toml … missing field
  `package`"): the root manifest is a pure workspace, so the gate's spelling must be
  the crate-scoped `maturin build --manifest-path
  crates/quantibbler-kernels/Cargo.toml` (RC=0, abi3 wheel built).
- `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane4 --strict`
  → "Change 'qk-nanoquant-alignment-lane4' is valid" (one change id per invocation).

### Row anchors that were rot, and the one uncovered row

- The two loss-site rows cited pre-change line numbers (`block_recon.py:667` and
  `:871`); the sites are the loss expressions in `_tune_fp` (`:809`) and
  `_tune_latent_ste` (`:1029`). The rows now cite the symbols.
- The TuneFP/STE test rows cited `-k tune_fp`, which collects 0: tests are named
  `test_scn_<scenario_id>` per the spec-naming rule, so the rows now carry the real
  selector. The TuneFP row also said "first-step losses differ" — no first-step loss
  value is recorded, so the fixture asserts on `stats["gradient_norm"]`.
- The gate row named `tests/ci/` and `tests/qk_report/`, which do not exist; they are
  `tests/test_ci_referee.py` and `tests/test_qk_report.py`.
- **Still uncovered (left unchecked):** the Phase-4 row pinning a fallback event's
  surfacing in the run record. The mechanism is present —
  `tools/quantize_hf_model.py:1313` writes
  `report["stages"] = [{"event": event, **info} …]`, so a recorded
  `tune_loss_uniform_fallback` does reach the artifact — but `stage_map()`
  (`qk_report.py:609`) keys only `_STAGE_LINES` entries (`tune_loss_uniform_fallback`
  is not one) and no test asserts the fallback's presence in any report. The task's
  named deliverable (the pin) does not exist.

### Open finding for the planner (spec prose vs the tree)

The delta's scenario `fixtures_discover_names_not_hardcode_qwen3` closes with "AND no
test file contains a hardcoded Qwen3 layer name literal (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) outside of arch-map declaration rows". The tree violates that
clause as literally written: `test_tunefp_teacher.py:83`, `test_root_cause_ablation.py:72`,
`test_ste_sign_adaptation.py:68-71` and `test_fp_resume_guard.py:48` all build toy
blocks out of `{"q_proj": …}` — fixtures that pre-date this lane and serve other
tests. The requirement's normative text is scoped to "weighting and resolution
fixtures", of which this lane's file is the newest and DOES comply (names discovered
from `named_modules()`, arch names confined to the `_ARCHS` map). No test enforces
the clause either way, so it is prose that can only be checked by hand. Not amended
here: rewriting a spec-delta scenario is a ruling, not a reconcile edit — and
`openspec/specs/**` is the requirement of record, so the fix belongs to whoever owns
the spec.
