# Tasks: qk-tune-loss-scale

Interpreter for every reading below: `/opt/venv` (`torch 2.14.1+cpu`, no
accelerator), invoked as `PYTHONPATH=/opt/data/quantibbler/src uv run --no-sync
--extra torch --extra kernels --extra hf …` — the shared venv's editable
install currently resolves `quantibbler` to a sibling worktree, so `PYTHONPATH`
is what names the tree the numbers came from.

## 1. Spec

- [x] 1.1 `openspec validate qk-tune-loss-scale --strict` is VALID, and every
      ADDED requirement body measures under the 500-character `--strict` limit.
      Evidence: `npx --yes @fission-ai/openspec validate qk-tune-loss-scale
      --strict` → `Change 'qk-tune-loss-scale' is valid` (RC=0); bodies measured
      446 / 406 / 361 chars.

## 2. Loss-weight normalization

- [x] 2.1 The terminal weight is normalized to unit mean once, at the resolve
      site in `reconstruct_block`, before either stage sees it; the raw robust
      diagonal is still what `_resolve_loss_weight` returns.
      Evidence: probe on the shared checkout shows the mean
      `reconstruct_block` hands to `_tune_fp` is exactly `1.0` (pre-change tree:
      `1.0605747542709931`); `pytest tests/ptq/test_tune_loss_weighting.py -q`
      → 12 passed (the raw-robust-diagonal pins are untouched).
- [x] 2.2 Both stage records carry `loss_weight_mean_before` and
      `loss_weight_mean_after` (`None` when the uniform/fallback path ran).
      Evidence: `pytest tests/ptq/test_tune_loss_scale.py -q` → 7 passed,
      including `test_scn_weight_is_normalized_before_the_loss_pipeline`.

## 3. Resolution-derived vacuous floor

- [x] 3.1 `_tune_fp` guards on `gradient_norm < floor` where
      `floor = opt.defaults["eps"] * finfo(coarsest param dtype).eps`, with exact
      zero a special case; the record carries the floor, the optimizer eps, the
      dtype resolution and the dtype name.
      Evidence: bf16 fixture, raw weight `1e-10` → `gradient_floor 7.8125e-11`,
      `gradient_norm 1.9717375279177577e-11`, `vacuous: true`, reason
      `below_resolution_gradient_floor`, `steps: 0`.
- [x] 3.2 `_tune_latent_ste` carries the same floor fields and the same guard
      (record + entry-snapshot restore + end-of-stage), per design.md's named
      delta.
      Evidence: `pytest tests/ptq/test_tunefp_teacher.py
      tests/ptq/test_tune_loss_scale.py -q` → green (exact-zero reason
      `zero_first_step_gradient` unchanged); STE row with weight `1e-30` →
      `vacuous: true`, entry latents byte-restored.

## 4. Inertness declaration

- [x] 4.1 A stage completing its budget with movement below the dtype relative
      resolution records `inert: true`; TuneLatentSTE records it per-linear.
      Evidence: bf16 + unit-mean weight → `weight_delta_rel` ~1.7e-4 < bf16 rel
      res 7.8125e-3 → `inert: true`; f64 + the same weight → ~3.8e-4 >
      f64 2.22e-16 → `inert: false`.
- [x] 4.2 RED → GREEN evidence for the v5 signature.
      RED (pre-change worktree at `cfa3311`): the new test file fails at
      collection (`ImportError: cannot import name 'LOSS_WEIGHT_MEAN_NORMALIZATION'`),
      and the behavioral probe reads `gradient_norm 1.97e-11`,
      `gradient_floor <ABSENT>`, `vacuous: false`, `inert <ABSENT>`, `steps: 2`.
      GREEN: same probe reads `gradient_floor 7.8125e-11`, `vacuous: true`,
      `inert: true`, `steps: 0`; the new file's 7 tests pass. Both transcripts in
      `design.md`'s RED → GREEN section.

## 5. Gates

- [x] 5.1 `tests/ptq` run one file at a time (43 files): 42 × `RC=0`, the one
      exception being `test_driver_stage_journal.py RC=5` — its single test is
      `@pytest.mark.integration` and is deselected, so the file collects nothing.
      That is pre-existing (identical on the pre-change tree), not this change's.
      Root lean tier re-quoted: `PYTHONPATH=… uv run --no-sync pytest -q` →
      **1043 passed, 7 deselected in 419.56s**, RC=0 — the baseline's 1036 passed
      plus this change's 7 new tests.
- [x] 5.2 `uv run --no-sync --with ruff ruff check` → `All checks passed!`;
      `ruff format --check` clean; `pre-commit run --files <this change's 6
      paths>` → all hooks Passed; `cd crates/quantibbler-kernels && cargo test`
      → 24 passed / 0 failed (unchanged by this change).
- [x] 5.3 Scoped commit landed; diff is `src/quantibbler/ptq/block_recon.py`,
      `tests/ptq/test_tune_loss_scale.py`, and this change directory only.
