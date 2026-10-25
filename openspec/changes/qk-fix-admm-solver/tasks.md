# qk-fix-admm-solver — tasks

## Phase 1 — SVID two-sided optimum + rho endpoint cap

- [x] (coder) `svid_proxy` (`lb_admm.py:162-203`) replaced with two-sided optimum via
      power iteration on `S∘P`. Z = S ⊙ (g hᵀ). Dead-column guard preserved.
      Docstring rewritten around [NQ] :592.
- [x] (coder) `_top_rank1` helper: power iteration with σ-delta convergence (tol 1e-14,
      cap 100). On near-degenerate spectra the vectors can hit the cap unconverged →
      suboptimal projection in tie cases. One docstring line recording that bound
      (@bug-hunter's nit).
- [x] (coder) `config.py` — `rho_end` 1e2 → 1e-1, field docstring recording the
      endpoint as an implementation choice ([NQ] App. C :3008 prescribes schedule
      shape only) calibrated for the normalized regime.
- [x] (coder) `tests/ptq/test_svid_optimality.py` — red-first verified: 2 failed
      against old proxy (planted rank-1 recovery, strict dominance), 2 passed
      (sign-preservation, dead-column floors). Now 13/13 green across the three
      touched test files (@bug-hunter verified).
- [x] (coder) `tests/ptq/test_lb_admm.py:84-96` — retired column-only closed-form
      assertion replaced with two-sided property (rank-1 magnitude profile on
      live columns + strict objective dominance over retired variant).
      Sign-preservation and sign(0)=+1 asserts untouched.
- [x] (coder) `tests/ptq/test_config.py:25` — `rho_end == 1e2` pin updated to 1e-1
      with reasoning recorded.
- [x] (coder) `tests/ptq/test_lb_admm_scale_equivariance.py` — still green.

## Phase 2 — Convergence test with corrected assertions

- [x] (coder) `tests/ptq/test_admm_convergence.py` — assertions revised per planner ruling:
      1. **`converged is True` → dropped.** The spec's scenario `admm_consensus_convergence`
         says "epsilon or at the K-iteration cap." With a ±1-structured SVID proxy,
         epsilon=1e-7 is unreachable by construction. Replace with: the consensus
         residual makes net progress across the **first 200 iterations** (proof of
         meaningful work), and the solver finishes at the K-iteration cap with
         `converged` recorded in the trace as a measurement, not a gate.
         *(Amended 2026-10-25: this row said "decreases monotonically", which is refuted
         by measurement — 67 of the first-half steps rise on the fixture and 20/20 real
         `layers.0` probe rows in `artifacts/lane0-fire-r3-20261024/rows.jsonl` are
         non-monotone. The folded requirement of record, `openspec/specs/qk-ptq/spec.md`,
         says only "decreases across iterations"; net progress over the first half plus a
         terminal value at the schedule's own floor is the asserted, measured form. The
         over-claiming delta scenario is reported to the planner, not amended here. See
         design.md "Implementation crystallization — 2026-10-25".)*
      2. **`cos >= warm_start cos` → dropped, two assertions replace it:**
         - `cos(UVᵀ, W) > 0.5` — the ADMM produces a usable initializer. Cos loss
           from the unconstrained truncated-SVD warm start is **expected** — the
           SVID proxy actively pushes latents into the ±1-signed rank-1 subspace,
           and the paper's pipeline compensates in STE refinement (Step 3).
         - `cos(UVᵀ, W) > cos(sign(zu) @ sign(zv)ᵀ, W)` — the continuous
           reconstruction beats a naive sign-only assignment, proving the ADMM
           iteration does meaningful work beyond trivial binarization.
      3. **Fixture:** planted rank-32 target, rank-64 factorization. Post-ADMM
         cos SHALL be ≥0.5 on a target the warm start represents at cos ≥0.99
         (the solver may degrade the unconstrained fit but must not collapse it).
- [x] (coder) Run suite: `tests/ptq + test_ci_referee + test_qk_report +
      test_run_ptq_wrapper`. All green required.
      *(Observed 2026-10-25 in `quantibbler-ptq:rocm-official` — the interpreter the fire
      runs — with `/dev/kfd` + `/dev/dri` passed and the working tree mounted:
      `399 passed, 4 skipped, 3 deselected in 93.92s`; python 3.12.3, torch
      2.14.0+rocm10.1.0, hip 7.16.26385, ndev 1, numpy 2.5.4, and
      `quantibbler.__file__=/repo/src/quantibbler/__init__.py` (mounted tree wins). The
      run was announced on this card before firing. Contrast on the dev CPU wheel, same
      403 collected: `57 failed, 346 passed, 3 deselected` (python 3.12.14, torch
      2.14.1+cpu, hip None, ndev 0) — 50 failures are `solver_backend.py:111` "requires an
      accelerator" and the other 6 are that same refusal surfacing as a subprocess-stdout
      AssertionError in driver/probe tests. The number is a property of the
      (interpreter x device) pair, and the accelerated pair is the binding one. The
      ticket's DoD path `tests/ci/ tests/qk_report/` does not exist (`file or directory
      not found`, RC=4); this row's own file names were used.)*
- [x] (planner) `openspec validate qk-fix-admm-solver --strict` — green.
      *(Observed 2026-10-25 (t_2d5b0a67): `npx --yes @fission-ai/openspec validate
      qk-fix-admm-solver --strict` → `Change 'qk-fix-admm-solver' is valid`, RC=0,
      exactly one change id per invocation.)*

## Phase 3 — Single-layer re-measure (no confound)

- [ ] (coder or planner) Run `SOLVER=admm PAPER_FAITHFUL=1` on `layers.0` with
      the **SVID fix only** (`rho_end` still 1e2 in the fork) to quantify the
      SVID improvement in isolation. Record per-linear raw cos.
      *(Disposition 2026-10-25 (t_2d5b0a67, planner): LEFT UNTICKED — @user ruling
      killed the faithful-ADMM probe + rho-cap arm (t_83fa0ef7). Premise text is
      false as written: the `rho_end = 1e2` fork this row measures against no longer
      exists (lane-3's normalized regime, `97a4725`, ships `rho in [0.0, 1.0]`), and
      the `layers.0` probe tool was deleted as a retired spike (`e5c6939`). No
      vehicle, no measurement owed.)*
- [ ] (coder or planner) Then stack the rho cap on the same branch and re-run
      to measure whether the cap moves any further metric. These two attribution
      numbers go in the run report alongside the packed-gate baseline.
      (@bug-hunter's process suggestion — avoids the confound that cost us
      the first decomposition.)
      *(Disposition 2026-10-25 (t_2d5b0a67, planner): LEFT UNTICKED — same @user
      ruling (t_83fa0ef7). The rho-cap-dominates-over-SVID attribution is already
      established from recorded faithful-run forensics, and stacking the cap onto a
      deleted probe vehicle is not runnable. The `rho_end`-cap requirement's
      reconciliation with lane-3's regime is OQ-3 in
      `changes/qk-nanoquant-alignment-lane3/design.md`, owned by the orchestrator.)*

## Phase 4 — Gate suite

- [x] (coder) Full gate suite (commands corrected 2026-10-25 — this repo has no
      `tests/ci/` or `tests/qk_report/`, and `maturin build` needs a
      `--manifest-path`): `uv run --no-sync --extra torch --extra kernels --extra hf
      pytest` (root, full suite) + `cargo test` + `cargo clippy --all-targets -D
      warnings` + `cargo fmt --check` (kernels-scoped) + `maturin build
      --manifest-path crates/quantibbler-kernels/Cargo.toml`. All green required.
      Record results.
      *(Closed 2026-10-25 (t_6ed481cc, the wave-disposition landing card). This note
      supersedes the attributed-RED status recorded here earlier the same day: the
      four `test_progress.py` reds and four `test_qk_report.py` reds it reported were
      co-tenant wave damage (the progress.py ETA rework and qk-launch-attribution's
      scratch-root extraction breaking qk-ci-referee's scratch-dir pin), and both
      surfaces were landed with their owners in that pass.
      Observed on the CLEAN tree: `tests` **756 passed** in 65.90s; `tests/ptq`
      **257 passed, 3 skipped, 4 deselected** in 626.96s; suite rc=0 (1013 passed,
      0 failed). Interpreter `/opt/venv`, `torch 2.14.1+cpu`,
      `cuda.is_available()=False` — the dev CPU wheel, so the accelerated pair is
      still unmeasured here. Rust: `cargo test` 24 passed rc=0; `cargo clippy
      --all-targets -- -D warnings` rc=0; `cargo fmt --all -- --check` rc=0;
      `pre-commit run --all-files` rc=0. The 4 deselected are the default marker
      filter's integration tests, not a silent skip.)*
