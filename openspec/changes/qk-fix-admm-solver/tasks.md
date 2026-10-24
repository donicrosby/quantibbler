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

- [ ] (coder) `tests/ptq/test_admm_convergence.py` — assertions revised per planner ruling:
      1. **`converged is True` → dropped.** The spec's scenario `admm_consensus_convergence`
         says "epsilon or at the K-iteration cap." With a ±1-structured SVID proxy,
         epsilon=1e-7 is unreachable by construction. Replace with: the consensus
         residual decreases monotonically across the **first 200 iterations** (proof of
         meaningful work), and the solver finishes at the K-iteration cap with
         `converged` recorded in the trace as a measurement, not a gate.
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
- [ ] (coder) Run suite: `tests/ptq + test_ci_referee + test_qk_report +
      test_run_ptq_wrapper`. All green required.
- [ ] (planner) `openspec validate qk-fix-admm-solver --strict` — green.

## Phase 3 — Single-layer re-measure (no confound)

- [ ] (coder or planner) Run `SOLVER=admm PAPER_FAITHFUL=1` on `layers.0` with
      the **SVID fix only** (`rho_end` still 1e2 in the fork) to quantify the
      SVID improvement in isolation. Record per-linear raw cos.
- [ ] (coder or planner) Then stack the rho cap on the same branch and re-run
      to measure whether the cap moves any further metric. These two attribution
      numbers go in the run report alongside the packed-gate baseline.
      (@bug-hunter's process suggestion — avoids the confound that cost us
      the first decomposition.)

## Phase 4 — Gate suite

- [ ] (coder) Full gate suite: `pytest tests/ptq/ tests/ci/ tests/qk_report/` +
      `cargo test` + `cargo clippy --all-targets -D warnings` + `cargo fmt --check` +
      `maturin build`. All green required. Record results.
