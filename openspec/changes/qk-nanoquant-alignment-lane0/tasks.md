# qk-nanoquant-alignment-lane0 — tasks

## Phase 0 — Pin the probe set (before any measurement)

- [ ] (planner) Pin the fixed probe set constant in `tools/lane0_precond_probe.py`
      stubs (model id, block idx, layer name per entry; attention + MLP, ≥2 blocks,
      loaded from the offline HF cache per OQ-1). Verify: `grep -n "PROBE_SET" tools/lane0_precond_probe.py`
      shows the pinned entries; no CLI flag can alter them.

## Phase 1 — Harness with tests first (RED-first per capability)

- [ ] (coder) `tests/ptq/test_lane0_probe.py` skeleton with one test per scenario:
      `measurement_only_no_export`, `shipped_pipeline_modules_untouched`,
      `fixed_probe_set_identity_recorded`, `both_arms_identical_probe_set`,
      `sqrt_fold_once_per_side`, `gradient_side_real_ce_backward`,
      `per_token_norm_clip_999`, `same_solver_pair_both_arms`,
      `pre_polish_stage_boundary`, `four_quantities_per_layer_per_arm`,
      `recon_cos_against_original_weight`, `winner_named_per_probed_layer`,
      `falsified_hypothesis_gates_lanes`. Verify: `uv run pytest tests/ptq/test_lane0_probe.py`
      FAILS (collection error on the absent tool module) — red-first recorded.
- [ ] (coder) `tools/lane0_precond_probe.py` module: PROBE_SET constant, per-layer
      weight loader (offline HF cache pattern), arm A (shipped `collect_stats` +
      `robust_diag`) and arm B (intended) builders, two-arm runner calling the
      shipped `lb_admm` + `magnitude_balance`, JSONL row writer (solver-probe
      conventions), report writer. Verify: `uv run python tools/lane0_precond_probe.py --help`
      exits 0; `uv run pytest tests/ptq/test_lane0_probe.py -k "not decisive"` green.
- [ ] (coder) Intended preconditioner unit tests pass: sqrt fold equals
      `D_out W D_in` exactly on a planted fixture; gradient statistic equals the
      mean-square of the CE-loss weight gradient (planted labels), NOT the
      uniform-cotangent value; p99.9 per-token clip caps an injected outlier token.
      Verify: `uv run pytest tests/ptq/test_lane0_probe.py -k "fold or ce or clip"` green.
- [ ] (coder) Two-arm synthetic smoke: planted rank-16 + noise target; both arms
      produce all four quantities; arm-B row carries the applied percentile.
      Verify: `uv run python tools/lane0_precond_probe.py --smoke` prints two complete rows.

## Phase 2 — Decisive run on the pinned probe set

- [ ] (coder or planner) Run both arms over PROBE_SET. Verify: JSONL has exactly
      2 rows per probed layer, every row complete (cos, full residual trajectory,
      n_iter, min_eigs floor, model/block/layer identity) — the completeness check
      is enforced by the writer, not by inspection.
- [ ] (coder or planner) Harness-sanity re-measurement: arm A's post-ADMM
      pre-polish recon cos on at least one production layer lands inside the
      measured ~0.02–0.17 band, proving the harness measures the same quantity
      the band was measured on. Verify: the arm-A row's cos value is in-band and
      the check is recorded in the run log.

## Phase 3 — Report, verdict, gate

- [ ] (planner) Write the quantize_report-style JSON (strict-parseable via
      `qk_report.json_safe`/`dump_report` conventions, provenance-complete,
      `artifact_exported: false`, no verdict band, no BPW). Verify:
      `python -c "import json; json.load(open('artifacts/lane0_precond_report.json'))"`
      succeeds; grep shows no `"status"` band field.
- [ ] (planner) Write the verdict paragraph: names the winner per probed layer by
      strictly higher post-ADMM pre-polish recon cos, or declares the hypothesis
      falsified per the design domination rule; record the Lanes 1–3 gate state
      (STOP and re-analyze if falsified). Verify: verdict paragraph present in the
      report artifact and quoted verbatim in the run record.
- [ ] (planner) If falsified: file the re-analysis entry naming the first
      re-analysis axis (percentile choice, sqrt fold, CE fallback coverage) and
      mark Lanes 1–3 BLOCKED in the report. Verify: grep "BLOCKED" in the report.

## Phase 4 — Gates and validation

- [ ] (planner) `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane0 --strict`
      — green.
- [ ] (coder) Run the full gate suite and record results: `pytest tests/ptq/ tests/ci/ tests/qk_report/`,
      `cargo test`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check`,
      `maturin build`. All green required; record the output in the run record.
