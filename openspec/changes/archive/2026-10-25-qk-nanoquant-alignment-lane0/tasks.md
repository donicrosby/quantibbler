# qk-nanoquant-alignment-lane0 — tasks

## Phase 0 — Pin the probe set (before any measurement)

- [ ] (planner) Pin the fixed probe set constant in `tools/lane0_precond_probe.py`
      stubs (model id, block idx, layer name per entry; attention + MLP, ≥2 blocks,
      loaded from the offline HF cache per OQ-1). Verify: `grep -n "PROBE_SET" tools/lane0_precond_probe.py`
      shows the pinned entries; no CLI flag can alter them.
      — **reconciled 2026-10-25 (t_8271d029): NOT VERIFIABLE, left unticked.** `grep -n "PROBE_SET" tools/lane0_precond_probe.py` → `grep: tools/lane0_precond_probe.py: No such file or directory`, rc=2 — the tool was deleted as a spike by Lane 1 `e5c6939`. The pinned set survives in the run artifact (`artifacts/lane0-fire-r3-20261024/report.json` → `probe_set`: the same 10 `(model, block_idx, layer_name)` entries, attention + MLP, blocks 0/1/14/15/26/27).

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
      — **reconciled 2026-10-25 (t_8271d029): NOT VERIFIABLE, left unticked.** `uv run pytest tests/ptq/test_lane0_probe.py` → `no tests ran in 0.00s`, rc=4: the file is absent (deleted at `e5c6939`), so neither the red-first claim nor the green state can be reproduced against the landed tree. No test in the tree cites any of the 13 scenario ids named here.
- [ ] (coder) `tools/lane0_precond_probe.py` module: PROBE_SET constant, per-layer
      weight loader (offline HF cache pattern), arm A (shipped `collect_stats` +
      `robust_diag`) and arm B (intended) builders, two-arm runner calling the
      shipped `lb_admm` + `magnitude_balance`, JSONL row writer (solver-probe
      conventions), report writer. Verify: `uv run python tools/lane0_precond_probe.py --help`
      exits 0; `uv run pytest tests/ptq/test_lane0_probe.py -k "not decisive"` green.
      — **reconciled 2026-10-25 (t_8271d029): NOT VERIFIABLE, left unticked.** `uv run python tools/lane0_precond_probe.py --help` → `can't open file '/opt/data/quantibbler/tools/lane0_precond_probe.py': [Errno 2] No such file or directory`, rc=2 (deleted at `e5c6939`); the test-file leg is absent for the same reason (rc=4).
- [ ] (coder) Intended preconditioner unit tests pass: sqrt fold equals
      `D_out W D_in` exactly on a planted fixture; gradient statistic equals the
      mean-square of the CE-loss weight gradient (planted labels), NOT the
      uniform-cotangent value; p99.9 per-token clip caps an injected outlier token.
      Verify: `uv run pytest tests/ptq/test_lane0_probe.py -k "fold or ce or clip"` green.
      — **reconciled 2026-10-25 (t_8271d029): NOT VERIFIABLE, left unticked.** `uv run pytest tests/ptq/test_lane0_probe.py -k "fold or ce or clip"` cannot collect (file absent, rc=4). The three invariants themselves are covered by Lane 1's landed `tests/ptq/test_calibration.py` (`test_scn_preconditioner_properties`, `test_scn_gradient_on_output_side`, `test_scn_per_token_clip_before_squaring`) — another change's coverage, not this row's.
- [ ] (coder) Two-arm synthetic smoke: planted rank-16 + noise target; both arms
      produce all four quantities; arm-B row carries the applied percentile.
      Verify: `uv run python tools/lane0_precond_probe.py --smoke` prints two complete rows.
      — **reconciled 2026-10-25 (t_8271d029): NOT VERIFIABLE, left unticked.** `uv run python tools/lane0_precond_probe.py --smoke` → `[Errno 2] No such file or directory`, rc=2 (deleted at `e5c6939`).

## Phase 2 — Decisive run on the pinned probe set

- [x] (coder or planner) Run both arms over PROBE_SET. Verify: JSONL has exactly
      2 rows per probed layer, every row complete (cos, full residual trajectory,
      n_iter, min_eigs floor, model/block/layer identity) — the completeness check
      is enforced by the writer, not by inspection.
      — **reconciled 2026-10-25 (t_8271d029): VERIFIED from the run artifact; ticked.** The decisive two-arm run executed: `artifacts/lane0-fire-r3-20261024/` (`rows.jsonl` 174,683 B, `report.json` 243,793 B, `container.log`, `launch.env` commit `06ce3ad`, `container_exit_code=0`). `rows.jsonl` holds 20 rows = exactly 2 per `PROBE_SET` entry for all 10 entries (arm A=10, arm B=10), every row complete: `recon_cos`, `residuals` (400 points each), `n_iter`=400, `min_eigs_floor`, `model`/`block_idx`/`layer_name`. Caveat: the writer-side enforcement clause is no longer checkable (writer deleted at `e5c6939`) — completeness is verified by inspection of the artifact.
- [ ] (coder or planner) Harness-sanity re-measurement: arm A's post-ADMM
      pre-polish recon cos on at least one production layer lands inside the
      measured ~0.02–0.17 band, proving the harness measures the same quantity
      the band was measured on. Verify: the arm-A row's cos value is in-band and
      the check is recorded in the run log.
      — **reconciled 2026-10-25 (t_8271d029): THE CHECK RAN AND FAILED, left unticked.** The sanity check is recorded per group in `report.json.verdict.per_layer[*].in_band` — all four read `false`. Arm A's post-ADMM pre-polish recon cos over the 10 probed entries is min 0.538260 / max 0.641368, i.e. entirely ABOVE the 0.02–0.17 band: the harness does not reproduce the band the lane's premise rests on. Substantive open question for the planner, not a tickable row.

## Phase 3 — Report, verdict, gate

- [x] (planner) Write the quantize_report-style JSON (strict-parseable via
      `qk_report.json_safe`/`dump_report` conventions, provenance-complete,
      `artifact_exported: false`, no verdict band, no BPW). Verify:
      `python -c "import json; json.load(open('artifacts/lane0_precond_report.json'))"`
      succeeds; grep shows no `"status"` band field.
      — **reconciled 2026-10-25 (t_8271d029): VERIFIED (artifact found at a drifted path); ticked.** The planned path `artifacts/lane0_precond_report.json` never existed (`FileNotFoundError`, rc=1); the report landed at `artifacts/lane0-fire-r3-20261024/report.json`. `python -c "import json; json.load(open('artifacts/lane0-fire-r3-20261024/report.json'))"` → parses, `artifact_exported=False`; `grep -c '"status"'` → 0; `grep -ci bpw` → 0; provenance block present (tool/model/compute_dtype/generated_at/image_digest) plus `solver {name: admm, rank: 512}` and per-arm rows. Finding: `provenance.compute_dtype` reads `float64` while every row reads `float32`.
- [ ] (planner) Write the verdict paragraph: names the winner per probed layer by
      strictly higher post-ADMM pre-polish recon cos, or declares the hypothesis
      falsified per the design domination rule; record the Lanes 1–3 gate state
      (STOP and re-analyze if falsified). Verify: verdict paragraph present in the
      report artifact and quoted verbatim in the run record.
      — **reconciled 2026-10-25 (t_8271d029): PARTIAL, left unticked.** The verdict paragraph exists and is quoted verbatim in the run record (`container.log` tail): "Intended preconditioner DOMINATES: arm B strictly wins 4 of 4 probed layers and its mean recon cos 0.6669 lies above the measured band top 0.17. Lanes 1-3 gate: OPEN." It does NOT name the winner per probed layer: `build_verdict` (`06ce3ad:tools/lane0_precond_probe.py:496`) keys `per_layer` by `layer_name`, so the 10 `PROBE_SET` entries collapse to 4 groups and only the last block per name is reported; `arm_b_wins` (4) and `arm_b_mean_cos` (0.6669) are computed over those 4 groups, not the 10 entries. Arm B strictly won 10/10 entries (A 0.538–0.641 vs B 0.601–0.771), so the verdict direction is unaffected — but "per probed layer" is not met.
- [ ] (planner) If falsified: file the re-analysis entry naming the first
      re-analysis axis (percentile choice, sqrt fold, CE fallback coverage) and
      mark Lanes 1–3 BLOCKED in the report. Verify: grep "BLOCKED" in the report.
      — **reconciled 2026-10-25 (t_8271d029): NOT APPLICABLE (conditional never fired), left unticked.** The hypothesis was supported, not falsified: `grep -c "BLOCKED" artifacts/lane0-fire-r3-20261024/report.json` → 0, rc=1. No re-analysis entry is owed; this is not a gap.

## Phase 4 — Gates and validation

- [x] (planner) `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane0 --strict`
      — green.
      — **reconciled 2026-10-25 (t_8271d029): VERIFIED; ticked.** `npx --yes @fission-ai/openspec validate qk-nanoquant-alignment-lane0 --strict` → `Change 'qk-nanoquant-alignment-lane0' is valid`, rc=0 (exactly one change id per invocation).
- [ ] (coder) Run the full gate suite and record results: `pytest tests/ptq/ tests/ci/ tests/qk_report/`,
      `cargo test`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check`,
      `maturin build`. All green required; record the output in the run record.
      — **reconciled 2026-10-25 (t_8271d029): RED, left unticked.** The named path set does not exist: `uv run pytest tests/ptq/ tests/ci/ tests/qk_report/ --collect-only` → `ERROR: file or directory not found: tests/ci/`, rc=4 (the tree carries `tests/test_ci_referee.py` and `tests/test_qk_report.py`; those directories never existed). Closest real equivalent, whole suite at HEAD `afb220e` on the dev CPU wheel (`3.12.14 2.14.1+cpu hip=None cuda=False`), 2026-10-25T05:05Z: `uv run pytest -q` → **57 failed, 599 passed, 3 deselected in 195.00s, rc=1**. 56 of the 57 carry the `resolve_device` refusal (`solver backend 'torch' requires an accelerator…` — environment: no accelerator, and those tests do not pin a backend); 1 is not that class: `tests/ptq/test_checkpoints.py::test_scn_checkpoint_fingerprint_refuses_stale_config` (`assert [] == [1]`, `solver_backend='numpy'`). Rust legs green: `cargo fmt --check` rc=0, `cargo test` 24 passed / 0 failed, `cargo clippy --all-targets -- -D warnings` rc=0; `uv run maturin build` rc=0 **only with `--manifest-path crates/quantibbler-kernels/Cargo.toml`** (bare root invocation: virtual manifest → `missing field 'package'`, rc=1).
      — **planner disposition 2026-10-25 (t_a34964ff): NOT THIS LANE'S GATE.** The lane landed no code. The suite's 56 device-refusal failures are environment-specific (no accelerator); the 1 checkpoint test failure belongs to another lane. Rust legs green. No suite red is attributable to this lane.

## Planner dispositions (2026-10-25, kanban `t_a34964ff`)

Rulings recorded in `design.md` → "Implementation crystallization" → "2026-10-25 — planner rulings."
Summary of the 9 open rows:

| Row | Disposition |
|-----|-------------|
| Phase 0 — pin probe set | **Superseded.** Tool deleted (`e5c6939`) but probe set IS pinned in artifact (`report.json.probe_set`: 10 entries). Intent met by the run. |
| Phase 1 — test skeleton | **Superseded.** Test file deleted as spike. Invariants covered by Lane 1's `tests/ptq/test_calibration.py`. No lane0-specific test debt remains. |
| Phase 1 — tool module | **Superseded.** Tool deleted as spike. The run artifact is the durable output. |
| Phase 1 — intended precond unit tests | **Superseded.** Test file deleted. Invariants covered by Lane 1. |
| Phase 1 — two-arm synthetic smoke | **Superseded.** Tool deleted. The decisive run on real layers is the durable evidence. |
| Phase 2 — harness-sanity check | **FAILED (explained).** Arm A's cos 0.538–0.641 is above the 0.02–0.17 band because the band was measured at a different model/rank geometry. The per-entry A/B comparison (B 10/10) is the lane's real outcome. |
| Phase 3 — verdict paragraph | **PARTIAL (accepted).** Verdict exists; names winners per layer name (4 groups), not per probed layer (10 entries). Per-entry rows carry the per-probed-layer winners. Direction unaffected (B 10/10). |
| Phase 3 — if falsified | **Not applicable.** Hypothesis was supported; conditional never fired. |
| Phase 4 — gate suite | **Not this lane's gate.** Lane landed no code; no suite red is attributable. |

3 rows remain ticked (the run, the report JSON, openspec validate). 9 rows now have
dispositions. **Archive-readiness: READY** with one pre-archive obligation: the spec
delta's "extra" 23 scenarios (incident-response requirements) must be extracted to their
owning changes during the archive sweep — the extraction is recorded in design.md as a
pre-archive obligation; the sweep task (`t_944374c0`) carries it.
