## MODIFIED Requirements

### Requirement: The probe SHALL be configurable in interval, bounded in cost, and fixed in threshold

The driver SHALL read probe configuration from environment variables `QK_PROBE_INTERVAL` (positive integer blocks, default 1) and `QK_PROBE_BATCHES` (positive integer eval batches, default 2). The abort threshold SHALL be the derived `COLLAPSE_LINE` with no per-run override, and probes SHALL begin no earlier than depth 2.

#### Scenario: probe_knob_defaults_and_validation

- **WHEN** the driver starts with no probe env vars set
- **THEN** the probe measures every block boundary from depth 2 on a 2-batch slice, and a non-positive `QK_PROBE_INTERVAL` or `QK_PROBE_BATCHES` is rejected at startup

#### Scenario: probe_env_var_override

- **WHEN** `QK_PROBE_INTERVAL=4` and `QK_PROBE_BATCHES=1` are set
- **THEN** the probe fires every fourth block on a 1-batch slice

#### Scenario: probe_threshold_not_overrideable

- **WHEN** any run executes
- **THEN** no flag or config value re-pins the probe's abort line above or below the shared `COLLAPSE_LINE` constant

#### Scenario: probe_math_lean_tested

- **WHEN** the suite runs in a lean environment without torch
- **THEN** the probe decision logic (slice-ratio verdict, boundary semantics, NaN first) is covered by tests that execute without skips, while model-eval glue tests live in the torch tier per the repo's tier conftest pattern

#### Scenario: probe_empty_env_var_treated_as_unset

- **WHEN** `QK_PROBE_INTERVAL=` (empty string) is set
- **THEN** the default interval 1 applies — an empty value is treated as unset, and the empty-string case is covered by a fixture test

#### Scenario: probe_non_integer_rejected

- **WHEN** `QK_PROBE_INTERVAL=abc` or `QK_PROBE_BATCHES=1.5` is set
- **THEN** the driver exits nonzero at startup naming the invalid input, with the same named refusal as a non-positive value

## ADDED Requirements

### Requirement: The report's fidelity label SHALL be a computed verdict from declared configuration against the paper regime

The report's `paper_faithful` label SHALL be the conjunction of eight config-match conditions plus all existing stage-level conditions. The `paper_faithful_flag` condition SHALL NOT exist. Every condition SHALL be reported. No hardcoded dtype SHALL replace the cut flag.

#### Scenario: all_conditions_satisfied_yields_true

- **WHEN** a run executes with `solver=admm`, `compute_dtype=bf16`, `factor_mode=binary`, `ablation=none`, `layer_scope=all`, `admm_iter=400`, `rho_end` in [0.0, 1.0], `device` on accelerator (not cpu), no runtime deviations, no polish rounds, no vacuous teacher, capture from fp stream, no stage failure, and every required stage executed
- **THEN** the label is `true` and every condition in the set is `true`

#### Scenario: condition_set_names_every_config_field

- **WHEN** the fidelity conditions are computed for any run
- **THEN** the condition record includes `config_solver_is_admm`, `config_compute_dtype_is_bf16`, `config_factor_mode_is_binary`, `config_ablation_is_none`, `config_layer_scope_is_all`, `config_admm_iter_matches_regime`, `config_admm_rho_end_matches_regime`, and `config_device_is_accelerator` — one condition per paper-regime config field — plus all existing stage-level conditions (`no_runtime_deviations`, `polish_off`, `no_discrete_flip_pass`, `no_scale_only_rescope`, `no_vacuous_teacher`, `capture_is_fp_stream`, `no_stage_failure`, `solver_is_admm`, and `required_*_executed`)

#### Scenario: solver_mismatch_names_the_deviation

- **WHEN** a run executes with `--solver svd` and all other config matches the paper regime
- **THEN** `config_solver_is_admm` is `false`, the label is `false`, and a reader can identify the denying condition from the condition record alone

#### Scenario: compute_dtype_mismatch_names_the_deviation

- **WHEN** a run executes with `--compute-dtype float32` and all other config matches
- **THEN** `config_compute_dtype_is_bf16` is `false` and the label is `false`

#### Scenario: admm_iter_mismatch_names_the_deviation

- **WHEN** a run executes with `--max-admm-iter 200` and all other config matches
- **THEN** `config_admm_iter_matches_regime` is `false` and the label is `false`

#### Scenario: device_mismatch_names_the_deviation

- **WHEN** a run executes with `--device cpu` and all other config matches
- **THEN** `config_device_is_accelerator` is `false` and the label is `false`

#### Scenario: no_hardcoded_dtype_in_cut_surface

- **WHEN** the implementation is complete
- **THEN** the cut surface (driver, report, fidelity conditions) contains no hardcoded `torch.float64`, `torch.float32`, `torch.bfloat16`, `np.float64`, or `np.float32` type constants — every dtype reference is parametric on `compute_dtype` or a named accumulation dtype per pipeline-dtype's no-hardcoded-dtype requirement

#### Scenario: label_denied_by_stage_condition_still_reports_config_match

- **WHEN** a run's config matches the paper regime but a vacuous-teacher event fires
- **THEN** all config-match conditions are `true`, `no_vacuous_teacher` is `false`, the label is `false`, and both facts are visible in the condition record

#### Scenario: no_request_flag_anywhere

- **WHEN** any run executes
- **THEN** no CLI flag, config field, or report identity field carries a fidelity-request value; the `paper_faithful_flag` key is absent from the condition record; the label is always computed and always reported

#### Scenario: eval_window_guard_dies_with_the_flag

- **WHEN** a run launches with a non-maximal `--eval-seqs` window
- **THEN** startup does not refuse it — the `--paper-faithful` eval-window guard is deleted with the flag; the sliced window is recorded in the report's evaluation record (selected blocks, source-token coverage) and never denies the computed label, because evaluation comparability is outside the method verdict

### Requirement: The driver CLI SHALL present an operator-grade help surface

Every surviving driver flag SHALL carry help text naming the flag's unit, its default value, and its effect on the artifact. Flags SHALL be grouped by function (model/data, solver/quality, output, resources). Help text SHALL NOT cite paper sections, change ids, lane names, or bot vocabulary. The word "paper" SHALL NOT appear in any flag's help string.

#### Scenario: help_output_has_operator_sections

- **WHEN** the driver is invoked with `--help`
- **THEN** the output includes grouped sections ("Model and data:", "Solver and quality:", "Output:", "Resources:") and every surviving flag appears under its section

#### Scenario: help_naming_unit_and_default

- **WHEN** the driver is invoked with `--help`
- **THEN** every flag's help text states its unit (tokens, sequences, iterations, fraction) and its default value

#### Scenario: help_free_of_bot_vocabulary

- **WHEN** the driver is invoked with `--help`
- **THEN** the output does not contain "paper", "faithful", change ids, or lane names

#### Scenario: cut_flags_absent_from_help

- **WHEN** the driver is invoked with `--help`
- **THEN** the output does not mention `--paper-faithful`, `--ste-flip-pass-budget`, `--probe-interval`, `--probe-batches`, `--factor-mode`, `--ablation`, or `--layer-scope`

### Requirement: Non-user mechanisms SHALL use environment variables with the QK_ prefix

Lane-internal controls that survive the CLI cut SHALL be settable only via environment variables with the `QK_` prefix. The driver SHALL read them at startup and SHALL NOT accept them as CLI flags. The controls this change migrates to env vars are `QK_PROBE_INTERVAL`, `QK_PROBE_BATCHES`, `QK_FACTOR_MODE`, `QK_ABLATION`, and `QK_LAYER_SCOPE`. Pre-existing `QK_*` env vars (`QK_RHO_END`, `QK_ADMM_REG`, `QK_SOLVER_PROBE`) are unaffected.

#### Scenario: env_var_wires_to_config

- **WHEN** `QK_FACTOR_MODE=fp` is set in the environment
- **THEN** `PTQConfig.factor_mode` is `"fp"` — the env var feeds the config field through the driver, not through a second config path

#### Scenario: env_var_defaults_preserve_behavior

- **WHEN** no `QK_*` env vars are set
- **THEN** the pipeline behavior is byte-identical to the pre-cut default run: `binary` mode, `none` ablation, `all` scope, probe interval 1, probe batches 2

#### Scenario: env_var_not_a_cli_flag

- **WHEN** a caller passes `--factor-mode fp` on the command line
- **THEN** the parser rejects it as an unrecognized argument — `QK_FACTOR_MODE` is the only path

#### Scenario: env_var_validation_rejects_invalid

- **WHEN** `QK_FACTOR_MODE=invalid` is set
- **THEN** the driver exits nonzero at startup naming the invalid value and the accepted choices

#### Scenario: env_var_empty_treated_as_unset

- **WHEN** `QK_FACTOR_MODE=` (empty string) is set
- **THEN** the default `"binary"` applies — an empty value is treated as unset, and this case is covered by a fixture test

#### Scenario: env_var_non_integer_rejected

- **WHEN** `QK_PROBE_INTERVAL=abc` or `QK_PROBE_BATCHES=1.5` is set
- **THEN** the driver exits nonzero at startup naming the invalid input, with the same named refusal path as a non-positive value

### Requirement: The discrete sign-flip pass SHALL be removed from the pipeline

The non-paper discrete sign-flip mechanism SHALL be deleted at every site: the `_discrete_flip_pass` definition, import, re-export, and call path. STE event records SHALL NOT carry a `flip_pass` key. `PTQConfig` SHALL NOT carry a `ste_flip_pass_budget` field. `no_discrete_flip_pass` SHALL remain in the condition set and SHALL always be `true`. The four runtime gates that consumed `PTQConfig.paper_faithful` SHALL re-key on the config-match conjunction.

#### Scenario: flip_pass_code_path_gone

- **WHEN** the STE stage completes on any layer
- **THEN** no discrete sign-flip pass executes; the post-STE path goes directly to sign extraction and packing per [NQ] Algorithm 1 lines 20–23, and the symbol `_discrete_flip_pass` does not exist anywhere in `src/` — definition (`_ste.py`), import (`_block_stage.py`), and re-export (`block_recon.py`) are all deleted

#### Scenario: flip_pass_never_in_conditions

- **WHEN** the fidelity conditions are computed for any run
- **THEN** `no_discrete_flip_pass` is `true` — it is always satisfied and can never be the denying condition

#### Scenario: ste_flip_pass_budget_field_absent

- **WHEN** `PTQConfig` is constructed
- **THEN** the object has no `ste_flip_pass_budget` attribute; accessing it raises `AttributeError`

#### Scenario: paper_ste_declared_from_config_match

- **WHEN** the STE stage completes on a paper-regime run (config matches all eight regime fields)
- **THEN** the per-linear `declared` field is `"paper_ste"` — the diagnostic path (discrete flip pass, scale-only rescope) is never invoked, and `no_scale_only_rescope` is always `true` on paper-regime runs

#### Scenario: non_paper_polish_runs_diagnostic_path

- **WHEN** a run with `factor_mode=binary` and `ablation=none` that deviates from the regime elsewhere (e.g. `--solver svd` or `--max-admm-iter 200`) reaches the STE stage
- **THEN** polish still runs (the gate is `factor_mode=binary ∧ ablation≠untuned-signs ∧ not config_matches_paper_regime`); on the post-STE ladder a linear with sign crossings declares `"paper_ste"` and a zero-crossing linear declares `"scale_only"` — the `declared` vocabulary is `{"paper_ste", "scale_only"}` and `discrete_flip_pass` no longer exists

#### Scenario: vacuous_teacher_abort_on_paper_regime_only

- **WHEN** a TuneFP pass encounters a zero first-step gradient
- **THEN** on a paper-regime run the driver raises `VacuousTeacherError` (fail-closed); on a non-paper run the event is recorded and the stage continues with pristine weights — the diagnostic, not the denial

#### Scenario: paper_regime_stage_budgets_are_app_c

- **WHEN** a paper-regime run (config matches all eight regime fields) constructs its stage budgets
- **THEN** the budgets are the [NQ] App. C triple — `stage_pre=StageHyper(1e-4, 4, 8)`, `stage_latent=StageHyper(1e-5, 1, 8)`, `stage_global=StageHyper(1e-6, 1, 8)` (lr, batch_size, epochs) — and any run failing the conjunction keeps the gate-shaped triple `StageHyper(1e-4, 4, 0)` / `StageHyper(2e-3, 4, 3)` / `StageHyper(1e-3, 4, 4)`

#### Scenario: stage_map_discrete_flip_pass_always_false_on_paper_regime

- **WHEN** the report stages summary is computed for a paper-regime run
- **THEN** the `ste.discrete_flip_pass` key is `false` — it stays in the schema for backward compatibility but is never `true` on a paper-regime run
