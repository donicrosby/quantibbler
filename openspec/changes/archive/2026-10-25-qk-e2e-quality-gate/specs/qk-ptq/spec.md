## ADDED Requirements

### Requirement: The real-model driver SHALL report an end-to-end perplexity ratio against the same run's bf16 baseline

The real-model driver SHALL compute `ppl_ratio = ppl_quant_native / ppl_baseline_bf16` over one held-out eval partition through one logits path (the host object quantize_model wrote the artifact back into), and SHALL record both perplexities, the ratio, and the byte-derived effective BPW in `quantize_report.json`.

#### Scenario: ratio_single_measurement_path

- **WHEN** the driver measures quantized perplexity after `quantize_model` writes the artifact back into the host
- **THEN** the reported ratio uses exactly that measurement — no re-import or second measurement path — and `ppl_baseline_bf16`, `ppl_quant_native`, and `ppl_ratio` all appear in `quantize_report.json`

#### Scenario: ratio_history_all_refused

- **WHEN** the verdict logic replays the recorded production shapes (ratios 1,156,577.1; 943.49; 5,517.8; 463,742.5; 524,894.8; and NaN)
- **THEN** every shape classifies `failed` — no historical artifact ever receives `ok` or `degraded`

### Requirement: Every driver run SHALL receive a three-value end-to-end quality verdict from a paper-calibrated ratio band

The driver SHALL classify each run `ok`, `degraded`, or `failed` from the end-to-end ratio against two derived band edges, expose the classification in the report `status` field, and recur the same classification into the process exit code.

#### Scenario: verdict_ok_within_functional_envelope

- **WHEN** a run's ratio is at or below `OK_LINE` = `ceil(23.74/6.10)` = 4.0 — the functional envelope at the extreme budget end of [LB2] Sec. 5.2 Table 1 (LittleBit-2 at 0.1 bpp, PPL 23.74, against the FP16 6.10)
- **THEN** the verdict is `ok` and the process exits 0

#### Scenario: verdict_degraded_between_lines

- **WHEN** a run's ratio exceeds `OK_LINE` and is at or below `COLLAPSE_LINE` = `ceil(241.96/6.10)` = 40.0 — the documented method collapse of [LB2] Sec. 5.2 Table 1 (STBLLM at 0.55 bpp, PPL 241.96, against the FP16 6.10)
- **THEN** the verdict is `degraded`, the process exits 0, and the report carries the degraded status rather than a silent pass

#### Scenario: verdict_failed_above_collapse_line

- **WHEN** a run's ratio exceeds `COLLAPSE_LINE`, or the ratio is missing or non-finite
- **THEN** the verdict is `failed` and the process exits nonzero; the missing/non-finite check always precedes any threshold comparison

#### Scenario: verdict_edge_arithmetic_pinned

- **WHEN** the suite runs
- **THEN** a test asserts both edges' derivations exactly — `ceil(241.96/6.10) == 40` and `ceil(23.74/6.10) == 4` — and asserts the boundary semantics (`ratio == 40.0` → `degraded`; `ratio == 40.0001` → `failed`) plus the trap that `nan > 40.0` is False so a threshold comparison alone can never catch NaN

#### Scenario: verdict_helpers_torch_free_tested

- **WHEN** the test suite runs in a lean environment without torch
- **THEN** the verdict and sanitizer tests execute (no skips, no collection errors) because the logic lives outside the torch-importing driver, in the compressor package

### Requirement: quantize_report.json SHALL be provenance-complete and strict-JSON-parseable

The report SHALL name the actual run — top-level `solver` and `scale_dtype` from the argument values, `status` from the verdict — and SHALL emit `regime.artifact` and `config_f64.dtype_note` from those same arguments rather than hardcoded literals, and SHALL write every non-finite float as `null` so the report parses under a NaN-rejecting strict parser.

#### Scenario: report_mirrors_run_args

- **WHEN** a run executes with `--solver svd --scale-dtype float32`
- **THEN** the report records `solver: svd` and `scale_dtype: float32` top-level, and `regime.artifact` names float32 scales — the float16 literal never appears for a float32 run

#### Scenario: report_fp16_opt_in_named

- **WHEN** a run executes with the `--scale-dtype float16` opt-in
- **THEN** `regime.artifact` and `config_f64.dtype_note` name float16 scales, recorded from the arguments rather than from a constant

#### Scenario: report_nonfinite_as_null

- **WHEN** any report value is NaN or infinite — including a numpy scalar float such as `np.float32(nan)` — or writing the full report fails for any reason
- **THEN** non-finite values are written as `null` with `status: failed` and the file parses under a `parse_constant`-rejecting strict JSON parser; if serializing or writing the report itself raises, the driver SHALL still write a minimal `{"status": "failed", "report_error": ...}` report, exit nonzero, and never leave the run without a durable status; a healthy report parses with `status: ok`

### Requirement: The driver process SHALL fail closed on a failed verdict while always exporting the artifact

`main()` SHALL return 0 only for `ok` or `degraded` verdicts and a nonzero exit code for `failed`, and SHALL write the qpk1 export and `quantize_report.json` regardless of verdict, so a failed run still leaves a forensically complete artifact.

#### Scenario: failed_run_exits_nonzero

- **WHEN** a run's verdict is `failed` (including the missing-or-non-finite ratio case)
- **THEN** the process exit code is nonzero, distinguishable from a pass by any CI or orchestrator chain

#### Scenario: healthy_run_exits_zero

- **WHEN** a run's verdict is `ok` or `degraded`
- **THEN** the process exits 0

#### Scenario: export_and_report_persist_on_failure

- **WHEN** the gate fails the run
- **THEN** the qpk1 model directory and `quantize_report.json` are still written, with the report carrying `status: failed` as the durable trace
