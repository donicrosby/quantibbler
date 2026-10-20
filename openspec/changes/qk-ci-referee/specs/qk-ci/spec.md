## ADDED Requirements

### Requirement: CI SHALL execute the full suite, never a skipped skeleton

The workflow's full-python job SHALL sync the `torch`, `hf`, and `kernels` extras explicitly and SHALL fail when pytest reports any skipped test; the lean-python job SHALL prove, in its own interpreter, that `torch`, `transformers`, and `tokenizers` cannot be imported, and only after that absence is proven SHALL it assert its documented expected skip count instead of ignoring skips.

#### Scenario: full_python_zero_skips

- **WHEN** the full-python job runs pytest
- **THEN** the skip count is parsed from the pytest summary line, and a count above zero fails the job — pre-extras suites (`62 passed, 16 skipped`) fail this job

#### Scenario: lean_python_asserts_absence_before_skips

- **WHEN** the lean-python job runs
- **THEN** it first proves absence in its interpreter — `python -c "import torch, transformers, tokenizers"` must exit nonzero, because skip discipline differs per extra and a torch-synced "lean" run is a false lean — and only then asserts its tier's documented expected skip count from the pytest summary line, legitimate tier skips being expected values, not ignorable noise

#### Scenario: lean_python_interpreter_built_without_extras

- **WHEN** the lean-python job builds its interpreter
- **THEN** it uses the proven recipe — a fresh `uv venv` with `uv pip install -e . pytest` and no extras — so the absence assertion tests the shipped dependency floor, not a hand-pruned environment

### Requirement: CI SHALL execute the driver end-to-end on a tiny offline model and assert the gate contract

The driver-smoke job SHALL build a from-scratch tiny `Qwen3Config`, run the driver fully offline, and SHALL assert on the produced `quantize_report.json`: `status` present, strict-parse clean under a `parse_constant`-rejecting parser, process `exit_code == verdict_exit_code(status)`, and provenance fields equal to the invoked arguments.

#### Scenario: smoke_proves_gate_executes

- **WHEN** the driver-smoke job completes
- **THEN** the report assertions hold — proving the gate's code path executes end-to-end at runtime, not merely that its unit tests pass

#### Scenario: smoke_claims_plumbing_only

- **WHEN** the driver-smoke job runs on the random-init tiny fixture
- **THEN** it is cited and labeled as plumbing-only coverage — a random-init model's next-token distribution is already uniform (PPL approx vocab, measured rank-1 quantized ratio 1.0000006 and a zeroed `o_proj` shifting PPL 3 ppm), so no claim of detection or abort-path coverage rides on it

#### Scenario: smoke_offline_by_env

- **WHEN** the driver-smoke job starts the driver
- **THEN** the HF-offline environment variables are set and no hub or network access is exercised — the smoke proves code paths, not connectivity

### Requirement: CI SHALL validate every openspec change strictly, one id per invocation

A workflow job SHALL iterate the non-archived `openspec/changes/*` directories and run `openspec validate <id> --strict` for each, failing the job on any invalid change — never a two-argument or silent-skip invocation.

#### Scenario: each_change_validated

- **WHEN** the openspec job runs
- **THEN** every active change dir is validated individually and zero return codes are required for the job to pass

### Requirement: CI SHALL prove the probe-abort path repeatably on a real checkpoint

A weights-cached leg SHALL run the recorded abort command (`Qwen/Qwen3-0.6B --rank 1 --solver svd --max-admm-iter 1 --seq-len 64 --calibration-seqs 1 --eval-seqs 2 --probe-batches 1`) fully offline and SHALL assert: the run aborts before block 3, wall time is minutes not tens of minutes, the report carries `status: failed` plus the `probe_abort` fields and the history so far, and the process exits nonzero — the catch→report→exit wiring executed under CI, not merely in a manual run.

#### Scenario: abort_leg_repeatable_detection_proof

- **WHEN** the weights-cached abort leg runs
- **THEN** the measured collapse (prefix ratio ~35,000 at depth 2 against `COLLAPSE_LINE` 40) aborts the driver in the recorded command's shape, with report and exit code asserted from the real process — this leg is the repeatable runtime proof the probe-abort path executes

#### Scenario: abort_leg_weights_cached_offline

- **WHEN** the abort leg needs the Qwen3-0.6B checkpoint and wikitext snapshot
- **THEN** they come from a restored cache (repo cache or self-hosted runner bind-mount — mechanism is the implementer's choice) with the HF-offline environment set, never a live hub download at run time

### Requirement: CI SHALL be shell-rigorous and supply-chain-hardened

Every workflow shell step SHALL run under `bash -euo pipefail`; every third-party action SHALL be pinned by full commit SHA (never a tag or branch); the workflow SHALL declare `permissions: contents: read`, trigger on `pull_request` and on `push` to `main` — never `pull_request_target` — and declare no secrets.

#### Scenario: push_to_main_trigger_is_safe

- **WHEN** the workflow triggers on push to main
- **THEN** it executes only content already at main's head, adding no PR-content execution surface — and without it this repo's direct-push landing practice would never be refereed

#### Scenario: no_masked_gate_rcs

- **WHEN** any step command pipes a gate's output
- **THEN** pipefail propagates the gate's exit status — a green filtered tail can never mask a red gate

#### Scenario: sha_pinned_only

- **WHEN** the workflow is audited
- **THEN** every `uses:` line names an immutable full-SHA reference, and no `pull_request_target` trigger exists

### Requirement: CI SHALL execute every crate that carries a purity guard

The rust job SHALL select its gate packages explicitly instead of scoping to the kernels manifest, so `crates/quantibbler-server` — which carries the no-libpython and cubek-purity guards — is executed rather than merely present; the referee SHALL pin that selection.

#### Scenario: rust_job_executes_the_server_purity_guards

- **WHEN** the rust job runs its test and clippy gates
- **THEN** both name `-p quantibbler-kernels` and `-p quantibbler-server`, because a `--manifest-path` selection of the kernels crate alone never executes `crates/quantibbler-server/tests/binary_purity.rs` (no libpython in the served binary) or `cubek_purity.rs` (cubek-attention reachable under `gpu`, `cpu-reference` never enabled) — a guard that exists but no job runs is decoration, and the cubek adoption would otherwise ship on arrival with its purity pair unenforced

#### Scenario: referee_pins_the_job_selection

- **WHEN** the workflow is audited
- **THEN** a pin asserts both package names appear in the rust job's test and clippy lines and that neither gate uses `--manifest-path`, so a quiet revert to kernels-only reds the referee instead of silently re-opening the local-only gap
