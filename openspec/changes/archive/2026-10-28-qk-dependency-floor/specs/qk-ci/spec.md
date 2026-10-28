# qk-dependency-floor — Spec Delta (qk-ci)

## REMOVED Requirements

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
- **THEN** it uses the proven recipe — a fresh `uv venv` with `uv pip install -e . pytest` plus the referee's test-harness dependency `pyyaml` (which is not in the product floor) and no extras — so the absence assertion still tests the shipped dependency floor, not a hand-pruned environment

## ADDED Requirements

### Requirement: CI SHALL run the full suite on the locked project environment and referee the product/dev dependency split

The full-python job SHALL sync the locked project environment (the product's own dependencies plus the default dev group) and SHALL fail when pytest reports any skipped test; the product-floor job SHALL install the project's declared dependencies alone and SHALL prove both halves of the split — the PTQ lane imports from them, and the test harness does not.

#### Scenario: full_python_zero_skips

- **WHEN** the full-python job runs pytest
- **THEN** the skip count is parsed from the pytest summary line, and a count above zero fails the job — a skipped ptq tier is a hole, not a lighter suite

#### Scenario: product_floor_carries_the_pipeline

- **WHEN** the product-floor job builds its interpreter with `uv pip install -e .` (the declared dependencies only — no extra, no dependency group)
- **THEN** `import numpy, torch, transformers, tokenizers, safetensors, pyarrow` succeeds in that interpreter, because the PTQ pipeline is the product and its dependencies are the product's own

#### Scenario: dev_tooling_is_not_product

- **WHEN** the product-floor interpreter is probed for the suite's harness
- **THEN** `import pytest, yaml` fails there, so a harness package in the product's declared dependency set reds the job

#### Scenario: product_floor_interpreter_floats

- **WHEN** the product-floor job builds its interpreter
- **THEN** it uses a fresh `uv venv` plus `uv pip install -e .` with no lock flag — it floats on purpose, because "does the shipped dependency set still install and import" is its signal, not determinism
