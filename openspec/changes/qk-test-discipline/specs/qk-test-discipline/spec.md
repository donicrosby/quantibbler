# Test discipline delta

## Why

The user set a hard budget: tests SHALL NOT exceed 30 seconds per 100 tests.
Measured baseline 2026-10-24: 349.36s wall for 628 tests = 55.6s/100 — red by
85%. A second class of defect is the production fire executed "as a test":
a PATH-shim test silently fell through to the real docker daemon and started
the production container name with no authorization. Both become law.

## What Changes

- `qk-test-discipline`: the suite time budget is a hard gate, a production
  fire under pytest is a defect by definition, and slow tests must justify
  themselves in their docstrings.

## Impact

- Test code and CI gating only; no production code paths change.
- Fixing the budget requires profiling the suite's slow tail (top offenders
  are the 2088-line `tests/test_qk_report.py` and the 1066-line
  `tests/test_ci_referee.py`) — behavior-preserving splits, never assertion
  weakening.

## ADDED Requirements

### Requirement: suite_time_budget

The full test suite SHALL complete within 30 seconds of wall time per 100
collected tests, measured on the CI lane with the BLAS determinism pin, and
recorded in the run report. The budget is a hard gate: a suite over budget is
red, not advisory. Baseline measured 2026-10-24: 349.36s for 628 tests
(55.6s/100) — the fix is profiling the slow tail and splitting oversized test
modules, never weakening assertions or shrinking coverage to meet the clock.

#### Scenario: over_budget_suite_is_red

- **WHEN** a full-suite run exceeds 30s per 100 collected tests
- **THEN** the run reports the ratio against the baseline and the suite is
  treated as failed
- **AND** the response is a measured slow-tail fix with the profile attached,
  not a coverage cut

### Requirement: no_production_fire_in_tests

Tests SHALL NOT execute a production driver run, invoke the launch wrapper
against a real daemon, or start containers. `docker` may appear in a test only
as text being parsed or asserted on. Any test that must invoke real
infrastructure SHALL prove its interception in-band (resolved binary is the
shim) before the code under test runs, and SHALL NOT hold the production
container name. The PATH-shim fall-through that started `quantibbler-ptq-run`
from pytest is the defect class this requirement removes.

#### Scenario: real_fire_under_pytest_is_a_defect

- **WHEN** a test's invocation path reaches a real daemon or a real container
- **THEN** the test is red by definition and the interception proof is the
  only permitted fix
- **AND** a test holding the production container name is refused at review

### Requirement: slow_test_justification

Any single test exceeding 5 seconds SHALL name its reason in its docstring —
what it exercises that cannot be made faster, and why the cost is worth it.
An unannotated slow test is a defect like any other failure.

#### Scenario: unannotated_slow_test_is_a_defect

- **WHEN** a test exceeds 5 seconds without a docstring justification
- **THEN** the test is reported as a defect alongside failures
