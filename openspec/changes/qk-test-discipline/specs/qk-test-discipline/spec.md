# Test discipline delta

## ADDED Requirements

### Requirement: suite_time_budget

The default (hot) lane SHALL complete within 30 seconds of wall time per 100 collected tests, measured with the BLAS determinism pin, and recorded in the run report. The budget is a hard gate: a suite over budget is red, not advisory. Baseline measured 2026-10-24: 349.36s for 628 tests (55.6s/100); after the honest split, 621 tests in 71.48s (11.5s/100). The fix is profiling the slow tail and splitting oversized test modules, never weakening assertions or shrinking coverage to meet the clock.

#### Scenario: over_budget_suite_is_red

- **WHEN** a hot-lane run exceeds 30s per 100 collected tests
- **THEN** the run reports the ratio against the baseline and the suite is treated as failed
- **AND** the response is a measured slow-tail fix with the profile attached, not a coverage cut

### Requirement: integration_marker_admission

`integration` SHALL mean the test loads a real model or executes the driver end-to-end. A CPU stub harness test SHALL be classified as a unit test — calling `build_real_arms` on a `tiny_model` stub is a harness unit test, and that classification is its entire value.

#### Scenario: death_class_guards_stay_hot

- **WHEN** a test is marked integration
- **THEN** it SHALL be one that loads a real model or runs the driver end-to-end — the three real stage-map driver runs, and nothing cheaper
- **AND** the lane0 stub fixtures born from each death class SHALL remain in the default lane, because the default lane is where the suite actually runs
- **AND** over-marking is a red: the 2026-10-24 sweep moved the two-batch Arm B fixture, the four-row surface-table pin, the capture-binding sentinel, and the stubbed-corpus e2e off the hot lane at zero clock savings — the three stage-map driver runs were the entire 222.85s, the stubs were never the cost

### Requirement: integration_lane_mandatory_before_fire

The integration lane (`pytest -m integration`) SHALL be green before staging any fire, and a default-lane run that deselects it SHALL announce the deselection loudly rather than silently. The fire staging proof SHALL carry the integration-lane record.

#### Scenario: fire_requires_integration_lane

- **WHEN** a fire is staged
- **THEN** the staging proof includes a green integration-lane run and the hot-lane clock record
- **AND** a default run that skipped integration without the loud banner is refused as a staging artifact

### Requirement: no_production_fire_in_tests

Tests SHALL NOT execute a production driver run, invoke the launch wrapper against a real daemon, or start containers. `docker` may appear in a test only as text being parsed or asserted on.

#### Scenario: real_fire_under_pytest_is_a_defect

- **WHEN** a test's invocation path reaches a real daemon or a real container
- **THEN** the test is red by definition
- **AND** any test that must invoke real infrastructure SHALL prove its interception in-band — the resolved binary is the shim — before the code under test runs, and SHALL NOT hold the production container name; the PATH-shim fall-through that started `quantibbler-ptq-run` from pytest is the defect class this requirement removes, and a test holding the production container name is refused at review

### Requirement: slow_test_justification

Any single test exceeding 5 seconds SHALL name its reason in its docstring —
what it exercises that cannot be made faster, and why the cost is worth it.
An unannotated slow test is a defect like any other failure.

#### Scenario: unannotated_slow_test_is_a_defect

- **WHEN** a test exceeds 5 seconds without a docstring justification
- **THEN** the test is reported as a defect alongside failures
