# qk-test-discipline Specification

## Purpose

Define the test-suite governance contract: a hard wall-time budget per 100 collected tests on the default hot lane, a strict `integration` marker definition that keeps cheap harness stubs in the default lane where the suite actually runs, and the admission criterion that every new test meets before it becomes part of the gate.

## Requirements

### Requirement: suite_time_budget

The default (hot) lane SHALL complete within 30 seconds of wall time per 100 collected tests, measured with the BLAS determinism pin, and recorded in the run report. The budget is a hard gate: a suite over budget is red, not advisory. The measurement SHALL be serial (`-n 0` / default execution) — pytest-xdist parallel wall time is a different quantity with different variance, and is never the budget's denominator.

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

#### Scenario: guards_home_acceptance

- **WHEN** the marker correction lands
- **THEN** `uv run pytest tests/ptq/test_lane0_probe.py` prints 36 passed, 0 deselected — the deselection banner's absence IS the test that the guards came home — and the default-lane wall line records the before/after clock
- **AND** the hot-lane cost of the guards SHALL be driven down by sizing the stub corpus to the property under test (two batches at reduced seq_len preserves the batch-boundary coverage at ~8× less trunk cost), never by re-homing them to another lane

### Requirement: integration_lane_standing_cadence

The integration lane SHALL run on a standing cadence — nightly — with a red run surfaced to the room, so the mandatory pre-fire check is a formality, not a discovery. A gate that runs only at pre-fire discovers its rot at the moment of maximum impatience and gets waived under pressure.

#### Scenario: nightly_integration_red_surfaces

- **WHEN** the standing integration-lane run is red
- **THEN** the red is surfaced to the room the same night, with the failing test named
- **AND** a pre-fire integration run is verification of the standing record, never the first execution of the lane since the last fire

### Requirement: integration_lane_mandatory_before_fire

The integration lane (`pytest -m integration`) SHALL be green before staging any fire, and a default-lane run that deselects it SHALL announce the deselection loudly rather than silently. The fire staging proof SHALL carry the integration-lane record.

#### Scenario: fire_requires_integration_lane

- **WHEN** a fire is staged
- **THEN** the staging proof includes a green integration-lane run and the hot-lane clock record
- **AND** a default run that skipped integration without the loud banner is refused as a staging artifact

### Requirement: prose_fence_stays_hot

The incident-narrative fence SHALL stay in the hot lane: `tests/test_prose_discipline.py` fails on the incident-ledger token class (`r137`, `run #N`, `post-mortem`, `user-directed`, `lane history`, `autopsy`) anywhere in `src/` or `tools/`. Derivation comments without run tags SHALL pass untouched. A commit that removes or waters down the fence is a red like any other failure.

#### Scenario: fence_regrowth_is_red

- **WHEN** an incident narrative reappears in product source
- **THEN** the hot lane fails on the token class, and the fix is exiling the story to its openspec change dir — keeping the durable rule and its measured numbers in 1–3 comment lines — never an exception added to the fence

### Requirement: no_production_fire_in_tests

Tests SHALL NOT execute a production driver run, invoke the launch wrapper against a real daemon, or start containers. `docker` may appear in a test only as text being parsed or asserted on.

#### Scenario: real_fire_under_pytest_is_a_defect

- **WHEN** a test's invocation path reaches a real daemon or a real container
- **THEN** the test is red by definition
- **AND** any test that must invoke real infrastructure SHALL prove its interception in-band before the code under test runs, and SHALL NOT hold the production container name; the PATH-shim fall-through that started `quantibbler-ptq-run` from pytest is the defect class this requirement removes, and a test holding the production container name is refused at review
- **AND** in-band proof means a logged call from an *executed* shim — the fixture SHALL execute the shim once and observe its log line before trusting it; a resolution check (`command -v`, `chmod`) is not proof, because PATH resolution and executability are different properties (`/tmp` is mounted `noexec`: the 2026-10-24 seam showed a resolved, executable-bit-carrying stub that returned EACCES on every call, indistinguishable from a free name by any resolution check)
- **AND** executable test fixtures SHALL stage on an exec-capable mount — every tmpfs on this host (`/tmp`, `/dev/shm`, `/var/tmp`) is mounted `noexec`, and pytest's default `tmp_path` lives under `/tmp`, so the default fixture location IS the trap; executables stage under `/opt/data` (ext4), e.g. `/opt/data/tmp/…` or pytest `--basetemp` pointed there, and the self-exec check is the belt that makes any drift back onto a tmpfs fail loudly instead of silently

### Requirement: slow_test_justification

Any single test exceeding 5 seconds SHALL name its reason in its docstring —
what it exercises that cannot be made faster, and why the cost is worth it.
An unannotated slow test is a defect like any other failure.

#### Scenario: unannotated_slow_test_is_a_defect

- **WHEN** a test exceeds 5 seconds without a docstring justification
- **THEN** the test is reported as a defect alongside failures

### Requirement: derivation_comments_on_harness_math

Any function in `tools/` that computes a quantity defined in `src/` or the paper SHALL carry a derivation comment naming that source, or a disposition naming why sharing is wrong. Inline blocks computing such quantities count as functions for this rule — a nameless copy is the drift class the inventory caught.

#### Scenario: unnamed_derivation_is_a_defect

- **WHEN** a `tools/` function or inline block computes a `src/` or paper quantity with no derivation comment and no valid disposition
- **THEN** it is a review defect, fixed by naming the source — or by extracting the inline block into a named function so a name can carry the derivation
- **AND** the disposition `(candidate under measurement; folding it into production would make the experiment vacuous)` is valid and earns a stay

### Requirement: transforms_carry_identity_property_tests

Any transform claiming to modify its input SHALL be born with a property test asserting output ≠ input on the adversarial class it claims to handle. The vacuous clip — clipping every entry at the vector's own maximum, which is no clip — survived review because nothing asserted the operation does anything.

#### Scenario: noop_transform_is_red

- **WHEN** a claimed transform's output equals its input on the class it exists to act on (entries above `tau_max`, for a clip)
- **THEN** the property test goes red in seconds against a single planted vector, and the transform is a no-op defect regardless of what its docstring cites

### Requirement: error_messages_name_the_invariant_the_code_tests

Any error message emitted by a guard SHALL state the invariant the code tests, and every remedy SHALL be valid for every instance of that condition. The message text is part of the line's contract; a fixture SHALL assert the emitted message names the invariant.

#### Scenario: correct_condition_with_wrong_remedy_is_red

- **WHEN** a guard tests condition C (e.g. `loss.requires_grad`) and states a remedy valid only for a strict subset of C's instances (e.g. "unfreeze your weights" — irrelevant to a detached head with a correct value but a severed graph)
- **THEN** the guard is a defect: the remedy sends the caller to fix something that isn't broken, and the condition itself — which IS correct — gets disbelieved because the message undermined it
- **AND** the fix states the invariant explicitly in the message ("The invariant being guarded is the GRAPH, not the weights…") so the caller can diagnose from the text alone without reading source

#### Scenario: invariant_naming_is_testable

- **WHEN** a guard's fixture asserts the emitted error message names the invariant the code tests (e.g. "GRAPH" for `loss.requires_grad`)
- **THEN** pinning text to contract is a mechanical assertion no different from pinning output to input — the guard's test proves it names the right thing the same way the property test proves the transform does the thing
- **AND** the three instances from Lane 1 (robust_clip_shrink cited Eq. 3 while clipping at its own max; the guard tested the graph and blamed the weights; the guard tested the graph and offered a remedy irrelevant to detached heads) prove the failure class: text attached to a line is part of its contract, and disbelieving a correct guard because its error message was wrong is the same outcome as a wrong guard with a correct message — the caller gets wrong information either way

### Requirement: green_tests_prove_detection_power

A test asserted green on current bytes SHALL have its detection power demonstrated before it counts in any tally or sign-off number. The standard is mutation: flip the guarded mechanism, assert the test fails on the real mechanism, restore, verify the tree byte-identical. A green that cannot be made to fail under any plausible regression is an instrument or theater, not coverage.

#### Scenario: mutation_proves_the_guard

- **WHEN** a newly green test joins the suite (e.g. a vacuity probe asserting a skip is by omission)
- **THEN** the guarded condition is mutated once (`idx >= 1` → `idx >= 0`), the test is shown to fail naming the real mechanism, the mutation is restored, and the tree is verified byte-identical afterward
- **AND** a green whose failure mode cannot fire — the Lane 2 task-4 vacuity assertion before its gradient measurement gave it teeth — is excluded from phase tallies and sign-off counts, or upgraded until it can fail
