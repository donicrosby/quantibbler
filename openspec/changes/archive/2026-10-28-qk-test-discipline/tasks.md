# Tasks: qk-test-discipline

## Wired (mechanism exists in tree)

- [x] prose_fence_stays_hot — tests/test_prose_discipline.py bans incident-narrative tokens in src/ and tools/
- [x] integration_marker_admission — pyproject.toml:57 addopts deselects integration; conftest.py prints loud deselection banner

## Partial (mechanism exists, incomplete)

- integration_lane_mandatory_before_fire — conftest.py announces deselection loudly, but no pre-fire enforcement (refusal without integration-lane green) exists anywhere in the tree
- no_production_fire_in_tests — tests/stubs/docker + wrapper PATH-shim tests exist, but no suite-wide docker-usage enforcement

## Debt (no mechanism — spec text only)

- suite_time_budget — no budget enforcement in CI, pytest config, or conftest
- integration_lane_standing_cadence — no nightly integration CI workflow
- slow_test_justification — no slow-test scanner
- derivation_comments_on_harness_math — no derivation scanner
- transforms_carry_identity_property_tests — no identity-property test gate
- error_messages_name_the_invariant_the_code_tests — no invariant-naming gate
- green_tests_prove_detection_power — no mutation-proving gate

## Deferred

- The nine unwired requirements stay as spec text only. They become active tasks when their owning lane is dispatched; do not archive this change until every requirement has a mechanism or is explicitly removed from the delta.
