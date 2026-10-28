# Design: qk-test-discipline

## 2026-10-25 — re-verdict of DELETE (planner card t_d320f8bc)

The spec-audit (t_b262f90d) verdict #21 said DELETE because the two-level gate supposedly returned zero mechanism sites. The executor (t_039035de) measured the gate and found L2 fails: at least two requirements have tracked mechanisms in the tree. The change dir is live (15 commits, most recent b2864ce), not a dead proposal.

### Ruling

**Keep alive as a spec-only change.** The dir gets tasks.md recording wired vs debt. No archive (9 of 11 requirements have no mechanism — archiving would create a main spec that falsely reads as done). No delete (2 wired requirements would lose their requirement-of-record).

### Requirement to mechanism evidence table

| Requirement | Status | Mechanism |
|---|---|---|
| prose_fence_stays_hot | WIRED | tests/test_prose_discipline.py — tracked, landed f4e5d37 |
| integration_marker_admission | WIRED | pyproject.toml:57 addopts plus conftest.py deselection accounting |
| integration_lane_mandatory_before_fire | PARTIAL | conftest.py announces deselection loudly — no pre-fire enforcement |
| no_production_fire_in_tests | PARTIAL | tests/stubs/docker plus wrapper PATH-shim tests — no suite-wide enforcement |
| suite_time_budget | DEBT | No mechanism |
| integration_lane_standing_cadence | DEBT | No nightly CI workflow |
| slow_test_justification | DEBT | No mechanism |
| derivation_comments_on_harness_math | DEBT | No mechanism |
| transforms_carry_identity_property_tests | DEBT | No mechanism |
| error_messages_name_the_invariant_the_code_tests | DEBT | No mechanism |
| green_tests_prove_detection_power | DEBT | No mechanism |

### Non-transfer clause

The wired requirements (prose fence, integration marker) stay wired regardless of this change dir's lifecycle. If this change dir is later deleted before archive, those requirements MUST be re-homed into another spec before deletion.
