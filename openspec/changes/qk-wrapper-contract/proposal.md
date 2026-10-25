# Proposal: qk-wrapper-contract — as-built wrapper contract specification

## Why

Seven wrapper-contract requirements were authored during the Lane 0 incident response and landed in `qk-nanoquant-alignment-lane0`'s spec delta, but their implementation lives in the surviving wrapper files (`tools/run-ptq-container.sh`, `tests/test_run_ptq_wrapper.py`) — not in the lane0 measurement experiment. The lane0 change is being archived; these universal requirements need their own capability spec before they're orphaned.

## What Changes

- New capability spec `qk-wrapper` captures 7 requirements extracted verbatim from `qk-nanoquant-alignment-lane0`: `container_log_witness`, `fire_monitoring_cadence`, `guards_shall_be_tri_state`, `wrapper_failclosed_branches_hermetic_tests`, `launch_lines_derived_at_firing`, `launch_env_single_writer`, `infrastructure_interception_proof`.
- No code changes — the implementation already exists in `tools/run-ptq-container.sh` and `tests/test_run_ptq_wrapper.py`.
- This is an as-built spec: it documents the wrapper's current contract for future changes to reference.

## Non-goals

- No new wrapper features, guards, or tests.
- No changes to `qk-launch-attribution` (crumb trails — a separate concern).
- No changes to the `qk-ptq` main spec (archive of lane0's core delta handles that).

## Impact

- Specs: `qk-wrapper` +7 requirements (new capability).
- Code: none — the implementation exists.
- No storage-format, solve, or server impact.
