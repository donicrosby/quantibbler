# Proposal: qk-test-discipline

## Why

The user set a hard budget: tests SHALL NOT exceed 30 seconds per 100 tests.
Measured baseline 2026-10-24: 349.36s wall for 628 tests = 55.6s/100 — red by
85%. A second defect class: a PATH-shim test silently fell through to the real
docker daemon and started the production container name from pytest. Both
become law.

## What Changes

- Hard suite-time gate: 30s per 100 collected tests, measured and recorded.
- Production fires under pytest are defects by definition; interception-proof
  tests are the only permitted path to real infrastructure.
- Any test over 5s justifies itself in its docstring or is reported.

## Impact

Test code and CI gating only. The fix path is profiling the slow tail and
splitting oversized test modules — never weakening assertions or shrinking
coverage to meet the clock.
