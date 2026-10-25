# Proposal: qk-launch-attribution — self-attributing crumb trails

## Why

The 2026-10-20 corpse (kernel-provable kill of a seq-2048-class tune
attempt, 33,480,724 kB against the 32g memcg) left a surviving trail
that still could not name its launcher. The launch protocol in
AGENTS.md is the discipline layer; the mechanism layer is missing. The
crumb start line already carries `pid` + `argv`
(`tools/quantize_hf_model.py:558`), but a run's *code identity* (git
rev) and *output target* (resolved out dir) are absent — and the
default trail path is shared, so two concurrent sandbox runs append
interleaved lines into one file, forcing a "two attempts or one
unknown death" choice instead of a read-off attribution. Discipline
then mechanism is this repo's conversion pattern (secrets, skip
counts, pin shapes); this is the mechanism.

## What Changes

- The launch crumb line gains `git rev` (resolved at runtime from the
  repo; literal `unknown` when unresolvable, e.g. outside a checkout)
  and the resolved output directory.
- The default trail path becomes per-launch: `breadcrumbs-<pid>.log`
  in the resolved KD scratch when `QUANTIBBLER_BREADCRUMBS` is unset,
  so concurrent runs never interleave. When the variable IS set (the
  wrapper's container contract), the configured path is honored
  exactly — single-run container, forensics reads one file.
- The start-line construction is pinned structurally in the
  driver-wiring tripwire set (same edit as the behavior, per the
  pin-shape discipline).

## Non-goals

- No launch authorization, no central log server, no network.
- No change to the container contract: the wrapper's dated run headers
  and bug-hunter's forensics procedure read the configured path
  unchanged.

## Impact

- Specs: qk-ptq +2 requirements (launch identity line; trail
  disambiguation).
- Code: `tools/quantize_hf_model.py` (start line + default path),
  `tests/ptq/test_launch_attribution.py` (new), tripwire pin
  extension in `tests/test_qk_report.py`.
- No storage-format, qpk1, or server impact.
