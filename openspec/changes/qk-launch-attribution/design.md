# qk-launch-attribution — Design

## Facts checked before authoring (2026-10-27)

- The crumb start line ALREADY carries pid + argv
  (`tools/quantize_hf_model.py:558`) — sec-eng's read of the trail as
  "no argv, no pid" is stale against the current tree. What is
  genuinely absent: the run's code identity (git rev), the resolved
  output target, and any guard against the shared default path.
- The shared-path hazard is real: the default trail is
  `$QUANTIBBLER_KD_TMPDIR/quantibbler-breadcrumbs.log`, so concurrent
  sandbox runs append interleaved lines — part of why the corpse
  reconciliation had to choose between "two attempts" and "unknown
  death."
- The container contract must not change: the wrapper sets
  `QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log` and
  appends dated run headers; forensics reads that exact path. The
  per-launch suffix therefore applies ONLY to the defaulted path; an
  explicit value is honored exactly. A container is single-run by
  construction, so the interleaving hazard does not exist there.

## Decisions

1. **Identity fields on the launch line, not a new file.** One line,
   first in every trail: `start pid=… argv=… rev=<git rev|unknown>
   out=<resolved out dir>`. A corpse then self-attributes regardless of
   which other mechanism (announcements, docker metadata, kernel
   records) failed.
2. **git rev at runtime, `unknown` when unresolvable.** The driver may
   legitimately run outside a checkout (CI smoke harness copies, bare
   scripts); the field must exist in every trail with an honest value.
3. **Pid-suffixed default, exact configured path.** Covers the
   concurrent-sandbox case without touching the container/forensics
   contract.
4. **Structural pin rides the same commit** — the launch-line
   construction joins the driver-wiring tripwire set
   (`tests/test_qk_report.py::_driver_wiring_findings`), same-edit
   discipline: the pin and the behavior move together, and a
   downstream edit that drops a field reds the tripwire test.

## Discarded alternatives

- **Always pid-suffix the trail file.** Rejected: breaks the wrapper's
  header/forensics contract for the container case, which has no
  concurrency to defend against.
- **A launch-registration service.** Rejected: network + a new
  component to secure; the trail file is already crash-proof and the
  corpse proved it survives the kill.
- **Prose-only rule.** Rejected: the protocol already exists in
  AGENTS.md and was insufficient — mechanism is the point of this
  change.

## Open Questions

- OQ-1 (owner: orchestrator): should the watchdog lines also carry the
  pid? No — the identity line plus per-launch files already binds
  every line to its launch; adding pid to each 5 s sample buys
  nothing. Status: CLOSED at authoring.
