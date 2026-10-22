# qk-launch-attribution — Spec Delta (qk-ptq)

## ADDED Requirements

### Requirement: The launch crumb line SHALL attribute the run to code, process, and output target

The PTQ driver's breadcrumb trail SHALL begin with a launch line
carrying the process id, the exact argv, the repository git revision
(literal `unknown` when unresolvable, e.g. running outside a
checkout), and the resolved output directory — so that a surviving
trail self-attributes its run even when the launch protocol is
violated, without recourse to kernel records or container metadata.

#### Scenario: launch identity line inside a checkout

- **WHEN** the driver starts inside a git checkout with a fake argv
  and a temp scratch dir,
- **THEN** the trail's first line contains the pid, the fake argv, the
  repo's `HEAD` revision, and the resolved output directory.

#### Scenario: launch identity line outside a checkout

- **WHEN** the driver starts with the repo directory absent (no `.git`
  reachable),
- **THEN** the trail's first line still contains pid, argv, and the
  resolved output directory, with the revision field literally
  `unknown` (never a crash, never a fabricated value).

### Requirement: Concurrent driver runs SHALL not interleave crumb trails

When `QUANTIBBLER_BREADCRUMBS` is unset, the trail file SHALL be
per-launch: `breadcrumbs-<pid>.log` inside the resolved KD scratch
directory, so two concurrent launches write two files. When the
variable is set, the configured path SHALL be honored exactly (the
single-run container contract; the wrapper's run headers and the
forensics read path are unchanged).

#### Scenario: default path disambiguates concurrent launches

- **WHEN** two driver processes run concurrently with the variable
  unset and a shared KD scratch dir,
- **THEN** each writes its own `breadcrumbs-<pid>.log` and no line
  interleaves.

#### Scenario: configured path honored exactly

- **WHEN** `QUANTIBBLER_BREADCRUMBS` points at a fixed path,
- **THEN** the driver appends only to that path (no pid suffix) and
  its launch identity line still appears first.
