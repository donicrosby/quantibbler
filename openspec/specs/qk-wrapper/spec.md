# qk-wrapper Specification

## Purpose

The launch-wrapper's contract: what a container fire must leave behind and what
its guard chain must do before anything launches. It covers the witness
(`container.log` captured wrapper-side, exit status recorded in `launch.env`, any
mode failing closed on an absent or empty witness), the monitoring cadence (poll
cadence and named owner announced at launch), the tri-state guard rule (held
refuses, free proceeds, unverifiable refuses with the captured failure), the
requirement that every fail-closed branch be exercised by a hermetic test,
launch lines re-derived at firing time rather than replayed, a single writer for
`launch.env`, and in-band proof that a test's `PATH` interception of real
infrastructure actually took effect. The capability exists so an unattributable
fire corpse or a guard that fell open is a defect the contract names, not an
incident to re-discover.

## Requirements

### Requirement: container_log_witness

The wrapper SHALL capture the container's stdout/stderr to `$OUT/container.log` — wrapper-level `tee`, no in-container changes, no image rebuild — and SHALL record the container exit status in `launch.env` via POSIX `PIPESTATUS`. A fire of ANY mode SHALL fail closed on completion when `container.log` is absent or empty.

#### Scenario: any_mode_death_leaves_testimony

- **WHEN** any fire — lane0 or ptq production — dies
- **THEN** its traceback SHALL exist in `container.log`: a death without a trace is a launch defect, not an acceptable outcome — green-16's batch-2 Arm B death and v2's fifth blind death both left none, and the CPU suite cannot be the witness for device-only faults

#### Scenario: witness_mechanics_verify_without_box_time

- **WHEN** the wrapper is invoked with an unpullable image
- **THEN** the redirect and status-capture mechanics prove themselves in seconds — the pull fails before any container exists — without spending box time or CPU hours on redirect theater
- **AND** the mechanical check records container.log presence and the propagated exit status in the artifact

#### Scenario: refire_requires_witness

- **WHEN** a rehearsal or decisive fire refires after a death
- **THEN** the refire SHALL complete with `container.log` on disk; any subsequent death leaves its traceback named in the artifact, and the first read of that traceback is the suspect from the prior death's boundary analysis — for green-16's class, `release_device_memory`'s grad-enabled `empty_cache()` between batches

### Requirement: fire_monitoring_cadence

Every fire SHALL announce its poll cadence at launch with a named owner, alongside its announced wall time — a fire nobody is scheduled to look at is unmonitored by construction. When the completion channel is known dead at launch, the launcher SHALL arm the poll at launch, not remember it later.

#### Scenario: v2_silence_class_shall_not_repeat

- **WHEN** a fire's wall time exceeds one hour
- **THEN** a poll cadence with a named owner SHALL exist from launch, and the launch announcement names both
- **AND** a fire launching with a dead completion channel SHALL have its first poll scheduled at launch — v2, a 21.8-hour fire with a known-dead channel and no poll order, died at 03:43:42Z and went unread for 21 hours; that class SHALL NOT exist again

### Requirement: guards_shall_be_tri_state

A guard whose verification step fails SHALL refuse, never proceed. Three states: held SHALL refuse; free SHALL proceed; unverifiable SHALL refuse with the captured failure in the message. `2>/dev/null || true` reading as free is the named defect class: the pre-flight name guard fell open exactly this way — an inspect that failed, or a payload on stderr, both read as free and the wrapper exited 0 through to `container: exit=0`.

#### Scenario: unverifiable_refuses_with_captured_stderr

- **WHEN** a guard's verification call returns rc≠0 or the payload lands on stderr
- **THEN** the guard refuses with the captured stderr in the refusal message — both collapse modes produce exit 2, not 0
- **AND** the fix lands red-first against two stub fixtures: one answering rc≠0, one emitting the payload on stderr; both currently yield rc=0, both SHALL yield exit 2 after the fix
- **AND** the stub logs three lines per invocation — argv, emitted stdout, rc — so form-selective dispatch cannot hide inside a pass

#### Scenario: seam_oracle_is_dry_run_rc

- **WHEN** a wrapper gate test runs
- **THEN** the wrapper SHALL execute under `DRY_RUN=1` — DRY_RUN walks every gate in order (digest probe, rocm-tag, name guard) and exits before anything can launch, so the run's rc IS the definitive seam verdict at zero launch risk
- **AND** the stub SHALL be form-selective: the `{{.Id}}` probe answers with the pinned digest via the `LANE0_IMAGE_DIGEST_ALLOW` door (a stub must satisfy the pin to reach the guard), and the name-inspect answers with a held payload
- **AND** one run proves the seam and the guard together: rc=0 means the seam is dead and the wrapper talked to the real daemon; rc=2 means the seam is alive AND the guard refused through the full chain — the 2026-10-24 dead-seam run read as a guard pass precisely because nothing stubby ever answered

#### Scenario: fail_direction_audit_required

- **WHEN** a guard chain is audited
- **THEN** every verification call SHALL be classified by fail direction, and a call that fails toward launch is a defect — the 2026-10-24 audit found the nets' label reads and the watchdog's `docker ps` fail safe, and the pre-flight guard the only fail-open on the launch path

### Requirement: wrapper_failclosed_branches_hermetic_tests

Every fail-closed branch of the wrapper SHALL be exercised by a hermetic test — a stub `docker` answering `inspect` with a fixture payload, the interception proven in-band before the wrapper runs — never by "verified by construction".

#### Scenario: refusal_branch_proven_before_fire

- **WHEN** a lane0 refire or decisive fire is staged
- **THEN** the staging proof includes the hermetic refusal test green — a fire SHALL NOT ride a guard that cannot be demonstrated
- **AND** the branch list SHALL cover: pre-flight held-name refusal, digest pin, rocm-tag gate, GPU gate, witness fail-closed, the nets' label-decline, and the watchdog's kill-path label check
- **AND** the named class is the 2026-10-24 pre-flight miss: claimed verified by construction, a hermetic stub measured it NOT firing — wrapper exited 0 through to `container: exit=0`
- **AND** a guard whose red test is reverted to keep the suite green is a staging blocker, not a silent gap
- **AND** the one legal exception to a red fixture in the lane is `xfail(strict=True)` when the red is a broken fixture harness, not a broken guard: the marker's reason names the harness defect, the docstring names the un-xfail condition, and `strict=True` makes the day the harness gets fixed and the test passes an XPASS failure — the suite itself demands the un-xfail; reverting the test out of the lane is never the exception

### Requirement: launch_lines_derived_at_firing

A staged launch line SHALL be re-derived against the tree at firing time — the wrapper's preflight is the derivation, and its printed launch record is what gets fired, not a remembered line. A knob value recorded under an old semantic is a defect to catch at the gate, not a launch to replay: v3 fast-failed because the staged line carried `EVAL_SEQS=18` in the old batches unit, which the re-expressed sequence-counting knob correctly refused.

#### Scenario: replayed_line_refused_at_gate

- **WHEN** a launch line recorded before a knob re-expression is replayed verbatim
- **THEN** the fire SHALL fail fast at the gate with the traceback in the witness — the correct response is re-deriving the value against HEAD (18 batches → 144 sequences), never bypassing the gate

### Requirement: launch_env_single_writer

`launch.env` SHALL have exactly one writer at a time: concurrent appends are a provenance defect, not a race to tolerate — v2's launch.env ended in a corrupted bare-`?` append line. Writers SHALL serialize (one append path) or write atomically (write-then-replace); a malformed launch.env invalidates the artifact's provenance block.

#### Scenario: corrupted_provenance_invalidates_artifact

- **WHEN** launch.env carries a corrupted line (v2's bare-`?` append)
- **THEN** the artifact's provenance block is refused as evidence until the writer defect is fixed — provenance that can be interleaved by two writers is not provenance

### Requirement: infrastructure_interception_proof

A test that overrides `PATH` to intercept a command which launches real infrastructure SHALL prove the interception took effect in-band — assert the resolved binary is the shim — before invoking the code under test. A silent fall-through to the real binary is not a red test; it is real infrastructure with the test's own timeout as the only alarm.

#### Scenario: shim_proven_before_code_runs

- **WHEN** a test shims a command that launches real infrastructure (docker, container runtimes)
- **THEN** the test SHALL assert the resolved binary is the shim in the same environment handed to the code under test, before that code runs — a standalone probe that passes while the real test falls through is the exact failure the rule catches

### Requirement: launch_ownership_required

The wrapper SHALL refuse to launch a fire that carries no owner stamp —
`QKB_LAUNCH_OWNER` (who is launching) plus `QKB_LAUNCH_CARD` (the owning kanban
card), the label spellings accepted as the same evidence — before it takes the
container name. An attributed launch SHALL stamp `qkb_launch_owner` and
`qkb_launch_card` beside `qkb_launch_id`, mirror both into the container env,
and name the owner in the announce, `launch.env`, and the trail's run header.

#### Scenario: wrapper_refuses_unattributed_launch

- **WHEN** the wrapper runs with none of the accepted owner spellings in its environment
- **THEN** it exits 2 with an error naming `QKB_LAUNCH_OWNER` and `QKB_LAUNCH_CARD`, and resolves no argv
- **AND** this is the 2026-10-25 `qk-pertarget-v4` incident's defect: an ownerless fire took the production container name, and no record said whose it was

#### Scenario: launch_ownership_label_spellings_accepted

- **WHEN** only the label spellings `qkb_launch_owner` / `qkb_launch_card` are present
- **THEN** the launch proceeds and stamps the contract labels — the incident ruling named the container labels, which a container can only receive as env, so those spellings are one piece of evidence under a second name

#### Scenario: attributed_launch_stamps_owner_labels

- **WHEN** the wrapper runs with an owner and an owning card id
- **THEN** the argv carries `--label qkb_launch_owner=…` and `--label qkb_launch_card=…` beside `--label qkb_launch_id=…`, the container env mirrors all three, and the announce and `launch.env` name the owner and the card
- **AND** the breadcrumb run header carries `owner=… card=…` — the owner-line announcement is machine-written at fire time, not remembered
