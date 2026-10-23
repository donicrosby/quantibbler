# qk-ptq delta: qk-nanoquant-alignment-lane2

## ADDED Requirements

### Requirement: Block reconstruction SHALL advance dual activation streams with an fp-stream teacher capture

Block reconstruction SHALL advance TWO streams: an fp stream through pristine blocks that produces the teacher target `Y_b = B^FP_b(X_fp)`, and a compressed stream through quantized blocks that produces the student input `X_b` ([NQ] Algorithm 1 :730); the student sees `X_b`. Feeding the teacher the fp-stream activation deviates from :735's literal `Y_b ← B^FP_b(X_b)`; design.md records the deviation and its rationale ([NQ] :525, :577).

#### Scenario: dual_stream_advances_in_lockstep

- **WHEN** block reconstruction advances from block b to block b+1
- **THEN** the fp stream advances through the pristine frozen-reference block and the compressed stream advances through the quantized block, in lockstep
- **AND** the student consumes the compressed stream: student sees `X_b`

#### Scenario: teacher_target_reads_fp_stream

- **WHEN** the block-b teacher capture runs
- **THEN** `Y_b = B^FP_b(X_fp)` — the pristine block applied to the fp-stream activation
- **AND** `X_b` is `M̂_<b(X_cal)` per [NQ] Algorithm 1 line 9 (:730), and `Y_b` is not a function of the compressed stream

#### Scenario: stream_separation_fixture

- **WHEN** a fixture corrupts the compressed stream before the block-b capture while the fp stream is intact
- **THEN** fixtures prove stream separation: corrupt the compressed stream, assert `Y_b` unchanged
- **AND** the assertion holds for every block after the first, through both direct `reconstruct_block` calls and the `reconstruct_model` driver loop

#### Scenario: target1_tunefp_vacuity_gone_by_construction

- **WHEN** TuneFP fires for a block's first target linear and at least one prefix block is quantized
- **THEN** the target-1 TuneFP objective is nonzero by construction — the streams have diverged — so the single-stream target-1 vacuity disappears
- **AND** at block 1 the prefix is the identity and the streams coincide: the block-1-target-1 skip remains, now documented as the single precise identity-prefix case (design.md)

#### Scenario: suite_green

- **WHEN** the lane-2 change runs the full gate suite
- **THEN** the suite is green: `pytest tests/ptq/ tests/ci/ tests/qk_report/`, `cargo test`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check`, and `maturin build` all pass

### Requirement: The tune_fp_vacuous_teacher guard SHALL remain a plumbing detector with a re-pinned rationale

The guard SHALL stay a detector, behavior unchanged: zero first-step gradient with zero initial residual still means broken plumbing (stream cross-talk, detached student, capture reading live or compressed state), recorded as `tune_fp_vacuous_teacher` and fail-closed under `paper_faithful`. The rationale SHALL be rewritten: under dual-stream every fired TuneFP call has a nonzero objective by construction, so a zero event is a construction defect, never nothing-to-compensate.

#### Scenario: guard_zero_grad_zero_residual_means_broken_plumbing

- **WHEN** a fired TuneFP call records a zero first-step gradient and a zero initial residual
- **THEN** the `tune_fp_vacuous_teacher` event fires carrying both norms, because zero-grad + zero-residual still means broken plumbing
- **AND** the event carries the stream provenance of the capture that fed the call

#### Scenario: guard_rationale_rewritten

- **WHEN** the guard's documentation is read (docstrings, the `paper_faithful` config comment, design.md)
- **THEN** the rationale names dual-stream separation as the reason a zero objective is a defect, and no text claims target-1 vacuity is inherent to the placement
- **AND** the rationale records that a genuinely zero prefix divergence still trips the guard — fail-closed, never silent

#### Scenario: paper_faithful_fails_closed_on_vacuous_event

- **WHEN** a run claiming paper fidelity records any `tune_fp_vacuous_teacher` event
- **THEN** the claim is invalidated exactly as before: the run fails or records `paper_faithful: false`, never silently continues

### Requirement: The paper_faithful verdict SHALL be re-pinned to fp-stream teacher provenance

The computed `paper_faithful` verdict SHALL be re-pinned to dual-stream capture semantics: fp-stream teacher provenance becomes a label requirement, so a capture recorded with compressed-stream provenance (the legacy single-stream capture) or with missing provenance is a mechanism deviation that denies the label. Every existing verdict condition — the flag AND no runtime deviations AND no mechanism deviations AND required stages executed — is unchanged.

#### Scenario: label_requires_fp_stream_teacher_provenance

- **WHEN** the run record's teacher captures carry stream provenance
- **THEN** every capture names its stream, and a paper-faithful label requires all captures to be fp-stream: a compressed-stream capture or an unprovenanced capture sets the label false
- **AND** the recorded deviation list names the offending capture

#### Scenario: dual_stream_deviation_marker_is_documented_not_label_flipping

- **WHEN** the record carries the dual-stream deviation marker (fp-stream teacher input versus [NQ] :735's literal compressed-prefix feed)
- **THEN** the marker is documentation recorded in design.md and the run record, and the re-pinned verdict treats fp-stream provenance as the faithful form, so the marker does not flip the label

### Requirement: Recorder events SHALL carry stream provenance

Capture, `tune_fp`, `ste`, and `tune_fp_vacuous_teacher` recorder events SHALL record which stream produced each tensor they reference — the teacher target's stream (fp-stream) and the student input's stream (compressed-stream) — alongside the existing teacher fingerprint and capture index, so provenance is provable from the artifact alone.

#### Scenario: recorder_events_gain_stream_provenance

- **WHEN** any reconstruction recorder event references `x_b` or `y_b`
- **THEN** recorder events gain stream provenance: each event names the stream that produced each referenced tensor
- **AND** the capture event keeps the teacher fingerprint and capture index fields, so the artifact proves both the teacher source and the stream separation

### Requirement: Prefix-probe and checkpoint fingerprints SHALL be unchanged in shape

The dual-stream capture SHALL introduce no new solver state: the progress probe keeps its signature `probe(model, depth, name)` and per-block call site, and the checkpoint fingerprint keeps its shape — sha256 over `asdict(PTQConfig)` plus code identity with `progress_probe` the only exclusion — with no new joined config field.

#### Scenario: prefix_probe_and_checkpoint_fingerprints_unchanged_in_shape

- **WHEN** a dual-stream run exercises the prefix probe and checkpoint resume paths
- **THEN** prefix-probe and checkpoint fingerprints are unchanged in shape: the probe still fires as `probe(model, depth, name)` after each block write-back, and the fingerprint payload keys and the joined PTQConfig field set are exactly as before the lane
- **AND** resume semantics are unchanged: a mismatched config or code state is refused, and the lane boundary changes the code identity, so a pre-lane checkpoint is refused resume exactly like any other stale-state case
