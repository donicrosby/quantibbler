## ADDED Requirements

### Requirement: A resumed run SHALL declare its restart in the report as a record, never a gate

The real-model driver SHALL emit a top-level `resume` record in `quantize_report.json` naming whether the run resumed, which blocks were restored from a prior launch, the checkpoint directory, the container image and digest, and the fingerprint that admitted the restore. The record SHALL be informational: it SHALL NOT enter the verdict, the exit code, or any gate, and SHALL NOT be folded into the checkpoint fingerprint.

#### Scenario: resume_record_on_resumed_run

- **WHEN** a launch resumes — its `--checkpoint-dir` holds block archives a prior launch wrote, admitted by `check_fingerprint`
- **THEN** `quantize_report.json` carries a top-level `resume` record with `resumed: true`, `restored_blocks` equal to the completed blocks present at launch (the `completed_blocks` scan), `checkpoint_dir` naming the directory, and `image`/`image_digest` read from the launch environment

#### Scenario: resume_record_fresh_run_consistent_shape

- **WHEN** a run starts fresh (no completed block archives at launch)
- **THEN** the `resume` key is still present on the report, with `resumed: false`, an empty `restored_blocks`, `checkpoint_dir: null`, `image`/`image_digest` read from the launch environment (or `"unknown"` when unset), and `admitted_by_fingerprint: null`, so the completion, abort, and stage-failure key sets stay identical and a strict consumer never KeyErrors on either shape

#### Scenario: resume_true_only_when_admitted

- **WHEN** a fingerprint check refuses the restore (the digest or model identity diverges)
- **THEN** the refusal propagates a `RuntimeError` before any report is written, so no report carries `resumed: true` from a refused resume — `resumed: true` SHALL appear only on runs whose `check_fingerprint` admitted the restored blocks
- **AND** a future change that catches the refusal downstream (post-report-write) SHALL flip `resumed: true` to an attempt-based record with an explicit `admitted: false` field, never silently land a false restoration claim

#### Scenario: resume_record_is_a_record_not_a_gate

- **WHEN** any run carries a resume record
- **THEN** the `ok`/`degraded`/`failed` verdict, the process exit code, the band edges, and every fail-fast, probe, and stage gate behave byte-identically to an otherwise-equivalent non-resumed run — the record changes no decision and is never read by a gate

#### Scenario: resume_marker_never_folds_the_fingerprint

- **WHEN** the resume record is emitted
- **THEN** no resume marker enters `PTQConfig`, the `config_fingerprint` payload, or the `_FINGERPRINT_PROVENANCE` record, so a resumed run computes the same config digest as the pre-resume blocks and admits itself rather than refusing its own checkpoint

#### Scenario: resume_identity_is_evidence_not_an_assertion

- **WHEN** the resume record names the restored blocks' provenance
- **THEN** it records the restored checkpoint's `fingerprint` digest, its `model_identity` digest, and its provenance members verbatim — the `_FINGERPRINT_PROVENANCE` tuple (`solver_backend`, `compute_dtype`, `solver`, `rank`, `factor_mode`, `admm_reg`, `rho_start`, `rho_end`) plus the separately recorded `solver_device`
- **AND** it does not emit a computed `identical_bytes` boolean — cross-reboot source-byte identity is a host-side fact (a git diff across the two tips) recorded in the launch trail, which the in-container driver without `git` cannot compute

#### Scenario: resume_record_on_failure_paths

- **WHEN** a resumed run aborts via the prefix probe or fails a fail-closed stage gate
- **THEN** the `resume` record appears on the abort and stage-failure report shapes too, with `resumed` and `restored_blocks` populated from launch state, so the declaration survives a run that died before completion
