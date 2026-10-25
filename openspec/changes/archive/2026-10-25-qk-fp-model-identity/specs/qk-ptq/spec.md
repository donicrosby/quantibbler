# qk-ptq delta: qk-fp-model-identity

## ADDED Requirements

### Requirement: Checkpoint resume SHALL bind the model identity

The resume fingerprint SHALL bind the identity of the model being quantized: the
quantized-parameter scope's module inventory, each module's `(dout, din)` shape, and
its weight dtype, all derived from the LIVE model at fingerprint-write time. A resume
SHALL refuse a diverged identity, naming the diverged component(s), and SHALL refuse a
directory with no recorded identity. A repository id or path SHALL NOT be the identity.

#### Scenario: model_identity_binds_the_resume

- **WHEN** a checkpoint directory written while quantizing model A is resumed by a run
  quantizing model B under an identical `PTQConfig`
- **THEN** `check_fingerprint` raises before any block archive is applied, and no
  block of B is restored from A's archives

#### Scenario: model_identity_mismatch_names_the_divergence

- **WHEN** the live model's quantized-parameter scope differs from the recorded
  identity in module membership, in a per-module `(dout, din)` shape, or in a weight
  dtype
- **THEN** the refusal names the diverged module(s) and what diverged about them, so
  the mismatch is diagnosable from the error alone

#### Scenario: legacy_fingerprint_without_model_identity_is_refused

- **WHEN** a checkpoint directory's `fingerprint.json` carries no model-identity
  component (it was written before this binding existed)
- **THEN** the resume refuses and says the directory predates the model-identity
  component, so its provenance with respect to the live model cannot be verified

#### Scenario: driver_default_checkpoint_dir_is_per_run

- **WHEN** `tools/quantize_hf_model.py` runs without an explicit `--checkpoint-dir`
- **THEN** it checkpoints into a FRESH per-run directory rather than a shared stable
  path, so two defaulted launches never share block archives; resumption is opt-in by
  naming the directory, and the chosen directory is named in the launch's output
