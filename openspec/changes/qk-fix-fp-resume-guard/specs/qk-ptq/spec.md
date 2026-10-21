# qk-ptq delta: qk-fix-fp-resume-guard

## ADDED Requirements

### Requirement: Checkpoint resume SHALL fail closed on empty or incomplete block archives

Resume SHALL never advance a block with fewer reconstructed layers than the
block is owed. Guards SHALL be driven from the live model's expected
quantizable names, not from the archive's contents.

#### Scenario: empty_archive_is_never_written

- **WHEN** a block completes with zero reconstructed layers
- **THEN** `save_block_checkpoint` raises and no `block-*.npz` file is
  written for that block

#### Scenario: empty_archive_is_never_counted_done

- **WHEN** `completed_blocks` scans a checkpoint directory
- **THEN** archives with zero keys (or no readable quantized-layer names) are
  excluded from the done set

#### Scenario: restore_asserts_expected_equals_present

- **WHEN** a resumed run applies checkpointed layers to a block
- **THEN** the set of layer names reconstructed from the archive is asserted
  equal to the block's expected quantizable names from the live model, and
  any symmetric difference raises an error naming the block and the missing
  names before the prefix advances
