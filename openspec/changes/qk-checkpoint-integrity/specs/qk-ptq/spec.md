# qk-ptq delta: qk-checkpoint-integrity

## ADDED Requirements

### Requirement: Block checkpoint archives SHALL carry and verify a content digest

Every `block-*.npz` this pipeline writes SHALL record a sha256 over its own
logical content, and a load SHALL refuse an archive whose content does not hash
to the recorded digest. The digest key itself SHALL NOT be part of the hashed
content. An archive carrying no digest SHALL load unverified, with a notice: an
absent digest is never a mismatch.

#### Scenario: block_archive_digest_written_at_save

- **WHEN** `save_block_checkpoint` writes an archive
- **THEN** the archive carries the sha256 of every array it stores — key name,
  dtype, shape and bytes, the reserved teacher-target key included — under a
  digest key that derives no layer name and is excluded from the hashed content

#### Scenario: stale_digest_refuses_rewritten_archive

- **WHEN** an archive's content is regenerated with different values — including
  a moved reserved teacher-target key — and the digest is carried over unchanged
- **THEN** `load_block_checkpoint` raises an error naming the archive and both
  digests, instead of rebuilding layer states from it

#### Scenario: unreadable_archive_is_refused_by_name

- **WHEN** an archive is bit-flipped or cut down in place so that its container
  no longer parses
- **THEN** `load_block_checkpoint` raises a named checkpoint refusal carrying
  the container's own exception class in its message, rather than letting that
  class escape to the caller

#### Scenario: legacy_archive_without_digest_loads_unverified

- **WHEN** an archive carries no digest
- **THEN** it loads, and the load prints a notice stating the archive is
  unverified

### Requirement: The checkpoint scan SHALL apply one contract to every name it cannot read

`completed_blocks` SHALL skip, with a printed notice naming the file, any
`block-*.npz` whose name carries no block index — the same notice-and-skip
contract it applies to an unreadable or empty archive. No file present in the
directory may abort the scan.

#### Scenario: malformed_block_stem_is_not_counted_done

- **WHEN** `completed_blocks` scans a directory holding an archive whose file
  name carries no block index
- **THEN** that archive is excluded from the done set with a printed notice
  naming it, and every well-formed archive in the directory is still reported

### Requirement: A fresh start SHALL preserve, never overwrite, prior checkpoint state

When a launch starts fresh in a checkpoint directory that already holds state
from a previous launch, the existing files SHALL be moved into a timestamped
subdirectory of that directory before the new fingerprint is written, leaving
the previous record (fingerprint, stage journal, archives) readable after the
relaunch. A directory with nothing to preserve SHALL NOT gain a snapshot
directory.

#### Scenario: fresh_start_preserves_prior_checkpoint_state

- **WHEN** a launch starts fresh in a checkpoint directory holding a previous
  launch's fingerprint, stage journal and an unfinished archive
- **THEN** those files are moved into a single `prior-<timestamp>` subdirectory
  with their bytes intact, and the directory's live fingerprint is the new
  launch's

#### Scenario: empty_directory_gains_no_snapshot

- **WHEN** a launch starts fresh in an empty checkpoint directory
- **THEN** no snapshot subdirectory is created

#### Scenario: prior_state_snapshots_are_additive

- **WHEN** a second fresh start happens in the same directory, including within
  the same second as the first
- **THEN** it archives the live state into its own `prior-<timestamp>`
  directory and leaves every earlier snapshot in place, un-nested
