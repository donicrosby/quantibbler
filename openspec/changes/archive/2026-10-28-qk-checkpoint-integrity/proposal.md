# Proposal: qk-checkpoint-integrity

## Why

Three LOW findings from the review of `qk-fix-fp-resume-guard` (sec-eng
`t_a76ffbcd` F3/F4/F5; orchestrator disposition 2026-10-25, board ticket
`t_14977580`) sit on ONE surface — the block-checkpoint directory — and share
one shape: the checkpoint format made no claim about its own contents, so the
code around it inferred one and got it wrong three times.

1. **The archive carries no content claim (F3).** `block-*.npz` records no
   digest of what was written, so the code relies on the container's own
   checksums. Measured on this tree: `np.savez` stores entries with
   `ZIP_STORED` + CRC-32, which does stop a byte flip *inside an entry* (as a
   `zipfile.BadZipFile` raised from inside `np.load`) — but a regenerated
   archive recomputes every CRC-32 for whatever content it is handed. Rewrite
   the same keys with a moved scale (or a moved teacher target) and the archive
   is a perfectly valid zip holding values no solve produced, restored as
   trusted state.
2. **Two contracts for one class of junk (F4).** `completed_blocks`
   `int()`-parses `block-*.npz` stems. An unreadable archive printed a notice
   and was skipped; `block-abc.npz` raised `ValueError` out of the scan and
   aborted the whole resume (measured: `ValueError: invalid literal for int()`
   at `checkpoints.py:237`).
3. **The fresh-start branch destroys the corpse (F5).** `reconstruct_model`'s
   `done == []` branch writes its fingerprint straight over whatever a previous
   launch left in the directory — before this launch has produced anything to
   replace it with. That directory is the only record a dead run leaves
   (fingerprint, stage journal, any half-landed archive), and the repo's
   run-launch protocol exists precisely so a corpse stays attributable.

## What Changes

- Every archive this code writes carries `content_sha256`: the sha256 of its own
  logical content (each key's name, dtype, shape and bytes, the reserved
  teacher-target key included), verified in `load_block_checkpoint`. A mismatch
  is a named refusal. An archive written before the digest existed loads
  unverified, with a notice — absent is not mismatched.
- `load_block_checkpoint` refuses a container that no longer parses with a named
  checkpoint verdict, reporting the container's own exception class in the
  message instead of letting it escape three frames deep.
- `completed_blocks` skips an unparseable stem with a notice — the contract it
  already applied to an unreadable archive.
- A fresh start MOVES the directory's prior state into `prior-<UTC stamp>/`
  before writing its own fingerprint.

## Impact

- **Affected specs**: `qk-ptq`, ADDED requirements only. The existing
  `Checkpoint resume SHALL fail closed on empty or incomplete block archives`
  requirement and all three of its scenarios are untouched.
- **Affected code**: `src/quantibbler/ptq/checkpoints.py`, the fresh-start branch
  of `reconstruct_model` in `block_recon.py`, and one new test module
  (`tests/ptq/test_checkpoint_integrity.py`).
- **Not invariant-touching**: no qpk1 format, kernels, or server change. The
  checkpoint directory is a local resume cache, never a shipped artifact, so no
  compatibility impact on any consumer of the compressor's output.
- **Compatibility**: archives written by an earlier revision still load (notice,
  unverified); a directory written by this revision leaves files an earlier
  revision ignores, and the `prior-*/` subdirectory is inert to every reader.

## Non-goals

- No change to what a checkpoint stores (still int8 bits + float64 scales).
- No keyed or signed digest: this detects drift of the bytes at rest, not an
  adversary who can rewrite the archive and its digest together.
- No `--force` gate on the driver. The finding offered "refuse with `--force`"
  as an alternative to preservation; preservation needs no operator decision
  and loses nothing, so the unconditional move is the smaller contract.
- No fingerprint change (`config_fingerprint` and its payload are untouched):
  model-identity binding and the shared driver default are a separate ticket
  (`t_08c2d9e9`). The digest is not folded into the fingerprint — never fold
  provenance into a digest, and never refuse a resume because a file moved.
