# Design: qk-checkpoint-integrity

> **Scaffold note (2026-10-25, `qk-spec-audit` executor, ruling R1).** This dir
> was scaffolded from the orphan patch (`t14977580.patch` v2, sha256
> `dfdb37fc68dc04d4e0b247e26a76dfddd36f561b7d16ec67706d055228880fa7`). The
> patch's CODE hunks are NOT applied to the working tree — two follow-up cards
> (`t_1b582b7d`, `t_ed0efb97`) own that landing with superseding revisions.
> Every figure under "Evidence" below is the **authoring run's** measurement,
> not a measurement of the current tree.

## The class: the artifact must make a claim about itself

All three findings are the same mistake in different places. The checkpoint
directory is the *only* state a resumed run trusts without re-deriving it, and
it was the only artifact in the pipeline that asserted nothing about its own
contents. Everything downstream then had to infer:

- "is this archive the bytes we wrote?" — inferred from the zip container's own
  checksums (F3);
- "is this file name a block index?" — inferred by calling `int()` on it (F4);
- "may I write here?" — inferred from `mkdir(exist_ok=True)` (F5).

Each inference fails quietly in exactly one direction: the container cannot see
a regenerated archive, `int()` fails loudly (crash instead of skip), and the
write succeeds while destroying evidence.

## Decisions

**The digest lives INSIDE the archive, over the logical content.** Keys are
iterated in sorted order and each contributes its name, dtype, shape and
C-order bytes; the digest key itself is excluded. Consequences:

- It covers every stored array including the reserved `y_b` teacher target — a
  digest over layer keys alone would leave that channel open.
- It is container-independent: regenerating the zip (which recomputes every
  CRC-32) cannot produce a matching digest for changed values, which is exactly
  the silent channel the finding names.
- It buys no keyed security: an adversary who can rewrite the archive can
  rewrite the digest beside it. Recorded, not claimed — this is integrity
  against drift at rest, not authentication.

**Absent digest = legacy, load with a notice; present-but-mismatched = refuse.**
The alternative (refuse on absent) bricks every checkpoint directory written by
an earlier revision — a full solve each, ~91 minutes on a real fire — which is
the tripwire the module docstring rules out ("a checkpoint that bricks because a
file moved or a tree was rebased is a tripwire, not provenance"). This is the
same ruling shape as the `y_b` field's own cross-change resume: a checkpoint
predating the field loads, and the reader is told what it is reading.

**An unreadable container is refused by NAME at the load site.** Measured
classes on this tree: a byte flip inside an entry raises
`zipfile.BadZipFile` (CRC-32), a cut-down archive raises `BadZipFile` ("not a
zip file") or a bare `ValueError` (bogus entry header). Letting those escape
reads as a numpy fault rather than a checkpoint verdict, and they are the same
failure class as a digest mismatch — the bytes on disk are not the bytes
written. The refusal is one class at the load site; the container's own
exception class is reported in the message so nothing is hidden.

**One contract per site, not per file: notice-and-skip in the SCAN.** The scan
decides membership, so junk always means "not done" (re-solve, never restore),
and it says so. That is now true for an unreadable archive, an empty/keyless
archive, and a name carrying no index. The restore site keeps the opposite
contract — refuse by name — because there the run is committed to a specific
block's state.

**Preservation, not a `--force` flag.** `preserve_prior_state` moves everything
(live state only; snapshots already taken stay put) into
`prior-<UTC stamp>/`, disambiguating the same-second case with a `-N` suffix.
A flag would need a driver surface, an operator decision, and a way to say no;
the move costs nothing, needs no decision, and is the smaller contract. The
fresh-start branch's own fingerprint/journal writes then land in an empty
directory, which is what they always assumed.

**Ordering at the call site: preserve BEFORE the new fingerprint.** Preserving
after would archive the file just written. The sibling journal reset that
already sat after `write_fingerprint` becomes a no-op once the journal has been
moved: its purpose (a later resume of this directory must not inherit another
config's events) still holds, since the top-level journal starts empty.
Consequence worth naming: if this launch already appended events before the
block loop (Phase-1 statistics does), those early lines ride into the snapshot
beside the corpse's own. That is what an auditor wants — the snapshot is a
picture of the directory at the moment of the fresh start — and the top-level
journal then holds only this launch's post-fingerprint events.

**Verification lives at `load_block_checkpoint`, not at `load_block_yb`.**
Both readers open the same untrusted archive, but the resume path reaches
`load_block_yb` only after the verified load of the same block, so a second
verification there would read the archive twice to close a channel already
closed. Removing that order dependency later means moving the check, not
duplicating it.

**No fingerprint change.** `config_fingerprint`, its payload, and
`_FINGERPRINT_PROVENANCE` are untouched: the digest is per-archive content
integrity, a different question from "was this directory produced by this
config?". Never fold one into the other.

## Discarded alternatives

- **A hash sidecar (`block-N.npz.sha256`) over the archive's file bytes.**
  Discarded: it is a second file that must be kept atomic with the archive,
  and it turns every legitimate whole-file rewrite into a spurious refusal —
  the in-tree legacy-simulation path (`test_dual_stream` rewrites archives with
  `np.savez` after stripping `y_b`, then resumes from them) would refuse a
  valid resume because a stale sidecar survived, and a hand-copied archive
  would too. The in-archive digest degrades to the honest "no digest: legacy,
  unverified" there.
- **Refusing an archive with no digest.** Discarded: see Decisions —
  it bricks every pre-existing checkpoint directory.
- **Size-threshold heuristics on the archive.** Re-discarded here for the same
  reason as in `qk-fix-fp-resume-guard`: the predicate that matters is
  "content unchanged", not "file big enough".
- **Refusing to start fresh when prior state exists (the `--force` variant of
  F5).** Discarded: it turns an unattended relaunch into a manual chore, and
  the repo's rule is that a control needing a human every time does not ship.
- **Extending the digest to cover the block index / file name.** Discarded:
  the name is already checked twice (the scan's parse contract and
  `apply_layer_states`' expected-vs-present guard), and a name is not content.

## Open questions

- None blocking. Recorded for the reader: the digest gives no protection against
  an archive swapped in from another block index (`cp block-3.npz block-5.npz`)
  — the name and the content are both internally consistent. Catching that
  needs the block index inside the archive's content, which is a format change
  with its own owner; it is not in this delta. (Owner: planner, if a fire ever
  shows the shape.)

## Implementation crystallization

### 2026-10-25 (landing)

- **The digest is the content, and the container is not the contract.** The
  measurement that shaped this delta: `np.savez` writes `ZIP_STORED` entries
  with CRC-32, so an in-place byte flip is already caught — but as an unnamed
  `zipfile.BadZipFile` out of `np.load`, and only because of a zip detail. A
  regenerated archive is invisible to it. Both facts are pinned by tests
  (`unreadable_archive_is_refused_by_name`, `stale_digest_refuses_rewritten_archive`)
  rather than asserted in prose.
- **The scan's parse contract and the restore's guard are complementary, not
  duplicated.** The scan decides membership from the archive's own keys only
  (a scan has no live model); the restore asserts expected-vs-present from the
  live block (`qk-fix-fp-resume-guard`). The new stem parse is a membership
  decision and therefore belongs with the first.
- **The fresh-start branch is a cross-lane hotspot.** A sibling lane's
  in-flight journal work adds `reset_stage_log` to the same branch. This change
  inserts its call before `write_fingerprint` and deliberately does NOT delete
  the sibling's line (removing it would leave `reset_stage_log` with zero
  readers, which is that lane's edit to make, not this ticket's). Flagged on
  the board as a hotspot.

## Evidence (this landing)

Interpreter: `/opt/venv/bin/python` — `torch 2.14.1+cpu`, `torch.version.hip
None`, `torch.cuda.is_available() False` (CPU-only box), tree
`PYTHONPATH=<worktree>/src` with `quantibbler.__file__` asserted into the
worktree. Tier: `SOLVER_BACKEND=numpy PYTHONPATH=... python -m pytest tests/ptq`.

- RED (pre-change tree): `tests/ptq/test_checkpoint_integrity.py` — 8 failed,
  RC=1. Failure reasons per item: `AttributeError: ... has no attribute
  '_DIGEST_KEY'` (digest not written); `zipfile.BadZipFile` escaping the load
  (unnamed refusal); `AssertionError: assert 'no content digest' in ''`
  (no legacy notice); `ValueError: invalid literal for int() with base 10:
  'abc'` at `checkpoints.py:237` (scan crash); `AssertionError: no single
  forensic snapshot of the corpse: ['block-0.npz', 'block-1.npz',
  'block-3.npz', 'fingerprint.json']` (clobber, measured).
- GREEN: `tests/ptq/test_checkpoint_integrity.py` 8 passed, RC=0.
- Regression surface: `test_fp_resume_guard.py`, `test_checkpoints.py`,
  `test_stage_journal.py`, `test_driver_stage_journal.py` — 19 passed /
  1 deselected (identical to the pre-change baseline of the same four files).
- Scoped tier on the FINAL bytes (re-measured for this handoff): the eight-file
  scoped set (`test_checkpoint_integrity`, `test_checkpoints`, `test_dual_stream`,
  `test_fp_resume_guard`, `test_stage_journal`, `test_driver_stage_journal`,
  `test_block_recon`, `test_api`) — **46 passed / 1 deselected in 6.76 s, RC=0**.
  An earlier revision of this bullet quoted a full-directory run taken BEFORE the
  fixture fix recorded below (**259 passed / 1 failed / 3 skipped / 4 deselected,
  1206.61 s**) and attributed its `test_dual_stream` failure to the foreign
  CPU-host class. That figure measures the pre-fix bytes, not the final ones, and
  the attribution is superseded by the finding below.
- Hygiene: `pre-commit run --files <the 8 changed paths>` PC_RC=0 (re-run for
  this handoff). On the changed paths `ruff check` / `ruff format --check` pass
  ("All checks passed!", "4 files already formatted"). A whole-tree
  `ruff check .` is RED on three SIBLING files this delta does not touch —
  `src/quantibbler/ptq/progress.py`, `tests/test_prose_discipline.py`,
  `tools/gen_attention_gate_record.py` — all unformatted/erroring from
  co-tenant in-flight edits in the shared checkout; attributed, not adopted.
- **The full-directory tier earned its keep**: it found a real break of mine that
  no scoped set would have shown. `test_dual_stream.py`'s legacy simulation
  rewrites an archive with `np.savez` while carrying my digest over, so the
  loader refused it — correctly, by its own contract, but the fixture was
  modelling a state that can no longer exist (a digest-bearing archive whose
  content changed). Because the format changed, the simulation had to change with
  it: it now drops `_DIGEST_KEY` with `y_b`. That is the honest reading of the
  failure — the loader behaved as specified and the FIXTURE was the stale party
  — and it is exactly the class of consumer a new format claim must be
  re-measured against.

Tree note: the shared checkout moved twice while this change was in flight (a
sibling landed `qk-fp-model-identity`, editing `checkpoints.py` and the same
`reconstruct_model` fresh-start branch). The patch was rebased onto the shared
tree's live state and re-verified there; the tier numbers above are from a
worktree whose base IS that state (`base snapshot 3`), with the patch applied.
