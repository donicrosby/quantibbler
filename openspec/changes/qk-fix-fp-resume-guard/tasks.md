# Tasks: qk-fix-fp-resume-guard

- [x] (coder) Implement per the spec delta: `save_block_checkpoint` raises
  on empty layers; `completed_blocks` excludes zero-key archives;
  `apply_layer_states` drives the guard from the live block's expected
  quantizable names with set-equality assertion and a symmetric-difference
  error message naming block, missing, and unexpected names. No qpk1 or
  format change.
- [x] (coder) Class-fix sweep: grep the compressor for guards built from
  loaded content (archive keys, `by_rel`, `missed` derivations); convert each
  to expected-vs-present or record the waiver in the change's design.md.
  `qk-divergence-guards` is cited, not modified.
- [x] (coder) Self-test: (a) an empty archive path raises loudly at save and
  at restore; (b) a truncated archive (one missing linear) raises naming the
  missing name; (c) a good archive restores bit-exactly as before; (d) the
  rung-1 archive format still loads (backward compatibility).
- [ ] (bug-hunter) Review on the landed bytes: prove by control-flow reading
  that an empty archive cannot pass resume silently on any
  path (write → count → load → apply); confirm the expected-names predicate
  is the same one the solver scopes with; re-run the rung-1 corpse and fp
  archive checks.
- [ ] (sec-eng) Integrity pass: the new raise cannot corrupt in-flight state
  (raise happens before any write-back); fail-closed semantics hold for
  hand-copied archives; the class-fix sweep report is complete or the
  waivers are explicit.
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-fp-resume-guard --strict`; commit (specs + code in one commit); do
  not archive until a resumed-run test against a deliberately truncated
  checkpoint directory fails loudly on record.

## Implementation notes (coder, 2026-10-21)

Landed on `src/quantibbler/ptq/checkpoints.py` and `block_recon.py`.
Evidence: `tests/ptq/test_fp_resume_guard.py` (6 tests, including a real
end-to-end resume against a hand-copied 22-byte empty archive).

**Class-fix sweep (sec-eng's ruling), complete:**

| site | guard set derived from | disposition |
| --- | --- | --- |
| `completed_blocks` | archive keys | CONVERTED — zero-key archives (and unreadable ones) are excluded from the done set, with a printed notice; the guard that matters is the restore-side one below. |
| `apply_layer_states` | `by_rel` built from archive keys | CONVERTED — `expected_names` (the live block's scope-filtered quantizable names, the SAME predicate `reconstruct_block` scopes with) is asserted set-equal to the archive's present set before any write-back; the symmetric difference names the block, the missing and the unexpected names. |
| `reconstruct_model` resume path | archive contents | CONVERTED — passes `expected_names` for the block, and only writes a checkpoint when the block is owed layers at all (`expected_names` non-empty). |
| `load_block_checkpoint` | archive keys | WAIVED — a loader, not a guard: it materializes whatever the archive holds and makes no claim about completeness. The claim is made where the live model is available. |
| `save_block_checkpoint` | caller's layer list | CONVERTED — raises on an empty list; an empty archive can no longer be written at all. |
| `api.quantize_model` (`missing` / out-of-scope) | expected quantizable names | WAIVED as already conforming — both guards are driven from `quant_names` (the live model), never from what reconstruction produced. |
| `block_recon.check_layer_magnitude` | measured ratio vs the fp target | WAIVED — a numeric gate on a measured quantity, not a membership guard. |
| `qk_report.probe_*` | measured ratios | WAIVED — same class; `qk-divergence-guards` owns those and is cited, not modified. |

**Crystallization:** `_archive_key()` normalizes BOTH sides of the guard
(archive names and expected names) into the checkpoint key space, so a flat
block (the block IS the Linear) and a nested one cannot disagree about
spelling; the pre-existing `missed = by_rel - applied` check is kept behind
it as the module-resolution backstop.
