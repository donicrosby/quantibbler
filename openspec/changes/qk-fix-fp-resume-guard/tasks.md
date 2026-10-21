# Tasks: qk-fix-fp-resume-guard

- [ ] (coder) Implement per the spec delta: `save_block_checkpoint` raises
  on empty layers; `completed_blocks` excludes zero-key archives;
  `apply_layer_states` drives the guard from the live block's expected
  quantizable names with set-equality assertion and a symmetric-difference
  error message naming block, missing, and unexpected names. No qpk1 or
  format change.
- [ ] (coder) Class-fix sweep: grep the compressor for guards built from
  loaded content (archive keys, `by_rel`, `missed` derivations); convert each
  to expected-vs-present or record the waiver in the change's design.md.
  `qk-divergence-guards` is cited, not modified.
- [ ] (coder) Self-test: (a) an empty archive path raises loudly at save and
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
