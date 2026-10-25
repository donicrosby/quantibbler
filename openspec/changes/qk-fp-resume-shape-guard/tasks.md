# Tasks: qk-fp-resume-shape-guard

- [x] (planner) Author the ADDED requirement
  `restore_refuses_mis_shaped_state` + `mis_shaped_refusal_leaves_block_untouched`
  under `qk-ptq`, ratifying the F2 shape refusal as a SHALL. No MODIFIED —
  sibling changes carry live deltas under this capability.
- [x] (planner) Record the dated archive-wording correction in design.md,
  superseding the archived sec-eng "raise happens before any write-back" row
  (cited by `apply_layer_states` phase ordering, not line numbers).
- [x] (planner) `npx --yes @fission-ai/openspec validate
  qk-fp-resume-shape-guard --strict` — green (single change id per invocation).
- [ ] (orchestrator) Land the F2 patch (`t_f387d19e-two-phase-apply.patch`,
  sha256 d578e0f2…) with this spec in one commit; the patch's test
  `test_shape_corrupt_archive_never_partially_writes` covers both scenarios
  and should cite the scenario IDs.

## Notes

Doc-only by design: no code change in this change dir. The F2 patch (code +
test) is the implementation; this change is the spec-of-record that covers it.
The archived `2026-10-25-qk-fix-fp-resume-guard` is left untouched.
