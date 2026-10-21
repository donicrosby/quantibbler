# Tasks: qk-fix-ste-sign-adaptation

- [ ] (coder) Implement per the spec delta: per-step flip-count
  instrumentation in `_tune_latent_ste` (both factors, per linear), plus
  the mechanism ladder per design.md — (1) diagnose and fix why latents
  never cross zero within [NQ] App. C parameter space (lr, T_post, init
  margins), attempting the paper's own STE mechanism first per the user
  directive; (2)-(4) fallbacks per design.md, each recorded in the run
  record with deviation status. The block-local objective uses the shared
  corpus-pinned `(x_b, y_b)` capture that feeds TuneFP ([NQ] Algorithm 1
  lines 9–10, 19), pristine-pinned, memory-only, provenance (source
  fingerprint + capture index) recorded. Binaries stay ±1 by construction;
  no qpk1 format change.
- [ ] (coder) Self-test: a toy block engineered with latents near zero must
  show ≥1 sign crossing under the diagnosed paper mechanism; a toy block
  with far-from-zero latents must either cross after the adopted mechanism
  or emit the scale-only declaration — no silent zero-flip completion.
  Reproducibility test: same seed, same sweep → identical flip sequence.
- [ ] (coder) Rehearse the stage-trace probe against the fixed bytes: post
  per-linear flip counts for the 7 scoped linears and the accepted-flip
  deltas; record the new probe sha and the adopted alternative per linear.
- [ ] (bug-hunter) Review on the landed bytes: instrumentation counts both
  factors and cannot be bypassed; the acceptance test is block-local only;
  the STE objective's `(x_b, y_b)` is the same captured teacher as TuneFP
  (pristine-pinned, provenance recorded), never recomputed from live state;
  the re-scoping marker is greppable in code, spec, and run record; zero-flip
  completion without a declaration is impossible (prove by reading the
  control flow).
- [ ] (sec-eng) Integrity pass: the flip pass adds no new persistence or
  network surface; deterministic order removes nondeterminism from the
  artifact; accepted flips cannot violate the round-trip byte-exact contract
  (bits are exported from the same tensors the pass wrote).
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fix-ste-sign-adaptation --strict`; commit (specs + code in one commit);
  do not archive until the fixed stage's flip counts land in a real run on
  record.
