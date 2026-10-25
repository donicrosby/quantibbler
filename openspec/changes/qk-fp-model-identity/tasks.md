# Tasks: qk-fp-model-identity

Board ticket: `t_08c2d9e9` (coder). Review tickets are created from this change's
rows on the board; this file does not rewrite them.

- [x] (coder) RED-first probe on the landed bytes: two models quantized with the
  same `PTQConfig` share one checkpoint directory. Probe 1: `check_fingerprint`
  passes for the other model and the end-to-end resume dies on an anonymous
  tensor-size error inside `apply_layer_states`. Probe 2: with the solver rigged
  to raise if called, the resume COMPLETES (`[checkpoint] block 1 (lin1):
  restored from checkpoint, skipping solve`, solver calls `0`) — a silent
  chimera. Commands, interpreter and RCs recorded on the ticket.
  (coder 2026-10-25: done — `probe_cross_model.py` and `probe_silent_chimera.py`
  in `/opt/data/tmp/t_08c2d9e9-probes/`, `/opt/venv` torch 2.14.1+cpu. RC=0 both;
  probe 1 printed `RESULT [check_fingerprint]: PASSED -- no model identity in the
  guard` then the anonymous `RuntimeError: The size of tensor a (6) must match the
  size of tensor b (8)`; probe 2 printed `[checkpoint] block 1 (lin1): restored
  from checkpoint, skipping solve`, `reconstruct_block calls during B: 0`, and
  `resumed B.lin1 == fresh B.lin1: False`. The AFTER probe
  (`probe_after.py`, same interpreter) shows both refused by name before the solver
  runs, plus the legacy-record refusal, plus a legitimate same-model resume still
  completing.)
- [x] (coder) Tests precede implementation: `tests/ptq/test_model_identity.py`
  with `test_scn_model_identity_binds_the_resume`,
  `test_scn_model_identity_mismatch_names_the_divergence`,
  `test_scn_legacy_fingerprint_without_model_identity_is_refused` and
  `test_scn_driver_default_checkpoint_dir_is_per_run`. RED observed before the
  implementation (`SOLVER_BACKEND=numpy`, CPU-only host), GREEN after.
  (coder 2026-10-25: RED observed first — `4 failed` in 26.59s, RC=1, three of them
  `TypeError: check_fingerprint() takes 2 positional arguments but 3 were given` and
  one `AttributeError: ... has no attribute '_checkpoint_launcher_dir'`; GREEN after
  — `23 passed in 44.70s` RC=0 on this file plus `test_checkpoints.py`,
  `test_fp_resume_guard.py` and `test_stage_journal.py`, `SOLVER_BACKEND=numpy`,
  `/opt/venv` torch 2.14.1+cpu.)
- [x] (coder) Implement the model-identity component in
  `src/quantibbler/ptq/checkpoints.py`: `model_identity(model)` over the
  quantized-parameter scope (module inventory + per-module `(dout, din)` +
  weight dtype), its digest recorded beside the config digest under
  `model_identity`, and `write_fingerprint` / `check_fingerprint` taking the live
  `model` as a REQUIRED argument. A mismatch refuses before any archive is
  loaded and names every diverged module and what diverged.
- [x] (coder) Thread the live model through the resume path — `block_recon`'s
  `check_fingerprint` / `write_fingerprint` call sites, `checkpoints`'
  `read_stage_log` and the `resume_stage_events` wrapper, and `api.quantize_model`'s
  seed read — and update every in-tree fingerprint call site honestly, including
  the key-set pin in `tests/ptq/test_checkpoints.py` (its expected payload grows
  one key) and the stage-journal lane's `tests/ptq/test_stage_journal.py` (six
  `read_stage_log` callers, broken by the required argument; see design.md,
  "Collateral, declared").
  (coder 2026-10-25: done — the rebase note in design.md explains why the call
  sites are `block_recon` + `api` and not the pre-rebase `reconstruct_model`
  seeding block.)
- [x] (coder) Driver default: `tools/quantize_hf_model.py` checkpoints into a
  fresh per-run directory when `--checkpoint-dir` is not given, names it in the
  run output, and keeps an explicitly passed directory as the resumable path.
  Audit the resume path for other shared-mutable defaults and record the result.
  (coder 2026-10-25: done — `_checkpoint_launcher_dir()` (module level, so the
  scenario test can drive it without `main()`), argparse default `None`, the dir
  printed on the launch whose default created it, and `_scratch_root()` shared with
  the KD-cache default. Audit table in design.md, "Shared-state audit of the resume
  surface": three audited clean, two audited/out-of-scope with the reason named.)
- [x] (bug-hunter) Review on the landed bytes: prove by control-flow reading that
  no resume path can apply another model's archive (write → check → load →
  apply), that the identity predicate is the same one the solver scopes with,
  and that a refusal cannot leave a partially-written prefix in the host. Re-run both
  RED probes and confirm they now refuse by name.
  (bug-hunter 2026-10-25: done, all clauses PASS on the shared tree @ `8c226fa`
  + live edits. Archive-surface enumeration: `fingerprint.json` written only by
  `write_fingerprint` (`checkpoints.py:271`) and read only by `check_fingerprint`
  (`:286`); `block-*.npz` read only by `completed_blocks` (`:334`, scan),
  `load_block_checkpoint` (`:530`) and `load_block_yb` (`:564`), whose only
  production callers are `block_recon.py:2081/2088`, inside the resume branch
  gated by `check_fingerprint` at `block_recon.py:2015`; `read_stage_log` is
  self-guarded (`checkpoints.py:449`) behind `resume_stage_events` (`:424`) from
  `api.quantize_model` (`:106`). Identity predicate = the solver's own
  `_quantizable_names` object (`checkpoints.py:168,177` → `block_recon.py:358`;
  solve-side scoping `:1308/:1389`, apply-side `:2075`); `layer_scope` excluded
  by design, carried by the config-digest leg. Refusal precedes every host
  write: the check runs before the block loop, and `apply_layer_states` asserts
  `expected == present` before any `copy_` (`checkpoints.py:617-625`). Probes
  re-run on the shared bytes (`/opt/venv` torch 2.14.1+cpu,
  `SOLVER_BACKEND=numpy`): `probe_silent_chimera.py` refused by name
  (`recorded but absent from the live model: lin2`); `probe_after.py` all four
  parts correct (named refusal / refused before the rigged solver / legacy
  refused / legitimate same-model resume completes); battery re-demonstrated the
  cross-shape d=6 named refusal direct + end-to-end and the same-shape/
  different-weights residual passing exactly as design.md names it. Instrument
  note: `probe_cross_model.py`'s own 2-arg `check_fingerprint` call now dies on
  the required-arity TypeError (RC=1) — that is the disarmed-guard arming
  firing, not a broken probe; its end-to-end half is unreachable on the fixed
  signature by construction. No blockers, no should-fixes; two nits recorded on
  the review card (probe instrument staleness; design.md's residual paragraph
  does not name bias-presence as a fourth uncovered attribute — silently
  dropped at `checkpoints.py:646`.)
- [x] (sec-eng) Integrity pass: the new refusal fails closed for a legacy
  directory and for a hand-edited `fingerprint.json`; the identity is derived
  from the live model and cannot be satisfied by editing a record; the per-run
  default leaves no shared-state path on the resume surface.
  (sec-eng 2026-10-25: PASS on all three clauses, `/opt/venv` torch 2.14.1+cpu,
  `SOLVER_BACKEND=numpy`, shared tree @ 8c226fa. Probe matrix
  `/opt/data/tmp/t_24807b11-seceng/seceng-probe.out` (RC=0): legacy and every
  malformed/mismatched/missing record shape refused or died before any restore
  (P1, P5a–j); a BOTH-SIDES fabrication — a correctly-computed sha256 over a
  FALSE inventory — still refuses (P3), the only passing edit class is a
  truth-telling one (P4), and the config leg refuses independently (P6);
  refusal preceded `load_block_checkpoint`/`apply_layer_states`/`load_block_yb`
  on both entry paths with sentinels never hit (P7), and the journal read
  cannot bypass the guard (P8). Shared-state audit table verified row by row
  against source: no row wrong, no row missing. Verdict comment on t_08c2d9e9.)
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate
  qk-fp-model-identity --strict`, then commit spec delta + code in ONE commit
  (leaf does not commit). Archive only when the change genuinely completes.
- [x] (coder) GATE SUITE (last task): run and record, with the interpreter and
  device stated — the ptq tier under `SOLVER_BACKEND=numpy` on this CPU-only host
  (`/opt/venv`, torch 2.14.1+cpu) for the touched files, `cargo test` +
  `cargo clippy --all-targets -- -D warnings` + `cargo fmt --check` in
  `crates/quantibbler-kernels/`, `maturin build` in `crates/quantibbler-kernels/`,
  and `pre-commit run --files <changed paths>`.
  (coder 2026-10-25: done. Interpreter `/opt/venv` torch 2.14.1+cpu, no
  accelerator, `SOLVER_BACKEND=numpy`, load 61. Full `tests/ptq` tier run ONE
  FILE PER SUBPROCESS (verdict per file, log
  `/opt/data/tmp/t_08c2d9e9-delivery/gate/tier.log`, evidence
  `evidence/gate-suite.txt`): **42/42 files, 255 passed / 1 failed / 3 skipped /
  4 deselected**. The single failure is
  `test_probe_knobs_reject_non_positive_values_at_parse_time[--probe-interval--2]`,
  where the driver child was SIGTERM'd (`returncode -15`, empty stdout/stderr) —
  the parse-time half of the driver, before any pipeline code, and the sibling
  `value=0` case passed in the same run; re-run in isolation on the same bytes
  `3 passed, 5 deselected` RC=0, i.e. box flake, not the diff.
  `test_driver_stage_journal.py` exits 5 because its one test is
  integration-marked and deselected by the default filter. `cargo test` 24 passed
  RC=0, `cargo clippy --all-targets -- -D warnings` RC=0, `cargo fmt --check`
  RC=0, `maturin build` in `crates/quantibbler-kernels/` RC=0 (wheel built) —
  note `maturin build`/`uv run maturin build` FROM THE REPO ROOT refuses with
  `Failed to parse Cargo.toml … missing field 'package'` (the root `Cargo.toml` is
  a virtual workspace), a pre-existing condition this change does not touch;
  `pre-commit run --files <13 changed paths>` all hooks Passed RC=0. Read-only
  whole-tree equivalent, because `pre-commit run --all-files`'s ruff-format hook
  would rewrite co-tenant bytes in this shared checkout: `ruff check .` passes;
  `ruff format --check .` reports 2 unformatted files, both foreign
  (`src/quantibbler/ptq/progress.py`, `tests/test_prose_discipline.py`).
  Live-tree caveat, stated because this checkout is shared: a sibling lane
  (stage-journal) edited `checkpoints.py` (`json_safe` / `allow_nan=False` in
  `append_stage_event`) at 09:55:13 and `tests/ptq/test_stage_journal.py` at
  10:05:07 — the first mid-tier. Both are additive to this change's hunks (3 hunk
  headers, none inside the model-identity code), so the scoped 9 files were
  RE-RUN after the edit on the current bytes: 9/9 files `RC=0`
  (`gate/scoped-rerun.log`). The 42-file tier is a mixed-bytes measurement; the
  scoped re-run is the one that matches the tree as handed off.)
