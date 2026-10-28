# Tasks: qk-checkpoint-integrity (scaffolded from the orphan patch)

Board ticket: `t_14977580` (coder). Source findings: sec-eng `t_a76ffbcd`
F3/F4/F5, ruled out of `qk-fix-fp-resume-guard` by the orchestrator
2026-10-25. This change dir was created by the `qk-spec-audit` executor
(ruling R1): the work had a spec delta, a design and an orphaned patch, but no
change dir, so nothing in `openspec/changes/` tracked it.

## Scaffold provenance — read this before applying anything

The openspec artifacts here (`proposal.md`, `design.md`,
`specs/qk-ptq/spec.md`) are extracted verbatim from patch **v2**, sha256
`dfdb37fc68dc04d4e0b247e26a76dfddd36f561b7d16ec67706d055228880fa7`
(`/opt/data/tmp/t14977580-final.patch`; a byte-identical copy sits at
`/opt/data/tmp/t_14977580/t14977580-v2.patch`). That patch ALSO carries the code
hunks (`src/quantibbler/ptq/checkpoints.py`, `block_recon.py`,
`tests/ptq/test_checkpoint_integrity.py`, `tests/ptq/test_dual_stream.py`);
**the executor did not apply them.** Two follow-up cards own the code landing
and both carry superseding revisions, all unlanded at scaffold time:

| revision | sha256 | artifact | note |
| --- | --- | --- | --- |
| v2 | `dfdb37fc…` | `/opt/data/tmp/t14977580-final.patch` | the revision the audit card named; provenance caveat outstanding (`t_1b582b7d`) |
| v3 | `ad530f11…` | `/opt/data/tmp/t_14977580/t14977580.patch` | v2 + a markdown-only provenance caveat (`t_1b582b7d`) |
| hardening | `79f4a144…` | `/opt/data/tmp/t_ed0efb97/t-ed0efb97.patch` | supersedes v2 wholesale; closes sec-eng F1/F2/F3 (`t_ed0efb97`) |

Land the code from the newest revision that is review-complete — never from v2
by habit.

## Phase 1 — Scaffold (executor)

- [x] Extract the openspec delta files from the patch and scaffold
  `openspec/changes/qk-checkpoint-integrity/`. Done 2026-10-25 by the
  `qk-spec-audit` executor (`t_039035de`); `openspec validate
  qk-checkpoint-integrity --strict` → valid, RC=0.
- [x] Write the tasks ledger (this file) from the coder's delivery — the patch
  carried the implementation but the ledger needed authorship.
- [ ] Apply the code hunks to `src/` and `tests/`. **Owned by `t_1b582b7d` /
  `t_ed0efb97`**, and the revision to land is the newest review-complete one,
  not v2. Evidence required: `tests/ptq/test_checkpoint_integrity.py` (8
  scenario tests) red on the pre-change tree → green after; the affected
  scoped tier recorded with its interpreter and device.

## Phase 2 — Validate + land

- [ ] `npx --yes @fission-ai/openspec validate qk-checkpoint-integrity --strict`
  — green at scaffold time, re-run after any delta edit.
- [ ] Gate suite: `set -o pipefail; uv run pytest` (quote interpreter, device
  and collected denominator), `cargo test --manifest-path
  crates/quantibbler-kernels/Cargo.toml`, `cargo clippy --manifest-path
  crates/quantibbler-kernels/Cargo.toml --all-targets -- -D warnings`,
  `cargo fmt --manifest-path crates/quantibbler-kernels/Cargo.toml --all --
  --check`, `uv run maturin build --manifest-path
  crates/quantibbler-kernels/Cargo.toml`.
- [ ] Scoped commit (orchestrator). Do not archive until the review rows of
  the landing patch are answered.

## Notes

The eight scenarios in the delta are unchanged from the coder's delivery:
`block_archive_digest_written_at_save`,
`stale_digest_refuses_rewritten_archive`,
`unreadable_archive_is_refused_by_name`,
`legacy_archive_without_digest_loads_unverified`,
`malformed_block_stem_is_not_counted_done`,
`fresh_start_preserves_prior_checkpoint_state`,
`empty_directory_gains_no_snapshot`,
`prior_state_snapshots_are_additive`.
