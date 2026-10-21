# Proposal: qk-ci-referee

## Why

The repo has no CI at all (`.github/workflows/` absent; GitHub API
confirms `total_count: 0` on main). Every green gate so far has been
sandbox-local, and — worse — "green" until today meant `62 passed, 16
skipped`: the entire ptq tier silently skipped because the extras were
never synced. With the extras installed the honest number is 137 passed,
0 skipped. A CI that reproduced the silent-skip shape would certify
nothing; the moon-landing standard needs an independent referee that
executes the full suite and can't be quietly hollowed out. The repo is
PUBLIC (verified), so the workflow is live attack surface from day one.

## What Changes

A single GitHub Actions workflow (new capability `qk-ci`) with six
jobs, all `bash -euo pipefail`, all third-party actions pinned by full
commit SHA, `permissions: contents: read`, triggers on `pull_request`
and `push` to `main` (never `pull_request_target`), no secrets:

1. **lean-python** — proven-recipe interpreter (fresh `uv venv` +
   `pip install -e . pytest`, no extras); asserts torch/transformers/
   tokenizers absence in its own interpreter before parsing anything,
   then asserts the lean tier's documented skip count (skips are
   legitimate here; the count is asserted, not ignored — and no count
   from an unproven interpreter is believed).
2. **full-python** — `uv sync --extra torch --extra hf --extra kernels`;
   pytest with an explicit requirement: **zero skipped** (parsed from the
   pytest summary, not eyeballed). This is the job that would have
   refused every pre-extras "green suite".
3. **rust-kernels** — cargo test + clippy `-D warnings` + fmt `--check`
   (manifest-path form) + `uv run maturin build --manifest-path
   crates/quantibbler-kernels/Cargo.toml`.
4. **openspec** — iterate `openspec/changes/*/` (non-archived), one id
   per invocation of `openspec validate <id> --strict`, fail on any
   invalid change.
5. **driver-smoke** — the torch-tier tiny-`Qwen3Config` driver run,
   from scratch, fully offline (no hub, no network): asserts the
   produced `quantize_report.json` has `status`, strict-parses under a
   `parse_constant`-rejecting parser, `exit_code ==
   verdict_exit_code(status)`, and provenance fields equal the invoked
   args. **Plumbing-only, by measured verdict:** a random-init model's
   next-token distribution is already uniform (measured rank-1 ratio
   1.0000006; zeroed `o_proj` shifts PPL 3 ppm), so this job proves the
   gate's code path executes end-to-end and the report/exit contract
   holds — it makes no detection or abort-coverage claim.
6. **abort-leg** — the weights-cached real-checkpoint leg: restores the
   Qwen3-0.6B checkpoint + wikitext snapshot from the cache (seeded by the
   **self-seeding** seed job, which restore-looks-up first — a hit is a no-op,
   a miss re-fetches — so a cold cache heals without a human in the loop), runs
   coder's recorded abort command fully offline, and asserts abort-before-
   block-3, minutes-not-tens-of-minutes wall time, `status: failed` +
   `probe_abort` fields + history in the report, and nonzero exit. A cache miss
   fails the job — fail-closed, never a silent live download; recovery is the
   next push to main, or a manual dispatch.

## Impact

- **Affected specs**: new capability `qk-ci` (ADDED requirements only,
  delta under `changes/qk-ci-referee/specs/qk-ci/spec.md`).
- **Affected code**: `.github/workflows/ci.yml` only; no product code.
- **Not invariant-touching**: no format, packing, kernels, or server
  change.
- **Operational**: from the first push carrying the workflow, every PR
  and push to main gets an independent full-suite referee; the first
  run is also the first CI this repo has ever had.

## Non-goals

- No CD, no release automation, no bench/codecov legs.
- No self-hosted runners or GPU legs.
- No secrets: not one workflow-provided secret today, so the PR-run
  attack surface stays secret-free by construction.
- No change to the fail-fast or gate changes; this lands after
  `qk-fail-fast-loop` to avoid driver-file stacking (workflow-only
  files, but the smoke job executes the driver the other change is
  editing — sequencing honesty noted in tasks).
