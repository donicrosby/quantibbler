# Tasks: qk-nq-recreation-bar

- [x] (coder) Tune/eval window split, RED first: tests for `split_mode_preserves_eval_geometry` and `default_single_geometry_unchanged`; observe RED (flag absent). The measured interim geometry — option (a) of the design's constraint section; this is what converts the seq-2048 protocol from OOM to a number.
      — **RED observed** on the new `tests/ptq/test_driver_tune_seq_len.py`, all
      three tests failing for their intended reasons (not one shared accident):
      `--tune-seq-len` absent from `--help`; a real driver run completing and
      then `KeyError: 'tune_seq_len'` on its report (so the fixture path is
      proven live, not assumed); and the split invocation exit 2 with
      `unrecognized arguments: --tune-seq-len 16`. `3 failed in 12.35s`.
- [x] (coder) GREEN: `--tune-seq-len` in the driver with pipeline slicing; report records both lengths; wrapper `TUNE_SEQ_LEN` env knob passthrough; lean/full suites green.
      — **Landed.** Driver: `--tune-seq-len` (`_positive_int`, default `None`)
      resolving to `--seq-len` when unset; `calib` SLICED into windows of that
      length rather than re-sampled, so every calibration token survives (the
      sampler's 2048-token rows are the paper's protocol — a 512-token sample
      would shrink the calibration budget to a quarter); `calibration_rows`
      counted BEFORE the slice because it counts sequences and slicing adds
      none. The report records the RESOLVED `tune_seq_len` on both paths
      (`report_single_shape_both_paths` holds). Wrapper: `TUNE_SEQ_LEN`
      passthrough, empty default, flag appended only when set — the wrapper does
      not author a tuning geometry. Suites: **full 438 passed**; **lean 353
      passed / 16 skipped**, `skip discipline holds`; pre-commit **RC=0**;
      `validate --strict` valid. Design-silent choices, the RED transcript and a
      guard finding are recorded in this change's design.md
      ("Implementation crystallization 2026-10-27").
- [ ] (bug-hunter) Review: the seq-2048 eval clause cannot be satisfied by eval-at-512; default (flag unset) geometry is behavior-identical to pre-flag; any memory claim comes only from the 32g-container microbench with crumb-format peaks.
- [ ] (coder) Suite test first per `bar_constants_asserted`/`bar_not_purchased_with_bits`: a torch-free module or test module pinning the four row constants + tolerance arithmetic with the derivation audit (constants asserted, not recomputed). Observe RED.
- [ ] (coder) Wire the test location per repo layering (torch-free constants live where the band constants live — `qk_report` or a sibling bar module if the change spirit demands separation); GREEN in lean env.
- [ ] (orchestrator) Author the acceptance-run ledger format (fields per `bar_recorded_not_gated` scenario) as a doc/format note in this change, not code.
- [ ] (bug-hunter) Review: the constants match the pinned paper copy by grep (docs/papers/nanoquant-2602.06694v3.md rows 27.56/33.79/52.94/12.66); tolerance is 1.15x verbatim, not rounded in code; the seq-2048 clause cannot be satisfied by seq-512 data; the bar never feeds a per-run verdict.
- [ ] (orchestrator) `npx --yes @fission-ai/openspec validate qk-nq-recreation-bar --strict`; scoped stage (spec + code ONE commit); gate-chained commit; fenced push; CI watch.
- [ ] (orchestrator + team) Full gate suite before landing — `uv run pytest`, `cargo test` in `crates/quantibbler-kernels`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` — and record results.
