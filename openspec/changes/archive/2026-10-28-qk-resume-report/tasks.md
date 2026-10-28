# Tasks: qk-resume-report

Each task is independently verifiable. Tests precede implementation per
capability; test names reference scenario IDs. Last task runs the full gate
suite.

- [x] 1.1 Add a torch-free `resume`-record builder in `src/quantibbler/qk_report/`
  that constructs the record dict from plain inputs (resumed flag,
  restored-blocks list, checkpoint-dir string, image/digest strings,
  fingerprint-admission dict); emits `resumed: false` + empty
  `restored_blocks` + `checkpoint_dir: null` + `admitted_by_fingerprint: null` for a fresh run; never asserts a computed
  `identical_bytes` boolean. Verification: a new lean test (no torch) covers
  `resume_record_on_resumed_run`, `resume_record_fresh_run_consistent_shape`, and
  `resume_identity_is_evidence_not_an_assertion` (no `identical_bytes` key),
  running without skips in a lean environment.

- [x] 1.2 Wire the record into `tools/quantize_hf_model.py`: read
  `completed_blocks(_ckpt_dir)` where `_ckpt_dir` is resolved (before
  `quantize_model`), read the restored `fingerprint.json` — its `fingerprint`
  digest, its `model_identity` digest, and its provenance members (the
  `_FINGERPRINT_PROVENANCE` tuple plus `solver_device`) — read
  `QUANTIBBLER_IMAGE` / `QUANTIBBLER_IMAGE_DIGEST` (default `"unknown"`),
  and thread the record through `_run_identity` so the completion,
  probe-abort, and stage-failure reports all carry the same `resume` key.
  Verification: `resume_record_on_resumed_run`,
  `resume_record_fresh_run_consistent_shape`,
  `resume_record_on_failure_paths`, `resume_record_is_a_record_not_a_gate`,
  `resume_true_only_when_admitted` —
  present on all three shapes; a wired-tier refusal test poisons the restored
  `fingerprint.json` digest (the `fingerprint` key), relaunches, and asserts
  the `RuntimeError` propagates unchecked (not converted to
  `PrefixProbeAbort`/`VacuousTeacherError`) and no `quantize_report.json` is
  written — red-first against a mutant that catches-and-continues;
  additionally, post-build mutate the resume record to adversarial garbage
  (`resumed: "banana"`, `restored_blocks: ["x"]`, `checkpoint_dir: 42`,
  `admitted_by_fingerprint: "junk"`) and assert verdict, exit code, band
  edges, and `dump_report` acceptance are unchanged (the record is never read
  by a gate — discriminating, not coincidental).

- [x] 1.3 Add the fingerprint non-folding guard (SHALL NOT): a suite test
  asserting the resume record does not change `config_fingerprint` for a fixed
  config, that `PTQConfig` gained no resume field, and that
  `_FINGERPRINT_PROVENANCE` is unchanged. Verification:
  `resume_marker_never_folds_the_fingerprint`, red-first — a poisoned config
  (a `resumed` field) changes `config_fingerprint`; the real config does not.

- [x] 1.4 Run the full gate suite (`uv run pytest`; kernels `cargo test` +
  `clippy --all-targets -- -D warnings` + `fmt --check`; `maturin build`) and
  `openspec validate qk-resume-report --strict`. Record results; every gate
  green before the change lands. Green, with one pre-existing integration-lane
  red outside this change's diff: `test_driver_stage_map.py`'s faithful stage-map
  leg asserts a hand-pinned `fidelity_conditions` set that no longer matches the
  record's own `capture_is_fp_stream` condition. Reproduced verbatim on the
  pre-change revision, so it is a stale pin for that leg's owner, not this one.
