# Tasks: qk-ste-joint

Interpreter for every reading below: `/opt/venv` (`torch 2.14.1+cpu`, no
accelerator), invoked as `PYTHONPATH=/opt/data/tmp/t_029df0de/src uv run
--no-sync …` — the shared venv's editable install resolves `quantibbler` to the
shared checkout, so `PYTHONPATH` is what names the tree the numbers came from.

Work ran in the isolated worktree `/opt/data/tmp/t_029df0de` (branch
`t_029df0de-ste-joint`), rebased `f538d0f` → `39f0fe9` after main moved under it.
Reset note: the shared checkout's `HEAD` moved and a peer merge discarded this
change's uncommitted edits mid-pass; everything below is measured in the
worktree, and the patch is landed by fast-forward from that branch, never by
staging in the live shared tree.

## 1. Spec

- [x] 1.1 `openspec validate qk-ste-joint --strict` is VALID, and every ADDED
      requirement body measures under the 500-character `--strict` limit.
      Evidence: `npx --yes @fission-ai/openspec validate qk-ste-joint --strict`
      → `Change 'qk-ste-joint' is valid` (RC=0); bodies measured 377 / 355 / 425
      / 277 chars.

## 2. The joint pass

- [x] 2.1 `reconstruct_block` invokes TuneLatentSTE exactly ONCE per block, after
      every factorization completes, handing it the block's whole factorized set.
      Evidence: spy over the call — GREEN `invocations: 1 sizes: [7]`; RED
      (pre-change) `invocations: 7 sizes: [1,1,1,1,1,1,1]`.
- [x] 2.2 The per-linear declaration ladder runs as a read-out of that one pass's
      rows (paper_ste / discrete_flip_pass / scale_only), unchanged in meaning.
      Evidence: `tests/ptq/test_ste_sign_adaptation.py` 15 passed (ladder,
      flip-pass reproducibility and acceptance, faithful exemption) and
      `tests/ptq/test_tunefp_teacher.py` 8 passed.
- [x] 2.3 The record is one block-scope record carrying per-linear rows, the
      `block_joint` scope marker, the block budget, and the record's own wall
      time; each row carries movement, flip and dead-column counts.
      Evidence: GREEN probe — `scope: block_joint total_steps 16 steps 16
      elapsed_s 0.2507`, 7 rows; `tests/ptq/test_ste_joint.py` 6 passed
      (`test_scn_ste_record_is_one_block_record_with_per_linear_rows`,
      `test_scn_dead_columns_are_recorded_per_linear`).

## 3. Consumers

- [x] 3.1 `qk_report.stage_row_times` counts one `ste` row per block fire (the
      record's own `elapsed_s`), not one per refined linear.
      Evidence: `tests/test_qk_report_stages.py` — both row-time tests amended to
      the new unit; root lean tier green, see 5.2.
- [x] 3.2 `qk_report._STAGE_LINES`' `ste` deviation describes the joint scope and
      names the deviation from line 19's prose.
      Evidence: the row is amended in `qk_report.py`; covered by the same root
      tier run.

## 4. RED → GREEN

- [x] 4.1 RED (pre-change, `840ce60`): seven per-linear rows with seven distinct
      objective readings; seven `_tune_latent_ste` invocations per block.
      Transcript (`/opt/data/tmp/t_029df0de-red`, `840ce60`):
      `RED invocations: 7 sizes: [1, 1, 1, 1, 1, 1, 1]`;
      `RED distinct residual_before: 7 [0.209847, 0.232117, 0.545375, 0.572653,
      1.008034, 1.461143, 1.565284]`; `RED record has scope: False`.
      The new test file also fails at collection there
      (`ImportError: cannot import name 'STE_JOINT_SCOPE'`) — no joint entry
      point and no record contract exist on the pre-change tree.
- [x] 4.2 GREEN (post-change): one invocation per block with the full factorized
      set; seven rows sharing one measurement; scope marker and dead columns on
      the record. Transcript:
      `GREEN invocations: 1 sizes: [7]`;
      `GREEN distinct residual_before: 1 [1.517289, …]`;
      `GREEN scope: block_joint total_steps 16 steps 16 elapsed_s 0.2507`;
      `GREEN rows: 7`; `tests/ptq/test_ste_joint.py` 6 passed.

## 5. Gates

- [x] 5.1 `tests/ptq` run one file at a time, per-file verdict printed: 44 files,
      42 × `RC=0`, `test_driver_stage_journal.py RC=5` (its single test is
      `@pytest.mark.integration` and is deselected — pre-existing), and
      `test_driver_probe.py RC=1` on one pass under a loaded box which then
      passed in isolation (`8 passed in 128.93s`) — the box's
      driver-subprocess load class, not the diff. At the pre-rebase revision the
      same file-by-file run was 43 × `RC=0` + the deselected file, 277 passed /
      7 deselected.
- [x] 5.2 Root lean tier re-quoted WITH its denominator and against a baseline
      taken at the same revision: base `39f0fe9` (no change) `808 passed in
      23.59s`; this change `808 passed in 25.09s` — the same count, RC=0 on both.
      (`tests/test_serialization.py` needed `pydantic`, which main's
      qk-serialization landing left out of the shared venv; the shared env was
      refreshed with `uv sync` from the shared checkout and announced on the
      card as an env note.)
- [x] 5.3 `ruff check` → `All checks passed!`; `ruff format --check` → `104 files
      already formatted`; `vulture --min-confidence 80` on `src`/`tools` reports
      only a pre-existing finding in `tools/capture_attention_activations.py`;
      `pre-commit` hooks passed on the commit; `cd crates/quantibbler-kernels &&
      cargo test --locked` → **24 passed / 0 failed**.
- [x] 5.4 Scoped commit landed: `src/quantibbler/ptq/block_recon.py`,
      `src/quantibbler/qk_report.py`, `tools/stage_row_time_bench.py`,
      `tests/ptq/test_ste_joint.py`, the four amended ptq test files,
      `tests/test_qk_report_stages.py`, and this change directory.

## 6. Findings for the planner (not amended here)

- `openspec/changes/qk-fix-ste-sign-adaptation/` (UNARCHIVED) carries a scenario
  requiring a FULL PER-LINEAR budget, which this change supersedes. The
  requirement of record (`openspec/specs/qk-ptq/spec.md`) already reads
  "jointly refined", so the conflict is with an unreleased proposal; rewriting a
  SHALL inside another change's delta is a planner call. Recorded in
  `design.md` §3a.

## 7. Not in this change

- No re-fire: the fire question returns to @user after the stage-fix set lands.
- No change to solver, factorization, polish, stage budget values, or the
  declaration ladder's per-linear semantics.
