# qk-launch-attribution — Tasks

Rules honored: tests precede implementation (RED observed before
code); every task independently verifiable; final task is the full
gate suite.

## 1. Reference first (tests)

- [x] 1.1 RED: author `tests/ptq/test_launch_attribution.py` covering
      the spec scenarios: `launch identity line inside a checkout`
      (fake argv, temp scratch, rev == repo HEAD), `launch identity
      line outside a checkout` (no .git reachable → literal
      `unknown`), `default path disambiguates concurrent launches`
      (two concurrent driver processes → two `breadcrumbs-<pid>.log`
      files, no interleave), `configured path honored exactly`
      (QUANTIBBLER_BREADCRUMBS set → exact path, identity line first).
      Run `uv run pytest tests/ptq/test_launch_attribution.py` and
      OBSERVE the failures (missing rev/out fields, unsuffixed shared
      path). Record the RED output in the task note.
- [x] 1.2 GREEN: implement in `tools/quantize_hf_model.py` — extend
      the launch line with `rev` (runtime-resolved, `unknown` outside
      a checkout) and resolved `out` dir; default trail path becomes
      `breadcrumbs-<pid>.log` in the KD scratch; configured path
      honored exactly. Same command exits 0 on all scenarios.
      Observed: `uv run --extra torch --extra hf pytest
      tests/ptq/test_launch_attribution.py -q` → `4 passed in 66.80s`,
      RC=0 (dev CPU wheel).
- [x] 1.3 Pin rides the same commit: the launch-line construction
      joins the driver-wiring tripwire set
      (`tests/test_qk_report.py::_driver_wiring_findings`) — a later
      edit dropping a field reds the tripwire test. Tripwire battery
      still green after the addition.
      Observed: battery green (`tests/test_qk_report.py` → `94 passed`,
      RC=0) and the three new poisons red it
      (`test_report_driver_tripwire_rejects_poisoned_drivers
      [launch_line_dropped_rev…]`, `[…dropped_out…]`,
      `[…fields_reordered…]` → `23 passed, 71 deselected`, RC=0).

## 2. Wrapper interplay

- [x] 2.1 Confirm the container contract unchanged: wrapper's
      `DRY_RUN=1` output still names the same mounted breadcrumb path;
      `tests/test_run_ptq_wrapper.py` green; the forensics read path
      (tail of the configured file) unchanged.
      Observed: `DRY_RUN=1 sh tools/run-ptq-container.sh` still emits
      `-v '/opt/data/tmp:/tmp/ptq'` +
      `'QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log'`;
      `tests/test_run_ptq_wrapper.py` → `15 passed`, RC=0;
      `git diff --stat -- tools/run-ptq-container.sh
      tools/qkb-stall-watchdog.sh` empty.

## 3. Landing

- [x] 3.1 Gate suite: lean pytest (documented skip count), scoped
      pre-commit, rust legs, wheel build, strict validate. All observed
      green on 2026-10-25; evidence on ticket t_38b135eb.
      Observed: lean (`/opt/data/quantibbler/.venv-lean`, built by the CI
      recipe — fresh `uv venv` + `pip install -e . pytest pyyaml`, torch/
      transformers/tokenizers proven absent first) → `718 passed, 16
      skipped in 157.30s`, rc=0, and
      `.venv-lean/bin/python tools/ci/pytest_summary.py --log … --pytest-rc 0
      --expect-skipped 16` → `skip discipline holds`, rc=0 (16 =
      `test_kernels_parity.py` Rust leg, asserted exactly). Rust legs:
      `cargo test` in the kernels crate → 24 passed, rc=0; `cargo test
      --locked -p quantibbler-kernels -p quantibbler-server --all-targets`
      → 150 passed / 10 ignored, rc=0; both clippy forms `-D warnings`
      rc=0; `cargo fmt --all -- --check` rc=0; `maturin build` rc=0
      (wheel `quantibbler_kernels-0.1.0-cp311-abi3-manylinux_2_35_x86_64`).
      `pre-commit run --files <this change's files>` → all hooks Passed,
      rc=0, no file modified (hash before == after).
      `openspec validate qk-launch-attribution --strict` → `is valid`, rc=0.
      The scoped commit is the orchestrator's (leaf never commits); the
      changed-path manifest rode the completion of t_38b135eb.
- [ ] 3.2 Full tier on the interpreter × device the fire runs in
      (`quantibbler-ptq:rocm-official`, `/dev/kfd` + `/dev/dri`, tree
      mounted, `PYTHONPATH=/repo/src`). NOT RUN in this lane: a container
      launch needs the AGENTS.md team-room announcement and the box was
      co-tenant saturated (load ~50). Measured instead, for scale: the
      dev CPU wheel (`/opt/venv`, torch 2.14.1+cpu) gives `968 passed,
      14 failed, 3 skipped`, every failure the fail-closed
      `resolve_device` refusal in a device-unpinned file — an
      interpreter×device property, not a defect: A/B on a pristine-HEAD
      worktree, same interpreter, changes nothing (14 failed, 4.66 s).
      Owner: orchestrator / @user to schedule.
