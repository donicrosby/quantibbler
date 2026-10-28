# Tasks: qk-launch-attribution (rewritten 2026-10-25)

> **Audit landing note (2026-10-25, `t_039035de`, ruling: verdicts from
> `qk-spec-audit`).** This `tasks.md` is a REWRITE, not the historical ledger:
> it replaces the prior task list so the open rows are the change's real
> backlog. Completed work is summarized above the open rows; the detailed
> row-by-row history is in git. Any numeric "suite state" reading below is the
> audit's, taken at the then-current HEAD — re-measure before quoting it.

## Completed (tasks 1–3.1)

Code landed: launch identity line (`rev`, resolved `out` dir, `breadcrums-<pid>.log`, configured path honored). Tasks 1.1–1.3 (RED→GREEN, tripwire pin), 2.1 (wrapper interplay unchanged), 3.1 (gate suite) all verified green on 2026-10-25:

- Lean tier: 718 passed, 16 skipped RC=0 (skip discipline holds)
- Rust legs: cargo test 24/0 RC=0, cargo test server 150 passed/10 ignored RC=0, clippy -D warnings RC=0, fmt --check RC=0
- `uv run maturin build --manifest-path crates/quantibbler-kernels/Cargo.toml` RC=0
- `npx --yes @fission-ai/openspec validate qk-launch-attribution --strict` VALID RC=0
- `pre-commit run --files <changed paths>` all Passed RC=0

## Completed (2026-10-25, launch ownership — incident card `t_29f52eb6`)

Row 4 is the incident-driven half of this change: at 18:10:32Z on 2026-10-25 an
unattributed paper-faithful fire ran (container `quantibbler-ptq-run`, launch id
`qk-pertarget-v4-20261025T1815Z`) with no owner stamp, no announcement and no GO,
and the launch path could not say whose run it was. Enforcement now lives at BOTH
ends: the wrapper refuses and stamps the labels, the driver refuses from inside
(a direct `docker run` never touches the wrapper) and records the owner in the
trail's first line.

- [x] 4.1 RED-first, both gates, against the pre-change bytes (`git show
      HEAD:<path>` into `/opt/data/tmp/prefix-red/`, so the observation is of the
      shipped tree rather than a hand-mutated copy): `DRY_RUN=1 WATCHDOG=0 sh
      <pre-change wrapper>` with no owner → **RC=0** and the full `'docker' 'run' …`
      argv printed; `<pre-change driver> --device cpu --model
      /nonexistent-model-xyz --out …` with no owner → started for real and wrote
      `… rev=unknown out=…` with NO owner field. Post-change both refuse:
      `error: launch ownership missing`, exit 2 / exit 1, no argv, no trail.
- [x] 4.2 Attributed launch green, and the refusal pinned. Every number below was
      re-measured AFTER the `ruff format` pass, so none of them predates the last
      code-touching edit: `uv run --no-sync --extra torch --extra kernels --extra hf
      python -m pytest tests/test_run_ptq_wrapper.py tests/test_qk_report.py
      tests/test_ci_referee.py tests/ptq/test_launch_attribution.py -q` → `194 passed
      in 37.56s`, RC=0 (20 + 98 + 69 + 7; the 7 are the four earlier scenarios plus
      the three ownership scenarios). The launch-line tripwire now pins ` owner=` and
      ` card=` and carries a `launch_line_dropped_owner` poison.
- [x] 4.3 Owner line machine-written at fire time: `--label qkb_launch_owner` /
      `qkb_launch_card` beside `qkb_launch_id`, both mirrored into the container env,
      named in the announce, in `launch.env`, in the trail's `=== run …` header, and
      in the driver's first line (`pid, argv, rev, out, owner, card`). Wrapper legs
      `test_scn_attributed_launch_stamps_owner_labels_and_owner_line` and
      `test_scn_attributed_fire_writes_the_owner_into_the_trail_header` (real launch
      path, not a source grep).
- [x] 4.4 Spec delta + dated crystallization + AGENTS.md paragraph in the same
      landing: `npx --yes @fission-ai/openspec validate qk-launch-attribution
      --strict` → `Change 'qk-launch-attribution' is valid`, RC=0 (new qk-wrapper
      delta + the qk-ptq requirement, both bodies under the 500-char strict limit:
      462 and 445 chars measured).
- [x] 4.5 The other driver-subprocess files, re-run after the change (each now
      sets `QKB_LAUNCH_OWNER=pytest`; they are the only files that exercise the
      driver's `main()` as a subprocess). Runner logs `FILE <path> RC=<rc>`,
      re-run after the format pass: `test_driver_probe.py` → `8 passed in 32.85s`
      RC=0; `test_driver_tune_seq_len.py` → `3 passed in 13.52s` RC=0;
      `test_geometry_conformance.py` → `5 passed in 5.73s` RC=0.
- [x] 4.6 Whole-tier re-measure on this cut, one file at a time (the box is
      co-tenant loaded, so a directory run blows the tool timeout with nothing to
      show): `tests/ptq` → **264 passed, 0 failed, 0 skipped**, 38/39 files RC=0 and
      `test_driver_stage_journal.py` RC=5 with `1 deselected` (every test in it is
      `integration`-marked, so the default marker filter runs none of it — the
      documented shape, not a red); `tests --ignore=tests/ptq` → `764 passed in
      24.86s` RC=0; `cargo test` in `crates/quantibbler-kernels` → 24 passed RC=0
      (no Rust bytes changed); `uv run --no-sync pre-commit run --files <changed
      paths>` → every hook Passed except `ruff format`, which reformatted two of
      this cut's Python files and is now clean (`ruff format --check` → `9 files
      already formatted`, `ruff check` → `All checks passed`).
      Interpreter: `/opt/venv` (torch 2.14.1+cpu, no accelerator), full extras,
      `--no-sync` on every `uv run`.

## Open

- [ ] 3.2 Full tier on the interpreter × device the fire runs in (`quantibbler-ptq:rocm-official`, `/dev/kfd` + `/dev/dri`). Container launch needs the AGENTS.md team-room announcement. This is a nice-to-have — not blocking.
  - **Note (2026-10-25)**: Dev CPU wheel gives 968 passed, 14 failed (all fail-closed `resolve_device` refusals), 3 skipped. The fire-lane number is a standing record that needs a quiet box.
