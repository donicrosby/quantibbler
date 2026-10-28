# Tasks: qk-cli-sane

## Phase 1 — RED-first: fidelity conditions without the request flag

- [x] **1.1** `fidelity_conditions` drops `requested` parameter and `paper_faithful_flag` condition.
  Add `config` parameter carrying `solver`, `compute_dtype`, `factor_mode`, `ablation`,
  `layer_scope`, `admm_iter`, `rho_end`, `device`. Add all eight config-match conditions
  (`config_solver_is_admm`, `config_compute_dtype_is_bf16`, `config_factor_mode_is_binary`,
  `config_ablation_is_none`, `config_layer_scope_is_all`, `config_admm_iter_matches_regime`,
  `config_admm_rho_end_matches_regime`, `config_device_is_accelerator`). Tests RED: all
  existing `fidelity_conditions` call sites now fail on the missing `requested` argument.
  - *Verification*: `uv run pytest tests/test_qk_report_stages.py -k "fidelity"` — RED
    on arity errors. Count the failures.

- [x] **1.2** `fidelity_verdict` drops `requested` parameter, passes `config` through.
  Tests RED. *Verification*: same as 1.1, arity errors on `fidelity_verdict` callers.

## Phase 2 — Flip to green: fidelity call sites

- [x] **2.1** Update `fidelity_conditions` call sites in `src/quantibbler/qk_report/reports.py`
  to pass `config` instead of `requested`. Remove `paper_faithful` read from report identity
  fields; always compute the label. Tests GREEN.
  - *Verification*: `uv run pytest tests/test_qk_report_stages.py -k "fidelity"` — all pass.
    Assert the condition set includes `config_*` keys and excludes `paper_faithful_flag`.

- [x] **2.2** Update `fidelity_verdict` call sites. *Verification*: same suite, all pass.

- [x] **2.3** Update driver report construction to not read `--paper-faithful`; always
  call `fidelity_verdict` with the live config. *Verification*: tiny-model smoke —
  `uv run python tools/quantize_hf_model.py --model Qwen/Qwen3-0.6B --rank 2 --calibration-seqs 8 --eval-seqs 4 --seq-len 64` — produces `quantize_report.json` with `paper_faithful` field present and `config_*` conditions in the record.

## Phase 3 — Cut flags from the parser

- [x] **3.1** Remove `--paper-faithful`, `--ste-flip-pass-budget`, `--probe-interval`,
  `--probe-batches`, `--factor-mode`, `--ablation`, `--layer-scope` from the
  `argparse` parser. Remove their consumption sites in the driver, including
  the eval-window guard at `quantize_hf_model.py:1243-1253` (dies with the
  flag per design) and the `quantized_config()` docstring/budget switch
  (:414-440, re-keyed in 3.3). Suite RED on the missing arguments.
  - *Verification*: `uv run pytest tests/test_qk_report_stages.py tests/ptq/ -k "not integration"` — parse-error tests that passed now fail. Count.

- [x] **3.2** Add env-var reads: `QK_PROBE_INTERVAL` (default 1), `QK_PROBE_BATCHES`
  (default 2), `QK_FACTOR_MODE` (default `"binary"`), `QK_ABLATION` (default
  `"none"`), `QK_LAYER_SCOPE` (default `"all"`). Wire them into the driver's
  config construction and probe setup, with edge semantics: empty string treated
  as unset, non-integer probe values rejected at startup with a named error.
  Tests GREEN.
  - *Verification*: `uv run pytest tests/ptq/test_driver_env_knobs.py` — driver-level
    subprocess legs, one per case: every knob EMPTY takes its default (the report's
    `fidelity_config` and `probe` record it), `QK_FACTOR_MODE=fp` reads back from
    the report's `fidelity_config.factor_mode`, the probe override
    (`QK_PROBE_INTERVAL=4 QK_PROBE_BATCHES=1`) lands in `probe`, an invalid choice
    refuses at startup naming the variable and its accepted values, and each of the
    seven cut flags is rejected by the parser. `tests/ptq/test_driver_probe.py`
    keeps the probe knobs' non-integer refusal legs.
    Discrimination: reading an empty env var as a VALUE reddens the two
    empty-string legs; ignoring the environment entirely reddens the two wiring
    legs (both measured).

- [x] **3.3** Re-key the four runtime gates on the config-match conjunction
  (`config_matches_paper_regime`): polish enable (`_block_stage.py:251`), STE
  declared path (`_block_stage.py:559`), vacuous-teacher abort (`_tuning.py:328`),
  and the stage-budget switch in `quantized_config()` (`quantize_hf_model.py:433-440`)
  — paper-regime runs build `StageHyper(1e-4,4,8)` / `StageHyper(1e-5,1,8)` /
  `StageHyper(1e-6,1,8)`, every other run keeps the gate-shaped triple. Rewrite
  the `quantized_config()` docstring (:414-425) to describe regime-derived budgets.
  - *Verification*: `uv run pytest tests/ptq/test_ste_sign_adaptation.py tests/ptq/test_root_cause_ablation.py -k "paper or polish or vacuous or budget"` — paper-regime
    runs produce `declared="paper_ste"` and no flip_pass, and carry the App. C
    StageHyper triple; vacuous teacher raises only on paper regime; non-paper
    binary runs still polish and carry the gate-shaped triple.

- [x] **3.4** Reword the fp-extraction refusal at `_factorized.py:245-251` from
  `--factor-mode fp` to `QK_FACTOR_MODE=fp` (the argv no longer exists post-cut).
  - *Verification*: `grep -rn -- '--factor-mode' src/` returns zero hits.

## Phase 4 — Remove the discrete flip-pass mechanism

- [x] **4.1** Delete the flip-pass mechanism at every site: the call path in
  `_block_stage.py` (:574-589 — zero-crossing branch goes straight to
  `"scale_only"` with rescope marker), the `_discrete_flip_pass` definition
  in `_ste.py:391`, the import at `_block_stage.py:39`, and the re-export at
  `block_recon.py:174-176`. Remove `PTQConfig.ste_flip_pass_budget`.
  Remove `flip_pass` key emission from STE event records. Tests RED on the
  removed field.
  - *Verification*: `uv run pytest tests/test_qk_report_stages.py` — tests asserting
    `flip_pass` in per_linear records now fail.

- [x] **4.2** Update test fixtures to expect no `flip_pass` key. Remove flip-pass
  fixture cases. Update the pinned `declared` set at
  `tests/ptq/test_ste_sign_adaptation.py:472` from three values to two
  (`discrete_flip_pass` leaves the vocabulary — the canary fails if it
  re-enters). Tests GREEN.
  - *Verification*: `uv run pytest tests/test_qk_report_stages.py` — all pass.

## Phase 5 — Operator-grade --help

- [x] **5.1** Rewrite the `argparse` description and every surviving flag's `help`
  string per the design.md table. Group flags with `add_argument_group`.
  The `--remainder-dtype` help SHALL state "default: tracks compute dtype" (per
  pipeline-dtype's remainder-dtype requirement).
  - *Verification*: `uv run python tools/quantize_hf_model.py --help` — output matches
    the design.md shape. Every flag names its unit and default. No flag uses
    the word "paper" or "faithful" in its help text. No citation of change ids
    or lane names.

- [x] **5.2** Add a gate test that asserts `--help` output contains the expected
  sections ("Model and data:", "Solver and quality:", "Output:", "Resources:"),
  the surviving flag names under their own section, and does NOT contain the cut
  flag names or bot vocabulary.
  - *Verification*: `uv run pytest tests/ptq/test_driver_help_surface.py -k help` —
    1 passed. The gate reads the SECTION BODIES only (the `usage:` summary is
    excluded, so a flag named there but missing from its group still reds),
    asserts each of the 22 surviving flags sits in its own group, and carries the
    negative half (all seven cut flags absent, no "paper"/"faithful").
    Discrimination: putting "faithful" back in the parser description, and putting
    a cut flag name back in a help string, each redden this leg (both measured).

## Phase 6 — Wrapper and CI migration (atomic with the parser cut)

- [x] **6.1** Migrate `tools/run-ptq-container.sh`: drop `--paper-faithful` and
  `--ste-flip-pass-budget` from argv (:360-367); replace `--factor-mode`,
  `--ablation`, `--layer-scope` pass-throughs (:351-359) with
  `QK_FACTOR_MODE=`, `QK_ABLATION=`, `QK_LAYER_SCOPE=` in the container
  environment. ALSO drop `PAPER_FAITHFUL`/`STE_FLIP_PASS_BUDGET` from the
  wrapper's entire surface: usage comment (:15), env parse (:65-72),
  usage echo (:88), advisory note (:173-179 — delete outright), announce
  field (:183), launch.env printf (:199-200), breadcrumbs header (:425-426).
  - *Verification*: `uv run pytest tests/test_run_ptq_wrapper.py` — all pass;
    `grep -n 'PAPER_FAITHFUL\|STE_FLIP_PASS_BUDGET' tools/run-ptq-container.sh`
    returns zero hits.

- [x] **6.2** Re-key `tools/ci/report_checks.py:146` from
  `paper_faithful_flag is True` to `config_matches_paper_regime`.
  - *Verification*: `uv run pytest tests/test_ci_referee.py -k "report_checks"` — the
    evidence detector fires on paper-regime runs and skips on non-paper runs.

- [x] **6.3** Migrate `tools/ci/driver_smoke.py:201` and `tools/ci/abort_leg.py:142`
  from `--probe-batches 1` to `QK_PROBE_BATCHES=1` in the subprocess environment.
  - *Verification*: Both CI legs pass `uv run pytest tests/test_ci_referee.py` — no
    argparse errors, probe batches are 1.

## Phase 7 — Spec and config hygiene

- [ ] **7.1** (not run in this leaf) Verify `openspec/config.yaml` line 103 (numerics invariant: "float64 by default")
  is owned by pipeline-dtype's delta — pipeline-dtype flips it to bf16 default. Confirm
  pipeline-dtype's delta already covers the change; this task is a dependency check, not a
  write. If line 103 still reads "float64 by default" after pipeline-dtype lands, flag it.
  - *Verification*: `grep -n "float64" openspec/config.yaml` — the line reflects bf16 default
    (post-pipeline-dtype). No self-assertion fidelity reference anywhere in config.yaml.

- [ ] **7.2** (not run in this leaf) Run `openspec validate qk-cli-sane --strict`. Resolve any warnings.
  - *Verification*: RC=0, "Change 'qk-cli-sane' is valid."

- [ ] **7.3** (not run in this leaf) Run the full gate suite: `uv run pytest` (lean), `uv run maturin build
  --manifest-path crates/quantibbler-kernels/Cargo.toml`, `cd crates/quantibbler-kernels
  && cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --all -- --check`.
  - *Verification*: All green. Record the lean passed/skipped count with the
    extras state demonstrated.

## Phase 8 — Canary test enumeration

- [ ] **8.1** (not run in this leaf: the mutation half) Enumerate canary tests per file:line that must fail under their own
  mutation. Minimum set (per review grep-verified hit list):
  `tests/test_qk_report_stages.py:223,317,474-513`;
  `tests/ptq/test_driver_stage_map.py:74,117-122,143,319-320,398-400`;
  `tests/test_ci_referee.py:244-247,403,424-425`;
  `tests/ptq/test_ste_sign_adaptation.py:25-26,366,472`;
  `tests/ptq/test_zz_flip_budget.py`;
  `tests/ptq/test_root_cause_ablation.py`;
  `tests/test_serialization.py`;
  `tests/test_run_ptq_wrapper.py`;
  `tests/test_qk_report_abort.py`;
  `tests/test_qk_report_driver_pin.py`.
  Each enumerated test SHALL fail when the field or condition it pins is mutated
  (removed key, renamed condition, wrong default).
  The `:472` pin updates with task 4.2 — the `declared` set shrinks to
  `{"paper_ste", "scale_only"}` and the canary fails if `discrete_flip_pass`
  re-enters the vocabulary.
  - *Verification*: `uv run pytest <each-file> -x` — every enumerated test passes
    on the clean tree. Mutation check: manually delete one field/condition and
    confirm the pinning test fails.

- [x] **8.2** Repurpose `serialization.py:189` `paper_faithful` field from
  self-assertion flag to the computed label (always present, no longer optional).
  - *Verification*: `uv run pytest tests/test_serialization.py` — all
    serialization tests pass with the computed label present in every ledger row.

## Phase 9 — Landing

- [ ] **9.1** (branch commit exists; landing awaits review) Commit: `Land qk-cli-sane: cut bot-authored CLI flags, re-anchor
  fidelity from config`. Scoped to `openspec/changes/qk-cli-sane/` plus the
  touched source, wrapper, and CI files. Gate chain in body.
