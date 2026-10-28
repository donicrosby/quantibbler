# Design: qk-cli-sane

## Cut list — per-flag justification

Each removed flag is audited through two levels: **(L1)** symbol-level usage grep
across the tree to confirm every consumer is identified, and **(L2)**
requirement→mechanism grep proving the flag serves bot automation, not an
operator running a compression.

### `--paper-faithful` — CUT

- **L1 usage**: Driver parser `--paper-faithful` argument (line 1071), wrapper
  pass-through (line 362), `PTQConfig.paper_faithful` (config.py:196-208),
  consumed at FIVE behavior-changing sites. Four runtime gates: polish enable
  (`_block_stage.py:251` — `not cfg.paper_faithful`), STE declared path
  (`_block_stage.py:559` — `if cfg.paper_faithful` yields `"paper_ste"`),
  vacuous-teacher abort (`_tuning.py:328` —
  `if cfg.paper_faithful: raise VacuousTeacherError`), and the App. C
  stage-budget switch in `quantized_config()` (`quantize_hf_model.py:433-440`
  — `getattr(args, "paper_faithful", False)` selects between the paper
  `StageHyper` triple and the gate-shaped triple). One startup guard: the
  eval-window maximality check (`quantize_hf_model.py:1243-1253`). Fidelity
  conditions consume it as `requested` (stages.py:100; driver call sites
  :538 identity read and :1541 `requested=`). The report identity fields carry
  it as a top-level key (reports.py:61-70). CI detector keys on
  `paper_faithful_flag` (report_checks.py:146). Ledger carries the field
  (serialization.py:189). Stage-map emits `discrete_flip_pass` keyed on
  flip_pass presence (stages.py:417-420).
- **L2 mechanism**: The flag is a self-assertion — the user claims fidelity, then
  the report verifies against stage records. An operator running a compression
  does not self-assert whether they followed a paper; they declare their config
  and the machine checks. The flag name itself ("paper-faithful") is a bot
  vocabulary term, not an operator's. The fidelity label SHOULD be computed from
  declared config vs paper regime with no request flag at all.
- **Migration**: `PTQConfig.paper_faithful` field is removed. All FOUR runtime
  gates re-key on the config-match conjunction (`config_matches_paper_regime =`
  solver=admm ∧ compute_dtype=bf16 ∧ factor_mode=binary ∧ ablation=none ∧
  layer_scope=all ∧ admm_iter=400 ∧ 0.0≤rho_end≤1.0 ∧ device≠cpu) — the
  re-anchoring's own philosophy applied consistently: polish runs only on
  non-paper binary runs, the paper_ste declaration fires on paper-regime runs,
  the vacuous-teacher abort is fail-closed on paper-regime runs, and the
  stage-budget switch in `quantized_config()` (`quantize_hf_model.py:433-440`)
  builds the App. C budgets (`StageHyper(1e-4,4,8)` / `StageHyper(1e-5,1,8)` /
  `StageHyper(1e-6,1,8)` — lr, batch_size, epochs) on paper-regime runs and the
  gate-shaped triple (`StageHyper(1e-4,4,0)` / `StageHyper(2e-3,4,3)` /
  `StageHyper(1e-3,4,4)`) on every other run. Consequence, stated plainly: a
  run whose config matches the regime IS a paper-regime run — including a
  defaults-only run once `qk-pipeline-dtype`'s bf16 default lands — and it gets
  the App. C schedule because its declared config is the paper's config. The
  budgets become derived state (a function of the regime detector), so no ninth
  config condition is needed: a regime-matching run mechanically carries the
  App. C schedule and cannot silently deviate from it. The switch and the
  `quantized_config()` docstring (:414-425) are rewritten to describe
  regime-derived budgets.
  The eval-window guard (`quantize_hf_model.py:1243-1253`) DIES DELIBERATELY
  with the flag: it enforced claim hygiene on a self-asserted flag; re-keying
  it on the conjunction would make `eval_seqs` a ninth regime field through
  the back door of a startup refusal; evaluation comparability is already
  outside the method verdict by the report's own separation — the sliced
  window stays recorded (selected blocks, source-token coverage in the
  evaluation record) and never denies the label. `fidelity_conditions` drops
  the `requested` parameter and the `paper_faithful_flag` condition. The label
  is always computed. The ledger's `paper_faithful` field
  (serialization.py:189) is repurposed to the computed label (always present).
  The stage-map `discrete_flip_pass` key (stages.py:417-420) stays for schema
  compatibility and is always-false on paper-regime runs.

### `--ste-flip-pass-budget` — CUT (mechanism removed)

- **L1 usage**: Driver parser (line 1080), wrapper pass-through (line 365),
  `PTQConfig.ste_flip_pass_budget` (config.py:209-217), consumed at the
  diagnostic flip-pass call site (`_block_stage.py:574` —
  `budget=cfg.ste_flip_pass_budget`). The mechanism's definition is
  `_ste.py:391` (emitting `"mechanism": "discrete_flip_pass"` at :424),
  imported at `_block_stage.py:39`, and re-exported at
  `block_recon.py:174-176` — all four sites are in the removal inventory.
  The STE event's `per_linear` records carry a `flip_pass` key when the
  pass ran (read by stages.py:91-92). CI smoke and abort legs do NOT pass
  this flag.
- **L2 mechanism**: The discrete sign-flip pass is a non-paper mechanism the bot
  added for ablation. The paper's path ([NQ] Algorithm 1 lines 20–23) extracts
  `sign(U)`, `sign(V)` and packs — no discrete refinement step. A human
  operator does not tune the budget of a deviation mechanism; the mechanism
  itself is bot scaffolding. Removing the flag AND the mechanism eliminates a
  named deviation from every run.
- **Migration**: The flip-pass mechanism is deleted at EVERY site — definition,
  import, re-export, and call:
  1. The call path in `_block_stage.py` (:574-589) is deleted; the
     zero-crossing branch goes straight to `"scale_only"` with the rescope
     marker (no flip attempt in between)
  2. The `_discrete_flip_pass` definition in `_ste.py:391` is deleted
  3. The import at `_block_stage.py:39` and the re-export at
     `block_recon.py:174-176` are deleted
  4. `PTQConfig.ste_flip_pass_budget` field is removed
  5. The `flip_pass` key is removed from STE `per_linear` records
  6. The fidelity condition `no_discrete_flip_pass` becomes always-true — it
     stays in the condition set for auditability but is never the denying
     condition

  **`declared` vocabulary ruling:** the post-cut set is two-valued —
  `{"paper_ste", "scale_only"}`. `discrete_flip_pass` leaves the vocabulary
  because the mechanism it names no longer exists. The pinned three-value
  set in `tests/ptq/test_ste_sign_adaptation.py:472` shrinks accordingly
  (a canary row, updated with the fixtures in task 4.2).

### `--probe-interval` — MOVE to `QK_PROBE_INTERVAL`

- **L1 usage**: Driver parser (line 993), consumed in the driver's progress-probe
  loop to control how often the per-block prefix probe fires. The wrapper does
  NOT pass this flag (it's a lane-internal diagnostic knob).
- **L2 mechanism**: This is a progress-monitoring knob for lane automation — a
  bot watching a long run wants to sample every block; an operator running a
  compression wants the probe ON (fail-fast) or OFF and does not tune the
  sampling interval per run. The interval is a lane instrumentation parameter,
  not a compression parameter.
- **Migration**: Driver reads `QK_PROBE_INTERVAL` env var (default 1). Parser
  drops the `--probe-interval` argument. `PTQConfig` gains no new field — the
  env var is read at the driver level only.

### `--probe-batches` — MOVE to `QK_PROBE_BATCHES`

- **L1 usage**: Driver parser (line 1000), consumed alongside probe-interval to
  set the eval-batch count in the progress-probe slice. CI legs
  `driver_smoke.py:201` and `abort_leg.py:142` pass `--probe-batches 1`.
- **L2 mechanism**: Same class as `--probe-interval`: lane instrumentation, not
  compression parameter. An operator wants the probe to work; they do not tune
  its batch count per run.
- **Migration**: Driver reads `QK_PROBE_BATCHES` env var (default 2). Parser
  drops the argument. Both CI legs migrate to `QK_PROBE_BATCHES=1`.

### `--factor-mode` — MOVE to `QK_FACTOR_MODE`

- **L1 usage**: Driver parser (line 1015), wrapper pass-through (line 352),
  `PTQConfig.factor_mode` (config.py field). The `binary` vs `fp` choice
  controls whether the pipeline uses binary latents or continuous latents.
  The fp-extraction refusal at `_factorized.py:250` names the flag in
  operator-facing prose ("Drop --factor-mode fp for a run that exports") —
  post-cut the named argv does not exist.
- **L2 mechanism**: `fp` mode is a root-cause ablation measurement — it removes
  the binary constraint to isolate whether degradation comes from the bits
  themselves or from the optimizer. An operator running a compression always
  wants `binary`; `fp` produces a measurement, never an artifact. This is a
  diagnostic knob the bot added for the root-cause line of work.
- **Migration**: `PTQConfig.factor_mode` stays (it's a config field, not a CLI
  surface concern). The driver reads `QK_FACTOR_MODE` env var (default
  `"binary"`) and passes it into `PTQConfig`. The parser drops the argument.
  The wrapper sets `QK_FACTOR_MODE` instead of `--factor-mode`. The refusal
  text at `_factorized.py:250` is reworded to name `QK_FACTOR_MODE` (set it
  to `binary` or unset it for a run that exports); the site joins the
  cut-surface inventory.

### `--ablation` — MOVE to `QK_ABLATION`

- **L1 usage**: Driver parser (line 1023), wrapper pass-through (line 355),
  `PTQConfig.ablation` (config.py field). The `untuned-signs` mode exports
  `sign()` of balance latents with zero refinement.
- **L2 mechanism**: Same class as `--factor-mode`: measurement-only diagnostic
  for the root-cause line of work. An operator never sets this.
- **Migration**: `PTQConfig.ablation` stays. Driver reads `QK_ABLATION` env var
  (default `"none"`). Parser drops the argument.

### `--layer-scope` — MOVE to `QK_LAYER_SCOPE`

- **L1 usage**: Driver parser (line 1032), wrapper pass-through (line 358),
  `PTQConfig.layer_scope` (config.py field). Restricts factorization to
  `self_attn` or `mlp` component only.
- **L2 mechanism**: Diagnostic knob for component-level ablation. An operator
  compressing a model always factorizes the full layer set.
- **Migration**: `PTQConfig.layer_scope` stays. Driver reads `QK_LAYER_SCOPE`
  env var (default `"all"`). Parser drops the argument.

### Surviving flags (no change)

Every flag NOT listed above stays. The complete surviving set after this change
plus `qk-pipeline-dtype`:

| Flag | Category | Rationale |
|---|---|---|
| `--model` | model/data | Operator picks the model |
| `--rank` | solver/quality | Core compression parameter |
| `--seq-len` | model/data | Evaluation geometry |
| `--tune-seq-len` | solver/quality | Memory-bound tuning window |
| `--calibration-seqs` | model/data | Calibration data size |
| `--eval-seqs` | model/data | Held-out evaluation size |
| `--max-admm-iter` | solver/quality | Solver iteration budget |
| `--solver` | solver/quality | ADMM vs SVD (real choice) |
| `--scale-dtype` | output | Scale storage precision |
| `--remainder-dtype` | output | Sidecar storage precision |
| `--checkpoint-dir` | output | Resume checkpoint location |
| `--no-checkpoints` | output | Disable checkpointing |
| `--compute-dtype` | solver/quality | Pipeline numerics (bf16/f32/f64) |
| `--device` | resources | CPU vs accelerator |
| `--max-threads` | resources | CPU thread cap |
| `--no-good-guest` | resources | Disable auto thread/memory tuning |
| `--gpu-memory-fraction` | resources | GPU memory cap |
| `--seed` | model/data | Reproducibility |
| `--out` | output | Artifact directory |
| `--container-format` | output | blob vs dir export |

## Fidelity ladder — re-anchored from config

### Current state

The report's `paper_faithful` field is a COMPUTED label (already correct — not
a forwarded flag), but the computation consumes a `requested` parameter that
originates from the `--paper-faithful` CLI flag. The condition set includes
`paper_faithful_flag: bool(requested)`. A run that does not set the flag always
gets `paper_faithful: false` regardless of whether the actual pipeline matches
the paper.

### Post-cut design

The `paper_faithful` label is always computed. There is no request flag — the
label reflects whether the declared config AND the observed execution match the
paper regime. The condition set drops `paper_faithful_flag` and adds
`config_matches_regime` conditions derived from the pipeline configuration:

```
fidelity_conditions(solver, runtime_deviations, stages, *, stage_failure, config) -> dict[str, bool]
```

**Paper regime definition** (the values [NQ] App. C prescribes):

| Config field | Paper value |
|---|---|
| `solver` | `"admm"` |
| `compute_dtype` | `"bf16"` |
| `factor_mode` | `"binary"` |
| `ablation` | `"none"` |
| `layer_scope` | `"all"` |
| `admm_iter` | = 400 (the paper budget; any other value is a deviation) |
| `admm_rho_end` | in [0.0, 1.0] (per existing spec) |
| `device` | accelerator (GPU-first; CPU is a deviation) |

**Config-match conditions** (one per regime field, named `config_<field>_matches_regime`):

- `config_solver_is_admm`: `config.solver == "admm"`
- `config_compute_dtype_is_bf16`: `config.compute_dtype == "bf16"` (pure string compare — `PTQConfig.compute_dtype` is `Literal["bf16", "float32", "float64"]` per pipeline-dtype)
- `config_factor_mode_is_binary`: `config.factor_mode == "binary"`
- `config_ablation_is_none`: `config.ablation == "none"`
- `config_layer_scope_is_all`: `config.layer_scope == "all"`
- `config_admm_iter_matches_regime`: `config.admm_iter == 400`
- `config_admm_rho_end_matches_regime`: `0.0 <= config.rho_end <= 1.0`
- `config_device_is_accelerator`: `config.device != "cpu"` (GPU-first; `auto` on a GPU box satisfies this, explicit CPU is a deviation)

The **config-match conjunction** for runtime gates:
`config_matches_paper_regime = (solver="admm" ∧ compute_dtype="bf16" ∧ factor_mode="binary" ∧ ablation="none" ∧ layer_scope="all" ∧ admm_iter=400 ∧ 0.0≤rho_end≤1.0 ∧ device≠cpu)`.

These replace `paper_faithful_flag`. The stage-level conditions are unchanged
but their semantics on paper-regime runs are now deterministic per the
config-match conjunction: `polish_off` is always-true (polish never fires on
paper regime), `no_discrete_flip_pass` is always-true (no diagnostic path
invoked on paper regime), `no_scale_only_rescope` is always-true (no
diagnostic rescope on paper regime), and `no_vacuous_teacher` is enforced
fail-closed (raise, not skip). The stage budgets are deterministic the same
way: the fourth gate derives them from the conjunction, so a
`paper_faithful: true` run mechanically carries the App. C `StageHyper`
triple (design.md `--paper-faithful` Migration rules the switch).

The **label** is the conjunction of ALL conditions — config-match AND stage-level.
A run with `--solver svd` gets `config_solver_is_admm: false` → label `false`,
naming the deviation (`solver = "svd"`) in the condition record. A run with
`--compute-dtype float32` gets `config_compute_dtype_is_bf16: false` → label
`false`. A run with perfect config but a vacuous-teacher event gets
`no_vacuous_teacher: false` → label `false`.

Every condition is always reported — a reader can audit which condition denied
the label from the artifact alone.

### Report field semantics

The report's `paper_faithful` field keeps its name (it is the durable label
readers already parse), but its semantics change: it is now a COMPUTED verdict
with config-match conditions in the condition set. The field is always present;
a run that never requested fidelity still gets the computed label. The
`paper_faithful_flag` condition is absent from the condition record (removed,
not just always-false).

## Operator-grade `--help`

The `argparse` description and per-flag help text are rewritten for an operator.
The structure:

```
usage: quantize_hf_model.py [-h] --model MODEL --rank R [options]

Compress a HuggingFace transformer model to sub-1-bit quantized weights.

Model and data:
  --model MODEL          HuggingFace model id or local path (default: Qwen/Qwen3-0.6B)
  --rank R               Factorization rank — the bit-budget control (default: 8)
  --seq-len N            Evaluation sequence length in tokens (default: 512)
  --tune-seq-len N       Per-block reconstruction window length; unset = --seq-len
  --calibration-seqs N   Calibration sequences, packed 8 rows/batch (default: 128)
  --eval-seqs N          Held-out evaluation sequences (default: 40)
  --seed N               Random seed for data ordering (default: 0)

Solver and quality:
  --solver {admm,svd}    Warm-start source: ADMM or truncated SVD (default: admm)
  --max-admm-iter N      ADMM iteration cap per layer (default: 400)
  --compute-dtype {bf16,float32,float64}
                         Pipeline numerics precision (default: bf16)

Output:
  --out DIR              Artifact directory (default: artifacts/model-qpk1)
  --container-format {blob,dir}
                         Export format: blob (.qpk file) or directory (default: blob)
  --scale-dtype {float16,float32}
                         Scale storage precision in the artifact (default: float32)
  --remainder-dtype {fp32,bf16}
                         Sidecar storage precision in the artifact (default: tracks compute dtype)
  --checkpoint-dir DIR   Per-block solve checkpoint directory for crash resume
  --no-checkpoints       Disable solve checkpointing entirely

Resources:
  --device {auto,cpu,cuda}
                         Where torch stages run; 'auto' takes the accelerator
                         or refuses (default: auto)
  --max-threads N        CPU thread cap for BLAS/solver parallelism
  --gpu-memory-fraction F
                         GPU memory fraction cap (default: 0.85)
  --no-good-guest        Disable automatic thread and memory tuning
```

### Help text rules

Each help string:
- Names the knob's unit (tokens, sequences, iterations)
- States the default
- Says what the knob affects (the artifact, the solver budget, memory)
- Does not cite paper sections, change ids, or lane names
- Does not use the word "paper" or "faithful"

## Non-user mechanisms — env-var discipline

Controls that survive behind env vars use the `QK_` prefix. The mapping for
the controls this change migrates:

| Removed flag | Env var | Default | Semantics |
|---|---|---|---|
| `--probe-interval` | `QK_PROBE_INTERVAL` | `1` | Blocks between progress-probe samples |
| `--probe-batches` | `QK_PROBE_BATCHES` | `2` | Eval batches in probe slice |
| `--factor-mode` | `QK_FACTOR_MODE` | `binary` | `binary` or `fp` (measurement) |
| `--ablation` | `QK_ABLATION` | `none` | `none` or `untuned-signs` (measurement) |
| `--layer-scope` | `QK_LAYER_SCOPE` | `all` | `all`, `self_attn`, or `mlp` (measurement) |

All five are driver-level reads (not `PTQConfig` fields — the config already
has `factor_mode`, `ablation`, `layer_scope`; the env vars just feed them).
Probe env vars are consumed directly in the driver's progress-probe
construction.

**Edge semantics:**

- **Empty string** (e.g. `QK_FACTOR_MODE=` with no value — the wrapper's naive
  migration `-e QK_FACTOR_MODE=$FACTOR_MODE` when `FACTOR_MODE` is unset
  produces exactly this) SHALL be treated as unset: the default applies.
- **Non-integer probe values** (`QK_PROBE_INTERVAL=abc` or
  `QK_PROBE_BATCHES=1.5`) SHALL produce the same startup rejection as a
  non-positive value, naming the invalid input.

### Wrapper migration (in scope, lands atomically)

The wrapper (`tools/run-ptq-container.sh`) currently passes `--paper-faithful`,
`--ste-flip-pass-budget`, `--factor-mode`, `--ablation`, and `--layer-scope` as
CLI arguments. When the driver parser drops these flags, the wrapper must land
in the **same commit** to prevent a no-fire window:

1. Drop `--paper-faithful` entirely (fidelity is always computed)
2. Drop `--ste-flip-pass-budget` entirely (mechanism removed)
3. Replace `--factor-mode "$FACTOR_MODE"` with `QK_FACTOR_MODE="$FACTOR_MODE"`
   in the container environment
4. Replace `--ablation "$ABLATION"` with `QK_ABLATION="$ABLATION"`
5. Replace `--layer-scope "$LAYER_SCOPE"` with `QK_LAYER_SCOPE="$LAYER_SCOPE"`

The two dead knobs leave the wrapper's ENTIRE surface, not just the argv
block (:351-367): the usage comment knob list (:15), the env parse and its
comment (:65-72), the refuse-arguments usage echo (:88), the
non-faithful-launch advisory note (:173-179 — deleted outright; post-cut the
launcher has no fidelity to announce, the report computes it), the resolved
config announce field (:183), the launch.env provenance printf (:199-200),
and the breadcrumbs run header (:425-426). A wrapper that parses, announces,
or records `PAPER_FAITHFUL`/`STE_FLIP_PASS_BUDGET` post-migration attests
knobs that do nothing — that violates its own attestation contract.
`FACTOR_MODE`/`ABLATION`/`LAYER_SCOPE` remain the wrapper-level knobs and
become the sources feeding `QK_FACTOR_MODE`/`QK_ABLATION`/`QK_LAYER_SCOPE`
in the container environment; launch.env and the breadcrumbs keep
`factor_mode`/`ablation`/`layer_scope` provenance and drop only the two dead
fields.

### CI referee consumers

Three CI scripts consume the cut surface and require migration in this landing:

- `tools/ci/report_checks.py:146`: keys the TuneFP/STE evidence detector on
  `paper_faithful_flag is True`. Must be re-keyed on the config-match
  conjunction (`config_matches_paper_regime`).
- `tools/ci/driver_smoke.py:201`: passes `--probe-batches 1`. Must migrate to
  `QK_PROBE_BATCHES=1` in the environment.
- `tools/ci/abort_leg.py:142`: passes `--probe-batches 1`. Same migration.

## `--ste-flip-pass-budget` mechanism removal

The discrete sign-flip pass (in `_block_stage.py`, gated by `cfg.ste_flip_pass_budget > 0`
and `not cfg.paper_faithful`) is removed. This is a non-paper mechanism that the
paper's path ([NQ] Algorithm 1 lines 20–23) never invokes. The removal:

1. Deletes the flip-pass code path in `_block_stage.py`
2. Removes `PTQConfig.ste_flip_pass_budget`
3. Removes the `flip_pass` key from STE `per_linear` records
4. The fidelity condition `no_discrete_flip_pass` becomes always-true — it stays
   in the condition set so existing report readers do not see a missing key,
   but it can never be the denying condition

## Config fingerprint update

The resume-safety fingerprint hashes config fields. Removing `paper_faithful`
and `ste_flip_pass_budget` from `PTQConfig` changes the fingerprint, which is
correct: a checkpoint written before this change must not resume after it
(config schema changed).

## Open questions

None at spec time. Implementation may surface:

- **OQ-1: Does the wrapper's migration break any CI lane?** The wrapper change
  is in scope and lands atomically; the CI smoke and abort legs are migrated
  in the same commit. This OQ resolves green when all CI passes.
- **~~OQ-2~~ RESOLVED (review, 2026-10-27): Are there any undocumented
  consumers of the cut flags?** Yes — the CI referee scripts
  (`tools/ci/report_checks.py:146`, `tools/ci/driver_smoke.py:201`,
  `tools/ci/abort_leg.py:142`) were missed by the initial L1 sweep. They are
  now documented and migrated in this change.

## Implementation crystallization

### 2026-10-27 — implementation

The spec left four things open that coding had to settle. Each is recorded here
so it is not re-litigated, and each names the alternative it rules out.

**C1. The regime's `device` axis is a DECLARED field on `PTQConfig`, not an env
read at the gate.** The conjunction includes `device != "cpu"`, so the four
runtime gates (which only ever see a `PTQConfig`) need the device where they
run. The alternatives were (a) reading `QUANTIBBLER_DEVICE` at each gate, which
makes the verdict depend on process state a config cannot carry and leaves three
resolvers to drift, and (b) a ninth condition that re-derives the device from
the tensors, which is not a config fact at all. `PTQConfig.device` (default
`"cpu"`, so a defaults-only unit fixture is NOT a regime run) is declared by the
driver from `--device`; the resolved device still travels separately in
`provenance.solver_device`, and the report records the DECLARED value in
`fidelity_config`, so the declaration and the placement are each named once.
Consequence worth stating: **on a CPU-only host no driver run can match the
regime** — an explicit `--device cpu` is itself a deviation — so the paper-regime
integration leg is a GPU-host leg and skips with a named reason elsewhere.

**C2. `rho_end` in the conjunction is the config field, and the driver writes
the EFFECTIVE endpoint onto it.** `QK_RHO_END` is a solve-time override, so a
conjunction reading only the field could bless a run whose solver used an
out-of-range endpoint. The driver resolves it once (the same helper the solver
calls) and passes it into `quantized_config`, so the config the gates read and
the endpoint the solver uses are the same number. Residual, named: a DIRECT API
caller that sets `QK_RHO_END` out of range and never sets the field is not seen
by the conjuction; the driver launch path is.

**C3. tasks.md 8.2's "(always present, no longer optional)" is implemented as
semantics, not as a pydantic tightening.** `QuantizeReport.paper_faithful` stays
`bool | None = None`. Making it a required field would make
`dump_report`'s validator refuse a partial payload, which is pinned behaviour:
`tests/test_qk_report_resume.py::test_a_report_written_before_the_record_existed_still_parses`
asserts that a legacy report with no such key still loads (`faithful is True`).
The repurpose the row asks for IS in: the field is documented as the COMPUTED
verdict, every exit path this build writes carries it, and both facts are now
pinned (`test_scn_no_request_flag_anywhere`, the driver legs' condition-set
assertions). Tightening the schema would have traded a named back-compat
contract for a type annotation the spec delta does not require.

**C4. The surviving-flags table omitted two flags; both survive, with operator
help.** `--gpu-memory-gb` and `--device-audit` are in the parser but absent from
design.md's "complete surviving set". Audited by the same two-level test: an
absolute per-process GPU memory cap is a real operator knob on a co-tenant host
(the fraction alone can still drain it), and a fail-closed device auditor is
what an operator debugging a device mismatch runs. Cutting a flag without a
per-flag justification is the failure this change exists to remove, so both stay
and the table in this document should gain them.

Two mechanical consequences of the re-anchoring, both measured in this landing:

- Every fixture that exercised the old `paper_faithful=True` path now stubs the
  regime verdict instead, patched on the module that HOLDS the call site
  (`ptq._block_stage`, `ptq._tuning`). Those legs run in the NumPy/float64
  parity tier, where bf16-on-an-accelerator cannot be declared, so the verdict
  is stated rather than reproduced; the verdict itself is covered by the report
  tests (both directions of every axis) and by the driver legs.
- `report_checks.stage_evidence_findings` now re-derives the conjunction from
  the report's `fidelity_config` via the shared predicate, so a report that
  cannot name its declaration cannot turn the effect floor on — the same
  fail-closed shape the old `paper_faithful_flag` read had.
