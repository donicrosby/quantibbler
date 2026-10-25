# Design: qk-fp-model-identity

## The binding

`checkpoints.py` grows one component, computed from the live model at
fingerprint-write time:

```
model_identity(model) -> { module-name -> [dout, din, weight-dtype] , ... }
```

over the **quantized-parameter scope** — the same `_quantizable_names(model)`
predicate the solver scopes with (A-9: embeddings and output heads excluded), so
the identity describes exactly the parameters the checkpoint restores. Planned
`layer_scope` is deliberately NOT applied: scope is a config field, already
inside the config digest, and an identity that moves with it would report one
divergence twice under two names.

`fingerprint.json` records it under one key beside the config digest:

```json
"model_identity": {"digest": "<sha256 of the canonical inventory>", "modules": {...}}
```

`digest` is what refuses; `modules` is what **names** the divergence — a bare
digest can only say "different", and the operator needs "which module". This is
the fold/record distinction the module already draws: env-resolved values are
*recorded* (visible, never refusing); the model identity is *folded* (a
divergence refuses).

## Refusal semantics

`check_fingerprint(ckpt_dir, cfg, model)` keeps its existing refusals and adds
two:

1. **No identity component in the record** (a directory written before this
   change): refuse, saying the directory predates the component so its
   provenance against the live model cannot be verified. Fail closed: the
   binding cannot be reconstructed after the fact, and "cannot verify" is not
   "verified".
2. **Digest mismatch**: refuse, and name each diverged module with what
   diverged — membership (`missing=` / `unexpected=` names) and per-module
   `(dout, din, dtype)` recorded-vs-live. The config leg keeps its own wording,
   so a config divergence still reads as a fingerprint mismatch naming the
   config.

The `model` parameter is **required, not defaulted** — the same doctrine the
sibling guard's `expected_names` carries (an omitted guard is a disarmed guard):
a caller that cannot name the model it is about to quantize has no business
blessing a checkpoint directory. All three in-tree call sites pass it.

The refusal is raised **before** any archive is loaded or applied, so a refused
resume cannot have written a partial prefix into the host.

## Driver default: the ruling

Ruled here: **the default `--checkpoint-dir` becomes a fresh per-run directory**
(a `tempfile.mkdtemp` under the driver's existing scratch root — the same
`/opt/data/tmp`-else-platform-temp preference the KD cache already uses, so the
archives stay off tmpfs where real disk exists), announced in the run output;
resumption is opt-in by passing `--checkpoint-dir` explicitly.

Why this and not "keep the stable default, refuse a resume": the shared default
is the *amplifier* — the defect is reachable by doing nothing, and a refusal
only converts a silent chimera into a failed launch that the operator must then
diagnose. A per-run default removes the collision instead of reporting it. It is
also the smaller behavior surface: no new refusal path in the driver, and the
resume workflow (name the directory) is unchanged for everyone who was already
resuming deliberately.

Consequence recorded: a crash-resume of a *defaulted* launch now needs the
directory printed by that launch, not a fixed path. That is the intended trade —
the path is in the run output and in the launch trail.

## Residual risk (named, not closed)

Two models of the **same architecture** (same inventory, same `(dout, din)`,
same dtype — e.g. a base checkpoint and a fine-tune of it) remain
indistinguishable to this identity, so archives could still be restored between
them if an operator points both runs at the same directory. Binding weights, a
model file digest, or an HF repo id would each close or narrow it, and each is
rejected below. The per-run default removes the *accidental* route; sharing one
directory across same-shape models is now an explicit choice by the operator.
Reported as a residual, not as a closed case.

## Discarded alternatives

- **Bind the HF repo id / model path.** Rejected by the orchestrator's ruling and
  independently: a moved, re-cloned, or renamed local path must not invalidate
  real checkpoints, and the path says nothing about the parameters quantized.
- **Bind a content hash of the model weights.** Rejected: it closes the
  same-shape case but asks the resume path for a full-model digest it must keep
  stable across reformatting/`bf16` round-trips, and any legitimate re-save would
  refuse a genuine resume — the tripwire-not-provenance failure mode the module
  docstring already rules out.
- **Bind a content hash of the block archives** (drift refusal per write).
  Rejected in the sibling change's crystallization and again here: it is
  integrity hardening beyond this delta, it re-reads every archive on every
  resume, and version control plus the CI gates already answer "did the code
  change".
- **Fold the model identity into `config_fingerprint`'s digest.** Rejected: the
  digest is pinned to the CONFIG (plus version and `git describe`) by a standing
  contract; folding model state into it makes a config-digest mismatch
  indistinguishable from a model mismatch, so the refusal could no longer name
  the diverged component. One recorded component with its own digest gives the
  same refusal and a diagnosable message.
- **Record the identity only (never refuse).** Rejected: it is exactly the
  status quo that produced the silent chimera — a record nothing reads is not a
  guard.
- **Keep the stable driver default and require `--checkpoint-dir` when resuming.**
  Rejected in the ruling above: a refusal reports the collision instead of
  removing it.
- **Refuse `--no-checkpoints` / delete the flag.** Out of scope: the flag is a
  documented ablation and is not a sharing hazard.

## Open questions

None outstanding. The two that were open are resolved above and recorded here so
they do not reopen:

1. *Identity scope: quantizable set, or the planned per-run `layer_scope`?* —
   RESOLVED: the quantizable set; scope lives in the config digest.
2. *Legacy directory (no identity component): trust or refuse?* — RESOLVED:
   refuse, fail closed, with the reason named in the message.

## Shared-state audit of the resume surface (task row 5)

Every default the resume path reads, and its disposition:

| default | disposition |
| --- | --- |
| `--checkpoint-dir` → `artifacts/ptq-checkpoints` | **FIXED** — the amplifier; now a fresh per-run directory (ruling above). |
| `QUANTIBBLER_KD_TMPDIR` → `/opt/data/tmp` else platform temp | Audited clean — a shared *root*, but each run creates its own `TemporaryDirectory` inside it; it is a teacher-logit cache, never a restore source. Now shares one `_scratch_root()` definition with the checkpoint default instead of restating the expression. |
| `QUANTIBBLER_BREADCRUMBS` → `breadcrumbs-<pid>.log` | Audited clean — per-launch by construction (pid), and deliberately honored exactly when set. |
| `QK_SOLVER_PROBE_PATH` → `/out/solver_probe.jsonl` | Audited, out of scope — a shared append ledger, but it is a *measurement* ledger (rows carry `(value, source)` and the run's own device/backend), it restores nothing, and its container path is a deliberate fail-closed contract: an unwritable path raises rather than measuring nothing silently. |
| `--out` → `artifacts/model-qpk1` | Audited, out of scope — a shared artifact destination, but it is written once at the end of a successful run (last writer wins); it is not read back by a resume, so it cannot contaminate a solve. |
| `SOLVER_BACKEND` / `QUANTIBBLER_DEVICE` | Audited clean by design — env-resolved and RECORDED in `fingerprint.json` with its source, deliberately never refusing (a checkpoint that bricks because an env var moved is a tripwire, not provenance). |

## Implementation crystallization

### 2026-10-25 (coder, ticket t_08c2d9e9)

- **RED first, on the landed bytes** (`/opt/venv`, torch 2.14.1+cpu, no
  accelerator; both probes archived with the ticket). Probe 1: `check_fingerprint`
  **PASSED** for a second model against the first model's directory, and the
  end-to-end resume then died on the anonymous tensor-size error. Probe 2 (the
  silent variant): model B — same architecture class, different quantizable scope,
  different weights — resumed model A's directory with the solver rigged to raise
  if called, **completed**, printed `[checkpoint] block 1 (lin1): restored from
  checkpoint, skipping solve`, solver calls `0`, and produced an artifact that
  differed from a fresh run of B. After the change both refuse by name, before any
  archive is read or applied: `model identity diverged: recorded but absent from the
  live model: lin2`.
- **`model_identity` is `{digest, modules}`, one folded component.** The digest is
  what refuses; the inventory is what names the divergence. Folding it into
  `config_fingerprint`'s digest instead was rejected (discarded alternatives): the
  config digest is pinned to the config, and a single digest could not say *which*
  component diverged.
- **Required, not defaulted.** `model` is a required argument on
  `write_fingerprint` / `check_fingerprint` / `read_stage_log` /
  `resume_stage_events` — the same rule the sibling guard's `expected_names`
  carries. Every in-tree call site was updated in the same commit.
- **The legacy refusal is a format boundary, not a drift tripwire.** A record with
  no `model_identity` cannot be verified against a model that was never named at
  write time, so it refuses with that reason. It is deleted (or given to a fresh
  run) exactly as a stale-config directory is; no migration is attempted, because a
  reconstructed identity would be an assertion, not a verification.
- **Rebased mid-flight onto a moving shared tree.** While this change was in
  development the stage-journal lane moved the resume-seeding out of
  `reconstruct_model` into `api.quantize_model` (new `resume_stage_events`) and
  dropped `stage_sink` from `reconstruct_model`. The change was rebased onto those
  bytes: the model is threaded through `resume_stage_events` at the new call site in
  `api.py`, and `reconstruct_model` carries only the two fingerprint call sites.
  The journal semantics are the sibling lane's; this change only makes the read they
  guard model-bound.
- **Collateral, declared: one sibling-owned test file.** `tests/ptq/test_stage_journal.py`
  (untracked, the stage-journal lane's) calls `read_stage_log` at six sites; the
  required `model` argument broke all six. They were updated in this change's patch
  (a `_identity_model()` fixture helper plus the six call sites) because leaving a
  known-broken caller in the tree would ship a red suite. The change is mechanical
  and does not alter the journal behavior that file tests.
- **No foreign lines.** An earlier draft of the diff carried `ruff format`'s reflow
  of a `magnitude_balance(...)` call authored by another lane; the rebase dropped it
  (that lane had already reformatted the line in the shared tree), so this patch
  touches only lines this change authors. **hotspot** flagged on the board for
  `tools/quantize_hf_model.py` / `block_recon.py` regardless.
- **Gate suite (task row 9), first complete `tests/ptq` denominator for this
  change.** One subprocess per test file (`/opt/venv`, torch 2.14.1+cpu, no
  accelerator, `SOLVER_BACKEND=numpy`, load 61): 42/42 files, 255 passed / 1
  failed / 3 skipped / 4 deselected. The failure is the load shape, not the diff
  — the driver child in
  `test_probe_knobs_reject_non_positive_values_at_parse_time[--probe-interval--2]`
  was SIGTERM'd (`returncode -15`, empty stdout/stderr) at parse time, and re-ran
  green in isolation on the same bytes (`3 passed` RC=0); the sibling `value=0`
  case passed in the same run. `test_driver_stage_journal.py` exits 5 because its
  single test is integration-marked and deselected by the default filter — that is
  the marker filter, recorded so "42 files" is not read as "42 green". The
  directory-tier denominator the first handoff had to report as unmeasured is
  therefore closed; raw log in the ticket's delivery bundle.

## Informed assumptions

- **The shape/inventory digest is the right identity** (orchestrator ruling,
  2026-10-25: "shape digest is the identity that matters"). Rationale: it
  distinguishes every model whose **quantized parameters** differ in what a
  checkpoint restores (which module, which shape, which dtype), while staying
  stable across path moves, renames, and re-materialization of the same
  architecture. *How a wrong assumption surfaces:* the residual above — a
  same-shape fine-tune restores silently from the other's archives; the per-run
  default makes that an operator-invoked case rather than a defaulted one.
- **`torch` weight dtype is stable between the fingerprint write and a resume of
  the same run.** Rationale: the driver loads and casts the model before
  reconstruction begins, so both legs observe the same dtype; the digest is
  computed from the live module. *How a wrong assumption surfaces:* a dtype-
  carrying pipeline change would turn a legitimate resume into a named refusal
  (`lin1 dtype recorded torch.bfloat16, live torch.float32`), which is loud and
  points at the inconsistent leg rather than silently mixing artifacts.
