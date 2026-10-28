# Proposal: qk-cli-sane

## Why

The driver CLI (`tools/quantize_hf_model.py`) carries flags a human operator would
never write: `--paper-faithful` (a self-assertion flag that the report then
verifies — a bot asking itself whether it followed the paper), `--ste-flip-pass-budget`
(a deviation knob for a non-paper mechanism), `--probe-interval` / `--probe-batches`
(progress-monitoring knobs for lane automation), and `--factor-mode` / `--ablation` /
`--layer-scope` (measurement-only controls for root-cause ablation).

The user ruling (2026-10-25) is explicit: "Remove the paper faithful crap (that
is not a fucking cli argument a human would ever write)… If it's something a bot
added to make their life easier and a human would never touch it, cut it."

The fidelity ladder must re-anchor from the self-asserted flag to a computed
verdict: the report compares the pipeline's declared configuration against the
paper regime and derives fidelity conditions from measurable facts, never from
a flag the user set. The surviving help text must read as an operator surface:
every flag names its unit, its default, and its effect on the artifact.

## What Changes

1. **Cut seven flags** from the driver parser. Each cut is justified in design.md
   with a two-level gate: symbol-level usage grep across the tree, and
   requirement→mechanism grep proving the flag is bot-authored, not operator-facing.

2. **Re-anchor the fidelity ladder.** The report's `paper_faithful` label (today
   keyed on the self-asserted `--paper-faithful` flag) becomes a computed
   verdict from the pipeline's declared config compared against the paper
   regime. The `paper_faithful_flag` condition is dropped from the condition
   set; the label is always computed and reported. A config that deviates
   from the paper regime names the deviation — it never silently claims
   fidelity.

3. **Operator-grade `--help`.** Every surviving flag carries help text written
   for an operator running a compression: the flag's unit, default value,
   effect on the artifact, and any coupling with other flags. Flags are
   grouped by function (model/data, solver/quality, output, resources).

4. **Surviving lane hooks move to env vars.** `--probe-interval`, `--probe-batches`,
   `--factor-mode`, `--ablation`, and `--layer-scope` become environment
   variables with the `QK_` prefix. The driver reads them at startup; the
   parser never exposes them. The `--ste-flip-pass-budget` mechanism is
   removed entirely — it was a named deviation from the paper that the
   paper's path never invokes.

5. **Spec delta.** MODIFIED requirement for probe configuration (flags → env
   vars). ADDED requirements for config-derived fidelity, operator-grade
   help text, and non-user mechanism discipline.

## Impact

- **Specs modified**: `qk-ptq` (1 MODIFIED, 4 ADDED requirements)
- **Code touched**: `tools/quantize_hf_model.py` (parser, stage-budget re-key, eval-window guard removal), `src/quantibbler/qk_report/stages.py` (fidelity conditions), `src/quantibbler/ptq/config.py` (config fields), `src/quantibbler/ptq/_block_stage.py` (flip-pass removal, two runtime gates), `src/quantibbler/ptq/_ste.py` (flip-pass definition deleted), `src/quantibbler/ptq/block_recon.py` (flip-pass re-export deleted), `src/quantibbler/ptq/_factorized.py` (refusal text names `QK_FACTOR_MODE`), `src/quantibbler/ptq/_tuning.py` (vacuous-teacher abort), `src/quantibbler/serialization.py` (ledger paper_faithful repurposed)
- **Wrapper**: `tools/run-ptq-container.sh` — wrapper pass-throughs and attestation surface migrate from cut CLI flags to `QK_*` env vars in this landing (atomic with parser change); `PAPER_FAITHFUL`/`STE_FLIP_PASS_BUDGET` dropped from entire wrapper surface per design
- **CI**: `tools/ci/report_checks.py` — paper_faithful_flag detector re-keyed on config-match conjunction; `tools/ci/driver_smoke.py`, `tools/ci/abort_leg.py` — `--probe-batches` migrated to `QK_PROBE_BATCHES` env var
- **Artifact format**: unchanged — the report gains fidelity conditions but the JSON schema is backward-compatible (conditions always emitted, label always computed)
- **Suite**: tests referencing cut CLI flags are updated; RED-first per the test discipline

## Non-goals

- Does not alter solver, stage, or numerics behavior on paper-regime runs — the
  four runtime gates re-keyed on the config-match conjunction preserve
  paper-regime semantics exactly (polish off, paper_ste declared,
  vacuous-teacher abort fail-closed, App. C stage budgets)
- Does not add new CLI flags beyond those already in-flight from `qk-pipeline-dtype`
- Does not remove the fidelity CONDITIONS from the report — only the self-assertion flag that requested them
