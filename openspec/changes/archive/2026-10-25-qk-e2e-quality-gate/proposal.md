# Proposal: qk-e2e-quality-gate

## Why

Every production run of `tools/quantize_hf_model.py` to date produced a
directionally dead artifact, and the repo had no instrument that could say
so. The run ledger (all ten `artifacts/*/quantize_report.json`):

| run | effective_bpw | ppl_ratio | vs baseline 28.67 |
|---|---|---|---|
| r137 (v0) | 0.219 | 1,156,577 | dead |
| **r137-v2** | 0.219 | **943.49** | **best ever, still dead** |
| r137-v3 / v4 | 0.219 | NaN | unparseable |
| r137-v7 | 0.242 | 5,517.8 | dead |
| r137-v8 | 0.242 | 463,742 | dead |
| r512-svd | 0.779 | 524,895 | dead |
| r8 line | — | 528,419 | dead |

Four linked defects kept this invisible (all confirmed in-tree):

1. **No end-to-end PPL gate exists anywhere.** All gates bound *local*
   quantities (per-layer Frobenius bands, per-kernel cosines, tiny-model
   logit deltas). Per the divergence-guards forensics, per-layer structure
   stayed coherent while the whole-model estimate compounded into random
   logits — local parity is not a proxy for the artifact.
2. **The report mislabels its own artifact.** `regime.artifact` hardcodes
   `"qpk1: +-1 bitplanes + float16 scales"` while `--scale-dtype` defaults
   to float32, so every report used for forensics lied about the storage
   dtype, and `solver` (the only field distinguishing r512-svd) was never
   recorded.
3. **NaN never stops anything.** r137-v3/v4 wrote a bare `NaN`
   `ppl_ratio`; any `ratio > MAX` gate silently passes NaN (every NaN
   comparison is False), and `json.dumps` emits bare NaN which strict
   parsers reject.
4. **`main()` ends in an unconditional `return 0`.** Nothing could ever
   fail the process; the report had no status field, so no durable trace
   distinguishes a WARN from a pass.

The design of the gate itself follows the repo's gate-calibration rule
honestly: a band must be calibrated on production measurements — but every
production health extreme ever measured is a *failure*. There is no
healthy production extreme to calibrate against. The band edges are
therefore anchored to the pinned papers' documented collapse/functional
lines (the nearest honest geography of the sub-1-bit regime), and the
spec makes production recalibration a named, mandatory follow-up the
moment the first `ok` artifact exists.

## What Changes

- `src/quantibbler/qk_report.py` (new, torch-free): verdict classification,
  non-finite sanitizer, and the two derived band constants, unit-testable
  in a lean (no-torch) environment.
- `tools/quantize_hf_model.py`: provenance-complete report (emitted
  `regime.artifact` and `config_f64.dtype_note` from run args, top-level
  `solver` + `scale_dtype`, `status` field), strict-JSON write, and the
  exit-code flip — `main()` returns 0 only for `ok`/`degraded`.
- New suite tests covering the verdict band, the NaN/inf traps, the strict
  parse, and the exit semantics.
- Spec delta for the `qk-ptq` capability (ADDED requirements only).

## Impact

- **Affected specs**: `qk-ptq` (four ADDED requirements under
  `changes/qk-e2e-quality-gate/specs/qk-ptq/spec.md`). No MODIFIED /
  REMOVED, so no merge interaction with the active `qk-divergence-guards`
  change (different surface: this change owns the driver's *report and
  process* contract, not solver internals).
- **Affected code**: `tools/quantize_hf_model.py`,
  `src/quantibbler/qk_report.py` (new), `tests/test_qk_report.py` (new).
- **Not invariant-touching**: no change to the qpk1 container grammar,
  packing domains, kernels contract, or server. No new dependency.
- **Process impact**: from the next production run, a dead artifact fails
  the process. The qpk1 export and report are still written, so forensics
  and hypothesis-testing loops keep working.

## Non-goals

- No algorithmic quality fixes (ADMM basins, polish, KD, rank policies).
  This change refuses to *bless* dead artifacts and makes reports
  trustworthy; reaching under the band is separate work it unblocks.
- No change to `qk-divergence-guards`, `qk-flash-attention`, or
  `qk-sage-attention`; their tasks and checkbox state remain untouched.
- No CLI threshold override in this change — the collapse line is
  derived physics, not a per-run taste knob. (Revisit only if the
  production recalibration produces evidence for one.)
- No change to how `ppl_quant` is measured — the gate consumes the exact
  number the report prints, on the write-back host.
