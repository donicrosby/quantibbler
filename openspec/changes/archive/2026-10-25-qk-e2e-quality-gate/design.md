# Design: qk-e2e-quality-gate

## The band (pinned arithmetic, asserted in suite)

- `COLLAPSE_LINE = 40.0 = ceil(241.96 / 6.10)` — [LB2] Sec. 5.2 Table 1,
  Llama-3-8B, WikiText-2: STBLLM's *documented method collapse* at
  0.55 bpp (PPL 241.96) against the FP16 reference (PPL 6.10). A run at or
  beyond a documented collapse of a method in the same regime is a failed
  run. Verified in-tree: historical best ratio 943.49 is ~24x this line.
- `OK_LINE = 4.0 = ceil(23.74 / 6.10)` — [LB2] Sec. 5.2 Table 1: the
  *functional* envelope at the extreme end of the budget range
  (LittleBit-2 at 0.1 bpp keeps PPL 23.74, ~3.89x). Ratios at or below
  this are the "working model" bar.

Verdict (deterministic, no threshold comparison may precede the
missing/non-finite check):

| condition | verdict | process exit |
|---|---|---|
| ratio missing or non-finite | `failed` | nonzero |
| `ratio > COLLAPSE_LINE` | `failed` | nonzero |
| `OK_LINE < ratio <= COLLAPSE_LINE` | `degraded` | 0 |
| `ratio <= OK_LINE` | `ok` | 0 |

Replay against all ten archived reports: every run classifies `failed`
(v0 1.16e6, v2 943.49, v3/v4 NaN, v7 5.5e3, v8 4.6e5, r512-svd 5.2e5,
r8 line 5.3e5). The gate would have refused every artifact ever shipped —
which is the point.

## Calibration-rule debt, declared

The repo's gate-calibration rule says: calibrate on production
measurements and pin a suite test asserting the honest extremes *pass*.
Here every production extreme is a failure, so asserting healthy extremes
would be fabricated comfort. This design instead:

1. anchors both edges to the pinned papers' own collapse/functional
   measurements (explicit arithmetic above, asserted in `tests/`), and
2. makes recalibration a spec-level obligation: when the first `ok`
   artifact lands, its ratio re-pins the band per the calibration rule
   (with margin arithmetic on both sides), as a spec update in the same
   change that ships it.

## Implementation crystallization

**2026-10-19 — tasks 1–3 landed (coder).**

- **Module split realized as designed.** `src/quantibbler/qk_report.py` (new,
  torch-free) holds `OK_LINE`/`COLLAPSE_LINE` (derived, not typed),
  `verdict()`, `exit_code()`, `json_safe()`, `provenance()`.
  `uv run pytest tests/test_qk_report.py` → **16 passed, 0 skipped** in the
  lean env (no torch installed), and `uv run pytest` → **43 passed, 16
  skipped**. The no-skips property is the point: in the previous revision the
  same sanitizer lived inside the torch-importing driver and was exercised
  only in a REPL transcript.
- **Design-silent choice — provenance is a returned dict, not driver string
  formatting.** `qk_report.provenance(solver, scale_dtype)` returns
  `solver`, `scale_dtype`, `regime_artifact`, `dtype_note`; the driver
  merges it and no longer constructs those strings. This is what makes
  `report_mirrors_run_args` / `report_fp16_opt_in_named` suite-testable
  without torch, and it makes "the report cannot disagree with the run it
  describes" a single-place property instead of two f-strings in the driver.
- **Design-silent choice — driver wiring is pinned by a source-level AST
  assertion.** `tests/test_qk_report.py::test_report_driver_is_wired_to_the_gate`
  parses `tools/quantize_hf_model.py` (unimportable in the lean env) and
  asserts exactly one `return` in `main()`, returning
  `verdict_exit_code(status)`; `status = verdict(ppl_ratio)`; the absence of
  `_json_safe`/`_report_status`; and the absence of both hardcoded literals
  (`"qpk1: +-1 bitplanes + float16 scales"`, `"scales fp16"`). Measured
  before the change: exactly one return, an unconditional `return 0`.
- **Measured exit ordering.** The qpk1 export and the report write both
  precede the single `return verdict_exit_code(status)`, so a `failed` run
  still leaves a forensic artifact — the requirement's
  `export_and_report_persist_on_failure` scenario holds by statement order,
  not by a guard that could be skipped.
- **Design-silent choice — `verdict()` treats non-numeric input as
  missing.** `math.isfinite` raises `TypeError` on non-numbers; that is
  caught and classified `failed`, because a non-numeric measurement is a
  defective measurement, never a good one.
- **Known limitation, stated plainly.** The driver's runtime path needs
  `--extra torch --extra hf`, which are not installed in this environment and
  no model weights are available offline, so the end-to-end driver run was
  not executed here. Its wiring is covered by the AST assertion above; the
  verdict, sanitizer, band arithmetic and exit semantics are covered by
  runtime tests in the torch-free module. `openspec validate` and the commit
  remain the orchestrator's.

**2026-10-19 — task 5 landed (coder): review findings S1 + S4.**

- **S1 — the tripwire now rejects what it exists to reject.** The wiring
  check was factored into `_driver_wiring_findings(src)`, which returns every
  contract violation it can find: exactly one `return` in `main()`, and it
  must be `verdict_exit_code(status)`; the `export_model_directory(...)` call
  and the report `write_text(...)` must both sit at a **greater** lineno than
  that return (forensics survive the gate — the regression a shape-only check
  let through); no `exit`/`quit`/`_exit`/`abort` call and no `raise
  SystemExit` anywhere in `main()`; the `quantibbler.qk_report` import must
  bind `verdict`, `verdict_exit_code` and `dump_report`; and no module-level
  def or assignment may shadow a gate name.
  `test_report_driver_tripwire_rejects_poisoned_drivers` applies each bypass
  sec-eng demonstrated to the **real** driver source and asserts it is
  caught: `raise SystemExit(0)` before the export, `sys.exit(0)`, a
  shadowing `def verdict`, a return hoisted above the export, a status
  pinned to a literal, and a *nested* `verdict = lambda r: "ok"` inside
  `main()` — bug-hunter's named residual, which escapes the module-level
  pass (a nested assignment is not a tree-level def) and the return-count
  pin (a lambda body is not an `ast.Return`). A guard that has never been
  shown rejecting anything is not a guard.
- **S4 — the report write can no longer cost the verdict.**
  `qk_report.json_safe` now converts numpy scalars first via
  `np.generic.item()`: `np.float32(np.inf)` is not a Python `float`, so it
  walked past the `isinstance(obj, float)` check straight into
  `json.dumps(..., allow_nan=False)` and raised — no report at all, on
  exactly the weirdest runs. New `qk_report.dump_report(report) ->
  (payload, faithful)` wraps serialization; on any `TypeError`/`ValueError`
  it returns the minimal `{"status": "failed", "report_error": ...}` payload
  with `faithful=False`, and the driver forces `status = FAILED` whenever
  `faithful` is False, so the durable trace and the exit code can never
  disagree.
- **The status assignment is pinned too.** The tripwire asserts the status
  assignments in `main()` are exactly `["verdict(ppl_ratio)", "FAILED"]` —
  the verdict, plus the serialization-failure fallback — so the S4 branch
  cannot be dropped silently either.
- **Measured (final, after the nested-shadow hardening):** `uv run pytest`
  → **54 passed, 16 skipped**, rc=0;
  `uv run pytest tests/test_qk_report.py` → **27 passed, 0 skipped**;
  `set -o pipefail; uv run pre-commit run --files …` → **RC=0** (re-run after
  the hooks reformatted; never piped through a filter).

## Discarded alternatives

1. **Pin MAX from the v2 baseline (943).** Rejected: v2 is 24x past a
   documented collapse; a gate built to admit it encodes a dead artifact
   as the boundary — the exact post-mortem-cycle mistake this change
   exists to break.
2. **Absolute PPL ceiling (e.g. `ppl_quant < 1000`).** Rejected:
   dataset- and model-specific, not portable to the next model/dataset
   pair; the ratio is the papers' own normalized comparison and what the
   report already prints.
3. **Refuse to write the artifact on failure.** Rejected: the whole
   forensic culture of this repo (and this very diagnosis) depends on
   exported evidence from failed runs. The gate fails the *process*, the
   evidence is always kept. Status in the report is the durable verdict.
4. **Binary ok/failed only.** Rejected: the functional envelope is
   QAT-grounded ([LB2]) while this pipeline is the PTQ reimplementation
   ([NQ]); PTQ may land in between the lines for a while. `degraded`
   records "alive, not yet working" honestly instead of moving the goalpost.
5. **CLI threshold override.** Rejected for this change: an override that
   relabels failures under a raised bound reintroduces the green-washing
   this gate exists to prevent. Revisit only with recalibration evidence.
6. **Keep helpers inside `tools/quantize_hf_model.py`.** Rejected: the
   driver imports torch at module top, so suite coverage of the verdict
   would silently skip in the lean env (the sanitizer already lived
   REPL-only once; it must not again). Extracting the torch-free module
   makes the tests run everywhere.

## Informed assumptions

1. **Ratio transfer across model family and context length.** The anchors
   (Llama-3-8B, WT2 seq 2048) move to Qwen3-0.6B, our in-repo WT2 seq 512
   eval. Rationale: ratio is the normalization that makes cross-scale
   comparison meaningful at all — it is the papers' own headline form.
   Wrong-surfaces-as: an artifact with `ok` verdict that is visibly
   degenerate on other evals; handled by the mandated recalibration.
2. **Paper functional envelope ~4x is reachable-or-degradable.** PTQ
   ([NQ] path) may not reach the QAT envelope immediately; that is what
   `degraded` is for. Wrong-surfaces-as: no verdict besides `failed` for
   an extended period → informs the algorithm workstream, not the gate.
3. **NaN/inf is failure with no known-innocent cause.** A non-finite
   ratio is a measurement or numerics defect in the artifact path; there
   is no legitimate "successful" run that produces one.

## Known limitations

1. **The gate certifies the host weights, not the served artifact.**
   `ppl_quant` is measured through the in-memory HF module after
   write-back. Everything between that object and the bytes the server
   loads — qpk1 packing domains (they coincide only when `r % 8 == 0`),
   norm-sidecar inventory, lm_head tying, engine forward conventions —
   is outside the measurement. Not re-importing is the right
   comparability call, but `status: ok` must never be read as "the
   served artifact works". The served-artifact gate is a separate,
   future obligation: the torch-tier tiny-`Qwen3Config` driver smoke
   through the real export path, plus a one-time server-side
   load-and-generate check on the first near-ok artifact before the
   pipeline is called working.

## Open Questions

**OQ-1 (raised in bug-hunter review): exit code for a `degraded` verdict.**
RESOLVED 2026-10-19 (orchestrator): `degraded` keeps exit 0. Rationale:
the exit code gates *process survival* — a dead artifact must stop the
chain — not doneness. `degraded` means "alive, not yet at the working
envelope", and during the PTQ climb nearly every forward step will start
there; nonzero exits for it would paint the research loop permanently red
and re-teach red-ignoring, the exact disease this change exists to cure.
The durable doneness distinction is the report's `status` field, which
this change makes machine-readable. Load-bearing consequence, accepted
(and restated on sec-eng review 2026-10-19): any consumer of the driver's
exit code **or its artifacts** — promotion wrappers, `&&`-chained landing
gates, cron jobs alike — MUST parse `status`; that consumer-level
requirement SHALL land in the same change as the first such consumer,
and must not silently inherit the driver's exit code.
