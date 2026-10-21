# Tasks: qk-root-cause-ablation

## 1. Harness (coder)

- [x] RED-first: `tests/ptq/test_root_cause_ablation.py` — **19 tests, 19
  failed**, every row for a missing-feature reason (`PTQConfig` fields absent,
  `_factor_terms` absent, new kwargs rejected, recorder payload absent); two
  fixture bugs were corrected before the clean RED transcript was recorded.
- [x] Library GREEN: `PTQConfig` fields (`factor_mode`, `ablation`,
  `layer_scope`, choices-validated and fingerprinted); one shared helper
  consumed by all THREE sites; fp identity in both regimes; binary grad/no-grad
  contract; fp skips polish; `untuned-signs` skips polish + STE; layer scope;
  conversion-site refusal; continuous host write-back with no fabricated binary
  layer payload. **21/21 new tests, full suite 470 passed, lean 364/16.**
- [x] Driver GREEN: flags + choices, both report shapes carry provenance, fp
  export backstop, `artifact_exported: false`, `effective_bpw: null`, report
  survives, and scale calibration is proved uncalled. Tiny fp driver completes
  with report-only output; an initially inverted export guard was caught and
  fixed before handoff. **Full 472 passed; lean 364/16; pre-commit RC=0.**
- [x] Review closure: `_set_linear` helper reachability is spied independently
  of bit-equality; a detached-alias negative row demonstrates the AST pin's
  residual and proves the shipped helper by real backward gradient.
- [x] OOM RED->GREEN: 3 failures / 1 parity pass before fix; streamed f64
  sum-of-squares + token counts only for `chunk_rows != None`; streamed vs
  reference within `3.9e-15` at 512 chunks against the `1e-12` budget; default
  f64 construction unchanged. Bug-hunter's append/branch/alias poison battery
  caught all three valid poisons. Exact real geometry crossed the former wall:
  `stats_done` at 6.44 GiB, 12.12 GiB peak versus attempt 1's 31.07 GiB kill.
- [x] Gates on the complete post-streaming tree: full **476 passed**, lean
  **364 passed / 16 skipped** with absence discipline, ablation **27 passed**,
  pre-commit RC=0, yamllint RC=0, strict-validate valid.

## 2. Review of the harness (bug-hunter)

- [x] All three sites share one helper in BOTH modes; the two live paths are
  spied, write-back is also bit-equality checked, and the real-backward
  analytic gradient contract — not the AST pin alone — catches dead STE.
- [x] `untuned-signs` really runs zero refinement (asserted on recorder
  events); scoped runs leave unscoped weights byte-identical; `fp` cannot reach
  export, and the conversion boundary hard-fails rather than fabricating bits.
- [x] Default-behavior regression proves unset knobs retain today's pipeline;
  bug-hunter reviewed the harness CLEAR and separately cleared D-7 after the
  three-shape poison battery.

## 3. Threat pass (sec-eng)

- [x] Measurement/artifact boundary: the fp abort report names every knob,
  writes `artifact_exported: false`, `effective_bpw: null`, and no qpk1 layer
  payload; conversion and directory-export boundaries both refuse fp mode.

## 4. Rung 1 measurement (coder, container per launch protocol)

- [x] Attempt 1 announced and fired at rank 512 / svd / seq-2048 /
  tune-window-512 / fp. It produced no depth-2 curve: Docker emitted `oom`,
  kernel memcg killed Python at 33,480,576 KiB, breadcrumbs peaked 31.07 GiB
  in `layers.0` immediately after TuneFP, exit 137. Root cause is the chunked
  `collect_stats` hook retaining 44.0 GiB of full f64 activations (D-7), not a
  numerics result.
- [x] D-7 retry used the identical announced line and crossed the former memory
  wall (`stats_done` 6.44 GiB; peak 12.12 GiB). It then **failed at depth 2**:
  `layers.1`, prefix PPL **2,049,545.180910646** against its own fp-prefix
  pedestal **18.88104607051466**, ratio **108,550.40410665018x** (within-run
  geometric mean **329.46988345924757x/block**), abort line 40, wall 357.2 s.
  Report provenance says `factor_mode=fp`, `ablation=none`, `layer_scope=all`,
  seq 2048/tune 512; `artifact_exported=false`, `effective_bpw=null`, status
  `failed`; checkpoint payloads are empty and no artifact exists. Registered
  interpretation: the explosion persists without signs, so the binary sign
  constraint is **not** the kill site; the defect is upstream in continuous
  factorization/scale/write-back. No cross-geometry comparison is claimed.

## 5. Rungs 2-4 measurements (coder, sequenced after rung 1 lands)

- [x] `untuned-signs` at the same geometry: depth 2 (`layers.1`), prefix PPL
  **1,599,091.605664219** vs fp pedestal **18.88104607051466**, ratio
  **84,692.95608368964x** (within-run geometric mean **291.0205423740559x/
  block**), wall 211.2 s, status `failed`. Zero polish and zero STE made the
  failure 1.28x less extreme than rung 1 — the two continuous-stage rungs are
  the same class, so refinement neither creates nor removes the explosion.
- [x] `layer_scope=self_attn` at the same geometry: depth 2, prefix PPL
  **4862.631823821083**, ratio **257.54038233160975x** (16.04806475347136x/
  block), wall 333.0 s, status `failed`. Attention-only quantization still
  exceeds collapse line 40. Its magnitude is not compared causally with the
  all-scope rungs because their factor/ablation knobs differ; the paired MLP
  run selects the layer-family branch.
- [x] `layer_scope=mlp` at the same geometry: depth 2 ratio
  **1.8699355654649084x** (ok band), then abort at depth 3 (`layers.2`):
  prefix PPL **1279.2058956388878**, ratio **67.75079573777128x** (marginal
  depth-2 to depth-3 multiplier **36.23162048416727x**), wall 480.9 s,
  status `failed`. Both layer families individually exceed the collapse line,
  so the defect is not specific to attention or MLP; MLP needs one more block
  to cross it. The exact announced geometry was already running under planner
  session `proc_e450f3f44e64`; its report landed at
  `/opt/data/tmp/rca-rung3-mlp/quantize_report.json`. The announced
  `/opt/data/tmp/rca-rung4-mlp` target was not fired a second time and does not
  exist.
- [x] Every run's report and complete probe history are recorded above;
  arithmetic and fail-closed provenance were independently audited. Verdicts
  are measurements -- expected `failed`, never tuned against (D-5, design.md).

## 6. Orchestrator

- [x] Strict validation and local gates passed on the landing bytes; commit
  `8dfab5c` was pushed to `main`, remote HEAD read back exactly, and GitHub
  Actions run `39536090909` completed **7/7 success**.
