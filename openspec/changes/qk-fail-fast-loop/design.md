# Design: qk-fail-fast-loop

## Probe contract

```
probe(model: nn.Module, depth: int, name: str) -> None   # may raise to abort
```

- Invoked by the pipeline`s sequential block reconstruction (`block_recon`),
  immediately after block `depth-1` has been reconstructed and written back
  into the host model — the live host is then exactly the quantized prefix
  of `depth` blocks, the same object the driver measures at the end.
- `depth` is the number of quantized blocks now live (1-based). `name` is
  the block module name. The probe SHALL NOT mutate the model; violation
  is a spec bug in the probe, not a capability of the hook.
- Abort semantics: the probe raises (dedicated `PrefixProbeAbort` carrying
  depth, name, prefix ratio, fp-prefix ratio, and the collision line) and
  the pipeline propagates it — the run fails at the first measured
  explosion, in minutes not at minute ~90.

## Why precompute fp-prefix baselines

The fp model is frozen; prefix baselines at every depth are one forward
per eval batch per depth through the stock trunk. Doing this once at
startup costs minutes in total and buys two things: (1) the probe
compares the quantized prefix against the **fp prefix at the same depth**,
so depth-dependent PPL floor is divided out — the ratio isolates exactly
the damage quantization inflicted; (2) the comparator is the same
`qk_report.verdict` band as the end-of-run gate, so "collapsed" means the
same thing at depth 3 as at depth 28: above `COLLAPSE_LINE` (40, the
[LB2]-derived documented-collapse line).

## Decision rules and false-positive guards

- Objective: never let an exploding prefix run on; a false positive costs
  a re-run, a false negative costs another 90 minutes (the disease).
- Guards voted in: (1) `--probe-interval` defaults to 1, but any value
  must be a positive int; (2) probes begin at `min_probe_depth = 2`
  (block-0-alone prefixes are the noisiest and block-0 damage shows up
  amplified at depth ≥ 2 anyway — the forensics' compounding pattern);
  (3) the slice is the same held-out batches in the same order, taken
  from the head of the eval set (a prefix of the same evaluation, not a
  second dataset generation); (4) the abort threshold is the derived
  `COLLAPSE_LINE` with no per-run override — a researcher tuning the line
  for probe feel-good is the exact green-washing regression this family
  of changes exists to kill.
- Probe PPL cost is `probe_batches × BATCH` rows per depth; with defaults
  that is 2 of the eval batches per block. Total added cost bounded by
  roughly half a full-model PPL pass spread over the whole run — recorded
  as measured numbers in crystallization, not asserted in advance.
- **Measured expectations (coder, checkpoint mtimes, 2026-10-19):** per-block
  cost is flat across blocks — r137 median 154.6 s/block (spread 149.7–170.6),
  r512-svd 274.2 s/block (spread 216–332) — there is no cheap-early-blocks
  effect, so `--probe-interval 1` is pinned as the correct default and the
  probe has no "wait for it to get expensive" excuse. A depth-2 abort saves
  the block-loop remainder only: r137 block loop 4240 s (70.7 min) with
  first probe at ~152 s; r512 7408 s (123.5 min) with first probe at ~237 s.
  Baseline precompute, report write, and export are not short-circuited
  (export and report persistence are the e2e gate's forensics contract).
  The honest end-to-end promise is therefore **~4x faster failure on the
  r137 line, ~5.5x on r512**, not a few minutes — stated here so no one
  promises "90 minutes → 3 minutes".

## Checkpoint interplay (fingerprint decision)

`progress_probe` is excluded from the checkpoint config fingerprint:
solver state does not depend on the observer. Consequence, explicit: a
resume created without a probe, relaunched WITH one, resumes solver work
and probes every block the resumed run *processes* — restored or solved
alike, since the restore path also writes the block back into the host
and advances the prefix through it, exactly the live-prefix boundary a
probe observes; a resumed run therefore keeps its own fail-fast
protection instead of inheriting a blind spot (pinned by
`test_probe_can_be_added_on_resume_without_refusing`; the earlier wording
"remaining blocks only" was about solve work and is superseded by this
implementation-crystallized reading). The fp-prefix baseline machinery
runs full-depth once per launch regardless, since prefix baselines are
cheap and deterministic against the frozen fp model.

## Orchestrator rulings on task-5 design-silent decisions (2026-10-20)

Coder's four decisions are accepted, each now a pinned ruling with its
boundary:

1. **Aborted runs export no artifact — the report is the whole trace.**
   Correct, and distinct from the e2e gate's persistence contract: the
   e2e `export_and_report_persist_on_failure` scenario governs the
   END-of-run verdict (full artifact exists; forensics keep it). A
   mid-run probe abort at depth d leaves a d/28 artifact that the e2e
   gate would refuse anyway — exporting it is dead weight. The two
   contracts are complementary, not contradictory; this line exists so
   nobody conflates them.
2. **The abort report carries the full history so far** — a superset of
   the three pinned fields. Accepted: dead runs are exactly the ones
   forensics-starved; the pinned fields remain the minimum, history is
   the bonus.
3. **No `--no-probe` escape hatch in the CLI.** Accepted — the knob would
   be the first thing a frustrated researcher disables and the last
   thing a post-mortem finds. The deliberate full-sweep forensic escape
   remains the Python API's `progress_probe=None` (caller-supplied
   pipeline), recorded here so the next post-mortem does not rediscover
   it.
4. **Defective measurement fails closed** (missing/non-finite/zero
   pedestal → abort with both PPLs recorded) — endorsed without
   reservation: re-measuring is "measure until it passes", the green-wash
   shape this repo exists to kill.

## Discarded alternatives

1. **Driver-side block loop (move the sequential reconstruction out of
   `quantize_model`).** Rejected: it would fork the pipeline contract for
   instrumentation, duplicate capture logic, and double the surface every
   future change must keep in sync. A config callable keeps one loop.
2. **Probe at every stage (post-balance, post-STE, post-polish).**
   Rejected: mid-stage prefix PPL is not monotone (STE/polish deliberately
   trades local MSE), so mid-stage probes would false-positive constantly.
   Block boundaries are the honest measurement points.
3. **Absolute PPL tripwire instead of depth-ratio.** Rejected: prefix PPL
   starts near the full baseline for a mostly-fp prefix, so an absolute
   line calibrated for depth 28 misfires at depth 2; the same-depth fp
   baseline is the whole trick.
4. **`--max-run-minutes` wall-clock abort.** Rejected: couples correctness
   to machine load; slow-and-correct is valid, fast-and-wrong is the
   enemy, and a clock can't tell them apart.
5. **Reuse `_note_block_progress` by string-sniffing in the driver.**
   Rejected: parseable telemetry prints, not a control signal; the hook
   must be a callable contract or it will rot silently (the AST tripwire
   lesson, applied).

## Informed assumptions

1. **Block-0/deep divergence manifests in prefix PPL before the full run.**
   The forensics show the explosion localizes then compounds; a prefix
   probe at the boundary where the ratio crosses 40 catches it at the
   boundary of first manifestation. Wrong-surfaces-as: a full-run failure
   with all prefix probes healthy → becomes an end-of-run `failed` verdict
   (the existing gate) and is evidence for the algorithm workstream.
2. **Probe PPL on a 2-batch slice is a stable enough ratio estimator at
   depth ≥ 2.** The ratio divides out depth noise; residual slice noise
   costs a re-run, which we accept. Wrong-surfaces-as: repeated healthy
   probes + failed end verdict; handled by assumption 1's path.

## Open Questions

**OQ-1: should the probe also prune the *capture* of downstream blocks
after an abort?** Not needed — abort stops the pipeline; no sub-question.

**OQ-2: does the first near-ok artifact justify a tighter probe line
under the recalibration rule?** STATUS: tracking the e2e gate's
recalibration obligation — resolved there, inherited here.

## Implementation crystallization

**2026-10-19 — tasks 2–3 landed (coder): the probe decision math.**

- `qk_report.MIN_PROBE_DEPTH = 2`, `qk_report.probe_ratio(prefix_ppl,
  fp_prefix_ppl)` and `qk_report.probe_aborts(depth, prefix_ppl,
  fp_prefix_ppl)`. `probe_aborts` calls the same `verdict()` over the same
  `COLLAPSE_LINE` as the end-of-run gate — no second threshold, no second
  band. Measured: `uv run pytest tests/test_qk_report.py` → **35 passed, 0
  skipped**; full suite **62 passed, 16 skipped**, rc=0; pre-commit RC=0.
- **Design-silent choice — a defective probe measurement fails closed.**
  `probe_ratio` returns `None` when either side is missing, non-finite, or
  the fp baseline is zero (a perplexity of zero is impossible), and
  `probe_aborts` treats that `None` as abort. Rationale is the asymmetry the
  spec states: a false positive costs a re-run, a false negative costs the
  hour-plus run this change exists to kill.
- **Design-silent choice — no decision below `MIN_PROBE_DEPTH`.** At depth
  < 2 `probe_aborts` returns `False` even for a non-finite measurement,
  rather than failing closed: the spec says probes begin no earlier than
  depth 2, so no decision is being made there and a "fail-closed" `True`
  would be a claim the depth rule does not support.
- **Design-silent choice — the shared-constant rule is enforced by test, not
  by review.** `test_probe_threshold_is_the_shared_constant_not_a_copy`
  AST-scans `qk_report.py` and asserts no numeric literal `40` or `40.0`
  exists in the module, turning bug-hunter's check (c) ("grep for a second
  `40`") into a suite assertion instead of an eyeball pass.
- **Task-4 wiring shape decided now, because the e2e tripwire's pins are
  exact.** Under `2e5cb03`, `_driver_wiring_findings` allows exactly one
  `return` in `main()`, exactly two `status` assignments
  (`["verdict(ppl_ratio)", "FAILED"]`), and flags every `raise` in `main()`.
  The probe-abort path therefore does NOT take the obvious shapes — not
  `except ProbeAbort: … return 1`, not `except: write(); raise`, not a third
  `status = …`. It folds into the pinned shape:
  1. `probe_abort = None` before the `quantize_model` call;
  2. `try: qm = quantize_model(...)` / `except PrefixProbeAbort as exc:
     probe_abort = exc` — catches, raises nothing, returns nothing;
  3. on the abort path `ppl_quant = None` and `ppl_ratio = None`, so the
     single existing `status = verdict(ppl_ratio)` still yields `failed` and
     the single `return verdict_exit_code(status)` still exits nonzero;
  4. `prov = provenance(...)` moves above the branch (it is independent of
     the export); `export_model_directory(...)` is skipped on abort because
     no `qm` exists, and its call-site lineno stays below the return's, so
     the ordering pin is unaffected;
  5. the abort-path report carries `probe: {block, depth, prefix_ppl,
     fp_prefix_ppl, ratio}` and is written through `dump_report`, so an
     `inf` prefix PPL sanitizes to `null` and the unfaithful-serialization
     fallback still forces `status = FAILED`.
  Consequence for review: the tripwire's expectations do **not** move, and
  the poison battery grows the two probe-shaped bypasses (`return`-from-
  `except`, re-`raise` after the report write) in the same edit as task 4, so
  the claim is tested rather than asserted.

**2026-10-19 — task 4 landed (coder): the hook, and it executes.**

- `PTQConfig.progress_probe` (`Any`, default `None`, `compare=False`,
  `repr=False`) with constructor validation (callable or `None`), and
  `block_recon.reconstruct_model` invokes `probe(model, len(results), name)`
  after the prefix advances through the block — that is, after the block's
  write-back and before the next block's capture. An unset probe costs one
  `is not None` test and no control-flow change.
- `PrefixProbeAbort` lives in `quantibbler.ptq.config` and carries `block`,
  `depth`, `prefix_ppl`, `fp_prefix_ppl`, `ratio`. **`line` is deliberately
  NOT a constructor parameter** — it is read from the shared
  `qk_report.COLLAPSE_LINE`, so there is no argument by which a caller could
  re-pin the abort threshold. A test asserts the parameter is absent.
- Fingerprint: `checkpoints.config_fingerprint` blanks `progress_probe`
  before `dataclasses.asdict` (which would otherwise try to serialize a
  function). Verified end to end, not just as a hash comparison: a
  checkpoint directory written by a probe-free config resumes under a
  probe-bearing config with no mismatch.
- **Measured — the torch tier now EXECUTES here instead of skipping.** The
  `torch`/`hf` extras were synced into this sandbox for this verification, so
  `uv run pytest tests/ptq/test_progress_probe.py` → **7 passed in 2.8 s**
  (rc=0) and the full suite runs **121 passed, 16 skipped** (rc=0) where it
  previously ran 62 passed with the whole ptq tier ignored. The remaining 16
  skips are `tests/test_kernels_parity.py`'s Rust leg — the maturin-built
  extension, a separate `--extra kernels` environment, not a tier gate.
  `pre-commit run --files` → RC=0.
- **Design-silent resolution — a restored block IS probed.** The resume
  paragraph above says a resumed run "runs probes for the remaining blocks
  only". The implemented behaviour is that every block the pipeline
  *processes* is probed, restored or solved, because the restore path also
  writes that block back into the host and advances the prefix through it —
  so the boundary the probe observes is exactly the live-prefix boundary a
  fresh run would observe, and a resumed run keeps its own fail-fast
  protection instead of inheriting a blind spot. Pinned by
  `test_probe_can_be_added_on_resume_without_refusing` (probe fires at both
  boundaries). The design sentence is about *solve* work; it should be worded
  as probe coverage or the code changed to match — flagged for @planner.
- **Interval is the driver's business, not the hook's.** `block_recon`
  invokes the probe at every processed boundary; `--probe-interval` will be
  implemented inside the driver's probe closure (task 5), so the pipeline
  contract stays "one call per block" with no interval knowledge.

**2026-10-19 — tasks 5–6 landed (coder): the driver wiring, measured.**

- The driver measures the fp pedestal **once, before quantization**, on
  `held_out[:--probe-batches]` — the head of the same held-out batches the
  end-of-run gate uses, same order, same seed discipline. `held_out` is the
  only eval dataset the file builds; the probe slice is a slice of it, never a
  second `tokenize_chunks` call. Consequence, measured: both probed depths
  divide by the *identical* pedestal (151949.86356532486 on the tiny fixture),
  so numerator and denominator differ only by quantization damage.
- `--probe-interval` is a **schedule** knob, not a threshold knob: probes fire
  at depth 2 and every N-th processed boundary after that, inside the driver's
  closure. A large interval delays detection of a collapse by up to N blocks;
  it can never let a collapse pass. The threshold has no flag anywhere, and
  the single `probe_aborts(depth, prefix_ppl, fp_probe_ppl)` call site passes
  no keyword that could re-pin it (pinned by a lean AST test, not by prose).
- **Measured kill, real model** — `Qwen/Qwen3-0.6B`, `--rank 1 --solver svd
  --max-admm-iter 1 --seq-len 64 --calibration-seqs 1 --eval-seqs 2
  --probe-batches 1`: abort at **depth 2**, block `layers.1`, prefix_ppl
  **2,906,604.68** vs fp pedestal 83.08 → ratio **34,983.7** > 40.0. 2 of 28
  blocks processed, 26 blocks of solver work skipped, **wall 69.6 s, exit 1**.
  The pre-change shape of that run is the 70-plus-minute sweep the user
  complained about; the same failure now costs about a minute and photographs
  the collapse in the report. There is no probe-free twin to compare against
  (the spec gives exactly two knobs and probes are always on), so the
  comparison is against the historical per-block cost, not a same-config
  control.
- **Design-silent decisions, all deliberate:** (1) an aborted run **exports no
  artifact** — `probe_abort` report only, `artifact_exported: false`, because a
  collapsed qpk1 directory is dead weight the e2e gate refuses anyway; (2) the
  abort report carries the **full probe history so far**, a superset of the
  three fields the spec pins, because runs that die are exactly the runs nobody
  has forensics for; (3) a **defective measurement fails closed** — a
  missing/non-finite/zero pedestal reaches `probe_aborts` as `None`, i.e. a
  `failed` verdict, so the run aborts with both perplexities recorded instead
  of continuing blind. If re-measuring is preferred to killing on a
  measurement bug, that is a spec sentence, not a code change.
- **The tiny-fixture limit, measured — this constrains the CI smoke.**
  A random-init tiny Qwen3 cannot trip this line: rank 8 → ratios
  1.0 / 0.99999758; rank 1 → 1.0000006 / 0.99999855; and zeroing an entire
  block's `o_proj` moves the tiny model's PPL by 3 parts per million
  (151947.93 → 151947.53). The cause is structural, not a wiring fault: a
  random model's next-token distribution is already uniform (PPL ≈ vocab =
  151,949), so quantization damage has no structure to destroy — while on the
  real model the same probe moved the ratio four orders of magnitude. An
  offline random-init smoke therefore proves the **plumbing** (invocation per
  block boundary, pedestal reuse, history in the report, exit-code contract)
  and cannot prove **detection**. The abort path's runtime evidence is the
  real-model run above; a CI abort leg needs a fixture with sharp predictions
  or a real checkpoint, and the smoke job must not be credited with a
  detection test it structurally cannot fail.

**2026-10-19 — blocker cleared (coder): the abort construction moved into the lean tier.**

- `qk_report.abort_report(exc, identity_fields, history) -> dict` now owns the
  catch→report→status construction, and the driver's `except PrefixProbeAbort`
  binds `report` to that call. The reason is the review's (e), accepted: the
  wiring that delivers the fail-fast ask had exactly one execution — a manual
  real-model run — and a test that needs weights is a test that skips. The
  builder is weight-free, torch-free and IO-free, so the path is exercised by
  the lean suite on every run.
- **The guarantees are structural, not copied.** The builder writes
  `status = FAILED`, `artifact_exported = False` and the four unmeasured
  end-of-run numerics (`ppl_quant_native`, `ppl_ratio`, `effective_bpw`,
  `qpk1_layer_bytes`) as `None` *after* merging the caller's identity fields, so
  a caller that smuggles `status: "ok"` or a made-up `ppl_ratio` cannot overwrite
  them. `exc` is duck-typed rather than imported, because `ptq.config` imports
  this module for `COLLAPSE_LINE` and a back-import would be a cycle; the
  attributes are read with no defaults, so the wrong object raises instead of
  filling the report with nulls.
- **One shape, both paths** (`report_single_shape_both_paths`). The completion
  report gained `artifact_exported: true` and `probe_abort: null`; the abort
  report carries every completion key, with the end-of-run numerics as `null`.
  The spec sentence is "identical key set", so `probe_abort` is present-but-null
  on a healthy run rather than absent — a consumer reading it cannot KeyError on
  either shape. If the intent was only "the completion's keys are a subset of
  the abort's", dropping that one null is a one-line change; the literal reading
  is the one implemented.
- **Tripwire moved minimally, per the battery.** The handler must bind `report`
  to exactly one `abort_report(...)` call — construction may not drift back
  inline into a torch-importing `main()` — and a `report[...] = ...` assignment
  anywhere in `main()` is now a finding, which is the new
  `abort_status_relabelled` poison (re-labeling a built report is the bypass the
  handler checks cannot see).

**2026-10-19 — regressed and re-fixed (coder): "lean" verified against a torch-absent interpreter.**

- The first version of the builder tests was **not lean**. `_abort_exc()` imported
  the production `PrefixProbeAbort`, and importing `quantibbler.ptq.config`
  executes `quantibbler/ptq/__init__.py` → `.api` → `import torch`. Checking the
  module was not checking the package it drags in: in a genuinely lean env all
  four builder tests died with `ModuleNotFoundError: No module named 'torch'`.
  The "47 passed, all lean" number was measured with torch still synced — the
  second time this repo has mistaken "present in my env" for "lean". The tests
  now use a local six-attribute stand-in, which is the *correct* level for a
  builder that is duck-typed by design (the exception cannot be imported into
  `qk_report` without a cycle), and a new lean test pins both halves of that
  contract from source: the builder's `exc.<field>` reads against
  `abort_report`'s own source, the type's declared fields against
  `ptq/config.py` read as text. A future "fix" that re-couples the lean tier to
  torch dies at collection.
- **How "lean" is verified from now on:** build the interpreter without the
  extras and prove absence before trusting any count. `uv venv <dir>` +
  `uv pip install -e . pytest`, then `python -c "import torch"`,
  `import transformers`, `import tokenizers` and the family import must each
  exit **1**. Measured that way: `tests/test_qk_report.py` → 50 passed, 0 failed,
  0 skipped; whole suite → 77 passed, 16 skipped (only the kernels Rust leg).
  In the torch env the same tree is 157 passed, 0 skipped. Both numbers are true
  of the same commit and neither is meaningful without its environment — which
  is why the CI lean leg must assert absence, not parse a skip count.
- **The relabel pin became a rule instead of a case list** (sec-eng's remedy,
  adopted): `report` is write-only-at-binding — exactly two bindings in `main()`
  and zero writes of any other AST shape rooted at it — which covers Assign,
  AugAssign, AnnAssign, `report[...][...] = ` at any nesting, attribute writes,
  deletes, and the mutating calls (`update`/`setdefault`/`pop`/`clear`) that
  bug-hunter's two bypasses used. Named residual, deliberately not chased: a
  callee that mutates the dict it is handed is invisible to any source pin, and
  that class belongs to the runtime abort leg (status failed, exit 1, full key
  set, real run).

**2026-10-19 — binding-shape hole closed (sec-eng's finding, test-only).**

- The frozen-name rule had a hole of exactly the cheap-drift kind it exists for:
  it counted bindings only as ``ast.Assign`` and swept only
  Assign/AugAssign/AnnAssign/Delete/mutator-Calls. ``(report := abort_report(...))``
  is an ``ast.NamedExpr`` — neither a counted binding nor a matched write — and
  ``for report in ...`` / ``with f() as report`` escape identically. The walrus
  is the one that matters, because it is the *natural* refactor of the except
  path, not an exotic case to leave to the runtime leg.
- Fix: ``_rooted_in_report`` recurses through destructuring targets
  (``a, report = ...``), and the pin now treats ``NamedExpr.target``,
  ``For``/``AsyncFor.target`` and ``With.optional_vars`` as writes into the
  report. Coverage is the class, not the instance: one battery poison
  (``abort_walrus_rebind``) plus a shape-by-shape test over walrus, for, with
  and destructuring. Measured after: lean ``tests/test_qk_report.py`` 52 passed /
  0 skipped in a torch-absent interpreter, lean suite 79 passed / 16 skipped,
  torch suite 159 passed / 0 skipped, pre-commit RC=0, strict-validate RC=0.

**2026-10-19 — second binding-shape hole (bug-hunter's AnnAssign): the predicate is now unified.**

- The rule above had already been widened once for the walrus and *still* had a
  gap, which is the useful part: the binding count walked only ``ast.Assign``
  while the write sweep skipped bare-``Name`` targets, so
  ``report: dict = {"status": "ok"}`` — an ``ast.AnnAssign`` **with** a value —
  was neither a counted binding nor a matched write. That is the one-token drift
  of the exact bypass class the pin exists to stop, so it was fixed rather than
  asterisked.
- **The remedy is structural, not a fourth case:** the count and the sweep now
  derive from one predicate, ``_binds_report_name``, which walks every
  name-binding shape an AST can express — ``Assign``, ``AnnAssign`` with a value,
  ``AugAssign``, ``NamedExpr``, ``for``, ``with ... as``, and destructuring
  patterns containing the name — with the offending lines named in the finding.
  The sweep keeps the other half (writes into the *value*: subscript/attribute at
  any nesting, deletes, destructuring, mutator calls). **Invariant banked for
  future pins, per planner's ruling: a binding count and a write sweep must
  derive from the same predicate** — twice now a syntax shape has slipped between
  two independently-written halves.
- Boundary calls, stated so they can be reversed: an ``AnnAssign`` with no value
  is a bare declaration and stays legal; ``report += ...`` is caught by the count
  even though it would fail closed at runtime (a dict has no ``__iadd__``) —
  rule uniformity over the runtime argument; and the *replace* form
  (``report: dict = abort_report(...)``, still two bindings by name) is caught by
  the separate builder-binding check, so the two halves together leave no
  annotated hole.
- Coverage: battery poison ``abort_annassign_rebind`` plus the shape test
  extended to walrus, annotated, augassign, for, with, destructuring, and the
  annotated-replace form. Measured after: lean ``tests/test_qk_report.py``
  **53 passed / 0 skipped** in a torch-absent interpreter; lean suite 80 passed /
  16 skipped; torch suite **160 passed / 0 skipped**; pre-commit RC=0;
  strict-validate RC=0.
