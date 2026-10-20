# Design: qk-nq-recreation-bar

## The bar (pinned arithmetic, asserted in suite)

Source: [NQ] Sec. 4.2 Table 2 (textually present rows for Qwen3-0.6B in
the pinned paper copy under docs/papers/):

| row | value | ratio vs 12.66 | 1.15x tolerance |
|---|---|---|---|
| FP16 baseline | 12.66 | 1.000 | — |
| NanoQuant @1.00 bits | 27.56 | 2.177 | 31.69 |
| NanoQuant @0.80 bits | 33.79 | 2.669 | 38.86 |
| NanoQuant @0.55 bits | 52.94 | 4.182 | 60.88 |

`PPL <= 1.15 * row`, at `effective_bpw <= row_bits` (byte-derived,
same accounting the report already prints). All four numbers and the
tolerance factor are constants, and a suite test asserts the arithmetic —
including that the row constants themselves are not recomputed at runtime
(the copy-vs-derivation audit, same discipline as the band edges).

**A bar claim is valid only at an existing row's bit budget.** Below
0.55 bits there is no [NQ] landing site: the r137 production line
(0.219 bpw) has no row and never will under "recreation" — sub-0.55
artifacts are beyond-paper research, publishable on their own terms but
barred from the word recreation and from any status the ledger would
lend them.

## Tolerance rationale

15% over a published number is already generous for a "recreation" and
takes no stand on implementation-language noise (fp64 solver vs the
paper's stacks, checkpoint-dtype differences). **Status: PROVISIONAL.**
It is the one unpinned number in the finish line: it has no paper source
and no healthy-run eval variance to derive from (the same bind the band
was in, resolved there by anchor + recalibration). Obligation, recorded
here so it cannot be forgotten: at the first artifact landing within 2x
of a targeted row's value, the tolerance SHALL be re-derived from that
artifact's measured run-to-run eval variance (same artifacts, repeated
runs) and re-pinned as a spec change; until then 1.15 stands only as the
pre-registered provisional. Any relaxation remains a spec change with
its own justification — never an edit to a ledger.

## Protocol honesty

[NQ] measures WT2 at seq 2048. The driver's current eval is seq 512.
The acceptance run SHALL execute at seq 2048 over the same WT2 test
partition family the paper used; the seq-512 numbers remain the
operational gate's feed and are NOT admissible as the finish line. If a
sequencing equivalence argument is ever claimed (512-slice scores
equivalent to 2048), it is recorded IN the acceptance ledger with both
number sets, and the 2048 run still executes.

## Run-geometry constraint (measured 2026-10-27)

**Lane ordering (planner, 2026-10-27):** this change's first
implementation tasks are the tune/eval window split — the seq-2048 run
is the thing that converts any of this into evidence; the constants
test and ledger follow.

The NQ seq-2048 line cannot pass the current CPU tune path. `_tune_fp`
is an Adam training loop that calls `loss.backward()`, so the NxN
attention tensor is a RETAINED autograd node, not one transient.
Measured at seq-512: a single fp32 N^2 tensor of 2.00 GiB produced a
+13.6 GiB tune transient (4.81 -> 18.38 -> 10.84 G) — a ~7x retention
factor. Scaled to seq-2048 (N^2 tensor 32 GiB), the tune requirement
brackets 100-200 GiB; the first seq-2048 attempt died ~2 min in at the
first tuning block under the 32g cap. Death record (reconciled
2026-10-27, wording per bug-hunter): the kernel-provable kill is
container `2d2d4b6df1e0`, 15:56:27Z, 33,480,724 kB against the 32g
memcg, launcher unknown and unestablishable post-mortem (--rm'd, no
journal, events not persisted); the room-launched nq2048 run died 43 s
later by userspace SIGKILL (last crumb 30.14G at 15:57:20Z, no kmsg
line; systemd-oomd class, unconfirmable from the sandbox). The plateau
is multi-day: four python memcg kills at essentially identical RSS
(33.48 GB) on Oct 18 (x2), Oct 19, and Oct 27. The constraint depends
on neither identity. Consequence ranking: (a) interim geometry — tune on
512-token windows, eval at 2048 — preserves the number the bar reads
(the eval PPL) at the protocol's geometry and needs no cap change; the
forward-only paths are measured fine at 2048 (fp pedestal 19.5277,
probe baseline 18.8810, both under the 32g cap); (a2) fused attention
inside the torch tune loop (SDPA mem-efficient/flash backend, or pip
flash-attn as backend in the rocm container) — candidate, small,
unblocks seq-2048 tune under the existing cap with no new kernel;
needs its own change; (b) cap-raise is
FALSIFIED by measurement — 64g sits below the low end of the bracket,
a coin flip you pay for with another 2-minute death; (c) the GPU
attention face — the real fix for the SERVER's decode memory, not the
tune loop — user-prioritized 2026-10-27 with the pip-benchmark bar
(qk-flash-attention task 4.2). Until one of (a)/(a2)/(c) lands,
seq-2048 full runs are
pedestal-producers, recorded alongside — exactly per doctrine.

Pending verification before any claim run: [NQ]'s calibration and eval
sequence counts. The wrapper knobs (SEQ_LEN / CALIBRATION_SEQS /
EVAL_SEQS) pass straight through; defaults stay 16/5 until the paper's
numbers are pinned, and every ledger entry states the counts it ran
with.

First measured r512 data point (recorded alongside, not a ledger
claim): seq-512 diagnostic, solver svd — probe abort at depth 2, ratio
21,161.8 (529x the collapse line; 22.4x worse than the best archived
943.49), wall 119.2 s; the bf16 pedestal reproduced the historical
28.66960268125778 to twelve significant figures. The meter is
calibrated; the recipe is what's failing.

## Relationship to the operational gate (no Goodhart loop)

The band (`ok <= 4 < degraded <= 40 < failed`) remains the per-run
verdict — it kills dead artifacts cheaply. The bar is a separate,
human-gated acceptance criterion with its own ledger entry (model,
commit hash, bit budget row targeted, eval protocol, measured PPL,
ratio, effective_bpw, verdicts of both gates). A run can be `ok` and
still far from the bar; a run at the bar must be `ok`. Exactly one
finish line, defined before the climb.

## Diagnostic context (not acceptance criteria)

[NQ] Table 6 (Qwen3-8B stage ablation, textually present): init-only
206.03 PPL — the paper's own documentation of the collapse class this
repo's production runs have been in (ratio 29.4 vs baseline 7.00) — and
full pipeline 12.47. Cited for diagnosis, never gated on.

## Discarded alternatives

1. **Accept the seq-512 numbers as the bar.** Rejected: finish-line
   drift; the paper's protocol is the anchor, not our convenience.
2. **Anchor the bar on [LB2] rows.** Rejected: wrong paper — the
   pipeline implements the [NQ] method; the LB2 numbers are comparison
   context only (and feed the operational band).
3. **Tolerance 2x or "within the band".** Rejected: post-hoc landing
   site; 15% is the stated, pre-registered number.
4. **Fold the bar into the e2e gate as a fourth verdict.** Rejected:
   the gate must remain cheap and per-artifact; the bar costs a
   full-protocol run and deserves human review in its ledger.

## Informed assumptions

1. **WT2-2048 rows transfer to a from-scratch reimplementation.** The
   numbers include the paper's own pipeline choices (calibration count,
   optimizer paths); a reimplementation may land slightly worse within
   15%. Wrong-surfaces-as: bar unreachable for many cycles → diagnostic
   signal the pipeline diverges from the paper's stages, not a reason to
   move the bar.
2. **effective_bpw (byte-derived, scales included) is comparable to the
   papers' Total Bits for gating the no-extra-bits clause.** The papers
   exclude fixed overhead in places; the clause is directional (never
   admit a PPL bought with more memory), not exact accounting.

## Open Questions

**OQ-1: is the first acceptance run manual or CI-scheduled?** Owner:
orchestrator. Resolution after the fail-fast loop lands and one artifact
reaches the band's neighborhood; the run costs hours and needs the
weights offline-cached, so it is scheduled, not automatic.

## Implementation crystallization (2026-10-27 — tune/eval split, tasks 1-2)

**RED first, and recorded.** `tests/ptq/test_driver_tune_seq_len.py` was
authored before the flag existed and failed for three separate reasons:
`--tune-seq-len` absent from `--help`; a real driver run completing and then
raising `KeyError: 'tune_seq_len'` on its report; and the split invocation
exiting 2 with `unrecognized arguments: --tune-seq-len 16`. Only then was the
flag wired.

**Windows are SLICED, not re-sampled — a method choice, not a memory trick.**
`calib` is built once at `--seq-len` and each row is chunked into
`--tune-seq-len` windows (`batch.split(tune_len, dim=1)`), so the tune stage
sees shorter sequences while every calibration token survives. The obvious
alternative — tokenizing the corpus at the shorter length — is rejected on
protocol grounds: [NQ] pins "128 samples ... with a sequence length of 2048"
(paper copy, Sec. 4.2), so 512-token sampling would silently cut the
calibration budget to a quarter and change the method being tested, not just
the memory. Slicing bounds the retained N² autograd node — the thing that
actually died — and leaves the sampler untouched.

**Design-silent choices, pinned here so they are reviewable:**
- **`calibration_rows` is counted BEFORE the slice.** It counts sequences
  (`n_chunks * BATCH`); slicing multiplies the *window* count per sequence, not
  the sequence count. Counting after the slice would have inflated it by the
  slice factor (16 → 64 at 2048/512) and made the report lie about how much
  calibration data was used. Found while writing the change; the count is
  hoisted above the slice.
- **The report records the RESOLVED tuning geometry** (`tune_seq_len`), not the
  raw flag: unset reports `--seq-len`, so the ledger's "protocol this run used"
  (`bar_recorded_not_gated`) is readable without knowing the defaults. Kept on
  both report paths — the abort payload carries it too, because
  `report_single_shape_both_paths` requires one key set.
- **`--tune-seq-len` is typed `_positive_int`** (the repo's existing knob
  convention), not a bare `int`. A bare `int` would let `--tune-seq-len 0` fall
  into `0 or args.seq_len` and silently restore 2048-token windows — the
  `default_single_geometry_unchanged` "no silent geometry change" clause reached
  through the back door.
- **The wrapper's `TUNE_SEQ_LEN` is passthrough with an empty default**, and the
  flag is appended only when non-empty: the wrapper does not author a tuning
  geometry, the same rule its header already states for the eval geometry.
- **The OOM-shaped launch is announced, not refused** (@bug-hunter's ruling).
  `SEQ_LEN=2048` with `TUNE_SEQ_LEN` unset prints a note naming the geometry whose
  tune stage died at the 32g ceiling and the flag that bounds it, and still starts
  the run — a deliberate unsplit 2048 run stays allowed. The wrapper judges
  nothing; it only refuses to let a bare launch pass for the claimable split. The
  split launch (`TUNE_SEQ_LEN` set) stays silent, and the advisory is pinned in
  `tests/test_run_ptq_wrapper.py`.

**Non-vacuity, at the report surface.** The split run has to be observably
different from the default run, or a driver could record the flag and ignore it:
the tests pin `tune_seq_len == seq_len` with the flag unset and
`tune_seq_len != seq_len` with it set, plus the driver's own slice line
(`calib sliced to 4 windows of 16`) as evidence the slicing ran, and
`calib 2 batches, eval 2 batches` as evidence the sequence count did not move.
The *location* claim — that the shortened windows are what the tune loop
allocates — is a memory fact and belongs to the 32g-container microbench in
review, not to an assertion in this tier. Stated so nobody reads the report
field as the memory proof.

**Guard finding — ruled and fixed in this commit.**
`tests/test_qk_report.py::test_abort_report_has_the_completion_reports_key_set`
compares the driver's *completion* report keys (parsed from source by
`_completion_report_keys()`) against `abort_report()` over a HAND-WRITTEN
`_ABORT_IDENTITY` fixture. Adding `tune_seq_len` reddened the pin until that
fixture was updated: the guard worked, but its mirror was manual — the next key
added to either report shape had to be remembered in that literal too. @bug-hunter
ruled: derive the fixture's key set from the driver's abort payload literal (the
same AST move `_completion_report_keys()` already makes) and keep the value
contracts hand-pinned. Fixed here: `_abort_payload_keys()` reads the key names out
of `main()`'s `abort_report(...)` dict literal and `_abort_identity()` joins them
to the hand-pinned `_ABORT_IDENTITY_VALUES`, asserting neither set is missing nor
extra — so a field added to the driver's abort path grows the fixture
automatically. Non-vacuity is asserted by mutating the driver source and watching
the derived set grow
(`test_abort_fixture_keys_are_derived_from_the_driver_not_remembered`).

**Gates at landing:** full suite **439 passed** (3 new driver tests + 1 wrapper
row + 1 derivation-guard row); lean tier **354 passed / 16 skipped**, `skip
discipline holds` (the new module is *ignored*, not skipped, by the tier gate's
`pytest_ignore_collect`, so the skip count is unchanged); pre-commit **RC=0**;
`openspec validate qk-nq-recreation-bar --strict` valid.
