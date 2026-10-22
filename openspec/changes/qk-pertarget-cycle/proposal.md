# qk-pertarget-cycle — the per-target-linear TuneFP cycle (Figure 3 re-ruled)

## Why

`qk-fix-trio` landed on 2026-10-21, and the fire meant to validate it — the
first paper-faithful run of the 0.6B target — ended in a **64,151x** perplexity
collapse with `paper_faithful: false`. No stage failed, no gate tripped, no
verdict condition went false on the way there.

All three of those instruments shared one reading of Figure 3: **one TuneFP
pass per block, fired after Q/K/V were factorized.** The figure does not say
that, and the reading had no red-first test anywhere — the shape it produced
was asserted into the stage map, the verdict and the suite as if it were
measured.

Two further consequences of the same reading compounded it:

- **A structurally impossible first pass.** Under "factorize Q/K/V, then tune",
  the first TuneFP of every block ran with the student bitwise equal to the
  pristine teacher on the same `x_b` — residual and gradient identically zero.
  The stage could not have done anything, and nothing recorded that it hadn't.
- **No effect evidence anywhere.** `TuneLatentSTE` booked 1024 steps on all 14
  linears of the fatal run, flipped zero signs, and declared `paper_ste`. Its
  only "evidence" — `zero_crossings` — is a restatement of the flip counters
  it was meant to corroborate. ~8.4 h of budget left no trace of whether the
  stage did anything.

## What changes

**Ruling (a), the reading this change is built on — quoted verbatim:**

> Factorize target 1 first, with no preceding TuneFP (a pass before it is
> bitwise vacuous: student == pristine teacher on the same `x_b`, residual and
> gradient identically zero). Then, per target: TuneFP(still-FP remainder)
> -> factorize the target -> STE. TuneFP therefore fires **N-1** times per
> block and STE **N** times.

Implemented in `src/quantibbler/ptq/block_recon.py`:

1. TuneFP is hoisted **into** the factorization loop, ahead of each target's
   own `_set_linear`. Its tunable set is the **still-FP remainder**
   (`quantizable_linears[idx:]`), its `factorized_in_block` is `idx`, and it
   is skipped at `idx == 0`.
2. The pre-group/remainder `split` machinery and `_is_attention_projection`
   are **deleted** — they existed only to serve the overruled reading.
3. The no-fire fallback names its two causes separately instead of merging
   them into one string: a block with fewer than two quantizable linears
   (legitimate) vs a truncated statistics book.
4. `TuneLatentSTE` records its own objective evidence per fire —
   `residual_before` / `residual_after` over the **full capture**, plus the
   first-step gradient norm.

And the instruments are re-pointed so they can go red:

- `src/quantibbler/qk_report.py` — `tune_fp` and `ste` stage-map cells now
  describe the per-target cycle instead of the overruled placement.
- `tests/ptq/test_driver_stage_map.py` — the driver-level pin
  (`tune_fp.invocations == capture.invocations`) is replaced by a **derived**
  invariant.
- `tools/ci/report_checks.py` — a stage-evidence gate (`stage_evidence_findings`).
- `tools/qkb-stall-watchdog.sh` + `tools/run-ptq-container.sh` — a host-side
  stall watchdog (see design.md, "Watchdog").

## What this does *not* establish

The 64,151x collapse is **not** explained by this change, and this change does
not claim it is. What it removes is the property that made the collapse
unattributable: a fidelity-bearing stage that could not show its own work, and
three instruments that shared its mistake. The cause of the collapse is still
open.
