# Design: qk-divergence-guards

## Context

The r137 incident (see proposal) localized a catastrophic quality failure to
block 0's solver initialization. The mechanism chain has two code-level
facts and one measurement left in flight:

1. `robust_diag(d_in, d_out, tau=None)` clips to `max(D)` **per vector** —
   provably a no-op (the cap equals the largest entry), so assumption A-1's
   "absolute cap" never engages on the default path.
2. The driver passes `tau=cfg.tau` and `PTQConfig.tau` defaults to `None`,
   so every production run to date ran with the protection disabled.
3. Block 0's raw statistics are the pipeline's most heavy-tailed (raw
   unnormalized embedding outputs enter the first block; the probe run
   records p99/median ratios per linear — see `l0_probe_results.json`
   artifacts referenced in tasks).

## Decisions

### D1: finite per-vector tau as the default ceiling policy

Replace the no-op default with a per-vector robust ceiling estimated from
the vector's own statistics: `tau = q95(D)` (95th percentile of the raw
diagonal entries of that vector), applied by the existing clip-then-shrink
path. Rationale:

- It is a real absolute cap: entries above the 95th percentile clip,
  bounding any single heavy tail.
- It is layer-local and vector-local (d_in and d_out each get their own
  estimate), matching A-1's per-layer "absolute cap" reading.
- Healthy layers have concentrated diagonals where q95 ≈ max, so the
  default path change is a near-no-op exactly where no treatment is
  needed (assumption A-iii below).

`tau=None` in `PTQConfig` is redefined to mean "derive per-vector q95";
an explicit numeric tau continues to override (spec A-1 absolute cap
semantics unchanged for callers who pin one).

### D2: post-balance sanity gate

Immediately after `magnitude_balance` produces `(Ub, Vb, s1, s2)` for a
layer (before STE, before any downstream block sees a poisoned prefix),
compute `rho = median(|diag(s1) (Ub Ub-signs) diag(s2)|) / median(|W|)` —
concretely `median(|Wq_binary|) / median(|W|)` — and require it inside a
band `[1/F, F]` with default `F = 10` (config knob `divergence_band_factor`).
Also reject any NaN/Inf scale entry. On violation raise
`QuantizationDivergenceError` naming the layer, the measured ratio, and
the band; the driver surfaces it as a nonzero exit before export and
before the dataset stages run.

- Placement inside `reconstruct_block` (per layer, at balance time) means
  a divergence aborts within the first minutes of the solve phase and
  never contaminates later blocks' calibration inputs.
- The binary-product median (not the continuous factors') is the gated
  quantity because the serialized artifact is the binary product; the
  incident's 250x signature was measured exactly there.
- Band calibration: the healthy population and the diverged population
  sit orders apart (incident: ~250x; coherent layers: O(1) at balance
  time — pinned from probe output during task execution and recorded in
  the crystallization note). F = 10 is the provisional default; the task
  adjusts it only with measured balance-time ratios from both artifacts.

### D3: thread pin re-benchmark (ops hygiene, same change)

The container runs `--cpus 24` with BLAS pinned to 16. Benchmark the r137
solver phase at 16 vs 24 threads under the 24-CPU quota and pin the
measured winner in `tools/run-ptq-container.sh` (and matching env in the
probe docs). The 59-minute wall and the 16-thread history stay in the
record; this task only claims a change if the measurement says so.

## Informed assumptions

- **A-i — the mechanism is preconditioner-fed divergence, not a solver
  bug.** Rationale: scales are coherent in layers 1-26 of both ranks;
  block 0 is the only block whose calibration inputs are raw embedding
  outputs; the tau no-op is verified in code. How it surfaces if wrong:
  the q95-treated block-0 probe arm still diverges (gate band violated
  with sane statistics) — then the investigation escalates to the solver
  (OQ-2), and this change's tau half is scope-reduced accordingly.
- **A-ii — the median-ratio band separates diverged from healthy.**
  Rationale: the two known populations differ by >2 orders; F=10 leaves
  >1 order of margin on each side. How it surfaces if wrong: a false
  abort on a healthy run names the layer and ratio explicitly, making
  the miscalibration immediately visible rather than silent.
- **A-iii — per-vector q95 leaves healthy layers' solves unchanged
  within noise.** Rationale: q95 ≈ max on concentrated diagonals, and
  clipping at max is the current behavior. How it surfaces if wrong:
  the r137 rerun's healthy-layer balance-time rel errors shift materially
  vs the wounded artifact's coherent layers; the rerun report diffs them.

## Discarded alternatives

- **Adaptive/recursive rho schedule** — [NQ] App. C/App. D.4 pin the
  linear schedule (config assumption A-2); deviating needs evidence the
  schedule itself is the defect. The tau cap addresses the measured
  upstream cause instead. Deferred as OQ-2.
- **Post-hoc repair (detect and rescale diverged layers after KD)** —
  treats the symptom downstream of the poisoning; rescaled layers fight
  the Eq. 7-9 balance invariants, and later blocks' calibration inputs
  were already contaminated. Prevention at balance time is simpler and
  cheaper.
- **Skip quantizing block 0** — abandons the hardest layer and breaks
  the full-model inventory contract ([NQ] Algorithm 1 quantizes every
  block; the qk-server inventory gate requires all linears).
- **Raise shrinkage gamma globally** — perturbs every layer's
  preconditioner to fix one block's tails; layer-local tau is the
  minimal-blast-radius fix.

## Open questions

- **OQ-1 — does the q95 tau heal block 0 at r137?** Owner: agent.
  Status: OPEN — isolated block-0 treatment screen in flight
  (`l0_probe.py` inside the ptq container); closes with measured
  per-treatment scale ratios and binary rel errors.
- **OQ-2 — adaptive rho fallback.** Owner: agent, deferred to a
  follow-up change; only activates if OQ-1 closes negative.
- **OQ-3 — exact gate band default.** Owner: agent; closes during task
  execution from measured balance-time ratios (provisional F=10).
- **OQ-4 — is r8's block-0 elevation the same mechanism?** Owner:
  agent; the probe's default-arm ratios are compared against the r8
  artifact's exported block-0 scales; closes with OQ-1's results.
