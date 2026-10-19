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

### D1: tail-gated per-vector tau as the default ceiling policy

Replace the no-op default with a per-vector robust ceiling, **tail-gated**:
`tau = q95(D)` becomes the ceiling only when the raw vector is genuinely
heavy-tailed (`max(D) > 10 x q95(D)`); otherwise no clipping occurs.
Rationale:

- Block-0-style heavy tails (the incident) exceed the gate by orders of
  magnitude and get clipped; degenerate/zero statistics still hit the
  positive floor.
- Healthy concentrated vectors pass through exactly untouched. This is a
  measured correction, not a preference: the unconditional-q95 variant
  degraded the honest tiny-gate fixture (logit delta 0.795 -> 1.43, over
  the A-8 bound) by clipping vectors where the cap at max was effectively
  inert (assumption A-iii was REFUTED — see crystallization).
- It is layer-local and vector-local (d_in and d_out each get their own
  estimate), matching A-1's per-layer "absolute cap" reading.

`tau=None` in `PTQConfig` means "derive the tail-gated per-vector cap";
an explicit numeric tau continues to override (A-1 verbatim semantics).

### D2: post-balance sanity gate (Frobenius-norm ratio)

Immediately after `magnitude_balance` produces `(Ub, Vb, s1, s2)` for a
layer (before STE, before any downstream block sees a poisoned prefix),
compute `||Wq||_F / ||W||_F` where `Wq` is the binary reconstruction
`diag(s1) (sign U)(sign V)^T diag(s2)`, and require it inside `[1/F, F]`
with default `F = 100` (`PTQConfig.divergence_band_factor`). Also reject
any NaN/Inf scale entry. On violation raise `QuantizationDivergenceError`
naming the layer, the measured ratio, and the band; the driver surfaces it
as a nonzero exit before export and before the dataset stages run.

- The **norm ratio** (not a median of entries) is the gated statistic,
  and the band default is **100** — both measured corrections (see
  crystallization): honest fixtures read as low as ~0.1 on the entry
  median with dead scale-floor rows pinning some layers at exactly 0,
  while the incident divergence is an energy explosion (~250x median,
  more in norm). The norm ratio separates them with >2 orders of margin
  on each side.
- Placement inside `reconstruct_block` (per layer, at balance time) means
  a divergence aborts within the first minutes of the solve phase and
  never contaminates later blocks' calibration inputs.

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

- **OQ-1 — does the tail-gated tau heal block 0 at r137?** Owner: agent.
  Status: OPEN — isolated block-0 treatment screen in flight
  (`l0_probe.py` inside the ptq container; arms re-baselined against the
  pre-change no-op default); closes with the rerun's per-layer scale audit.
- **OQ-2 — adaptive rho fallback.** Owner: agent, deferred to a
  follow-up change; only activates if OQ-1 closes negative.
- **OQ-3 — exact gate band default.** Owner: agent.
  RESOLVED 2026-10-19 (implementation crystallization, below): F=100,
  Frobenius-norm ratio.
- **OQ-4 — is r8's block-0 elevation the same mechanism?** Owner:
  agent; closes with OQ-1's results.

## Implementation crystallization (2026-10-19)

Measured during task execution, revising the as-authored design:

- **A-iii REFUTED (unconditional q95 harms healthy layers).** The
  tiny-gate fixture's honest quantization degraded from logit delta
  0.795 (old default) to 1.434 (unconditional q95), over the A-8 bound
  of 1.0 — clipping concentrated vectors is NOT a no-op. D1 became the
  tail-gated policy (max > 10x q95 engages the cap); with the gate,
  healthy vectors are bit-identical to the no-clip path
  (`preconditioner_tau_tail_gate_inert`).
- **Gate statistic and band recalibrated by measurement.** The tiny-gate
  fixture's honest median-ratio is 0.0997 (and two fixture layers read
  exactly 0 from scale-floor dead rows), so the as-authored
  median-ratio/F=10 gate false-positived on the suite itself. The
  Frobenius-norm ratio with F=100 passes all honest fixtures
  (32/32) and sits >2 orders below the incident signature (~250x
  median, more in norm). The dead-row anatomy of the tiny fixtures is
  itself a finding: the pre-gate pipeline silently shipped layers whose
  binary reconstruction had zero median magnitude; the norm gate now
  bounds exactly that pathology (a fully-dead layer reads ratio 0 and
  aborts).
- The isolated probe (v3, driver-native capture) CLOSED OQ-1 and OQ-4
  on 2026-10-19, POSITIVE. Driver-faithful block-0 capture (robust_diag
  monkeypatch, solve stubbed, recorder unwind — 84s, driver memory
  profile), then offline treatment screen on the captured state:
  - Reproduction: under the HISTORICAL default (no-op cap), block-0
    down_proj diverges at norm_ratio 6.28e4 (s1_max 35.4) while its six
    siblings sit at 1e-4..1e-2 — the wounded artifact's exact species
    break (exported s1 max 48). Block 0's raw stats are extreme
    heavy-tailed (d_in max/median: q 2.3e3, v 6.3e3, up_proj 2.4e4) —
    precisely what the no-op ceiling passed through unclipped.
  - Treatment: the landed tail-gated default heals it (down_proj
    norm_ratio 0.027, all 7 layers in 0.021-0.060, inside the gate
    band). Unconditional q95 only shrinks the explosion (83) and is
    independently refuted on healthy layers; median-clip heals but is
    strictly more aggressive than the tail gate. No adaptive-rho work
    needed (OQ-2 stays closed unless the rerun says otherwise).
  - Note: block-0 rel_err(Wq, W) ~ 1.0 in ALL arms including healed
    ones — the flat down_proj spectrum (see skill notes) means the
    binary rank-137 factorization captures little entrywise structure
    at block 0 regardless; STE+KD absorbing that is the algorithm's
    contract. Divergence is the qualitative scale explosion (1e4x),
    not the rel_err floor — the gate bounds the former.
- The v1 probe's trunk-wide stats construction was an OOM/time trap
  (32G kill at 40 min) superseded by the driver-native capture
  pattern; recorded in the agent skill notes, not design (ops, not
  spec).
