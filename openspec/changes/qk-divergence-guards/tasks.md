# Tasks: qk-divergence-guards

## 1. Block-0 evidence + treatment screen (isolated probe)

- [x] 1.1 Run the isolated block-0 treatment probe inside the ptq
      container (driver-native capture: robust_diag monkeypatch +
      stubbed solve + recorder unwind; then offline tau treatment
      screen at rank 137); results in
      `/tmp/ptq/l0_probe_results.json` + probe log: default (no-op)
      arm reproduces the divergence on down_proj (norm_ratio 6.28e4,
      s1_max 35.4); tail-gated default heals (0.027).
- [x] 1.2 Compare the probe's default-arm ratios against the wounded
      artifact's exported block-0 scales and against the r8 artifact's
      block-0 scales; confirm or refute A-i/A-iv in a crystallization
      note appended to design.md (OQ-1/OQ-4 close here). CONFIRMED:
      same species break, same mechanism (raw-stat tails), r8 is the
      milder manifestation; recorded in design.md crystallization.

## 2. tau default policy (D1)

- [x] 2.1 RED: add `tests/ptq/test_preconditioner_tau.py` covering
      scenario IDs `preconditioner_tau_default_q95` (tau=None yields a
      finite cap strictly below max(D) on a heavy-tailed vector),
      `preconditioner_tau_explicit_override` (numeric tau unchanged), and
      `preconditioner_tau_healthy_noop` (concentrated vector: q95 clip
      leaves entries within 1 ulp of the no-op path). Observe RED.
- [x] 2.2 GREEN: implement the q95 default in `robust_diag`
      (calibration.py) behind the same `tau=None` signature; update the
      `PTQConfig.tau` docstring and the qk-ptq spec delta wording if the
      scenario language needs it. Observe GREEN.
      (Tail-gate refinement + `preconditioner_tau_tail_gate_inert`
      amendment recorded in design.md crystallization — A-iii refuted
      by the tiny-gate fixture mid-implementation.)
- [x] 2.3 Verify no healthy-layer regression: run the existing
      preconditioner/solver pytest tier plus the tiny-model e2e gate;
      confirm the gate's BPW and logit-delta assertions still pass.

## 3. Post-balance divergence gate (D2)

- [x] 3.1 RED: add `tests/ptq/test_divergence_gate.py` covering
      `divergence_gate_healthy_passes` (ratio in band → no raise, layers
      proceed), `divergence_gate_diverged_aborts` (inflated s1 → raise
      naming the layer and measured ratio), and
      `divergence_gate_nan_rejected` (NaN scale → raise). Observe RED.
- [x] 3.2 GREEN: implement the gate in `block_recon.reconstruct_block`
      at balance time; add `divergence_band_factor` to `PTQConfig`
      (default 100.0 after measurement) and a dedicated error type in the
      ptq package; wire the driver to fail fast with a nonzero exit
      before export. Observe GREEN. (Metric: Frobenius-norm ratio — see
      design.md crystallization; the as-authored entry-median metric
      false-positived on honest fixtures.)
- [x] 3.3 End-to-end tripwire proof: run the tiny-model quantize gate
      with an artificially inflated block-0 scale via a test-only hook;
      assert the run aborts naming `layers.0` and that no artifact is
      written. (`tests/ptq/test_divergence_tripwire.py`: abort raised
      inside quantize_model, before any QuantizedModel/serialized bytes
      exist.)

## 4. Thread pin measurement (D3)

- [x] 4.1 Benchmark the isolated r137 layer-15 solve at 16 vs 24 BLAS
      threads under the 24-CPU container quota (2x per config, interleaved
      order); record medians in the run notes. RESULT: 16T median 12.8s
      vs 24T 13.8s (subprocess-isolated, interleaved, block-0 down_proj
      at rank 137) — 24 buys nothing on these shapes; KEEP 16
      (`/tmp/ptq/thread_bench.json`). No script change needed.

## 5. Rerun + close-out

- [ ] 5.1 Full r137 rerun through the container driver with the guards
      live; expect either a clean artifact (ppl bounded well below
      baseline x2 and sane scales across all 28 blocks) or a fast,
      named abort. Record the report JSON in the run notes.
- [ ] 5.2 Run the full gate suite (pytest, cargo test, clippy
      -D warnings, fmt --check, maturin build) and record results.
