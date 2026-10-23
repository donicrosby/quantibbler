# qk-solver-backend — design

## Backend architecture (resolved)

- The torch backend mirrors the NumPy iteration algebra step-for-step: Gram builds, ridge-stabilized solves, SVID proxy, dual accumulation, magnitude balancing. No algorithmic divergence rides the port.
- Device/dtype derive per solve from the resolved knob; weights hand off through `.cpu().numpy()` on the NumPy path only.
- fp32 is the earning dtype on gfx1151 (RDNA3.5 fp64 ≈ 1/16 fp32 rate); torch-fp64 is kept only as a parity-checkable arm, not a speed arm.

## Fidelity semantics (resolved)

- `solver_is_admm` pins the ALGORITHM; backend and dtype are provenance. A fidelity-claiming run records them. fp32-solver arithmetic on the faithful path is a user directive; the measured tolerance (±0.002 on production layers, ±0.002 campaign table) is the standing acceptance bound, re-verified by the parity scenario per layer.

## Dependency

- Acceptance rides the probe-set harness owned by `qk-nanoquant-alignment-lane0` (cited, not restated). Implementation order: Lane 0 harness first, then this change's arm.
- NOT gated on Lane 0's hypothesis verdict — the user directive stands regardless of which preconditioner wins; the backend parity runs on whichever preconditioner Lane 0's harness pins.

## Known limitations

- The 10×-over-CPU-fp32 estimate is from the RDNA3.5 fp64 ratio and the solver's measured flop count; the parity run's row-time column is the measured value that supersedes it.
- NumPy remains the default until the parity run reports; flipping the default is a separate decision with the user.
- The rank-1 init path (`np.linalg.eigh` and the power iteration) has sign/branch ambiguity that different backends may resolve differently on near-degenerate spectra. A re-oriented init is a different factorization with the same cos, so cos-based acceptance is correct — but the residual-trajectory column (required by the parity scenario) is what makes a re-oriented init visible in the data rather than inferred.
