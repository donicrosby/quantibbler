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

## Sequencing mandate (user directive, 2026-10-23)

GPU acceleration lands PRIOR to the next production run — the user has ruled the numerics question settled by the measured A/B campaign (fp32 GPU ≡ fp64 CPU within tolerance on tested layers). This mandate is a SEQUENCING constraint, not a parity waiver: the parity scenarios in the spec remain the acceptance gate, and the torch fp32 arm becomes the production arm only after the probe-set parity passes per layer.

## Known limitations

- The 10×-over-CPU-fp32 estimate is from the RDNA3.5 fp64 ratio and the solver's measured flop count; the parity run's row-time column is the measured value that supersedes it.
- The numpy-to-torch default flip is decided by the sequencing mandate, contingent on parity acceptance — no longer an open decision.
- The rank-1 init path (`np.linalg.eigh` and the power iteration) has sign/branch ambiguity that different backends may resolve differently on near-degenerate spectra. A re-oriented init is a different factorization with the same cos, so cos-based acceptance is correct — but the residual-trajectory column (required by the parity scenario) is what makes a re-oriented init visible in the data rather than inferred.

## Landing efficiency (revised by user directive, 2026-10-23)

The user has reversed the numpy-first sequencing: the decisive Lane 0 fire happens ON the torch backend once the port lands — torch-only, 20 solves, no NumPy runs. One execution delivers the Lane 0 verdict and the measured row-times; target wall time ~5–10 min (at GPU fp32 rates vs 90–150 s/solve on CPU). Lane 0's harness therefore does NOT fire before the torch port exists.

## NumPy scope and the fire grid (user directive, 2026-10-23, revised)

NumPy executes ONLY inside unit tests — the parity suite is its sole sanctioned home. The in-fire oracle grid is therefore REMOVED: the fire carries **20 torch verdict solves and no NumPy runs**. Parity acceptance rests on (a) the unit-test parity suite (sign-ambiguity-safe, reconstructed-product comparisons) and (b) the campaign's measured tolerance record (fp32 GPU ≡ fp64 CPU within ±0.002 on production layers). Any tolerance breach in a parity unit test blocks acceptance and names the operation.

## Fire sequencing (v2 complete, 2026-10-23)

v2 finished at 16:42Z — nothing is in flight, and the earlier note sequencing the Lane 0 fire behind it is historical. The only things in front of the GPU fire are the port, the parity tests, and the harness GPU wiring. The `lb_admm.py` namespace swap that makes `SOLVER_BACKEND` live is no longer blocked by the v2 import graph; it lands with the harness wiring. The torch port (`solver_backend.py` + tests) is already landed outside the fire's import graph.
