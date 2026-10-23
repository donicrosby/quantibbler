# qk-solver-backend — proposal

## Why

User directive: "ensure that we have GPU accelerated ADMM. DO NOT just leave it on the CPU." The solver (`lb_admm.py`) is 536 lines of NumPy, no torch, no device plumbing; the hot loop is 400 iterations of small Gram solves (~4×10¹¹ flops per linear at r=512), measured 90–150 s per linear on 32 CPU cores.

The arithmetic decides the device dtype: gfx1151 (RDNA3.5) runs fp64 at ~1/16 of its fp32 rate, so a torch-fp64 port lands near the CPU's fp64 throughput — a wash. A torch-**fp32** port takes the full-rate path, roughly an order of magnitude over the CPU's fp32, and the campaign's own production-layer measurements put fp32-vs-fp64 cosine movement at ±0.002 — inside the 1e-3 tolerance class. The earning arm is fp32-on-GPU.

Amdahl honesty, stated in the spec: ADMM is ~15–25% of row-time at 0.6B (STE dominates), so a 10× solver win buys ~15–20% of a row there. At the 1–4B scale target the CPU solver goes to tens of minutes per row and the GPU path becomes the difference between feasible and infeasible.

## What Changes

1. **Backend knob**: `SOLVER_BACKEND=numpy|torch` with device + dtype selection, using the same provenance idiom as `QK_RHO_END`/`COMPUTE_DTYPE` (fail-closed resolution, recorded in launch.env, checkpoint fingerprint, and probe rows).
2. **NumPy stays the default reference** — the determinism anchor for the kernels' parity gate is untouched unless the knob moves it.
3. **Parity acceptance = the Lane 0 probe set** (`qk-nanoquant-alignment-lane0` owns the harness definition; this change ADDs the backend arm to it): same fixed production layers, both backends through the existing lb_admm, reporting post-ADMM pre-polish cos, consensus residual trajectory, n_iter, min_eigs floor, AND row-time, so the Amdahl translation is in the report rather than implied.
4. **Fidelity semantics**: `solver_is_admm` stays an algorithm condition, not a backend condition — admm-on-torch is still ADMM. Backend and dtype become provenance fields on every probe row and launch.env; a fidelity-claiming run records them. fp32-solver arithmetic on the faithful path is a user directive with the measured tolerance on record.

## Impact

- `src/quantibbler/ptq/lb_admm.py` — torch backend for the ADMM arithmetic (Gram builds, solves, proxy updates), same iteration algebra.
- `src/quantibbler/ptq/config.py` — `SOLVER_BACKEND` knob + dtype/device fields; provenance plumbing.
- `src/quantibbler/qk_report.py` — backend/dtype fields on probe rows and report provenance.
- Tests — parity fixtures comparing numpy vs torch backends on the fixed probe set within the 1e-3 class.

## Non-goals

- Does NOT change the ADMM iteration algebra, the regime (that's `qk-nanoquant-alignment-lane3`), the preconditioner (lane1), or the stopping rule semantics.
- Does NOT re-spec the Lane 0 harness — it cites the sibling change's requirement rather than restating it.
- Does NOT make torch the default; does NOT remove the NumPy path.
