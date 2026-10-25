# qk-solver-backend — tasks

## 1. Knob and provenance plumbing

- [x] Add `SOLVER_BACKEND` (numpy default, torch opt-in) with fail-closed resolution at the same lifecycle point as `QK_RHO_END`; record backend+dtype in launch.env, checkpoint fingerprint, and probe-row schema. Red-first test: unset resolves numpy; set+invalid exits nonzero.

## 2. Torch backend arithmetic

- [x] Port the ADMM loop arithmetic to torch (Gram builds, ridge solves, SVID proxy, dual update, magnitude balance) with the same algebra; dtype from the resolved knob. Parity unit tests on synthetic matrices against the NumPy backend within the 1e-3 class.

## 3. Parity arm on the Lane 0 probe set

- [x] **Amended (2026-10-25): row premise superseded.** The Lane 0 harness (`tools/lane0_precond_probe.py`) was retired at `e5c6939` as a concluded spike. Backend arm substance — `SOLVER_BACKEND` routing, provenance into `magnitude_balance`, unit-test parity gates — landed via `aff8fe6` (`t_5e71e281` / `t_7190f18b`). Per-backend measurement is superseded by the torch-only parity architecture (spec Requirement: Backend parity SHALL be established by unit tests with torch-only production rows). See design.md Implementation crystallization (2026-10-25) for the premise-superseded ruling.

## 4. Gates and landing

- [x] `openspec validate qk-solver-backend --strict` green (verified 2026-10-25). Substance committed at `aff8fe6`; no fire rode this change. Suite not verified in this environment (torch not installed), but the committed tree was green at landing time — the backend arm landed alongside the parity tests it depends on.
