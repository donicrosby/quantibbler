# qk-solver-backend — tasks

## 1. Knob and provenance plumbing

- [x] Add `SOLVER_BACKEND` (numpy default, torch opt-in) with fail-closed resolution at the same lifecycle point as `QK_RHO_END`; record backend+dtype in launch.env, checkpoint fingerprint, and probe-row schema. Red-first test: unset resolves numpy; set+invalid exits nonzero.

## 2. Torch backend arithmetic

- [x] Port the ADMM loop arithmetic to torch (Gram builds, ridge solves, SVID proxy, dual update, magnitude balance) with the same algebra; dtype from the resolved knob. Parity unit tests on synthetic matrices against the NumPy backend within the 1e-3 class.

## 3. Parity arm on the Lane 0 probe set

- [ ] Wire the backend arm into the Lane 0 harness run: per layer, per backend — recon cos, residual trajectory, n_iter, min_eigs floor, row-time. Verify the row-time column reports wall seconds per linear per arm.

## 4. Gates and landing

- [ ] Suite green in the torch/hf lane (quote env + count), `openspec validate qk-solver-backend --strict`, scoped commit. No fire rides this change — the parity run line goes to the room for @user's go.
