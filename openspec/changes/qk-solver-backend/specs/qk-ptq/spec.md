# qk-ptq delta: qk-solver-backend

## ADDED Requirements

### Requirement: The ADMM solver SHALL default to the torch backend with NumPy as oracle-only

The solver SHALL default to the torch backend, the only backend that produces artifacts, resolved fail-closed once per run with the `QK_RHO_END` provenance idiom. NumPy SHALL be reachable only by explicit name, solely as the determinism oracle for parity acceptance — never producing artifacts — with NO fallback: a torch request without an accelerator SHALL fail rather than resolve to NumPy. `solver_is_admm` SHALL remain an algorithm condition.

#### Scenario: backend_knob_provenance

- **WHEN** a run selects `SOLVER_BACKEND=torch` (or resolves the default)
- **THEN** launch.env, the checkpoint fingerprint, and every solver-probe row record backend and dtype
- **AND** the default resolves to torch — the accelerated backend
- **AND** a torch backend without an available accelerator device fails closed at startup, on the same pattern as `GPU=1` requiring a rocm-tagged image — a torch backend running on CPU is a slowdown wearing an optimization's name

#### Scenario: numpy_is_oracle_only

- **WHEN** the NumPy backend executes
- **THEN** it is reachable only by explicit name, produces no artifacts, and exists solely to generate the reference numbers the parity acceptance compares against

#### Scenario: admm_algorithm_unchanged

- **WHEN** either backend factorizes a matrix
- **THEN** it executes the same continuous-update, SVID-proxy, dual-update, and magnitude-balance steps, and `solver_is_admm` remains true on either backend

### Requirement: Backend parity SHALL be measured on the fixed production-layer probe set with row-time reported

The torch backend SHALL be accepted only after a parity run on the Lane 0 probe set reports, per layer and per backend: post-ADMM pre-polish recon cos, consensus residual trajectory, n_iter, min_eigs floor, and row-time. The oracle arm runs the stratified six-cell grid; a tolerance breach in any probed cell SHALL expand the oracle to all ten entries. fp32-vs-fp64 cosine movement SHALL stay within the ±0.002 tolerance class; a larger movement names the layer and fails acceptance for that layer.

#### Scenario: parity_on_probe_set

- **WHEN** the parity run executes both backends on the fixed probe set
- **THEN** per-layer cosines agree within the tolerance class and the report carries both arms' row-times
- **AND** the row-time column translates the win through the measured row-time share rather than reporting solver-seconds in isolation

#### Scenario: tolerance_breach_named

- **WHEN** any probed layer moves outside the tolerance class between backends
- **THEN** the report names the layer and its delta, and acceptance fails for that layer instead of passing silently
