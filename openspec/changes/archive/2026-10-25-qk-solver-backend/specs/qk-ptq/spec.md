# qk-ptq delta: qk-solver-backend

## ADDED Requirements

### Requirement: The ADMM solver SHALL default to the torch backend with NumPy unit-test-only

The solver SHALL default to the torch backend, the only backend that produces artifacts, resolved fail-closed once per run with the `QK_RHO_END` provenance idiom. NumPy SHALL execute ONLY inside unit tests, as the parity reference — never in a fire, harness, or integration path — with NO fallback: a torch request without an accelerator SHALL fail rather than resolve to NumPy. `solver_is_admm` SHALL remain an algorithm condition.

#### Scenario: backend_knob_provenance

- **WHEN** a run selects `SOLVER_BACKEND=torch` (or resolves the default)
- **THEN** launch.env, the checkpoint fingerprint, and every solver-probe row record backend and dtype
- **AND** the default resolves to torch — the accelerated backend
- **AND** a torch backend without an available accelerator device fails closed at startup, on the same pattern as `GPU=1` requiring a rocm-tagged image — a torch backend running on CPU is a slowdown wearing an optimization's name

#### Scenario: numpy_is_unit_test_only

- **WHEN** the NumPy backend executes
- **THEN** it executes inside a unit test and nowhere else — no fire, no harness, no integration path

#### Scenario: admm_algorithm_unchanged

- **WHEN** either backend factorizes a matrix
- **THEN** it executes the same continuous-update, SVID-proxy, dual-update, and magnitude-balance steps, and `solver_is_admm` remains true on either backend

### Requirement: Backend parity SHALL be established by unit tests with torch-only production rows

The torch backend SHALL be accepted on its unit-test parity suite — the only place the NumPy reference executes — together with the campaign's measured tolerance record (fp32 GPU ≡ fp64 CPU within ±0.002 on production layers). Production and probe fires are torch-only, and every probe row SHALL carry row-time so the measured win replaces the estimate.

#### Scenario: torch_only_fire_rows

- **WHEN** the Lane 0 fire executes
- **THEN** all 20 solves run on torch, no NumPy process runs, and every row carries row-time alongside recon cos, residual trajectory, n_iter, and min_eigs

#### Scenario: unit_test_parity_is_the_gate

- **WHEN** the backend's parity suite runs as unit tests
- **THEN** tests compare reconstructed products (sign-ambiguity-safe) within the tolerance class, and a failing parity test blocks acceptance

#### Scenario: tolerance_breach_named

- **WHEN** any parity unit test moves outside the tolerance class
- **THEN** the failing test names the operation and its delta, and acceptance fails instead of passing silently
