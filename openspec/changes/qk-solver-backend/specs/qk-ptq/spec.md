# qk-ptq delta: qk-solver-backend

## ADDED Requirements

### Requirement: The ADMM solver SHALL support a torch backend behind a provenance-recorded knob

The solver SHALL support `SOLVER_BACKEND=numpy|torch`, resolved fail-closed once per run with the same provenance idiom as `QK_RHO_END` (recorded in launch.env, the checkpoint fingerprint, and every probe row). The NumPy backend SHALL remain the default and the determinism reference; the torch backend SHALL run the same iteration algebra on a device with a recorded dtype. The fidelity condition `solver_is_admm` SHALL remain an algorithm condition — the backend does not change it.

#### Scenario: backend_knob_provenance

- **WHEN** a run selects `SOLVER_BACKEND=torch`
- **THEN** launch.env, the checkpoint fingerprint, and every solver-probe row record backend and dtype
- **AND** an unset knob resolves to numpy, preserving today's behavior byte-for-byte
- **AND** a torch backend without an available accelerator device fails closed at startup, on the same pattern as `GPU=1` requiring a rocm-tagged image — a torch backend running on CPU is a slowdown wearing an optimization's name

#### Scenario: admm_algorithm_unchanged

- **WHEN** the torch backend factorizes a matrix
- **THEN** it executes the same continuous-update, SVID-proxy, dual-update, and magnitude-balance steps as the NumPy backend, and `solver_is_admm` remains true on either backend

### Requirement: Backend parity SHALL be measured on the fixed production-layer probe set with row-time reported

The torch backend SHALL be accepted only after a parity run on the Lane 0 probe set (owned by `qk-nanoquant-alignment-lane0`) reports, per layer and per backend: post-ADMM pre-polish recon cos, consensus residual trajectory, n_iter, min_eigs floor, and row-time. fp32-vs-fp64 cosine movement SHALL stay within the campaign's measured tolerance class (±0.002 on production layers); a larger movement names the layer and fails the acceptance for that layer.

#### Scenario: parity_on_probe_set

- **WHEN** the parity run executes both backends on the fixed probe set
- **THEN** per-layer cosines agree within the tolerance class and the report carries both arms' row-times
- **AND** the row-time column translates the win through the measured row-time share rather than reporting solver-seconds in isolation

#### Scenario: tolerance_breach_named

- **WHEN** any probed layer moves outside the tolerance class between backends
- **THEN** the report names the layer and its delta, and acceptance fails for that layer instead of passing silently
