"""qk-ptq-speedups — the `min_eigs` eigenvalue probe is gated behind the

solver-probe flag (`QK_SOLVER_PROBE`).

Spec oracle: `openspec/changes/qk-ptq-speedups/specs/qk-ptq/spec.md`
(Requirement: *Eigenvalue diagnostics SHALL be gated behind the probe flag*).

Scenarios pinned here, by their verbatim ids from that delta:

* ``production_run_pays_no_diagnostic_tax`` — flag off: NO eigenvalue call
  happens at all (a recorder on the one linalg seam both backends route
  through counts them), and the trace carries no column.
* ``diagnostic_run_keeps_the_column`` — flag on: every solve still records
  `min_eigs` (the Lane 0 harness / Lane 3 sweep requirement).
* ``gating_is_behavior_neutral`` — the same target factorizes bit-identically
  with the flag on and off: gating moves wall time, nothing else.
* ``spd_signal_preserved`` — `trace.n_jitter`, the always-on non-SPD record
  carried by the Cholesky solve itself, is untouched by the gate.

The tests pin `solver_backend="numpy"` so they run identically on the dev CPU
wheel and in the ROCm image: the gate is one `if` in the solve loop, not a
backend-specific path.
"""

import numpy as np
import pytest

from quantibbler.ptq import solver_backend
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import SOLVER_PROBE_ENV, _cholesky_solve, lb_admm

_NP = solver_backend.get_backend("numpy")


def _target(m: int = 16, n: int = 12, r: int = 4, seed: int = 7) -> np.ndarray:
    """Exact rank-r matrix built from ±1 factors (the tier's usual fixture)."""
    gen = np.random.default_rng(seed)
    ub = np.where(gen.random((m, r)) < 0.5, -1.0, 1.0)
    vb = np.where(gen.random((n, r)) < 0.5, -1.0, 1.0)
    return ub @ vb.T


def _cfg() -> PTQConfig:
    return PTQConfig(rank=4, rho_start=1e-2, rho_end=1e2, solver_backend="numpy")


@pytest.fixture
def eig_calls(monkeypatch):
    """Recorder on `_LinalgProxy.eigvalsh` — the ONE seam both backends use.

    Counting here (rather than asserting on `trace.min_eigs`) is the stronger
    reading of the requirement: the production path must not pay the
    computation, not merely not store it.
    """
    calls: list[tuple[int, ...]] = []
    real = solver_backend._LinalgProxy.eigvalsh

    def _record(self, a):  # noqa: ANN001, ANN202
        calls.append(tuple(np.shape(a)))
        return real(self, a)

    monkeypatch.setattr(solver_backend._LinalgProxy, "eigvalsh", _record)
    return calls


def test_scn_production_run_pays_no_diagnostic_tax(monkeypatch, eig_calls):
    """`production_run_pays_no_diagnostic_tax`: flag off => zero eigvalsh."""
    monkeypatch.delenv(SOLVER_PROBE_ENV, raising=False)

    res = lb_admm(_target(), _cfg())

    assert res.n_iter > 0, "the solve did not iterate: the fixture is broken"
    assert eig_calls == [], (
        f"production path called eigvalsh {len(eig_calls)} time(s) at shapes "
        f"{sorted(set(eig_calls))}: the diagnostic tax is not gated"
    )
    assert res.trace.min_eigs == [], "the min_eigs column was recorded with the flag off"


def test_scn_diagnostic_run_keeps_the_column(monkeypatch):
    """`diagnostic_run_keeps_the_column`: flag on => `min_eigs` still emitted."""
    monkeypatch.setenv(SOLVER_PROBE_ENV, "1")

    res = lb_admm(_target(), _cfg())

    # Two continuous half-updates per iteration (U-side then V-side).
    assert len(res.trace.min_eigs) == 2 * res.n_iter
    assert all(np.isfinite(res.trace.min_eigs))
    assert all(eig > 0.0 for eig in res.trace.min_eigs)


def test_scn_gating_is_behavior_neutral(monkeypatch):
    """`gating_is_behavior_neutral`: same target, identical solver state."""
    monkeypatch.delenv(SOLVER_PROBE_ENV, raising=False)
    off = lb_admm(_target(), _cfg())
    monkeypatch.setenv(SOLVER_PROBE_ENV, "1")
    on = lb_admm(_target(), _cfg())

    assert (off.n_iter, off.converged) == (on.n_iter, on.converged)
    assert off.trace.residuals == on.trace.residuals
    assert np.array_equal(off.U, on.U)
    assert np.array_equal(off.V, on.V)
    assert np.array_equal(off.P_u, on.P_u)
    assert np.array_equal(off.P_v, on.P_v)
    # ... and the only difference IS the diagnostic column.
    assert off.trace.min_eigs == []
    assert len(on.trace.min_eigs) == 2 * on.n_iter


def test_scn_spd_signal_preserved(monkeypatch):
    """`spd_signal_preserved`: `n_jitter` stays always-on with the flag off."""
    monkeypatch.delenv(SOLVER_PROBE_ENV, raising=False)

    from quantibbler.ptq.lb_admm import ADMMTrace

    trace = ADMMTrace()
    h_spd = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    x_spd, jittered = _cholesky_solve(_NP, h_spd, b, trace)
    assert jittered is False
    assert trace.n_jitter == 0
    assert np.all(np.isfinite(x_spd))

    # Indefinite by construction: the stabilized fallback engages and the
    # always-on SPD signal records it, flag or no flag.
    h_bad = h_spd - 10.0 * np.eye(2)
    x_bad, jittered_bad = _cholesky_solve(_NP, h_bad, b, trace)
    assert jittered_bad is True
    assert trace.n_jitter == 1, "the non-SPD event went unrecorded"
    assert np.all(np.isfinite(x_bad))
