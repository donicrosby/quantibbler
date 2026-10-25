"""Backend parity: the torch lane must compute what the NumPy lane computes.

The solver's algebra is one implementation; these tests pin the arithmetic the
two backends actually perform, so "the torch backend runs the same solver" is a
measured claim rather than a structural one. Every comparison is on CPU in
float64, where torch and NumPy are both exact references -- accelerator-specific
divergence is a separate question and belongs to the probe's tolerance column,
not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantibbler.ptq.solver_backend import available_backends, get_backend


@pytest.fixture(scope="module")
def backends():
    np_b = get_backend("numpy")
    t_b = get_backend("torch", device="cpu")
    return np_b, t_b


def _spd(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_available_backends_names_what_can_be_built():
    names = available_backends()
    assert "numpy" in names
    # torch is an optional extra; if it imports, it must be reported.
    try:
        import torch  # noqa: F401
    except ImportError:
        assert "torch" not in names
    else:
        assert "torch" in names


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown solver backend"):
        get_backend("numpyish")


def test_default_backend_is_torch_and_fails_closed_without_accelerator():
    """Contract (qk-solver-backend spec.md:5-7): torch default, numpy test-only.

    "The ADMM solver SHALL default to the torch backend with NumPy
    unit-test-only ... a torch request without an accelerator SHALL fail rather
    than resolve to NumPy." This pin previously asserted a numpy default, which
    was the pre-amendment contract; the spec is the authority and the refusal is
    half of what it requires.
    """
    import torch

    from quantibbler.ptq.solver_backend import DEFAULT_SOLVER_BACKEND

    assert DEFAULT_SOLVER_BACKEND == "torch"
    if torch.cuda.is_available():
        assert get_backend().name == "torch"
    else:
        with pytest.raises(RuntimeError, match="requires an accelerator"):
            get_backend()
    # NumPy is the unit-test parity reference, reachable by explicit request.
    assert get_backend("numpy").name == "numpy"
    # The parity/diagnostic path stays reachable on an explicit device.
    assert get_backend("torch", device="cpu").name == "torch"


def test_torch_backend_refuses_missing_accelerator():
    """Fail closed: an explicit cuda request on a CPU box is not a fallback."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("accelerator present; the refusal path cannot be exercised")
    with pytest.raises(RuntimeError, match="no accelerator is available"):
        get_backend("torch", device="cuda")


def test_implicit_torch_without_accelerator_is_refused():
    """Tri-state on the fail-closed branch (ruled: diagnostic reading).

    implicit + no accelerator -> raises; explicit cpu -> succeeds (the parity /
    diagnostic path, and the only thing that makes the numpy-oracle comparison
    runnable on a CPU-only box); explicit cuda without cuda -> raises.
    """
    import torch

    from quantibbler.ptq.solver_backend import resolve_device

    if torch.cuda.is_available():
        pytest.skip("accelerator present; the implicit-refusal path cannot be exercised")

    with pytest.raises(RuntimeError, match="requires an accelerator"):
        get_backend("torch")

    assert get_backend("torch", device="cpu").name == "torch"
    assert resolve_device("torch", device="cpu") == "cpu"

    with pytest.raises(RuntimeError, match="no accelerator is available"):
        resolve_device("torch", device="cuda")


def test_solver_backend_env_knob_is_fail_closed(monkeypatch):
    """SOLVER_BACKEND mirrors QK_RHO_END: unset=config, valid=env, garbage=raise."""
    from quantibbler.ptq.solver_backend import solver_backend_provenance

    monkeypatch.delenv("SOLVER_BACKEND", raising=False)
    assert solver_backend_provenance("numpy") == ("numpy", "config")

    monkeypatch.setenv("SOLVER_BACKEND", "torch")
    assert solver_backend_provenance("numpy") == ("torch", "env")

    monkeypatch.setenv("SOLVER_BACKEND", "  NUMPY  ")
    assert solver_backend_provenance("torch") == ("numpy", "env")

    monkeypatch.setenv("SOLVER_BACKEND", "cuda-ish")
    with pytest.raises(ValueError, match="SOLVER_BACKEND"):
        solver_backend_provenance("numpy")


def test_resolve_device_numpy_is_not_a_device():
    """The numpy lane has no device dimension; the digest records it as n/a."""
    from quantibbler.ptq.solver_backend import resolve_device

    assert resolve_device("numpy") == "n/a"


def test_cholesky_solve_agrees(backends):
    np_b, t_b = backends
    a = _spd(64, seed=1)
    y = np.random.default_rng(2).standard_normal(64)
    dt_n, dt_t = np_b.dtype("float64"), t_b.dtype("float64")
    ln = np_b.linalg.cholesky(np_b.asarray(a, dt_n))
    lt = t_b.linalg.cholesky(t_b.asarray(a, dt_t))
    assert np.allclose(ln, lt.cpu().numpy(), atol=1e-12)
    sn = np_b.linalg.solve(np_b.asarray(a, dt_n), np_b.asarray(y, dt_n))
    st = t_b.linalg.solve(t_b.asarray(a, dt_t), t_b.asarray(y, dt_t))
    assert np.allclose(sn, st.cpu().numpy(), atol=1e-12)


def test_svd_and_eigh_agree(backends):
    np_b, t_b = backends
    a = _spd(48, seed=3)
    dt_n, dt_t = np_b.dtype("float64"), t_b.dtype("float64")
    u_n, s_n, vt_n = np_b.linalg.svd(np_b.asarray(a, dt_n), full_matrices=False)
    u_t, s_t, vt_t = t_b.linalg.svd(t_b.asarray(a, dt_t), full_matrices=False)
    assert np.allclose(s_n, s_t.cpu().numpy(), atol=1e-12)
    # Singular vectors are sign-ambiguous; compare the reconstructed product,
    # which is what the solver consumes and what a re-orientation cannot move.
    recon_n = u_n @ np.diag(s_n) @ vt_n
    recon_t = (u_t @ t_b.torch.diag(s_t) @ vt_t).cpu().numpy()
    assert np.allclose(recon_n, recon_t, atol=1e-10)
    w_n, v_n = np_b.linalg.eigh(np_b.asarray(a, dt_n))
    w_t, v_t = t_b.linalg.eigh(t_b.asarray(a, dt_t))
    assert np.allclose(w_n, w_t.cpu().numpy(), atol=1e-12)
    assert np.allclose(
        v_n @ np.diag(w_n) @ v_n.T, (v_t @ t_b.torch.diag(w_t) @ v_t.T).cpu().numpy(), atol=1e-10
    )


def test_elementwise_and_reduction_semantics_agree(backends):
    np_b, t_b = backends
    x = np.array([-2.0, -0.0, 0.0, 3.0, np.nan])
    dt_n, dt_t = np_b.dtype("float64"), t_b.dtype("float64")
    xn, xt = np_b.asarray(x, dt_n), t_b.asarray(x, dt_t)
    # sign(0) must stay 0 in BOTH lanes: the solver applies its own sign(0)=+1
    # convention, so a backend that normalized it would move packed signs.
    # equal_nan: sign(NaN) is NaN in both lanes, and NaN != NaN otherwise.
    assert np.array_equal(np_b.sign(xn), t_b.sign(xt).cpu().numpy(), equal_nan=True)
    assert np.array_equal(np_b.isfinite(xn), t_b.isfinite(xt).cpu().numpy())
    assert np_b.any(np_b.asarray([False, True])) == t_b.any(t_b.asarray([False, True]))
    assert np_b.all(np_b.asarray([True, True])) == t_b.all(t_b.asarray([True, True]))
    assert np_b.count_nonzero(xn) == int(t_b.count_nonzero(xt))
    assert np.allclose(
        np_b.where(xn >= 0, np_b.asarray(1.0, dt_n), np_b.asarray(-1.0, dt_n)),
        t_b.where(xt >= 0, t_b.asarray(1.0, dt_t), t_b.asarray(-1.0, dt_t)).cpu().numpy(),
    )


def test_method_level_conversions_exist_in_both_lanes(backends):
    """`astype` and `copy` are NumPy *methods*; torch has neither.

    This is the class a grep for ``np.`` misses entirely, so it is pinned by
    test: the backend must supply both spellings or the port fails at runtime.
    """
    np_b, t_b = backends
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np_b.astype(np_b.asarray(a), np_b.dtype("float32")).dtype == np.dtype("float32")
    assert t_b.astype(t_b.asarray(a), t_b.dtype("float32")).dtype == t_b.torch.float32
    cloned = np_b.clone(np_b.asarray(a))
    assert cloned is not np_b.asarray(a)
    assert np.array_equal(cloned, a)
    t_cloned = t_b.clone(t_b.asarray(a))
    assert t_cloned is not None and t_cloned.shape == (2, 3)
    assert np_b.scalar(1, np_b.dtype("int8")) == 1
    assert int(t_b.scalar(1, t_b.dtype("int8"))) == 1


def test_construction_ops_agree(backends):
    np_b, t_b = backends
    dt_n, dt_t = np_b.dtype("float64"), t_b.dtype("float64")
    assert np.array_equal(np_b.zeros((2, 3), dt_n), t_b.zeros((2, 3), dt_t).cpu().numpy())
    assert np.array_equal(np_b.full((2,), 2.5, dt_n), t_b.full((2,), 2.5, dt_t).cpu().numpy())
    assert np.array_equal(np_b.eye(3, dt_n), t_b.eye(3, dt_t).cpu().numpy())
    assert np.array_equal(
        np_b.outer(np_b.asarray([1.0, 2.0], dt_n), np_b.asarray([3.0, 4.0], dt_n)),
        t_b.outer(t_b.asarray([1.0, 2.0], dt_t), t_b.asarray([3.0, 4.0], dt_t)).cpu().numpy(),
    )
    assert np.array_equal(
        np_b.diag(np_b.asarray([1.0, 2.0], dt_n)), np_b.diag(np_b.asarray([1.0, 2.0], dt_n))
    )
    assert np.allclose(
        np_b.nanmax(np_b.asarray([1.0, np.nan, 3.0], dt_n)),
        float(t_b.nanmax(t_b.asarray([1.0, np.nan, 3.0], dt_t))),
    )
    assert np_b.finfo(dt_n).eps == t_b.finfo(dt_t).eps
    assert float(np_b.inf) == float(t_b.inf) == float("inf")
