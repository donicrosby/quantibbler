"""Dispatch-semantics tests for quantibbler.kernels (spec D6 / dispatch req).

Scenarios: dispatch_default_auto, dispatch_env_override,
dispatch_explicit_rust_missing_raises, dispatch_explicit_numpy_always_works,
plus the identical-signature surface check on both backends.
"""

from __future__ import annotations

import numpy as np
import pytest

import quantibbler.kernels as k


@pytest.fixture()
def no_extension(monkeypatch):
    """Simulate extension absence (module not importable on fresh import)."""
    monkeypatch.setattr(k, "HAVE_KERNELS", False)
    monkeypatch.setattr(k, "_kernels", None)
    yield


_KERNELS = (
    "pack_bits",
    "unpack_bits",
    "sign_estimate",
    "packed_matvec",
    "packed_matmul",
    "joint_itq",
)


def test_dispatch_default_auto(monkeypatch):
    monkeypatch.delenv("QUANTIBBLER_BACKEND", raising=False)
    monkeypatch.setattr(k, "backend", k.backend, raising=False)
    expected = "rust" if k.HAVE_KERNELS else "numpy-reference"
    assert k.backend() == expected
    # no env var, no kwarg: resolution follows importability per call
    factor = np.ones((2, 3), dtype=np.int8)
    packed = k.pack_bits(factor)
    assert packed.shape == (2, 1)


def test_dispatch_env_override(monkeypatch):
    monkeypatch.setenv("QUANTIBBLER_BACKEND", "numpy-reference")
    assert k.backend() == "numpy-reference"
    # force the extension flag on: env var still wins
    monkeypatch.setattr(k, "HAVE_KERNELS", True)
    monkeypatch.setattr(
        k, "_kernels", type("FakeExt", (), {"pack_bits": staticmethod(lambda f: f)})()
    )
    x = np.array([[1, -1]], dtype=np.int8)
    out = k.pack_bits(x)
    assert np.array_equal(out, k.pack_bits(x, backend="numpy-reference"))


def test_dispatch_env_override_rust_missing_raises(monkeypatch):
    monkeypatch.setenv("QUANTIBBLER_BACKEND", "rust")
    monkeypatch.setattr(k, "HAVE_KERNELS", False)
    monkeypatch.setattr(k, "_kernels", None)
    with pytest.raises(RuntimeError) as excinfo:
        k.sign_estimate(np.ones(3))
    msg = str(excinfo.value)
    assert "quantibbler_kernels" in msg
    assert "kernels" in msg


def test_dispatch_explicit_rust_missing_raises(no_extension):
    with pytest.raises(RuntimeError) as excinfo:
        k.pack_bits(np.ones((1, 3), dtype=np.int8), backend="rust")
    msg = str(excinfo.value)
    assert "quantibbler_kernels" in msg
    assert "kernels" in msg and "extra" in msg


def test_dispatch_explicit_numpy_always_works(no_extension):
    x = np.array([0.0, -1.0, 2.0])
    out = k.sign_estimate(x, backend="numpy-reference")
    assert np.array_equal(out, np.array([1, -1, 1], dtype=np.int8))
    packed = k.pack_bits(np.array([[-1, 1]], dtype=np.int8), backend="numpy-reference")
    assert packed.dtype == np.uint8


def test_dispatch_env_var_numpy_reference_beats_importable_extension(monkeypatch):
    monkeypatch.setenv("QUANTIBBLER_BACKEND", "numpy-reference")
    # even with a fake importable extension recorded, env var wins
    calls = []

    class FakeExt:
        @staticmethod
        def sign_estimate(x):
            calls.append(1)
            return x

    monkeypatch.setattr(k, "HAVE_KERNELS", True)
    monkeypatch.setattr(k, "_kernels", FakeExt())
    k.sign_estimate(np.ones(2))
    assert not calls, "env var must select NumPy reference, not the extension"


def test_dispatch_invalid_backend_kwarg_raises():
    with pytest.raises(ValueError):
        k.pack_bits(np.ones((1, 2), dtype=np.int8), backend="cuda")  # type: ignore


def test_dispatch_invalid_env_var_raises(monkeypatch):
    monkeypatch.setenv("QUANTIBBLER_BACKEND", "opencl")
    with pytest.raises(ValueError):
        k.backend()


def test_dispatch_identical_signatures_on_both_backends(no_extension):
    """Every kernel exists with identical signatures on both backends."""
    import inspect

    for name in _KERNELS:
        public = getattr(k, name)
        params = inspect.signature(public).parameters
        assert "backend" in params, f"{name} lacks the backend kwarg"
    # NumPy references exist for all six, same positional surface
    for name in _KERNELS:
        assert callable(getattr(k, f"_numpy_{name}"))


def test_dispatch_env_restored_between_tests(monkeypatch):
    monkeypatch.delenv("QUANTIBBLER_BACKEND", raising=False)
    # backend() auto-detects; the two baseline tests must stay green
    assert k.backend() in {"rust", "numpy-reference"}


# ---------------------------------------------------------------------------
# Dtype-tagged product dispatch (change qk-kernels-dtype-generic, design D5).
# The routing contract: ``float64`` keeps the v0.1 entry point, every other tag
# goes to the dtype-aware one — so an existing f64 call site cannot start
# taking a different numerical path by accident.
# ---------------------------------------------------------------------------


class _RecordingExt:
    """A stand-in extension recording which entry point each call reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def _entry(*_args: object) -> np.ndarray:
            self.calls.append(name)
            return np.zeros(2)

        return _entry


@pytest.mark.parametrize(
    ("product", "dtype", "expected"),
    [
        ("packed_matvec", "float64", "packed_matvec"),
        ("packed_matvec", "float32", "packed_matvec_dtype"),
        ("packed_matvec", "bfloat16", "packed_matvec_dtype"),
        ("packed_matmul", "float64", "packed_matmul"),
        ("packed_matmul", "float32", "packed_matmul_dtype"),
        ("packed_matmul", "bfloat16", "packed_matmul_dtype"),
    ],
)
def test_dispatch_dtype_tag_routes_to_the_matching_entry_point(
    monkeypatch, product: str, dtype: str, expected: str
):
    ext = _RecordingExt()
    monkeypatch.setattr(k, "HAVE_KERNELS", True)
    monkeypatch.setattr(k, "_kernels", ext)
    packed = k.pack_bits(np.ones((2, 4), dtype=np.int8), backend="numpy-reference")
    operand = (
        np.ones(4, dtype=np.float32) if product == "packed_matvec" else np.ones((4, 1), np.float32)
    )

    getattr(k, product)(packed, 4, operand, dtype=dtype, backend="rust")
    assert ext.calls == [expected], f"{product} dtype={dtype} dispatched through {ext.calls}"
