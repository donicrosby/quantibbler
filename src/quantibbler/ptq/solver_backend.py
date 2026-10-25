"""Array namespace for the LB-ADMM solver: NumPy reference and torch accelerator.

The solver's algebra exists once. Which arithmetic executes it is selected at
the entry point, so the NumPy path stays the determinism reference (it is what
the Rust kernels are validated against) and the torch path can put the same
operations on an accelerator.

The surface is deliberately *explicit* rather than ``__getattr__``-delegated: a
backend that silently resolved an operation the solver never vetted would turn a
porting mistake into a different numerical path, which is the failure this
module exists to make impossible. Every operation the solver uses is named here;
anything else fails at the call site.

Sign semantics are NOT normalized. ``sign(0)`` is 0 in both libraries, and the
magnitude-balance step's documented ``sign(0) = +1`` convention is applied by
the solver, not here -- a backend that quietly changed it would move packed
signs without moving any test that checks magnitudes.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DEFAULT_SOLVER_BACKEND",
    "NumpyBackend",
    "SOLVER_BACKEND_ENV",
    "TorchBackend",
    "available_backends",
    "get_backend",
    "resolve_device",
    "solver_backend_provenance",
]

# Production execution is the accelerated backend. NumPy is retained as the
# parity oracle — the reference the torch lane is accepted against — and is
# reachable only by requesting it by name. No production path resolves to it.
DEFAULT_SOLVER_BACKEND = "torch"

# Env knob for the backend, same idiom as ``QK_RHO_END``: unset resolves to the
# config default, a valid value overrides it, and anything else RAISES naming the
# variable. Never a silent fallback -- a run must not be mislabelled about the
# arm it executed.
SOLVER_BACKEND_ENV = "SOLVER_BACKEND"
_SOLVER_BACKEND_NAMES = ("numpy", "torch")


def solver_backend_provenance(default: str) -> tuple[str, str]:
    """Effective backend name and where it came from: ``"env"`` or ``"config"``."""
    raw = os.environ.get(SOLVER_BACKEND_ENV)
    if raw is None:
        return default, "config"
    value = raw.strip().lower()
    if value not in _SOLVER_BACKEND_NAMES:
        msg = f"{SOLVER_BACKEND_ENV} must be one of {_SOLVER_BACKEND_NAMES}, got {raw!r}"
        raise ValueError(msg) from None
    return value, "env"


def resolve_device(name: str, *, device: str | None = None) -> str:
    """The device a backend will actually execute on, without constructing it.

    One source of truth for the resolver AND for the fingerprint: the digest must
    record the device the solve really used, otherwise ``solver_backend="torch"``
    hashes identically for a cuda write and a cpu resume -- real numerics
    (reduction order) hidden behind an identical label.

    Diagnostic reading (ruled): an EXPLICIT ``device`` is honoured, so
    ``resolve_device("torch", device="cpu")`` is the parity/diagnostic path and
    the only thing that makes the numpy-oracle comparison runnable on a CPU-only
    sandbox. The IMPLICIT resolution refuses without an accelerator: a torch
    backend that silently fell back to CPU would report CPU timings under an
    acceleration label.
    """
    if name == "numpy":
        # No device dimension: NumPy runs where the interpreter runs.
        return "n/a"
    if name != "torch":
        raise ValueError(f"unknown solver backend {name!r}: expected 'numpy' or 'torch'")
    import torch  # noqa: PLC0415

    available = bool(torch.cuda.is_available())
    if device is not None:
        if device.startswith("cuda") and not available:
            raise RuntimeError(
                f"solver backend 'torch' requested device={device!r} "
                "but no accelerator is available"
            )
        return device
    if not available:
        raise RuntimeError(
            "solver backend 'torch' requires an accelerator: implicit resolution "
            "refuses to run torch on CPU (it is a slowdown wearing an "
            "acceleration's name). Pass an explicit device='cpu' for the "
            "diagnostic/parity path, or select backend='numpy'."
        )
    return "cuda"


class _LinalgProxy:
    """The ``linalg`` namespace, re-exposed per backend.

    ``eigh``/``svd``/``cholesky``/``solve``/``eigvalsh``/``norm`` have matching
    signatures across NumPy and torch, so the proxy adds no behaviour -- it
    exists so the solver writes ``xp.linalg.cholesky(...)`` in both lanes and the
    call sites do not fork.
    """

    __slots__ = ("_linalg", "LinAlgError")

    def __init__(self, linalg: Any) -> None:
        self._linalg = linalg
        # the solver catches xp.linalg.LinAlgError; both libraries expose the
        # same exception name on their linalg namespace.
        self.LinAlgError = linalg.LinAlgError

    def cholesky(self, a: Any) -> Any:
        return self._linalg.cholesky(a)

    def eigvalsh(self, a: Any) -> Any:
        return self._linalg.eigvalsh(a)

    def eigh(self, a: Any) -> tuple[Any, Any]:
        return self._linalg.eigh(a)

    def svd(self, a: Any, full_matrices: bool = True) -> tuple[Any, Any, Any]:
        return self._linalg.svd(a, full_matrices=full_matrices)

    def solve(self, a: Any, b: Any) -> Any:
        return self._linalg.solve(a, b)

    def norm(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        return self._linalg.norm(a, *args, **kwargs)


class _BaseBackend:
    """Shared surface; subclasses bind ``_np``/``_xp`` and the differing ops."""

    name = ""

    def __init__(self, module: Any) -> None:
        self._m = module
        self.linalg = _LinalgProxy(module.linalg)
        self.LinAlgError = module.linalg.LinAlgError

    # --- construction / conversion ------------------------------------------
    def asarray(self, x: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def ascontiguousarray(self, x: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def dtype(self, spec: str) -> Any:
        raise NotImplementedError

    def scalar(self, value: Any, dtype: Any = None) -> Any:
        """A 0-d value of ``dtype``.

        The solver builds packed signs with ``np.int8(1)``/``np.int8(-1)``; torch
        has no per-dtype scalar constructor, so the backend supplies the one
        spelling both lanes can use.
        """
        raise NotImplementedError

    @property
    def ndarray(self) -> Any:
        """The array *type*, for annotations that must stay backend-true."""
        raise NotImplementedError

    # --- conversions the solver uses as methods ------------------------------
    def astype(self, x: Any, dtype: Any) -> Any:
        """``x.astype(dtype)`` in NumPy terms; ``.to(dtype)`` for torch.

        Method-level rather than ``np.``-prefixed, which is exactly why a grep
        for ``np.`` misses it: ``astype`` exists on NumPy arrays and NOT on
        torch tensors, so a port that only rewrote the ``np.`` call sites would
        still fail here at runtime.
        """
        raise NotImplementedError

    def clone(self, x: Any) -> Any:
        """``x.copy()`` in NumPy terms; ``.clone()`` for torch."""
        raise NotImplementedError

    # --- elementwise ---------------------------------------------------------
    def sqrt(self, x: Any) -> Any:
        return self._m.sqrt(x)

    def abs(self, x: Any) -> Any:
        return self._m.abs(x)

    def sign(self, x: Any) -> Any:
        return self._m.sign(x)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._m.maximum(a, b)

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        return self._m.where(cond, a, b)

    def isfinite(self, x: Any) -> Any:
        return self._m.isfinite(x)

    def nanmax(self, x: Any) -> Any:
        raise NotImplementedError

    # --- reductions ----------------------------------------------------------
    def any(self, x: Any) -> Any:
        return self._m.any(x)

    def all(self, x: Any) -> Any:
        return self._m.all(x)

    def mean(self, x: Any, axis: Any = None) -> Any:
        return self._m.mean(x, axis=axis)

    def count_nonzero(self, x: Any) -> Any:
        return self._m.count_nonzero(x)

    def array_equal(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def argsort(self, x: Any) -> Any:
        return self._m.argsort(x)

    # --- construction --------------------------------------------------------
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def zeros_like(self, x: Any) -> Any:
        return self._m.zeros_like(x)

    def full(self, shape: Any, value: Any, dtype: Any = None) -> Any:
        raise NotImplementedError

    def eye(self, n: int, dtype: Any = None) -> Any:
        raise NotImplementedError

    def diag(self, x: Any) -> Any:
        return self._m.diag(x)

    def outer(self, a: Any, b: Any) -> Any:
        return self._m.outer(a, b)

    # --- scalar constants ----------------------------------------------------
    @property
    def inf(self) -> float:
        return float(self._m.inf)

    def finfo(self, dtype: Any) -> Any:
        raise NotImplementedError


class NumpyBackend(_BaseBackend):
    """The reference backend. Float64 by default, as the solver documents."""

    name = "numpy"

    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        super().__init__(np)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        return self._m.asarray(x, dtype=dtype)

    def ascontiguousarray(self, x: Any, dtype: Any = None) -> Any:
        return self._m.ascontiguousarray(x, dtype=dtype)

    def dtype(self, spec: str) -> Any:
        return self._m.dtype(spec)

    def scalar(self, value: Any, dtype: Any = None) -> Any:
        return self._m.array(value, dtype=dtype)

    @property
    def ndarray(self) -> Any:
        return self._m.ndarray

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.astype(dtype)

    def clone(self, x: Any) -> Any:
        return x.copy()

    def nanmax(self, x: Any) -> Any:
        return self._m.nanmax(x)

    def array_equal(self, a: Any, b: Any) -> Any:
        return self._m.array_equal(a, b)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self._m.zeros(shape, dtype=dtype)

    def full(self, shape: Any, value: Any, dtype: Any = None) -> Any:
        return self._m.full(shape, value, dtype=dtype)

    def eye(self, n: int, dtype: Any = None) -> Any:
        return self._m.eye(n, dtype=dtype)

    def finfo(self, dtype: Any) -> Any:
        return self._m.finfo(dtype)


class TorchBackend(_BaseBackend):
    """Accelerator backend. Same algebra; arithmetic on ``device``.

    ``torch.as_tensor`` copies when the source is not already a tensor on the
    requested device, which is the behaviour the solver's ``asarray(..., dtype)``
    call sites already assume. ``ascontiguousarray`` becomes ``contiguous()``:
    torch has no separate non-contiguous type, so the operation is an identity
    for tensor input and a no-op copy otherwise, matching NumPy's contract of
    returning something the kernels can consume.
    """

    name = "torch"

    def __init__(self, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415

        super().__init__(torch)
        self.device = device
        self.torch = torch

    def _t(self, x: Any) -> Any:
        """Coerce a Python scalar to a tensor.

        NumPy's elementwise functions accept Python floats; torch's do not, and
        the solver legitimately hands them scalars (``isfinite(0.0)``,
        ``sqrt(n_v / n_u)``). Without this the port fails at runtime on inputs
        the reference implementation accepts happily -- a difference in calling
        convention, not in arithmetic, so it is absorbed here.
        """
        if isinstance(x, self.torch.Tensor):
            return x
        return self.torch.as_tensor(x, device=self.device)

    def sqrt(self, x: Any) -> Any:
        return self.torch.sqrt(self._t(x))

    def abs(self, x: Any) -> Any:
        return self.torch.abs(self._t(x))

    def isfinite(self, x: Any) -> Any:
        return self.torch.isfinite(self._t(x))

    def maximum(self, a: Any, b: Any) -> Any:
        return self.torch.maximum(self._t(a), self._t(b))

    def sign(self, x: Any) -> Any:
        """``sign`` that preserves NaN, matching NumPy.

        torch.sign(nan) returns 0.0 where np.sign(nan) returns nan. Left alone
        that is a silent-corruption path: a non-finite latent would be packed as
        a legitimate sign instead of tripping the solver's isfinite divergence
        gate. Parity includes the non-finite case, so it is restored here rather
        than papered over in the caller.
        """
        return self.torch.where(self.torch.isnan(x), x, self.torch.sign(x))

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        return self.torch.as_tensor(x, dtype=dtype, device=self.device)

    def ascontiguousarray(self, x: Any, dtype: Any = None) -> Any:
        tensor = self.torch.as_tensor(x, dtype=dtype, device=self.device)
        return tensor.contiguous()

    def dtype(self, spec: str) -> Any:
        return getattr(self.torch, spec)

    def scalar(self, value: Any, dtype: Any = None) -> Any:
        return self.torch.tensor(value, dtype=dtype, device=self.device)

    @property
    def ndarray(self) -> Any:
        return self.torch.Tensor

    def astype(self, x: Any, dtype: Any) -> Any:
        return x.to(dtype)

    def clone(self, x: Any) -> Any:
        return x.clone()

    def nanmax(self, x: Any) -> Any:
        """``np.nanmax``: the max ignoring NaN. torch has no ``nanmax``.

        Built from ``nan_to_num`` + ``amax``. One documented divergence: an
        all-NaN input returns ``-inf`` here where NumPy returns ``nan`` and
        warns. The solver's single call site sits behind an ``isfinite`` gate,
        so an all-NaN input is already an error condition upstream.
        """
        return self.torch.amax(self.torch.nan_to_num(x, nan=float("-inf")))

    def array_equal(self, a: Any, b: Any) -> Any:
        return bool(self.torch.equal(a, b))

    @staticmethod
    def _shape(shape: Any) -> Any:
        """NumPy accepts an int as a shape; torch requires a tuple.

        The harness passes ``zeros(n)``-style int sizes, so the calling style has
        to be absorbed here or the port fails on inputs the reference accepts.
        """
        return (shape,) if isinstance(shape, int) else tuple(shape)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self.torch.zeros(self._shape(shape), dtype=dtype, device=self.device)

    def full(self, shape: Any, value: Any, dtype: Any = None) -> Any:
        # torch.full rejects a 0-d tensor as the fill value; NumPy accepts an
        # array-valued fill and broadcasts it. The solver passes 1/sqrt(n).
        fill = float(value) if isinstance(value, self.torch.Tensor) else value
        return self.torch.full(self._shape(shape), fill, dtype=dtype, device=self.device)

    def eye(self, n: int, dtype: Any = None) -> Any:
        return self.torch.eye(self._shape(n)[0], dtype=dtype, device=self.device)

    def finfo(self, dtype: Any) -> Any:
        return self.torch.finfo(dtype)


def get_backend(name: str | None = None, *, device: str | None = None) -> _BaseBackend:
    """Resolve a backend by name, defaulting to :data:`DEFAULT_SOLVER_BACKEND`.

    There is deliberately no NumPy fallback: if the accelerator cannot be used,
    the call fails rather than quietly returning the reference implementation.

    ``torch`` without an accelerator **and without an explicit device** is
    refused (see :func:`resolve_device`): a torch backend that silently fell back
    to CPU would report CPU timings under an acceleration label. An explicit
    ``device="cpu"`` remains available as the parity/diagnostic path.
    """
    if name is None:
        name = DEFAULT_SOLVER_BACKEND
    if name == "numpy":
        return NumpyBackend()
    if name == "torch":
        return TorchBackend(device=resolve_device("torch", device=device))
    raise ValueError(f"unknown solver backend {name!r}: expected 'numpy' or 'torch'")


def available_backends() -> tuple[str, ...]:
    """Backend names this interpreter can actually construct."""
    names = ["numpy"]
    try:
        import torch  # noqa: PLC0415, F401

        names.append("torch")
    except ImportError:
        pass
    return tuple(names)
