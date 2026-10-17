"""Kernel dispatch: Rust extension when available, NumPy reference otherwise.

The NumPy implementations are the executable specification of every kernel;
the Rust extension must match within the tolerances pinned in
``openspec/config.yaml`` (verification-gates section): float64 outputs at
relative tolerance 1e-12, packed outputs byte-identical.

Dispatch resolution order, applied per call:

1. explicit ``backend=`` keyword (``"rust"`` | ``"numpy-reference"``);
2. ``QUANTIBBLER_BACKEND`` environment variable;
3. automatic: the Rust extension when importable, else the NumPy reference.

Auto mode silently falls back to the NumPy reference when
``quantibbler_kernels`` is absent. An explicit Rust request (kwarg or env
var) without the extension raises :class:`RuntimeError` naming the missing
module and the ``kernels`` extra.

Packed layout (the qpk1 bitplane contract, in-memory): packed output is
``uint8`` with one row of ``ceil(n_bits / 8)`` bytes per input row,
row-major element order, little-endian bit order within each byte
(element ``8j + k`` goes to bit ``k`` of byte ``j``), ``+1`` maps to bit 1,
``-1`` maps to bit 0, and trailing pad bits are written as zero and ignored
on read.

Zero-handling: :func:`sign_estimate` maps ``x > 0`` to ``+1`` and
``x <= 0`` to ``-1`` with the documented tie-break that every exact zero
(``+0.0`` and ``-0.0``) maps to ``+1`` (``sign(0) = +1``); no output value
is ever zero.
"""

from __future__ import annotations

import math
import os

import numpy as np

try:  # pragma: no cover - depends on install extras
    import quantibbler_kernels as _kernels  # type: ignore[import-not-found]

    HAVE_KERNELS = True
except ImportError:
    _kernels = None  # type: ignore[assignment]
    HAVE_KERNELS = False

_VALID_BACKENDS = ("rust", "numpy-reference")
_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_TWO_POW_53 = float(2**53)


def backend() -> str:
    """Name of the active kernel backend: ``rust`` or ``numpy-reference``."""
    return _resolve_backend(None)


def _resolve_backend(explicit: str | None) -> str:
    """Resolve the backend per call: kwarg -> env var -> auto."""
    if explicit is not None:
        if explicit not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {explicit!r}")
        return explicit
    env = os.environ.get("QUANTIBBLER_BACKEND")
    if env:
        if env not in _VALID_BACKENDS:
            raise ValueError(f"QUANTIBBLER_BACKEND must be one of {_VALID_BACKENDS}, got {env!r}")
        return env
    return "rust" if HAVE_KERNELS else "numpy-reference"


def _rust_has(name: str) -> bool:
    """True when the extension is importable and provides kernel ``name``."""
    return HAVE_KERNELS and _kernels is not None and hasattr(_kernels, name)


def _call_rust(name: str, *args: object, **kwargs: object) -> object:
    """Forward to the extension (caller checked ``_wants_rust`` first)."""
    return getattr(_kernels, name)(*args, **kwargs)


def _wants_rust(name: str, explicit: str | None) -> bool:
    """Decide whether ``name`` runs on the Rust extension for this call.

    Resolution: explicit kwarg -> ``QUANTIBBLER_BACKEND`` -> auto. An
    explicit Rust request without the extension (or without the specific
    kernel) raises RuntimeError; auto mode falls back to NumPy silently —
    including when an importable extension predates the v0.1 kernel
    surface.
    """
    mode = _resolve_backend(explicit)
    if mode == "numpy-reference":
        return False
    if _rust_has(name):
        return True
    if explicit is not None or os.environ.get("QUANTIBBLER_BACKEND") == "rust":
        raise RuntimeError(
            f"backend='rust' requested but the quantibbler_kernels extension "
            f"is not available (or does not provide {name!r}); install "
            f"quantibbler with the 'kernels' extra "
            f"(e.g. `uv sync --extra kernels`) to use the Rust kernels"
        )
    return False


# ---------------------------------------------------------------------------
# NumPy reference implementations (the executable specification)
# ---------------------------------------------------------------------------


def _numpy_pack_bits(factor: object) -> np.ndarray:
    """Pack a ±1 factor into uint8 bitplanes (row-major, LSB-first).

    Any dimensionality is flattened row-major to ``n x m`` first; byte
    ``j`` of a row holds elements ``8j .. 8j+7`` with element ``8j + k`` at
    bit ``k`` (LSB-first). ``+1 -> bit 1``, ``-1 -> bit 0``, pad bits zero.
    """
    x = np.asarray(factor)
    if x.ndim == 0:
        raise ValueError("pack_bits expects an array with at least one dimension")
    xf = np.asarray(x, dtype=np.float64).reshape(x.shape[0], x.shape[-1])
    n_rows, n_cols = xf.shape
    n_words = (n_cols + 7) // 8
    out = np.zeros((n_rows, n_words), dtype=np.uint8)
    if n_cols == 0:
        return out
    for i in range(n_rows):
        for j in range(n_words):
            byte = 0
            for k in range(8):
                idx = j * 8 + k
                if idx >= n_cols:
                    break
                val = xf[i, idx]
                if val == 1.0:
                    byte |= 1 << k
                elif val != -1.0:
                    raise ValueError(
                        "pack_bits expects values in {-1, +1}; got "
                        f"{val!r} at row {i}, column {idx}"
                    )
            out[i, j] = byte
    return out


def _numpy_unpack_bits(packed: object, n_bits: int, dtype: object = "int8") -> np.ndarray:
    """Restore a ±1 factor from packed bitplanes with explicit logical width.

    Pad bits beyond ``n_bits`` are dropped, never validated. ``dtype`` must
    be ``int8`` or ``float64``.
    """
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("unpack_bits expects a 2-D uint8 packed array")
    if not _is_integral(n_bits) or int(n_bits) < 0:
        raise ValueError(f"n_bits must be a non-negative integer, got {n_bits!r}")
    n_bits_i = int(n_bits)
    n_rows, n_words = p.shape
    if n_bits_i > n_words * 8:
        raise ValueError(
            f"n_bits={n_bits_i} exceeds the packed width {n_words} ({n_words * 8} bits)"
        )
    target = _unpack_dtype(dtype)
    out = np.empty((n_rows, n_bits_i), dtype=target)
    for i in range(n_rows):
        for k in range(n_bits_i):
            bit = (int(p[i, k >> 3]) >> (k & 7)) & 1
            out[i, k] = 1 if bit == 1 else -1
    return out


def _unpack_dtype(dtype: object) -> np.dtype:
    d = np.dtype(dtype)
    if d == np.dtype(np.int8):
        return d
    if d == np.dtype(np.float64):
        return d
    raise ValueError(f"unpack_bits dtype must be 'int8' or 'float64', got {dtype!r}")


def _is_integral(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, float):
        return float(value).is_integer()
    return False


def _numpy_sign_estimate(x: object) -> np.ndarray:
    """int8 sign of ``x``: ``>0 -> +1``, ``<=0 -> -1``, exact zeros -> ``+1``.

    The zero tie-break (``sign(0) = +1``, including ``-0.0``) is the
    documented contract so the output feeds ``pack_bits`` directly.
    """
    xf = np.asarray(x, dtype=np.float64)
    return np.where(xf >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)


def _decode_row_bit(packed: np.ndarray, row: int, k: int) -> int:
    return (int(packed[row, k >> 3]) >> (k & 7)) & 1


def _numpy_packed_matvec(packed: object, n_bits: int, vec: object) -> np.ndarray:
    """``unpack_bits(packed, n_bits, float64) @ vec`` without materializing.

    Accumulation is pinned left-to-right in row-element order in every
    backend so float results are bit-identical, not merely 1e-12-close.
    """
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("packed_matvec expects a 2-D uint8 packed array")
    if not _is_integral(n_bits) or int(n_bits) < 0:
        raise ValueError(f"n_bits must be a non-negative integer, got {n_bits!r}")
    n_bits_i = int(n_bits)
    v = np.asarray(vec, dtype=np.float64)
    n_rows, n_words = p.shape
    if n_bits_i > n_words * 8:
        raise ValueError(
            f"n_bits={n_bits_i} exceeds the packed width {n_words} ({n_words * 8} bits)"
        )
    if v.ndim != 1 or v.shape[0] != n_bits_i:
        raise ValueError(f"vec must have shape ({n_bits_i},), got {v.shape}")
    out = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        acc = 0.0
        for k in range(n_bits_i):
            s = 1.0 if _decode_row_bit(p, i, k) == 1 else -1.0
            acc += s * float(v[k])
        out[i] = acc
    return out


def _numpy_packed_matmul(packed: object, n_bits: int, mat: object) -> np.ndarray:
    """Column-batched :func:`_numpy_packed_matvec` (packed x dense-float only)."""
    p = np.asarray(packed, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("packed_matmul expects a 2-D uint8 packed array")
    if not _is_integral(n_bits) or int(n_bits) < 0:
        raise ValueError(f"n_bits must be a non-negative integer, got {n_bits!r}")
    n_bits_i = int(n_bits)
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("packed_matmul expects a 2-D float64 matrix")
    n_rows, n_words = p.shape
    if n_bits_i > n_words * 8:
        raise ValueError(
            f"n_bits={n_bits_i} exceeds the packed width {n_words} ({n_words * 8} bits)"
        )
    if m.shape[0] != n_bits_i:
        raise ValueError(f"mat must have {n_bits_i} rows, got {m.shape[0]}")
    k_cols = m.shape[1]
    out = np.zeros((n_rows, k_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(k_cols):
            acc = 0.0
            for t in range(n_bits_i):
                s = 1.0 if _decode_row_bit(p, i, t) == 1 else -1.0
                acc += s * float(m[t, j])
            out[i, j] = acc
    return out


def _splitmix_matrix(seed: int, rank: int) -> np.ndarray:
    """Deterministic ``rank x rank`` float64 matrix in [-1, 1) from splitmix64.

    Filled row-major; mirrored bit-exactly by the Rust implementation so a
    seeded R0 matches across backends.
    """
    state = int(seed) & _MASK64
    out = np.empty((rank, rank), dtype=np.float64)
    for i in range(rank):
        for j in range(rank):
            state = (state + _GOLDEN) & _MASK64
            z = state
            z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
            z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
            z = z ^ (z >> 31)
            u01 = (z >> 11) / _TWO_POW_53
            out[i, j] = u01 * 2.0 - 1.0
    return out


def _gram_schmidt(a: np.ndarray) -> np.ndarray:
    """Modified Gram-Schmidt orthonormalization, column by column.

    Every dot product and update is an explicit left-to-right float64 loop
    so NumPy and Rust produce bit-identical rotations.
    """
    rank = a.shape[0]
    r0 = np.array(a, dtype=np.float64, copy=True)
    for j in range(rank):
        v = r0[:, j].copy()
        for i in range(j):
            d = 0.0
            for t in range(rank):
                d += r0[t, i] * v[t]
            for t in range(rank):
                v[t] -= d * r0[t, i]
        s = 0.0
        for t in range(rank):
            s += v[t] * v[t]
        nrm = math.sqrt(s)
        if nrm == 0.0:
            raise ValueError("joint_itq produced a degenerate zero column building R0")
        for t in range(rank):
            r0[t, j] = v[t] / nrm
    return r0


def _matmul_lr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` with explicit left-to-right float64 accumulation."""
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    for i in range(m):
        for j in range(p):
            acc = 0.0
            for t in range(n):
                acc += a[i, t] * b[t, j]
            out[i, j] = acc
    return out


def _matmul_at_b_lr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a.T @ b`` with explicit left-to-right float64 accumulation."""
    n, m = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    for i in range(m):
        for j in range(p):
            acc = 0.0
            for t in range(n):
                acc += a[t, i] * b[t, j]
            out[i, j] = acc
    return out


def _numpy_joint_itq(
    u_hat: object,
    v_hat: object,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Joint-ITQ alternating Procrustes ([LB2] Sec 4.4, Algorithm 1 lines 6-11).

    Builds ``Z = [u_hat; v_hat]``, seeds a random orthogonal R0 (splitmix64
    fill + modified Gram-Schmidt when ``seed`` is given; OS entropy
    otherwise), and alternates ``B <- sign(Z R)`` / ``R <- Psi Phi^T`` from
    the r x r SVD of ``B^T Z`` until ``||R_new - R_old||_F <= tol`` or
    ``max_iter``. Returns ``(u_t, v_t, n_iter, converged)`` with int8 ±1
    factors.
    """
    u = np.asarray(u_hat, dtype=np.float64)
    v = np.asarray(v_hat, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("joint_itq expects 2-D inputs u_hat (dout, r), v_hat (din, r)")
    if not _is_integral(max_iter) or int(max_iter) < 0:
        raise ValueError(f"max_iter must be a non-negative integer, got {max_iter!r}")
    max_iter_i = int(max_iter)
    if seed is not None and (not _is_integral(seed) or int(seed) < 0):
        raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")
    dout, rank = u.shape
    din, rank_v = v.shape
    if rank != rank_v:
        raise ValueError(f"joint_itq rank mismatch: u_hat r={rank}, v_hat r={rank_v}")
    if rank == 0 or dout + din == 0:
        raise ValueError("joint_itq expects non-empty factors")
    if seed is None:
        raw = np.random.default_rng().random((rank, rank)) * 2.0 - 1.0
    else:
        raw = _splitmix_matrix(int(seed), rank)
    rr = _gram_schmidt(raw)
    z = np.concatenate([u, v], axis=0)
    n_iter = 0
    converged = False
    for it in range(1, max_iter_i + 1):
        zr = _matmul_lr(z, rr)
        b = (zr >= 0.0).astype(np.float64) * 2.0 - 1.0
        btz = _matmul_at_b_lr(b, z)
        umat, _s, vt = np.linalg.svd(btz, full_matrices=True)
        r_new = np.empty((rank, rank), dtype=np.float64)
        for i in range(rank):
            for j in range(rank):
                acc = 0.0
                for t in range(rank):
                    acc += vt[t, i] * umat[j, t]
                r_new[i, j] = acc
        delta = 0.0
        for i in range(rank):
            for j in range(rank):
                d = r_new[i, j] - rr[i, j]
                delta += d * d
        delta = math.sqrt(delta)
        rr = r_new
        n_iter = it
        if tol > 0.0 and delta <= tol:
            converged = True
            break
    zr = _matmul_lr(z, rr)
    signs = np.where(zr >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    return signs[:dout].copy(), signs[dout:].copy(), n_iter, converged


# ---------------------------------------------------------------------------
# Public kernel surface (dispatch per design D6)
# ---------------------------------------------------------------------------


def pack_bits(factor: object, *, backend: str | None = None) -> np.ndarray:
    """Pack a ±1 factor into uint8 bitplanes; see ``_numpy_pack_bits``."""
    if _wants_rust("pack_bits", backend):
        return _call_rust("pack_bits", factor)  # type: ignore[return-value]
    return _numpy_pack_bits(factor)


def unpack_bits(
    packed: object, n_bits: int, dtype: object = "int8", *, backend: str | None = None
) -> np.ndarray:
    """Restore a ±1 factor from packed bitplanes; see ``_numpy_unpack_bits``."""
    target = _unpack_dtype(dtype)
    if _wants_rust("unpack_bits", backend):
        return _call_rust("unpack_bits", packed, n_bits, str(target))  # type: ignore
    return _numpy_unpack_bits(packed, n_bits, target)


def sign_estimate(x: object, *, backend: str | None = None) -> np.ndarray:
    """int8 sign with ``sign(0) = +1``; see ``_numpy_sign_estimate``."""
    if _wants_rust("sign_estimate", backend):
        return _call_rust("sign_estimate", x)  # type: ignore[return-value]
    return _numpy_sign_estimate(x)


def packed_matvec(
    packed: object, n_bits: int, vec: object, *, backend: str | None = None
) -> np.ndarray:
    """Packed x dense-float matvec; see ``_numpy_packed_matvec``."""
    if _wants_rust("packed_matvec", backend):
        return _call_rust("packed_matvec", packed, n_bits, vec)  # type: ignore
    return _numpy_packed_matvec(packed, n_bits, vec)


def packed_matmul(
    packed: object, n_bits: int, mat: object, *, backend: str | None = None
) -> np.ndarray:
    """Packed x dense-float matmul; see ``_numpy_packed_matmul``."""
    if _wants_rust("packed_matmul", backend):
        return _call_rust("packed_matmul", packed, n_bits, mat)  # type: ignore
    return _numpy_packed_matmul(packed, n_bits, mat)


def joint_itq(
    u_hat: object,
    v_hat: object,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
    seed: int | None = None,
    backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Joint-ITQ alternating Procrustes; see ``_numpy_joint_itq``."""
    if _wants_rust("joint_itq", backend):
        return _call_rust(  # type: ignore[return-value]
            "joint_itq", u_hat, v_hat, max_iter=max_iter, tol=tol, seed=seed
        )
    return _numpy_joint_itq(u_hat, v_hat, max_iter=max_iter, tol=tol, seed=seed)
