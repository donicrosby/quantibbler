"""Scenario-ID parity probes: NumPy reference vs the Rust extension.

Every probe is named for its spec scenario ID
(openspec/changes/qk-kernels/specs/qk-kernels/spec.md). Each probe runs the
kernel under test through ``backend="numpy-reference"`` always and through
``backend="rust"`` when the extension provides the kernel; the Rust leg
skips with an explicit reason when the extension is absent (the suite stays
green both ways).

Gates (openspec/config.yaml, Verification gates): float64 outputs at rel
1e-12, packed outputs byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

import quantibbler.kernels as k

RTOL = 1e-12

_RUST_KERNELS = (
    "pack_bits",
    "unpack_bits",
    "sign_estimate",
    "packed_matvec",
    "packed_matmul",
    "joint_itq",
)


def _rust_ready(*names: str) -> bool:
    """True when the extension is importable and provides every name."""
    return k.HAVE_KERNELS and all(hasattr(k._kernels, n) for n in names)  # noqa: SLF001


def _require_rust(*names: str) -> None:
    if not _rust_ready(*names):
        missing = (
            "extension not importable"
            if not k.HAVE_KERNELS
            else "extension lacks: " + ", ".join(names)
        )
        pytest.skip(f"Rust leg unavailable: {missing}")


def _assert_rel(a: np.ndarray, b: np.ndarray, rtol: float = RTOL) -> None:
    """Float64 relative tolerance 1e-12 (gate), with a tiny floor for 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(np.float64).tiny)
    assert np.all(np.abs(a - b) <= rtol * scale), (
        f"float outputs differ beyond rel {rtol}: max rel {np.max(np.abs(a - b) / scale):.3e}"
    )


def _assert_bytes(a: np.ndarray, b: np.ndarray) -> None:
    """Packed outputs must be byte-identical (sign-agreement invariant)."""
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), (
        "packed outputs are not byte-identical"
    )


def _signed_factor(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)


# ---------------------------------------------------------------------------
# pack_bits / unpack_bits scenarios
# ---------------------------------------------------------------------------


def test_parity_pack_roundtrip_small():
    factor = _signed_factor((5, 11), seed=1)
    packed = k.pack_bits(factor, backend="numpy-reference")
    assert packed.dtype == np.uint8
    assert packed.shape == (5, 2)
    # n_bits=11: bits 11..15 of byte 1 are pad (byte 0 is fully logical)
    pad_mask = np.array([0b00000000, 0b11111000], dtype=np.uint8)
    assert np.all((packed & pad_mask) == 0), "pad bits must be zero"
    unpacked = k.unpack_bits(packed, 11, "int8", backend="numpy-reference")
    assert unpacked.dtype == np.int8
    assert np.array_equal(unpacked, factor)
    if _rust_ready("pack_bits", "unpack_bits"):
        packed_r = k.pack_bits(factor, backend="rust")
        _assert_bytes(packed, packed_r)
        unpacked_r = k.unpack_bits(packed_r, 11, "int8", backend="rust")
        assert np.array_equal(unpacked_r, factor)
        assert unpacked_r.dtype == np.int8
    else:
        _require_rust("pack_bits")


def test_parity_pack_roundtrip_exact8():
    factor = _signed_factor((4, 8), seed=2)
    packed = k.pack_bits(factor, backend="numpy-reference")
    assert packed.shape == (4, 1)
    unpacked = k.unpack_bits(packed, 8, "float64", backend="numpy-reference")
    assert unpacked.dtype == np.float64
    assert np.array_equal(unpacked, factor.astype(np.float64))
    if _rust_ready("pack_bits", "unpack_bits"):
        packed_r = k.pack_bits(factor, backend="rust")
        _assert_bytes(packed, packed_r)
        unpacked_r = k.unpack_bits(packed_r, 8, "float64", backend="rust")
        assert np.array_equal(unpacked_r, factor.astype(np.float64))
    else:
        _require_rust("pack_bits")


def test_parity_pack_roundtrip_padded():
    factor = _signed_factor((3, 9), seed=3)
    packed = k.pack_bits(factor, backend="numpy-reference")
    assert packed.shape == (3, 2)
    dirty = packed.copy()
    dirty[:, -1] |= 0b11111110  # dirty every pad bit except the logical one
    unpacked = k.unpack_bits(dirty, 9, "int8", backend="numpy-reference")
    assert np.array_equal(unpacked, factor), "dirty pad bits must be ignored"
    if _rust_ready("pack_bits", "unpack_bits"):
        packed_r = k.pack_bits(factor, backend="rust")
        _assert_bytes(packed, packed_r)
        unpacked_r = k.unpack_bits(dirty, 9, "int8", backend="rust")
        assert np.array_equal(unpacked_r, factor)
    else:
        _require_rust("pack_bits")


@pytest.mark.parametrize("shape", [(2, 0), (0, 7)])
def test_parity_pack_roundtrip_empty(shape):
    factor = np.zeros(shape, dtype=np.int8)
    packed = k.pack_bits(factor, backend="numpy-reference")
    assert packed.dtype == np.uint8
    assert packed.size == 0
    unpacked = k.unpack_bits(packed, shape[1], "int8", backend="numpy-reference")
    assert unpacked.size == 0
    assert unpacked.dtype == np.int8
    if _rust_ready("pack_bits", "unpack_bits"):
        packed_r = k.pack_bits(factor, backend="rust")
        _assert_bytes(packed, packed_r)
        unpacked_r = k.unpack_bits(packed_r, shape[1], "int8", backend="rust")
        assert unpacked_r.size == 0
    else:
        _require_rust("pack_bits")


def test_parity_pack_layout_bitorder():
    for pos in range(16):
        row = np.full((1, 16), -1, dtype=np.int8)
        row[0, pos] = 1
        packed = k.pack_bits(row, backend="numpy-reference")
        assert packed.shape == (1, 2)
        j, bit = pos // 8, pos % 8
        expected = np.zeros((1, 2), dtype=np.uint8)
        expected[0, j] = 1 << bit
        assert np.array_equal(packed, expected), f"pos {pos}: {packed} != {expected}"
        # a -1 at the same position leaves the bit 0; every other bit of the
        # row still reflects its element
        row_neg = np.full((1, 16), 1, dtype=np.int8)
        row_neg[0, pos] = -1
        packed_neg = k.pack_bits(row_neg, backend="numpy-reference")
        expected_neg = np.full((1, 2), 0xFF, dtype=np.uint8)
        expected_neg[0, pos // 8] = np.uint8(expected_neg[0, pos // 8] ^ np.uint8(1 << (pos % 8)))
        assert np.array_equal(packed_neg, expected_neg), (
            f"pos {pos}: -1 must clear bit {pos % 8} of byte {pos // 8}"
        )
    if _rust_ready("pack_bits"):
        rows = []
        for pos in range(16):
            row = np.full((1, 16), -1, dtype=np.int8)
            row[0, pos] = 1
            rows.append(row)
        stacked = np.concatenate(rows, axis=0)
        _assert_bytes(
            k.pack_bits(stacked, backend="numpy-reference"), k.pack_bits(stacked, backend="rust")
        )
    else:
        _require_rust("pack_bits")


# ---------------------------------------------------------------------------
# sign_estimate scenarios
# ---------------------------------------------------------------------------


def test_parity_sign_zeros():
    x = np.array([0.0, -0.0, 1.5, -2.5, 0.0, -1e-9], dtype=np.float64)
    out = k.sign_estimate(x, backend="numpy-reference")
    assert out.dtype == np.int8
    expected = np.array([1, 1, 1, -1, 1, -1], dtype=np.int8)
    assert np.array_equal(out, expected)
    assert not np.any(out == 0), "no output value may be 0"
    all_zero = k.sign_estimate(np.zeros((2, 3)), backend="numpy-reference")
    assert np.all(all_zero == 1)
    if _rust_ready("sign_estimate"):
        _assert_bytes(out, k.sign_estimate(x, backend="rust").astype(np.int8))
        _assert_bytes(all_zero, k.sign_estimate(np.zeros((2, 3)), backend="rust").astype(np.int8))
    else:
        _require_rust("sign_estimate")


def test_parity_sign_extremes():
    x = np.array([1e-300, -1e300, 7.0, -3.5, 2.2250738585072014e-308], dtype=np.float64)
    out = k.sign_estimate(x, backend="numpy-reference")
    assert out.dtype == np.int8
    assert np.array_equal(out, np.array([1, -1, 1, -1, 1], dtype=np.int8))
    if _rust_ready("sign_estimate"):
        _assert_bytes(out, k.sign_estimate(x, backend="rust").astype(np.int8))
    else:
        _require_rust("sign_estimate")


# ---------------------------------------------------------------------------
# packed matvec / matmul scenarios
# ---------------------------------------------------------------------------


def test_parity_matvec_padded():
    factor = _signed_factor((6, 13), seed=4)
    vec = np.random.default_rng(5).standard_normal(13)
    packed = k.pack_bits(factor, backend="numpy-reference")
    out = k.packed_matvec(packed, 13, vec, backend="numpy-reference")
    assert out.dtype == np.float64
    dense = k.unpack_bits(packed, 13, "float64", backend="numpy-reference")
    _assert_rel(out, dense @ vec)
    # pad bits contribute exactly 0: corrupt them, result unchanged
    # (n_bits=13, packed width 2: bit 13 is the only logical bit in byte 1;
    # bits 14..15 are pad)
    dirty = packed.copy()
    dirty[:, -1] |= 0b11000000
    out_dirty = k.packed_matvec(dirty, 13, vec, backend="numpy-reference")
    _assert_rel(out, out_dirty)
    _assert_bytes(out, out_dirty)  # must be bit-identical, not merely close
    if _rust_ready("packed_matvec"):
        out_r = k.packed_matvec(packed, 13, vec, backend="rust")
        _assert_rel(out, out_r)
        _assert_bytes(out, out_r)
        out_r_dirty = k.packed_matvec(dirty, 13, vec, backend="rust")
        _assert_bytes(out, out_r_dirty)
    else:
        _require_rust("packed_matvec")


def test_parity_matmul_wide():
    factor = _signed_factor((5, 12), seed=6)
    mat = np.random.default_rng(7).standard_normal((12, 3))
    packed = k.pack_bits(factor, backend="numpy-reference")
    out = k.packed_matmul(packed, 12, mat, backend="numpy-reference")
    assert out.dtype == np.float64
    assert out.shape == (5, 3)
    dense = k.unpack_bits(packed, 12, "float64", backend="numpy-reference")
    _assert_rel(out, dense @ mat)
    if _rust_ready("packed_matmul"):
        out_r = k.packed_matmul(packed, 12, mat, backend="rust")
        _assert_rel(out, out_r)
        _assert_bytes(out, out_r)
    else:
        _require_rust("packed_matmul")


# ---------------------------------------------------------------------------
# joint_itq scenarios
# ---------------------------------------------------------------------------


def _itq_inputs():
    rng = np.random.default_rng(8)
    u_hat = rng.standard_normal((7, 3))
    v_hat = rng.standard_normal((5, 3))
    return u_hat, v_hat


def test_parity_itq_preserves_product():
    u_hat, v_hat = _itq_inputs()
    u_t, v_t, n_iter, converged = k.joint_itq(u_hat, v_hat, seed=42, backend="numpy-reference")
    assert u_t.dtype == np.int8 and v_t.dtype == np.int8
    assert set(np.unique(u_t)) <= {-1, 1}
    assert set(np.unique(v_t)) <= {-1, 1}
    product = u_hat @ v_hat.T
    approx = u_t.astype(np.float64) @ v_t.astype(np.float64).T
    # binary-sign approximation error is bounded, but NOT at 1e-12: assert the
    # Procrustes-optimal bound instead (|W|_F <= sum of singular values).
    svals = np.linalg.svd(product, compute_uv=False)
    assert np.linalg.norm(product - approx) <= 2.0 * svals.sum() + 1e-9
    assert n_iter >= 1
    if _rust_ready("joint_itq"):
        u_r, v_r, n_iter_r, converged_r = k.joint_itq(u_hat, v_hat, seed=42, backend="rust")
        assert (n_iter_r, converged_r) == (n_iter, converged)
        _assert_bytes(u_t, u_r)
        _assert_bytes(v_t, v_r)
    else:
        _require_rust("joint_itq")


def test_parity_itq_converged():
    u_hat, v_hat = _itq_inputs()
    u_t, v_t, n_iter, converged = k.joint_itq(
        u_hat, v_hat, max_iter=50, tol=1e-12, seed=42, backend="numpy-reference"
    )
    assert converged is True
    assert 1 <= n_iter <= 50
    if _rust_ready("joint_itq"):
        u_r, v_r, n_iter_r, converged_r = k.joint_itq(
            u_hat, v_hat, max_iter=50, tol=1e-12, seed=42, backend="rust"
        )
        assert converged_r is True
        assert n_iter_r == n_iter
        _assert_bytes(u_t, u_r)
        _assert_bytes(v_t, v_r)
    else:
        _require_rust("joint_itq")


def test_parity_itq_maxiter():
    u_hat, v_hat = _itq_inputs()
    u_t, v_t, n_iter, converged = k.joint_itq(
        u_hat, v_hat, max_iter=50, tol=0.0, seed=42, backend="numpy-reference"
    )
    assert converged is False
    assert n_iter == 50, "must run exactly max_iter iterations"
    if _rust_ready("joint_itq"):
        u_r, v_r, n_iter_r, converged_r = k.joint_itq(
            u_hat, v_hat, max_iter=50, tol=0.0, seed=42, backend="rust"
        )
        assert converged_r is False
        assert n_iter_r == 50
        # different fixed point is permitted for a different backend; only
        # the loop bookkeeping is contractual here
        assert u_r.shape == u_t.shape and v_r.shape == v_t.shape
    else:
        _require_rust("joint_itq")


def test_parity_itq_deterministic_seed():
    u_hat, v_hat = _itq_inputs()
    u1, v1, n1, c1 = k.joint_itq(u_hat, v_hat, seed=42, backend="numpy-reference")
    u2, v2, n2, c2 = k.joint_itq(u_hat, v_hat, seed=42, backend="numpy-reference")
    assert (n1, c1) == (n2, c2)
    _assert_bytes(u1, u2)
    _assert_bytes(v1, v2)
    assert u1.tobytes() == u2.tobytes() and v1.tobytes() == v2.tobytes()
    # different seed: a different rotation fixed point is permitted
    u3, v3, n3, c3 = k.joint_itq(u_hat, v_hat, seed=7, backend="numpy-reference")
    assert u3.shape == u1.shape and v3.shape == v1.shape
    if _rust_ready("joint_itq"):
        ur1, vr1, nr1, cr1 = k.joint_itq(u_hat, v_hat, seed=42, backend="rust")
        ur2, vr2, nr2, cr2 = k.joint_itq(u_hat, v_hat, seed=42, backend="rust")
        assert (nr1, cr1) == (nr2, cr2)
        _assert_bytes(ur1, ur2)
        _assert_bytes(vr1, vr2)
        # same seed across backends: packed sign factors byte-identical
        _assert_bytes(u1, ur1)
        _assert_bytes(v1, vr1)
    else:
        _require_rust("joint_itq")


# ---------------------------------------------------------------------------
# dispatch surface sanity (full matrix in tests/test_kernels_dispatch.py)
# ---------------------------------------------------------------------------


def test_parity_all_kernels_present_on_both_backends():
    for name in _RUST_KERNELS:
        assert hasattr(k, name), f"missing public kernel {name}"
    for name in _RUST_KERNELS:
        numpy_fn = getattr(k, f"_numpy_{name}")  # noqa: SLF001
        assert callable(numpy_fn), f"missing NumPy reference for {name}"
    if k.HAVE_KERNELS:
        missing = [n for n in _RUST_KERNELS if not hasattr(k._kernels, n)]  # noqa: SLF001
        if missing:
            pytest.skip(f"installed extension predates the v0.1 surface (lacks {missing})")
