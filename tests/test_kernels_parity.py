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
# scenario probes: pack_rejects_non_binary, unpack_dtype_selection, itq_converged
# ---------------------------------------------------------------------------


def test_parity_pack_rejects_non_binary():
    """Scenario: pack_rejects_non_binary — BOTH faces must raise ValueError.

    Regression guard: the PyO3 face once silently truncated 1.0001 -> 1 via
    a blind `as i8` cast, diverging from the NumPy reference on exactly the
    input class this scenario probes.
    """
    for bad_value in (0.0, 2.0, 0.5, 1.0001, -1.5):
        arr = np.array([bad_value, 1.0], dtype=np.float64)
        with pytest.raises(ValueError):
            k.pack_bits(arr, backend="numpy-reference")
    if _rust_ready("pack_bits"):
        for bad_value in (0.0, 2.0, 0.5, 1.0001, -1.5):
            arr = np.array([bad_value, 1.0], dtype=np.float64)
            with pytest.raises(ValueError):
                k.pack_bits(arr, backend="rust")
    else:
        _require_rust("pack_bits")


def test_parity_unpack_dtype_selection():
    """Scenario: unpack_dtype_selection — one packed input, both dtypes."""
    signs = np.array([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]])
    packed_ref = k.pack_bits(signs, backend="numpy-reference")
    i8 = k.unpack_bits(packed_ref, 8, "int8", backend="numpy-reference")
    f64 = k.unpack_bits(packed_ref, 8, "float64", backend="numpy-reference")
    assert i8.dtype == np.int8
    assert f64.dtype == np.float64
    assert set(np.unique(i8)) == {-1, 1}, "int8 output must be exactly ±1"
    _assert_rel(f64, signs)
    assert (i8.astype(np.float64) == f64).all(), "both dtypes decode the same ±1 pattern"
    if _rust_ready("pack_bits", "unpack_bits"):
        packed_r = k.pack_bits(signs, backend="rust")
        _assert_bytes(packed_ref, packed_r)
        i8_r = k.unpack_bits(packed_r, 8, "int8", backend="rust")
        f64_r = k.unpack_bits(packed_r, 8, "float64", backend="rust")
        assert i8_r.dtype == np.int8 and f64_r.dtype == np.float64
        assert set(np.unique(i8_r)) == {-1, 1}
        assert (i8_r == i8).all()
        _assert_rel(f64_r, f64)
    else:
        _require_rust("pack_bits", "unpack_bits")


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


# ---------------------------------------------------------------------------
# Dtype-generic FP-remainder products (change qk-kernels-dtype-generic).
# Scenario IDs: dtype_tag_resolution, scalar_trait_covers_three_dtypes,
# dtype_f32_matvec, dtype_bf16_matvec, dtype_f32_matmul, dtype_bf16_matmul,
# f64_product_bit_identical, bf16_packed_still_bit_identical,
# pyo3_dtype_tag_selects_output_dtype, pyo3_unknown_dtype_refused,
# bf16_wire_form_is_uint16.
#
# bf16 deviation is bounded as a BACKWARD error on the row's term scale
# Sigma_k |v_k|: the dominant term is the bf16 rounding of the OPERAND, which
# scales with the terms, and a forward-relative bound is governed by the sum's
# condition number (a canceling row sends it to infinity). These are the
# constants rust-side `parity.rs` declares.
# ---------------------------------------------------------------------------

_PRODUCT_DTYPES = ("float64", "float32", "bfloat16")
F32_BACKWARD_TOL = 1e-6
BF16_BACKWARD_TOL = 1e-2


def _torch_bf16_bits(x: np.ndarray) -> np.ndarray:
    """bf16 bit patterns from TORCH — an encoder independent of this package."""
    import torch

    return torch.tensor(np.asarray(x, np.float32), dtype=torch.bfloat16).view(torch.uint16).numpy()


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u = np.asarray(bits, np.uint16).astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


def _dense(name: str, x: np.ndarray) -> np.ndarray:
    """A dense operand in ``name``'s wire form (bf16 crosses as uint16 bits)."""
    if name == "bfloat16":
        return _torch_bf16_bits(x)
    return np.asarray(x, dtype=name)


def _decode(name: str, x: np.ndarray) -> np.ndarray:
    """A result's wire form widened to float64 for comparison."""
    if name == "bfloat16":
        return _bf16_bits_to_f32(x).astype(np.float64)
    return np.asarray(x).astype(np.float64)


def _term_scale(factor: np.ndarray, operand: np.ndarray) -> np.ndarray:
    """Sigma_k |v_k| per row — the backward-error denominator."""
    return np.abs(factor.astype(np.float64)) @ np.abs(np.asarray(operand, np.float64))


def test_scn_dtype_tag_resolution():
    """Three dtype names resolve; anything else is refused naming the value."""
    packed = k.pack_bits(_signed_factor((2, 4), seed=11), backend="numpy-reference")
    op = np.ones(4, dtype=np.float32)
    for name in _PRODUCT_DTYPES:
        k.packed_matvec(packed, 4, _dense(name, op), dtype=name, backend="numpy-reference")
    with pytest.raises(ValueError) as excinfo:
        k.packed_matvec(packed, 4, op, dtype="float16", backend="numpy-reference")
    assert "float16" in str(excinfo.value)
    if _rust_ready("packed_matvec_dtype"):
        with pytest.raises(ValueError) as excinfo:
            k.packed_matvec(packed, 4, op, dtype="float16", backend="rust")
        assert "float16" in str(excinfo.value), "the Rust face must name the value too"
    else:
        _require_rust("packed_matvec_dtype")


def test_scn_scalar_trait_covers_three_dtypes():
    """The accepted set is the same on both faces — no second private list."""
    packed = k.pack_bits(_signed_factor((2, 4), seed=12), backend="numpy-reference")
    op = np.ones(4, dtype=np.float32)
    probes = [*_PRODUCT_DTYPES, "float16", "int8", "bfloat32", "float64x2"]

    def accepted(backend: str) -> list[str]:
        out = []
        for name in probes:
            operand = _dense(name if name in _PRODUCT_DTYPES else "float32", op)
            try:
                k.packed_matvec(packed, 4, operand, dtype=name, backend=backend)
            except ValueError:
                continue
            out.append(name)
        return out

    numpy_accepted = accepted("numpy-reference")
    assert numpy_accepted == list(_PRODUCT_DTYPES)
    if _rust_ready("packed_matvec_dtype"):
        assert accepted("rust") == numpy_accepted
    else:
        _require_rust("packed_matvec_dtype")


def _check_dtype_matvec(dtype: str, *, n_bits: int, rows: int, seed: int) -> None:
    factor = _signed_factor((rows, n_bits), seed=seed)
    packed = k.pack_bits(factor, backend="numpy-reference")
    vec = np.random.default_rng(seed + 1).standard_normal(n_bits)
    operand = _dense(dtype, vec)
    ref = k.packed_matvec(packed, n_bits, operand, dtype=dtype, backend="numpy-reference")
    assert ref.dtype == (np.uint16 if dtype == "bfloat16" else np.dtype(dtype))

    # the declared bound is exercised against the exact float64 product of the
    # SAME operand: the different-precision computation this dtype replaces
    exact_operand = (
        _bf16_bits_to_f32(operand) if dtype == "bfloat16" else operand.astype(np.float32)
    )
    exact = k.packed_matvec(
        packed, n_bits, exact_operand.astype(np.float64), backend="numpy-reference"
    )
    scale = _term_scale(factor, exact_operand)
    got = _decode(dtype, ref)
    tol = BF16_BACKWARD_TOL if dtype == "bfloat16" else F32_BACKWARD_TOL
    assert np.all(np.abs(got - exact) <= tol * scale), (
        f"{dtype} matvec exceeds the declared backward bound {tol}: "
        f"worst {np.max(np.abs(got - exact) / scale):.3e} of the term scale"
    )

    if _rust_ready("packed_matvec_dtype"):
        rust = k.packed_matvec(packed, n_bits, operand, dtype=dtype, backend="rust")
        assert rust.dtype == ref.dtype, "the tagged dtype must come back on both faces"
        assert rust.tobytes() == ref.tobytes(), "the two faces must agree bit-exactly"
    else:
        _require_rust("packed_matvec_dtype")


def _check_dtype_matmul(dtype: str, *, n_bits: int, rows: int, cols: int, seed: int) -> None:
    factor = _signed_factor((rows, n_bits), seed=seed)
    packed = k.pack_bits(factor, backend="numpy-reference")
    mat = np.random.default_rng(seed + 1).standard_normal((n_bits, cols))
    operand = _dense(dtype, mat)
    ref = k.packed_matmul(packed, n_bits, operand, dtype=dtype, backend="numpy-reference")
    assert ref.dtype == (np.uint16 if dtype == "bfloat16" else np.dtype(dtype))

    exact_operand = (
        _bf16_bits_to_f32(operand) if dtype == "bfloat16" else operand.astype(np.float32)
    )
    exact = k.packed_matmul(
        packed, n_bits, exact_operand.astype(np.float64), backend="numpy-reference"
    )
    scale = np.stack([_term_scale(factor, exact_operand[:, j]) for j in range(cols)], axis=1)
    got = _decode(dtype, ref)
    tol = BF16_BACKWARD_TOL if dtype == "bfloat16" else F32_BACKWARD_TOL
    assert np.all(np.abs(got - exact) <= tol * scale), (
        f"{dtype} matmul exceeds the declared backward bound {tol}: "
        f"worst {np.max(np.abs(got - exact) / scale):.3e} of the term scale"
    )

    if _rust_ready("packed_matmul_dtype"):
        rust = k.packed_matmul(packed, n_bits, operand, dtype=dtype, backend="rust")
        assert rust.dtype == ref.dtype
        assert rust.tobytes() == ref.tobytes(), "the two faces must agree bit-exactly"
    else:
        _require_rust("packed_matmul_dtype")


def test_scn_dtype_f32_matvec():
    _check_dtype_matvec("float32", n_bits=13, rows=6, seed=101)


def test_scn_dtype_bf16_matvec():
    _check_dtype_matvec("bfloat16", n_bits=13, rows=6, seed=102)


def test_scn_dtype_f32_matmul():
    _check_dtype_matmul("float32", n_bits=12, rows=5, cols=3, seed=103)


def test_scn_dtype_bf16_matmul():
    _check_dtype_matmul("bfloat16", n_bits=12, rows=5, cols=3, seed=104)


def test_scn_f64_product_bit_identical():
    """The refactor is structural: the float64 path is the v0.1 bytes."""
    factor = _signed_factor((6, 13), seed=21)
    packed = k.pack_bits(factor, backend="numpy-reference")
    vec = np.random.default_rng(22).standard_normal(13)
    ref = k.packed_matvec(packed, 13, vec, backend="numpy-reference")
    dense = factor.astype(np.float64)
    manual = np.array([sum(dense[i, kk] * vec[kk] for kk in range(13)) for i in range(6)])
    assert ref.tobytes() == manual.tobytes(), "the reference is left-to-right float64"
    if _rust_ready("packed_matvec", "packed_matvec_dtype"):
        assert k.packed_matvec(packed, 13, vec, backend="rust").tobytes() == ref.tobytes()
        tagged = k.packed_matvec(packed, 13, vec, dtype="float64", backend="rust")
        assert tagged.tobytes() == ref.tobytes(), "the tagged float64 leg must be the v0.1 bytes"
    else:
        _require_rust("packed_matvec", "packed_matvec_dtype")


def test_scn_f64_surface_signature_and_error_unchanged():
    """The v0.1 f64 call shape and diagnostics, preserved verbatim.

    Scenario ``f64_surface_signature_and_error_unchanged``. Both halves are
    probed as VALUES: ``pytest.raises(ValueError)`` cannot tell a preserved
    message from a rewritten one, and a new POSITIONAL parameter would break a
    v0.1 call site that the keyword-only additions do not.
    """
    import inspect

    for fn, third in ((k.packed_matvec, "vec"), (k.packed_matmul, "mat")):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params[:3]] == ["packed", "n_bits", third]
        assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:3])
        # every ADDED parameter is keyword-only, so no v0.1 call site breaks
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[3:])
        assert "dtype" in {p.name for p in params[3:]}

    packed = k.pack_bits(_signed_factor((6, 13), seed=71), backend="numpy-reference")
    packed_m = k.pack_bits(_signed_factor((5, 12), seed=72), backend="numpy-reference")

    # NumPy reference: the v0.1 f64 literals
    with pytest.raises(ValueError) as excinfo:
        k.packed_matvec(packed, 13, np.ones(5), backend="numpy-reference")
    assert "vec must have shape (13,), got (5,)" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        k.packed_matmul(packed_m, 12, np.ones((5, 3)), backend="numpy-reference")
    assert "mat must have 12 rows, got 5" in str(excinfo.value)

    if _rust_ready("packed_matvec", "packed_matmul"):
        # the Rust face's own literals, reached through PyO3 as ValueError
        with pytest.raises(ValueError) as excinfo:
            k.packed_matvec(packed, 13, np.ones(5), backend="rust")
        assert "vec must have length 13, got 5" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            k.packed_matmul(packed_m, 12, np.ones((5, 3)), backend="rust")
        assert "mat length 15 is not a multiple of n_bits 12" in str(excinfo.value)
    else:
        _require_rust("packed_matvec", "packed_matmul")


def test_scn_bf16_packed_still_bit_identical():
    """The dtype touches the dense operand and nothing else."""
    factor = _signed_factor((5, 12), seed=31)
    packed = k.pack_bits(factor, backend="numpy-reference")
    mat = np.random.default_rng(32).standard_normal((12, 3))
    results = [
        k.packed_matmul(packed, 12, _dense(dtype, mat), dtype=dtype, backend="numpy-reference")
        for dtype in ("float32", "bfloat16")
    ]
    assert all(r.shape == (5, 3) for r in results)
    decoded = k.unpack_bits(packed, 12, "int8", backend="numpy-reference")
    assert k.pack_bits(decoded, backend="numpy-reference").tobytes() == packed.tobytes()
    if _rust_ready("packed_matmul_dtype"):
        for dtype in ("float32", "bfloat16"):
            operand = _dense(dtype, mat)
            rust = k.packed_matmul(packed, 12, operand, dtype=dtype, backend="rust")
            numpy_ref = k.packed_matmul(packed, 12, operand, dtype=dtype, backend="numpy-reference")
            assert rust.tobytes() == numpy_ref.tobytes()
    else:
        _require_rust("packed_matmul_dtype")


def test_scn_pyo3_dtype_tag_selects_output_dtype():
    packed = k.pack_bits(_signed_factor((4, 9), seed=41), backend="numpy-reference")
    vec = np.random.default_rng(42).standard_normal(9)
    if not _rust_ready("packed_matvec_dtype"):
        _require_rust("packed_matvec_dtype")
    for dtype, expected in (
        ("float64", np.float64),
        ("float32", np.float32),
        ("bfloat16", np.uint16),
    ):
        got = k.packed_matvec(packed, 9, _dense(dtype, vec), dtype=dtype, backend="rust")
        assert got.dtype == np.dtype(expected), f"{dtype} came back as {got.dtype}"


def test_scn_pyo3_unknown_dtype_refused():
    packed = k.pack_bits(_signed_factor((2, 3), seed=51), backend="numpy-reference")
    if not _rust_ready("packed_matvec_dtype", "packed_matmul_dtype"):
        _require_rust("packed_matvec_dtype", "packed_matmul_dtype")
    with pytest.raises(ValueError) as excinfo:
        k.packed_matvec(packed, 3, np.ones(3, np.float32), dtype="float8", backend="rust")
    assert "float8" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        k.packed_matmul(packed, 3, np.ones((3, 2), np.float32), dtype="float8", backend="rust")
    assert "float8" in str(excinfo.value)


def test_scn_bf16_wire_form_is_uint16():
    """bf16 crosses as uint16 bit patterns; torch's cast is the independent check."""
    import torch

    vec = np.random.default_rng(61).standard_normal(7).astype(np.float32)
    bits = _torch_bf16_bits(vec)
    assert bits.dtype == np.uint16
    assert k._f32_to_bf16_bits(vec).tobytes() == bits.tobytes(), (  # noqa: SLF001
        "the package's bf16 rounding must be torch's rounding (round-to-nearest-even)"
    )
    assert _bf16_bits_to_f32(bits).dtype == np.float32
    tensor = torch.tensor(vec, dtype=torch.bfloat16)
    assert tensor.view(torch.uint16).numpy().tobytes() == bits.tobytes()
