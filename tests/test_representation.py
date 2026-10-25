"""Tests for the qpk1 container: capability qk-representation (SF-1.1..SF-6.1).

Every test name references its scenario ID in
``openspec/changes/qk-representation/specs/qk-representation/spec.md``.
Byte literals below were derived by hand from design.md D1-D2 (24-byte
header; row-major, little-endian-within-bytes bit packing; ``+1 -> bit 1``
for the version the writer emits) and cross-checked with an independent
bit-loop oracle written from the spec prose before the implementation
existed.

Two container versions are exercised:

* version 2 (SF-1.5) is what ``serialize`` writes: header, scales, the
  ``(dout,)`` bias vector, then the bitplanes;
* version 1 (SF-1.6) is the pre-bias artifact. It is a READ-ONLY legacy
  reading: the fixed 48-byte literal below is its byte contract, it parses
  with ``bias is None``, and ``serialize`` never emits it.

The version-2 pinned literal is the same byte string the Rust faces assert
(``crates/quantibbler-qpk1/src/lib.rs`` and
``crates/quantibbler-server/tests/e2e_fixture.rs``), so one bias-carrying
vector is checked against both faces of the contract.
"""

import math
import struct

import numpy as np
import pytest

from quantibbler import qpk1

# --- Known-layer fixture (SF-1.3): dout=3, din=5, r=7, float16 scales. ---

UB_3X7 = np.array(
    [
        [1, 1, -1, 1, -1, 1, -1],
        [-1, 1, 1, -1, -1, 1, 1],
        [1, -1, 1, 1, 1, -1, -1],
    ],
    dtype=np.int8,
)
VB_5X7 = np.array(
    [
        [1, -1, 1, -1, 1, -1, 1],
        [-1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, -1, -1, -1],
        [-1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1],
    ],
    dtype=np.int8,
)
S1_F16 = np.array([1.0, 0.5, -2.0], dtype=np.float16)
S2_F16 = np.array([0.25, -0.5, 2.0, 1.0, 0.125], dtype=np.float16)
BIAS_3_F16 = np.array([1.0, -0.5, 2.0], dtype=np.float16)

# Magic 'QPK1'; u32 LE (version=1, tag=1, dout=3, din=5, r=7); s1/s2 float16
# LE; Ub bitplane 2b 73 07; Vb bitplane 55 e6 41 35 03. 48 bytes total: the
# pre-bias byte contract, hand-derived and frozen.
LEGACY_3_5_7_F16 = bytes.fromhex(
    "51504b31"
    "01000000"
    "01000000"
    "03000000"
    "05000000"
    "07000000"
    "003c003800c0"
    "003400b80040003c0030"
    "2b7307"
    "55e6413503"
)

# The same layer at version 2: u32 LE (version=2, tag=1, dout=3, din=5, r=7),
# the scales, then the bias vector 003c 00b8 0040 (= 1.0, -0.5, 2.0), then the
# unchanged bitplanes. 24 + (2*3 + 5)*2 + 3 + 3 = 54 bytes.
EXPECTED_3_5_7_F16_V2 = bytes.fromhex(
    "51504b31"
    "02000000"
    "01000000"
    "03000000"
    "05000000"
    "07000000"
    "003c003800c0"
    "003400b80040003c0030"
    "003c00b80040"
    "2b7307"
    "55e6413503"
)

# The same rank-feasible twin as a PRE-BIAS stream (44 bytes: the version-2
# bytes minus the bias vector, header back at version 1). This is the shape a
# committed pre-bias artifact has, so it is the legacy reading under test.
LEGACY_3_5_3_F16 = bytes.fromhex(
    "51504b310100000001000000030000000500000003000000003c003800c0003400b80040003c00307301e535"
)

# The rank-feasible twin of the same vector (r=3 <= min(3, 5) = 3), so it is a
# READER-accepted stream: 50 bytes. Pinned identically on the Rust faces.
EXPECTED_3_5_3_F16_V2 = bytes.fromhex(
    "51504b31"
    "02000000"
    "01000000"
    "03000000"
    "05000000"
    "03000000"
    "003c003800c0"
    "003400b80040003c0030"
    "003c00b80040"
    "7301"
    "e535"
)


def _base_stream() -> bytes:
    return qpk1.serialize(UB_3X7, VB_5X7, S1_F16, S2_F16, bias=BIAS_3_F16, dtype=np.float16)


def _random_factors(rng, dout, din, r):
    ub = rng.integers(0, 2, size=(dout, r), dtype=np.int8) * 2 - 1
    vb = rng.integers(0, 2, size=(din, r), dtype=np.int8) * 2 - 1
    return ub, vb


def test_sf1_3_known_layer_exact_bytes():
    stream = qpk1.serialize(UB_3X7, VB_5X7, S1_F16, S2_F16, bias=BIAS_3_F16, dtype=np.float16)
    assert len(stream) == 54
    assert stream == EXPECTED_3_5_7_F16_V2
    # Field-for-field at the documented offsets.
    assert stream[:4] == b"QPK1"
    assert struct.unpack("<5I", stream[4:24]) == (2, 1, 3, 5, 7)
    assert stream[24:30] == S1_F16.astype("<f2").tobytes()
    assert stream[30:40] == S2_F16.astype("<f2").tobytes()
    assert stream[40:46] == BIAS_3_F16.astype("<f2").tobytes()
    assert stream[46:49] == bytes.fromhex("2b7307")
    assert stream[49:54] == bytes.fromhex("55e6413503")

    # SF-1.1: the length formula holds across shapes, dtypes, odd and even r.
    rng = np.random.default_rng(11)
    shapes = [(1, 1, 1), (3, 5, 7), (64, 64, 9), (100, 200, 33), (4096, 11008, 100)]
    for dout, din, r in shapes:
        for dtype in (np.float16, np.float32):
            ub, vb = _random_factors(rng, dout, din, r)
            s1 = rng.standard_normal(dout).astype(dtype)
            s2 = rng.standard_normal(din).astype(dtype)
            bias = rng.standard_normal(dout).astype(dtype)
            stream = qpk1.serialize(ub, vb, s1, s2, bias=bias, dtype=dtype)
            w = np.dtype(dtype).itemsize
            expected = 24 + (2 * dout + din) * w + math.ceil(dout * r / 8) + math.ceil(din * r / 8)
            assert len(stream) == expected, f"size formula failed at {(dout, din, r, dtype)}"
            assert qpk1.expected_size(dout, din, r, dtype) == expected
            # The pre-bias model is exactly the same stream minus the bias
            # vector's `dout` floats.
            assert qpk1.expected_size(dout, din, r, dtype, with_bias=False) == expected - dout * w


def test_sf1_5_version2_layout_carries_the_bias_vector():
    """The writer's layout: the `(dout,)` bias vector sits between the scales
    and the bitplanes, in the stream's own dtype, at every shape."""
    stream = qpk1.serialize(UB_3X7, VB_5X7, S1_F16, S2_F16, bias=BIAS_3_F16, dtype=np.float16)
    assert struct.unpack("<I", stream[4:8])[0] == 2
    assert stream[40:46] == BIAS_3_F16.astype("<f2").tobytes()

    # Omitting the argument writes the zero vector rather than dropping the
    # field: the layout is fixed for every version-2 stream.
    zero = qpk1.serialize(UB_3X7, VB_5X7, S1_F16, S2_F16, dtype=np.float16)
    assert len(zero) == len(stream)
    assert zero[40:46] == np.zeros(3, np.float16).astype("<f2").tobytes()
    assert zero == qpk1.serialize(
        UB_3X7, VB_5X7, S1_F16, S2_F16, bias=np.zeros(3), dtype=np.float16
    )

    # A bias whose width is not dout is refused, not truncated.
    with pytest.raises(qpk1.Qpk1Error):
        qpk1.serialize(UB_3X7, VB_5X7, S1_F16, S2_F16, bias=np.zeros(4), dtype=np.float16)


def test_sf1_6_legacy_version1_reads_without_bias():
    """The pre-bias artifact: the frozen 48-byte literal carries version 1
    (the same byte contract the Rust faces pin), and its rank-feasible twin
    reads with no bias field and the factors/scales it stored."""
    # The writer-only SF-1.3 literal is rank-infeasible (7 > min(3, 5)), so it
    # is a byte contract, not a readable stream: assert its header here.
    assert LEGACY_3_5_7_F16[:4] == b"QPK1"
    assert struct.unpack("<5I", LEGACY_3_5_7_F16[4:24]) == (1, 1, 3, 5, 7)

    layer = qpk1.deserialize(LEGACY_3_5_3_F16)
    assert layer.version == qpk1.LEGACY_REPRESENTATION_VERSION
    assert layer.bias is None
    assert layer.Ub.shape == (3, 3)  # (dout, r)
    assert layer.Vb.shape == (5, 3)  # (din, r)
    assert layer.s1.tolist() == [1.0, 0.5, -2.0]
    assert layer.s2.tolist() == [0.25, -0.5, 2.0, 1.0, 0.125]
    assert layer.Ub.tolist() == UB_3X7[:, :3].tolist()
    assert layer.Vb.tolist() == VB_5X7[:, :3].tolist()

    # The legacy body length under the version-2 header is a TRUNCATION, never
    # a silent zero bias.
    with pytest.raises(qpk1.Qpk1Error):
        qpk1.deserialize(EXPECTED_3_5_7_F16_V2[:48])
    # And a version-2 stream one bias-vector short of its extent likewise.
    with pytest.raises(qpk1.Qpk1Error):
        qpk1.deserialize(EXPECTED_3_5_3_F16_V2[:44])


def test_sf1_2_dtype_tag_selects_precision():
    s1 = np.array([0.1, -3.5, 7.0], dtype=np.float32)
    s2 = np.array([1.5, -0.25, 2.0, -1.0, 0.5], dtype=np.float32)
    bias = np.array([0.25, -0.75, 1.5], dtype=np.float32)
    ub, vb = UB_3X7[:, :2], VB_5X7[:, :2]  # r = 2 views of the known factors

    stream32 = qpk1.serialize(ub, vb, s1, s2, bias=bias, dtype=np.float32)
    assert struct.unpack("<I", stream32[8:12])[0] == 2  # dtype tag
    layer32 = qpk1.deserialize(stream32)
    assert layer32.s1.dtype == np.float32
    assert layer32.s2.dtype == np.float32
    assert layer32.bias.dtype == np.float32
    np.testing.assert_array_equal(layer32.s1, s1)  # exact under the stream dtype
    np.testing.assert_array_equal(layer32.s2, s2)
    np.testing.assert_array_equal(layer32.bias, bias)

    stream16 = qpk1.serialize(ub, vb, s1, s2, bias=bias, dtype=np.float16)
    assert struct.unpack("<I", stream16[8:12])[0] == 1  # dtype tag
    layer16 = qpk1.deserialize(stream16)
    assert layer16.s1.dtype == np.float16
    assert layer16.bias.dtype == np.float16
    np.testing.assert_array_equal(layer16.s1, s1.astype(np.float16))  # rounded once, at write
    np.testing.assert_array_equal(layer16.bias, bias.astype(np.float16))

    for bad_tag in (0, 3, 0xFFFFFFFF):
        patched = stream16[:8] + struct.pack("<I", bad_tag) + stream16[12:]
        with pytest.raises(qpk1.Qpk1Error):
            qpk1.deserialize(patched)


def test_sf1_4_malformed_headers_rejected():
    base = _base_stream()  # 54 valid bytes

    def patch(offset, value):
        return base[:offset] + struct.pack("<I", value) + base[offset + 4 :]

    cases = {
        "bad magic": b"NOPE" + base[4:],
        "version 0": patch(4, 0),
        "version 3": patch(4, 3),
        "dtype tag 0": patch(8, 0),
        "dtype tag 3": patch(8, 3),
        "dtype tag 0xFFFFFFFF": patch(8, 0xFFFFFFFF),
        "r = 0": patch(20, 0),
        "r > min(dout, din)": patch(20, 6),
        "dims at u32 ceiling": struct.pack("<4sIIIII", b"QPK1", 2, 1, 0xFFFFFFFF, 0xFFFFFFFF, 1),
    }
    for offset in range(24):
        cases[f"truncated at header offset {offset}"] = base[:offset]
    cases["truncated mid-scales"] = base[:30]
    cases["truncated mid-bias"] = base[:44]
    cases["truncated one byte short"] = base[:-1]
    cases["one trailing byte"] = base + b"\x00"

    for label, stream in cases.items():
        try:
            qpk1.deserialize(stream)
        except qpk1.Qpk1Error:
            continue
        pytest.fail(f"malformed stream accepted: {label}")


def test_sf2_1_byte_identical_roundtrip():
    # SF-2.1's domain is valid streams, i.e. reader-accepted ones; the
    # SF-1.3 fixture stays writer-only (see the crystallization note on the
    # rank bound), so the byte-level anchor here uses a rank-feasible twin.
    feasible = qpk1.serialize(
        UB_3X7[:, :3], VB_5X7[:, :3], S1_F16, S2_F16, bias=BIAS_3_F16, dtype=np.float16
    )  # (3, 5, r=3): same dims, rank-feasible (r <= min(dout, din) = 3)
    assert feasible == EXPECTED_3_5_3_F16_V2
    layer = qpk1.deserialize(feasible)
    again = qpk1.serialize(
        layer.Ub, layer.Vb, layer.s1, layer.s2, bias=layer.bias, dtype=layer.dtype
    )
    assert again == feasible

    rng = np.random.default_rng(1)
    for dout, din, r in [(3, 5, 3), (5, 4, 3), (8, 8, 8), (13, 9, 5), (64, 33, 17)]:
        for dtype in (np.float16, np.float32):
            ub, vb = _random_factors(rng, dout, din, r)
            s1 = rng.standard_normal(dout).astype(dtype)
            s2 = rng.standard_normal(din).astype(dtype)
            bias = rng.standard_normal(dout).astype(dtype)
            stream = qpk1.serialize(ub, vb, s1, s2, bias=bias, dtype=dtype)
            layer = qpk1.deserialize(stream)
            reserialized = qpk1.serialize(
                layer.Ub, layer.Vb, layer.s1, layer.s2, bias=layer.bias, dtype=layer.dtype
            )
            assert reserialized == stream, f"byte round-trip failed at {(dout, din, r, dtype)}"


def test_sf2_1_and_sf2_2_randomized_shape_sweep():
    """Task 2.3: seeded randomized-shape sweep extending SF-2.1/SF-2.2.

    Shapes 1x1 through 4096x11008, r from 1 up, odd and even, both scale
    dtypes: byte identity, exact factor/scale/bias recovery, and
    ``expected_size(...) == len(serialize(...))`` per case.
    """
    rng = np.random.default_rng(2026)
    shapes = [
        (1, 1, 1),
        (1, 7, 1),
        (7, 1, 1),
        (2, 3, 2),
        (5, 5, 4),  # even r
        (5, 5, 5),  # odd r at the rank bound
        (8, 8, 8),
        (9, 13, 7),
        (16, 16, 15),
        (17, 31, 16),  # even r, odd dims
        (33, 64, 33),
        (100, 200, 33),
        (256, 512, 128),
        (1000, 1000, 999),
        (4096, 11008, 1),
        (4096, 11008, 100),
    ]
    for dout, din, r in shapes:
        assert r <= min(dout, din)
        for dtype in (np.float16, np.float32):
            ub, vb = _random_factors(rng, dout, din, r)
            s1 = rng.standard_normal(dout)
            s2 = rng.standard_normal(din)
            bias = rng.standard_normal(dout)
            stream = qpk1.serialize(ub, vb, s1, s2, bias=bias, dtype=dtype)
            assert len(stream) == qpk1.expected_size(dout, din, r, dtype)
            layer = qpk1.deserialize(stream)
            np.testing.assert_array_equal(layer.Ub, ub)
            np.testing.assert_array_equal(layer.Vb, vb)
            np.testing.assert_array_equal(layer.s1, s1.astype(dtype))
            np.testing.assert_array_equal(layer.s2, s2.astype(dtype))
            np.testing.assert_array_equal(layer.bias, bias.astype(dtype))
            assert (
                qpk1.serialize(layer.Ub, layer.Vb, layer.s1, layer.s2, bias=layer.bias, dtype=dtype)
                == stream
            )


def test_sf2_2_factors_scales_recovered():
    rng = np.random.default_rng(2)
    dout, din, r = 7, 13, 5
    ub, vb = _random_factors(rng, dout, din, r)
    s1_64 = rng.standard_normal(dout)
    s2_64 = rng.standard_normal(din)
    bias_64 = rng.standard_normal(dout)

    layer16 = qpk1.deserialize(qpk1.serialize(ub, vb, s1_64, s2_64, bias=bias_64, dtype=np.float16))
    assert layer16.Ub.dtype == np.int8
    np.testing.assert_array_equal(layer16.Ub, ub)  # bit-exact sign recovery
    np.testing.assert_array_equal(layer16.Vb, vb)
    # float64 inputs are rounded exactly once, at serialize time (IA4).
    np.testing.assert_array_equal(layer16.s1, s1_64.astype(np.float16))
    np.testing.assert_array_equal(layer16.s2, s2_64.astype(np.float16))
    np.testing.assert_array_equal(layer16.bias, bias_64.astype(np.float16))

    layer32 = qpk1.deserialize(qpk1.serialize(ub, vb, s1_64, s2_64, bias=bias_64, dtype=np.float32))
    np.testing.assert_array_equal(layer32.s1, s1_64.astype(np.float32))  # no further loss
    np.testing.assert_array_equal(layer32.s2, s2_64.astype(np.float32))
    np.testing.assert_array_equal(layer32.bias, bias_64.astype(np.float32))

    # Bit-level identity for float16-representable inputs, incl. subnormals/extremes.
    s1_h = np.array([0.1, 1.0, -65504.0, 6.1e-5], dtype=np.float16)
    s2_h = np.array([2.0, -0.0, 7.3e-6], dtype=np.float16)
    bias_h = np.array([0.5, -0.0, 3.0, -1.0], dtype=np.float16)
    ub4, vb4 = _random_factors(rng, 4, 3, 2)
    layer = qpk1.deserialize(qpk1.serialize(ub4, vb4, s1_h, s2_h, bias=bias_h, dtype=np.float16))
    np.testing.assert_array_equal(layer.s1.view(np.uint16), s1_h.view(np.uint16))
    np.testing.assert_array_equal(layer.s2.view(np.uint16), s2_h.view(np.uint16))
    np.testing.assert_array_equal(layer.bias.view(np.uint16), bias_h.view(np.uint16))


def test_sf2_3_bias_roundtrips_byte_exactly():
    """The bias vector is a first-class stream field: it is recovered exactly
    under the stream's dtype and re-serializes to the same bytes."""
    layer = qpk1.deserialize(EXPECTED_3_5_3_F16_V2)
    assert layer.version == qpk1.REPRESENTATION_VERSION
    assert layer.bias.tolist() == [1.0, -0.5, 2.0]
    assert layer.bias.dtype == np.float16
    assert (
        qpk1.serialize(layer.Ub, layer.Vb, layer.s1, layer.s2, bias=layer.bias, dtype=layer.dtype)
        == EXPECTED_3_5_3_F16_V2
    )

    # Changing only the bias changes only the bias bytes: the factors and the
    # scales are untouched by the field's presence.
    other = qpk1.serialize(
        layer.Ub,
        layer.Vb,
        layer.s1,
        layer.s2,
        bias=np.array([-1.0, 0.5, -2.0], dtype=np.float16),
        dtype=layer.dtype,
    )
    assert other[:40] == EXPECTED_3_5_3_F16_V2[:40]
    assert other[46:] == EXPECTED_3_5_3_F16_V2[46:]
    assert other[40:46] != EXPECTED_3_5_3_F16_V2[40:46]


def test_sf3_1_row_major_layout():
    dout, din, r = 3, 5, 5  # dout*r = 15 bits: non-byte-aligned
    ub = np.array(
        [[1, -1, 1, -1, 1], [-1, 1, -1, 1, -1], [1, 1, -1, -1, 1]],
        dtype=np.int8,
    )
    vb = np.ones((din, r), dtype=np.int8)
    s1 = np.ones(dout, dtype=np.float16)
    s2 = np.ones(din, dtype=np.float16)
    stream = qpk1.serialize(ub, vb, s1, s2, dtype=np.float16)

    ub_offset = 24 + (2 * dout + din) * 2
    plane_len = (dout * r + 7) // 8
    plane = stream[ub_offset : ub_offset + plane_len]
    assert plane == bytes.fromhex("554d")  # hand-computed reference bitplane

    # Element (i, j) lives at stream bit i*r + j: byte k//8, bit k%8 from the LSB.
    for i in range(dout):
        for j in range(r):
            k = i * r + j
            bit = (plane[k // 8] >> (k % 8)) & 1
            assert bit == (1 if ub[i, j] == 1 else 0), f"element ({i}, {j}) misplaced at bit {k}"


def test_sf3_2_sign_mapping():
    dout, din, r = 2, 3, 2  # r a multiple of 8 is unreachable here; any r works
    # The scenario wants rows at a byte-aligned r; take r = 8 with dout = din
    # large enough to keep the reader happy (r <= min(dout, din)).
    dout, din, r = 8, 8, 8
    ub = np.array([[1] * 8, [-1] * 8] + [[1, -1] * 4] * 6, dtype=np.int8)
    vb = np.array(
        [[1] * 8, [-1] * 8, [1] * 8, [-1] * 8, [1, -1] * 4, [1, -1] * 4, [1] * 8, [-1] * 8],
        dtype=np.int8,
    )
    stream = qpk1.serialize(
        ub, vb, np.ones(dout, np.float16), np.ones(din, np.float16), dtype=np.float16
    )
    ub_offset = 24 + (2 * dout + din) * 2
    ub_len = (dout * r + 7) // 8
    assert stream[ub_offset : ub_offset + 2] == b"\xff\x00"  # +1 row -> 0xFF, -1 row -> 0x00
    vb_offset = ub_offset + ub_len
    assert stream[vb_offset : vb_offset + 4] == b"\xff\x00\xff\x00"

    layer = qpk1.deserialize(stream)
    np.testing.assert_array_equal(layer.Ub, ub)
    np.testing.assert_array_equal(layer.Vb, vb)


def test_sf4_1_bpw_from_stream_length():
    stream = _base_stream()  # 54 bytes for dout=3, din=5
    measured = qpk1.bpw_from_bytes(len(stream), 3, 5)
    assert measured == 8 * len(stream) / (3 * 5)
    assert measured == pytest.approx(432 / 15)  # 28.8: header, bias and padding included

    rng = np.random.default_rng(5)
    ub, vb = _random_factors(rng, 3, 5, 2)
    s1 = rng.standard_normal(3).astype(np.float32)
    s2 = rng.standard_normal(5).astype(np.float32)
    stream32 = qpk1.serialize(ub, vb, s1, s2, dtype=np.float32)  # 71 bytes
    assert len(stream32) == 71
    assert qpk1.bpw_from_bytes(len(stream32), 3, 5) == 8 * 71 / 15


def test_sf4_2_bpw_matches_formula():
    rng = np.random.default_rng(3)
    for dout, din, r in [(3, 5, 7), (64, 64, 9), (100, 200, 33), (4096, 11008, 100)]:
        for dtype in (np.float16, np.float32):
            ub, vb = _random_factors(rng, dout, din, r)
            s1 = rng.standard_normal(dout).astype(dtype)
            s2 = rng.standard_normal(din).astype(dtype)
            stream = qpk1.serialize(ub, vb, s1, s2, dtype=dtype)
            w = np.dtype(dtype).itemsize
            analytical = (
                24 + (2 * dout + din) * w + math.ceil(dout * r / 8) + math.ceil(din * r / 8)
            )
            assert len(stream) == analytical
            assert qpk1.expected_size(dout, din, r, dtype) == analytical
            # Same integer byte count -> BPWs agree exactly (SF-4.2).
            assert qpk1.bpw_from_bytes(len(stream), dout, din) == qpk1.bpw_from_bytes(
                analytical, dout, din
            )


def test_sf4_3_sub1bpw_at_target_rank():
    dout, din = 4096, 11008
    rng = np.random.default_rng(4)
    ranks = [100, qpk1.rank_for_target_bpw(dout, din, 0.5, dtype=np.float16)]
    for r in ranks:
        ub, vb = _random_factors(rng, dout, din, r)
        s1 = rng.standard_normal(dout).astype(np.float16)
        s2 = rng.standard_normal(din).astype(np.float16)
        stream = qpk1.serialize(ub, vb, s1, s2, dtype=np.float16)
        measured = qpk1.bpw_from_bytes(len(stream), dout, din)
        assert measured < 1.0, f"BPW {measured} at r={r} is not sub-1-bit"

    # The documented shape at r=100 with zero scales and zero bias:
    # 227224 bytes -> BPW ~ 0.0403. The bias vector adds 2*4096 bytes to the
    # pre-bias 219032, and storage stays far under one bit per weight.
    ub, vb = _random_factors(rng, dout, din, 100)
    stream = qpk1.serialize(
        ub, vb, np.zeros(dout, np.float16), np.zeros(din, np.float16), dtype=np.float16
    )
    assert len(stream) == 227224
    assert qpk1.bpw_from_bytes(len(stream), dout, din) == pytest.approx(0.0403, abs=1e-3)


def test_sf5_1_rank_hits_target():
    for dout, din, b in [
        (4096, 4096, 0.55),
        (4096, 11008, 0.5),
        (1024, 2048, 0.75),
        (512, 512, 0.9),
    ]:
        r = qpk1.rank_for_target_bpw(dout, din, b, dtype=np.float16)
        assert r >= 1
        rng = np.random.default_rng(dout + din)
        ub, vb = _random_factors(rng, dout, din, r)
        s1 = rng.standard_normal(dout).astype(np.float16)
        s2 = rng.standard_normal(din).astype(np.float16)
        stream = qpk1.serialize(ub, vb, s1, s2, dtype=np.float16)
        measured = qpk1.bpw_from_bytes(len(stream), dout, din)

        def rank_bpw(rank, *, dout=dout, din=din):
            return qpk1.bpw_from_bytes(qpk1.expected_size(dout, din, rank, np.float16), dout, din)

        step = max(rank_bpw(r + 1) - rank_bpw(r), rank_bpw(r) - rank_bpw(max(r - 1, 1)))
        assert abs(measured - b) <= step, f"r={r} lands {measured} vs target {b}"
        # Re-checked derivation: inverting the measured bytes recovers the rank.
        r_back = (measured * dout * din - 8 * (24 + (2 * dout + din) * 2)) / (dout + din)
        assert abs(r_back - r) <= 1.0

    # Infeasible targets (non-positive numerator) are rejected, not clamped.
    with pytest.raises(ValueError):
        qpk1.rank_for_target_bpw(4096, 4096, 1e-9, dtype=np.float16)


def test_sf5_2_formula_reduces_to_papers():
    # With the 24-byte header and byte-rounding set aside, the byte model at
    # w=2 is exactly r*(dout+din) + 16*(2*dout+din) bits: the paper's weight
    # storage ([NQ] App. F.5 Eq. 58; [LB] App. D Eq. 16 with the residual
    # factor of 2 and the latent row removed) plus the container's bias
    # vector at 16 bits per output row.
    for dout, din, r in [(4096, 4096, 546), (4096, 11008, 100), (100, 100, 7)]:
        size = qpk1.expected_size(dout, din, r, np.float16)
        rounding = (8 * math.ceil(dout * r / 8) - dout * r) + (8 * math.ceil(din * r / 8) - din * r)
        qpk1_bits = 8 * (size - 24) - rounding
        paper_bits = r * (dout + din) + 16 * (dout + din)
        assert qpk1_bits == paper_bits + 16 * dout

    # Dropping the header term from the D6 inversion leaves Eq. 59's r(b),
    # shifted by exactly the bias vector's bits.
    b = 0.55
    r_eq59 = (b * 4096 * 4096 - 16 * (4096 + 4096)) / (4096 + 4096)
    r_headerless_nobias = (b * 4096 * 4096 - 8 * (4096 + 4096) * 2) / (4096 + 4096)
    assert r_headerless_nobias == pytest.approx(r_eq59)
    r_headerless = (b * 4096 * 4096 - 8 * (2 * 4096 + 4096) * 2) / (4096 + 4096)
    assert r_eq59 - r_headerless == pytest.approx(16 * 4096 / (4096 + 4096))


def test_sf6_1_stream_extent_one_layer():
    base = _base_stream()
    with pytest.raises(qpk1.Qpk1Error):
        qpk1.deserialize(base + b"\x00")  # any bytes beyond the declared extent
    with pytest.raises(qpk1.Qpk1Error):
        qpk1.deserialize(base + base)  # two layers concatenated are not one stream

    # The public surface is exactly the pinned names — no multi-layer API, and
    # the legacy version constant is part of the versioning contract.
    assert set(qpk1.__all__) == {
        "QPK1_MAGIC",
        "REPRESENTATION_VERSION",
        "LEGACY_REPRESENTATION_VERSION",
        "DTYPE_TAG_FLOAT16",
        "DTYPE_TAG_FLOAT32",
        "Qpk1Error",
        "Qpk1Layer",
        "bpw_from_bytes",
        "deserialize",
        "expected_size",
        "rank_for_target_bpw",
        "serialize",
    }
