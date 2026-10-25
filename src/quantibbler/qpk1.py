"""qpk1: canonical serialized container for one factorized linear layer.

Implements the ``qk-representation`` capability (openspec change
``qk-representation``; design decisions D1-D9):

- 24-byte u32 LE header (magic ``b"QPK1"``, version, dtype tag, dims), then
  the scale vectors (float16 default / float32 opt-in, one dtype per stream),
  then the ``(dout,)`` bias vector in that same dtype (version 2 onward),
  then packed uint8 ``Ub``/``Vb`` bitplanes (D1);
- canonical packing: row-major, little-endian within bytes, ``+1 -> bit 1``,
  ``-1 -> bit 0``, zero padding to the byte boundary (D2);
- byte-identical round-trip (D4); BPW counted only from stream length (D5);
- rank planning formula (D6); fail-closed reader (D8); version gate (D9).

Version semantics (D10/D11):

- ``REPRESENTATION_VERSION`` is 2: the writer emits the bias vector and the
  reader recovers it. This is the only version new artifacts are born on.
- ``LEGACY_REPRESENTATION_VERSION`` is 1: a pre-bias artifact carries no bias
  vector. Reading one is the declared legacy rule -- the layer's bias is the
  zero vector it always implicitly was -- NOT a silent default for a missing
  field on a current stream. A version-1 stream is read, never written.

NumPy is the executable specification; the Rust kernels lane must match
these packed bits exactly (sign-agreement invariant, config.yaml).
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = [
    "DTYPE_TAG_FLOAT16",
    "DTYPE_TAG_FLOAT32",
    "LEGACY_REPRESENTATION_VERSION",
    "QPK1_MAGIC",
    "REPRESENTATION_VERSION",
    "Qpk1Error",
    "Qpk1Layer",
    "bpw_from_bytes",
    "deserialize",
    "expected_size",
    "rank_for_target_bpw",
    "serialize",
]

QPK1_MAGIC = b"QPK1"
#: Version the writer emits: scales + a ``(dout,)`` bias vector + bitplanes.
REPRESENTATION_VERSION = 2
#: Version 1 = pre-bias artifacts. Read-only legacy (D11): bias absent means
#: the zero vector, which is exactly the map a biasless linear applies.
LEGACY_REPRESENTATION_VERSION = 1
DTYPE_TAG_FLOAT16 = 1
DTYPE_TAG_FLOAT32 = 2

_HEADER = struct.Struct("<4sIIIII")  # magic, version, dtype tag, dout, din, r
_U32_MAX = 0xFFFFFFFF

_TAGS_TO_DTYPES = {
    DTYPE_TAG_FLOAT16: np.dtype("float16"),
    DTYPE_TAG_FLOAT32: np.dtype("float32"),
}
_DTYPES_TO_TAGS = {dt: tag for tag, dt in _TAGS_TO_DTYPES.items()}
_READABLE_VERSIONS = (LEGACY_REPRESENTATION_VERSION, REPRESENTATION_VERSION)


class Qpk1Error(ValueError):
    """A byte stream is not a valid qpk1 layer (fail-closed reader, D8)."""


class Qpk1Layer:
    """One deserialized factorized linear layer.

    Attributes:
        Ub: ``(dout, r)`` int8 binary factor, entries exactly ``-1``/``+1``.
        Vb: ``(din, r)`` int8 binary factor, entries exactly ``-1``/``+1``.
        s1: ``(dout,)`` scale vector in the stream's dtype.
        s2: ``(din,)`` scale vector in the stream's dtype.
        dtype: the stream's scale dtype (``float16`` or ``float32``).
        version: the stream's representation version (2 carries a bias
            vector; 1 is the pre-bias legacy reading).
        bias: ``(dout,)`` bias vector in the stream's dtype, or ``None`` for
            a legacy version-1 stream (which carries no bias field).
    """

    __slots__ = ("Ub", "Vb", "s1", "s2", "dtype", "version", "bias")

    def __init__(self, Ub, Vb, s1, s2, dtype, version=REPRESENTATION_VERSION, bias=None):
        self.Ub = Ub
        self.Vb = Vb
        self.s1 = s1
        self.s2 = s2
        self.dtype = dtype
        self.version = version
        self.bias = bias

    def __repr__(self) -> str:
        dout, r = self.Ub.shape
        din = self.Vb.shape[0]
        return (
            f"Qpk1Layer(dout={dout}, din={din}, r={r}, v={self.version}, "
            f"dtype={np.dtype(self.dtype)!s}, "
            f"bpw={bpw_from_bytes(expected_size(dout, din, r, self.dtype), dout, din):.4f})"
        )


def _canonical_dtype(dtype) -> np.dtype:
    """Normalize a dtype argument to one of the two storable scale dtypes."""
    dt = np.dtype(dtype)
    if dt not in _DTYPES_TO_TAGS:
        raise Qpk1Error(f"unsupported scale dtype {dt!s}; qpk1 stores float16 or float32")
    return dt


def _require_dims(dout, din, r) -> None:
    """Structural dimension validation (D8): positive and u32-bounded."""
    if dout <= 0 or din <= 0 or r <= 0:
        raise Qpk1Error(f"dout, din and r must all be >= 1, got {(dout, din, r)}")
    if dout > _U32_MAX or din > _U32_MAX or r > _U32_MAX:
        raise Qpk1Error("dout, din and r must fit in u32 (header fields are u32 LE)")


def _require_rank_feasible(dout, din, r) -> None:
    """Reader-side rank sanity (D8/SF-1.4): ``r <= min(dout, din)``.

    Enforced by :func:`deserialize` only. ``serialize``/``expected_size``
    stay bound-free so byte-layout fixtures like the SF-1.3 known layer
    (``3 x 5 x r=7``) remain writable; see the Implementation
    crystallization entry in the change's design.md.
    """
    if r > min(dout, din):
        raise Qpk1Error(f"r={r} exceeds min(dout, din)={min(dout, din)}")


def _pack_bits(factors: np.ndarray) -> bytes:
    """Pack a +/-1 factor matrix per D2: row-major, LSB-first, +1 -> bit 1."""
    bits = (np.asarray(factors) > 0).astype(np.uint8)
    return np.packbits(bits, axis=None, bitorder="little").tobytes()


def _unpack_bits(data: bytes, count: int) -> np.ndarray:
    """Inverse of :func:`_pack_bits` for the first ``count`` bits."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little", count=count)
    return bits.astype(np.int8) * np.int8(2) - np.int8(1)


def serialize(Ub, Vb, s1, s2, bias=None, dtype=np.float16) -> bytes:
    """Serialize one factorized linear layer to the canonical qpk1 stream (D1).

    Args:
        Ub: ``(dout, r)`` binary factor, entries exactly ``-1``/``+1``.
        Vb: ``(din, r)`` binary factor, entries exactly ``-1``/``+1``.
        s1: ``(dout,)`` scale vector (any float dtype; rounded once to
            ``dtype`` at write time).
        s2: ``(din,)`` scale vector, same rules as ``s1``.
        bias: ``(dout,)`` bias vector, same rounding rule as the scales.
            ``None`` (the default) writes the zero vector — a zero bias is
            the identity element of the map, so this is not a hidden default:
            the field is always present on a version-2 stream.
        dtype: scale storage dtype — ``float16`` (default, dtype tag 1) or
            ``float32`` (opt-in, dtype tag 2). All three vectors share one
            dtype.

    Returns:
        The canonical byte stream; its length is exactly
        ``expected_size(dout, din, r, dtype)``.

    Raises:
        Qpk1Error: on shape, rank, dtype, or dimension violations.
    """
    dt = _canonical_dtype(dtype)
    ub = np.asarray(Ub)
    vb = np.asarray(Vb)
    sv1 = np.asarray(s1)
    sv2 = np.asarray(s2)

    if ub.ndim != 2 or vb.ndim != 2:
        raise Qpk1Error(f"factors must be 2-D, got Ub{ub.shape} Vb{vb.shape}")
    dout, rank_u = ub.shape
    din, rank_v = vb.shape
    if rank_u != rank_v:
        raise Qpk1Error(f"factor ranks disagree: Ub has r={rank_u}, Vb has r={rank_v}")
    for name, vec, n in (("s1", sv1, dout), ("s2", sv2, din)):
        if vec.shape != (n,):
            raise Qpk1Error(f"{name} must have shape ({n},), got {vec.shape}")
    if bias is None:
        bv = np.zeros(dout)
    else:
        bv = np.asarray(bias)
        if bv.shape != (dout,):
            raise Qpk1Error(f"bias must have shape ({dout},), got {bv.shape}")
    for name, mat in (("Ub", ub), ("Vb", vb)):
        if not np.all((mat == 1) | (mat == -1)):
            raise Qpk1Error(f"{name} entries must be exactly +1 or -1")
    _require_dims(dout, din, rank_u)
    header = _HEADER.pack(
        QPK1_MAGIC, REPRESENTATION_VERSION, _DTYPES_TO_TAGS[dt], dout, din, rank_u
    )
    dt_le = np.dtype(dt.name).newbyteorder("<")
    s1_bytes = sv1.astype(dt, copy=False).astype(dt_le, copy=False).tobytes()
    s2_bytes = sv2.astype(dt, copy=False).astype(dt_le, copy=False).tobytes()
    bias_bytes = bv.astype(dt, copy=False).astype(dt_le, copy=False).tobytes()
    return b"".join((header, s1_bytes, s2_bytes, bias_bytes, _pack_bits(ub), _pack_bits(vb)))


def deserialize(data) -> Qpk1Layer:
    """Parse exactly one qpk1 layer from a byte stream (fail-closed, D8).

    Rejects bad magic, versions other than 1 or 2 (D9; a version-1 stream is
    the declared pre-bias legacy reading and parses with ``bias is None``),
    unknown dtype tags, non-positive or rank-infeasible dims, dims beyond the
    u32 ceiling, truncated streams, and trailing garbage.
    """
    raw = bytes(data)
    if len(raw) < _HEADER.size:
        raise Qpk1Error(f"stream is {len(raw)} bytes; shorter than the {_HEADER.size}-byte header")
    magic, version, tag, dout, din, r = _HEADER.unpack_from(raw)
    if magic != QPK1_MAGIC:
        raise Qpk1Error(f"bad magic {magic!r}; expected {QPK1_MAGIC!r}")
    if version not in _READABLE_VERSIONS:
        raise Qpk1Error(
            f"unsupported representation version {version}; this reader implements "
            f"version {LEGACY_REPRESENTATION_VERSION} (pre-bias legacy) and "
            f"version {REPRESENTATION_VERSION} (current, with bias)"
        )
    if tag not in _TAGS_TO_DTYPES:
        raise Qpk1Error(f"unknown dtype tag {tag}; expected 1 (float16) or 2 (float32)")
    _require_dims(dout, din, r)
    _require_rank_feasible(dout, din, r)

    dt = _TAGS_TO_DTYPES[tag]
    has_bias = version >= REPRESENTATION_VERSION
    expected = expected_size(dout, din, r, dt, with_bias=has_bias)
    if len(raw) < expected:
        raise Qpk1Error(f"truncated stream: {len(raw)} bytes; header declares {expected}")
    if len(raw) > expected:
        raise Qpk1Error(
            f"trailing garbage: {len(raw)} bytes; one layer at this header is {expected}"
        )

    offset = _HEADER.size
    dt_le = np.dtype(dt.name).newbyteorder("<")

    def take(n: int) -> bytes:
        nonlocal offset
        chunk = raw[offset : offset + n]
        offset += n
        return chunk

    s1 = np.frombuffer(take(dout * dt.itemsize), dtype=dt_le).copy()
    s2 = np.frombuffer(take(din * dt.itemsize), dtype=dt_le).copy()
    bias = np.frombuffer(take(dout * dt.itemsize), dtype=dt_le).copy() if has_bias else None
    ub = _unpack_bits(take((dout * r + 7) // 8), dout * r).reshape(dout, r)
    vb = _unpack_bits(take((din * r + 7) // 8), din * r).reshape(din, r)
    return Qpk1Layer(ub, vb, s1, s2, dt, version=version, bias=bias)


def bpw_from_bytes(num_bytes: int, dout: int, din: int) -> float:
    """Effective bits per weight from the serialized byte count only (D5).

    ``BPW = 8 * num_bytes / (dout * din)`` — header, scales, bias and
    padding included; never derived from parameter counts.
    """
    if dout <= 0 or din <= 0:
        raise Qpk1Error(f"dout and din must be >= 1, got {(dout, din)}")
    return 8 * num_bytes / (dout * din)


def expected_size(dout: int, din: int, r: int, dtype=np.float16, *, with_bias=True) -> int:
    """Exact serialized byte count for a layer at ``(dout, din, r, dtype)``.

    Version 2 (``with_bias=True``, the writer's version):
    ``24 + (2*dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)`` — the
    two scale vectors plus the ``(dout,)`` bias vector. Version 1
    (``with_bias=False``, read-only legacy) drops the bias term:
    ``24 + (dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)`` (D1).
    """
    dt = _canonical_dtype(dtype)
    _require_dims(dout, din, r)
    floats = (2 * dout + din) if with_bias else (dout + din)
    return _HEADER.size + floats * dt.itemsize + (dout * r + 7) // 8 + (din * r + 7) // 8


def rank_for_target_bpw(dout: int, din: int, b: float, dtype=np.float16) -> int:
    """Planning rank hitting target BPW ``b`` (D6), re-checked as an int.

    Inverts the version-2 byte-size model:
    ``r(b) = (b * dout * din - 8 * (24 + (2*dout + din) * w)) / (dout + din)``,
    rounded to the nearest integer. A planning aid only — the authoritative
    BPW for any artifact remains ``bpw_from_bytes(len(serialized), ...)``.

    Raises:
        ValueError: if ``b`` is not positive or the target is infeasible
            (the fixed overhead alone exceeds the bit budget, or the needed
            rank exceeds ``min(dout, din)``).
    """
    dt = _canonical_dtype(dtype)
    _require_dims(dout, din, 1)
    if b <= 0:
        raise ValueError(f"target BPW must be positive, got {b}")
    numerator = b * dout * din - 8 * (_HEADER.size + (2 * dout + din) * dt.itemsize)
    if numerator <= 0:
        raise ValueError(
            f"target BPW {b} is infeasible at ({dout}, {din}) with {dt}: the fixed "
            f"header-plus-scale-plus-bias overhead alone needs more bits than the budget"
        )
    r = round(numerator / (dout + din))
    if r > min(dout, din):
        raise ValueError(f"target BPW {b} requires rank {r} > min(dout, din)={min(dout, din)}")
    return max(1, r)
