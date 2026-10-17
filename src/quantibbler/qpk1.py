"""qpk1: canonical serialized container for one factorized linear layer.

Implements the ``qk-representation`` capability (openspec change
``qk-representation``; design decisions D1-D9):

- 24-byte u32 LE header (magic ``b"QPK1"``, version 1, dtype tag, dims),
  then scales (float16 default / float32 opt-in, one dtype per stream),
  then packed uint8 ``Ub``/``Vb`` bitplanes (D1);
- canonical packing: row-major, little-endian within bytes, ``+1 -> bit 1``,
  ``-1 -> bit 0``, zero padding to the byte boundary (D2);
- byte-identical round-trip (D4); BPW counted only from stream length (D5);
- rank planning formula (D6); fail-closed reader (D8); version gate (D9).

NumPy is the executable specification; the Rust kernels lane must match
these packed bits exactly (sign-agreement invariant, config.yaml).
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = [
    "DTYPE_TAG_FLOAT16",
    "DTYPE_TAG_FLOAT32",
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
REPRESENTATION_VERSION = 1
DTYPE_TAG_FLOAT16 = 1
DTYPE_TAG_FLOAT32 = 2

_HEADER = struct.Struct("<4sIIIII")  # magic, version, dtype tag, dout, din, r
_U32_MAX = 0xFFFFFFFF

_TAGS_TO_DTYPES = {
    DTYPE_TAG_FLOAT16: np.dtype("float16"),
    DTYPE_TAG_FLOAT32: np.dtype("float32"),
}
_DTYPES_TO_TAGS = {dt: tag for tag, dt in _TAGS_TO_DTYPES.items()}


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
    """

    __slots__ = ("Ub", "Vb", "s1", "s2", "dtype")

    def __init__(self, Ub, Vb, s1, s2, dtype):
        self.Ub = Ub
        self.Vb = Vb
        self.s1 = s1
        self.s2 = s2
        self.dtype = dtype

    def __repr__(self) -> str:
        dout, r = self.Ub.shape
        din = self.Vb.shape[0]
        return (
            f"Qpk1Layer(dout={dout}, din={din}, r={r}, dtype={np.dtype(self.dtype)!s}, "
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


def serialize(Ub, Vb, s1, s2, dtype=np.float16) -> bytes:
    """Serialize one factorized linear layer to the canonical qpk1 stream (D1).

    Args:
        Ub: ``(dout, r)`` binary factor, entries exactly ``-1``/``+1``.
        Vb: ``(din, r)`` binary factor, entries exactly ``-1``/``+1``.
        s1: ``(dout,)`` scale vector (any float dtype; rounded once to
            ``dtype`` at write time).
        s2: ``(din,)`` scale vector, same rules as ``s1``.
        dtype: scale storage dtype — ``float16`` (default, dtype tag 1) or
            ``float32`` (opt-in, dtype tag 2). Both vectors share one dtype.

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
    return b"".join((header, s1_bytes, s2_bytes, _pack_bits(ub), _pack_bits(vb)))


def deserialize(data) -> Qpk1Layer:
    """Parse exactly one qpk1 layer from a byte stream (fail-closed, D8).

    Rejects bad magic, versions other than ``REPRESENTATION_VERSION``
    (D9), unknown dtype tags, non-positive or rank-infeasible dims, dims
    beyond the u32 ceiling, truncated streams, and trailing garbage.
    """
    raw = bytes(data)
    if len(raw) < _HEADER.size:
        raise Qpk1Error(f"stream is {len(raw)} bytes; shorter than the {_HEADER.size}-byte header")
    magic, version, tag, dout, din, r = _HEADER.unpack_from(raw)
    if magic != QPK1_MAGIC:
        raise Qpk1Error(f"bad magic {magic!r}; expected {QPK1_MAGIC!r}")
    if version != REPRESENTATION_VERSION:
        raise Qpk1Error(
            f"unsupported representation version {version}; this reader implements "
            f"version {REPRESENTATION_VERSION} only"
        )
    if tag not in _TAGS_TO_DTYPES:
        raise Qpk1Error(f"unknown dtype tag {tag}; expected 1 (float16) or 2 (float32)")
    _require_dims(dout, din, r)
    _require_rank_feasible(dout, din, r)

    dt = _TAGS_TO_DTYPES[tag]
    expected = expected_size(dout, din, r, dt)
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
    ub = _unpack_bits(take((dout * r + 7) // 8), dout * r).reshape(dout, r)
    vb = _unpack_bits(take((din * r + 7) // 8), din * r).reshape(din, r)
    return Qpk1Layer(ub, vb, s1, s2, dt)


def bpw_from_bytes(num_bytes: int, dout: int, din: int) -> float:
    """Effective bits per weight from the serialized byte count only (D5).

    ``BPW = 8 * num_bytes / (dout * din)`` — header, scales, and padding
    included; never derived from parameter counts.
    """
    if dout <= 0 or din <= 0:
        raise Qpk1Error(f"dout and din must be >= 1, got {(dout, din)}")
    return 8 * num_bytes / (dout * din)


def expected_size(dout: int, din: int, r: int, dtype=np.float16) -> int:
    """Exact serialized byte count for a layer at ``(dout, din, r, dtype)``.

    ``24 + (dout + din) * itemsize + ceil(dout*r/8) + ceil(din*r/8)`` (D1).
    """
    dt = _canonical_dtype(dtype)
    _require_dims(dout, din, r)
    return _HEADER.size + (dout + din) * dt.itemsize + (dout * r + 7) // 8 + (din * r + 7) // 8


def rank_for_target_bpw(dout: int, din: int, b: float, dtype=np.float16) -> int:
    """Planning rank hitting target BPW ``b`` (D6), re-checked as an int.

    Inverts the byte-size model:
    ``r(b) = (b * dout * din - 8 * (24 + (dout + din) * itemsize)) / (dout + din)``,
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
    numerator = b * dout * din - 8 * (_HEADER.size + (dout + din) * dt.itemsize)
    if numerator <= 0:
        raise ValueError(
            f"target BPW {b} is infeasible at ({dout}, {din}) with {dt}: the fixed "
            f"header-plus-scale overhead alone needs more bits than the budget"
        )
    r = round(numerator / (dout + din))
    if r > min(dout, din):
        raise ValueError(f"target BPW {b} requires rank {r} > min(dout, din)={min(dout, din)}")
    return max(1, r)
