"""Declared storage dtype for a qpk1 model directory's remainder sidecars.

The per-layer qpk1 streams carry the factorized weights (``qk-representation``);
the FP tensors they do NOT carry — the embedding, an untied lm_head, and the
norm weights — live in the model directory's sidecar files. This module owns
their dtype vocabulary and codec, and it is the Python half of the byte
contract ``crates/quantibbler-qpk1/src/loader.rs`` enforces:

- the dtype is DECLARED — an optional ``dtype`` key in the manifest's
  ``[sidecar]`` block, whose absence is the fp32 reading every pre-existing
  artifact already has. The declaration is the ONLY thing that selects the
  itemsize; a file's byte length never infers a dtype.
- ``fp32`` is IEEE binary32 LE (4 bytes/element); ``bf16`` is IEEE bfloat16 LE
  (2 bytes/element) — the high half of an f32 word.
- widening is exact in both directions of the hop: bf16 -> f32 zero-fills the
  low 16 bits and f32 -> f64 is exact, so a decode never invents a bit.
  ENCODING to bf16 is round-to-nearest-even and therefore lossy — 8 mantissa
  bits — which is why a bf16 remainder is compared against its fp32 original
  under a declared bf16 tolerance and never under the 1e-12 NumPy-parity bound.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "DEFAULT_REMAINDER_DTYPE",
    "REMAINDER_DTYPES",
    "RemainderDtypeError",
    "cast_to_bf16",
    "decode",
    "encode",
    "itemsize",
]

#: Declared dtype vocabulary, in the order the specs state it.
REMAINDER_DTYPES = ("fp32", "bf16")
#: The reading of an ABSENT ``[sidecar] dtype`` key: the historical byte
#: contract, so a pre-change artifact never has to be re-declared.
DEFAULT_REMAINDER_DTYPE = "fp32"

_ITEMSIZE = {"fp32": 4, "bf16": 2}


class RemainderDtypeError(ValueError):
    """A remainder declaration or payload is not one this build reads."""


def itemsize(dtype: str) -> int:
    """Bytes per element for a declared remainder dtype."""
    try:
        return _ITEMSIZE[dtype]
    except KeyError:
        raise RemainderDtypeError(
            f"unknown remainder dtype {dtype!r}; declared dtypes are "
            f"{' and '.join(REMAINDER_DTYPES)} (an absent declaration reads as fp32)"
        ) from None


def cast_to_bf16(values) -> torch.Tensor:
    """The ONE round-to-nearest-even fp -> bf16 cast, finiteness-gated.

    Every bf16 write in the repo goes through here (the exporter's bf16 path and
    the repack utility), so the two faces cannot drift. bf16's exponent is eight
    bits wide, so a finite f32 above its range rounds to inf — and a round trip
    through a non-finite word is a payload no reader expects. Both the source and
    the result are therefore checked, and a non-finite value is a named refusal.
    torch.bfloat16 is the runtime the pipeline's model side already evaluates in,
    so this adds no dependency and no second definition of "bf16".
    """
    source = torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.float32)
    if not bool(torch.isfinite(source).all()):
        raise RemainderDtypeError("bf16 encoding refuses non-finite values")
    cast = source.to(torch.bfloat16)
    if not bool(torch.isfinite(cast).all()):
        raise RemainderDtypeError(
            "bf16 encoding refuses values outside bf16's finite range (they overflow to inf)"
        )
    return cast


def encode(values, dtype: str) -> bytes:
    """Little-endian bytes of ``values`` at the declared dtype."""
    if dtype == "bf16":
        words = cast_to_bf16(values).view(torch.uint16).numpy()
        return np.ascontiguousarray(words, dtype="<u2").tobytes()
    if dtype == "fp32":
        return np.ascontiguousarray(values, dtype="<f4").tobytes()
    raise RemainderDtypeError(
        f"unknown remainder dtype {dtype!r}; declared dtypes are {' and '.join(REMAINDER_DTYPES)}"
    )


def decode(raw: bytes | bytearray | memoryview, dtype: str) -> np.ndarray:
    """Widen a remainder payload to float64 at the declared dtype."""
    if dtype == "bf16":
        words = np.frombuffer(bytes(raw), dtype="<u2")
        return (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    if dtype == "fp32":
        return np.frombuffer(bytes(raw), dtype="<f4").astype(np.float64)
    raise RemainderDtypeError(
        f"unknown remainder dtype {dtype!r}; declared dtypes are {' and '.join(REMAINDER_DTYPES)}"
    )
