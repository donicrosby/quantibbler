"""Named accumulation dtypes: where bf16's 7-bit mantissa is insufficient.

The pipeline computes in ONE declared dtype (``PTQConfig.compute_dtype``) and
never silently upcasts. A site that must widen says so by name: the accumulation
dtype is a pure function of the compute dtype — float32 under bf16 compute,
identity at float32 and float64 — never a second user knob (a tunable
``accumulation_dtype`` invites a rank-4096 solve that silently accumulates in
bf16, which is garbage with no error).

Why float32 under bf16: a dot product over ``d`` terms accumulates roughly one
relative ULP of rounding per term, and bf16's relative ULP is 2^-7 (~1.4). The
float32 accumulator has 23 mantissa bits (~16 guard bits over bf16) and stays
safe for ranks up to ~4096 before a float64 accumulator would be required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "ACCUM_DTYPE_SPECS",
    "TORCH_DTYPE_SPECS",
    "accumulation_np_dtype",
    "accumulation_spec",
    "accumulation_torch_dtype",
    "torch_dtype",
]

#: compute dtype -> accumulation dtype. Pure mapping, deliberately not a knob.
ACCUM_DTYPE_SPECS: dict[str, str] = {
    "bf16": "float32",
    "float32": "float32",
    "float64": "float64",
}

#: pipeline dtype name -> torch attribute name. The two vocabularies differ
#: exactly once (bf16 / bfloat16), which is enough for a ``getattr`` to fail.
TORCH_DTYPE_SPECS: dict[str, str] = {
    "bf16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
}


def accumulation_spec(compute_dtype: str) -> str:
    """The accumulation dtype NAME for a declared compute dtype."""
    try:
        return ACCUM_DTYPE_SPECS[compute_dtype]
    except KeyError:
        raise ValueError(
            f"unknown compute_dtype {compute_dtype!r}: expected one of {tuple(ACCUM_DTYPE_SPECS)}"
        ) from None


def accumulation_np_dtype(compute_dtype: str) -> np.dtype:
    """The accumulation dtype as a NumPy dtype."""
    return np.dtype(accumulation_spec(compute_dtype))


def accumulation_torch_dtype(compute_dtype: str) -> torch.dtype:
    """The accumulation dtype as a torch dtype (lazy torch import)."""
    import torch  # noqa: PLC0415

    return getattr(torch, accumulation_spec(compute_dtype))


def torch_dtype(spec: str) -> torch.dtype:
    """A declared compute dtype as a torch dtype (lazy torch import).

    An explicit mapping, never ``getattr(torch, spec)``: the pipeline's name for
    the dtype is ``bf16`` and torch's is ``bfloat16``, so the naive lookup raises
    ``AttributeError`` on the pipeline's OWN default — the failure mode that
    would red every bf16 run at its first tensor creation.
    """
    import torch  # noqa: PLC0415

    name = TORCH_DTYPE_SPECS.get(spec)
    if name is None:
        raise ValueError(
            f"unknown compute dtype {spec!r}: expected one of {tuple(TORCH_DTYPE_SPECS)}"
        )
    return getattr(torch, name)
