"""Accumulation dtype separation (qk-ptq's "Where bf16 mantissa width is
insufficient, each accumulation site SHALL declare an explicit accumulation
dtype" requirement, scenarios accumulation_dtype_is_named_per_site and
accumulation_identity_at_higher_dtypes).

The mapping is a pure function of compute_dtype — float32 under bf16 compute,
identity at float32 and float64 — and it is deliberately NOT a second config
knob: a tunable accumulation dtype invites a rank-4096 solve that accumulates in
bf16 without an error.
"""

import numpy as np
import pytest

from quantibbler.ptq.accumulation import (
    ACCUM_DTYPE_SPECS,
    accumulation_np_dtype,
    accumulation_spec,
    accumulation_torch_dtype,
    torch_dtype,
)


def test_scn_accumulation_dtype_is_named_per_site():
    """bf16 storage widens to a float32 accumulator at named sites."""
    assert accumulation_spec("bf16") == "float32"
    assert accumulation_np_dtype("bf16") == np.dtype(np.float32)


def test_scn_accumulation_identity_at_higher_dtypes():
    """float32 and float64 accumulate in their own dtype (identity)."""
    assert accumulation_spec("float32") == "float32"
    assert accumulation_spec("float64") == "float64"
    assert accumulation_np_dtype("float32") == np.dtype(np.float32)
    assert accumulation_np_dtype("float64") == np.dtype(np.float64)


def test_accumulation_is_not_a_config_knob():
    """The mapping is derived, not settable: unknown specs raise by name."""
    assert set(ACCUM_DTYPE_SPECS) == {"bf16", "float32", "float64"}
    with pytest.raises(ValueError):
        accumulation_spec("float16")


def test_torch_dtype_maps_the_pipeline_spelling_to_torch():
    """The pipeline spells the default `bf16`; torch spells it `bfloat16`.

    A `getattr(torch, spec)` lookup fails on exactly that name — the one that is
    now the default — so the mapping is explicit and asserted here.
    """
    torch = pytest.importorskip("torch")
    assert torch_dtype("bf16") is torch.bfloat16
    assert torch_dtype("float32") is torch.float32
    assert torch_dtype("float64") is torch.float64
    with pytest.raises(ValueError):
        torch_dtype("float16")


def test_accumulation_torch_dtype_tracks_the_spec():
    torch = pytest.importorskip("torch")
    assert accumulation_torch_dtype("bf16") is torch.float32
    assert accumulation_torch_dtype("float32") is torch.float32
    assert accumulation_torch_dtype("float64") is torch.float64
