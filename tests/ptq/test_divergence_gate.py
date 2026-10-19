"""Scenario tests: post-balance divergence gate (qk-divergence-guards).

Pins the qk-ptq ADDED requirement scenarios `divergence_gate_healthy_passes`,
`divergence_gate_diverged_aborts`, and `divergence_gate_nan_rejected`.
The gate lives in block_recon and fires at balance time per layer.
"""

import numpy as np
import pytest

from quantibbler.ptq.block_recon import check_layer_magnitude
from quantibbler.ptq.config import PTQConfig


def _factors(seed, dout=32, din=64, r=8, scale=1.0):
    rng = np.random.default_rng(seed)
    U = scale * rng.standard_normal((dout, r))
    V = scale * rng.standard_normal((din, r))
    s1 = np.full(dout, 0.5)
    s2 = np.full(din, 0.25)
    return U, V, s1, s2


def test_scn_divergence_gate_healthy_passes():
    """In-band ratio: no raise; the helper returns the measured ratio."""
    U, V, s1, s2 = _factors(0)
    W = (s1[:, None] * (np.sign(U) @ np.sign(V).T)) * s2[None, :]
    cfg = PTQConfig(rank=8)
    ratio = check_layer_magnitude("layers.0.inner.mlp.down_proj", U, V, s1, s2, W, cfg)
    assert ratio == pytest.approx(1.0, rel=1e-12)


def test_scn_divergence_gate_diverged_aborts():
    """Out-of-band inflation: dedicated error names layer, ratio, and band."""
    U, V, s1, s2 = _factors(1)
    W = (s1[:, None] * (np.sign(U) @ np.sign(V).T)) * s2[None, :]
    W_small = W / 1000.0  # reconstructed >> fp target: the incident signature
    cfg = PTQConfig(rank=8, divergence_band_factor=100.0)
    with pytest.raises(Exception) as ei:
        check_layer_magnitude("layers.0.inner.mlp.down_proj", U, V, s1, s2, W_small, cfg)
    msg = str(ei.value)
    assert "layers.0.inner.mlp.down_proj" in msg
    assert "1000" in msg or "1e+03" in msg or "1.0e+03" in msg
    assert "100" in msg  # the band factor appears


def test_scn_divergence_gate_nan_rejected():
    """Any non-finite scale entry is rejected with the same contract."""
    U, V, s1, s2 = _factors(2)
    s1_bad = s1.copy()
    s1_bad[3] = np.nan
    W = np.ones((32, 64))
    cfg = PTQConfig(rank=8)
    with pytest.raises(Exception) as ei:
        check_layer_magnitude("layers.0.inner.mlp.down_proj", U, V, s1_bad, s2, W, cfg)
    assert "layers.0.inner.mlp.down_proj" in str(ei.value)


def test_scn_divergence_gate_band_respected_custom_factor():
    """A custom band factor widens/narrows the acceptance region."""
    U, V, s1, s2 = _factors(3)
    W = (s1[:, None] * (np.sign(U) @ np.sign(V).T)) * s2[None, :]
    cfg = PTQConfig(rank=8, divergence_band_factor=100.0)
    ratio = check_layer_magnitude("layers.0.inner.o_proj", U, V, s1, s2, W / 50.0, cfg)
    assert ratio == pytest.approx(50.0, rel=1e-12)
