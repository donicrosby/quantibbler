"""Scenario tests: preconditioner tau default policy (qk-divergence-guards).

Pins the qk-ptq delta scenarios `preconditioner_tau_default_q95`,
`preconditioner_tau_explicit_override`, and `preconditioner_tau_healthy_noop`:
tau=None must derive a finite per-vector q95 ceiling (engaging on heavy
tails), explicit tau keeps verbatim A-1 semantics, and concentrated
(healthy) vectors are unaffected within noise.
"""

import numpy as np

from quantibbler.ptq.calibration import robust_diag


def _heavy_tailed(d=512, seed=7):
    """Diagonal statistic with a median ~1 and rare huge entries (block-0-like)."""
    rng = np.random.default_rng(seed)
    v = rng.lognormal(mean=0.0, sigma=0.25, size=d)
    v[rng.choice(d, size=6, replace=False)] = rng.lognormal(6.0, 0.5, size=6)
    return v


def test_scn_preconditioner_tau_default_q95():
    """tau=None derives the per-vector q95 cap on heavy tails; clips below max."""
    v = _heavy_tailed()
    d = robust_diag(v, gamma=0.0)  # shrinkage off: cap effect isolated
    # OLD default (cap at max) returned max(v) verbatim; the tail-gated q95
    # default must strictly bound below the raw maximum here.
    assert d.max() < v.max()
    q95 = float(np.percentile(v, 95))
    # The fixture is genuinely heavy-tailed: max far above the gate.
    assert v.max() > 10.0 * q95
    # Every entry above the q95 estimate is pinned at (about) the cap.
    assert np.isclose(d.max(), q95, rtol=1e-12)
    # Entries below the cap pass through exactly (gamma=0).
    below = v <= q95
    assert np.allclose(d[below], v[below], rtol=0, atol=0)


def test_scn_preconditioner_tau_tail_gate_inert():
    """tau=None leaves a moderately-tailed vector untouched (gate inert)."""
    rng = np.random.default_rng(13)
    v = np.exp(rng.normal(0.0, 1.0, size=512))  # lognormal: max ~ 4-8x q95
    gate = float(v.max()) / float(np.percentile(v, 95))
    assert gate <= 10.0  # fixture is below the tail gate
    d_new = robust_diag(v, gamma=0.0)
    d_old = robust_diag(v, tau=np.inf, gamma=0.0)  # no-clip reference
    assert np.array_equal(d_new, d_old)


def test_scn_preconditioner_tau_explicit_override():
    """An explicit numeric tau is the verbatim A-1 absolute cap."""
    v = _heavy_tailed()
    tau = float(np.percentile(v, 50))
    d = robust_diag(v, tau=tau, gamma=0.0)
    assert d.max() <= tau + 1e-15
    assert np.isclose(d.max(), tau, rtol=1e-12)
    below = v <= tau
    assert np.allclose(d[below], v[below], rtol=0, atol=0)


def test_scn_preconditioner_tau_healthy_noop():
    """On a concentrated vector the q95 default matches the old max-cap path
    within the clip gap (max - q95 is sub-percent of the median there)."""
    rng = np.random.default_rng(11)
    v = 1.0 + 0.1 * rng.random(size=256)  # max/q95 ~ 1.0x: no meaningful tail
    d_new = robust_diag(v, gamma=0.2)
    d_old = robust_diag(v, tau=float(v.max()), gamma=0.2)
    gap = (float(v.max()) - float(np.percentile(v, 95))) / float(np.median(v))
    assert gap < 0.02  # fixture is genuinely concentrated
    assert np.allclose(d_new, d_old, rtol=0.0, atol=gap)


def test_scn_preconditioner_tau_default_constant_vector():
    """Degenerate flat vector: q95 == max, floor keeps entries finite-positive."""
    v = np.full(64, 3.0)
    d = robust_diag(v, gamma=0.2)
    assert np.all(np.isfinite(d)) and np.all(d > 0)
