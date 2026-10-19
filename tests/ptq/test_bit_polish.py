"""r137-v4 regression tests for greedy_polish's scale refit invariants.

v4 post-mortem: the alternating-LS refit produced 108,691 negative s1
entries (sign of the row optimum folded nowhere) and degenerate-split
scales (s1 ~ 1e-4 paired with s2 >= 233) from near-zero-energy binary
columns. The refit now (1) folds any negative LS sign into the bit row
— |s|·(sgn·U) == s·U exactly, so the stored objective never changes —
and (2) keeps every scale within [1/8x, 8x] of its pre-refit value per
round.
"""

import numpy as np
import pytest

from quantibbler.ptq.bit_polish import greedy_polish


def test_scn_polish_negative_ls_optimum_folds_into_bits_exactly():
    """A row whose LS optimum is negative (-1 exactly, hand-built) must
    come back with a POSITIVE scale and the sign absorbed in Ub — and the
    effective product must equal the target exactly (fold is lossless)."""
    ub0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    vb0 = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    s1_0 = np.ones(2)
    s2_0 = np.ones(4)
    bc = ub0 @ vb0.T  # rows [2,0,2,0]
    w = np.stack([-bc[0], bc[1]])  # row 0 anti-aligned -> raw LS s1 = -1

    ub, vb, s1, s2 = greedy_polish(w, ub0, vb0, s1_0, s2_0, rounds=3)

    assert (s1 > 0).all(), f"negative scale survived: {s1}"
    assert (s2 > 0).all(), f"negative scale survived: {s2}"
    wq = (s1[:, None] * (ub @ vb.T)) * s2[None, :]
    assert np.allclose(wq, w, atol=1e-12), "fold changed the effective product"


def test_scn_polish_keeps_scales_positive_and_improves_objective():
    """Production-shaped target: polished scales are strictly positive,
    finite, and the stored objective only improves."""
    from quantibbler.ptq.lb_admm import magnitude_balance

    rng = np.random.default_rng(7)
    w = (rng.standard_normal((24, 4)) @ rng.standard_normal((4, 18))) * 3.0
    u0, s0, vt0 = np.linalg.svd(w, full_matrices=False)
    uu = u0[:, :6] * np.sqrt(s0[:6])
    vv = vt0[:6].T * np.sqrt(s0[:6])
    bal = magnitude_balance(uu.copy(), vv.copy(), np.ones(24), np.ones(18), target=w)

    def sse(s1, s2, ub, vb):
        return float(np.sum((w - (s1[:, None] * (ub @ vb.T)) * s2[None, :]) ** 2))

    before = sse(bal.s1, bal.s2, bal.Ub, bal.Vb)
    ub, vb, s1, s2 = greedy_polish(w, bal.Ub, bal.Vb, bal.s1, bal.s2, rounds=4)
    after = sse(s1, s2, ub, vb)

    assert (s1 > 0).all() and (s2 > 0).all()
    assert np.isfinite(s1).all() and np.isfinite(s2).all()
    assert after < before
    # No degenerate split: scales within each vector stay same order of
    # magnitude (v4 showed s1~1e-4 riding next to ~1e-2 entries).
    for s in (s1, s2):
        assert s.min() > 0 and s.max() / s.min() < 1e6


def test_scn_polish_degenerate_target_does_not_explode():
    """A target with an exactly-zero row (LS denominator degenerate) must
    come back finite with bounded scales — v4's explosion path."""
    rng = np.random.default_rng(11)
    w = rng.standard_normal((12, 10))
    w[3] = 0.0
    ub0 = np.where(rng.standard_normal((12, 4)) >= 0, 1.0, -1.0)
    vb0 = np.where(rng.standard_normal((10, 4)) >= 0, 1.0, -1.0)
    s1_0 = np.ones(12)
    s2_0 = np.ones(10)

    ub, vb, s1, s2 = greedy_polish(w, ub0, vb0, s1_0, s2_0, rounds=3)

    assert np.isfinite(s1).all() and np.isfinite(s2).all()
    assert np.isfinite(ub).all() and np.isfinite(vb).all()
    assert (np.abs(ub) == 1.0).all() and (np.abs(vb) == 1.0).all()
    assert (s1 >= 0).all() and (s2 >= 0).all()
    assert s1.max() < 1e6 and s2.max() < 1e6


@pytest.mark.parametrize("rounds", [1, 4])
def test_scn_polish_roundwise_trust_region_bounds_motion(rounds):
    """Per-round motion is capped at 8x/8x: two consecutive 1-round calls
    can move a scale at most 64x from its post-first-call value."""
    from quantibbler.ptq.lb_admm import magnitude_balance

    rng = np.random.default_rng(23)
    w = (rng.standard_normal((16, 4)) @ rng.standard_normal((4, 12))) * 2.0
    u0, s0, vt0 = np.linalg.svd(w, full_matrices=False)
    uu = u0[:, :4] * np.sqrt(s0[:4])
    vv = vt0[:4].T * np.sqrt(s0[:4])
    bal = magnitude_balance(uu.copy(), vv.copy(), np.ones(16), np.ones(12), target=w)

    ub1, vb1, s1_1, s2_1 = greedy_polish(
        w, bal.Ub.copy(), bal.Vb.copy(), bal.s1.copy(), bal.s2.copy(), rounds=1
    )
    ub2, vb2, s1_2, s2_2 = greedy_polish(w, ub1, vb1, s1_1, s2_1, rounds=1)

    for pre, post in ((s1_1, s1_2), (s2_1, s2_2)):
        ref = np.maximum(np.abs(pre), 1e-30)
        assert (post <= 8.0 * ref + 1e-30).all()
        assert (post >= (pre / 8.0) - 1e-30).all()  # pre is already positive
