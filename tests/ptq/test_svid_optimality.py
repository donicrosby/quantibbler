"""Red-first: SVID must be the paper's OPTIMAL two-sided rank-1, not a
column-only restriction.

[NQ] :592 defines SVID as the "optimal rank-1 approximation that preserves the
sign structure". With S = sign(P) and S^2 = 1 entrywise,

    min_{g,h} ||P - S o (g h^T)||_F  ==  min_{g,h} || |P| - g h^T ||_F

so the optimum is the top rank-1 SVD of |P| = sign(P) o P. The retired
column-only proxy fixed the row vector at 1 (one scalar per column), which is
optimal only inside that restricted class and cannot represent row structure.

This file pins the optimum and proves the test discriminates: the same fixture
is run through the retired column-only closed form and asserted to FAIL.
"""

import numpy as np

from quantibbler.ptq.lb_admm import DEFAULT_SCALE_FLOOR, svid_proxy
from quantibbler.ptq.solver_backend import get_backend

_NP = get_backend("numpy")


def _fixture():
    """16x8 with a planted rank-1 magnitude and a deterministic sign pattern."""
    rows = np.arange(16, dtype=np.float64)
    cols = np.arange(8, dtype=np.float64)
    g = 0.5 + rows / 4.0  # non-constant: column-only cannot fit this
    h = 1.0 + cols / 2.0
    signs = np.where(((rows[:, None].astype(int) + cols[None, :].astype(int)) % 2) == 0, 1.0, -1.0)
    return signs, np.outer(g, h)


def _retired_column_only(z):
    """The proxy this change replaces, verbatim from the old closed form."""
    zf = np.asarray(z, dtype=np.float64)
    s = np.sign(zf)
    s[s == 0.0] = 1.0
    num = np.einsum("ij,ij->j", s, zf)
    den = np.einsum("ij,ij->j", s, s)
    return s * (num / den)[None, :]


def test_two_sided_svid_recovers_the_planted_rank1_value_exactly():
    """The optimum reproduces a planted rank-1 magnitude to 1e-12."""
    signs, mag = _fixture()
    P = signs * mag
    Z = svid_proxy(_NP, P)
    assert Z.shape == P.shape
    assert np.allclose(Z, P, rtol=1e-12, atol=1e-12)


def test_the_retired_column_only_proxy_cannot_recover_it():
    """Discrimination guard: the old variant fails this same fixture.

    If this ever starts passing, the fixture stopped discriminating and the
    test above proves nothing about optimality.
    """
    signs, mag = _fixture()
    P = signs * mag
    col_only = _retired_column_only(P)
    assert not np.allclose(col_only, P, rtol=1e-12, atol=1e-12)
    # and it is strictly worse in the objective the paper defines
    assert np.linalg.norm(P - svid_proxy(_NP, P)) < np.linalg.norm(P - col_only)


def test_sign_structure_is_preserved():
    """Signs of the input survive, with sign(0) = +1."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(16, 8))
    z[3, 2] = 0.0
    Z = svid_proxy(_NP, z)
    expected = np.sign(z)
    expected[expected == 0.0] = 1.0
    assert np.array_equal(np.sign(Z), expected)


def test_dead_columns_get_the_positive_scale_floor():
    """All-zero input: sign(0) = +1 and no output entry is ever zero."""
    Z = svid_proxy(_NP, np.zeros((16, 8)))
    assert np.all(Z == DEFAULT_SCALE_FLOOR)
    assert np.all(np.sign(Z) == 1.0)
