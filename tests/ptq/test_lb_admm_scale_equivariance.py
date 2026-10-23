"""Red-first: scale equivariance of lb_admm's returned factors.

`lb_admm` normalizes its target by ``||w||_F`` on entry and rescales the
factors on exit. The entry comment claims that pair is "exactly
scale-equivariant". It is only equivariant if the exit rescale lands on the
PRODUCT once: ``W ~ U V^T`` has two factors, so multiplying BOTH by ``t``
multiplies the reconstructible matrix by ``t^2``. Rescaling both factors by
``||w||_F`` therefore overshoots the reconstructed matrix by exactly
``||w||_F`` -- ~2.76e11 on layers.0.self_attn.q_proj at rank 512, which is
the scale the packed artifact carries.
"""

import numpy as np
import pytest

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm


def _ground_truth(m: int, n: int, r: int, seed: int) -> np.ndarray:
    """Exact rank-r matrix built from +-1 factors."""
    gen = np.random.default_rng(seed)
    ub = np.where(gen.random((m, r)) < 0.5, -1.0, 1.0)
    vb = np.where(gen.random((n, r)) < 0.5, -1.0, 1.0)
    return ub @ vb.T


def _product_norm(res) -> float:
    return float(np.linalg.norm(np.asarray(res.P_u) @ np.asarray(res.P_v).T))


@pytest.mark.parametrize("factor", [1e3, 1e6, 2.759e11])
def test_returned_product_is_scale_equivariant(factor):
    """Scaling the target by c must scale the reconstructed product by c.

    The solve normalizes internally, so the two runs follow an identical
    trajectory and the ratio is exact -- not approximate. An overshoot of
    ``factor`` (i.e. a ratio of ``factor**2``) is the double-rescale defect.
    """
    r = 4
    w = _ground_truth(20, 14, r, seed=7).astype(np.float64)
    cfg = PTQConfig(solver_backend="numpy", rank=r, rho_start=1e-2, rho_end=1e2)

    base = _product_norm(lb_admm(w, cfg))
    scaled = _product_norm(lb_admm(w * factor, cfg))

    ratio = scaled / base
    assert ratio == pytest.approx(factor, rel=1e-5), (
        f"target scaled by {factor:.6e} produced a product scaled by "
        f"{ratio:.6e} -- an overshoot of {ratio / factor:.6e}x, i.e. one "
        f"extra factor of ||w||_F={np.linalg.norm(w):.6e}"
    )
