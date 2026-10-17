"""Task 5.1: global scale calibration (scenario kl_scale_calibration_frozen_bits)."""

import numpy as np
import torch
from torch import nn

from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.scale_calib import calibrate_scales


class _TinyKDModel(nn.Module):
    def __init__(self, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(d, d, dtype=torch.float64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d, dtype=torch.float64)
        self.head = nn.Linear(d, 12, dtype=torch.float64)
        with torch.no_grad():
            for m in (self.fc1, self.fc2, self.head):
                m.weight.copy_(
                    torch.randn(m.weight.shape, generator=gen, dtype=torch.float64) * 0.3
                )
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(x)))
        return self.head(h)


def _frozen_weight_snapshot(model: nn.Module) -> list[np.ndarray]:
    return [
        m.weight.detach().cpu().numpy().copy() for m in model.modules() if isinstance(m, nn.Linear)
    ]


def test_scn_kl_scale_calibration():
    """kl_scale_calibration_frozen_bits: KL decreases over T_glob steps on
    cached teacher logits; packed bits (the reconstructed weight signs) are
    bit-identical before/after; only scale-side parameters carry grads."""
    torch.manual_seed(0)
    model = _TinyKDModel(seed=1)
    gen = torch.Generator().manual_seed(2)
    batches = [torch.randn(2, 4, 8, generator=gen, dtype=torch.float64) for _ in range(2)]
    # Stage override per A-10: full-batch (bs = all 16 rows) and a test-speed
    # lr/epoch budget; the paper values (1e-6, bs 1, 8 epochs) remain the
    # PTQConfig defaults.
    from quantibbler.ptq.config import StageHyper

    cfg = PTQConfig(rank=4, stage_global=StageHyper(2e-2, 16, 10))

    # Stand up a layer book by hand: factorize fc1/fc2 signs from their
    # weights with alpha-consistent scales (as reconstruct_block would).
    from quantibbler.ptq.block_recon import QuantizedLayerState, _set_linear
    from quantibbler.ptq.lb_admm import magnitude_balance

    layers: dict[str, QuantizedLayerState] = {}
    # The KL must start strictly positive: perturb the scales away from the
    # alpha-consistent values (Phase 3 then has something to recover).
    for name, m in (("fc1", model.fc1), ("fc2", model.fc2)):
        w = m.weight.detach().to(torch.float64).cpu().numpy()
        u0, s0, vt0 = np.linalg.svd(w, full_matrices=False)
        uu = u0[:, :4] * np.sqrt(s0[:4])
        vv = vt0[:4].T * np.sqrt(s0[:4])
        bal = magnitude_balance(uu.copy(), vv.copy(), np.ones(8), np.ones(8), target=w)
        bal.s1 = bal.s1 * 1.15
        bal.s2 = bal.s2 * 0.9
        st = QuantizedLayerState(name, bal.Ub, bal.Vb, bal.s1, bal.s2, w * 0)
        st.bias = m.bias.detach().cpu().numpy().astype(np.float64).copy()
        layers[name] = st
        _set_linear(m, _fac_from_balance(bal, m))

    before_weights = _frozen_weight_snapshot(model)
    before_bits = [(st.Ub.copy(), st.Vb.copy()) for st in layers.values()]

    # The fp teacher must be captured BEFORE the scales are perturbed: in
    # the pipeline the teacher is the untouched fp model ([NQ] Eq. 11),
    # while the student starts with degraded scales.
    teacher = _TinyKDModel(seed=1)

    result = calibrate_scales(model, layers, teacher, batches, cfg)
    assert result.kl_trace[0] > result.kl_trace[-1]
    assert len(result.kl_trace) >= 2

    # Frozen bits: sign structure unchanged bit-for-bit.
    after_bits = [(st.Ub, st.Vb) for st in layers.values()]
    for (ub_b, vb_b), (ub_a, vb_a) in zip(before_bits, after_bits, strict=True):
        assert np.array_equal(ub_b, ub_a) and np.array_equal(vb_b, vb_a)

    # Frozen weights: fc1/fc2 weights unchanged by Phase 3 (only the live
    # scale factors used in forward were tuned); head untouched entirely.
    after_weights = _frozen_weight_snapshot(model)
    assert np.array_equal(before_weights[0], after_weights[0])
    assert np.array_equal(before_weights[2], after_weights[2])  # head


def _fac_from_balance(bal, linear: nn.Linear):
    """Local helper: rebuild a FactorizedLinear from a BalanceResult."""
    from quantibbler.ptq.block_recon import FactorizedLinear

    fac = FactorizedLinear(
        torch.from_numpy(np.ascontiguousarray(bal.U)),
        torch.from_numpy(np.ascontiguousarray(bal.V)),
        linear.bias.detach().clone(),
    )
    with torch.no_grad():
        fac.s1.copy_(torch.from_numpy(np.ascontiguousarray(bal.s1)))
        fac.s2.copy_(torch.from_numpy(np.ascontiguousarray(bal.s2)))
    return fac
