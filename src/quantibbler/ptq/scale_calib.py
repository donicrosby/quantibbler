"""Phase 3 global scale calibration (tasks 5.1-5.2; [NQ] Sec 3.3, Eq. 11,
Algorithm 1 lines 26-27).

With the packed binaries bit-frozen, tunes only the per-layer scale vectors
{ s1, s2 } of all quantized layers to minimize the KL divergence between
teacher and quantized logit distributions under temperature T (Eq. 11).
Teacher logits are computed once and cached. The frozen weight matrices
never receive gradients: the live forward applies scales as
``x @ ((s2 ⊙ Vb) (Ub ⊙ s1))ᵀ`` with the factorized parameters in an autograd
graph of their own, while the underlying ``nn.Linear`` weights stay fixed.

Stage hyperparameters per [NQ] App. C: lr 1e-6, batch size 1, 8 epochs,
cosine schedule (config-overridable; assumption A-10 allows tiny-model
overrides for test speed).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .block_recon import QuantizedLayerState
from .config import PTQConfig

__all__ = ["ScaleCalibResult", "calibrate_scales"]


class ScaleCalibResult:
    """Diagnostics of one Phase 3 run.

    Attributes:
        kl_trace: KL value after each optimization step (float64 list).
        n_steps: optimization steps executed.
    """

    __slots__ = ("kl_trace", "n_steps")

    def __init__(self, kl_trace: list[float], n_steps: int) -> None:
        self.kl_trace = kl_trace
        self.n_steps = n_steps


def _kl_with_logits(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float):
    """KL(softmax(z_T) || softmax(ẑ_T)) of Eq. 11, token-averaged (nats)."""
    t = max(float(temperature), 1e-12)
    log_p = torch.log_softmax(teacher_logits / t, dim=-1)
    log_q = torch.log_softmax(student_logits / t, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


def calibrate_scales(
    model: nn.Module,
    layers: dict[str, QuantizedLayerState],
    teacher: nn.Module,
    calibration_batches,
    cfg: PTQConfig,
) -> ScaleCalibResult:
    """Tune only the scale vectors under the KD objective of Eq. 11.

    Args:
        model: the quantized host model (quantized linears carry frozen
            factorized weights; scales are applied from ``layers``).
        layers: per-layer artifacts whose ``s1``/``s2`` receive the updates.
        teacher: the fp teacher (its logits are cached once up front).
        calibration_batches: token-id batches.
        cfg: pipeline configuration (temperature, stage_global, cosine).

    Returns:
        :class:`ScaleCalibResult` with the KL trace.

    The packed binaries are never touched: gradients flow only to the
    scale tensors held in ``layers`` (frozen-bit invariant asserted by
    test_scn_kl_scale_calibration).
    """
    batches = list(calibration_batches)
    if not batches:
        raise ValueError("scale calibration needs calibration batches")

    linears = dict((name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear))
    # Scale tensors live outside the modules so no Linear weight is ever in
    # the autograd graph of this phase.
    s1_params = {
        name: torch.from_numpy(np.ascontiguousarray(st.s1, dtype=np.float64))
        for name, st in layers.items()
    }
    s2_params = {
        name: torch.from_numpy(np.ascontiguousarray(st.s2, dtype=np.float64))
        for name, st in layers.items()
    }
    for t in (*s1_params.values(), *s2_params.values()):
        t.requires_grad_(True)
    opt = torch.optim.Adam([*s1_params.values(), *s2_params.values()], lr=cfg.stage_global.lr)

    # Build the quantized-prefix input by running the batches through the
    # frozen model once (input token ids feed embeddings upstream; for a
    # bare block stack the batches may already be activations).
    def host_forward(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    def quantized_forward_with_scales(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    # Teacher logits, cached once ([NQ] Sec 3.3 memory note).
    teacher_logits = torch.cat([teacher(b) for b in batches], dim=0).detach()

    n = sum(b.shape[0] for b in batches)
    bs = max(1, min(cfg.stage_global.batch_size, n))
    total = max(cfg.stage_global.epochs * (n // bs or 1), 1)
    kl_trace: list[float] = []
    inputs = torch.cat(batches, dim=0)
    all_logits = teacher_logits

    def scale_forward(x: torch.Tensor) -> torch.Tensor:
        """Forward with the tuned scales applied at each quantized layer."""
        handles = []
        try:
            for name, st in layers.items():
                linear = linears.get(name)
                if linear is None:
                    continue
                s1 = s1_params[name]
                s2 = s2_params[name]
                ub = torch.from_numpy(np.ascontiguousarray(st.Ub.astype(np.float64)))
                vb = torch.from_numpy(np.ascontiguousarray(st.Vb.astype(np.float64)))

                def scaled_forward(x_in, _ub=ub, _vb=vb, _s1=s1, _s2=s2, _lin=linear):
                    # W_q = (s1 ⊙ Ub) @ (s2 ⊙ Vb)ᵀ, shape (dout, din) — the
                    # artifact weight convention (nn.Linear applies x @ W.T).
                    w = (_s1.unsqueeze(1) * _ub) @ (_s2.unsqueeze(1) * _vb).T
                    return x_in @ w.to(x_in.dtype).T + _lin.bias.to(x_in.dtype)

                handles.append((linear, linear.forward))
                linear.forward = scaled_forward  # type: ignore[method-assign]
            return model(x)
        finally:
            for linear, original in handles:
                linear.forward = original  # type: ignore[method-assign]

    step = 0
    for _epoch in range(cfg.stage_global.epochs):
        for lo in range(0, n, bs):
            idx = slice(lo, min(lo + bs, n))
            x_bs = inputs[idx]
            y_bs = all_logits[idx]
            if cfg.cosine_schedule and total > 1:
                import math

                lr = cfg.stage_global.lr * 0.5 * (1 + math.cos(math.pi * step / total))
                for group in opt.param_groups:
                    group["lr"] = lr
            student = scale_forward(x_bs)
            loss = _kl_with_logits(y_bs, student, cfg.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            kl_trace.append(float(loss.detach()))
            step += 1
            if step >= total:
                break
        if step >= total:
            break

    # Write the tuned scales back into the layer artifacts (float64).
    for name, st in layers.items():
        st.s1 = s1_params[name].detach().cpu().numpy().astype(np.float64).copy()
        st.s2 = s2_params[name].detach().cpu().numpy().astype(np.float64).copy()
    return ScaleCalibResult(kl_trace=kl_trace, n_steps=step)
