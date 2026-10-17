"""Phase 1 global calibration (tasks 2.1-2.3, [NQ] Step 2-1, Algorithm 1
lines 2-5).

``collect_stats`` runs calibration batches through a module once, capturing
per-linear-layer input/output activation statistics plus gradient statistics
(one backward pass), and ``robust_diag`` builds the robust diagonal
preconditioners D~_in / D~_out by clipping raw diagonal statistics to a
ceiling ``tau`` (assumption A-1) and applying Ledoit-Wolf-style shrinkage
(Eq. 3 of [NQ] Step 2-1): ``D <- (1-gamma) * clip(D, tau) + gamma *
mean(clip(D, tau))``.

All accumulators are float64 regardless of the model dtype (config.yaml
numerics policy); zero-variance statistics yield finite, positive entries.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["LayerStatsBook", "LayerStatsEntry", "collect_stats", "robust_diag"]

# Dead-channel floor: entries of a raw statistic that carry no signal are
# raised to this value (near the smallest positive normal float) so the
# preconditioners stay finite and positive for zero-variance inputs.
_TINY = float(np.finfo(np.float64).tiny) * 16.0


class LayerStatsEntry:
    """Raw per-layer statistics captured in Phase 1.

    Attributes:
        name: module name in declaration order (``named_modules`` order).
        d_in_raw: ``(din,)`` float64 diagonal input statistic (squared
            activation scale plus the mean-squared gradient statistic).
        d_out_raw: ``(dout,)`` float64 diagonal output statistic (same
            construction on the layer output).
    """

    __slots__ = ("name", "d_in_raw", "d_out_raw")

    def __init__(self, name: str, d_in_raw: np.ndarray, d_out_raw: np.ndarray) -> None:
        self.name = name
        self.d_in_raw = d_in_raw
        self.d_out_raw = d_out_raw


class LayerStatsBook:
    """Ordered collection of :class:`LayerStatsEntry`, one per linear layer."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[LayerStatsEntry] = []

    def add(self, entry: LayerStatsEntry) -> None:
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, i: int) -> LayerStatsEntry:
        return self._entries[i]

    @property
    def num_layers(self) -> int:
        return len(self._entries)

    @property
    def layer_names(self) -> list[str]:
        return [e.name for e in self._entries]


@torch.no_grad()
def _reduce_tokens(x: torch.Tensor) -> torch.Tensor:
    """Flatten activations to ``(n_tokens, features)`` float64."""
    return x.detach().to(torch.float64).reshape(-1, x.shape[-1])


def _layer_statistic(feat_sq: torch.Tensor, grad_stat: torch.Tensor) -> torch.Tensor:
    """Diagonal curvature proxy per input feature: mean(x^2) + mean-square
    gradient statistic, float64.

    Gradient statistics enter the preconditioners per design section 1
    ([NQ] Step 2-1); we mirror the output-side construction on the input
    side (informed choice for OQ-2, recorded in the change's design.md
    implementation notes).
    """
    return feat_sq.mean(dim=0) + grad_stat


def collect_stats(module: nn.Module, calibration_batches) -> LayerStatsBook:
    """Run calibration batches forward (and backward) once through ``module``.

    For every ``nn.Linear`` in module declaration order, records the mean
    square of each input/output feature plus the mean square of the
    output-gradient feature (backward pass). ``module`` may be a single
    layer or a whole model; gradients are enabled for the statistics pass
    and the module's parameter ``.grad`` fields are restored afterwards.
    """
    linears = [(name, m) for name, m in module.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("collect_stats found no nn.Linear modules to calibrate")

    grad_enabled = torch.is_grad_enabled()
    book = LayerStatsBook()
    feats_in: dict[str, list[torch.Tensor]] = {name: [] for name, _ in linears}
    feats_out: dict[str, list[torch.Tensor]] = {name: [] for name, _ in linears}
    handles = []
    for name, m in linears:
        handles.append(
            m.register_forward_hook(
                lambda mod, inp, out, key=name: feats_out[key].append(_reduce_tokens(out))
            )
        )
        handles.append(
            m.register_forward_pre_hook(
                lambda mod, inp, key=name: feats_in[key].append(_reduce_tokens(inp[0]))
            )
        )

    params = [p for _, m in linears for p in m.parameters()]
    had_grad = [bool(p.grad is not None) for p in params]
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    grad_in2: dict[str, list[torch.Tensor]] = {name: [] for name, _ in linears}
    try:
        # The backward-style pass over calibration data produces the gradient
        # statistics; a uniform output-gradient cotangent touches every
        # feature channel.
        for batch in calibration_batches:
            torch.set_grad_enabled(True)
            model_out = module(batch)
            if not isinstance(model_out, torch.Tensor):
                model_out = model_out[0]
            model_out.backward(torch.ones_like(model_out))
            torch.set_grad_enabled(False)
            for name, m in linears:
                if m.weight.grad is not None:
                    g = m.weight.grad.to(torch.float64)  # (dout, din)
                    # Mean square per input feature, per batch -> (din,).
                    grad_in2[name].append(g.pow(2).mean(dim=0))
                else:
                    din = m.weight.shape[1]
                    grad_in2[name].append(torch.zeros(din, dtype=torch.float64))
    finally:
        torch.set_grad_enabled(grad_enabled)
        for h in handles:
            h.remove()
        for p, had, saved in zip(params, had_grad, saved_grads, strict=True):
            p.grad = saved if had else None

    for name, m in linears:
        din, dout = m.weight.shape[1], m.weight.shape[0]
        feat_in = torch.cat(feats_in[name], dim=0)
        feat_out = torch.cat(feats_out[name], dim=0)
        grad_stat = torch.stack(grad_in2[name]).mean(dim=0)  # (din,)
        d_in_raw = (
            _layer_statistic(feat_in.pow(2), grad_stat).detach().cpu().numpy().astype(np.float64)
        )
        d_out_raw = feat_out.pow(2).mean(dim=0).detach().cpu().numpy().astype(np.float64)
        d_in_raw = np.maximum(d_in_raw, _TINY)
        d_out_raw = np.maximum(d_out_raw, _TINY)
        if d_in_raw.shape != (din,) or d_out_raw.shape != (dout,):
            raise ValueError(f"statistics shape mismatch for layer {name!r}")
        book.add(LayerStatsEntry(name, d_in_raw, d_out_raw))
    return book


def robust_diag(d_in_raw, d_out_raw=None, *, tau=None, gamma=0.2):
    """Robust diagonal preconditioners D~_in / D~_out (Eq. 3, [NQ] Step 2-1).

    Entries are clipped to the ceiling ``tau`` **before** shrinkage
    (assumption A-1: tau is an absolute cap; ``None`` adapts per vector to
    ``max(D)``), then shrunk toward the clipped mean with coefficient
    ``gamma``: ``D <- (1-gamma)*clip(D, tau) + gamma*mean(clip(D, tau))``.
    A tiny positive floor keeps zero-variance statistics finite and
    positive. Returns a ``(d_in, d_out)`` tuple; with ``d_out_raw=None`` the
    scalar form returns a single vector.
    """
    scalar = d_out_raw is None
    vecs = [np.asarray(d_in_raw, dtype=np.float64).copy()]
    if not scalar:
        vecs.append(np.asarray(d_out_raw, dtype=np.float64).copy())
    out = []
    for vec in vecs:
        if vec.ndim != 1:
            raise ValueError(f"raw statistics must be 1-D, got shape {vec.shape}")
        np.maximum(vec, _TINY, out=vec)
        if tau is not None:
            vec = np.minimum(vec, float(tau))
        mean = vec.mean()
        shrunk = (1.0 - gamma) * vec + gamma * mean
        # The shrinkage is a convex combination, so it cannot exceed the
        # pre-shrinkage maximum; clamp the float ulp overshoot so entries
        # provably lie in [0, tau] after (spec preconditioner_shape_robustness).
        np.minimum(shrunk, vec.max(), out=shrunk)
        np.maximum(shrunk, _TINY, out=shrunk)
        out.append(shrunk)
    if scalar:
        return out[0]
    return out[0], out[1]
