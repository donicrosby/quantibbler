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

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .block_recon import QuantizedLayerState
from .config import PTQConfig

__all__ = ["ScaleCalibResult", "calibrate_scales"]


def _kd_scratch_dir() -> str | None:
    """The nominated KD scratch directory, created if a caller named one.

    ``tempfile.TemporaryDirectory`` creates its own leaf but NOT its parents, so
    a nominated path that does not exist yet must be made first: otherwise a
    set-but-missing override — or a default naming a directory that exists on
    one machine and nowhere else, which is how the first full-python CI run went
    red — dies with a ``FileNotFoundError`` raised from inside tempfile, far from
    the configuration that caused it. ``None`` means "use the platform default".
    """
    nominated = os.getenv("QUANTIBBLER_KD_TMPDIR")
    if not nominated:
        return None
    Path(nominated).mkdir(parents=True, exist_ok=True)
    return nominated


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
    # Pre-stage snapshots (trust-region references + restore sources).
    pre_s1 = {n: t.numpy().copy() for n, t in s1_params.items()}
    pre_s2 = {n: t.numpy().copy() for n, t in s2_params.items()}
    for t in (*s1_params.values(), *s2_params.values()):
        t.requires_grad_(True)
    # Per-tensor RELATIVE lr: an
    # absolute lr 1e-3 is irrelevant to the tensors it steps — at polish
    # exit s1 can live at 1e-7 while s2 lives at 1e4, so one absolute rate
    # is 1e4x too small for one side and 1e4x too big for the other. Each
    # scale tensor now steps at lr_rel * its own median magnitude.
    groups = []
    for t in (*s1_params.values(), *s2_params.values()):
        lr_scale = max(float(t.detach().abs().median()), 1e-30)
        groups.append({"params": [t], "lr": cfg.stage_global.lr * lr_scale, "lr_scale": lr_scale})
    opt = torch.optim.Adam(groups)

    # Teacher logits, cached once ([NQ] Sec 3.3 memory note). Under a
    # container memory cap the cache is DISK-BACKED (np.memmap): a real
    # 152k-vocab model x 128 rows x 512 positions is tens of GB in RAM (the
    # papers cache on 80GB H100s; we trade a sequential write + mmap reads
    # for the RAM). The teacher's logits for one calibration batch
    # materialize only transiently, per time-chunk of the write. f64 models
    # (the tiny fixtures) cache f64 exactly as before — bit-identical to
    # the in-RAM path; lower-precision models cache float32 (already below
    # the artifacts' fp16 scale precision impact).
    import tempfile

    # Teacher logits materialize in 2-row slices: a full 8x512x152k batch
    # of bf16 logits is ~2.4GB at once; 2 rows is ~600MB. The cache write
    # below walks the same slice shape, so peak transient stays bounded.
    probe = teacher(batches[0][:2])
    logit_shape = tuple(probe.shape)
    vocab = logit_shape[-1]
    rows_per_sample = logit_shape[1] if len(logit_shape) == 3 else 1
    n_rows = sum(b.shape[0] for b in batches) * rows_per_sample
    cache_dt = np.float64 if probe.dtype == torch.float64 else np.float32
    torch_cache_dt = torch.float64 if probe.dtype == torch.float64 else torch.float32
    tmp = tempfile.TemporaryDirectory(prefix="quantibbler-kd-", dir=_kd_scratch_dir())
    teacher_mm = np.memmap(
        f"{tmp.name}/teacher_logits.bin",
        dtype=cache_dt,
        mode="w+",
        shape=(n_rows, vocab),
    )
    row = 0
    with torch.no_grad():
        for b in batches:
            for lo in range(0, b.shape[0], 2):
                t = teacher(b[lo : lo + 2]).detach().reshape(-1, vocab)
                teacher_mm[row : row + t.shape[0]] = t.to(torch_cache_dt).cpu().numpy()
                row += t.shape[0]
                del t
    teacher_mm.flush()
    del probe

    n = sum(b.shape[0] for b in batches)
    bs = max(1, min(cfg.stage_global.batch_size, n))
    total = max(cfg.stage_global.epochs * (n // bs or 1), 1)
    kl_trace: list[float] = []
    inputs = torch.cat(batches, dim=0)
    # Disk-backed cache: row-slices page in per step (bs x rows_per_sample
    # logits only), so KD-step RAM is O(batch), not O(dataset).
    all_logits = teacher_mm

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
                    # Bias-free linears (every modern LLM's attention/MLP
                    # projections) skip the bias term entirely.
                    w = (_s1.unsqueeze(1) * _ub) @ (_s2.unsqueeze(1) * _vb).T
                    out = x_in @ w.to(x_in.dtype).T
                    if _lin.bias is not None:
                        out = out + _lin.bias.to(x_in.dtype)
                    return out

                handles.append((linear, linear.forward))
                linear.forward = scaled_forward  # type: ignore[method-assign]
            return model(x)
        finally:
            for linear, original in handles:
                linear.forward = original  # type: ignore[method-assign]

    step = 0
    for _epoch in range(cfg.stage_global.epochs):
        for lo in range(0, n, bs):
            hi = min(lo + bs, n)
            x_bs = inputs[lo:hi]
            if cfg.cosine_schedule and total > 1:
                import math

                cos_f = 0.5 * (1 + math.cos(math.pi * step / total))
                for group in opt.param_groups:
                    group["lr"] = group["lr_scale"] * cfg.stage_global.lr * cos_f
            student = scale_forward(x_bs)
            # Page the step's teacher logits in from the disk cache. The
            # cache's row space is (samples x positions) FLATTENED while the
            # input index space is samples: rows_per_sample bridges them
            # (identical to the old in-RAM path when rows_per_sample == 1,
            # which is what the tiny fixtures exercise). Per-sample slices
            # also keep the KL-loss transient at bs x vocab, not n x vocab.
            row0 = lo * rows_per_sample
            row1 = hi * rows_per_sample
            y_flat = torch.from_numpy(
                np.ascontiguousarray(all_logits[row0:row1], dtype=cache_dt)
            ).to(student.dtype)
            y_bs = y_flat.view(hi - lo, rows_per_sample, -1) if rows_per_sample > 1 else y_flat
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
    tmp.cleanup()  # release the disk-backed teacher cache

    # Finiteness armor + trust region (measured): this stage had
    # NO armor while every tuning stage upstream got it, and its absolute
    # lr (1e-3, 128 steps) is orders of magnitude larger than the polished
    # scales it tunes (q_proj s1 ~ 1e-2..1e-1). One early blow-up poisons
    # every tensor Adam touches; 72/196 layers exported NaN scales, born
    # depth-ordered in exactly this stage. Two guards, mirroring
    # _tune_fp/_tune_latent_ste:
    #   (1) non-finite post-stage tensors -> restore pre-stage snapshot
    #       (poisoned scales never reach ``layers``; the KL trace shows
    #       when the loss blew up);
    #   (2) per-tensor trust region: Adam's unit-scale steps may not move
    #       any scale vector by more than 2x / below 1/2x its pre-stage
    #       values (per-entry, pre-stage |s| as reference; the reference
    #       floor avoids pinning legitimately tiny scales to zero).
    s1_fin = {n: np.isfinite(t.detach().cpu().numpy()).all() for n, t in s1_params.items()}
    s2_fin = {n: np.isfinite(t.detach().cpu().numpy()).all() for n, t in s2_params.items()}
    for name, st in layers.items():
        if not (s1_fin[name] and s2_fin[name]):
            # Poisoned: restore the EXACT pre-stage snapshot. (s1_params
            # share memory with the layer book via from_numpy, so the
            # stale st.s1 arrays are themselves NaN — rebinding is not
            # optional here.)
            st.s1 = np.array(pre_s1[name], dtype=np.float64, copy=True)
            st.s2 = np.array(pre_s2[name], dtype=np.float64, copy=True)
            continue
        pre1 = np.maximum(np.abs(pre_s1[name]), 1e-30)
        pre2 = np.maximum(np.abs(pre_s2[name]), 1e-30)
        t1 = s1_params[name].detach().cpu().numpy()
        t2 = s2_params[name].detach().cpu().numpy()
        st.s1 = np.clip(t1, 0.5 * pre1, 2.0 * pre1).astype(np.float64).copy()
        st.s2 = np.clip(t2, 0.5 * pre2, 2.0 * pre2).astype(np.float64).copy()
    n_restored = sum(
        1
        for name in layers
        if not (s1_fin[name] and s2_fin[name])
        or not np.isfinite(s1_params[name].detach().cpu().numpy()).all()
        or not np.isfinite(s2_params[name].detach().cpu().numpy()).all()
    )
    if n_restored:
        print(
            f"[calibrate_scales] restored {n_restored}/{len(layers)} layers "
            "(non-finite post-KD scales -> pre-stage snapshot)",
            flush=True,
        )
    return ScaleCalibResult(kl_trace=kl_trace, n_steps=step)
