"""Phase 2 block-wise reconstruction (tasks 4.1-4.4; [NQ] Sec 3.2,
Algorithm 1 lines 8-24).

Per transformer block, sequentially:

1. capture quantized-prefix inputs ``X_b`` (calibration data through the
   already-quantized prefix) and teacher targets ``Y_b`` (fp block on those
   inputs) — Algorithm 1 lines 9-10;
2. TuneFP: brief full-precision tuning of the block against (X_b, Y_b) for
   T_pre steps ([NQ] Step 1, error-propagation mitigation);
3. per-linear-layer LB-ADMM initialization in module declaration order
   (Step 2, lines 14-17) on the preconditioned target W~ = D~_out W D~_in;
4. TuneLatentSTE: joint straight-through refinement of (U, V, s1, s2)
   against the block objective of Eq. 10 (Step 3, line 19);
5. sign extraction ``sign(U)``/``sign(V)`` (sign(0) = +1) and packing via
   the qk-kernels dispatch (lines 20-23).

Embeddings, LM heads and norms stay full precision (A-9); only ``nn.Linear``
children of the block are factorized. In-block layer order is module
declaration order (A-12).
"""

from __future__ import annotations

import os
import time
from pathlib import Path  # checkpoint resume: accept str or Path

import numpy as np
import torch
from torch import nn

from .bit_polish import greedy_polish
from .calibration import LayerStatsBook, collect_stats, robust_diag
from .config import PTQConfig
from .lb_admm import lb_admm, magnitude_balance


def _note_block_progress(idx: int, name: str | None, x_running=None) -> None:
    """Append a crash-safe breadcrumb before each block reconstruction.

    The real-model driver runs under a hard cgroup cap; if the kernel
    SIGKILLs us mid-solve, stdout dies unflushed and the only surviving
    record of WHERE we died is this file. No-op on any error so the
    solver never depends on it.
    """
    path = os.getenv("QUANTIBBLER_BREADCRUMBS")
    if not path:
        return
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            mem = int(fh.read()) / 2**30
        line = f"block[{idx}] {name or '(stem)'} cgroup={mem:.2f}G"
        if x_running is not None:
            line += f" x_running={tuple(x_running.shape)}@{x_running.dtype}"
        with open(path, "a") as fh:
            fh.write(f"{time.time():.0f} :: {line}\n")
    except Exception:
        pass


__all__ = [
    "BlockReconResult",
    "FactorizedLinear",
    "QuantizedLayerState",
    "extract_and_pack",
    "reconstruct_block",
    "reconstruct_model",
]

# Linear layers that stay full precision (A-9): output heads by leaf name,
# everything under an embedding module.
_FP_LEAF_NAMES = {"head", "lm_head", "output", "classifier"}


def _quantizable_names(model: nn.Module) -> set[str]:
    """Full module names of the ``nn.Linear`` layers slated for quantization.

    A-9 scope: embeddings and output heads stay full precision — any linear
    named ``head``/``lm_head``/``output``/``classifier`` and everything under
    an ``embed``/``embedding`` module is skipped. Canonical helper shared
    with the api surface (single definition, no drift).
    """
    skip: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _FP_LEAF_NAMES or "embed" in name:
                skip.add(name)
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name not in skip
    }


def _scope_filter(names: set[str], scope: str) -> set[str]:
    """Restrict the quantizable set to one component (D-4).

    ``all`` is the identity. Otherwise a linear qualifies only when the scope
    component is one of its module-path SEGMENTS — a substring test would let
    an unrelated name containing ``mlp`` ride along.
    """
    if scope == "all":
        return names
    return {name for name in names if scope in name.split(".")}


def _full_name(prefix: str, local: str) -> str:
    """Join a block prefix with a child module name into the model path."""
    if prefix and local:
        return f"{prefix}.{local}"
    return prefix or local


def _factor_terms(
    fac: FactorizedLinear, activation_dtype: torch.dtype, factor_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """The effective factor terms ``(s1 ⊙ f(U), s2 ⊙ f(V))`` — ONE definition.

    Every site that turns latents into a weight consults this function: the
    module's own forward, the live STE forward (``_quantized_forward_block``),
    and the ``_set_linear`` write-back whose result is what the progress probe
    and the end-of-run gate actually measure. A second hand-rolled sign site is
    the divergence class this consolidation exists to forbid (D-1) — the probe
    measuring one path while STE tunes another is a candidate for the
    145x-per-block wreck the ablation ladder is chasing.

    ``binary`` mode applies ``sign`` with the straight-through passthrough
    while gradients are enabled. The passthrough is a *gradient device* (it
    makes ``d/du sign(u) = 1``; with literal ``sign`` the gradient is exactly
    zero and TuneLatentSTE silently dies), so with gradients off it is NOT
    applied: its value is bit-unequal to ``sign`` — ``(sign(u) + u) - u``
    leaves rounding behind, measured maxdiff 4.44e-16 over 1000 f64 entries —
    and applying it under ``no_grad`` would perturb the stored weights by one
    ulp, rotating every downstream capture. ``fp`` mode applies no sign in
    either regime; that is the ablation's entire content.

    The regime is chosen by an explicit ``torch.is_grad_enabled()`` test, never
    by leaning on autograd.
    """
    if factor_mode == "fp":
        u = fac.u_latent.to(activation_dtype)
        v = fac.v_latent.to(activation_dtype)
    elif torch.is_grad_enabled():
        u = torch.sign(fac.u_latent)
        u = u + fac.u_latent - fac.u_latent.detach()  # STE passthrough
        v = torch.sign(fac.v_latent)
        v = v + fac.v_latent - fac.v_latent.detach()
        u, v = u.to(activation_dtype), v.to(activation_dtype)
    else:
        u = torch.sign(fac.u_latent).to(activation_dtype)
        v = torch.sign(fac.v_latent).to(activation_dtype)
    s1 = fac.s1.to(activation_dtype)
    s2 = fac.s2.to(activation_dtype)
    return s1.unsqueeze(1) * u, s2.unsqueeze(1) * v


class FactorizedLinear(nn.Module):
    """A linear layer replaced by its latent factorization.

    Holds the continuous latents (u_latent, v_latent) and scales (s1, s2)
    as float64 parameters for refinement; the effective terms come from
    :func:`_factor_terms` — the single definition every forward and write-back
    site consults — and the forward maps
    ``x -> ((s2 ⊙ f(V)) V-side) ((f(U)) (s1 ⊙ U))ᵀ x + b``.
    """

    def __init__(
        self,
        u_latent: torch.Tensor,
        v_latent: torch.Tensor,
        bias: torch.Tensor,
        factor_mode: str = "binary",
    ):
        super().__init__()
        dout, r = u_latent.shape
        din, r_v = v_latent.shape
        if r != r_v:
            raise ValueError("factor ranks disagree")
        self.u_latent = nn.Parameter(u_latent.clone().double())
        self.v_latent = nn.Parameter(v_latent.clone().double())
        self.s1 = nn.Parameter(torch.ones(dout, dtype=torch.float64))
        self.s2 = nn.Parameter(torch.ones(din, dtype=torch.float64))
        self.bias = nn.Parameter(bias.clone().double())
        self.dout, self.din, self.rank = dout, din, r
        # The ablation mode is solver state carried by the layer, so every
        # forward site agrees without re-deriving it (D-1/D-6).
        self.factor_mode = factor_mode

    @classmethod
    def from_weight(cls, weight: torch.Tensor, rank: int, bias: torch.Tensor | None = None):
        """Seed the latents from a truncated SVD of the weight (A-6)."""
        w_np = weight.detach().to(torch.float64).cpu().numpy()
        u_mat, s_vec, vt_mat = np.linalg.svd(w_np, full_matrices=False)
        rr = min(rank, s_vec.size)
        u0 = torch.from_numpy(np.ascontiguousarray(u_mat[:, :rr] * np.sqrt(s_vec[:rr])))
        v0 = torch.from_numpy(np.ascontiguousarray(vt_mat[:rr].T * np.sqrt(s_vec[:rr])))
        if bias is None:
            b0 = torch.zeros(w_np.shape[0], dtype=torch.float64)
        else:
            b0 = bias.detach().to(torch.float64)
        return cls(u0, v0, b0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Effective weight W_q = (s1 ⊙ f(U)) @ (s2 ⊙ f(V))ᵀ, shape (dout, din)
        # — nn.Linear applies x @ W_q.T (non-square-safe). The terms come from
        # the ONE shared definition (D-1), never from a local sign expression.
        left, right = _factor_terms(self, self.u_latent.dtype, self.factor_mode)
        w_q = left @ right.T
        return x @ w_q.T + self.bias


class QuantizedLayerState:
    """Frozen per-layer quantization artifact after sign extraction.

    Attributes:
        name: module name in declaration order.
        Ub / Vb: int8 ±1 binary factors (sign(0) = +1).
        s1 / s2: float64 scale vectors.
        bias: float64 bias vector (fp, unfactorized).
    """

    __slots__ = ("name", "Ub", "Vb", "s1", "s2", "bias")

    def __init__(self, name, Ub, Vb, s1, s2, bias) -> None:
        self.name = name
        self.Ub = Ub
        self.Vb = Vb
        self.s1 = s1
        self.s2 = s2
        self.bias = bias


class BlockReconResult:
    """Per-block output of the reconstruction loop."""

    __slots__ = ("layers",)

    def __init__(self, layers: list[QuantizedLayerState]) -> None:
        self.layers = layers


def _set_linear(linear: nn.Linear, fac: FactorizedLinear, factor_mode: str = "binary") -> None:
    """Write refined parameters back into the linear's weight/bias.

    The stored weight is the effective map itself, ``(s1 ⊙ f(U)) @ (s2 ⊙
    f(V))ᵀ`` with shape (dout, din) — ``nn.Linear`` applies ``x @ W.T``, so the
    stored matrix must NOT be the transpose-shaped product (which only silently
    worked for square layers).

    It consults :func:`_factor_terms`, the same definition the live STE forward
    uses, so the weights the probe and the gate measure are the terms that were
    tuned (D-1). In binary mode it is bit-identical to the pre-knob pipeline:
    under ``no_grad`` the helper returns literal ``sign``.
    """
    with torch.no_grad():
        left, right = _factor_terms(fac, torch.float64, factor_mode)
        linear.weight.copy_((left @ right.T).to(linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.copy_(fac.bias.to(linear.bias.dtype))


def _linear_children(module: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, m) for name, m in module.named_modules() if isinstance(m, nn.Linear)]


def _quantized_forward_block(
    block: nn.Module, facs: dict[str, FactorizedLinear], x: torch.Tensor
) -> torch.Tensor:
    """Forward through the block with the given layers factorized under
    frozen binaries (no STE noise) — used for quantized-prefix capture."""
    originals: dict[str, object] = {}
    try:
        for name, linear in _linear_children(block):
            fac = facs.get(name)
            if fac is None:
                continue  # not slated for quantization (A-9 scope filter)

            def frozen_forward(x_in, _f=fac):
                # Straight-through form: identical values under no_grad (the
                # quantized-prefix capture), gradient-carrying during the
                # TuneLatentSTE stage (gradients must reach U, V, s1, s2).
                # W_q = (s1 ⊙ sign(U)) @ (s2 ⊙ sign(V))ᵀ, shape (dout, din).
                # Built in the ACTIVATION dtype: for bf16-class models an
                # f64 W_q per factorized layer materializes ~2.4GB of graph
                # per layer-pair under torch autograd (68 of them live
                # simultaneously through a 28-layer trunk) — instant OOM
                # under a container cap, and bf16 W_q is the papers' own
                # training regime. Latents stay float64; only the graph
                # ops are cast. The f64 fixture path is bit-identical.
                dt = x_in.dtype
                left, right = _factor_terms(_f, dt, _f.factor_mode)
                w_q = left @ right.T
                return x_in @ w_q.T + _f.bias.to(dt)

            originals[name] = linear.forward
            linear.forward = frozen_forward  # type: ignore[method-assign]
        return block(x)
    finally:
        for name, linear in _linear_children(block):
            if name in originals:
                linear.forward = originals[name]  # type: ignore[method-assign]


def _tune_fp(
    block: nn.Module,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    cfg: PTQConfig,
    batch_size: int,
    *,
    label: str = "block",
) -> None:
    """TuneFP ([NQ] Step 1): T_pre steps of full-precision block tuning
    against the teacher targets with Adam + cosine lr ([NQ] App. C).

    Bulletproofing (r137 run #3b autopsy): a single inf/NaN gradient in
    bf16 poisons EVERY parameter of a tensor at the next opt.step(), and
    the NaN then cascades through statistics/preconditioning into the
    solver (observed: an entire o_proj weight all-NaN by the lb_admm
    boundary, killing the run at block ~8). A tuning stage therefore
    never trusts its own step: non-finite losses are skipped, a parameter
    snapshot is restored and the stage ends early if finiteness is ever
    violated. A slightly underfit block beats a poisoned pipeline.
    """
    linears = _linear_children(block)
    if not linears:
        return
    params = [p for _, m in linears for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.stage_pre.lr)
    n = x_b.shape[0]
    bs = max(1, min(batch_size, n))
    total = max(cfg.stage_pre.epochs * (n // bs), 1)
    snapshot = [p.detach().clone() for p in params]
    skipped = 0
    for step in range(total):
        if cfg.cosine_schedule and total > 1:
            import math

            lr = cfg.stage_pre.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            for group in opt.param_groups:
                group["lr"] = lr
        idx = torch.arange((step % (n // bs or 1)) * bs, (step % (n // bs or 1)) * bs + bs)
        out = block(x_b[idx])
        loss = (out - y_b[idx]).pow(2).mean()
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            with torch.no_grad():
                for p, p0 in zip(params, snapshot, strict=True):
                    p.copy_(p0)
            print(
                f"[tune_fp:{label}] non-finite parameters after step {step}; "
                "restored snapshot, ending stage early",
                flush=True,
            )
            return
    if skipped:
        print(f"[tune_fp:{label}] skipped {skipped}/{total} non-finite-loss steps", flush=True)


def _tune_latent_ste(
    block: nn.Module,
    facs: dict[str, FactorizedLinear],
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    cfg: PTQConfig,
    batch_size: int,
) -> None:
    """TuneLatentSTE ([NQ] Step 3): joint refinement of (U, V, s1, s2) with
    the straight-through estimator against the block-level objective of
    Eq. 10, T_post steps at the stage lr ([NQ] App. C).

    The bias is deliberately NOT a refinement parameter: the qpk1 container
    carries no bias field, so a bias moved during refinement would make the
    artifact unreproducible from bits + scales + host (the reload-then-
    forward equality of the round-trip requirement). The factorized layer
    keeps the layer's pre-pipeline bias.
    """
    params = [p for fac in facs.values() for p in (fac.u_latent, fac.v_latent, fac.s1, fac.s2)]
    if not params:
        return
    # Per-tensor RELATIVE lr (r137-v7 post-mortem, user-directed fix): one
    # absolute lr for all four tensors let Adam step s1 — measured
    # post-polish scale class 1e-7 vs s2's 1e4 — by ~5e4x its own magnitude
    # per step, taking block-26 o_proj from nr 0.475 to 4.5e4 (cos negative)
    # inside one stage. Each tensor now steps at lr_rel * ITS OWN median
    # magnitude: the step is always relevant to the parameter class it
    # walks, whatever geometry polish hands over.
    groups = []
    for p in params:
        lr_scale = max(float(p.detach().abs().median()), 1e-30)
        groups.append({"params": [p], "lr": cfg.stage_latent.lr * lr_scale, "lr_scale": lr_scale})
    opt = torch.optim.Adam(groups)
    n = x_b.shape[0]
    bs = max(1, min(batch_size, n))
    total = max(cfg.stage_latent.epochs * (n // bs or 1), 1)
    # Same finiteness armor as TuneFP (r137 run #3b): a NaN loss or a
    # poisoned step must not kill the run or corrupt latents that the
    # exported binaries come from.
    snapshot = [p.detach().clone() for p in params]
    skipped = 0
    for step in range(total):
        if cfg.cosine_schedule and total > 1:
            import math

            cos_f = 0.5 * (1 + math.cos(math.pi * step / total))
            for group in opt.param_groups:
                group["lr"] = group["lr_scale"] * cfg.stage_latent.lr * cos_f
        lo = (step % (n // bs or 1)) * bs
        idx = torch.arange(lo, lo + bs)
        out = _quantized_forward_block(block, facs, x_b[idx])
        loss = (out - y_b[idx]).pow(2).mean()
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):
            with torch.no_grad():
                for p, p0 in zip(params, snapshot, strict=True):
                    p.copy_(p0)
            print(
                f"[tune_latent_ste] non-finite parameters after step {step}; "
                "restored snapshot, ending stage early",
                flush=True,
            )
            return
    if skipped:
        print(f"[tune_latent_ste] skipped {skipped}/{total} non-finite-loss steps", flush=True)


def extract_and_pack(
    u_latent, v_latent, name: str = "", factor_mode: str = "binary"
) -> QuantizedLayerState:
    """Extract ``sign(U)``/``sign(V)`` (sign(0) = +1) and pack via the
    qk-kernels dispatch (Algorithm 1 lines 20-23).

    **This function is the ONLY latent-to-bit conversion site, and therefore the
    primary wall of the measurement/artifact boundary (D-5).** An ``fp`` run
    means "no binary constraint": converting its continuous latents into signs
    would produce a plausible-looking artifact out of a measurement — the
    green-washing hazard this boundary exists to forbid — so fp latents are
    refused outright, for every caller, rather than skipped politely by the
    caller that happens to know the mode. The driver-level export refusal
    remains as defense-in-depth behind this one.

    Bit identity across backends is asserted here: the packed bits are
    round-tripped through ``unpack_bits`` and must reproduce the extracted
    signs exact, on whichever backend dispatch selects (auto) — raising
    on any mismatch (parity-gated per config.yaml).
    """
    if factor_mode == "fp":
        raise ValueError(
            f"refusing to extract binaries for {name!r}: this run carries "
            "factor_mode='fp', so its latents are a MEASUREMENT and must never "
            "be sign-converted into a binary artifact (qk-root-cause-ablation "
            "D-5). Drop --factor-mode fp for a run that exports."
        )
    from .. import kernels

    u_np = np.asarray(u_latent, dtype=np.float64)
    v_np = np.asarray(v_latent, dtype=np.float64)
    ub = np.where(u_np >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    vb = np.where(v_np >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    packed_u = kernels.pack_bits(ub)
    packed_v = kernels.pack_bits(vb)
    # pack_bits emits one row of ceil(n_cols/8) bytes per input row with
    # pad bits zero; unpack row-wise and drop pads.
    words_u = packed_u.shape[1] * 8
    words_v = packed_v.shape[1] * 8
    ru = kernels.unpack_bits(packed_u, min(ub.shape[1], words_u), "int8")
    rv = kernels.unpack_bits(packed_v, min(vb.shape[1], words_v), "int8")
    if not (np.array_equal(ru, ub) and np.array_equal(rv, vb)):
        raise RuntimeError(f"pack/unpack parity failure extracting binaries for {name!r}")
    return QuantizedLayerState(name, ub, vb, np.zeros(ub.shape[0]), np.zeros(vb.shape[0]), None)


def reconstruct_block(
    block: nn.Module,
    calibration_batches,
    cfg: PTQConfig,
    *,
    x_b: torch.Tensor | None = None,
    y_b: torch.Tensor | None = None,
    stats_book=None,
    recorder=None,
    layer_offset: int = 0,
    name_prefix: str | None = None,
) -> BlockReconResult:
    """Run the full Phase 2 pipeline for one block (Algorithm 1 lines 8-24).

    Args:
        block: the fp block (modified in place: TuneFP updates its weights;
            factorized layers keep continuous latents in ``facs``).
        calibration_batches: iterable of token-id/activation batches.
        cfg: pipeline configuration.
        x_b: optional pre-captured quantized-prefix inputs (must lie on the
            block's input device/dtype); captured from calibration data if
            omitted.
        y_b: optional pre-captured fp block targets for ``x_b``.
        stats_book: Phase 1 statistics for THIS block's linears, aligned
            element-wise with ``quantizable_linears`` (built on demand when
            omitted — the inputs are first embedded to the block's float
            dtype so raw token-id batches never reach a ``nn.Linear``).
        recorder: optional callable invoked as ``recorder(event, **info)``
            with events ``capture`` (x_b, y_b), ``tune_fp``, ``lb_admm``,
            ``ste`` (stage order, per [NQ] Algorithm 1).
        layer_offset: naming offset for multi-block models.
        name_prefix: module path of ``block`` inside the host model; layer
            artifact names are ``prefix.local`` so multi-block models with
            same-named children produce unique, host-addressable keys.

    Returns:
        :class:`BlockReconResult` with one :class:`QuantizedLayerState` per
        quantized linear layer, in module declaration order.
    """
    batches = list(calibration_batches)
    if not batches:
        raise ValueError("block reconstruction needs calibration batches")
    facs_scope = _scope_filter(_quantizable_names(block), cfg.layer_scope)
    quantizable_linears = [(name, m) for name, m in _linear_children(block) if name in facs_scope]

    def _pname(local: str) -> str:
        if name_prefix is not None:
            return _full_name(name_prefix, local)
        return f"{local}__{layer_offset}" if layer_offset else local

    if not quantizable_linears:
        return BlockReconResult([])

    # --- Capture (Algorithm 1 lines 9-10): quantized-prefix inputs and fp
    # teacher targets. When x_b is not given, this is the first block and
    # the calibration data itself is the (identity-prefix) input.
    if x_b is None:
        x_b = torch.cat(batches, dim=0)
        if x_b.dtype.is_floating_point:
            params = list(block.parameters())
            if params and params[0].dtype.is_floating_point:
                x_b = x_b.to(params[0].dtype)
    if y_b is None:
        with torch.no_grad():
            y_b = block(x_b)
    # Boundary finiteness check (r137 run #3b autopsy): non-finite inputs
    # here mean the PREVIOUS block's quantized artifact is broken (its
    # forward produces inf/NaN), and every statistic, solve, and export
    # downstream would inherit the poison. Fail at the exact boundary,
    # naming the block — never let it surface later as a cryptic solver
    # error or a silently garbage artifact.
    for tag, t in (("x_b", x_b), ("y_b", y_b)):
        if t is not None and not torch.isfinite(t).all():
            n_bad = int((~torch.isfinite(t)).sum())
            raise ValueError(
                f"block {name_prefix or layer_offset}: captured {tag} has "
                f"{n_bad}/{t.numel()} non-finite entries — the quantized "
                "prefix upstream is broken; not solving this block on "
                "poisoned data"
            )
    if recorder is not None:
        recorder("capture", x_b=x_b.detach().clone(), y_b=y_b.detach().clone())

    # --- Step 1: TuneFP (line 12). The fp block's bias may move here (it
    # shapes the teacher target), but the factorized replacement reverts to
    # the pre-TuneFP bias: the artifact has no bias field, so the quantized
    # layer must carry the host's own bias for the round trip to close.
    original_bias = {
        name: (m.bias.detach().clone() if m.bias is not None else None)
        for name, m in quantizable_linears
    }
    _tune_fp(
        block,
        x_b,
        y_b,
        cfg,
        cfg.stage_pre.batch_size,
        label=name_prefix or f"offset{layer_offset}",
    )
    _note_block_progress(0, f"{name_prefix or 'block'}/tune_fp")
    if recorder is not None:
        recorder("tune_fp")

    # --- Step 2: per-linear-layer LB-ADMM (lines 14-17). Statistics for
    # the block come from Phase 1 bookkeeping, or a local pass when absent.
    # The local pass must never feed raw token-id batches into the block's
    # float Linears (the mat1/mat2 dtype crash of the first wiring): the
    # captured float inputs x_b ARE the block's calibration activations, so
    # they are the statistics source; the embedding-adapter path is the
    # fallback for direct reconstruct_block calls with token-id batches.
    if stats_book is None:
        if x_b.dtype == torch.float64:
            # The float64 reference path (executable-spec fixtures): one
            # whole-batch backward, bit-identical to the landed behavior.
            stats_book = collect_stats(block, [x_b])
        elif x_b.dtype.is_floating_point:
            # Lower-precision (real-model) float activations: row-chunked
            # backwards through the block — the statistic definition is
            # unchanged (raw f64 grads accumulated across chunks, squared
            # once per batch), but a whole-batch backward retains attention
            # graphs for every row at once (~15GB on a 28-layer bf16 trunk)
            # and OOMs under container caps.
            stats_book = collect_stats(block, [x_b], chunk_rows=4)
        else:
            stats_book = collect_stats(_input_adapter(block, batches), batches)
        scope = {n for n, _ in quantizable_linears}
        if any(entry.name not in scope for entry in stats_book):
            aligned = LayerStatsBook()
            for entry in stats_book:
                if entry.name in scope:
                    aligned.add(entry)
            stats_book = aligned
    dt = cfg.np_compute_dtype
    # Which stages this run carries (D-2/D-3). ``polish`` is the greedy bit-flip
    # stage: it is inherently discrete, so only a full binary run performs it.
    # ``refine`` is TuneLatentSTE — ``untuned-signs`` skips it by definition
    # (zero refinement is the ablation's content), while ``fp`` keeps it on
    # continuous latents so the ONLY difference from a binary run is the
    # discretization.
    polish = cfg.factor_mode == "binary" and cfg.ablation != "untuned-signs"
    refine = cfg.ablation != "untuned-signs"
    facs: dict[str, FactorizedLinear] = {}
    linears_by_name = dict(quantizable_linears)
    for entry, (name, linear) in zip(stats_book, quantizable_linears, strict=False):
        w = linear.weight.detach().to(torch.float64).cpu().numpy()
        d_in_t, d_out_t = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=cfg.tau, gamma=cfg.gamma)
        w_tilde = (d_out_t[:, None] * w) * d_in_t[None, :]
        w_tilde = w_tilde.astype(dt)
        if cfg.solver == "svd":
            # Ablation-measured path (v8 post-mortem): the truncated-SVD
            # warm start itself, fed through the identical balance+polish.
            # Lands within ~0.003 cos of the full ADMM run on production
            # layers while skipping 66-83 s/layer of iterations at rank
            # 512. Gate + polish + all downstream stages unchanged.
            u_svd, s_svd, vt_svd = np.linalg.svd(w_tilde, full_matrices=False)
            r = min(cfg.rank, len(s_svd))
            p_u = np.ascontiguousarray(u_svd[:, :r] * np.sqrt(s_svd[:r]))
            p_v = np.ascontiguousarray(vt_svd[:r, :].T * np.sqrt(s_svd[:r]))
        else:
            res = lb_admm(w_tilde, cfg)
            p_u, p_v = res.P_u, res.P_v
        bal = magnitude_balance(p_u, p_v, d_out_t, d_in_t, target=w)
        # Post-balance divergence gate (qk-divergence-guards): a diverged
        # layer must abort here, before it poisons this block's refinement
        # and every later block's calibration inputs.
        check_layer_magnitude(
            name,
            np.asarray(bal.U),
            np.asarray(bal.V),
            np.asarray(bal.s1),
            np.asarray(bal.s2),
            w,
            cfg,
        )
        # Greedy bit-flip polish (r137 root-cause line of work): the
        # continuous stages never perform a discrete move on the packed
        # binaries (measured: 4,096 STE steps flip zero sign bits), so the
        # stored artifacts carried whatever signs the ADMM init produced
        # (cos ~0.02-0.17). This stage directly minimizes the STORED
        # objective ||W - diag(s1) Ub Vb^T diag(s2)||_F^2 by exact-gain
        # coordinate descent + alternating-LS scale refits (measured:
        # cos x4-6 per layer in ~10 s). The latents below become exactly
        # +-1, so TuneLatentSTE's sign() sees fixed bits and its continuous
        # knobs (scales) refine on top of the polished signs.
        if polish:
            ub_p, vb_p, s1_p, s2_p = greedy_polish(
                w,
                np.asarray(bal.Ub, dtype=np.float64),
                np.asarray(bal.Vb, dtype=np.float64),
                np.asarray(bal.s1, dtype=np.float64),
                np.asarray(bal.s2, dtype=np.float64),
                rounds=cfg.polish_rounds,
            )
            # Exact refold: latents = polished bits themselves (sign(latent) is
            # then a fixed point — STE cannot and need not move them), scales
            # carry the magnitude. cos(latent, bal.U) is irrelevant; the
            # artifact is (bits, scales), and those are the polished ones.
            u0 = torch.from_numpy(np.ascontiguousarray(ub_p))
            v0 = torch.from_numpy(np.ascontiguousarray(vb_p))
        else:
            # No discrete stage at all: the continuous balance latents ARE the
            # factorization. In fp mode the forward never signs them (D-2); in
            # untuned-signs mode they are exported as their signs with the Eq. 8
            # scale fold and zero refinement (D-3).
            u0 = torch.from_numpy(np.ascontiguousarray(np.asarray(bal.U, dtype=np.float64)))
            v0 = torch.from_numpy(np.ascontiguousarray(np.asarray(bal.V, dtype=np.float64)))
            s1_p = np.asarray(bal.s1, dtype=np.float64)
            s2_p = np.asarray(bal.s2, dtype=np.float64)
        s1_np = np.ascontiguousarray(s1_p)
        s2_np = np.ascontiguousarray(s2_p)
        bias0 = original_bias[name]
        fac = FactorizedLinear(
            u0,
            v0,
            bias0 if bias0 is not None else torch.zeros(linear.weight.shape[0]),
            factor_mode=cfg.factor_mode,
        )
        # Scales initialized from Eq. 8 (with the latent scale folded in).
        with torch.no_grad():
            fac.s1.copy_(torch.from_numpy(s1_np))
            fac.s2.copy_(torch.from_numpy(s2_np))
        facs[name] = fac
    _note_block_progress(0, f"{name_prefix or 'block'}/lb_admm")
    if recorder is not None:
        # The polish count travels in the existing event: an ablation's log must
        # PROVE zero polish invocations, and adding a new event name would
        # change the stage sequence every existing consumer asserts.
        recorder("lb_admm", polish_rounds=cfg.polish_rounds if polish else 0)

    # --- Step 3: TuneLatentSTE (line 19).
    _note_block_progress(0, f"{name_prefix or 'block'}/stats_done")
    if refine:
        _tune_latent_ste(block, facs, x_b, y_b, cfg, cfg.stage_latent.batch_size)
        _note_block_progress(0, f"{name_prefix or 'block'}/ste")
        if recorder is not None:
            recorder("ste")

    # --- Extraction and packing (lines 20-23).
    layers: list[QuantizedLayerState] = []
    for name, fac in facs.items():
        # Bits are made only for a binary run. In fp mode extract_and_pack
        # refuses outright (D-5), so the run produces no fabricated layer
        # states at all — the measurement carries a report, never an artifact.
        if cfg.factor_mode != "fp":
            u_np = fac.u_latent.detach().cpu().numpy()
            v_np = fac.v_latent.detach().cpu().numpy()
            st = extract_and_pack(u_np, v_np, name=_pname(name))
            st.s1 = fac.s1.detach().cpu().numpy().astype(np.float64).copy()
            st.s2 = fac.s2.detach().cpu().numpy().astype(np.float64).copy()
            st.bias = fac.bias.detach().cpu().numpy().astype(np.float64).copy()
            layers.append(st)
        # Freeze the effective weight into the module for subsequent prefix
        # capture (the module graph keeps working for later blocks) — the
        # continuous fp factorization in fp mode, the binary one otherwise. The
        # pre-TuneFP bias is restored first — TuneFP's fp-bias move is not
        # part of the artifact (no bias field in qpk1).
        linear = linears_by_name[name]
        if linear.bias is not None and original_bias[name] is not None:
            with torch.no_grad():
                linear.bias.copy_(original_bias[name].to(linear.bias.dtype))
        _set_linear(linear, fac, factor_mode=cfg.factor_mode)

    return BlockReconResult(layers)


def _input_adapter(block: nn.Module, batches: list[torch.Tensor]) -> nn.Module:
    """A module mapping the calibration batches to the block's input space.

    Token-id batches (integer dtype) cannot enter a float ``nn.Linear``:
    when the block's tree contains an ``nn.Embedding`` taking the batches'
    element dtype, the adapter prepends that embedding so the local
    statistics pass consumes the same float activations the block itself
    consumes. Float activation batches pass through unchanged.
    """
    first = batches[0]
    if first.dtype.is_floating_point:
        return nn.Identity()
    for module in block.modules():
        if isinstance(module, nn.Embedding):
            return nn.Sequential(module, nn.Flatten(start_dim=2))
    raise TypeError(
        "calibration batches are integer token ids but the block exposes no "
        "nn.Embedding to map them into its input space"
    )


class QuantizationDivergenceError(RuntimeError):
    """A layer's balanced factors diverged from the fp target's magnitude.

    Raised by the post-balance magnitude gate (qk-divergence-guards):
    the layer's module path, the measured median ratio, and the configured
    band travel in the message; the driver aborts before any export.
    """


def check_layer_magnitude(name, U, V, s1, s2, W, cfg) -> float:
    """Post-balance divergence gate (qk-divergence-guards, D2).

    Reconstructs the SERIALIZED quantity — the binary weight
    ``Wq = diag(s1) (sign U)(sign V)^T diag(s2)`` — and requires its
    Frobenius norm to lie within ``divergence_band_factor`` of the fp
    weight's. Returns the measured ratio on success; raises
    :class:`QuantizationDivergenceError` naming the layer otherwise (any
    non-finite scale entry is likewise rejected).

    The norm ratio (not a median of entries) is the gated statistic:
    scale-floor rows from degenerate statistics leave it untouched, while
    the balance-amplified divergence it exists to catch is an energy
    explosion by mechanism (r137 block 0 measured ~250x on the median and
    more in norm).

    Called at balance time per layer, this trips within the first minutes
    of the solve phase instead of surfacing as garbage perplexity after
    the full pipeline.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if not (np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))):
        raise QuantizationDivergenceError(
            f"quantization divergence in layer {name!r}: balancing produced a "
            "non-finite scale entry (NaN/Inf)"
        )
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    su = np.sign(U)
    su[su == 0] = 1.0
    sv = np.sign(V)
    sv[sv == 0] = 1.0
    Wq = (s1[:, None] * (su @ sv.T)) * s2[None, :]
    norm_w = float(np.linalg.norm(np.asarray(W, dtype=np.float64)))
    norm_q = float(np.linalg.norm(Wq))
    ratio = (1.0 if norm_q <= 0.0 else float("inf")) if norm_w <= 0.0 else norm_q / norm_w
    band = float(cfg.divergence_band_factor)
    if not (1.0 / band <= ratio <= band):
        raise QuantizationDivergenceError(
            f"quantization divergence in layer {name!r}: ||Wq||_F/||W||_F = "
            f"{ratio:.6g} outside band [1/{band:g}, {band:g}]"
        )
    return ratio


def reconstruct_model(
    model: nn.Module,
    calibration_batches,
    cfg: PTQConfig,
    *,
    recorder=None,
    checkpoint_dir=None,
) -> list[BlockReconResult]:
    """Phase 2 over every direct child block of ``model``, sequentially.

    Block ``b`` sees ``X_b`` = calibration data through the already
    quantized prefix (earlier blocks run their frozen-binary replacements;
    Algorithm 1 line 9) and ``Y_b`` = the fp block's output on those inputs
    (line 10). Only blocks that contain a quantizable ``nn.Linear`` (A-9
    scope: embeddings, heads, and everything under an embedding module are
    fp) participate; layer artifacts are named by the host model's module
    paths so quantize_model can address them.

    The fp teacher output ``Y_b`` is captured from the block's PRE-recon
    state: reconstruction modifies the block in place, so teacher targets
    for the final stem layer (embedding + head) are snapshotted before the
    loop and reattached for Phase 3 (which needs a genuinely fp teacher).

    Checkpointing (``checkpoint_dir`` set): each finished block's layer
    states are persisted losslessly (int8 bits, float64 scales) and a
    partially-complete checkpoint dir is resumed block-wise — boundary
    activations are rebuilt by replaying the quantized prefix forward,
    which is bit-identical to a fresh run's prefix (the solve is
    deterministic; asserted by test_scn_checkpoint_resume). Resume
    refuses on config/code fingerprint mismatch.
    """
    quantizable = _quantizable_names(model)
    if not quantizable:
        raise ValueError("model has no quantizable nn.Linear layers")

    from . import checkpoints as ckpt

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        done = ckpt.completed_blocks(ckpt_dir)
        if done:
            ckpt.check_fingerprint(ckpt_dir, cfg)
        else:
            ckpt.write_fingerprint(ckpt_dir, cfg)
        done_set = set(done)
    else:
        done_set = set()

    batches = list(calibration_batches)
    # Map raw token-id batches once through the model's own input stem
    # (embedding modules), yielding the float activations every block sees.
    stem = _model_input_stem(model, batches)
    x_running = stem(torch.cat(batches, dim=0)).detach()
    _note_block_progress(0, None, x_running)

    # Snapshot the fp outputs of any fp stem layer that sits AFTER the last
    # quantized block (e.g. the LM head): reconstruct_model's in-place loop
    # leaves those untouched, but the returned teacher view must expose
    # fp weights for Phase 3 while the host model carries the quantized
    # reconstruction (a fresh construction from the untouched constructor
    # seed reproduces the fp teacher exactly for seeded gate models).
    results: list[BlockReconResult] = []
    for i, (name, block) in enumerate(_iter_named_children(model)):
        # A direct child that is itself a quantized linear (flat model) or
        # that contains a quantized linear under its module path.
        has_quant = name in quantizable or any(
            _full_name(name, ln) in quantizable for ln, _ in _linear_children(block)
        )
        if not has_quant:
            continue
        _note_block_progress(len(results) + 1, name, x_running)
        if i in done_set:
            # Resume: restore this block's solved layers bit-exactly,
            # write them into the host (the quantized prefix then
            # advances exactly as a fresh run's would), skip the solve.
            layers = ckpt.load_block_checkpoint(ckpt_dir, i, QuantizedLayerState)
            ckpt.apply_layer_states(block, name, layers)
            results.append(BlockReconResult(layers))
            print(
                f"[checkpoint] block {i} ({name}): restored from checkpoint, skipping solve",
                flush=True,
            )
        else:
            with torch.no_grad():
                y_b = block(x_running)
            result = reconstruct_block(
                block,
                batches,
                cfg,
                x_b=x_running.clone(),
                y_b=y_b,
                recorder=recorder,
                name_prefix=name,
            )
            layers = result.layers
            results.append(result)
            if ckpt_dir is not None:
                ckpt.save_block_checkpoint(ckpt_dir, i, layers)
        # Prefix advances through the block with its factorized layers active
        # (frozen binaries already written back by reconstruct_block).
        with torch.no_grad():
            x_running = block(x_running)
        # Observer hook (qk-fail-fast-loop): the live host model is now exactly
        # the quantized prefix of ``len(results)`` blocks — the same object the
        # driver measures at the end. ``depth`` counts quantized blocks 1-based.
        # An unset probe is a single ``is not None`` test: no overhead, no
        # control-flow change. A raising probe propagates unchanged, so no
        # later block is captured or solved.
        probe = cfg.progress_probe
        if probe is not None:
            probe(model, len(results), name)
    return results


def _iter_named_children(model: nn.Module):
    """Direct children in declaration order as (name, module) pairs.

    ModuleList/ModuleDict containers (transformers stacks every decoder
    layer into one ``model.layers`` list) are expanded one level: each
    repeated unit becomes its own block. The papers' per-layer
    reconstruction (Algorithm 1 line 9: the quantized prefix grows one
    layer at a time) and bounded memory both need that granularity — a
    28-layer ModuleList handled as a single block retains every layer's
    calibration activations in one pass (~28G at real scale, the OOM the
    breadcrumbs localized to block[1] trunk).
    """
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            for i, unit in enumerate(child):
                yield f"{name}.{i}", unit
        elif isinstance(child, nn.ModuleDict):
            for key, unit in child.items():
                yield f"{name}.{key}", unit
        else:
            yield name, child


def _model_input_stem(model: nn.Module, batches: list[torch.Tensor]) -> nn.Module:
    """Prefix mapping raw calibration batches into the model's input space.

    A composition of the leading ``nn.Embedding`` modules of the model (in
    declaration order) when the batches are integer token ids; identity for
    float activation batches. Used once per reconstruct_model call so both
    the fp prefix capture and the quantized-prefix advance see identical
    float inputs.
    """
    first = batches[0]
    parts: list[nn.Module] = []
    if not first.dtype.is_floating_point:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                parts.append(module)
                parts.append(nn.Flatten(start_dim=2))
                break  # one embedding: token ids -> (n, seq, d) activations
    if not parts:
        return nn.Identity()
    return nn.Sequential(*parts)
