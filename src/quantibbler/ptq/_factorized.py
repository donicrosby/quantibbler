"""The factorized-layer representation: continuous latents, the ONE
effective-term definition, the write-back, packed sign extraction, and the
divergence guard.

Split out of ``block_recon``; same code, same call graph.

``_factor_terms`` is the single definition of the effective factor terms (D-1)
and all three expression sites live here -- ``FactorizedLinear.forward``, the
live STE forward ``_quantized_forward_block`` (``frozen_forward``) and
``_set_linear`` -- because the D-1 shape pin in
``tests/ptq/test_root_cause_ablation.py`` parses THIS module's source. The three
sites must not be separated again.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .solver_backend import to_host_np


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
    u = u.to(s1.device)
    v = v.to(s2.device)
    return s1.unsqueeze(1) * u, s2.unsqueeze(1) * v


class FactorizedLinear(nn.Module):
    """A linear layer replaced by its latent factorization.

    Holds the continuous latents (u_latent, v_latent), scales (s1, s2) and bias
    as parameters at the declared ``compute_dtype`` — the ONE dtype the pipeline
    computes in. The effective terms come from :func:`_factor_terms` — the single
    definition every forward and write-back site consults — and the forward maps
    ``x -> ((s2 ⊙ f(V)) V-side) ((f(U)) (s1 ⊙ U))ᵀ x + b``.
    """

    def __init__(
        self,
        u_latent: torch.Tensor,
        v_latent: torch.Tensor,
        bias: torch.Tensor,
        factor_mode: str = "binary",
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        dout, r = u_latent.shape
        din, r_v = v_latent.shape
        if r != r_v:
            raise ValueError("factor ranks disagree")
        # Unset means "the dtype this layer's own latents already carry": a caller
        # that built device-native latents at the pipeline dtype must not be
        # silently re-cast to a default. bf16 has no NumPy scalar type, so a
        # float64 default here would be unreachable for the bf16 arm anyway.
        dtype = compute_dtype if compute_dtype is not None else u_latent.dtype
        self.u_latent = nn.Parameter(u_latent.clone().to(dtype))
        self.v_latent = nn.Parameter(v_latent.clone().to(dtype))
        self.s1 = nn.Parameter(torch.ones(dout, dtype=dtype))
        self.s2 = nn.Parameter(torch.ones(din, dtype=dtype))
        self.bias = nn.Parameter(bias.clone().to(dtype))
        self.dout, self.din, self.rank = dout, din, r
        self.compute_dtype = dtype
        # The ablation mode is solver state carried by the layer, so every
        # forward site agrees without re-deriving it (D-1/D-6).
        self.factor_mode = factor_mode

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        rank: int,
        bias: torch.Tensor | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        """Seed the latents from a truncated SVD of the weight (A-6).

        Two lanes, because bf16 SVD is platform-dependent and may not converge:

        * accelerator: ``torch.linalg.svd`` at float32 accumulation dtype — an
          explicit, named accumulation site, not the device dtype;
        * CPU: the NumPy SVD verification fixture at the declared dtype
          (float64), cast to ``compute_dtype`` before it is stored.

        Either way the returned module's parameters are ``compute_dtype``: the
        SVD is an initialization, not a storage decision.
        """
        target = compute_dtype if compute_dtype is not None else weight.dtype
        if weight.device.type == "cuda":
            # float32 accumulation: a bf16 SVD of a d_out x d_in matrix has ~7
            # mantissa bits in the vectors and is device-dependent; float32 is
            # stable for the ranks the pipeline uses and stays one ulp class
            # below the reference it is compared against.
            u_mat, s_vec, vt_mat = torch.linalg.svd(
                weight.detach().to(torch.float32), full_matrices=False
            )
            rr = min(rank, int(s_vec.numel()))
            root = s_vec[:rr].sqrt()
            u0 = (u_mat[:, :rr] * root).contiguous()
            v0 = (vt_mat[:rr].T * root).contiguous()
            if bias is None:
                b0 = torch.zeros(weight.shape[0], device=weight.device, dtype=target)
            else:
                b0 = bias.detach().to(target)
            return cls(u0, v0, b0, compute_dtype=target)
        # reference-path: float64 by design — the NumPy lane's declared dtype.
        w_np = weight.detach().to(torch.float64).cpu().numpy()
        u_mat, s_vec, vt_mat = np.linalg.svd(w_np, full_matrices=False)
        rr = min(rank, s_vec.size)
        # Latents live where the weights live. Derived from `weight` rather than from a
        # threaded-out device setting: a second source of truth can disagree with where the
        # parameters actually are, and a device mismatch surfaces frames from the cause. On
        # CPU this is a no-op, so the reference path is unchanged by construction.
        _dev = weight.device
        u0 = torch.from_numpy(np.ascontiguousarray(u_mat[:, :rr] * np.sqrt(s_vec[:rr]))).to(_dev)
        v0 = torch.from_numpy(np.ascontiguousarray(vt_mat[:rr].T * np.sqrt(s_vec[:rr]))).to(_dev)
        # Built in the WEIGHT's own dtype and cast once by the constructor: the
        # width of the transient host numpy array above is the oracle's, not the
        # module's storage decision.
        if bias is None:
            b0 = torch.zeros(w_np.shape[0], device=weight.device, dtype=weight.dtype)
        else:
            b0 = bias.detach().to(weight.dtype)
        return cls(u0, v0, b0, compute_dtype=target)

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
        s1 / s2: scale vectors at the pipeline's declared host grade.
        bias: bias vector (fp, unfactorized) at the same grade.
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
        # The layer's OWN dtype — the declared compute dtype it stores its
        # parameters at (D-1). A hardcoded width here would write a weight the
        # pipeline's other stages never computed.
        left, right = _factor_terms(fac, fac.u_latent.dtype, factor_mode)
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
                # The packed weight is assembled through a NumPy round-trip and is born on the
                # CPU; the activation is not. Coerce here, after the assignment that produces
                # it -- same class as the latents and the stem batches. No-op on CPU.
                w_q = w_q.to(x_in.device)
                return x_in @ w_q.T + _f.bias.to(device=x_in.device, dtype=dt)

            originals[name] = linear.forward
            linear.forward = frozen_forward  # type: ignore[method-assign]
        return block(x)
    finally:
        for name, linear in _linear_children(block):
            if name in originals:
                linear.forward = originals[name]  # type: ignore[method-assign]


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
            "D-5). Unset QK_FACTOR_MODE (or set it to 'binary') for a run that "
            "exports."
        )
    from .. import kernels

    u_np = to_host_np(u_latent)
    v_np = to_host_np(v_latent)
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
    explosion by mechanism (measured: block 0 ~250x on the median and
    more in norm).

    Called at balance time per layer, this trips within the first minutes
    of the solve phase instead of surfacing as garbage perplexity after
    the full pipeline.
    """
    # accumulation-site: float64 by arithmetic justification. This gate decides
    # CATASTROPHE (band [1/F, F] with F ~ 1e4), and the statistic is a Frobenius
    # norm summing over every d_out x d_in entry — a bf16 accumulator would carry
    # ~7 mantissa bits into a ratio whose whole job is to be right about orders of
    # magnitude. The host arrays it consumes are already at the declared host
    # grade (to_host_np); widening them here is the gate's, not the pipeline's.
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
