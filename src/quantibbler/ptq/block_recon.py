"""Phase 2 block-wise reconstruction (tasks 4.1-4.4; [NQ] Sec 3.2,
Algorithm 1 lines 8-24).

Per transformer block, sequentially:

1. capture quantized-prefix inputs ``X_b`` (calibration data through the
   already-quantized prefix) and teacher targets ``Y_b`` (the PRISTINE fp
   block on those inputs) — Algorithm 1 lines 9-10; the teacher's weights
   come from a frozen reference copy of the block, never the live module;
2. factorize the block's attention projections (Q/K/V) in module
   declaration order (Step 2, lines 14-17) on the preconditioned target
   W~ = D~_out W D~_in, freezing each into the module as it is solved;
3. TuneFP: brief full-precision tuning of the block's STILL-FP remainder
   (O/G/U/D) against the cached teacher for T_pre steps ([NQ] Step 1,
   error-propagation mitigation) — fired ONCE per block at the [NQ]
   Figure 3 placement, after Q/K/V are factorized;
4. factorize the remaining linears (O/G/U/D), refining each one's
   factors (U, V, s1, s2) with TuneLatentSTE against the block objective
   of Eq. 10 (Step 3, line 19) right after its own factorization;
5. sign extraction ``sign(U)``/``sign(V)`` (sign(0) = +1) and packing via
   the qk-kernels dispatch (lines 20-23).

Embeddings, LM heads and norms stay full precision (A-9); only ``nn.Linear``
children of the block are factorized. In-block layer order is module
declaration order (A-12).

Stage changes folded here:

* ``qk-fix-tunefp-teacher`` — the teacher is the pristine FP block's output
  on the compressed-prefix stream, captured once per block from a frozen
  reference copy (A1), memory-only (A2); TuneFP fires at the Figure 3
  placement with the still-FP remainder as its tunable set; a zero
  first-step gradient aborts with ``tune_fp_vacuous_teacher`` carrying both
  the gradient norm and the initial residual norm (A3).
  Under the dual streams the first target's skip is NOT a vacuity
  claim: ``Y_b`` comes from the fp reference reading a different input than
  the block's own forward, so the target-1 gradient is no longer ~0 and the
  engine has real work. The skip stands only as "nothing to compensate yet"
  -- at ``idx == 0`` the block is still entirely FP and the teacher is its
  deepcopy, so there is genuinely nothing to tune until something is
  factorized.
* ``qk-fix-ste-sign-adaptation`` — per-linear, per-factor flip
  instrumentation, per-layer refinement on the just-factorized linear with
  a deterministic split of the line-19 T_post budget, and flip acceptance
  judged against the block-reconstruction objective.
* ``qk-fix-fp-resume-guard`` — the resume guard is driven by the live
  block's expected quantizable names, not by the archive's contents.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path  # checkpoint resume: accept str or Path

import numpy as np
import torch
from torch import nn

from .bit_polish import greedy_polish
from .calibration import LayerStatsBook, collect_stats, robust_diag
from .config import PTQConfig
from .lb_admm import SOLVER_PROBE_ENV, lb_admm, magnitude_balance
from .progress import ProgressReporter
from .solver_backend import solver_backend_provenance, to_host_np

# Artifact markers (greppable in code, spec, and the run record).
#
# ``TUNE_FP_LOSS_MARKER`` — the corpus names the TuneFP call AND anchors the loss
# FORM ("weighted MSE", [NQ] App. C :3012) but writes no weight formula. This stage
# weights the block-output squared residual against the cached teacher by the terminal
# linear's robust-diagonalized Phase-1 output statistic; WHICH statistic does the
# weighting is the design's informed assumption, not a paper pin. The marker
# therefore declares the anchor (nq_3012) and the assumption together, so a reader of
# a run record cannot mistake one for the other.
# ``STE_RESCOPE_MARKER`` — a scoped linear whose refinement produced zero
# accepted sign flips is declared scale-only in the run record under this
# marker; a zero-flip completion without the declaration is impossible by
# construction.
TUNE_FP_LOSS_MARKER = "tune_fp_loss: weighted_mse_nq_3012 (weight source: informed_assumption)"
STE_RESCOPE_MARKER = "ste_scale_only_declared"


class VacuousTeacherError(RuntimeError):
    """A TuneFP pass saw a zero first-step gradient in a fidelity-claiming run.

    Raised by :func:`_tune_fp` when ``cfg.paper_faithful`` is set and the
    pre-tune objective gradient norm is exactly zero: under the Figure 3
    placement the operative call always has factorized in-block layers, so a
    zero gradient is a construction defect (broken teacher or detached
    student), never legitimate "nothing to compensate". A run that claims
    paper fidelity must fail rather than print "paper-faithful" on a stage
    that silently did not execute (qk-fix-tunefp-teacher A3).

    Carries the two norms and the reason so the caller can write a
    forensically complete failure report (see
    :func:`quantibbler.qk_report.stage_failure_report`).
    """

    def __init__(
        self,
        *,
        block: str,
        reason: str,
        gradient_norm: float,
        initial_residual_norm: float,
    ) -> None:
        self.stage = "tune_fp"
        self.block = block
        self.reason = reason
        self.gradient_norm = gradient_norm
        self.initial_residual_norm = initial_residual_norm
        super().__init__(
            f"tune_fp vacuous teacher at block {block!r} ({reason}): gradient norm "
            f"{gradient_norm!r}, initial residual norm {initial_residual_norm!r} — a "
            "run claiming paper fidelity cannot continue on a stage that did not execute"
        )


# `SOLVER_PROBE_ENV` is imported from ``lb_admm``: the same flag gates the
# solver's per-iteration eigenvalue probe, and this writer must never be able to
# name a different variable than the gate reads.
SOLVER_PROBE_PATH_ENV = "QK_SOLVER_PROBE_PATH"
SOLVER_PROBE_DEFAULT_PATH = "/out/solver_probe.jsonl"


def _record_solver_probe(name, w, bal, res, cfg) -> None:
    """Append one JSONL row per solved linear: the solver's own numbers.

    Diagnostic, read-only, OFF by default -- with QK_SOLVER_PROBE unset this
    returns before touching an array, so a default run is unchanged.

    Why it exists: the attribution probe needs per-linear cos and the solve's
    iteration count, and NOTHING in this tree records either. Every ``cos`` in
    the solver path is the cosine LR schedule; ``n_iter``/``converged`` live on
    ADMMResult and are dropped; the PTQ stage prints nothing per block. And
    stage records only reach disk in the end-of-run report, so a per-layer
    probe that is stopped early -- which a per-layer probe must be -- leaves no
    numbers at all. Hence JSONL appended as each linear solves.

    Both cos numbers are recorded deliberately, because they answer different
    questions and the gap between them is measured history: the SVD arm split
    raw 0.84 / pack 0.45, and a single number would have hidden that the pack
    stage, not the solve, is the bottleneck.

      cos_raw  = cos(W, U V^T)                              [NQ] Eq. 9 state
      cos_pack = cos(W, s1 o (sign(U) sign(V)^T) o s2^T)    [NQ] Eq. 10 packed

    Both against the ORIGINAL (un-preconditioned) FP weight W, never W~, per
    the stage-boundary pin. The effective rho_end and its source ride along, so
    a probe row can never be mislabelled about which arm produced it.

    ``rel_raw`` measures ``||W - U V^T||_F / ||W||_F`` on the CONTINUOUS
    product, which is deliberately NOT fit to W: the alpha refit fits the
    *packed* product (see lb_admm.magnitude_balance -- it corrects
    ``s1 o Ub Vb^T o s2^T`` and never rescales the latents). Nothing consumes
    the continuous product either: every site that turns latents into a weight
    goes through ``_factor_terms``, whose binary mode applies ``sign``, i.e.
    the packed class -- the module forward, the live STE forward
    (``_quantized_forward_block``) and the ``_set_linear`` write-back alike.

    So ``rel_raw > 1`` is EXPECTED on the ADMM arm and is not a defect signal.
    It reads as magnitude drift of the continuous latents, which is the
    consensus shaping that makes the packed product good in the first place
    (measured: ADMM packs at -0.133, SVD at -0.389 on the same real weight).
    Driving that drift toward 1.0 would trade away the pack
    fidelity the solver is chosen for. Do not close it as a balance-stage bug;
    the useful reading is that raw fit is a weak predictor of the served
    artifact, which is the same lesson the SVD arm taught from the other side.
    """
    if not os.environ.get(SOLVER_PROBE_ENV):
        return

    from .lb_admm import admm_reg_provenance, rho_end_provenance
    from .solver_backend import resolve_device

    u = to_host_np(bal.U)
    v = to_host_np(bal.V)
    s1 = to_host_np(bal.s1)
    s2 = to_host_np(bal.s2)
    wf = to_host_np(w)

    def _cos(recon: np.ndarray) -> float:
        num = float(np.sum(wf * recon))
        den = float(np.linalg.norm(wf) * np.linalg.norm(recon))
        return num / den if den > 0.0 else float("nan")

    recon_raw = u @ v.T
    recon_pack = (s1[:, None] * (np.sign(u) @ np.sign(v).T)) * s2[None, :]
    nw = float(np.linalg.norm(wf))

    def _rel(recon: np.ndarray) -> float:
        return float(np.linalg.norm(wf - recon)) / nw if nw > 0.0 else float("nan")

    value, source = rho_end_provenance(cfg.rho_end)
    # Regime identity, resolved exactly as the solver resolves it. A row
    # without these cannot say which regime produced its cos, which is the whole
    # point of the regime label (old-regime rows are inadmissible for the claim).
    reg_value, reg_source = admm_reg_provenance(cfg.admm_reg)
    # Backend arm + the device it actually executed on. "torch" alone does not
    # identify a row: cpu and cuda are different numerics (reduction order), so
    # an unlabelled device is the same mislabelling class as an unlabelled rho.
    backend_value, backend_source = solver_backend_provenance(cfg.solver_backend)
    solver_device = resolve_device(backend_value, device=os.environ.get("QUANTIBBLER_DEVICE"))
    row = {
        "name": name,
        "solver": cfg.solver,
        # The dtype the solve actually ran in. Two arms whose rows differ only
        # in precision are otherwise indistinguishable -- the same provenance
        # gap as an unlabelled rho_end, and worse here, because an fp32 row's
        # cos is not comparable to an fp64 row's.
        "compute_dtype": cfg.compute_dtype,
        # Image identity, carried in from the wrapper via the container env: the
        # driver cannot inspect its own image, and a tag is mutable, so the
        # digest is what makes "did both arms run the same binary" answerable
        # from the artifact. Absent outside a container, and stated as such.
        "image_digest": os.environ.get("QUANTIBBLER_IMAGE_DIGEST", "unknown"),
        # Which device the torch stages ran on. An fp32 row on a GPU and an fp32
        # row on a CPU are different measurements of the same thing.
        "device": os.environ.get("QUANTIBBLER_DEVICE", "unknown"),
        "rho_end": value,
        "rho_end_source": source,
        "rho_start": cfg.rho_start,
        "rho_start_source": "config",
        "admm_reg": reg_value,
        "admm_reg_source": reg_source,
        "solver_backend": backend_value,
        "solver_backend_source": backend_source,
        "solver_device": solver_device,
        "n_iter": getattr(res, "n_iter", None),
        "converged": getattr(res, "converged", None),
        "w_fro": nw,
        "wq_raw_fro": float(np.linalg.norm(recon_raw)),
        "wq_pack_fro": float(np.linalg.norm(recon_pack)),
        "cos_raw": _cos(recon_raw),
        "cos_pack": _cos(recon_pack),
        "rel_raw": _rel(recon_raw),
        "rel_pack": _rel(recon_pack),
        "rank": int(u.shape[1]),
        "ts": time.time(),
    }
    path = os.environ.get(SOLVER_PROBE_PATH_ENV) or SOLVER_PROBE_DEFAULT_PATH
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        # The numbers ARE the deliverable of a probe run: a probe that cannot
        # write them has measured nothing, so fail loudly rather than return.
        raise RuntimeError(f"solver probe enabled but cannot write {path}: {exc}") from exc
    print(
        f"[solver_probe:{name}] solver={row['solver']} rank={row['rank']} "
        f"n_iter={row['n_iter']} converged={row['converged']} "
        f"cos_raw={row['cos_raw']:.6f} cos_pack={row['cos_pack']:.6f} "
        f"rel_raw={row['rel_raw']:.6f} rel_pack={row['rel_pack']:.6f}",
        flush=True,
    )


#: Row chunk for forwards over the full capture tensor.
#:
#: At the paper geometry the capture is (128 seqs x 8) = 1024 rows x 2048 tokens
#: x 1024 d_model (4 GiB at bf16). The block's MLP expands that to
#: (1024, 2048, 3072), so ``act(gate(x)) * up(x)`` alone requires TWO 12 GiB
#: intermediates live at once -- 24 GiB, plus the 4 GiB capture and a 4 GiB
#: clone, against a 32 GiB container cap. That is what killed the first
#: paper-faithful run: dmesg anon-rss 33,480,244kB = 99.8% of the cap, a
#: one-window spike no polling watchdog can catch.
#:
#: Chunking bounds the peak by ``chunk_rows`` instead of by capture geometry --
#: 64 rows holds 0.75 GiB per MLP intermediate instead of 12 GiB -- which is
#: what makes the cost independent of rank/seq/calib growth rather than tuned to
#: today's numbers. Same idiom as ``calibration.collect_stats``'s ``chunk_rows``;
#: note that fp32 would DOUBLE the intermediates, so this is required on any
#: device, not merely a CPU workaround.
CAPTURE_CHUNK_ROWS = 64


def _forward_chunked(module, x, chunk_rows: int = CAPTURE_CHUNK_ROWS):
    """Run ``module(x)`` without ever holding more than ``chunk_rows`` of activations.

    Row-chunked and concatenated, which is exact for a row-wise block -- the
    module sees each row independently, so the only difference is floating-point
    blocking: BLAS picks its reduction order from the batch shape, exactly as the
    pinned thread count does, so results match within rounding rather than
    bit-for-bit. Cost is compute-neutral to a few percent (16 forwards instead of
    1 at the paper geometry) in exchange for a peak bounded by chunk size.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0 or x.shape[0] <= chunk_rows:
        return module(x)
    return torch.cat(
        [module(x[lo : lo + chunk_rows]) for lo in range(0, x.shape[0], chunk_rows)], dim=0
    )


def _weights_fingerprint(module: nn.Module) -> str:
    """Stable sha256 over a module's parameters (the pristine teacher source).

    The TuneFP teacher's provenance is only provable if the artifact can name
    the WEIGHTS the teacher was built from (qk-fix-tunefp-teacher A1): this
    hash is recorded alongside the capture index, so a reader can tell a
    pristine-pinned teacher from one taken off live (already written-back)
    module state.
    """
    digest = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(param.shape)).encode("utf-8"))
        digest.update(str(param.dtype).encode("utf-8"))
        digest.update(param.detach().to(torch.float64).contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


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
    "STE_RESCOPE_MARKER",
    "TUNE_FP_LOSS_MARKER",
    "VacuousTeacherError",
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
    u = u.to(s1.device)
    v = v.to(s2.device)
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
        # Latents live where the weights live. Derived from `weight` rather than from a
        # threaded-out device setting: a second source of truth can disagree with where the
        # parameters actually are, and a device mismatch surfaces frames from the cause. On
        # CPU this is a no-op, so the reference path is unchanged by construction.
        _dev = weight.device
        u0 = torch.from_numpy(np.ascontiguousarray(u_mat[:, :rr] * np.sqrt(s_vec[:rr]))).to(_dev)
        v0 = torch.from_numpy(np.ascontiguousarray(vt_mat[:rr].T * np.sqrt(s_vec[:rr]))).to(_dev)
        if bias is None:
            b0 = torch.zeros(w_np.shape[0], dtype=torch.float64, device=weight.device)
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


# The block's canonical terminal linear ([NQ] App. C :3012 -- the output projection
# whose Phase-1 output statistic weights the tuning loss; Algorithm 1 lines 2-5).
# ``down_proj`` is the Llama-family name. Other architectures name the same function
# differently, and those aliases are deliberately NOT enumerated here: the tree holds
# no such list, and inventing one would be inventing the paper's vocabulary. Until the
# alias set is pinned, an unrecognised terminal name takes the uniform fallback and
# the caller must record it as such.
CANONICAL_TERMINAL_NAMES: tuple[str, ...] = ("down_proj",)

# The recorded fallback: a block with no canonical terminal linear (or one carrying no
# usable statistic) is tuned with uniform weights, and that must leave a trace naming
# the block and the missing layer. A uniform-weighted stage WITHOUT this event is
# impossible by construction -- silent fail-open is the pattern this lane closed.
TUNE_LOSS_FALLBACK_EVENT = "tune_loss_uniform_fallback"


def _canonical_terminal_pair(pairs):
    """The ``(entry, (name, linear))`` pair for the block's canonical terminal linear.

    Name-based, never positional: book order follows the block's own module order,
    which is not the paper's rule. Matching is on the final dotted segment, so both
    ``down_proj`` and ``mlp.down_proj`` resolve. ``None`` means the block carries no
    canonical terminal name at all -- the caller must then fall back to uniform
    weights AND record ``tune_loss_uniform_fallback`` naming the block and the
    missing canonical layer.
    """
    for entry, named in pairs:
        if named[0].rsplit(".", 1)[-1] in CANONICAL_TERMINAL_NAMES:
            return entry, named[0]
    return None


def _usable_statistic(entry) -> bool:
    """Is this entry's output statistic usable as a weight (~[NQ] Phase-1 output)?

    ``None``, empty, or non-finite raw statistics are not weights. They take the same
    recorded fallback as a missing terminal rather than poisoning the loss with NaN --
    the spec's "or carries no Phase-1 statistic" arm.
    """
    raw = getattr(entry, "d_out_raw", None)
    if raw is None:
        return False
    arr = np.asarray(raw, dtype=np.float64)
    return arr.size > 0 and bool(np.all(np.isfinite(arr)))


def _resolve_loss_weight(cfg: PTQConfig, pairs) -> tuple[np.ndarray | None, str, str | None]:
    """Resolve the weighting and say WHY, so the caller cannot record a silence.

    Returns ``(weight, source, missing_canonical_layer)``:

    * ``(array, "terminal_grad:<linear>", None)`` -- canonical terminal found and usable.
    * ``(None, "uniform", None)`` -- the explicit ablation (``block_loss_weight ==
      "uniform"``). Needs no terminal and is NOT a fallback: no event.
    * ``(None, "uniform_fallback", "<canonical name>")`` -- the block has no canonical
      terminal, or its statistic is unusable. The CALLER must record
      ``TUNE_LOSS_FALLBACK_EVENT`` naming the block and this layer.
    """
    if cfg.block_loss_weight == "uniform":
        return None, "uniform", None
    found = _canonical_terminal_pair(pairs)
    if found is None or not _usable_statistic(found[0]):
        return None, "uniform_fallback", CANONICAL_TERMINAL_NAMES[0]
    entry, name = found
    _d_in, d_out = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=cfg.tau, gamma=cfg.gamma)
    return np.asarray(d_out, dtype=np.float64), f"terminal_grad:{name}", None


def _terminal_loss_weight(cfg: PTQConfig, pairs) -> np.ndarray | None:
    """Weighted-MSE weight for the block's canonical terminal linear, else ``None``.

    The weight is the robust diagonal of that linear's Phase-1 output statistic
    ``d_out_raw`` ([NQ] App. C :3012 weighted MSE; the statistic's construction is
    Algorithm 1 lines 2-5, Step 2-1 Eq. 2). It is detached and non-trainable by
    construction: a host array, never a graph node.

    ``None`` means uniform weights, and it has THREE causes the caller must not
    conflate: ``cfg.block_loss_weight == "uniform"`` (the explicit ablation, which
    needs no terminal at all and is NOT a fallback), an empty or canonical-free
    statistics book, and a canonical terminal carrying no usable statistic. Only the
    latter two are the fallback the spec requires to be recorded.
    """
    return _resolve_loss_weight(cfg, pairs)[0]


# The unit-mean normalization's derivation, recorded beside the weights it scales
# (qk-tune-loss-scale). The robust diagonal's ABSOLUTE scale is a free numerical
# degree of freedom: its relative per-feature structure is the statistical content
# ([NQ] App. C weighted MSE, Algorithm 1 lines 2-5 Eq. 2). At raw scale -- a
# mean-SQUARED-gradient statistic, ~1e-10 at hidden-state level -- the weighted loss
# sits ~1e10x under the residual, the step-0 Adam gradient falls below the
# optimizer's absolute eps, and the step rounds back on the parameter dtype: the
# measured inert-stage regime (qk-pertarget-cycle design.md 23, the v5 report).
LOSS_WEIGHT_MEAN_NORMALIZATION = "unit_mean_normalization"


def _normalize_weight_to_unit_mean(weight) -> tuple[np.ndarray, float, float]:
    """Rescale a resolved terminal weight to unit mean (qk-tune-loss-scale).

    Preserves the robust diagonal's per-feature RELATIVE structure and restores its
    absolute scale to the residual's own scale. Returns ``(normalized, mean_before,
    mean_after)`` so both means can travel in the stage record: the scale applied to
    the loss is then auditable from the artifact, not inferred from the code.

    A non-finite or non-positive mean is a degenerate weight -- nothing to rescale,
    and dividing by it would poison the loss -- so the vector is handed back
    unchanged with equal means, which is what the record must be able to say.
    """
    arr = np.asarray(weight, dtype=np.float64)
    if arr.size == 0:
        return arr, 0.0, 0.0
    pre = float(arr.mean())
    if not np.isfinite(pre) or pre <= 0.0:
        return arr, pre, pre
    norm = arr / pre
    return norm, pre, float(norm.mean())


def _optimizer_gradient_floor(optimizer_eps: float, params) -> tuple[float, str | None, float]:
    """Resolution-derived first-step gradient floor (qk-tune-loss-scale).

    ``floor = optimizer_eps * dtype_relative_resolution``, on the COARSEST parameter
    dtype in the stage: a bf16 parameter is the one that rounds back, so a mixed
    fp32/bf16 block is guarded by its bf16 members.

    Derivation of record (design.md): at step 0 the bias-corrected Adam update is
    ``lr * g / (|g| + eps)``, which in the eps-dominated regime ``|g| << eps``
    degrades to ``lr * g / eps``. A parameter dtype represents relative change only
    above ``finfo(dtype).eps``, so the asymptotically-smallest gradient whose
    eps-damped step still clears that floor for a unit-magnitude parameter is the
    product ``optimizer_eps * finfo(dtype).eps``. Measured on the v5 rows: bf16
    gives 7.8125e-11 and every recorded step-0 gradient (7.9e-12..6.2e-11) sits
    below it. Exact zero stays a special case (``0.0 < floor``).

    Returns ``(floor, dtype_name, relative_resolution)`` -- all three are recorded,
    so a reader can re-derive the floor from the artifact.
    """
    rel = 0.0
    dtype_name: str | None = None
    for p in params:
        eps_dtype = float(torch.finfo(p.dtype).eps)
        if dtype_name is None or eps_dtype >= rel:
            rel = eps_dtype
            dtype_name = str(p.dtype)
    return float(optimizer_eps) * rel, dtype_name, rel


def _tune_fp(
    block: nn.Module,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    cfg: PTQConfig,
    batch_size: int,
    *,
    label: str = "block",
    tune_names: list[str] | None = None,
    factorized_in_block: int = 0,
    recorder=None,
    capture_index: int | None = None,
    teacher_fingerprint: str | None = None,
    teacher_source: str = "frozen_reference_copy",
    loss_weight: np.ndarray | None = None,
    loss_weight_source: str = "uniform",
    loss_weight_mean_before: float | None = None,
    loss_weight_mean_after: float | None = None,
) -> dict:
    """TuneFP ([NQ] Step 1): T_pre steps of full-precision block tuning
    against the cached teacher with Adam + cosine lr ([NQ] App. C).

    Placement and tunable set (qk-fix-tunefp-teacher, [NQ] Figure 3): the
    caller fires this ONCE per block, after the attention projections are
    factorized and before the remaining linears are, and passes the
    still-FP remainder as ``tune_names`` — so the objective
    ``||block_tuned(x_b) - y_b||**2`` is nonzero by construction (the
    factorized Q/K/V are the compensation source). ``y_b`` is the cached,
    pristine-pinned teacher captured once per block; it is never recomputed
    from tuned weights.

    Guard (A3, fail-closed; extended by qk-tune-loss-scale): a first-step
    gradient norm below the resolution-derived floor (``optimizer eps x
    parameter-dtype relative resolution``) — or any call emitted with
    ``factorized_in_block == 0`` (the legacy pre-loop placement the figure
    rules out) — aborts with the recorder event
    ``tune_fp_vacuous_teacher`` carrying BOTH the gradient norm and the
    initial residual norm, plus the floor and the inputs it was derived from
    (gradient-zero + residual-zero = broken teacher; gradient-zero +
    residual-nonzero = detached student; nonzero-but-below-floor = the measured
    inert regime where every step rounds back on the parameter dtype).

    The per-target loop's first-target skip is a separate matter and is NOT
    this guard: under dual streams it is justified as "nothing to compensate
    yet" (the block is still entirely FP and the teacher is its deepcopy),
    not as inherent vacuity. Only a call that actually runs and returns a
    zero first-step gradient reaches the fail-closed path below. In a diagnostic
    run the stage continues with pristine weights and the event stays on
    record; in a run claiming paper fidelity it raises
    :class:`VacuousTeacherError` — there is no trivial-zero carve-out under
    the figure placement.

    Returns the honest stage record (steps, gradient/residual norms, the fp
    weight delta versus the pre-tune snapshot) for the run artifact.

    Bulletproofing (measured): a single inf/NaN gradient in
    bf16 poisons EVERY parameter of a tensor at the next opt.step(), and
    the NaN then cascades through statistics/preconditioning into the
    solver (observed: an entire o_proj weight all-NaN by the lb_admm
    boundary, killing the run at block ~8). A tuning stage therefore
    never trusts its own step: non-finite losses are skipped, a parameter
    snapshot is restored and the stage ends early if finiteness is ever
    violated. A slightly underfit block beats a poisoned pipeline.
    """
    t_stage = time.perf_counter()

    def _stamp() -> None:
        """Close the stage's row time on EVERY return path.

        A stage that returns early (no tunables, vacuous teacher) still spent
        wall time, and the row that reports it must not read as 0 s.
        """
        stats["elapsed_s"] = time.perf_counter() - t_stage

    stats: dict = {
        "stage": "tune_fp",
        "label": label,
        "tunable": sorted(tune_names) if tune_names is not None else None,
        "factorized_in_block": int(factorized_in_block),
        "capture_index": capture_index,
        "teacher_source": teacher_source,
        "teacher_fingerprint": teacher_fingerprint,
        "marker": TUNE_FP_LOSS_MARKER,
        "steps": 0,
        "gradient_norm": None,
        "initial_residual_norm": None,
        "residual_after": None,
        "weight_delta_rel": None,
        "vacuous": False,
        # Inertness is DECLARED, not implied (qk-tune-loss-scale, requirement 3):
        # `None` means the stage did not run far enough to have an answer, never
        # "it was fine".
        "inert": None,
        # Weighting actually used, on the record from construction so the
        # vacuous-teacher early return carries it too.
        "loss_weight": "uniform" if loss_weight is None else "terminal_grad",
        "loss_weight_source": loss_weight_source,
        # The scale applied to the weighted MSE: the raw resolved weight's mean
        # and the mean actually handed to the loss. Both `None` when the
        # uniform/fallback path ran (there is no scale to audit).
        "loss_weight_mean_before": loss_weight_mean_before,
        "loss_weight_mean_after": loss_weight_mean_after,
        "loss_weight_normalization": (
            LOSS_WEIGHT_MEAN_NORMALIZATION if loss_weight_mean_after is not None else None
        ),
        # The resolution-derived vacuous floor and its derivation inputs.
        # Filled once the optimizer exists.
        "gradient_floor": None,
        "floor_optimizer_eps": None,
        "floor_dtype_rel_resolution": None,
        "floor_param_dtype": None,
        # The device the stage's tensors ran on, and the stage's own row time.
        # Both are read off the ACTIVATION, the authority for where this stage
        # executes: an unlabelled device cannot say which numerics produced the
        # row, and a row without its own time cannot replace the proposal's
        # per-stage cost estimate with a measured one.
        "device": str(x_b.device),
        "elapsed_s": None,
    }
    all_linears = _linear_children(block)
    if tune_names is None:
        linears = all_linears
    else:
        wanted = set(tune_names)
        linears = [(n, m) for n, m in all_linears if n in wanted]
    params = [p for _, m in linears for p in m.parameters()]
    if not params:
        stats["steps"] = 0
        _stamp()
        return stats

    def _residual() -> float:
        with torch.no_grad():
            out = _forward_chunked(block, x_b)
            return float((out - y_b).pow(2).mean().sqrt())

    opt = torch.optim.Adam(params, lr=cfg.stage_pre.lr)
    # The resolution-derived floor is read off THIS optimizer's eps and THESE
    # parameters' dtypes -- never a module constant -- so a reader re-derives it
    # from the record (qk-tune-loss-scale, requirement 2).
    opt_eps = float(opt.defaults["eps"])
    floor, floor_dtype, floor_rel = _optimizer_gradient_floor(opt_eps, params)
    stats["gradient_floor"] = floor
    stats["floor_optimizer_eps"] = opt_eps
    stats["floor_dtype_rel_resolution"] = floor_rel
    stats["floor_param_dtype"] = floor_dtype
    n = x_b.shape[0]
    bs = max(1, min(batch_size, n))
    total = max(cfg.stage_pre.epochs * (n // bs), 1)
    snapshot = [p.detach().clone() for p in params]
    before_residual = _residual()
    skipped = 0
    # TuneFP is one of the two stages that dominate wall time, so it is the one
    # the user is most blind during without progress output.
    reporter = ProgressReporter("tunefp", total)
    steps_done = 0
    for step in range(total):
        steps_done = step + 1
        reporter.tick(steps_done)
        if cfg.cosine_schedule and total > 1:
            import math

            lr = cfg.stage_pre.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            for group in opt.param_groups:
                group["lr"] = lr
        idx = torch.arange(
            (step % (n // bs or 1)) * bs, (step % (n // bs or 1)) * bs + bs, device=x_b.device
        )
        out = block(x_b[idx])
        # Weighted MSE over the block-output residual ([NQ] App. C :3012): one
        # weight per output feature, applied in the stage's compute dtype. The
        # uniform case runs the SAME expression with all-ones weights, so the
        # ablated path and the shipped path are one path. Multiplying by exactly
        # 1.0 is bit-identical, so a uniform run is unchanged by this change.
        if loss_weight is None:
            w = torch.ones(out.shape[-1], dtype=out.dtype, device=out.device)
        else:
            w = torch.as_tensor(loss_weight, dtype=out.dtype, device=out.device)
        loss = (w[None, :] * (out - y_b[idx]).pow(2)).mean()
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        if step == 0:
            grad_sq = 0.0
            for p in params:
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().pow(2).sum())
            grad_norm = float(np.sqrt(grad_sq))
            residual_norm = float(loss.detach().sqrt())
            stats["gradient_norm"] = grad_norm
            stats["initial_residual_norm"] = residual_norm
            # The guard is a FLOOR, not an equality (qk-tune-loss-scale,
            # requirement 2): a nonzero gradient below the resolution-derived
            # threshold is dead for exactly the same reason an identically-zero
            # one is -- both round every step back on the parameter dtype -- and
            # exact zero stays a special case of `grad_norm < floor`.
            if grad_norm < floor or factorized_in_block <= 0:
                if factorized_in_block <= 0:
                    reason = "no_in_block_layer_factorized"
                elif grad_norm == 0.0:
                    reason = "zero_first_step_gradient"
                else:
                    reason = "below_resolution_gradient_floor"
                stats["vacuous"] = True
                stats["vacuous_reason"] = reason
                # A vacuous stage restored its entry snapshot: it moved nothing,
                # so the inertness declaration is the honest one (requirement 3).
                stats["inert"] = True
                event = {
                    "block": label,
                    "gradient_norm": grad_norm,
                    "initial_residual_norm": residual_norm,
                    "reason": reason,
                    "gradient_floor": floor,
                    "floor_optimizer_eps": opt_eps,
                    "floor_dtype_rel_resolution": floor_rel,
                    "floor_param_dtype": floor_dtype,
                    "capture_index": capture_index,
                    "teacher_source": teacher_source,
                    "teacher_fingerprint": teacher_fingerprint,
                }
                if recorder is not None:
                    recorder("tune_fp_vacuous_teacher", **event)
                print(
                    f"[tune_fp:{label}] VACUOUS TEACHER ({reason}): "
                    f"grad_norm={grad_norm!r} initial_residual={residual_norm!r}",
                    flush=True,
                )
                with torch.no_grad():
                    for p, p0 in zip(params, snapshot, strict=True):
                        p.copy_(p0)
                if cfg.paper_faithful:
                    raise VacuousTeacherError(
                        block=label,
                        reason=reason,
                        gradient_norm=grad_norm,
                        initial_residual_norm=residual_norm,
                    )
                _stamp()
                return stats
        opt.step()
        stats["steps"] = step + 1
        if not all(bool(torch.isfinite(p).all()) for p in params):
            with torch.no_grad():
                for p, p0 in zip(params, snapshot, strict=True):
                    p.copy_(p0)
            print(
                f"[tune_fp:{label}] non-finite parameters after step {step}; "
                "restored snapshot, ending stage early",
                flush=True,
            )
            break
    after_residual = _residual()
    num = den = 0.0
    with torch.no_grad():
        for p, p0 in zip(params, snapshot, strict=True):
            num += float((p.detach() - p0).pow(2).sum())
            den += float(p0.pow(2).sum())
    stats["residual_after"] = after_residual
    stats["residual_before"] = before_residual
    delta_rel = float(np.sqrt(num) / np.sqrt(den)) if den > 0.0 else 0.0
    stats["weight_delta_rel"] = delta_rel
    # Inertness is DECLARED, not implied (qk-tune-loss-scale, requirement 3): a
    # stage that ran steps and moved its parameters by less than the parameter
    # dtype's relative resolution did nothing measurable, and the record says so
    # instead of leaving a reader to reconcile `steps: 256` with
    # `weight_delta_rel: 0.0`.
    stats["inert"] = bool(stats["steps"] > 0 and delta_rel < floor_rel)
    stats["skipped_steps"] = skipped
    if skipped:
        print(f"[tune_fp:{label}] skipped {skipped}/{total} non-finite-loss steps", flush=True)
    reporter.complete(steps_done)  # tunefp: the count actually run, never the budget
    _stamp()
    return stats


def _tune_latent_ste(
    block: nn.Module,
    facs: dict[str, FactorizedLinear],
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    cfg: PTQConfig,
    batch_size: int,
    *,
    steps: int | None = None,
    label: str = "",
    loss_weight: np.ndarray | None = None,
    loss_weight_source: str = "uniform",
    loss_weight_mean_before: float | None = None,
    loss_weight_mean_after: float | None = None,
) -> dict:
    """TuneLatentSTE ([NQ] Step 3): joint refinement of (U, V, s1, s2) with
    the straight-through estimator against the block-level objective of
    Eq. 10, T_post steps at the stage lr ([NQ] App. C).

    Per-layer scheduling (qk-fix-ste-sign-adaptation, Figure 3 right panel):
    the caller invokes this once per just-factorized linear; the refinement
    unit is that linear and it consumes the FULL per-linear App. C budget
    ``epochs x (n // batch)`` (Step-3 prose names "the current target linear
    layer" singular; Eq. 10's variables are one linear's; App. C attaches the
    factorized stage's 8 epochs there). The block forward is the module as it
    stands, so every other block parameter is held at its current value.

    Instrumentation (same change, scenarios ``flip_counts_are_recorded`` /
    ``sign_adaptation_is_effective_or_declared``): per-step, per-factor sign
    crossings of the latents, the relative latent movement, and the scale
    (s1, s2) movement are recorded for EVERY factorized linear. A linear
    that finishes with zero crossings on both factors is what the caller
    turns into either an accepted discrete flip or the scale-only
    declaration — a silent zero-flip completion cannot be produced here,
    because the counts leave the stage in the returned record.

    The bias is deliberately NOT a refinement parameter: the qpk1 container
    carries no bias field, so a bias moved during refinement would make the
    artifact unreproducible from bits + scales + host (the reload-then-
    forward equality of the round-trip requirement). The factorized layer
    keeps the layer's pre-pipeline bias.
    """
    t_stage = time.perf_counter()
    params = [p for fac in facs.values() for p in (fac.u_latent, fac.v_latent, fac.s1, fac.s2)]
    stats: dict = {
        "stage": "ste",
        "label": label,
        "linears": sorted(facs),
        "steps": 0,
        "per_linear": {},
        # The weighted-MSE scale actually applied, the resolution-derived floor,
        # and the declared inertness. Same contract as the TuneFP row -- `None`
        # means "not measured", never "fine".
        "loss_weight_mean_before": loss_weight_mean_before,
        "loss_weight_mean_after": loss_weight_mean_after,
        "loss_weight_normalization": (
            LOSS_WEIGHT_MEAN_NORMALIZATION if loss_weight_mean_after is not None else None
        ),
        "gradient_floor": None,
        "floor_optimizer_eps": None,
        "floor_dtype_rel_resolution": None,
        "floor_param_dtype": None,
        "vacuous": False,
        "inert": None,
        # The device the stage's tensors ran on, and its own row time.
        # Read off the activation — the authority for where this stage executes.
        "device": str(x_b.device),
        "elapsed_s": None,
    }

    def _stamp() -> None:
        """Close the stage's row time on EVERY return path.

        A row time is attributable to a linear only when the call refined
        exactly that linear (the shipped per-target call pattern). A joint call
        over several facs cannot split its wall time by linear, so its rows
        report null rather than a fabricated share.
        """
        elapsed = time.perf_counter() - t_stage
        stats["elapsed_s"] = elapsed
        if len(stats["per_linear"]) == 1:
            for rec in stats["per_linear"].values():
                rec["elapsed_s"] = elapsed

    stats["loss_weight"] = "uniform" if loss_weight is None else "terminal_grad"
    stats["loss_weight_source"] = loss_weight_source
    if not params:
        _stamp()
        return stats
    # Per-tensor RELATIVE lr: one
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
    # Same resolution-derived floor as TuneFP (qk-tune-loss-scale, requirement 2),
    # read off THIS optimizer's eps and THESE parameters' dtypes. With f64 latents
    # the product is ~2.2e-24, so the guard fires only for a weight scale that puts
    # the step-0 gradient below it -- the RED fixture drives it there.
    opt_eps = float(opt.defaults["eps"])
    floor, floor_dtype, floor_rel = _optimizer_gradient_floor(opt_eps, params)
    stats["gradient_floor"] = floor
    stats["floor_optimizer_eps"] = opt_eps
    stats["floor_dtype_rel_resolution"] = floor_rel
    stats["floor_param_dtype"] = floor_dtype
    n = x_b.shape[0]
    bs = max(1, min(batch_size, n))
    if steps is None:
        total = max(cfg.stage_latent.epochs * (n // bs or 1), 1)
    else:
        total = max(int(steps), 0)
    per_linear: dict[str, dict] = {
        name: {
            "steps": 0,
            "flips_u": 0,
            "flips_v": 0,
            "per_step_flips": [],
            "first_flip_step": None,
            "u_rel_move": 0.0,
            "v_rel_move": 0.0,
            "s1_rel_move": 0.0,
            "s2_rel_move": 0.0,
            "zero_crossings": True,
            # Declared inertia (qk-tune-loss-scale, requirement 3): set False/True
            # once the row's movement is measured against the dtype resolution.
            "inert": None,
            # Every STE row names the device its refinement ran on: the report's
            # per-linear rows are what a reader compares across arms, and a row
            # that cannot say where it ran cannot be compared with one that can.
            "device": str(x_b.device),
            "elapsed_s": None,
        }
        for name in facs
    }
    stats["per_linear"] = per_linear

    def _residual() -> float:
        """The stage's own objective, over the WHOLE capture.

        ``||block_quant(x_b) - y_b||`` (mean-sqrt), the same quantity the
        optimizer minimizes per minibatch. Measured over the full ``x_b`` and
        not the first minibatch, so ``residual_before``/``residual_after`` are
        comparable and the recorded effect size is the stage's, not one
        batch's. Without this pair the 1024-step budget left no trace of
        whether the stage did anything (the flip counters alone cannot
        distinguish an effective stage from an inert one).
        """
        with torch.no_grad():
            out = _forward_chunked(lambda t: _quantized_forward_block(block, facs, t), x_b)
            return float((out - y_b).pow(2).mean().sqrt())

    residual_before = _residual()
    if total == 0:
        for rec in per_linear.values():
            rec["residual_before"] = residual_before
            rec["residual_after"] = residual_before
            rec["gradient_norm"] = 0.0
        _stamp()
        return stats
    # Same finiteness armor as TuneFP: a NaN loss or a
    # poisoned step must not kill the run or corrupt latents that the
    # exported binaries come from.
    snapshot = {name: [p.detach().clone() for p in _fac_params(fac)] for name, fac in facs.items()}
    entry = {name: [p.detach().clone() for p in _fac_params(fac)] for name, fac in facs.items()}
    prev_sign = {
        name: (torch.sign(fac.u_latent.detach()), torch.sign(fac.v_latent.detach()))
        for name, fac in facs.items()
    }
    first_grad_norm = 0.0
    skipped = 0
    reporter = ProgressReporter("ste", total)
    steps_done = 0
    for step in range(total):
        steps_done = step + 1
        reporter.tick(steps_done)
        if cfg.cosine_schedule and total > 1:
            import math

            cos_f = 0.5 * (1 + math.cos(math.pi * step / total))
            for group in opt.param_groups:
                group["lr"] = group["lr_scale"] * cfg.stage_latent.lr * cos_f
        lo = (step % (n // bs or 1)) * bs
        idx = torch.arange(lo, lo + bs, device=x_b.device)
        out = _quantized_forward_block(block, facs, x_b[idx])
        # Weighted MSE, same construction as TuneFP above ([NQ] App. C :3012):
        # one detached weight per output feature in the stage's compute dtype,
        # all-ones for the uniform case so both paths are one path.
        if loss_weight is None:
            w = torch.ones(out.shape[-1], dtype=out.dtype, device=out.device)
        else:
            w = torch.as_tensor(loss_weight, dtype=out.dtype, device=out.device)
        loss = (w[None, :] * (out - y_b[idx]).pow(2)).mean()
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        if step == 0:
            # Same point TuneFP takes its gradient norm (the first step's
            # backward, before the optimizer moves anything): a stage whose
            # gradient is identically zero is inert by construction, and this
            # is the field that says so instead of leaving it to inference.
            grad_sq = 0.0
            for p in params:
                if p.grad is not None:
                    grad_sq += float(p.grad.detach().pow(2).sum())
            first_grad_norm = float(np.sqrt(grad_sq))
            # Resolution-derived floor, not exact zero (qk-tune-loss-scale,
            # requirement 2). The stage has no fidelity-fail mechanism today, so
            # its fail-closed path is: restore the entry snapshot, declare the
            # stage vacuous AND inert on the record, and end it (design.md's
            # named delta) -- never run the rest of a budget that cannot move a
            # parameter.
            if first_grad_norm < floor:
                reason = (
                    "zero_first_step_gradient"
                    if first_grad_norm == 0.0
                    else "below_resolution_gradient_floor"
                )
                stats["vacuous"] = True
                stats["vacuous_reason"] = reason
                stats["inert"] = True
                with torch.no_grad():
                    for name, fac in facs.items():
                        for p, p0 in zip(_fac_params(fac), snapshot[name], strict=True):
                            p.copy_(p0)
                print(
                    f"[tune_latent_ste:{label}] VACUOUS ({reason}): "
                    f"grad_norm={first_grad_norm!r} floor={floor!r}",
                    flush=True,
                )
                break
        opt.step()
        # The finiteness gate runs BEFORE the counters, and the order is the
        # contract: a step the armor rolls back contributed no sign change, so
        # counting it first would poison ``zero_crossings`` (the mechanism
        # ladder's input) with flips the snapshot restore undid — the block
        # would skip the ladder and declare ``paper_ste`` on latents restored
        # uncrossed, and the stage's step count would disagree with the
        # per-linear record's.
        if not all(bool(torch.isfinite(p).all()) for p in params):
            with torch.no_grad():
                for name, fac in facs.items():
                    for p, p0 in zip(_fac_params(fac), snapshot[name], strict=True):
                        p.copy_(p0)
            print(
                f"[tune_latent_ste:{label}] non-finite parameters after step {step}; "
                "restored snapshot, ending stage early",
                flush=True,
            )
            break
        step_flips = 0
        for name, fac in facs.items():
            rec = per_linear[name]
            rec["steps"] += 1
            su = torch.sign(fac.u_latent.detach())
            sv = torch.sign(fac.v_latent.detach())
            fu = int((su != prev_sign[name][0]).sum())
            fv = int((sv != prev_sign[name][1]).sum())
            prev_sign[name] = (su, sv)
            rec["flips_u"] += fu
            rec["flips_v"] += fv
            rec["per_step_flips"].append(fu + fv)
            step_flips += fu + fv
            if (fu or fv) and rec["first_flip_step"] is None:
                rec["first_flip_step"] = step
        stats["steps"] = step + 1
        if step_flips:
            stats.setdefault("flip_steps", []).append(step)
    residual_after = _residual()
    with torch.no_grad():
        for name, fac in facs.items():
            rec = per_linear[name]
            cur = _fac_params(fac)
            start = entry[name]
            rec["u_rel_move"] = _rel_move(cur[0], start[0])
            rec["v_rel_move"] = _rel_move(cur[1], start[1])
            rec["s1_rel_move"] = _rel_move(cur[2], start[2])
            rec["s2_rel_move"] = _rel_move(cur[3], start[3])
            # The stage's objective evidence. ``zero_crossings`` is NOT it:
            # it is a restatement of the flip counters (below), so on its own
            # the record could not tell an effective stage from an inert one.
            rec["residual_before"] = residual_before
            rec["residual_after"] = residual_after
            rec["gradient_norm"] = first_grad_norm
            rec["zero_crossings"] = bool(rec["flips_u"] == 0 and rec["flips_v"] == 0)
            # Declared inertia (qk-tune-loss-scale, requirement 3): this row's own
            # movement measured against the coarsest parameter dtype's relative
            # resolution. A row that cannot say whether it moved is unreadable as
            # evidence, so the key is always present once the row ran.
            moves = (
                rec["u_rel_move"],
                rec["v_rel_move"],
                rec["s1_rel_move"],
                rec["s2_rel_move"],
            )
            # A row that moved less than the dtype resolution (including a row a
            # vacuous stage restored to its entry value) is inert; a row that
            # cleared it is not.
            rec["inert"] = bool(max(moves) < floor_rel)
    if stats["per_linear"]:
        stats["inert"] = bool(all(rec["inert"] for rec in stats["per_linear"].values()))
    stats["skipped_steps"] = skipped
    if skipped:
        print(
            f"[tune_latent_ste:{label}] skipped {skipped}/{total} non-finite-loss steps", flush=True
        )
    reporter.complete(steps_done)  # ste: the count actually run, never the budget
    _stamp()
    return stats


def _fac_params(fac: FactorizedLinear) -> tuple[torch.Tensor, ...]:
    """The paper's joint refinement parameter set for one linear (Eq. 10)."""
    return (fac.u_latent, fac.v_latent, fac.s1, fac.s2)


def _rel_move(current: torch.Tensor, start: torch.Tensor) -> float:
    """Relative L2 movement of a parameter tensor versus its entry value."""
    den = float(start.pow(2).sum())
    if den <= 0.0:
        return 0.0
    return float((current.detach() - start).pow(2).sum().sqrt()) / float(np.sqrt(den))


def _discrete_flip_pass(
    block: nn.Module,
    linear: nn.Linear,
    name: str,
    fac: FactorizedLinear,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    *,
    budget: int,
    factor_mode: str,
) -> dict:
    """Bounded discrete sign-flip pass (qk-fix-ste-sign-adaptation alt. 3).

    The paper has no discrete flip pass, so this is an EXPLICIT DEVIATION:
    it exists only to serve a linear whose continuous refinement produced no
    sign crossing (the measured geometry: warm-start latents sit far from
    the decision boundary, so 4,096 STE steps move them 5-33% without
    crossing zero). Each proposal flips ONE factor entry of the linear under
    refinement and is accepted only when the block-reconstruction objective
    ``||block_tuned(x_b) - y_fp||**2`` DECREASES with every other block
    parameter held at its current value (scenario
    ``flip_acceptance_uses_block_reconstruction_objective``).

    The sweep order is deterministic — candidate entries sorted by ascending
    ``|latent|`` (smallest margin first), ties broken by (factor, row, col)
    — so the same seed and sweep produce the identical flip sequence.
    ``budget`` bounds the proposals PER FACTOR; 0 disables the pass.

    Returns a record naming the deviation and the per-factor acceptance
    counts.
    """
    record = {
        "linear": name,
        "mechanism": "discrete_flip_pass",
        "deviation": "paper has no discrete sign-flip pass (design alt. 3)",
        "budget_per_factor": int(budget),
        "accepted_u": 0,
        "accepted_v": 0,
        "proposals": 0,
        "objective_before": None,
        "objective_after": None,
    }
    if budget <= 0:
        return record
    with torch.no_grad():
        objective = float((_forward_chunked(block, x_b) - y_b).pow(2).mean())
    record["objective_before"] = objective
    order: list[tuple[str, int, int]] = []
    for tag, latent in (("u", fac.u_latent), ("v", fac.v_latent)):
        flat = latent.detach().abs().reshape(-1)
        k = min(int(budget), int(flat.numel()))
        if k <= 0:
            continue
        # Ascending |latent|, ties broken by first-seen index order (row-major
        # within a factor, u's entries ahead of v's): torch.argsort with
        # stable=True keeps that order, which is what makes the sweep
        # backend-independent.
        picked = torch.argsort(flat, stable=True)[:k].tolist()
        rows, cols = latent.shape
        for flat_idx in picked:
            order.append((tag, flat_idx // cols, flat_idx % cols))
    for tag, i, j in order:
        latent = fac.u_latent if tag == "u" else fac.v_latent
        with torch.no_grad():
            latent.data[i, j] = -latent.data[i, j]
        _set_linear(linear, fac, factor_mode=factor_mode)
        with torch.no_grad():
            candidate = float((_forward_chunked(block, x_b) - y_b).pow(2).mean())
        record["proposals"] += 1
        if candidate < objective:
            objective = candidate
            record["accepted_u" if tag == "u" else "accepted_v"] += 1
        else:
            with torch.no_grad():
                latent.data[i, j] = -latent.data[i, j]
            _set_linear(linear, fac, factor_mode=factor_mode)
    record["objective_after"] = objective
    return record


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


def _block_stats_slice(global_book, name_prefix: str | None, block: nn.Module, cfg: PTQConfig):
    """This block's share of the Phase-1 statistics book, in linear order.

    Phase 1 (Algorithm 1 lines 2-5) collects per-linear RobustDiag statistics
    once, over the PRISTINE FP model on the RAW calibration set, before any
    block is reconstructed. The block loop then consumes that pre-collected
    book: ``reconstruct_block`` receives the entries for its own scoped
    linears, aligned element-wise with ``quantizable_linears`` (declaration
    order), and never re-collects statistics from the compressed-prefix
    stream (line 9's tuning input is not the statistics source).

    A missing entry is a hard error, not a silent fallback to a local pass: a
    block quietly computing its own statistics would look identical in the
    record to one consuming the Phase-1 book.
    """
    if global_book is None:
        return None
    scope = _scope_filter(_quantizable_names(block), cfg.layer_scope)
    ordered = [name for name, _ in _linear_children(block) if name in scope]
    by_name = {entry.name: entry for entry in global_book}
    out = LayerStatsBook()
    for local in ordered:
        full = _full_name(name_prefix, local)
        entry = by_name.get(full)
        if entry is None and name_prefix is None:
            entry = by_name.get(local)
        if entry is None:
            raise ValueError(
                f"Phase-1 statistics book has no entry for {full!r}; the whole-model "
                "statistics pass and the block loop disagree about the layer inventory"
            )
        out.add(entry)
    return out


def reconstruct_block(
    block: nn.Module,
    calibration_batches,
    cfg: PTQConfig,
    *,
    x_b: torch.Tensor | None = None,
    y_b: torch.Tensor | None = None,
    x_fp: torch.Tensor | None = None,
    stats_book=None,
    recorder=None,
    layer_offset: int = 0,
    name_prefix: str | None = None,
    teacher_block: nn.Module | None = None,
    teacher_source: str | None = None,
    teacher_fingerprint: str | None = None,
    capture_index: int | None = None,
    cotangent: torch.Tensor | None = None,
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
        y_b: optional pre-captured teacher targets for ``x_b`` (the pristine
            fp block's output on the compressed-prefix stream).
        stats_book: Phase 1 statistics for THIS block's linears, aligned
            element-wise with ``quantizable_linears`` (built on demand when
            omitted — the inputs are first embedded to the block's float
            dtype so raw token-id batches never reach a ``nn.Linear``).
        cotangent: caller-supplied output gradient for the on-demand local
            statistics pass — the fail-closed guard's remedy for a bare block
            (spec: caller_supplied_head_for_bare_block). Ignored when
            ``stats_book`` is supplied.
        recorder: optional callable invoked as ``recorder(event, **info)``
            with events ``capture`` (x_b, y_b, teacher provenance),
            ``tune_fp`` (with the stage record), ``lb_admm``, ``ste`` (with
            the per-linear flip/scale-movement record), and
            ``tune_fp_vacuous_teacher`` on the fail-closed guard path.
        layer_offset: naming offset for multi-block models.
        name_prefix: module path of ``block`` inside the host model; layer
            artifact names are ``prefix.local`` so multi-block models with
            same-named children produce unique, host-addressable keys.
        teacher_block: frozen reference copy of this block, at pristine
            weights, used for the teacher capture when ``y_b`` is omitted
            (qk-fix-tunefp-teacher A1). Never mutated; when absent a local
            ``copy.deepcopy(block)`` is taken so the teacher never reads the
            live module's mutable state.
        teacher_source: provenance label for the teacher weights.
        teacher_fingerprint: sha256 of the teacher block's parameters.
        capture_index: which block this teacher capture belongs to.

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
        # Teacher capture (Algorithm 1 lines 9-10):
        # ``Y_b = B^FP_b(X_b)`` — the PRISTINE fp block applied to the
        # compressed-prefix stream. The weights come from a frozen reference
        # copy, never the live module: by block N the live model carries
        # earlier blocks written back quantized, and reading live state would
        # silently turn the teacher into a quantized-prefix teacher with
        # every gate still green. The capture is detached and under
        # ``no_grad`` (memory-only, A2 — never persisted).
        teacher = teacher_block if teacher_block is not None else copy.deepcopy(block)
        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            y_b = _forward_chunked(teacher, x_b)
        if teacher_fingerprint is None:
            teacher_fingerprint = _weights_fingerprint(teacher)
        teacher_source = teacher_source or "frozen_reference_copy"
        teacher.train(was_training)
        if teacher_block is None:
            del teacher
    else:
        teacher_source = teacher_source or "caller_supplied_frozen_capture"
    # Boundary finiteness check (measured): non-finite inputs
    # here mean the PREVIOUS block's quantized artifact is broken (its
    # forward produces inf/NaN), and every statistic, solve, and export
    # downstream would inherit the poison. Fail at the exact boundary,
    # naming the block — never let it surface later as a cryptic solver
    # error or a silently garbage artifact.
    # Both streams: x_fp is the fp reference reading, so a non-finite value there
    # is a broken reference forward rather than a poisoned quantized prefix, and
    # the message should say which.
    for tag, t in (("x_b", x_b), ("y_b", y_b), ("x_fp", x_fp)):
        if t is not None and not torch.isfinite(t).all():
            n_bad = int((~torch.isfinite(t)).sum())
            raise ValueError(
                f"block {name_prefix or layer_offset}: captured {tag} has "
                f"{n_bad}/{t.numel()} non-finite entries — the quantized "
                "prefix upstream is broken; not solving this block on "
                "poisoned data"
            )
    if recorder is not None:
        recorder(
            "capture",
            x_b=x_b.detach().clone(),
            y_b=y_b.detach().clone(),
            # The fp-stream activation at this block. Recorded so the lockstep
            # advance is verifiable from the artifact -- a capture that
            # carries only the compressed reading cannot distinguish "both streams
            # advanced together" from "only the student's did".
            x_fp=x_fp.detach().clone() if x_fp is not None else None,
            # Stream provenance, by construction rather than by inspection: x_b is
            # the compressed prefix's reading (the student's input) and y_b is the
            # fp reference's (the teacher target). The dual-stream construction
            # exists precisely to make y_b come from the fp stream, so these labels
            # are what the verdict's provenance condition reads.
            x_b_stream="compressed",
            y_b_stream="fp",
            teacher_source=teacher_source,
            teacher_fingerprint=teacher_fingerprint,
            capture_index=capture_index,
        )

    # --- Step 1: TuneFP (line 12) is fired INSIDE the factorization loop
    # below, at the [NQ] Figure 3 placement (after Q/K/V, before O/G/U/D).
    # The fp block's bias may move there (it shapes the teacher target), but
    # the factorized replacement reverts to the pre-TuneFP bias: the artifact
    # has no bias field, so the quantized layer must carry the host's own
    # bias for the round trip to close.
    original_bias = {
        name: (m.bias.detach().clone() if m.bias is not None else None)
        for name, m in quantizable_linears
    }

    # --- Step 2: per-linear-layer LB-ADMM (lines 14-17). Statistics for
    # the block come from Phase 1 bookkeeping, or a local pass when absent.
    # The local pass must never feed raw token-id batches into the block's
    # float Linears (the mat1/mat2 dtype crash of the first wiring): the
    # captured float inputs x_b ARE the block's calibration activations, so
    # they are the statistics source; the embedding-adapter path is the
    # fallback for direct reconstruct_block calls with token-id batches.
    if stats_book is None:
        # Fail-closed:
        # a bare block carries no scalar loss, so collection refuses rather
        # than invent one. Measured: the natural candidate — the
        # teacher reconstruction loss ``2*(block(x_b) - y_b)/N`` — is
        # IDENTICALLY ZERO here, because at collection time the live block is
        # still pristine and the teacher is its frozen deepcopy (same
        # function, same input). Zero cotangent means E[g^2] clamps to _TINY
        # and the whitened solve trips the magnitude divergence gate on every
        # fixture. The paper's gradient source is the task loss itself
        # (App. C :3011, CE through the model) — a real solve supplies the
        # Phase-1 book; direct callers supply a documented loss-bearing
        # cotangent (the spec's ``0.5*||z - t||**2`` factory shape).
        if x_b.dtype == torch.float64:
            # The float64 reference path (executable-spec fixtures): one
            # whole-batch backward, bit-identical to the landed behavior.
            stats_book = collect_stats(block, [x_b], cotangent=cotangent)
        elif x_b.dtype.is_floating_point:
            # Lower-precision (real-model) float activations: row-chunked
            # backwards through the block — the statistic definition is
            # unchanged (raw f64 grads accumulated across chunks, squared
            # once per batch), but a whole-batch backward retains attention
            # graphs for every row at once (~15GB on a 28-layer bf16 trunk)
            # and OOMs under container caps.
            stats_book = collect_stats(block, [x_b], chunk_rows=4, cotangent=cotangent)
        else:
            stats_book = collect_stats(_input_adapter(block, batches), [x_b], cotangent=cotangent)
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
    polish = (
        cfg.factor_mode == "binary" and cfg.ablation != "untuned-signs" and not cfg.paper_faithful
    )
    refine = cfg.ablation != "untuned-signs"
    facs: dict[str, FactorizedLinear] = {}
    linears_by_name = dict(quantizable_linears)
    # --- Figure 3 placement: the figure depicts a PER-TARGET-LINEAR cycle —
    # tune the still-FP remainder, factorize the target, refine the target —
    # not one pass per block. The old pre-group/remainder `split` machinery
    # existed only to serve the falsified once-per-block reading and is gone.
    #
    # Target 1 is factorized first, WITHOUT a preceding TuneFP. A pass before
    # the first factorization has nothing factorized, so the student block IS
    # the pristine teacher block on the same ``x_b``: the objective is
    # identically zero and so is its gradient — exactly the vacuous shape this
    # placement removes. Every later target gets TuneFP(still-FP remainder) ->
    # factorize -> STE, so TuneFP fires N-1 times per block and STE fires N
    # times (target 1 included).
    # --- TuneLatentSTE budget: the refinement unit is the LINEAR — Step-3 prose
    # names "the current target linear layer" (singular), Eq. 10's optimization
    # variables are ONE linear's (U, V, s1, s2), and App. C attaches the
    # factorized stage's 8 epochs to that per-linear refinement. Every scoped
    # linear therefore consumes the FULL per-linear budget `epochs x (n // batch)`
    # on the just-factorized linear per Figure 3's right panel; there is no
    # block-pooled split. The schedule record carries each linear's full
    # scheduled steps.
    n_rows = int(x_b.shape[0])
    bs_latent = max(1, min(cfg.stage_latent.batch_size, n_rows))
    total_ste = max(cfg.stage_latent.epochs * (n_rows // bs_latent or 1), 1)
    ste_records: dict[str, dict] = {}
    tune_fp_record: dict | None = None
    _tune_pairs = list(zip(stats_book, quantizable_linears, strict=False))
    terminal_loss_weight, loss_weight_source, missing_terminal = _resolve_loss_weight(
        cfg, _tune_pairs
    )
    # --- Normalize the terminal weight to unit mean BEFORE either stage consumes
    # it (qk-tune-loss-scale, requirement 1). The robust diagonal's relative
    # per-feature structure is the statistical content; its absolute scale is a
    # free numerical degree of freedom, and at raw scale (~1e-10) the weighted
    # loss sits ~1e10x under the residual, every Adam step rounds back on the
    # parameter dtype, and both stages run full budgets that move nothing (v5).
    # The normalization belongs HERE, not inside a stage: a stage that normalized
    # its own input could never observe the un-normalized regime the floor guard
    # exists to catch. `_resolve_loss_weight` keeps returning the raw diagonal --
    # that is the pre-normalization quantity the record must be able to name.
    loss_weight_mean_before: float | None = None
    loss_weight_mean_after: float | None = None
    if terminal_loss_weight is not None:
        terminal_loss_weight, loss_weight_mean_before, loss_weight_mean_after = (
            _normalize_weight_to_unit_mean(terminal_loss_weight)
        )
    if missing_terminal is not None and recorder is not None:
        recorder(
            TUNE_LOSS_FALLBACK_EVENT,
            block=name_prefix or f"offset{layer_offset}",
            missing_canonical_layer=missing_terminal,
            requested_mode=cfg.block_loss_weight,
            reason="absent_canonical_layer",
        )
    # --- Solver arm for THIS block's balance stage. ``magnitude_balance`` is
    # part of the algorithm on EITHER backend, and it refits the solve that
    # just ran, so
    # it must resolve the arm exactly as ``lb_admm`` resolves its own solve:
    # SOLVER_BACKEND over cfg.solver_backend, an explicit QUANTIBBLER_DEVICE
    # honoured, implicit torch on a box with no accelerator refused. Resolving
    # it HERE is the point -- ``magnitude_balance``'s own default is
    # DEFAULT_SOLVER_BACKEND, so a call site that omits it executes the balance
    # on torch regardless of the knobs: unrunnable on an accelerator-less host,
    # and a numpy-arm run silently mislabelled about the arm its balance used.
    # Env is process-global, so one resolution per block covers every layer.
    balance_backend, _balance_source = solver_backend_provenance(cfg.solver_backend)
    for idx, (entry, (name, linear)) in enumerate(
        zip(stats_book, quantizable_linears, strict=False)
    ):
        # --- Step 1 (line 12), per-target cycle: TuneFP tunes the still-FP
        # remainder BEFORE this target is factorized. Its tunable set is
        # ``quantizable_linears[idx:]`` — every linear not yet factorized,
        # this target included — and ``factorized_in_block`` is the number
        # already frozen, so the objective is nonzero by construction.
        if idx >= 1:
            remainder = [nm for nm, _ in quantizable_linears[idx:]]
            label = name_prefix or f"offset{layer_offset}"
            tune_fp_record = _tune_fp(
                block,
                x_b,
                y_b,
                cfg,
                cfg.stage_pre.batch_size,
                label=label,
                tune_names=remainder,
                factorized_in_block=idx,
                recorder=recorder,
                capture_index=capture_index,
                teacher_fingerprint=teacher_fingerprint,
                teacher_source=teacher_source,
                loss_weight=terminal_loss_weight,
                loss_weight_source=loss_weight_source,
                loss_weight_mean_before=loss_weight_mean_before,
                loss_weight_mean_after=loss_weight_mean_after,
            )
            _note_block_progress(0, f"{name_prefix or 'block'}/tune_fp")
            if recorder is not None:
                recorder("tune_fp", **tune_fp_record)

        w = linear.weight.detach().to(torch.float64).cpu().numpy()
        d_in_t, d_out_t = robust_diag(entry.d_in_raw, entry.d_out_raw, tau=cfg.tau, gamma=cfg.gamma)
        # [NQ] B.1 Eq. 26 folds the preconditioner as a square root on each side:
        #   W~ = sqrt(D~_out) W sqrt(D~_in)
        # Folding the raw diagonals charges the scales' magnitude twice across the
        # factor pair. The sqrt fold (Arm B) beats the raw fold on every probed
        # layer (mean delta +0.0825), which is why this ships and the shipped raw
        # fold is retired. Both the factorization and the balance below
        # consume these values, so the fold and its un-fold stay consistent by
        # construction rather than by two matching edits.
        d_in_t = np.sqrt(d_in_t)
        d_out_t = np.sqrt(d_out_t)
        w_tilde = (d_out_t[:, None] * w) * d_in_t[None, :]
        w_tilde = w_tilde.astype(dt)
        if cfg.solver == "svd":
            # Ablation-measured path: the truncated-SVD
            # warm start itself, fed through the identical balance+polish.
            # Lands within ~0.003 cos of the full ADMM run on production
            # layers while skipping 66-83 s/layer of iterations at rank
            # 512. Gate + polish + all downstream stages unchanged.
            u_svd, s_svd, vt_svd = np.linalg.svd(w_tilde, full_matrices=False)
            r = min(cfg.rank, len(s_svd))
            p_u = np.ascontiguousarray(u_svd[:, :r] * np.sqrt(s_svd[:r]))
            p_v = np.ascontiguousarray(vt_svd[:r, :].T * np.sqrt(s_svd[:r]))
            res = None  # no iterate to report; the probe row records that
        else:
            res = lb_admm(w_tilde, cfg)
            p_u, p_v = res.P_u, res.P_v
        bal = magnitude_balance(p_u, p_v, d_out_t, d_in_t, target=w, backend=balance_backend)
        # Post-balance divergence gate: a diverged
        # layer must abort here, before it poisons this block's refinement
        # and every later block's calibration inputs.
        check_layer_magnitude(
            name,
            to_host_np(bal.U),
            to_host_np(bal.V),
            to_host_np(bal.s1),
            to_host_np(bal.s2),
            w,
            cfg,
        )
        _record_solver_probe(name, w, bal, res, cfg)
        # Greedy bit-flip polish: the
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
                to_host_np(bal.Ub),
                to_host_np(bal.Vb),
                to_host_np(bal.s1),
                to_host_np(bal.s2),
                rounds=cfg.polish_rounds,
            )
            # Exact refold: latents = polished bits themselves (sign(latent) is
            # then a fixed point — STE cannot and need not move them), scales
            # carry the magnitude. cos(latent, bal.U) is irrelevant; the
            # artifact is (bits, scales), and those are the polished ones.
            _dev = linear.weight.device
            u0 = torch.from_numpy(np.ascontiguousarray(ub_p)).to(_dev)
            v0 = torch.from_numpy(np.ascontiguousarray(vb_p)).to(_dev)
        else:
            # No discrete stage at all: the continuous balance latents ARE the
            # factorization. In fp mode the forward never signs them (D-2); in
            # untuned-signs mode they are exported as their signs with the Eq. 8
            # scale fold and zero refinement (D-3).
            _dev = linear.weight.device
            u0 = torch.from_numpy(np.ascontiguousarray(to_host_np(bal.U))).to(_dev)
            v0 = torch.from_numpy(np.ascontiguousarray(to_host_np(bal.V))).to(_dev)
            s1_p = to_host_np(bal.s1)
            s2_p = to_host_np(bal.s2)
        s1_np = np.ascontiguousarray(s1_p)
        s2_np = np.ascontiguousarray(s2_p)
        bias0 = original_bias[name]
        fac = FactorizedLinear(
            u0,
            v0,
            bias0
            if bias0 is not None
            else torch.zeros(linear.weight.shape[0], device=linear.weight.device),
            factor_mode=cfg.factor_mode,
        )
        # Scales initialized from Eq. 8 (with the latent scale folded in).
        with torch.no_grad():
            # copy_ requires matching devices; the numpy round trip through the CPU solver
            # is deliberate (the solver is NumPy), so the return leg lands on the latents' device.
            fac.s1.copy_(torch.from_numpy(s1_np).to(fac.s1.device))
            fac.s2.copy_(torch.from_numpy(s2_np).to(fac.s2.device))
        facs[name] = fac
        # Freeze this linear's factorized weight into the module NOW: every
        # later stage in the block (the per-layer STE below, the TuneFP pass,
        # the next layer's statistics-free solve) must see the quantized
        # version of the layers already factorized. The pre-TuneFP bias is
        # restored first — TuneFP's fp-bias move is not part of the artifact
        # (no bias field in qpk1).
        if original_bias[name] is not None:
            with torch.no_grad():
                linear.bias.copy_(original_bias[name].to(linear.bias.dtype))
        _set_linear(linear, fac, factor_mode=cfg.factor_mode)

        # --- Step 3 (line 19): per-layer TuneLatentSTE on the
        # just-factorized linear, with the FULL per-linear App. C budget.
        # Every other block parameter is held at its current value: the
        # module carries the frozen weights of the previously factorized
        # layers and the untouched fp weights of the later ones.
        if refine:
            rec = _tune_latent_ste(
                block,
                {name: fac},
                x_b,
                y_b,
                cfg,
                cfg.stage_latent.batch_size,
                steps=total_ste,
                label=name,
                loss_weight=terminal_loss_weight,
                loss_weight_source=loss_weight_source,
                loss_weight_mean_before=loss_weight_mean_before,
                loss_weight_mean_after=loss_weight_mean_after,
            )
            per = rec["per_linear"][name]
            if cfg.paper_faithful:
                # The paper's Stage 2 -> 3 mechanism, to the letter: STE on
                # the continuous latents for the full budget, then sign+pack
                # whatever signs result (Algorithm 1 lines 20-23). A
                # zero-crossing outcome IS the paper's outcome — a
                # measurement, not a re-scope — so neither the bounded
                # discrete flip pass nor the scale-only declaration fire on
                # this path. The measured count travels in the record.
                per["declared"] = "paper_ste"
            elif per["zero_crossings"]:
                # Diagnostic path only: no continuous sign crossing, so fall
                # back to the bounded discrete flip pass (design alt. 3, an
                # explicit deviation), and if that also accepts nothing,
                # declare the linear scale-only under the greppable re-scope
                # marker. A silent zero-flip completion is impossible here.
                flip = _discrete_flip_pass(
                    block,
                    linear,
                    name,
                    fac,
                    x_b,
                    y_b,
                    budget=cfg.ste_flip_pass_budget,
                    factor_mode=cfg.factor_mode,
                )
                per["flip_pass"] = flip
                if flip["accepted_u"] + flip["accepted_v"] > 0:
                    per["declared"] = "discrete_flip_pass"
                else:
                    per["declared"] = "scale_only"
                    per["rescope_marker"] = STE_RESCOPE_MARKER
            else:
                per["declared"] = "paper_ste"
            per["scheduled_steps"] = int(total_ste)
            _set_linear(linear, fac, factor_mode=cfg.factor_mode)
            ste_records[name] = per

    _note_block_progress(0, f"{name_prefix or 'block'}/lb_admm")
    if recorder is not None:
        # The polish count travels in the existing event: an ablation's log must
        # PROVE zero polish invocations, and adding a new event name would
        # change the stage sequence every existing consumer asserts.
        recorder("lb_admm", polish_rounds=cfg.polish_rounds if polish else 0)

    # --- Step 3 record (line 19), emitted once per block after the
    # factorization loop so the stage sequence every existing consumer
    # asserts (capture -> tune_fp -> lb_admm -> ste) is unchanged.
    _note_block_progress(0, f"{name_prefix or 'block'}/stats_done")
    if refine:
        if tune_fp_record is None:
            # No TuneFP fired. Two distinct causes, named rather than merged:
            # a block with fewer than two quantizable linears has no second
            # target for the per-target cycle (legitimate, and run-level
            # execution counting means a per-block skip cannot false-deny a
            # faithful run), while a statistics book shorter than the linear
            # list means the loop was TRUNCATED before the cycle completed.
            truncated = len(facs) < len(quantizable_linears)
            tune_fp_record = {
                "stage": "tune_fp",
                "label": name_prefix or f"offset{layer_offset}",
                "skipped": (
                    "factorization_loop_truncated_before_placement"
                    if truncated
                    else "block_carries_no_second_target"
                ),
                "tunable": [],
                "factorized_in_block": len(facs),
                "capture_index": capture_index,
                "teacher_source": teacher_source,
                "teacher_fingerprint": teacher_fingerprint,
                "marker": TUNE_FP_LOSS_MARKER,
                "steps": 0,
                "vacuous": False,
                # A skip is not an execution: inertness is unknown (None), and the
                # weighting scale is carried so the row still names its regime
                # (qk-tune-loss-scale).
                "inert": None,
                "loss_weight": ("uniform" if terminal_loss_weight is None else "terminal_grad"),
                "loss_weight_source": loss_weight_source,
                "loss_weight_mean_before": loss_weight_mean_before,
                "loss_weight_mean_after": loss_weight_mean_after,
                "loss_weight_normalization": (
                    LOSS_WEIGHT_MEAN_NORMALIZATION if loss_weight_mean_after is not None else None
                ),
                "gradient_floor": None,
                "floor_optimizer_eps": None,
                "floor_dtype_rel_resolution": None,
                "floor_param_dtype": None,
            }
            if recorder is not None:
                recorder("tune_fp", **tune_fp_record)
        _note_block_progress(0, f"{name_prefix or 'block'}/ste")
        if recorder is not None:
            recorder(
                "ste",
                # Weighting actually used by the stage, on the record.
                loss_weight=("uniform" if terminal_loss_weight is None else "terminal_grad"),
                loss_weight_source=loss_weight_source,
                loss_weight_mean_before=loss_weight_mean_before,
                loss_weight_mean_after=loss_weight_mean_after,
                loss_weight_normalization=(
                    LOSS_WEIGHT_MEAN_NORMALIZATION if loss_weight_mean_after is not None else None
                ),
                per_linear=ste_records,
                total_steps=total_ste,
                teacher_source=teacher_source,
                teacher_fingerprint=teacher_fingerprint,
                capture_index=capture_index,
                rescope_marker=STE_RESCOPE_MARKER,
            )

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

    if recorder is not None:
        # Stage map (Algorithm 1 lines 20-23): recorded so the faithful-run
        # record can show that sign extraction + packing executed for every
        # factorized linear (and that an fp measurement extracted none).
        recorder(
            "sign_extract_pack",
            layers=sorted(facs),
            extracted=len(layers),
            factor_mode=cfg.factor_mode,
        )

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
    explosion by mechanism (measured: block 0 ~250x on the median and
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
    teacher_model: nn.Module | None = None,
    stats_book=None,
) -> list[BlockReconResult]:
    """Phase 2 over every direct child block of ``model``, sequentially.

    Block ``b`` sees ``X_b`` = calibration data through the already
    quantized prefix (earlier blocks run their frozen-binary replacements;
    Algorithm 1 line 9) and ``Y_b`` = the PRISTINE fp block's output on those
    inputs (line 10). Only blocks that contain a quantizable ``nn.Linear``
    (A-9 scope: embeddings, heads, and everything under an embedding module
    are fp) participate; layer artifacts are named by the host model's module
    paths so quantize_model can address them.

    ``teacher_model`` is the frozen reference copy of the whole model at
    pristine weights (qk-fix-tunefp-teacher A1): each block's teacher output
    is read from THAT copy, never from the live block — by block N the live
    model already carries earlier blocks written back quantized, so a teacher
    read off live state would silently be a quantized-prefix teacher. When
    the caller supplies none (a direct ``reconstruct_model`` call), one is
    deep-copied here with gradients disabled. Each block's teacher capture
    records the teacher weights' sha256 fingerprint and the block index, so
    provenance is provable from the artifact alone.

    Checkpointing (``checkpoint_dir`` set): each finished block's layer
    states are persisted losslessly (int8 bits, float64 scales) and a
    partially-complete checkpoint dir is resumed block-wise — boundary
    activations are rebuilt by replaying the quantized prefix forward,
    which is bit-identical to a fresh run's prefix (the solve is
    deterministic; asserted by test_scn_checkpoint_resume). Resume
    refuses on config/code fingerprint mismatch, and both the save and the
    restore are driven by the LIVE block's expected quantizable names
    (qk-fix-fp-resume-guard), never by what an archive happened to contain.

    ``stats_book`` (optional) is the Phase-1 statistics book collected once
    over the pristine FP model on the raw calibration set (Algorithm 1 lines
    2-5). When supplied, each block's scoped slice of it is handed to
    ``reconstruct_block`` and no per-block statistics pass runs at all; the
    compressed-prefix stream stays the TUNING input (line 9), never the
    statistics source. When omitted (direct callers) the legacy per-block
    collection remains, so a caller that has no Phase-1 pass is not silently
    handed fabricated statistics.
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
            # The LIVE model is the identity these checkpoints must belong to:
            # a config-only fingerprint accepts another model's block archives
            # under the same PTQConfig. (The stage-journal
            # seeding that used to live here is api.quantize_model's now —
            # resume_stage_events — so the model is threaded there too.)
            ckpt.check_fingerprint(ckpt_dir, cfg, model)
        else:
            ckpt.write_fingerprint(ckpt_dir, cfg, model)
            # Fresh start: the journal on disk belongs to the fingerprint just
            # replaced (see checkpoints.reset_stage_log).
            ckpt.reset_stage_log(ckpt_dir)
        done_set = set(done)
    else:
        done_set = set()

    if teacher_model is None:
        teacher_model = copy.deepcopy(model)
        for param in teacher_model.parameters():
            param.requires_grad_(False)
    teacher_children = dict(_iter_named_children(teacher_model))

    batches = list(calibration_batches)
    # Map raw token-id batches once through the model's own input stem
    # (embedding modules), yielding the float activations every block sees.
    stem = _model_input_stem(model, batches)
    # Chunked like every other forward over the capture. The stem is the FIRST
    # full-width forward of a block and the one the dead run completed in bf16
    # before the MLP blew the cap, so its measured-to-fit evidence was
    # bf16-only -- and at fp32 its intermediates double like everything else.
    # `batches` come off the tokenizer on the CPU; `stem` may hold device-resident
    # weights. Derive the device from the stem's own parameters, and fall back to
    # leaving the batch alone when it has none -- a bare `next(...)` raises
    # StopIteration on a parameterless stem, which is exactly what the block-loop
    # fixtures are. No-op on CPU.
    _batches = torch.cat(batches, dim=0)
    _stem_param = next(iter(stem.parameters()), None)
    if _stem_param is not None:
        _batches = _batches.to(_stem_param.device)
    x_running = _forward_chunked(stem, _batches).detach()
    # The fp reference stream. Identical at the stem (so block 1's skip stays
    # the documented identity case), then advanced through the PRISTINE
    # children while ``x_running`` advances through the quantized ones. The
    # divergence between the two IS the lane.
    x_fp = x_running.clone()
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
        # Expected names for THIS block, from the LIVE model and through the
        # SAME predicate the solver scopes with: the checkpoint write and the
        # checkpoint restore are both driven by what
        # the block is OWED, never by what an archive contains.
        expected_names = _scope_filter(_quantizable_names(block), cfg.layer_scope)
        _note_block_progress(len(results) + 1, name, x_running)
        if i in done_set:
            # Resume: restore this block's solved layers bit-exactly,
            # write them into the host (the quantized prefix then
            # advances exactly as a fresh run's would), skip the solve.
            layers = ckpt.load_block_checkpoint(ckpt_dir, i, QuantizedLayerState)
            ckpt.apply_layer_states(block, name, layers, expected_names=expected_names)
            # Resume: recover the fp stream from the checkpoint rather than
            # recomputing it, so the resumed stream is exactly the activation the
            # interrupted solve had advanced to. A legacy checkpoint predating the
            # field has no y_b, and there the stream is recomputed through the
            # pristine copy -- never silently started from a wrong activation.
            _yb = ckpt.load_block_yb(ckpt_dir, i)
            if _yb is not None:
                x_fp = torch.as_tensor(_yb, dtype=x_fp.dtype, device=x_fp.device)
            else:
                # Legacy fallback: no stored target, so recompute the stream through
                # this block's pristine copy. ``teacher_block`` is resolved HERE and
                # not hoisted: the solve branch below resolves its own, and a hoisted
                # binding is what the checkpoint simplification removed. Referencing
                # the solve branch's name from this branch is an UnboundLocalError on
                # exactly the legacy-resume path, which has no other test.
                _tb = teacher_children.get(name)
                with torch.no_grad():
                    x_fp = _forward_chunked(_tb if _tb is not None else block, x_fp)
            results.append(BlockReconResult(layers))
            print(
                f"[checkpoint] block {i} ({name}): restored from checkpoint, skipping solve",
                flush=True,
            )
        else:
            # Teacher capture from the FROZEN reference copy (A1): the live
            # block is still pristine at this instant, but pinning the source
            # structurally means a later reordering of this loop cannot
            # silently turn the teacher into a quantized-prefix teacher.
            teacher_block = teacher_children.get(name)
            with torch.no_grad():
                # The teacher target is the fp-stream reading
                # Y_b = B^FP_b(x_fp), NOT the compressed stream. Feeding x_running
                # here made the teacher a function of the quantized prefix — the
                # defect this construction removes.
                y_b = _forward_chunked(teacher_block if teacher_block is not None else block, x_fp)
            result = reconstruct_block(
                block,
                batches,
                cfg,
                x_b=x_running.clone(),
                y_b=y_b,
                x_fp=x_fp.clone(),
                stats_book=_block_stats_slice(stats_book, name, block, cfg),
                recorder=recorder,
                name_prefix=name,
                teacher_block=teacher_block,
                teacher_source="frozen_reference_copy",
                teacher_fingerprint=_weights_fingerprint(teacher_block)
                if teacher_block is not None
                else None,
                capture_index=i,
            )
            # y_b already IS the next fp activation: Y_b = B^FP_b(x_fp) is the
            # same computation as x_fp_next = B^FP_b(x_fp). Assigning it makes the
            # invariant structural -- the stream block b+1's teacher reads is
            # by-construction the target block b recorded -- and avoids a second
            # pristine forward per block (and its chunk-boundary reassociation).
            x_fp = y_b
            layers = result.layers
            results.append(result)
            if ckpt_dir is not None and expected_names:
                ckpt.save_block_checkpoint(ckpt_dir, i, layers, y_b=y_b)
        # Prefix advances through the block with its factorized layers active
        # (frozen binaries already written back by reconstruct_block).
        with torch.no_grad():
            x_running = _forward_chunked(block, x_running)
        # Observer hook: the live host model is now exactly the quantized
        # prefix of ``len(results)`` blocks — the same object the
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
