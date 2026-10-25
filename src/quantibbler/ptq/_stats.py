"""Phase-1 statistics seam: the quantizable inventory and scope filter, this
block's slice of the Phase-1 book, the tuning-loss weight resolved from the
block's terminal-linear statistic, capture sizing, and the pristine-teacher
fingerprint.

Split out of ``block_recon``; same code, same call graph.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from ._factorized import _linear_children
from .calibration import LayerStatsBook, robust_diag
from .config import PTQConfig

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
