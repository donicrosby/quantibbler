"""Phase 1 global calibration (tasks 2.1-2.3, [NQ] Step 2-1, Algorithm 1
lines 2-5).

``collect_stats`` runs calibration batches through a module once, capturing
per-linear-layer input/output activation statistics plus gradient statistics
(one backward pass), and ``robust_diag`` builds the robust diagonal
preconditioners D~_in / D~_out by clipping raw diagonal statistics to a
ceiling ``tau`` (assumption A-1) and applying Ledoit-Wolf-style shrinkage
(Eq. 3 of [NQ] Step 2-1): ``D <- (1-gamma) * clip(D, tau) + gamma *
mean(clip(D, tau))``.

Accumulators follow the caller's dtype; nothing here widens precision.
Zero-variance statistics yield finite, positive entries.
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

# Tail gate for the tau=None default policy: the
# derived q95 ceiling engages only when the raw vector is genuinely
# heavy-tailed (max exceeds 10x the q95 entry). Block-0-style embedding
# statistics exceed this by orders of magnitude; healthy concentrated
# vectors (where clipping measurably degrades the solve) stay untouched.
_TAU_TAIL_GATE = 10.0


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
    return x.detach().reshape(-1, x.shape[-1])


@torch.no_grad()
def _accumulate(rows: torch.Tensor, total: torch.Tensor, count: int) -> tuple[torch.Tensor, int]:
    """Stream one chunk into per-feature sum-of-squares plus a token count.

    D-7: the state is O(feature width) and the chunk's float64 copy dies with
    this call, so retention does not grow with the row or chunk count. The
    final per-feature mean of squares is ``total / count`` — the same quantity
    the reference path computes as ``x.pow(2).mean(dim=0)``, re-associated.
    """
    flat = rows.detach().reshape(-1, rows.shape[-1])
    # `total` is a capture accumulator created before any activation arrives, so
    # its device is only knowable here, at the first real input. Coerce at the
    # boundary rather than guessing at creation: the activation's device is the
    # only authority on where the sum must happen. A no-op on CPU.
    if torch.is_tensor(total):
        total = total.to(flat.device)
    return total + flat.pow(2).sum(dim=0), count + int(flat.shape[0])


# Per-token norm clipping happens at the robust 99.9th percentile, applied to
# the token's L2 norm BEFORE squaring ([NQ] App. B.1). Before, not after: clipping
# the squared vector would not bound a single token's contribution the same way,
# and the whole point is that one extreme token must not dominate a per-feature
# mean. Order statistic, not interpolated quantile -- the spec's "robust max" is
# the 99.9th value, and interpolation would return a norm between two samples.
_TOKEN_CLIP_QUANTILE = 0.999


def _clip_token_norms(t: torch.Tensor) -> torch.Tensor:
    """Scale each token vector down to the 99.9th-percentile L2 norm if above it.

    Derivation: [NQ] App. B.1 -- per-token norm clipping at the robust 99.9th
    percentile, before squaring. Tokens at or below the cap are untouched, so the
    transform is the identity on the bulk of the batch and only acts on its
    adversarial class (rare extreme tokens) -- which is the property its test
    asserts.
    """
    if t.numel() == 0:
        return t
    norms = t.norm(dim=-1, keepdim=True)
    flat = norms.reshape(-1)
    if flat.numel() < 2:
        return t
    cap = torch.quantile(flat.float(), _TOKEN_CLIP_QUANTILE, interpolation="lower")
    scale = torch.clamp(cap / norms.clamp_min(_TINY), max=1.0)
    return t * scale


def require_grad_or_name(loss: torch.Tensor) -> None:
    """Fail with a reason when ``loss`` cannot produce a gradient.

    Every caller-side ``loss.backward()`` in this pipeline routes through here.
    The bare-block path runs its backward outside ``collect_stats`` -- the caller
    computes ``head -> loss -> backward -> cotangent`` itself -- so a frozen or
    detached graph on that path cannot be caught inside the module, no matter
    where a guard is placed. One guard, one message, every caller.
    """
    if not loss.requires_grad:
        raise ValueError(
            "the loss you supplied does not require grad, so no gradient exists to "
            "measure: unfreeze the parameters, or supply a head whose loss depends "
            "on the module output."
        )


def cotangent_grad_sq_sum(g: torch.Tensor) -> torch.Tensor:
    """Per-feature sum of squared output-cotangents.

    Derivation: [NQ] Eq. 2, the K-FAC diagonal reading -- the output-side
    preconditioner is ``sqrt(E[g^2])`` with ``g`` the loss gradient at the layer's
    OUTPUT, from a real cross-entropy backward routed through the model head
    (App. C :3011).

    This is the SINGLE blessed implementation of that accumulation, and the body
    is the form Arm B measured, adapted to one device per the dtype
    ruling. Three non-negotiables, each of them a defect when dropped:

    * ``reshape(-1, g.shape[-1])`` -- module outputs are 3D, ``[batch, seq, d_out]``.
      Summing ``dim=0`` without reshaping sums over the BATCH axis and returns
      ``[seq, d_out]``, which is not a per-feature diagonal at all. The fixtures
      missed it because they fed 2D cotangents, so the test below asserts shape on
      a 3D input.
    * ``.detach()`` -- a hook-body accumulator that keeps autograd graph
      references is the retention class that killed three boxes.
    Runs at the caller's dtype. Widening precision inside the statistic would make
    the number incomparable with the run it came from; a caller that wants wider
    accumulation widens it at the call.
    """
    return g.detach().reshape(-1, g.shape[-1]).pow(2).sum(dim=0)


def _to_numpy_stat(t: torch.Tensor) -> np.ndarray:
    """Serialise a collected statistic to the f64 storage schema (boundary).

    Accumulation stays at caller dtype (the dtype ruling); only the recorded
    schema widens here. bf16/f16 have no NumPy counterpart — lift exactly
    those to f32 before the ``.numpy()`` hop (bf16's 8-bit mantissa widens
    to f32 exactly, so nothing is invented); f32/f64 pass through untouched.
    """
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return t.detach().cpu().numpy().astype(np.float64)


def collect_stats(
    module: nn.Module, calibration_batches, *, chunk_rows: int | None = None, cotangent=None
) -> LayerStatsBook:
    """Run calibration batches forward (and backward) once through ``module``.

    For every ``nn.Linear`` in module declaration order, records the mean
    square of each input/output feature plus the mean square of the
    output-gradient feature (backward pass). ``module`` may be a single
    layer or a whole model; gradients are enabled for the statistics pass
    and the module's parameter ``.grad`` fields are restored afterwards.

    ``chunk_rows`` splits each batch's backward into row chunks (the raw f64
    gradient is accumulated across chunks and squared once, preserving the
    per-batch statistic's definition). Default ``None`` = one backward per
    batch — the float64 reference path stays bit-identical to the landed
    single-pass behavior; lower-precision real-model paths opt in because a
    whole-batch backward retains attention graphs for every row at once and
    OOMs under container caps. Setting ``chunk_rows`` also switches the
    activation hooks to streamed statistics (D-7): per-feature sum-of-squares
    plus a token count, state O(feature width), instead of retaining every
    chunk's full float64 activation — the retention that OOM-killed the first
    rung-1 launch. ``chunk_rows=None`` keeps the landed list/concatenate path
    byte-for-byte.
    """
    linears = [(name, m) for name, m in module.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("collect_stats found no nn.Linear modules to calibrate")

    # The calibration batches come off the tokenizer on the CPU; the module's weights
    # may be on an accelerator. Move them ONCE here, before any use, so the fp-prefix
    # probe and the chunked capture forward are both device-true -- the auditor named
    # each boundary it crossed otherwise (calibration.py:225, then :232). The parameter
    # is `calibration_batches`; an earlier attempt wrote `batches` and hit NameError,
    # which the red-test gate caught before any commit or fire. Explicit None guard:
    # never truthiness on a tensor, never a bare next().
    _audit_ref = next(iter(module.parameters()), None)
    if _audit_ref is not None:
        calibration_batches = [b.to(_audit_ref.device) for b in calibration_batches]
    grad_enabled = torch.is_grad_enabled()
    book = LayerStatsBook()
    # D-7: the chunked path streams (sum of squares, token count) per feature
    # and drops each chunk's activation; the reference path retains and
    # concatenates exactly as landed.
    streaming = chunk_rows is not None
    feats_in: dict[str, list[torch.Tensor]] = {name: [] for name, _ in linears}
    # Accumulators are born on the weights' device, not the default (CPU) one:
    # the hook feeds them activations from wherever the model ran. A no-op on CPU.
    sum_sq_in = {
        name: torch.zeros(m.weight.shape[1], dtype=m.weight.dtype, device=m.weight.device)
        for name, m in linears
    }
    sum_sq_out = {
        name: torch.zeros(m.weight.shape[0], dtype=m.weight.dtype, device=m.weight.device)
        for name, m in linears
    }
    rows_in = {name: 0 for name, _ in linears}
    rows_out = {name: 0 for name, _ in linears}
    # d_out = E[g^2] from the REAL loss gradient at each layer's output ([NQ]
    # Eq. 2, K-FAC diagonal reading). The uniform-ones cotangent that used to
    # feed an INPUT-side gradient term is retired: it was loss-blind, and on a
    # benign unit-norm fixture it inflated d_in by ~6e6 (measured 2911800.35
    # against an analytic 0.5).
    grad_sq_out = {
        name: torch.zeros(m.weight.shape[0], dtype=m.weight.dtype, device=m.weight.device)
        for name, m in linears
    }

    def _accumulate_grad(key: str, g: torch.Tensor) -> None:
        rows = g.reshape(-1, g.shape[-1])
        grad_sq_out[key] = grad_sq_out[key] + cotangent_grad_sq_sum(rows).to(grad_sq_out[key].dtype)

    handles = []
    for name, m in linears:
        if streaming:

            def _pre_hook(mod, inp, key=name):  # noqa: ANN001
                sum_sq_in[key], rows_in[key] = _accumulate(
                    _clip_token_norms(inp[0]), sum_sq_in[key], rows_in[key]
                )

            def _post_hook(mod, inp, out, key=name):  # noqa: ANN001
                clipped = _clip_token_norms(out)
                sum_sq_out[key], rows_out[key] = _accumulate(
                    clipped, sum_sq_out[key], rows_out[key]
                )
                if out.requires_grad:
                    out.register_hook(lambda g, key=key: _accumulate_grad(key, g))

            handles.append(m.register_forward_pre_hook(_pre_hook))
            handles.append(m.register_forward_hook(_post_hook))
        else:

            def _ref_post_hook(mod, inp, out, key=name):  # noqa: ANN001
                # The reference path counts rows and captures the real output
                # gradient too. Wiring the gradient hook on the streaming branch
                # only left this path with rows_out == 0, which the assembly then
                # divided by -- and the f64 reference is the DEFAULT path, so the
                # gap was on the most-used branch, not the opt-in one.
                rows_out[key] += int(out.reshape(-1, out.shape[-1]).shape[0])
                if out.requires_grad:
                    out.register_hook(lambda g, key=key: _accumulate_grad(key, g))

            handles.append(m.register_forward_hook(_ref_post_hook))
            handles.append(
                m.register_forward_pre_hook(
                    lambda mod, inp, key=name: feats_in[key].append(
                        _reduce_tokens(_clip_token_norms(inp[0]))
                    )
                )
            )

    params = [p for _, m in linears for p in m.parameters()]
    had_grad = [bool(p.grad is not None) for p in params]
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    for p in params:
        p.grad = None
    # Three cases ([NQ] App. C :3011):
    #   1. whole-model collection -- cotangent is None and the module carries a
    #      head (``.head`` on the driver wrapper, ``.lm_head`` on a raw HF model),
    #      so the loss is next-token CE and the cotangent falls out of its
    #      backward, with the batch's input_ids as labels;
    #   2. bare block -- the caller supplies cotangent=, the gradient at the
    #      module output, and collect_stats runs backward from it. One entry
    #      shape, not two: a head= callable was retired in favour of this,
    #      because it would have grown a second way in beside the cotangent;
    #   3. neither -- fail closed.
    whole_model = cotangent is None
    detected_head = detected_trunk = None
    if whole_model:
        # Route from the TRUNK, not from the module's forward output. Applying the
        # head to a forward output that already ends in a head double-applies it:
        # on the driver wrapper that is [*, vocab] into Linear(hidden, vocab), a
        # RuntimeError at first collection on every block. Width-sniffing the
        # output to guess which case we are in is the fragile-collision class this
        # lane removes, so instead we compose the two handles the production
        # wrapper already maintains (tools/quantize_hf_model.py:200-205):
        # ``head(trunk(input_ids=x).last_hidden_state)``. A module that exposes a
        # head but no trunk fails detection honestly rather than crashing later.
        detected_head = getattr(module, "lm_head", None) or getattr(module, "head", None)
        stock_trunk = getattr(module, "_stock_trunk", None)
        # The trunk is exposed under one of three names: the driver wrapper's
        # ``_stock_trunk`` list, a raw HF model's ``.model``, or a bare ``.trunk``
        # on a model that composes the two halves itself.
        detected_trunk = (
            (stock_trunk[0] if stock_trunk else None)
            or getattr(module, "model", None)
            or getattr(module, "trunk", None)
        )
    if whole_model and (detected_head is None or detected_trunk is None):
        raise ValueError(
            "collect_stats cannot find a loss for d_out = E[g^2]: this module is "
            "neither a whole model exposing a trunk and a head separately (no "
            "reachable .head/.lm_head, or no .model/_stock_trunk) nor a bare block "
            "given cotangent= (the gradient at the module output). Pass "
            "cotangent=<tensor> for a bare block."
        )
    try:
        for batch in calibration_batches:
            if whole_model:
                ids = batch if torch.is_tensor(batch) else batch["input_ids"]
                hidden = detected_trunk(input_ids=ids).last_hidden_state
                logits = detected_head(hidden)
                # Autoregressive shift: position t predicts token t+1.
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1).to(logits.device),
                    ignore_index=-100,
                )
                require_grad_or_name(loss)
                loss.backward()
            else:
                out_t = module(batch)
                grad = cotangent if torch.is_tensor(cotangent) else torch.as_tensor(cotangent)
                grad = grad.to(device=out_t.device, dtype=out_t.dtype).reshape(out_t.shape)
                out_t.backward(gradient=grad)
    finally:
        torch.set_grad_enabled(grad_enabled)
        for h in handles:
            h.remove()
        for p, had, saved in zip(params, had_grad, saved_grads, strict=True):
            p.grad = saved if had else None

    for name, m in linears:
        din, dout = m.weight.shape[1], m.weight.shape[0]
        if rows_out[name] == 0:
            raise ValueError(f"layer {name!r} saw no output-gradient rows")
        if streaming:
            if rows_in[name] == 0 or rows_out[name] == 0:
                raise ValueError(f"streamed statistics for layer {name!r} saw no rows")
            # sum(x^2)/n per feature is the quantity the reference computes as
            # feat.pow(2).mean(dim=0); only the summation order differs.
            d_in_raw = _to_numpy_stat(sum_sq_in[name] / rows_in[name])
            d_out_raw = _to_numpy_stat(grad_sq_out[name] / rows_out[name])
        else:
            feat_in = torch.cat(feats_in[name], dim=0)
            d_in_raw = _to_numpy_stat(feat_in.pow(2).mean(dim=0))
            d_out_raw = _to_numpy_stat(grad_sq_out[name] / rows_out[name])
        d_in_raw = np.maximum(d_in_raw, _TINY)
        d_out_raw = np.maximum(d_out_raw, _TINY)
        if d_in_raw.shape != (din,) or d_out_raw.shape != (dout,):
            raise ValueError(f"statistics shape mismatch for layer {name!r}")
        book.add(LayerStatsEntry(name, d_in_raw, d_out_raw))
    return book


def robust_diag(d_in_raw, d_out_raw=None, *, tau=None, gamma=0.2):
    """Robust diagonal preconditioners D~_in / D~_out (Eq. 3, [NQ] Step 2-1).

    Entries are clipped to the ceiling tau **before** shrinkage
    (assumption A-1: tau is an absolute cap). An explicit numeric tau is
    used verbatim. ``None`` derives the cap per vector, tail-gated: the
    q95 percentile becomes the ceiling ONLY when the vector is heavy-tailed
    (max > _TAU_TAIL_GATE x q95); otherwise no clipping occurs, because a
    cap at max(D) is a no-op and clipping a concentrated vector measurably
    degrades healthy quantization (tiny-gate logit 0.795 -> 1.43; see the
    qk-divergence-guards change). Block-0-style heavy tails (raw embedding
    statistics) exceed the gate by orders of magnitude and
    are clipped. Then shrinkage toward the clipped mean with coefficient
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
            cap = float(tau)
        else:
            q95 = float(np.percentile(vec, 95))
            cap = q95 if float(vec.max()) > _TAU_TAIL_GATE * q95 else float("inf")
        vec = np.minimum(vec, cap)
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
