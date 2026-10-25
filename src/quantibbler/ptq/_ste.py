"""TuneLatentSTE ([NQ] Step 3) and the bounded discrete sign-flip pass: joint
refinement of (U, V, s1, s2) through the straight-through estimator, with the
per-factor flip instrumentation the caller turns into an accepted flip or the
scale-only declaration.

Split out of ``block_recon``; same code, same call graph.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ._factorized import FactorizedLinear, _quantized_forward_block, _set_linear
from ._stats import LOSS_WEIGHT_MEAN_NORMALIZATION, _forward_chunked, _optimizer_gradient_floor
from .config import PTQConfig
from .progress import ProgressReporter


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

    Scope: ``facs`` IS the refinement unit. The caller hands over the block's
    ENTIRE factorized set once per block, after every factorization completes
    and before the stream swap, so ONE optimizer holds every ``(U, V, s1, s2)``
    and ONE loss is taken on the block output against ``y_b``. That is what lets
    the correction trade across linears — one linear's residual partially
    cancelled by another — which a per-target pass cannot express, because there
    the other linears are frozen at their already-degraded values and every
    linear is optimized as if its neighbours were exact. Algorithm 1 line 19's
    "the current target linear layer" prose is a DECLARED deviation from this
    reading, not the other way round.

    Budget: each call consumes the App. C factorized-stage regime
    ``epochs x (n // batch)`` at the stage lr with the cosine schedule, ONCE.
    There is no per-linear or block-pooled allocation on top of it. The block
    forward is the module as it stands, so every block parameter outside
    ``facs`` is held at its current value.

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
        exactly that linear. A joint call over several facs cannot split its
        wall time by linear -- the one measurement belongs to the whole set, and
        dividing it would report a share no clock observed -- so its rows report
        null and the call's own ``elapsed_s`` carries the stage's time.
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
            # Dead rank components: rank indices whose outer-product column
            # carries no sign, because one of the two factors' columns is
            # identically zero. Their sign is arbitrary and the latents they
            # index cannot move the artifact, so a row that cannot say how many
            # it carries cannot be read as evidence of an effective refinement.
            "dead_columns": 0,
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
        for name, rec in per_linear.items():
            rec["residual_before"] = residual_before
            rec["residual_after"] = residual_before
            rec["gradient_norm"] = 0.0
            rec["dead_columns"] = _dead_columns(facs[name])
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
            rec["dead_columns"] = _dead_columns(fac)
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


def _dead_columns(fac: FactorizedLinear) -> int:
    """Rank components of ``fac`` that contribute no sign to the artifact.

    A rank index is dead when either factor's COLUMN is identically zero: the
    outer-product column ``(s1 ⊙ f(U))[:, r] ⊗ (s2 ⊙ f(V))[:, r]`` is then zero
    for every (output, input) pair, so that component's sign is arbitrary and
    the latents indexing it cannot move the stored weight. Same notion as the
    dead column ``lb_admm`` floors in place of a sign, applied to the refinement
    record so a reader can tell an effective joint pass from one whose rank
    budget is partly inert by construction.
    """
    with torch.no_grad():
        u = fac.u_latent.detach()
        v = fac.v_latent.detach()
        dead_u = u.abs().sum(dim=0) == 0.0
        dead_v = v.abs().sum(dim=0) == 0.0
        return int((dead_u | dead_v).sum())


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
