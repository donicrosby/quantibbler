"""TuneFP ([NQ] Step 1): T_pre steps of full-precision tuning of the block's
still-FP remainder against the cached pristine teacher.

Also carries the two greppable artifact markers and ``VacuousTeacherError``:
they sit in one documented block in the source, and ``_block_stage`` imports
the markers from here. The STE-only ``STE_RESCOPE_MARKER`` is declared in this
shared block rather than beside its user.

Split out of ``block_recon``; same code, same call graph.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ._factorized import _linear_children
from ._stats import LOSS_WEIGHT_MEAN_NORMALIZATION, _forward_chunked, _optimizer_gradient_floor
from .config import PTQConfig
from .progress import ProgressReporter

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
# ``STE_JOINT_SCOPE`` — the refinement's scope. TuneLatentSTE runs ONCE per
# block over the block's whole factorized set, so a firing's scope has to be
# readable from the record rather than inferred from the pipeline's shape: the
# same ``declared`` values mean something different when one joint pass over N
# linears produced them than when N per-linear passes did, and the rows alone
# cannot tell a reader which.
STE_JOINT_SCOPE = "block_joint"


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
        # The aggregate relative movement of the stage's tuned parameters.
        # `weight_delta_rel = None` has TWO causes and the row names which one
        # rather than overloading the bare `None`: the stage that never ran far
        # enough to have an answer (an empty tunable set, or a vacuous-teacher
        # early return), or a stage whose every tuned parameter was exactly 0.0
        # at entry -- a zero-init output projection, a LoRA B matrix, a gated
        # unit -- where the relative movement has no denominator at all. The
        # division is then undefined, not zero, so the field keeps this `None`
        # and never a confident 0.0 (design D12); `steps` separates the two.
        "weight_delta_rel": None,
        # Bias is a first-class tuned quantity (the container carries it), so
        # its movement is recorded separately from the aggregate: the weight
        # figure alone cannot say whether the stage's bias compensation is
        # real.
        #
        # `bias_delta_rel = None` names WHICH case left it without a figure
        # rather than overloading the bare `None`: `bias_tunables == 0` is the
        # no-bias-compartment case -- a bias-free model has nothing to move, so
        # `None` is the honest figure and never a confident 0.0 (design D12) --
        # while `bias_delta_rel is None` beside a NON-ZERO `bias_tunables` is
        # the stage that has no measurable figure: it never ran far enough to
        # have an answer (empty tunable set, or a vacuous-teacher early return),
        # or every bias baseline was exactly 0.0, so the relative movement has
        # no denominator at all -- the division is undefined, not zero, and
        # D12's "no value, never a confident 0.0" rules out recording 0.0. A
        # reader can therefore tell "the model is bias-free" from "the bias
        # figure was not computed" off the record alone, without consulting the
        # design doc; `steps` separates "never ran" from "ran over an all-zero
        # baseline".
        "bias_delta_rel": None,
        # The number of bias-bearing tunables the stage had: the discriminator
        # beside `bias_delta_rel`, filled once the tunable set is resolved (so
        # the empty-params early return carries it too). 0 means the model has
        # no bias compartment; an early-returned stage over bias-bearing
        # linears keeps its non-zero count.
        "bias_tunables": 0,
        "vacuous": False,
        # Inertness is DECLARED, not implied (qk-tune-loss-scale, requirement 3):
        # `None` means the stage did not run far enough to have an answer -- or
        # ran over a baseline with no denominator to compare a movement
        # against -- never "it was fine".
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
    bias_params = [m.bias for _, m in linears if m.bias is not None]
    # The count beside `bias_delta_rel`, recorded before any early return so an
    # unrun row and a run row are distinguishable by the record alone: 0 = the
    # model carries no bias to move; > 0 with a `None` movement figure = the
    # stage stopped before it could measure (see the field comment above).
    stats["bias_tunables"] = len(bias_params)
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
    bias_snapshot = [p.detach().clone() for p in bias_params]
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
    # An all-zero weight baseline (a zero-init output projection, a LoRA B
    # matrix, a gated unit) leaves the relative movement UNDEFINED -- 0.0 would
    # read as "the weights never moved" about weights the stage may have tuned
    # off zero. Record the `None` initial value instead (design D12: no value,
    # never a confident 0.0); `steps` says whether the stage ran. The bias arm
    # below is the same rule for the bias compartment.
    if den > 0.0:
        stats["weight_delta_rel"] = float(np.sqrt(num) / np.sqrt(den))
    if bias_params:
        bnum = bden = 0.0
        with torch.no_grad():
            for p, p0 in zip(bias_params, bias_snapshot, strict=True):
                bnum += float((p.detach() - p0).pow(2).sum())
                bden += float(p0.pow(2).sum())
        # An all-zero bias baseline (a zero-init bias projection) leaves the
        # relative movement UNDEFINED -- 0.0 would read as "the bias never
        # moved" about a bias the stage may have tuned off zero. Record the
        # `None` initial value instead (design D12: no value, never a confident
        # 0.0); `bias_tunables > 0` above still says the bias compartment
        # exists, and `steps` says whether the stage ran.
        if bden > 0.0:
            stats["bias_delta_rel"] = float(np.sqrt(bnum) / np.sqrt(bden))
    # Inertness is DECLARED, not implied (qk-tune-loss-scale, requirement 3): a
    # stage that ran steps and moved its parameters by less than the parameter
    # dtype's relative resolution did nothing measurable, and the record says so
    # instead of leaving a reader to reconcile `steps: 256` with
    # `weight_delta_rel: 0.0`. An all-zero baseline has no relative movement to
    # compare, so the verdict is unwitnessed rather than False and the field
    # stays `None` -- the same no-answer a stage that never ran carries.
    delta_rel = stats["weight_delta_rel"]
    stats["inert"] = (
        None if delta_rel is None else bool(stats["steps"] > 0 and delta_rel < floor_rel)
    )
    stats["skipped_steps"] = skipped
    if skipped:
        print(f"[tune_fp:{label}] skipped {skipped}/{total} non-finite-loss steps", flush=True)
    reporter.complete(steps_done)  # tunefp: the count actually run, never the budget
    _stamp()
    return stats
