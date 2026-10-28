"""The block-wise reconstruction loop ([NQ] Algorithm 1 lines 8-24): capture,
TuneFP, precondition, LB-ADMM solve, magnitude balance, latent refinement and
sign extraction for ONE block, in place -- plus the block's input adapter.

The factorization loop carries the Figure 3 per-target cycle: per target, TuneFP
on the still-FP remainder, factorize the target, TuneLatentSTE on that one
target against the block-output objective ([NQ] Eq. 10), harden its signs, next
target.

Split out of ``block_recon``; same code, same call graph.
``reconstruct_block`` is re-exported from ``block_recon``, so every caller and
every test that patches ``block_recon.reconstruct_block`` keeps working.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from ._factorized import (
    BlockReconResult,
    FactorizedLinear,
    QuantizedLayerState,
    _linear_children,
    _set_linear,
    check_layer_magnitude,
    extract_and_pack,
)
from ._probe import _note_block_progress, _record_solver_probe
from ._stats import (
    LOSS_WEIGHT_MEAN_NORMALIZATION,
    TUNE_LOSS_FALLBACK_EVENT,
    _forward_chunked,
    _full_name,
    _normalize_weight_to_unit_mean,
    _quantizable_names,
    _resolve_loss_weight,
    _scope_filter,
    _weights_fingerprint,
)
from ._ste import _tune_latent_ste, harden_target
from ._tuning import STE_PER_TARGET_SCOPE, STE_RESCOPE_MARKER, TUNE_FP_LOSS_MARKER, _tune_fp
from .accumulation import (
    accumulation_np_dtype,
    accumulation_spec,
    accumulation_torch_dtype,
    torch_dtype,
)
from .bit_polish import greedy_polish
from .calibration import LayerStatsBook, collect_stats, robust_diag
from .config import PTQConfig, config_matches_paper_regime
from .lb_admm import lb_admm, magnitude_balance
from .solver_backend import solver_backend_provenance, to_host_np


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
            ``tune_fp`` (with the stage record), ``lb_admm``, ``ste`` (ONE
            record per target linear — its flip/movement/dead-column row plus
            the ``per_target`` scope marker and the target's name), and
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
    # Per-stage row time (qk-pipeline-dtype 5.5): every stage appends its own
    # wall seconds to its record, so the per-stage aggregate the report quotes
    # against the ~8h v6-class baseline is measured, never inferred.
    t_capture = time.perf_counter()
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
            elapsed_s=time.perf_counter() - t_capture,
        )

    # --- Step 1: TuneFP (line 12) is fired INSIDE the factorization loop
    # below, at the [NQ] Figure 3 placement (after Q/K/V, before O/G/U/D).
    # TuneFP's optimizer step moves the remainder linears' bias along with
    # their weights, and the container carries a per-linear bias, so the
    # factorized replacement takes the bias the tuning produced: [NQ] App. C
    # makes the tuned values the product of the stage, and the compensation
    # the stage computed must not be discarded at factorization.

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
        # The Phase-1 sum-of-squares runs at the DECLARED accumulation grade:
        # the statistics are what the preconditioner and the loss weight are
        # built from, and a bf16 accumulator would carry 7-bit mantissa loss
        # into both.
        _accum_torch = accumulation_torch_dtype(cfg.compute_dtype)
        if x_b.dtype == torch.float64:
            # The float64 reference path (executable-spec fixtures): one
            # whole-batch backward, bit-identical to the landed behavior.
            stats_book = collect_stats(block, [x_b], cotangent=cotangent, accum_dtype=_accum_torch)
        elif x_b.dtype.is_floating_point:
            # Lower-precision (real-model) float activations: row-chunked
            # backwards through the block — the statistic definition is
            # unchanged (raw f64 grads accumulated across chunks, squared
            # once per batch), but a whole-batch backward retains attention
            # graphs for every row at once (~15GB on a 28-layer bf16 trunk)
            # and OOMs under container caps.
            stats_book = collect_stats(
                block, [x_b], chunk_rows=4, cotangent=cotangent, accum_dtype=_accum_torch
            )
        else:
            stats_book = collect_stats(
                _input_adapter(block, batches),
                [x_b],
                cotangent=cotangent,
                accum_dtype=_accum_torch,
            )
        scope = {n for n, _ in quantizable_linears}
        if any(entry.name not in scope for entry in stats_book):
            aligned = LayerStatsBook()
            for entry in stats_book:
                if entry.name in scope:
                    aligned.add(entry)
            stats_book = aligned
    # ONE declared dtype for every tensor this stage creates. `np_compute_dtype`
    # is deliberately NOT consulted: it refuses bf16 by name (NumPy has no bf16
    # scalar type), and the torch dtype is the only mapping the pipeline's own
    # storage contract needs.
    _compute_dtype_torch = torch_dtype(cfg.compute_dtype)
    # Which stages this run carries (D-2/D-3). ``polish`` is the greedy bit-flip
    # stage: it is inherently discrete, so only a full binary run performs it,
    # and only when the run is NOT the paper regime — the regime test replaces
    # the self-asserted fidelity flag, so "is this the paper's run" is now read
    # off the declared config rather than claimed by the launcher.
    # ``refine`` is TuneLatentSTE — ``untuned-signs`` skips it by definition
    # (zero refinement is the ablation's content), while ``fp`` keeps it on
    # continuous latents so the ONLY difference from a binary run is the
    # discretization.
    paper_regime = config_matches_paper_regime(cfg)
    polish = cfg.factor_mode == "binary" and cfg.ablation != "untuned-signs" and not paper_regime
    refine = cfg.ablation != "untuned-signs"
    facs: dict[str, FactorizedLinear] = {}
    linears_by_name = dict(quantizable_linears)
    # --- Figure 3 placement: the figure and [NQ] Sec 3.2 Step 3's singular "the
    # current target linear layer" (:641) depict a PER-TARGET-LINEAR cycle —
    # tune the still-FP remainder, factorize the target, refine the target —
    # not one pass per block. The old pre-group/remainder `split` machinery
    # existed only to serve the falsified once-per-block reading and is gone.
    #
    # Target 1 is factorized first, WITHOUT a preceding TuneFP. A pass before
    # the first factorization has nothing factorized, so the student block IS
    # the pristine teacher block on the same ``x_b``: the objective is
    # identically zero and so is its gradient — exactly the vacuous shape this
    # placement removes. Every target then runs TuneFP(still-FP remainder) ->
    # factorize -> TuneLatentSTE(this target) -> harden, so TuneFP fires N-1
    # times and the refinement N times per block.
    # --- TuneLatentSTE budget: the refinement unit is the LINEAR, so the App. C
    # factorized-stage regime (stage lr, 8 epochs, cosine) is consumed afresh by
    # each of the block's N targets — the block's total is N x the regime, where
    # one block-joint pass spent it once. `epochs x (n // batch)` below is
    # therefore the PER-FIRE budget, and each fire's row books it as its own
    # `scheduled_steps` so a row cannot be misread about the budget it ran under.
    n_rows = int(x_b.shape[0])
    bs_latent = max(1, min(cfg.stage_latent.batch_size, n_rows))
    total_ste = max(cfg.stage_latent.epochs * (n_rows // bs_latent or 1), 1)
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
    # The block's solve row is ONE row carrying the block's solve cost, so it is
    # the SUM of the per-target solve phases — never a window over the whole
    # loop: the cycle's refinement fires run inside that loop, and a span
    # covering them would charge the solve with every target's refinement time.
    solve_elapsed_s = 0.0
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

        # Re-anchor the clock at THIS target's solve phase. The refinement fire
        # further down is a different stage with its own row; the solve row must
        # not wear its wall time.
        t_solve_iter = time.perf_counter()
        # The target reaches the solver at the DECLARED host grade — float64 for a
        # float64 run (unchanged), float32 for bf16/float32, which is bf16's exact
        # widening and the only NumPy grade it has. Deriving it from compute_dtype
        # keeps a bf16 run from being silently upcast at the solver's doorstep,
        # which is the dtype laundering this change exists to remove.
        w = np.ascontiguousarray(
            to_host_np(linear.weight, dtype=accumulation_np_dtype(cfg.compute_dtype))
        )
        d_in_t, d_out_t = robust_diag(
            entry.d_in_raw,
            entry.d_out_raw,
            tau=cfg.tau,
            gamma=cfg.gamma,
            dtype=accumulation_spec(cfg.compute_dtype),
        )
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
        # No dtype cast here: the solver backend casts to the declared compute
        # dtype at its own entry, and a NumPy cast could not express bf16 anyway.
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
        bal = magnitude_balance(
            p_u,
            p_v,
            d_out_t,
            d_in_t,
            target=w,
            backend=balance_backend,
            dtype=accumulation_spec(cfg.compute_dtype),
        )
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
                dtype=accumulation_spec(cfg.compute_dtype),
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
        # The LIVE bias, i.e. whatever TuneFP just tuned for this target (the
        # pre-TuneFP value only for target 1, which no TuneFP pass precedes).
        # The layer stores at the DECLARED compute dtype, named here rather than
        # inferred from the latents: the polish path hands over exact +-1 BITS
        # (int8), which are storage values, not a precision.
        if linear.bias is not None:
            bias0 = linear.bias.detach().to(_compute_dtype_torch).clone()
        else:
            # Named device from the owning weight (test_device_constructors is
            # the grep that keeps this class unrepresentable).
            _dev = linear.weight.device
            _rows = linear.weight.shape[0]
            bias0 = torch.zeros(_rows, device=_dev, dtype=_compute_dtype_torch)
        fac = FactorizedLinear(
            u0, v0, bias0, factor_mode=cfg.factor_mode, compute_dtype=_compute_dtype_torch
        )
        # Scales initialized from Eq. 8 (with the latent scale folded in).
        with torch.no_grad():
            # copy_ requires matching devices; the numpy round trip through the CPU solver
            # is deliberate (the solver is NumPy), so the return leg lands on the latents' device.
            fac.s1.copy_(torch.from_numpy(s1_np).to(fac.s1.device))
            fac.s2.copy_(torch.from_numpy(s2_np).to(fac.s2.device))
        facs[name] = fac
        # Freeze this linear's factorized weight into the module NOW: every
        # later stage in the block (this target's STE, the next target's TuneFP,
        # the next layer's statistics-free solve) must see the quantized version
        # of the layers already factorized. The tuned bias goes with it (see the
        # loop head) — this linear is out of the tunable remainder from here on,
        # so its bias is final.
        _set_linear(linear, fac, factor_mode=cfg.factor_mode)
        # This target's solve phase closes here, before its refinement fires.
        solve_elapsed_s += time.perf_counter() - t_solve_iter

        # --- Step 3 (line 19), the per-target cycle: refine THIS target against
        # the block-output objective ([NQ] Eq. 10, :649-651) at the full App. C
        # factorized-stage budget, then harden it before the next target's TuneFP
        # and STE run. Algorithm 1 line 19's block-scoped call (:780, one
        # ``TuneLatentSTE`` after the factorization loop closes) is the declared
        # tension against Sec 3.2 Step 3's singular target prose (:641) and
        # Figure 3; the per-target reading governs.
        if refine:
            ste_stats = _tune_latent_ste(
                block,
                {name: fac},
                x_b,
                y_b,
                cfg,
                cfg.stage_latent.batch_size,
                steps=total_ste,
                label=name_prefix or f"offset{layer_offset}",
                loss_weight=terminal_loss_weight,
                loss_weight_source=loss_weight_source,
                loss_weight_mean_before=loss_weight_mean_before,
                loss_weight_mean_after=loss_weight_mean_after,
            )
            # The fire declares its SCOPE and its TARGET. The row alone cannot
            # say whether one pass over N linears or N passes over one produced
            # it, and the same ``declared`` values mean different things in the
            # two shapes.
            ste_stats["scope"] = STE_PER_TARGET_SCOPE
            ste_stats["target"] = name
            per = ste_stats["per_linear"][name]
            if paper_regime:
                # The paper's Stage 2 -> 3 mechanism, to the letter: STE on
                # the continuous latents for the full budget, then sign+pack
                # whatever signs result (Algorithm 1 lines 20-23). A
                # zero-crossing outcome IS the paper's outcome — a
                # measurement, not a re-scope — so the scale-only declaration
                # does not fire on this path. The measured count travels in
                # the record.
                per["declared"] = "paper_ste"
            elif per["zero_crossings"]:
                # Diagnostic path only: the continuous refinement produced no
                # sign crossing, so the linear is declared scale-only under the
                # greppable re-scope marker. There is no discrete fallback
                # behind it: a silent zero-flip completion is impossible here
                # because this declaration is the alternative.
                per["declared"] = "scale_only"
                per["rescope_marker"] = STE_RESCOPE_MARKER
            else:
                per["declared"] = "paper_ste"
            per["scheduled_steps"] = int(total_ste)
            if cfg.factor_mode != "fp":
                # Commit the target (Algorithm 1 lines 20-23) BEFORE the next
                # one: the neighbours of a later fire must be the discrete
                # values the artifact will carry, not continuous latents the
                # next pass could keep moving. An fp run is excluded by
                # construction — its latents ARE the measurement, so
                # extract_and_pack refuses to sign them (D-5).
                harden_target(fac, _pname(name))
                _set_linear(linear, fac, factor_mode=cfg.factor_mode)
            _note_block_progress(0, f"{name_prefix or 'block'}/ste")
            if recorder is not None:
                # ONE record per fire, carrying this target's row. The stage's
                # own summary (steps, wall time, gradient floor, vacuity,
                # inertness, scope, target) sits beside it, and the weighting
                # fields are the same (value, source) pair the TuneFP row
                # carries, so the two stages cannot be read as different
                # regimes.
                payload = dict(ste_stats)
                payload["total_steps"] = total_ste
                payload["teacher_source"] = teacher_source
                payload["teacher_fingerprint"] = teacher_fingerprint
                payload["capture_index"] = capture_index
                payload["rescope_marker"] = STE_RESCOPE_MARKER
                recorder("ste", **payload)
    _note_block_progress(0, f"{name_prefix or 'block'}/lb_admm")
    if recorder is not None:
        # The polish count travels in the existing event: an ablation's log must
        # PROVE zero polish invocations, and adding a new event name would
        # change the stage sequence every existing consumer asserts.
        recorder(
            "lb_admm",
            polish_rounds=cfg.polish_rounds if polish else 0,
            elapsed_s=solve_elapsed_s,
        )

    # --- The block's stage records are closed while the loop runs (each target
    # emits its own tune_fp and ste records at its own position in the cycle).
    # What is left for this point is the ABSENCE case below: a block whose loop
    # fired no TuneFP at all still owes the record that says so.
    _note_block_progress(0, f"{name_prefix or 'block'}/stats_done")
    if refine and tune_fp_record is None:
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

    # --- Extraction and packing (lines 20-23), for the artifact's layer states.
    # The cycle already hardened each target through this same ONE latent-to-bit
    # site, so on a binary run this pass re-extracts bits that are already exact
    # (the latents hold ±1 and sign(±1) is itself) — it is the artifact
    # constructor, not a second conversion with its own convention. An fp run
    # extracted nothing here and hardens nothing: its latents are the
    # measurement.
    layers: list[QuantizedLayerState] = []
    t_pack = time.perf_counter()
    for name, fac in facs.items():
        # Bits are made only for a binary run. In fp mode extract_and_pack
        # refuses outright (D-5), so the run produces no fabricated layer
        # states at all — the measurement carries a report, never an artifact.
        if cfg.factor_mode != "fp":
            # Host hop through to_host_np: the latents carry the declared compute
            # dtype, and bf16 tensors have no NumPy scalar type, so a bare
            # ``.cpu().numpy()`` raises on exactly the dtype this pipeline
            # defaults to. Each array lands at its own declared host grade.
            u_np = to_host_np(fac.u_latent)
            v_np = to_host_np(fac.v_latent)
            st = extract_and_pack(u_np, v_np, name=_pname(name))
            st.s1 = to_host_np(fac.s1).copy()
            st.s2 = to_host_np(fac.s2).copy()
            st.bias = to_host_np(fac.bias).copy()
            layers.append(st)
        # Freeze the effective weight into the module for subsequent prefix
        # capture (the module graph keeps working for later blocks) — the
        # continuous fp factorization in fp mode, the binary one otherwise.
        # The bias written here is the one the artifact carries, so the host
        # forward equals the reloaded artifact's forward.
        linear = linears_by_name[name]
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
            elapsed_s=time.perf_counter() - t_pack,
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
