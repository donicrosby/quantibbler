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

Module map (same code, same call graph -- every moved symbol
is re-exported from this module, so external import paths are unchanged):

* ``_factorized`` -- ``FactorizedLinear``, the ONE effective-term definition,
  the write-back, packed sign extraction, the divergence guard.
* ``_stats`` -- Phase-1 statistics inventory/scope, the per-block book slice,
  the tuning-loss weight, capture sizing, the teacher fingerprint.
* ``_probe`` -- solver-probe JSONL rows and the crash-safe block breadcrumb.
* ``_tuning`` / ``_ste`` -- the TuneFP and TuneLatentSTE stages.
* ``_block_stage`` -- ``reconstruct_block`` (one block, Algorithm 1 lines 8-24).
* this module -- ``reconstruct_model``, the model-level loop, and the public
  surface re-exported for external callers.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from ._block_stage import (
    _input_adapter as _input_adapter,
)
from ._block_stage import (
    reconstruct_block,
)
from ._factorized import (
    BlockReconResult,
    FactorizedLinear,
    QuantizedLayerState,
    extract_and_pack,
)
from ._factorized import (
    QuantizationDivergenceError as QuantizationDivergenceError,
)
from ._factorized import (
    _factor_terms as _factor_terms,
)
from ._factorized import (
    _linear_children as _linear_children,
)
from ._factorized import (
    _quantized_forward_block as _quantized_forward_block,
)
from ._factorized import (
    _set_linear as _set_linear,
)
from ._factorized import (
    check_layer_magnitude as check_layer_magnitude,
)
from ._probe import (
    SOLVER_PROBE_DEFAULT_PATH as SOLVER_PROBE_DEFAULT_PATH,
)
from ._probe import (
    SOLVER_PROBE_PATH_ENV as SOLVER_PROBE_PATH_ENV,
)
from ._probe import (
    _note_block_progress as _note_block_progress,
)
from ._probe import (
    _record_solver_probe as _record_solver_probe,
)
from ._stats import (
    _FP_LEAF_NAMES as _FP_LEAF_NAMES,
)
from ._stats import (
    CANONICAL_TERMINAL_NAMES as CANONICAL_TERMINAL_NAMES,
)
from ._stats import (
    CAPTURE_CHUNK_ROWS as CAPTURE_CHUNK_ROWS,
)
from ._stats import (
    LOSS_WEIGHT_MEAN_NORMALIZATION as LOSS_WEIGHT_MEAN_NORMALIZATION,
)
from ._stats import (
    TUNE_LOSS_FALLBACK_EVENT as TUNE_LOSS_FALLBACK_EVENT,
)
from ._stats import (
    _block_stats_slice as _block_stats_slice,
)
from ._stats import (
    _canonical_terminal_pair as _canonical_terminal_pair,
)
from ._stats import (
    _forward_chunked as _forward_chunked,
)
from ._stats import (
    _full_name as _full_name,
)
from ._stats import (
    _normalize_weight_to_unit_mean as _normalize_weight_to_unit_mean,
)
from ._stats import (
    _optimizer_gradient_floor as _optimizer_gradient_floor,
)
from ._stats import (
    _quantizable_names as _quantizable_names,
)
from ._stats import (
    _resolve_loss_weight as _resolve_loss_weight,
)
from ._stats import (
    _scope_filter as _scope_filter,
)
from ._stats import (
    _terminal_loss_weight as _terminal_loss_weight,
)
from ._stats import (
    _usable_statistic as _usable_statistic,
)
from ._stats import (
    _weights_fingerprint as _weights_fingerprint,
)
from ._ste import (
    _discrete_flip_pass as _discrete_flip_pass,
)
from ._ste import (
    _fac_params as _fac_params,
)
from ._ste import (
    _rel_move as _rel_move,
)
from ._ste import (
    _tune_latent_ste as _tune_latent_ste,
)
from ._tuning import (
    STE_RESCOPE_MARKER,
    TUNE_FP_LOSS_MARKER,
    VacuousTeacherError,
)
from ._tuning import (
    _tune_fp as _tune_fp,
)
from .config import PTQConfig

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
