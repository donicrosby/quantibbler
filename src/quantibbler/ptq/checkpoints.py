"""Per-block solve checkpoints for reconstruct_model (resume mid-run).

Rationale: four consecutive production runs
died in the pipeline TAIL - KD, eval, storage cast, export
- after the ~91-minute solver had finished, burning the entire solve
each time. The solve is deterministic and block-sequential, so its
results are exactly resumable: finished blocks are stored losslessly
(int8 bits, float64 scales - storage casting stays an export-time
concern), and a resumed run replays the quantized prefix forward to
rebuild boundary activations bit-identically.

Resume safety is enforced by a FINGERPRINT: sha256 over the full
PTQConfig (including stage hypers), the quantibbler version, and
git describe, PLUS a model-identity component derived from the LIVE model
(the quantized-parameter scope's module inventory, each module's (dout, din)
and its weight dtype). Resume refuses on a config mismatch - resuming a
rank-137 solve into a rank-64 config would produce plausible-looking garbage
rather than a crash - and on a model-identity mismatch, which is what keeps
one model's block archives from being restored into another model's solve
under an identical config (qk-fp-model-identity). The identity is a
shape/inventory digest, never a repo id or a path: local paths get moved and
renamed, and a real checkpoint must survive that. Env-resolved values are
RECORDED beside the digest in fingerprint.json, for a human to diff when a
resume looks wrong. They are not locks: a checkpoint that bricks because a
file moved or a tree was rebased is a tripwire, not provenance.

Bit-identity contract (asserted by test_scn_checkpoint_resume):
a resumed run's serialized artifact is BYTE-IDENTICAL to a
never-interrupted run's. Same bits or it's broken - there is no
"approximately recovered".

Beside the archives sits the STAGE JOURNAL (``stage-events.jsonl``): the run's
stage events, appended as they fire, so a run resumed after a crash in the
pipeline tail can seed its record with the stages the resume does not re-run
(without it a tail-resumed run's record attests only what the resume itself
emitted — an honest false-deny on a run that executed every stage). It is read
only behind the fingerprint guard, only for a directory with a completed block,
and validated line by line (payload shape AND finiteness — a saturated
measurement is the monotone-leverage corruption class and must deny, never
bless); a missing journal seeds nothing, never a synthesized event. See
:func:`read_stage_log`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

import quantibbler

from ..qk_report import json_safe, stage_journal_record

_FINGERPRINT_NAME = "fingerprint.json"

#: The stage journal beside the checkpoints: one JSON line per stage event,
#: appended as the event fires, so a run
#: that dies in the pipeline tail can still be resumed with a record that
#: attests the stages it actually executed. See :func:`read_stage_log` for the
#: fingerprint guard and the per-line shape check that keep a re-read record
#: from blessing a name-only one.
_STAGE_LOG_NAME = "stage-events.jsonl"

# Observer-only config fields, excluded from the fingerprint: a probe observes
# the solver, it does not steer it, so a resume created without a probe may be
# relaunched WITH one without tripping stale-state protection (and vice
# versa).
_FINGERPRINT_EXCLUDED = ("progress_probe",)

# Fields whose EFFECTIVE value can be overridden at solve time by an env var.
# The RECORD carries the effective value and its source, so a run whose schedule
# came from the environment cannot misattribute itself in the artifact. It is
# recorded, not folded: an env difference is visible in fingerprint.json for a
# human to read, and does not by itself refuse a resume.
_ENV_OVERRIDABLE = ("rho_end", "admm_reg", "solver_backend")


def effective_solver_state(cfg) -> dict[str, object]:
    """Effective (env-resolved) solver state and the source of each value.

    Mirrors the solve-time resolution in ``lb_admm`` so the record carries the
    values the iteration actually used -- not the dataclass defaults it was
    constructed with. Imported lazily to keep module import order independent.
    """
    from .lb_admm import admm_reg_provenance, rho_end_provenance
    from .solver_backend import resolve_device, solver_backend_provenance

    rho_end_value, rho_end_source = rho_end_provenance(cfg.rho_end)
    reg_value, reg_source = admm_reg_provenance(cfg.admm_reg)
    # The backend NAME alone says where nothing: "torch" records identically
    # whether the run executed cuda or cpu, and the numerics differ. Record the
    # RESOLVED DEVICE beside it, so a cross-device resume is visible.
    backend_value, backend_source = solver_backend_provenance(cfg.solver_backend)
    device = resolve_device(backend_value, device=os.environ.get("QUANTIBBLER_DEVICE"))
    return {
        "rho_end": rho_end_value,
        "rho_end_source": rho_end_source,
        "admm_reg": reg_value,
        "admm_reg_source": reg_source,
        "solver_backend": backend_value,
        "solver_backend_source": backend_source,
        "solver_device": device,
    }


def config_fingerprint(cfg) -> str:
    """Stable sha256 over the complete PTQConfig, version, and git describe.

    ``progress_probe`` is blanked before hashing (see
    ``_FINGERPRINT_EXCLUDED``): a callable is code-state, not solver-state,
    and ``dataclasses.asdict`` would otherwise try to serialize a function.

    The digest covers the CONFIG. Env-resolved values are recorded beside it in
    ``fingerprint.json`` instead of being folded in: a digest that moves with an
    env value locks the config to a run state, and the record is what a human
    needs to see when a resume looks wrong.
    """
    overrides = {name: None for name in _FINGERPRINT_EXCLUDED if hasattr(cfg, name)}
    payload = {
        "config": dataclasses.asdict(dataclasses.replace(cfg, **overrides)),
        "quantibbler_version": getattr(quantibbler, "__version__", "unknown"),
    }
    try:
        git = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        payload["git_describe"] = git.stdout.strip() if git.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        payload["git_describe"] = "unknown"
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


#: The model-identity component's key in fingerprint.json.
#: Unlike the fields in ``_FINGERPRINT_PROVENANCE`` below, this one is FOLDED, not
#: merely recorded: a divergence refuses the resume. It carries the digest (what
#: refuses) and the inventory it was built from (what NAMES the divergence) --
#: a bare digest can only say "different", and the operator needs "which module".
_MODEL_IDENTITY_KEY = "model_identity"


def model_identity(model) -> dict[str, list]:
    """Canonical inventory of ``model``'s quantized-parameter scope.

    The scope predicate is the SAME one the solver scopes with
    (``block_recon._quantizable_names``, A-9), so the identity describes exactly
    the parameters a checkpoint restores: one entry per quantizable ``nn.Linear``
    naming, in sorted order, ``[dout, din, weight-dtype]``. ``cfg.layer_scope`` is
    deliberately NOT applied -- scope is a config field and already lives in the
    config digest; an identity that moved with it would report one divergence
    under two names.

    It is an identity of SHAPES, not of weights: two models of the same
    architecture (a base checkpoint and a fine-tune of it) share it. See the
    change's design.md, residual risk.
    """
    from .block_recon import _quantizable_names  # lazy: block_recon imports this module

    modules = dict(model.named_modules())
    return {
        name: [
            int(modules[name].out_features),
            int(modules[name].in_features),
            str(modules[name].weight.dtype).removeprefix("torch."),
        ]
        for name in sorted(_quantizable_names(model))
    }


def model_identity_digest(model) -> str:
    """sha256 over the canonical inventory, for the refuse/accept decision."""
    blob = json.dumps(model_identity(model), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _identity_repr(entry) -> str:
    """``(dout, din, dtype)`` as the refusal prints it."""
    return f"({entry[0]}, {entry[1]}, {entry[2]})"


def _identity_divergence(recorded: dict, live: dict) -> str:
    """What diverged, module by module, for the refusal message.

    Naming the modules is the whole point of recording the inventory beside the
    digest: "the model changed" is not diagnosable, "lin2 recorded but absent
    from the live model" is.
    """
    changed, missing, added = [], [], []
    for name in sorted(set(recorded) | set(live)):
        if name not in live:
            missing.append(name)
        elif name not in recorded:
            added.append(name)
        elif list(recorded[name]) != list(live[name]):
            changed.append(
                f"{name} {_identity_repr(recorded[name])} -> {_identity_repr(live[name])}"
            )
    parts = []
    if changed:
        parts.append("changed: " + "; ".join(changed))
    if missing:
        parts.append("recorded but absent from the live model: " + ", ".join(missing))
    if added:
        parts.append("present in the live model but not recorded: " + ", ".join(added))
    return "; ".join(parts) or "no per-module difference (the recorded encoding changed)"


# Provenance fields recorded ALONGSIDE the hash. The hash already covers them
# (``config_fingerprint`` hashes ``dataclasses.asdict(cfg)``), but a bare digest
# cannot answer "which backend and dtype produced these checkpoints?" without
# re-deriving it. The scenario that matters: a run resumed on a different
# solver backend or dtype must be identifiable from the artifact alone.
_FINGERPRINT_PROVENANCE = (
    "solver_backend",
    "compute_dtype",
    "solver",
    "rank",
    "factor_mode",
    # Regime state, recorded so the artifact names the regime that
    # produced the stored blocks. A diff against a relaunch is visible to a
    # human reader; it does not refuse the resume.
    "admm_reg",
    "rho_start",
    "rho_end",
)


def write_fingerprint(ckpt_dir: Path, cfg, model) -> None:
    """Write the directory's fingerprint record, for ``cfg`` AND ``model``.

    ``model`` is REQUIRED, not defaulted: the record's whole job is to say which
    model these checkpoints belong to, and a caller that cannot name the model it
    is about to quantize has no business writing that record (the same
    disarmed-guard rule ``apply_layer_states.expected_names`` follows).
    """
    record = {"fingerprint": config_fingerprint(cfg)}
    identity = model_identity(model)
    record[_MODEL_IDENTITY_KEY] = {
        "digest": model_identity_digest(model),
        "modules": identity,
    }
    effective = effective_solver_state(cfg)
    for name in _FINGERPRINT_PROVENANCE:
        # Env-overridable fields record the EFFECTIVE value the solve will use,
        # beside its source -- a bare config default would misattest a run whose
        # schedule came from the environment.
        if name in _ENV_OVERRIDABLE:
            record[name] = effective[name]
            record[f"{name}_source"] = effective[f"{name}_source"]
        else:
            record[name] = getattr(cfg, name, "unknown")
    # The resolved device is recorded, not digested: cuda and cpu give different
    # numerics, so a cross-device resume is visible in the artifact instead of
    # silent.
    record["solver_device"] = effective["solver_device"]
    # The process imports from ``quantibbler.__file__``; recording the imported
    # path makes an artifact imported from OUTSIDE the tree it was built in
    # visible from the artifact itself (the stale-install case).
    record["quantibbler_file"] = str(Path(quantibbler.__file__).resolve())
    (ckpt_dir / _FINGERPRINT_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def check_fingerprint(ckpt_dir: Path, cfg, model) -> None:
    """Refuse to resume checkpoints that are not this run's, and say why.

    Two folded components: the config/code digest and the model identity. Both
    are checked before any archive is read or applied, so a refusal cannot leave
    a partially-restored prefix in the host. The model identity is REQUIRED (see
    ``write_fingerprint``), and a record that predates it is refused rather than
    trusted -- it cannot be verified after the fact, and "cannot verify" is not
    "verified".
    """
    fp_path = ckpt_dir / _FINGERPRINT_NAME
    if not fp_path.exists():
        raise RuntimeError(
            f"resume requested but {fp_path} is missing; cannot verify the "
            "checkpoint was produced by an identical configuration"
        )
    record = json.loads(fp_path.read_text(encoding="utf-8"))
    recorded_identity = record.get(_MODEL_IDENTITY_KEY)
    if not isinstance(recorded_identity, dict) or "digest" not in recorded_identity:
        raise RuntimeError(
            f"checkpoint fingerprint mismatch: {fp_path} carries no model-identity "
            "component (it was written before the resume fingerprint bound one), so "
            "this run cannot verify the checkpoints were produced by the model it is "
            "about to quantize. Refusing to resume (stale-state protection). Delete "
            "the checkpoint directory to start fresh."
        )
    diverged = []
    if record.get("fingerprint") != config_fingerprint(cfg):
        diverged.append("a different PTQConfig / code state")
    if recorded_identity["digest"] != model_identity_digest(model):
        detail = _identity_divergence(recorded_identity.get("modules", {}), model_identity(model))
        diverged.append(f"a different model (model identity diverged: {detail})")
    if diverged:
        raise RuntimeError(
            "checkpoint fingerprint mismatch: the on-disk checkpoints were produced "
            f"by {' and '.join(diverged)}, not by this run. Refusing to resume "
            "(stale-state protection). Delete the checkpoint directory to start fresh."
        )


def completed_blocks(ckpt_dir: Path) -> list[int]:
    """Sorted indices of fully-written block checkpoints (atomic-rename
    discipline means a partial write never becomes visible).

    A block archive counts as done only when it actually carries readable
    quantized-layer keys (qk-fix-fp-resume-guard, scenario
    ``empty_archive_is_never_counted_done``). A zero-key archive is what an
    fp-mode run used to write — 22 bytes of zip end-of-central-directory —
    and counting it as done made the next resume SKIP that block's solve
    with nothing restored, so the quantized prefix advanced on pristine
    weights. Membership is decided here from the archive's own keys only
    because this is a SCAN (which indices have an archive); the
    expected-vs-present guard lives in :func:`apply_layer_states`, which
    knows what the live block is owed.
    """
    done: list[int] = []
    for path in ckpt_dir.glob("block-*.npz"):
        try:
            with np.load(path) as npz:
                # The loader's key grammar, applied scan-side too: only
                # ``|``-separated keys are layer fields. The reserved
                # teacher-target key y_b carries no separator, so a
                # y_b-only archive derives no layer name here either and
                # is not counted as a done block.
                names = {key.rsplit("|", 1)[0] for key in npz.files if "|" in key}
        except Exception as exc:  # unreadable archive: never a silent "done"
            print(
                f"[checkpoint] ignoring unreadable archive {path.name}: {exc!r}",
                flush=True,
            )
            continue
        if not names:
            print(
                f"[checkpoint] ignoring empty archive {path.name} "
                "(no quantized-layer keys): not counted as done",
                flush=True,
            )
            continue
        done.append(int(path.stem.split("-")[1]))
    return sorted(done)


def stage_log_path(ckpt_dir: Path) -> Path:
    """The stage journal beside the block checkpoints."""
    return Path(ckpt_dir) / _STAGE_LOG_NAME


def append_stage_event(ckpt_dir: Path, event: str, info: dict) -> None:
    """Append ONE stage event to the journal, as its own JSON line.

    Append-per-event, one open/write/close per event: the failure this exists
    for is a run that dies in the pipeline TAIL, so every event must be on disk
    by the time the stage it describes has returned — there is no buffered
    writer whose tail a crash could lose. Only the report-shaped payload the
    recorder already sanitized is journaled (no tensors: ``json`` could not
    encode one).

    Creates the checkpoint directory if the launch has not yet: Phase-1
    statistics fires BEFORE the block loop, which is where a run first needs the
    directory, so a writer that assumed the directory existed would die on the
    first event of every fresh run.
    """
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # ``json_safe`` first, ``allow_nan=False`` as the backstop: a non-finite
    # measurement (the monotone-leverage class) is recorded as ``null`` instead
    # of the non-standard ``NaN``/``Infinity`` tokens Python's default
    # ``allow_nan=True`` emits — those make the line unreadable for strict
    # JSON parsers, and a resume that must read its own journal would fail on
    # its own legitimate degenerate metric. Anything the sanitizer misses
    # raises here rather than writing a line no strict reader can parse.
    line = json.dumps(json_safe({"event": event, "info": info}), sort_keys=True, allow_nan=False)
    with open(stage_log_path(ckpt_dir), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def reset_stage_log(ckpt_dir: Path) -> None:
    """Drop the journal of a launch that STARTS FRESH rather than resumes.

    Called where the fingerprint is written (a directory with no completed
    block): at that moment the journal on disk belongs to the fingerprint being
    replaced, and a later resume of this directory must not inherit another
    config's events. A missing journal is not an error — that is the
    pre-journal case, and the empty result is the same either way.

    Consequence, named because it is subtle: this also drops the CURRENT
    launch's events emitted before the fingerprint was established — Phase-1
    statistics, which fires before the block loop. That is safe for the resume
    seed by construction (Phase-1 statistics is not checkpointed and re-fires on
    every launch, so a resume never needs it from disk) and it keeps the journal
    to the events a resume cannot re-emit.
    """
    stage_log_path(ckpt_dir).unlink(missing_ok=True)


def resume_stage_events(ckpt_dir: Path, cfg, model) -> list[tuple[str, dict]]:
    """The stage events a RESUMING launch inherits; ``[]`` for a fresh start.

    A directory with no completed block checkpoint is a fresh start, and any
    journal on disk belongs to a launch whose blocks were never written: this
    launch re-runs every stage, so nothing is inherited. Otherwise the journal
    is read through the fingerprint guard and the per-line shape check — against
    the LIVE model, because the fingerprint binds model identity as well as
    config (qk-fp-model-identity).

    Call this BEFORE the launch emits its own first event. The journal is the
    launch's own append log, so a read that happened after a few appends would
    seed the launch with its own prefix events.
    """
    if not completed_blocks(ckpt_dir):
        return []
    return read_stage_log(ckpt_dir, cfg, model)


def read_stage_log(ckpt_dir: Path, cfg, model) -> list[tuple[str, dict]]:
    """The journal's stage events, for seeding a RESUMING run's record.

    Fingerprint-guarded LIVE — ``check_fingerprint`` runs here, first, so this
    reader cannot be called against a directory whose fingerprint does not match
    the run reading it (``model`` is the live model, threaded through from the
    caller and REQUIRED: the journal is this run's only if the checkpoints are) —
    and shape-validated per line: every line must be a JSON
    object naming an event the recorder can emit, whose payload carries the keys
    that event always sends (``qk_report._STAGE_PAYLOAD_KEYS``). A poisoned or
    garbage line — a name-only record, an unregistered event name, a truncated
    write — RAISES. It never yields a record that satisfies the required-stage
    conditions by itself, which is the property those conditions cannot check
    for themselves: they are payload-blind, so a re-read record must not be
    accepted on its names alone.

    A missing journal returns ``[]``: that is the behavior before this journal
    existed (the run's record is then only what this launch emitted), and nothing
    is synthesized to stand in for events that were never recorded.
    """
    check_fingerprint(ckpt_dir, cfg, model)
    path = stage_log_path(ckpt_dir)
    if not path.exists():
        return []
    records: list[tuple[str, dict]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            # A trailing newline is not a record, and a blank line carries
            # nothing that could be read as an event.
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{lineno}: stage journal line is not JSON ({exc})") from exc
        try:
            records.append(stage_journal_record(record))
        except ValueError as exc:
            raise RuntimeError(f"{path}:{lineno}: {exc}") from exc
    return records


def save_block_checkpoint(ckpt_dir: Path, index: int, layers: list, *, y_b=None) -> None:
    """Losslessly persist one finished block's QuantizedLayerStates.

    int8 bits and float64 scales are stored exactly as the solver left
    them (npz has no casting step); a tmp+rename keeps partial writes
    invisible to completed_blocks().

    Refuses to write an EMPTY archive (qk-fix-fp-resume-guard, scenario
    ``empty_archive_is_never_written``): ``np.savez`` with no arrays emits
    a 22-byte zip end-of-central-directory, which then read as "block done"
    and let a resume skip the solve with nothing restored. An empty layer
    list means the caller believed it solved a block and did not — that is
    a defect at the call site, never a checkpoint.
    """
    if not layers:
        raise RuntimeError(
            f"refusing to write an empty block checkpoint for block {index} "
            f"({ckpt_dir}): a block that is owed quantized layers produced none, "
            "so an archive here would make the next resume skip its solve with "
            "nothing restored. Fix the solve, not the checkpoint."
        )
    arrays: dict[str, np.ndarray] = {}
    # The fp-stream teacher target rides along with the block. The resume
    # path then recovers ``x_fp`` as exactly the activation the solve path
    # advanced to, so both branches are ``x_fp = y_b`` and there is no conditional
    # pristine forward whose placement can desync the streams. Stored float64
    # exactly as produced: the stream feeds a later teacher forward, so a lossy
    # cast here would silently move every downstream target.
    if y_b is not None:
        if hasattr(y_b, "detach"):
            _yb = y_b.detach().cpu()
            # Serialization boundary: NumPy has no bfloat16 dtype and torch's
            # .numpy() refuses it ("unsupported ScalarType BFloat16"), so a real
            # bf16 run died at block 0 with default checkpoints. Widen bf16/f16
            # to float32 AT THE HOP -- both are exact subsets of float32 (bf16
            # 8-bit and f16 10-bit mantissas), so nothing is lost, and a fresh
            # run and a resumed run widen identically, preserving the
            # byte-identical resume contract.
            if _yb.dtype in (torch.bfloat16, torch.float16):
                _yb = _yb.float()
            _yb = _yb.numpy()
        else:
            _yb = np.asarray(y_b)
        arrays[_YB_KEY] = _yb
    for st in layers:
        p = f"{st.name}"
        arrays[f"{p}|Ub"] = np.asarray(st.Ub, dtype=np.int8)
        arrays[f"{p}|Vb"] = np.asarray(st.Vb, dtype=np.int8)
        arrays[f"{p}|s1"] = np.asarray(st.s1, dtype=np.float64)
        arrays[f"{p}|s2"] = np.asarray(st.s2, dtype=np.float64)
        if st.bias is not None:
            arrays[f"{p}|bias"] = np.asarray(st.bias, dtype=np.float64)
    tmp = ckpt_dir / f"block-{index}.npz.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:  # handle, not path: savez appends .npz otherwise
        np.savez(fh, **arrays)
    os.replace(tmp, ckpt_dir / f"block-{index}.npz")


def load_block_checkpoint(ckpt_dir: Path, index: int, layer_cls) -> list:
    """Rebuild QuantizedLayerState objects, bit-exact, from a checkpoint."""
    with np.load(ckpt_dir / f"block-{index}.npz") as npz:
        # Only keys carrying a ``|`` are layer fields. Without this filter the
        # reserved y_b key derives a bogus layer named "y_b" and the next line
        # raises KeyError on "y_b|Ub" -- a resume-only failure that appears as a
        # missing-layer error rather than a checkpoint-format one.
        names = sorted({key.rsplit("|", 1)[0] for key in npz.files if "|" in key})
        layers = []
        for name in names:
            layers.append(
                layer_cls(
                    name,
                    npz[f"{name}|Ub"],
                    npz[f"{name}|Vb"],
                    npz[f"{name}|s1"],
                    npz[f"{name}|s2"],
                    npz[f"{name}|bias"] if f"{name}|bias" in npz.files else None,
                )
            )
    return layers


#: Reserved npz key for the fp-stream teacher target. Carries no ``|`` by design,
#: which is why the layer-name derivation filters on that separator.
_YB_KEY = "y_b"


def load_block_yb(ckpt_dir: Path, index: int):
    """Return the block's stored fp-stream target, or None if it predates Lane 2.

    A checkpoint written before the field existed has no y_b; that is a resume
    across the change, and the caller falls back to recomputing the stream rather
    than silently starting from a wrong activation.
    """
    path = ckpt_dir / f"block-{index}.npz"
    with np.load(path) as npz:
        if _YB_KEY not in npz.files:
            return None
        return npz[_YB_KEY]


def _archive_key(name: str, block_name: str) -> str:
    """Normalize a layer name into the checkpoint archive's key space.

    Nested linears are keyed by their path relative to the block
    (``self_attn.q_proj``); a flat block (the block IS the Linear) is keyed
    by the block name itself. Both the archive's stored names and the live
    block's expected names go through this one function so the two sides of
    the guard can never disagree about spelling.
    """
    if not name or name == block_name:
        return block_name
    if name.startswith(block_name + "."):
        return name[len(block_name) + 1 :]
    return name


def apply_layer_states(
    block: nn.Module, block_name: str, layers: list, expected_names: set
) -> None:
    """Write checkpointed bits+scales back into a block's linears.

    This reproduces reconstruct_block's final write-back (_set_linear)
    exactly: the stored weight IS the quantized map
    ``W_q = (s1 ⊙ Ub) @ (s2 ⊙ Vb)ᵀ`` copied into the linear's dtype, so a
    resumed run's quantized prefix forward is bit-identical to a fresh
    run's. Layer names are host-global (``layers.N.inner...``); the
    block-name prefix is stripped to address submodules.

    ``expected_names`` is the set of quantizable names the LIVE block is
    owed (the same predicate the solver scopes with — ``_quantizable_names``
    filtered by ``layer_scope``), in the block's relative name space. The
    archive's present set is asserted EQUAL to it before any write-back,
    and a symmetric difference raises naming the block and the
    missing/unexpected names (qk-fix-fp-resume-guard, scenario
    ``restore_asserts_expected_equals_present``). The parameter is
    required, not defaulted: a caller that cannot name what the live block
    is owed has no business restoring into it, and an omitted guard is a
    disarmed guard. The guard is driven by what the block is OWED, never
    by what the archive happened to contain: a hand-copied or truncated
    archive therefore fails loudly instead of advancing the prefix on
    partially-restored weights.
    """
    by_rel = {}
    for st in layers:
        by_rel[_archive_key(st.name, block_name)] = st
    expected = {_archive_key(name, block_name) for name in expected_names}
    present = set(by_rel)
    if expected != present:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise RuntimeError(
            f"checkpoint restore for block {block_name!r} is incomplete: the "
            f"live block is owed {len(expected)} quantized layer(s), the archive "
            f"holds {len(present)}; missing={missing}, unexpected={unexpected} "
            "— refusing to advance the quantized prefix on a partial archive"
        )
    applied = set()
    for rel, module in block.named_modules():
        # Flat blocks (the block IS the Linear, e.g. _TinyApiModel.lin1)
        # yield relative name '' — fall back to the block-name key.
        st = by_rel.get(block_name) if rel == "" else by_rel.get(rel)
        if st is None or not isinstance(module, nn.Linear):
            continue
        applied.add(rel if rel != "" else block_name)
        # Compute Wq with the SAME torch ops as block_recon._set_linear:
        # torch f64 matmul and numpy matmul can differ in last-ulp
        # summation order, and a ulp in the prefix activations can flip a
        # downstream polish acceptance — byte-identity requires the
        # reference arithmetic path, not merely the same math.
        ub = torch.from_numpy(np.ascontiguousarray(st.Ub, dtype=np.float64))
        vb = torch.from_numpy(np.ascontiguousarray(st.Vb, dtype=np.float64))
        s1 = torch.from_numpy(np.ascontiguousarray(st.s1, dtype=np.float64))
        s2 = torch.from_numpy(np.ascontiguousarray(st.s2, dtype=np.float64))
        w_q = (s1.unsqueeze(1) * ub) @ (s2.unsqueeze(1) * vb).T
        with torch.no_grad():
            module.weight.copy_(w_q.to(module.weight.dtype))
            if module.bias is not None and st.bias is not None:
                module.bias.copy_(
                    torch.from_numpy(np.ascontiguousarray(st.bias)).to(module.bias.dtype)
                )
    missed = set(by_rel) - applied
    if missed:
        raise RuntimeError(
            f"checkpoint restore matched no module for layer(s) {sorted(missed)} "
            f"in block {block_name!r} — refusing to continue with unfrozen weights "
            "(the quantized prefix would silently diverge)"
        )
