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
git describe. Resume refuses on mismatch - resuming a rank-137 solve
into a rank-64 config, or across a code change, would produce
plausible-looking garbage rather than a crash.

Bit-identity contract (asserted by test_scn_checkpoint_resume):
a resumed run's serialized artifact is BYTE-IDENTICAL to a
never-interrupted run's. Same bits or it's broken - there is no
"approximately recovered".
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

_FINGERPRINT_NAME = "fingerprint.json"

# Observer-only config fields, excluded from the fingerprint: a probe observes
# the solver, it does not steer it, so a resume created without a probe may be
# relaunched WITH one without tripping stale-state protection (and vice
# versa). Recorded as an explicit consequence in qk-fail-fast-loop's design.md.
_FINGERPRINT_EXCLUDED = ("progress_probe",)


def config_fingerprint(cfg) -> str:
    """Stable sha256 over the complete PTQConfig + code identity.

    ``progress_probe`` is blanked before hashing (see
    ``_FINGERPRINT_EXCLUDED``): a callable is code-state, not solver-state,
    and ``dataclasses.asdict`` would otherwise try to serialize a function.
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


# Provenance fields recorded ALONGSIDE the hash. The hash already covers them
# (``config_fingerprint`` hashes ``dataclasses.asdict(cfg)``), but a bare digest
# cannot answer "which backend and dtype produced these checkpoints?" without
# re-deriving it. The scenario that matters: a run resumed on a different
# solver backend or dtype must be identifiable from the artifact alone.
_FINGERPRINT_PROVENANCE = ("solver_backend", "compute_dtype", "solver", "rank", "factor_mode")


def write_fingerprint(ckpt_dir: Path, cfg) -> None:
    record = {"fingerprint": config_fingerprint(cfg)}
    for name in _FINGERPRINT_PROVENANCE:
        record[name] = getattr(cfg, name, "unknown")
    (ckpt_dir / _FINGERPRINT_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )


def check_fingerprint(ckpt_dir: Path, cfg) -> None:
    fp_path = ckpt_dir / _FINGERPRINT_NAME
    if not fp_path.exists():
        raise RuntimeError(
            f"resume requested but {fp_path} is missing; cannot verify the "
            "checkpoint was produced by an identical configuration"
        )
    recorded = json.loads(fp_path.read_text(encoding="utf-8"))["fingerprint"]
    if recorded != config_fingerprint(cfg):
        raise RuntimeError(
            "checkpoint fingerprint mismatch: the on-disk checkpoints were "
            "produced by a different PTQConfig / code state than this run. "
            "Refusing to resume (stale-state protection). Delete the "
            "checkpoint directory to start fresh."
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
                names = {key.rsplit("|", 1)[0] for key in npz.files}
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
    # Lane 2: the fp-stream teacher target rides along with the block. The resume
    # path then recovers ``x_fp`` as exactly the activation the solve path
    # advanced to, so both branches are ``x_fp = y_b`` and there is no conditional
    # pristine forward whose placement can desync the streams. Stored float64
    # exactly as produced: the stream feeds a later teacher forward, so a lossy
    # cast here would silently move every downstream target.
    if y_b is not None:
        _yb = y_b.detach().cpu().numpy() if hasattr(y_b, "detach") else np.asarray(y_b)
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
    block: nn.Module, block_name: str, layers: list, expected_names=None
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
    filtered by ``layer_scope``), in the block's relative name space. When
    supplied, the archive's present set is asserted EQUAL to it before any
    write-back, and a symmetric difference raises naming the block and the
    missing/unexpected names (qk-fix-fp-resume-guard, scenario
    ``restore_asserts_expected_equals_present``). The guard is driven by
    what the block is OWED, never by what the archive happened to contain:
    a hand-copied or truncated archive therefore fails loudly instead of
    advancing the prefix on partially-restored weights.
    """
    by_rel = {}
    for st in layers:
        by_rel[_archive_key(st.name, block_name)] = st
    if expected_names is not None:
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
