"""Per-block solve checkpoints for reconstruct_model (resume mid-run).

Rationale (2026-10 session data): four consecutive production runs
(r137 v3-v6) died in the pipeline TAIL - KD, eval, storage cast, export
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


def config_fingerprint(cfg) -> str:
    """Stable sha256 over the complete PTQConfig + code identity."""
    payload = {
        "config": dataclasses.asdict(cfg),
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


def write_fingerprint(ckpt_dir: Path, cfg) -> None:
    (ckpt_dir / _FINGERPRINT_NAME).write_text(
        json.dumps({"fingerprint": config_fingerprint(cfg)}, indent=2), encoding="utf-8"
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
    discipline means a partial write never becomes visible)."""
    return sorted(int(p.stem.split("-")[1]) for p in ckpt_dir.glob("block-*.npz"))


def save_block_checkpoint(ckpt_dir: Path, index: int, layers: list) -> None:
    """Losslessly persist one finished block's QuantizedLayerStates.

    int8 bits and float64 scales are stored exactly as the solver left
    them (npz has no casting step); a tmp+rename keeps partial writes
    invisible to completed_blocks().
    """
    arrays: dict[str, np.ndarray] = {}
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
        names = sorted({key.rsplit("|", 1)[0] for key in npz.files})
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


def apply_layer_states(block: nn.Module, block_name: str, layers: list) -> None:
    """Write checkpointed bits+scales back into a block's linears.

    This reproduces reconstruct_block's final write-back (_set_linear)
    exactly: the stored weight IS the quantized map
    ``W_q = (s1 ⊙ Ub) @ (s2 ⊙ Vb)ᵀ`` copied into the linear's dtype, so a
    resumed run's quantized prefix forward is bit-identical to a fresh
    run's. Layer names are host-global (``layers.N.inner...``); the
    block-name prefix is stripped to address submodules.
    """
    by_rel = {}
    for st in layers:
        rel = st.name[len(block_name) + 1 :] if st.name.startswith(block_name + ".") else st.name
        by_rel[rel] = st
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
