"""Artifact sanity gate: the per-layer unit-gain leg.

For every packed projection of an exported artifact this recomputes the norm
ratio of the reconstructed product against the original fp weight:

    ratio = ||(s1 .* Ub) @ (s2 .* Vb).T||_F / ||W||_F

The artifact's gain is whatever the pipeline's scale extraction, solve-side
reconciliation and downstream tuning stages (which step at the App. C absolute
lrs per F1) together produce. Whatever that shape is, a healthy artifact lands
at a band-calibrated fraction of its continuous-product ratio -- measured median
0.77, band-calibrated and NOT unity -- and a collapsed or blown-up gain is a
sanity finding rather than a nuance.

Band calibration (production layers, NOT fixtures)
--------------------------------------------------
Measured 2026-10-28 across the 196 packed projections of the v6b production
artifact (`qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget`), reconstructing
each qpk1 stream against its `Qwen/Qwen3-0.6B` fp weight:

    min = 0.6759417062632295   (layers.16.inner.mlp.up_proj)
    max = 0.858367324034136    (layers.12.inner.self_attn.k_proj)
    median = 0.7678096712635808

The edges carry a 1.35x margin on each side of its own measured extreme, so the
honest extremes pass with headroom while a gain collapse (the measured
post-balance catastrophe reaches ~2e-4) or a blow-up (~6.3e4) stays far outside:

    floor = 0.6759417062632295 / 1.35 = 0.5006975...  -> 0.50
    ceil  = 0.858367324034136  * 1.35 = 1.1587959...  -> 1.16

A fixture-calibrated band would be a defect: fixtures do not carry the
production solver basins or preconditioner regimes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

#: Calibrated unit-gain band (see the module docstring for the arithmetic).
UNIT_GAIN_BAND: tuple[float, float] = (0.50, 1.16)


def reconstructed_norm_ratio(w, ub, vb, s1, s2) -> float:
    """``||(s1 .* Ub) @ (s2 .* Vb).T||_F / ||W||_F`` for one layer.

    Computed in float64 regardless of the container's storage dtype: the
    statistic is a Frobenius norm summed over every `dout x din` entry and its
    whole job is to be right about orders of magnitude.
    """
    w = np.asarray(w, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    wq = (s1[:, None] * ub) @ (s2[:, None] * vb).T
    n_w = float(np.linalg.norm(w))
    n_q = float(np.linalg.norm(wq))
    if n_w <= 0.0:
        return 0.0 if n_q <= 0.0 else math.inf
    return n_q / n_w


def unit_gain_findings(
    ratios: Mapping[str, float], band: tuple[float, float] = UNIT_GAIN_BAND
) -> list[str]:
    """One finding string per layer whose gain leaves the calibrated band.

    An empty list is the pass. Non-finite ratios are findings, never passes.
    """
    lo, hi = band
    findings: list[str] = []
    for name in sorted(ratios):
        ratio = float(ratios[name])
        if not math.isfinite(ratio):
            findings.append(f"unit-gain: layer {name!r} norm ratio is not finite ({ratio!r})")
        elif not (lo <= ratio <= hi):
            findings.append(
                f"unit-gain: layer {name!r} norm ratio {ratio:.6g} outside band [{lo:g}, {hi:g}]"
            )
    return findings


def hf_weight_name(host_layer_name: str) -> str:
    """Map a host-global layer path to its HF state-dict key.

    ``layers.12.inner.self_attn.q_proj`` -> ``model.layers.12.self_attn.q_proj.weight``
    """
    parts = host_layer_name.split(".")
    if len(parts) < 4 or parts[0] != "layers" or parts[2] != "inner":
        raise ValueError(f"unexpected host layer path {host_layer_name!r}")
    return "model.layers." + parts[1] + "." + ".".join(parts[3:]) + ".weight"


def _host_name(artifact_dir: Path, blob: Path) -> str:
    rel = blob.relative_to(artifact_dir / "layers")
    return ".".join(rel.with_suffix("").parts)


def measure_artifact_unit_gain(
    artifact_dir, weight_getter: Callable[[str], np.ndarray]
) -> dict[str, float]:
    """Per-layer unit-gain ratio for every ``*.qpk1`` under ``layers/``.

    ``weight_getter`` receives the HF state-dict key and returns the fp weight
    (any floating dtype; converted to float64 here). Fail-closed on a missing
    weight: a layer the gate cannot compare is a hole in the gate, never a pass.
    """
    from .. import qpk1

    artifact_dir = Path(artifact_dir)
    layers_root = artifact_dir / "layers"
    blobs = sorted(layers_root.glob("**/*.qpk1"))
    if not blobs:
        raise FileNotFoundError(f"no *.qpk1 streams under {layers_root}")
    ratios: dict[str, float] = {}
    for blob in blobs:
        host = _host_name(artifact_dir, blob)
        w = weight_getter(hf_weight_name(host))
        layer = qpk1.deserialize(blob.read_bytes())
        ratios[host] = reconstructed_norm_ratio(w, layer.Ub, layer.Vb, layer.s1, layer.s2)
    return ratios


def safetensors_weight_getter(path) -> Callable[[str], np.ndarray]:
    """A ``weight_getter`` over one HF safetensors shard (bf16-aware via torch)."""
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(path))

    def get(key: str) -> np.ndarray:
        if key not in tensors:
            raise KeyError(f"{key!r} absent from {path}")
        return tensors[key].detach().to("cpu").to(torch.float64).numpy()

    return get


def report_artifact_unit_gain(artifact_dir, safetensors_path, band=UNIT_GAIN_BAND) -> dict:
    """Full gate result for a CLI: ratios, findings, and the measured extremes."""
    ratios = measure_artifact_unit_gain(artifact_dir, safetensors_weight_getter(safetensors_path))
    values = list(ratios.values())
    return {
        "n": len(ratios),
        "band": list(band),
        "min": min(values),
        "max": max(values),
        "median": float(np.median(values)),
        "findings": unit_gain_findings(ratios, band),
        "ratios": ratios,
    }
