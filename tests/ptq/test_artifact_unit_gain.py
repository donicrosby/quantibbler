"""Artifact sanity gate: the per-layer norm-band leg.

The leg recomputes, for every packed projection of an exported artifact, the
norm ratio of the reconstructed product against the original fp weight:

    ratio = ||(s1 .* Ub) @ (s2 .* Vb).T||_F / ||W||_F

The leg is a pure artifact-side sanity check: it reads the packed streams and
the fp weights and asserts the reconstructed product's norm stays near unit
gain. It is landed with the F1 stage-lr anchoring, which keeps the downstream
tuning stages' compensation in the App. C absolute regime.

The band is calibrated on PRODUCTION layers - fixture-calibrated bands are a
defect - so the honest-extremes arm reads a real artifact when one is
resolvable and skips (naming why) when it is not.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from quantibbler.ptq.unit_gain import (
    UNIT_GAIN_BAND,
    measure_artifact_unit_gain,
    reconstructed_norm_ratio,
    safetensors_weight_getter,
    unit_gain_findings,
)

PRODUCTION_ARTIFACT = "qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget"


def _repo_root() -> Path:
    import quantibbler

    return Path(quantibbler.__file__).resolve().parents[2]


def _find_artifact(name: str) -> Path | None:
    override = os.environ.get("QK_UNIT_GAIN_ARTIFACT")
    if override:
        return Path(override) if Path(override).is_dir() else None
    for base in (_repo_root(), *_repo_root().parents):
        candidate = base / "artifacts" / name
        if candidate.is_dir():
            return candidate
    return None


def _find_safetensors() -> Path | None:
    override = os.environ.get("QK_UNIT_GAIN_SAFETENSORS")
    if override:
        return Path(override) if Path(override).is_file() else None
    roots = [
        Path("/opt/data/hf-cache/hub"),
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        if root is None or not root.is_dir():
            continue
        for snap in sorted(root.glob("models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")):
            return snap
    return None


def test_unit_gain_ratio_is_unity_for_an_exact_factorization():
    """A product that IS the stored weight measures exactly unit gain."""
    gen = np.random.default_rng(3)
    ub = np.where(gen.random((12, 5)) < 0.5, -1.0, 1.0)
    vb = np.where(gen.random((10, 5)) < 0.5, -1.0, 1.0)
    s1 = gen.uniform(0.2, 1.5, size=12)
    s2 = gen.uniform(0.2, 1.5, size=10)
    w = (s1[:, None] * ub) @ (s2[:, None] * vb).T
    assert reconstructed_norm_ratio(w, ub, vb, s1, s2) == pytest.approx(1.0, rel=1e-12)


def test_unit_gain_gate_flags_gain_collapse_and_blowup():
    """The band must actually gate: a collapsed or blown-up gain is a finding,
    an honest layer is not."""
    lo, hi = UNIT_GAIN_BAND
    assert unit_gain_findings({"ok": (lo + hi) / 2.0}) == []
    collapse = unit_gain_findings({"collapsed": lo / 500.0})
    assert collapse and "collapsed" in collapse[0]
    blowup = unit_gain_findings({"blown": hi * 500.0})
    assert blowup and "blown" in blowup[0]
    assert unit_gain_findings({"nan": float("nan")})


def test_unit_gain_band_is_declared_tight_around_the_measured_production_range():
    """The band is the production-calibrated one, not a fixture placeholder."""
    lo, hi = UNIT_GAIN_BAND
    # measured production extremes (196 projections of the v6b artifact):
    # min 0.675942, max 0.858367. The band must CONTAIN them (honest extremes
    # pass) and stay within a couple of doublings of the measured range.
    assert 0.40 <= lo <= 0.675942, "floor must sit below the measured production min, with margin"
    assert 0.858367 <= hi <= 1.30, "ceiling must sit above the measured production max, with margin"


def test_scn_artifact_sanity_gate_per_layer_norm_band():
    """artifact_sanity_gate_per_layer_norm_band: every honest production layer
    lands inside the calibrated band, and the measured extremes pass."""
    artifact = _find_artifact(os.environ.get("QK_UNIT_GAIN_NAME", PRODUCTION_ARTIFACT))
    if artifact is None:
        pytest.skip(
            "no production artifact resolvable for the honest-extremes arm "
            f"(looked for artifacts/{PRODUCTION_ARTIFACT}; set QK_UNIT_GAIN_ARTIFACT)"
        )
    safetensors = _find_safetensors()
    if safetensors is None:
        pytest.skip("no Qwen3-0.6B safetensors resolvable; set QK_UNIT_GAIN_SAFETENSORS")

    ratios = measure_artifact_unit_gain(artifact, safetensors_weight_getter(safetensors))
    assert len(ratios) == 196, f"expected 196 packed projections, measured {len(ratios)}"
    lo, hi = UNIT_GAIN_BAND
    measured_min = min(ratios.values())
    measured_max = max(ratios.values())
    assert measured_min >= lo, f"honest min {measured_min!r} below the calibrated floor {lo!r}"
    assert measured_max <= hi, f"honest max {measured_max!r} above the calibrated ceiling {hi!r}"
    assert unit_gain_findings(ratios) == []
