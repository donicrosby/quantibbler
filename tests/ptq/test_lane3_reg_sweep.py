"""Lane-3 reg sweep staging (ruling a): schema, sweep points, ordering gate.

The sweep is the gate in front of the faithful run. These tests pin the parts
that can be pinned WITHOUT lane-0 targets: the three-point sweep shape, the
refusal when targets are absent, and the row schema/regime identity that the
default decision will be read from. The synthetic run is explicitly provisional.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "lane3_reg_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("lane3_reg_sweep", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_tool()


def test_sweep_points_add_a_geometric_mid():
    """{1e-5, 3e-2} plus a geometric mid, so the decision sees the curve."""
    points = sweep.sweep_points([1e-5, 3e-2])
    assert len(points) == 3
    assert points[0] == pytest.approx(1e-5)
    assert points[-1] == pytest.approx(3e-2)
    assert points[1] == pytest.approx((1e-5 * 3e-2) ** 0.5)
    assert points[0] < points[1] < points[2]


def test_sweep_points_reject_non_positive():
    with pytest.raises(ValueError, match="reg must be > 0"):
        sweep.sweep_points([0.0, 3e-2])


def test_refuses_to_run_without_lane0_targets(tmp_path):
    """No lane-0 targets and no --synthetic: refuse, never fabricate rows."""
    missing = tmp_path / "nope.npz"
    assert sweep.main(["--targets", str(missing)]) == 2


def test_requires_a_target_source():
    """--targets and --synthetic are mutually exclusive and one is required."""
    with pytest.raises(SystemExit):
        sweep.main([])


def test_synthetic_smoke_emits_one_complete_row_per_point(tmp_path, monkeypatch):
    """A provisional synthetic sweep produces the full schema for every point."""
    # The tool solves with the production default (torch) and this leg is a
    # schema smoke, not a device claim: on a host with no accelerator the
    # implicit resolution refuses by design, so the test declares the documented
    # CPU reading explicitly. QUANTIBBLER_DEVICE is the resolution's own knob --
    # no resolver is patched and the refusal stays reachable.
    monkeypatch.setenv("QUANTIBBLER_DEVICE", "cpu")
    out = tmp_path / "sweep.jsonl"
    assert sweep.main(["--synthetic", "--out", str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    regs = sweep.sweep_points([1e-5, 3e-2])
    targets = {row["name"] for row in rows}
    assert len(rows) == len(targets) * len(regs), (
        f"expected one row per target x reg, got {len(rows)} rows over {targets}"
    )
    assert {round(row["reg"], 12) for row in rows} == {round(r, 12) for r in regs}

    required = {
        "name",
        "reg",
        "reg_source",
        "rho_start",
        "rho_end",
        "normalization",
        "n_iter",
        "converged",
        "cos_raw",
        "rel_raw",
        "min_eigs",
        "residual_final",
        "residual_stall",
        "ridge_scale_mean",
        "w_fro",
        "provisional",
        "ts",
    }
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        assert required <= set(row), f"row missing fields: {sorted(required - set(row))}"
        # Regime identity: a row can never be mislabelled about its arm.
        assert row["reg_source"] == "env"
        assert row["normalization"] == "column"
        assert row["rho_start"] == pytest.approx(0.0)
        assert row["rho_end"] == pytest.approx(1.0)
        assert row["provisional"] is True
        # SPD floor: the reg floor bounds the smallest solved eigenvalue.
        assert row["min_eigs"] >= row["reg"] * (1.0 - 1e-9)
        assert 0.0 < row["cos_raw"] <= 1.0 + 1e-12
        by_name.setdefault(row["name"], []).append(row)

    # Column normalization makes the solve scale-free: the same target rescaled
    # by 1e3 must give the same post-ADMM cos at a given reg.
    small = sorted(by_name["synthetic-rank4-scale1"], key=lambda r: r["reg"])
    large = sorted(by_name["synthetic-rank4-scale1000"], key=lambda r: r["reg"])
    assert len(small) == len(large) == len(regs)
    for a, b in zip(small, large, strict=True):
        assert a["cos_raw"] == pytest.approx(b["cos_raw"], rel=1e-9)
