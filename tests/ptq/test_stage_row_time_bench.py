"""The per-stage wall-time benchmark: schema, leg resolution, and the refusal.

The tool produces the number that supersedes the proposal's per-stage cost
estimate, so what is pinned here is that its readings are self-describing (each
row carries the stage record's own ``elapsed_s`` beside the wall clock) and that
a leg which cannot run refuses instead of silently producing a one-sided table.
Tiny geometry: the benchmark's own production defaults are a manual run.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "stage_row_time_bench.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("stage_row_time_bench", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_tool()

_TINY_BASE = [
    "--d-model",
    "16",
    "--mlp",
    "24",
    "--rank",
    "4",
    "--rows",
    "2",
    "--seq",
    "8",
    "--tunefp-epochs",
    "1",
    "--tunefp-batch",
    "2",
    "--ste-steps",
    "2",
]
_TINY = [*_TINY_BASE, "--legs", "cpu"]


def test_cpu_leg_reports_both_stages_with_their_own_row_time(tmp_path):
    """Every reading names its device and the stage's own measured row time."""
    out = tmp_path / "rows.json"
    assert bench.main([*_TINY, "--out", str(out)]) == 0

    readings = json.loads(out.read_text())
    assert set(readings) == {"tune_fp", "ste"}
    for stage, legs in readings.items():
        assert set(legs) == {"cpu"}, stage
        entry = legs["cpu"]
        assert entry["device"] == "cpu", stage
        assert entry["wall_s"] > 0.0, stage
        # The stage record's own time, which is what the run report carries: a
        # tool that measured its own fixture would report a different quantity.
        assert entry["reported_elapsed_s"] > 0.0, stage
    # One TuneFP row is a fire (8 epochs x (2 // 2) = ... see the config): the
    # step counts come from the stage, never from the tool's own loop.
    assert readings["tune_fp"]["cpu"]["steps"] == bench.parse_args(_TINY).tunefp_epochs
    assert readings["ste"]["cpu"]["steps"] == bench.parse_args(_TINY).ste_steps


def test_explicit_leg_without_an_accelerator_refuses(monkeypatch):
    """A cuda leg on a box that cannot run it is an error, not a skipped row."""
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no accelerator is available"):
        bench.resolve_legs("cuda")
    # The default (no --legs) is one-sided BY DISCLOSURE, and says so.
    assert bench.resolve_legs(None) == ["cpu"]


def test_default_legs_include_the_accelerator_when_one_is_present(monkeypatch):
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    assert bench.resolve_legs(None) == ["cpu", "cuda"]


def test_unknown_leg_is_rejected():
    with pytest.raises(SystemExit, match="unknown leg"):
        bench.resolve_legs("tpu")


@pytest.mark.integration
def test_accelerator_leg_runs_the_same_configuration(tmp_path):
    """The accelerator arm is the CPU arm with a different (device, dtype).

    Marked ``integration``: the cuda arm needs an accelerator, and the default
    tier's referee fails on ANY skip, so this leg is DESELECTED there rather than
    counted as a skip. The skip below still fires in the lane that does select it
    (``pytest -m integration``) on a box without an accelerator, and the
    requirement itself stays testable in the refusal pin above.
    """
    if not torch.cuda.is_available():
        pytest.skip("no accelerator: the cuda arm needs one")
    out = tmp_path / "rows.json"
    assert bench.main([*_TINY_BASE, "--legs", "cpu,cuda", "--out", str(out)]) == 0
    readings = json.loads(out.read_text())
    for stage, legs in readings.items():
        assert set(legs) == {"cpu", "cuda"}, stage
        assert readings[stage]["cpu"]["steps"] == readings[stage]["cuda"]["steps"], stage
        assert legs["cuda"]["device"].startswith("cuda"), stage
        assert legs["cuda"]["wall_s"] > 0.0, stage


def test_accelerator_leg_is_held_out_of_the_default_tier() -> None:
    """The cuda arm needs an accelerator: deselected, never a default-tier skip.

    The default ledger must report zero skips, and a leg that requests a cuda arm
    cannot run on a GPU-less runner — so it is held out by the ``integration``
    marker (deselected) while the CPU leg stays in the default tier. The refusal
    pin above is what keeps the accelerator requirement itself testable here.
    """

    def marks_of(fn: object) -> set[str]:
        return {mark.name for mark in getattr(fn, "pytestmark", ())}

    held_out = test_accelerator_leg_runs_the_same_configuration
    in_default = test_cpu_leg_reports_both_stages_with_their_own_row_time
    assert "integration" in marks_of(held_out)
    assert "integration" not in marks_of(in_default)
