"""The fail-fast progress probe.

Split out of ``tests/test_qk_report.py`` (surface 3 of 6); every test body is
verbatim. Kill-only by construction: a 2-batch prefix supports *death* (every
dead run sat 20x+ past the line) and can never support *success*. Only the
gate's `failed` verdict may halt a run. The runtime half lives in
``tests/ptq/test_driver_probe.py``.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np

from quantibbler import qk_report
from quantibbler.qk_report import (
    COLLAPSE_LINE,
    MIN_PROBE_DEPTH,
    OK_LINE,
    probe_aborts,
    probe_ratio,
)

# --------------------------------------------------------------------------
# probe_* — the fail-fast progress probe
#
# Kill-only by construction: a 2-batch prefix supports *death* (every dead
# run sat 20x+ past the line) and can never support *success*. Only the
# gate's `failed` verdict may halt a run, so nobody can later "optimize" the
# full held-out measurement away because a prefix already looked fine.
# --------------------------------------------------------------------------


def test_probe_comparator_reuses_gate_band():
    """The probe decision IS the shared band, not a second threshold."""
    assert probe_aborts(2, COLLAPSE_LINE * 2.0, 1.0) is True
    assert probe_aborts(2, COLLAPSE_LINE * 1.0001, 1.0) is True
    assert probe_aborts(2, COLLAPSE_LINE, 1.0) is False  # the line is degraded
    assert probe_aborts(2, OK_LINE, 1.0) is False
    assert probe_aborts(2, 1.0, 1.0) is False


def test_probe_threshold_is_the_shared_constant_not_a_copy():
    """No second literal 40 lives in qk_report (bug-hunter's check (c))."""
    tree = ast.parse(pathlib.Path(qk_report.__file__).read_text())
    numeric = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    assert 40 not in numeric
    assert 40.0 not in numeric
    # ...and the module really does carry the derived constant.
    assert COLLAPSE_LINE == 40.0


def test_probe_min_depth_two():
    """Probes begin no earlier than depth 2; block-0-alone is the noisiest."""
    assert MIN_PROBE_DEPTH == 2
    for depth in (0, 1):
        assert probe_aborts(depth, float("inf"), 1.0) is False, depth
    for depth in (2, 3, 28):
        assert probe_aborts(depth, COLLAPSE_LINE * 2.0, 1.0) is True, depth


def test_probe_boundary_semantics():
    assert probe_aborts(2, COLLAPSE_LINE, 1.0) is False
    assert probe_aborts(2, COLLAPSE_LINE + 1e-4, 1.0) is True
    assert probe_aborts(2, 4.0, 1.0) is False  # ok
    assert probe_aborts(2, 12.0, 1.0) is False  # degraded — alive, not working


def test_probe_nan_check_runs_first():
    """`nan > COLLAPSE_LINE` is False, so a bare comparison passes a dead run."""
    assert (float("nan") > COLLAPSE_LINE) is False
    assert probe_aborts(2, float("nan"), 1.0) is True
    assert probe_aborts(2, float("inf"), 1.0) is True
    assert probe_aborts(2, 1.0, float("nan")) is True
    assert probe_aborts(2, 1.0, 0.0) is True  # defective baseline: fail closed
    assert probe_aborts(2, None, 1.0) is True
    assert probe_aborts(2, 1.0, None) is True


def test_probe_is_kill_only_and_never_passes_early():
    """No probe result can report success — only the full gate can."""
    for ratio in (0.5, 1.0, 3.99, OK_LINE, OK_LINE + 1e-4, 10.0, 39.999):
        assert probe_aborts(2, ratio, 1.0) is False, ratio


def test_probe_accepts_numpy_scalars():
    assert probe_aborts(2, np.float32(1000.0), np.float32(1.0)) is True
    assert probe_aborts(2, np.float64(1.0), np.float64(1.0)) is False


def test_probe_ratio_helper_isolates_quantization_damage():
    """The depth-ratio divides out the depth-dependent PPL floor."""
    assert probe_ratio(80.0, 2.0) == 40.0
    assert probe_ratio(2.0, 2.0) == 1.0
    for dead in (
        (float("inf"), 2.0),
        (2.0, float("nan")),
        (None, 2.0),
        (2.0, None),
        (2.0, 0.0),  # a zero fp baseline is impossible: defective measurement
        ("nope", 2.0),
    ):
        assert probe_ratio(*dead) is None, dead
