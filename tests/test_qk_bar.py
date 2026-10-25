"""Suite coverage for the [NQ] paper-recreation bar - change qk-nq-recreation-bar.

The bar's pins live in ``quantibbler.qk_bar``, a TORCH-FREE module at the package
root (the layering ``quantibbler.qk_report`` uses), so the constants are asserted
in a lean environment where ``torch`` is absent: a pin that silently skips is not
a pin.

The values below are the ORACLE, copied from the delta
(``openspec/changes/qk-nq-recreation-bar/specs/qk-ptq/spec.md``, scenarios
``bar_constants_asserted`` and ``bar_not_purchased_with_bits``); the module's
constants are asserted against them, never read back from the module into the
oracle, so a module-side edit cannot move both sides at once.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

from quantibbler import qk_bar
from quantibbler.qk_bar import (
    BAR_ROWS,
    FP16_BASELINE_PPL,
    TOLERANCE,
    bar_tolerance,
    row_meets_bar,
)

#: The module's own source, read as text: the derivation audit is a statement
#: about the code, so it parses the file rather than consulting the values it
#: produced.
_MODULE_SRC = pathlib.Path(qk_bar.__file__).read_text(encoding="utf-8")

#: The pinned paper copy the row constants are COPIED from ([NQ] Sec. 4.2
#: Table 2, Qwen3-0.6B column).
_PAPER = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "papers" / "nanoquant-2602.06694v3.md"
)

#: `(label, row PPL, row bit budget, paper line)` - the delta's pins.
_PINNED_ROWS = (
    ("nanoquant_1.00_bits", 27.56, 1.00, 1035),
    ("nanoquant_0.80_bits", 33.79, 0.80, 1079),
    ("nanoquant_0.55_bits", 52.94, 0.55, 1123),
)
_PINNED_BASELINE_PPL = 12.66
_PINNED_BASELINE_LINE = 881
_PINNED_TOLERANCE = 1.15
#: The delta's design table renders the ceilings to 2dp; the CODE must not.
_PINNED_TOLERANCE_CEILINGS = (31.69, 38.86, 60.88)

# --------------------------------------------------------------------------
# paper provenance - the copy, not a derivation
# --------------------------------------------------------------------------


def _paper_cell(lines: list[str], number: int) -> float:
    """The numeric cell on 1-based paper line ``number``."""
    return float(lines[number - 1].strip().lstrip("|").strip())


def _paper_cells(lines: list[str]) -> list[float]:
    """Every numeric table cell in the pinned copy, in document order."""
    cells: list[float] = []
    for line in lines:
        text = line.strip()
        if not text.startswith("|"):
            continue
        try:
            cells.append(float(text.lstrip("|").strip()))
        except ValueError:
            continue
    return cells


def _derivation_findings(src: str) -> list[str]:
    """Every way ``src`` stops pinning the bar as literals copied from the paper.

    The four rows are TEXTUALLY present in [NQ] Table 2, so they are copies and
    never derivations: this is the copy-vs-derivation half of
    ``bar_constants_asserted``. A row whose value is recomputed at import time
    (``6.33 * 2.0``, ``float("12.66")``, a name resolved elsewhere) has re-opened
    the provenance question the paper already answered, and the audit names it.
    """
    findings: list[str] = []
    assigned: dict[str, ast.expr | None] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value
    scalars = (("FP16_BASELINE_PPL", _PINNED_BASELINE_PPL), ("TOLERANCE", _PINNED_TOLERANCE))
    for name, value in scalars:
        node = assigned.get(name)
        if node is None:
            findings.append(f"{name}: not assigned at module level")
        elif not (isinstance(node, ast.Constant) and node.value == value):
            findings.append(f"{name}: `{ast.unparse(node)}` is not the literal {value!r}")
    rows = assigned.get("BAR_ROWS")
    if not isinstance(rows, ast.Tuple):
        findings.append("BAR_ROWS: not a module-level tuple of literal rows")
        return findings
    if len(rows.elts) != len(_PINNED_ROWS):
        findings.append(f"BAR_ROWS: {len(rows.elts)} rows, not {len(_PINNED_ROWS)}")
    for index, element in enumerate(rows.elts):
        if not isinstance(element, ast.Tuple) or len(element.elts) != 4:
            findings.append(f"BAR_ROWS[{index}]: not a 4-tuple")
            continue
        if index >= len(_PINNED_ROWS):
            continue
        pinned = _PINNED_ROWS[index]
        for column, value in ((1, pinned[1]), (2, pinned[2])):
            cell = element.elts[column]
            if not (isinstance(cell, ast.Constant) and cell.value == value):
                findings.append(
                    f"BAR_ROWS[{index}][{column}]: "
                    f"`{ast.unparse(cell)}` is not the literal {value!r}"
                )
    return findings


# --------------------------------------------------------------------------
# bar_constants_asserted
# --------------------------------------------------------------------------


def test_scn_bar_constants_asserted():
    """The four row constants and the 1.15x arithmetic are the delta's pins."""
    assert FP16_BASELINE_PPL == _PINNED_BASELINE_PPL
    assert tuple(BAR_ROWS) == _PINNED_ROWS
    assert TOLERANCE == _PINNED_TOLERANCE
    # `1.15 * row`, computed per row...
    assert [bar_tolerance(row[1]) for row in _PINNED_ROWS] == [
        1.15 * row[1] for row in _PINNED_ROWS
    ]
    # ...and the design table's ceilings are those numbers rendered to 2dp
    # (31.694 -> 31.69): equal within half a displayed unit, not equal outright.
    assert [bar_tolerance(row[1]) for row in _PINNED_ROWS] == pytest.approx(
        _PINNED_TOLERANCE_CEILINGS, abs=5e-3
    )
    # ...and never rounded to the design table's 2dp rendering of it.
    assert bar_tolerance(27.56) != 31.69
    assert bar_tolerance(33.79) != 38.86
    assert bar_tolerance(52.94) != 60.88


def test_scn_bar_constants_match_the_pinned_paper_copy():
    """Each row constant is the value on the paper line it names, and is unique."""
    lines = _PAPER.read_text(encoding="utf-8").splitlines()
    cells = _paper_cells(lines)
    assert _paper_cell(lines, _PINNED_BASELINE_LINE) == FP16_BASELINE_PPL
    assert cells.count(FP16_BASELINE_PPL) == 1
    for label, ppl, _bits, line in _PINNED_ROWS:
        assert _paper_cell(lines, line) == ppl, label
        assert cells.count(ppl) == 1, label


def test_scn_bar_constants_are_never_recomputed_at_runtime():
    """The copy-vs-derivation audit: the pins are literals in the module source."""
    assert _derivation_findings(_MODULE_SRC) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "FP16_BASELINE_PPL: float = 12.66",
            "FP16_BASELINE_PPL: float = 6.33 * 2.0",
            "FP16_BASELINE_PPL",
        ),
        ("TOLERANCE: float = 1.15", "TOLERANCE: float = 115 / 100", "TOLERANCE"),
        (
            '("nanoquant_1.00_bits", 27.56, 1.00, 1035)',
            '("nanoquant_1.00_bits", float("27.56"), 1.00, 1035)',
            "BAR_ROWS[0][1]",
        ),
        (
            '("nanoquant_0.55_bits", 52.94, 0.55, 1123)',
            '("nanoquant_0.55_bits", _ROW_PPL, 0.55, 1123)',
            "BAR_ROWS[2][1]",
        ),
    ],
)
def test_bar_derivation_audit_rejects_a_recomputed_constant(old, new, expected):  # noqa: ANN001
    """Each recomputed shape is caught, against the real module source."""
    poisoned = _MODULE_SRC.replace(old, new)
    assert poisoned != _MODULE_SRC, "mutation did not apply"
    findings = _derivation_findings(poisoned)
    assert findings, f"the audit accepted a recomputed constant: {new}"
    assert any(expected in finding for finding in findings), (expected, findings)


# --------------------------------------------------------------------------
# bar_not_purchased_with_bits
# --------------------------------------------------------------------------


def test_scn_bar_not_purchased_with_bits():
    """A lower PPL at a higher bit budget meets no row."""
    row = BAR_ROWS[0]  # 27.56 @ 1.00 bits
    assert row_meets_bar(27.56, 1.00, row) is True
    assert row_meets_bar(bar_tolerance(27.56), 1.00, row) is True
    # Better perplexity, more bits: no row is met by either the 1.00-bit row or
    # the 0.55-bit row it could otherwise be pointed at.
    assert row_meets_bar(20.0, 1.20, row) is False
    assert row_meets_bar(20.0, 1.20, BAR_ROWS[2]) is False
    assert row_meets_bar(bar_tolerance(27.56) + 1e-6, 1.00, row) is False
    assert row_meets_bar(27.56, 1.00 + 1e-9, row) is False
    # Below the lowest budget there is no landing site at all.
    assert min(r[2] for r in BAR_ROWS) == 0.55
    # Defective measurements fail closed.
    for bad in (None, "junk", float("nan"), float("inf")):
        assert row_meets_bar(bad, 1.00, row) is False
        assert row_meets_bar(27.56, bad, row) is False


def test_bar_module_is_torch_free():
    """Import and run in a process that has never seen torch."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import quantibbler.qk_bar as m; "
            "assert 'torch' not in sys.modules, sorted(sys.modules); "
            "assert m.bar_tolerance(27.56) == 1.15 * 27.56; "
            "print('ok', int('torch' in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.split() == ["ok", "0"], proc.stdout
