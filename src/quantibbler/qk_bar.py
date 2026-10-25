"""The [NQ] paper-recreation bar: the finish line, pinned before the climb.

Change: ``qk-nq-recreation-bar``. This module is deliberately TORCH-FREE and
lives in the compressor package (not in ``tools/quantize_hf_model.py``) so the
bar's constants get real suite coverage in a lean environment — the driver
imports torch at module top, so a pin kept there would silently skip.

The bar is NOT the per-run verdict
----------------------------------

``quantibbler.qk_report`` owns the ``ok/degraded/failed`` band, which classifies
every artifact cheaply and remains the ONLY per-run verdict. This module owns a
separate, human-gated acceptance criterion: recreating [NQ]'s published rows. A
run can be ``ok`` and still be far below the bar; nothing here is read by
``qk_report.verdict()``, and nothing here gates a run.

The rows
--------

Source: [NQ] Sec. 4.2 Table 2, the Qwen3-0.6B column of the pinned copy
(``docs/papers/nanoquant-2602.06694v3.md``):

    FP16 baseline          12.66 PPL
    NanoQuant @1.00 bits   27.56 PPL     row budget 1.00 bits
    NanoQuant @0.80 bits   33.79 PPL     row budget 0.80 bits
    NanoQuant @0.55 bits   52.94 PPL     row budget 0.55 bits

The perplexities are TEXTUALLY present in that table, so they are COPIES, never
derivations: each row records the paper line it was copied from and the suite
re-reads that line, so a constant that drifts from the paper fails instead of
quietly redefining the bar.

A row is met by a perplexity at or below ``TOLERANCE * row`` AND a byte-derived
``effective_bpw`` at or below that row's own bit budget — a lower perplexity
bought with extra bits meets no row. Below the lowest budget (0.55 bits) no row
exists at all: that is beyond-paper research, never a recreation.

Tolerance: PROVISIONAL
----------------------

``1.15`` is the pre-registered provisional (the change's ``design.md``,
"Tolerance rationale"): it has no paper source and no measured run-to-run eval
variance behind it. It is re-derived from the first artifact to land within 2x of
a targeted row and re-pinned as a spec change — never edited in a ledger.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "TOLERANCE",
    "FP16_BASELINE_PPL",
    "BAR_ROWS",
    "bar_tolerance",
    "row_meets_bar",
]

#: Row tolerance, verbatim from the delta (``PPL <= 1.15 * row``). PROVISIONAL,
#: per the module docstring: never rounded, never rescaled, never applied to a
#: row other than the one the artifact targets.
TOLERANCE: float = 1.15

#: The FP16 reference the [NQ] Table 2 rows are measured against. Not a target
#: row: a run does not "meet the baseline".
FP16_BASELINE_PPL: float = 12.66

#: Rows as ``(label, row_ppl, row_bits, paper_line)``.
#:
#: ``paper_line`` is the 1-based line of the pinned paper copy the perplexity was
#: copied from; ``row_bits`` is that row's bit budget, the ceiling the artifact's
#: serialized-byte ``effective_bpw`` must not exceed.
BAR_ROWS: tuple[tuple[str, float, float, int], ...] = (
    ("nanoquant_1.00_bits", 27.56, 1.00, 1035),
    ("nanoquant_0.80_bits", 33.79, 0.80, 1079),
    ("nanoquant_0.55_bits", 52.94, 0.55, 1123),
)


def bar_tolerance(row_ppl: float) -> float:
    """The admitted ceiling for a row: ``1.15 * row``, exactly.

    Computed from the row value, never from a rounded literal. The change's
    design table displays 31.69 / 38.86 / 60.88, which are 2dp renderings of
    31.694 / 38.8585 / 60.881 and are NOT the bar.
    """
    return TOLERANCE * row_ppl


def _measured(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` if that is not possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_meets_bar(ppl: Any, effective_bpw: Any, row: tuple) -> bool:
    """Whether an artifact meets ONE bar row — both halves, in either direction.

    ``effective_bpw <= row_bits`` and ``ppl <= 1.15 * row``. A lower perplexity
    at a higher bit budget meets no row (scenario
    ``bar_not_purchased_with_bits``), so the bit test is a condition and not a
    tie-breaker. ``effective_bpw`` is the byte-derived serialized figure the
    report already prints; this predicate only compares it to the row.

    Fails closed when either measurement is missing, non-numeric, or non-finite
    — the rule the per-run band applies too: a bar claim rests on two numbers,
    and a defective one is not evidence of meeting the row.
    """
    _label, row_ppl, row_bits, _line = row
    measured_ppl = _measured(ppl)
    measured_bpw = _measured(effective_bpw)
    if measured_ppl is None or measured_bpw is None:
        return False
    return measured_bpw <= row_bits and measured_ppl <= bar_tolerance(row_ppl)
