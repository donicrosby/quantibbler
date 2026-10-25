"""The kill-only prefix probe: the depth gate and the shared-band ratio verdict."""

from __future__ import annotations

from typing import Any

from .band import FAILED, _as_finite_float, verdict

#: Prefix probes begin no earlier than this depth. Block-0-alone prefixes are
#: the noisiest, and block-0 damage surfaces amplified at depth >= 2 anyway
#: (damage compounds along the depth chain).
MIN_PROBE_DEPTH = 2


def probe_ratio(prefix_ppl: Any, fp_prefix_ppl: Any) -> float | None:
    """Quantized-prefix / fp-prefix ratio at the SAME depth.

    Dividing by the fp prefix at the same depth is the whole trick: it divides
    out the depth-dependent PPL floor, so the ratio isolates the damage
    quantization inflicted rather than the fact that a 2-block prefix predicts
    worse than a 28-block one.

    ``None`` when either side is missing, non-finite, or the fp baseline is
    zero. A perplexity of zero is impossible and a non-finite one is a
    numerics defect, so neither is a ratio — it is a defective measurement,
    and the caller fails closed on it.
    """
    prefix = _as_finite_float(prefix_ppl)
    baseline = _as_finite_float(fp_prefix_ppl)
    if prefix is None or baseline is None or baseline == 0.0:
        return None
    return prefix / baseline


def probe_aborts(depth: int, prefix_ppl: Any, fp_prefix_ppl: Any) -> bool:
    """Whether the progress probe must halt the run at this block boundary.

    Runs the SAME comparator as the end-of-run gate — ``verdict()`` over the
    shared ``COLLAPSE_LINE`` — with the missing/non-finite check first, so
    "collapsed" means the same thing at depth 3 as at depth 28 and there is
    no second copy of the line to drift.

    Kill-only by construction: it returns ``True`` only for the gate's
    ``failed`` verdict. A probe can never report success. A 2-batch prefix
    supports *death* (every dead run measured 20x+ past the line, far outside
    slice noise) but it can never support *success* — a lucky prefix ratio
    below ``OK_LINE`` on a model that is degraded deeper is exactly the case
    this asymmetry forbids acting on. ``ok`` and ``degraded`` come only from
    the full held-out measurement.

    A defective measurement (missing, non-finite, or a zero baseline) fails
    closed: a false positive costs a re-run, a false negative costs the
    hour-plus run this probe exists to kill.
    """
    if depth is None or depth < MIN_PROBE_DEPTH:
        return False
    ratio = probe_ratio(prefix_ppl, fp_prefix_ppl)
    if ratio is None:
        return True
    return verdict(ratio) == FAILED
