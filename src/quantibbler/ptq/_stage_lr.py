"""Stage learning-rate derivation ([NQ] App. C) and stage-boundary scale
anchoring.

Disposition F1 (2026-10-28): the polish-era per-tensor RELATIVE anchor
(``lr * median|p|``) strangled effective steps 20-100x below [NQ] App. C when the
stage input was the balanced class (latents O(0.01-1), scales well conditioned).
App. C names ABSOLUTE stage lrs (:3005 factorized ``1e-5``, :3006 global
``1e-6``); those now govern balanced inputs and the relative form survives only
as the else-branch for the pathological class that motivated it.

Scale-class anchoring moves to the stage boundary: before lr derivation the
s1/s2 pair is collapsed to a common median by an EXACT product-invariant gamma
rescale (``(s1*gamma) * (s2/gamma) == s1*s2``), so the optimizer always sees the
two scale classes as comparable and the factor can be recorded for a consumer
that needs to unwind it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

#: A tensor (or a stage's whole tunable set) is BALANCED when its median
#: magnitude lies inside this band. Outside it the polish-era pathological class
#: applies and the per-tensor relative form is the else-branch.
BALANCED_MEDIAN_LOW = 0.01
BALANCED_MEDIAN_HIGH = 1.0

_MEDIAN_FLOOR = 1e-30


def median_magnitude(tensor) -> float:
    """Median |entry| of one parameter tensor, floored away from exact zero."""
    return max(float(tensor.detach().abs().median()), _MEDIAN_FLOOR)


def rebalance_scale_pair(s1, s2) -> float:
    """Collapse an s1/s2 scale-class split in place, product-invariantly.

    Rescales ``s1 *= gamma`` and ``s2 /= gamma`` with
    ``gamma = sqrt(median|s2| / median|s1|)``, which equalizes the two vectors'
    median magnitudes at their geometric mid. The stored product
    ``(s1 .* Ub) @ (s2 .* Vb).T`` is unchanged entry-for-entry (``gamma`` cancels
    exactly), so this can never move the artifact weight; it only makes the two
    classes comparable for the step-size decision that follows.

    Returns the factor so a stage record can hand it to a consumer that needs to
    unwind the rescale.
    """
    m1 = median_magnitude(s1)
    m2 = median_magnitude(s2)
    gamma = float(math.sqrt(m2 / m1))
    if not math.isfinite(gamma) or gamma <= 0.0:
        return 1.0
    s1.mul_(gamma)
    s2.div_(gamma)
    return gamma


def derive_stage_lr_groups(params: Iterable, base_lr: float) -> tuple[list[dict], dict]:
    """Build Adam param groups at [NQ] App. C lr semantics for one stage.

    The STAGE input is classified once:

    * **balanced** — every tensor's median magnitude lies in
      ``[BALANCED_MEDIAN_LOW, BALANCED_MEDIAN_HIGH]``: each group's lr is the
      App. C ABSOLUTE value ``base_lr`` (``lr_scale = 1``).
    * **pathological** — otherwise: the per-tensor RELATIVE form
      ``base_lr * median|p|`` is the else-branch.

    Every group also carries ``lr_scale`` so a cosine schedule can rescale the
    rate per step without re-deriving the class, and ``median`` so the stage
    record can say what it classified.

    Returns ``(groups, provenance)`` where ``provenance`` carries the class
    determination, the representative median magnitude, the effective lr, and
    the per-group detail.
    """
    tensors = list(params)
    medians = [median_magnitude(p) for p in tensors]
    balanced = all(BALANCED_MEDIAN_LOW <= m <= BALANCED_MEDIAN_HIGH for m in medians)
    stage_median = float(sorted(medians)[len(medians) // 2]) if medians else 0.0
    groups: list[dict] = []
    detail: list[dict] = []
    for p, med in zip(tensors, medians, strict=True):
        lr_scale = 1.0 if balanced else med
        lr = float(base_lr) * lr_scale
        groups.append({"params": [p], "lr": lr, "lr_scale": lr_scale, "median": med})
        detail.append({"median": med, "lr": lr})
    provenance = {
        "class": "balanced" if balanced else "pathological",
        "median": stage_median,
        "effective_lr": float(base_lr) if balanced else max((d["lr"] for d in detail), default=0.0),
        "groups": detail,
    }
    return groups, provenance
