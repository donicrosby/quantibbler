"""End-to-end quality verdict for real-model production runs.

Change: ``qk-e2e-quality-gate``. This module is deliberately TORCH-FREE and
lives in the compressor package (not in ``tools/quantize_hf_model.py``) so
the verdict and sanitizer get real suite coverage in a lean environment —
the driver imports torch at module top, so helpers kept there would silently
skip. The sanitizer previously lived only in a REPL transcript; it must not
again.

The band
--------

Both edges are DERIVED from the pinned papers' own measurements, not chosen.
Anchors are [LB2] Sec. 5.2 Table 1 (Llama-3-8B, WikiText-2): STBLLM's
documented method collapse at 0.55 bpp (PPL 241.96) against the FP16
reference (PPL 6.10), and the functional envelope at the extreme end of the
budget range, LittleBit-2 at 0.1 bpp (PPL 23.74) against the same 6.10.

    COLLAPSE_LINE = ceil(241.96 / 6.10) = 40
    OK_LINE       = ceil(23.74 / 6.10)  = 4

Reaching the papers' own documented collapse for a method in this regime is
a failed run. The historical best ratio ever measured in this repo is 943.49
(grep ``artifacts/*/quantize_report.json``) — ~24x past the collapse line.
Under this gate every historical run classifies ``failed``; that is the
point.

Calibration-rule debt (declared in the change's design.md): the repo's
gate-calibration rule wants the band set from production healthy extremes,
and none exist — every production extreme is a failure. The band is
paper-anchored instead, and the owning change makes recalibration from the
first ``ok`` artifact a mandatory spec update.
"""

from __future__ import annotations

from .band import (
    COLLAPSE_LINE,
    DEGRADED,
    FAILED,
    OK,
    OK_LINE,
    STEPS_EFFECT_EPS,
    dump_report,
    exit_code,
    json_safe,
    verdict,
)
from .probe import MIN_PROBE_DEPTH, probe_aborts, probe_ratio
from .reports import (
    EVAL_EVALUATOR,
    abort_report,
    evaluation_record,
    geometry_deviations,
    provenance,
    resume_record,
    stage_failure_report,
)
from .stages import (
    _STAGE_LINES,
    _STAGE_PAYLOAD_KEYS,
    PAPER_REGIME_CONFIG_FIELDS,
    config_matches_paper_regime,
    fidelity_conditions,
    fidelity_verdict,
    paper_regime_conditions,
    stage_journal_record,
    stage_map,
    stage_row_times,
)

# Every re-export is listed, as in the other façade (``ptq/kernels_shim``): the
# namespace a caller sees is declared here in full, not implied by the imports.
__all__ = [
    "COLLAPSE_LINE",
    "STEPS_EFFECT_EPS",
    "OK_LINE",
    "MIN_PROBE_DEPTH",
    "FAILED",
    "DEGRADED",
    "OK",
    "verdict",
    "exit_code",
    "json_safe",
    "dump_report",
    "probe_ratio",
    "probe_aborts",
    "abort_report",
    "stage_failure_report",
    "resume_record",
    "stage_map",
    "stage_row_times",
    "geometry_deviations",
    "fidelity_conditions",
    "fidelity_verdict",
    "PAPER_REGIME_CONFIG_FIELDS",
    "paper_regime_conditions",
    "config_matches_paper_regime",
    "stage_journal_record",
    "provenance",
    "EVAL_EVALUATOR",
    "evaluation_record",
    "_STAGE_LINES",
    "_STAGE_PAYLOAD_KEYS",
]
