"""quantibbler.ptq: the NanoQuant PTQ pipeline (openspec change qk-ptq).

Produces qpk1 artifacts from a caller-supplied fp model and caller-supplied
calibration data (no-network rule of openspec/config.yaml). Kernels are
consumed through the parity-gated dispatch of the qk-kernels capability;
access the dispatch surface via ``quantibbler.ptq.block_recon.kernels``.
"""

from .api import QuantizedModel, quantize_model
from .calibration import collect_stats
from .config import PTQConfig, StageHyper

__all__ = ["PTQConfig", "StageHyper", "collect_stats", "quantize_model", "QuantizedModel"]
