"""PTQ pipeline configuration (task 1.2).

Defaults follow [NQ] (nanoquant-2602.06694v3) Appendix C where the paper
states values; everything else is a qk-ptq design informed assumption and is
commented with its assumption ID (design.md section 8).

Numerics policy (config.yaml method invariants): solvers compute in float64
by default; float32 is a documented, opt-in accelerator, never a silent
default — ``compute_dtype`` accepts only ``"float64"`` / ``"float32"`` and
anything else raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PTQConfig", "StageHyper"]

_COMPUTE_DTYPES = ("float64", "float32")
_SCALE_DTYPES = ("float16", "float32")


@dataclass(frozen=True)
class StageHyper:
    """Learning rate / batch size / epoch budget for one tuning stage.

    Paper defaults ([NQ] App. C): pre-factorization tuning lr 1e-4, bs 4;
    factorized-component tuning lr 1e-5, bs 1; global scale tuning lr 1e-6,
    bs 1; all stages 8 epochs with a cosine learning-rate schedule.
    """

    lr: float
    batch_size: int
    epochs: int

    def __post_init__(self) -> None:
        if self.lr <= 0.0 or not float(self.lr) < float("inf"):
            raise ValueError(f"lr must be a positive finite float, got {self.lr!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if isinstance(self.lr, int):
            # Normalize int-typed lr (e.g. StageHyper(1, 4, 8)) to float.
            object.__setattr__(self, "lr", float(self.lr))
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")


@dataclass(frozen=True)
class PTQConfig:
    """Configuration for the NanoQuant PTQ pipeline ([NQ] Algorithm 1 inputs).

    Attributes:
        rank: factorization rank r (a single global default; per-layer
            schedules are OQ-3, out of v0.1 scope).
        compute_dtype: solver numerics — ``"float64"`` default per
            config.yaml, ``"float32"`` only by explicit opt-in.
        scale_dtype: qpk1 scale storage dtype (qk-representation container
            stores float16 by default; float32 opt-in).
        seed: global seed ([NQ] App. C uses seed 0).
        temperature: KD temperature T of Eq. 11 (assumption A-7: default 1.0,
            the paper states the objective but not a value).
        tau: robust-diagonal clipping ceiling (assumption A-1: absolute cap
            applied to raw diagonal statistics before shrinkage; ``None``
            means the tail-gated per-vector cap derived inside
            ``robust_diag``: q95 engages only on genuinely heavy-tailed
            vectors, max > 10x q95).
        gamma: shrinkage coefficient of Eq. 3 ([NQ] Step 2-1: ~0.2 small
            models, ~0.6 large; 0.2 is the v0.1 default).
        divergence_band_factor: post-balance magnitude gate band F: the
            binary reconstruction's ||Wq||_F/||W||_F must lie within
            [1/F, F] (qk-divergence-guards D2). This is a CATASTROPHE
            bound, not a tight invariant: balance-time scales are Eq. 8
            initializations (pre-STE/KD), and the measured honest range on
            production layers spans 0.0002-6.3 across preconditioner
            regimes and solver basins. Default 1000.0 passes all honest
            observations with >=7x margin and trips the r137 block-0
            divergence (6.3e4) with 60x margin.
        admm_max_iter: K, the ADMM iteration cap ([NQ] App. C: 400).
        rho_start / rho_end: linear penalty schedule endpoints over K
            iterations (assumption A-2: 1e-2 -> 1e+2).
        admm_lam: ridge lambda of Eq. 4 (assumption A-3: 1e-6).
        admm_epsilon: consensus-residual early-stop tolerance (assumption
            A-4: 1e-7, relative form of design section 2).
        stage_pre: TuneFP hyperparameters ([NQ] App. C).
        stage_latent: TuneLatentSTE hyperparameters ([NQ] App. C).
        stage_global: TuneScalesKD hyperparameters ([NQ] App. C).
        cosine_schedule: cosine lr schedule per stage ([NQ] App. C).
        verbose: progress flag; no callback protocol in v0.1 (design section
            6).
    """

    rank: int = 8
    compute_dtype: str = "float64"
    scale_dtype: str = "float16"
    seed: int = 0
    temperature: float = 1.0
    tau: float | None = None
    gamma: float = 0.2
    divergence_band_factor: float = 1000.0
    admm_max_iter: int = 400
    rho_start: float = 1e-2
    rho_end: float = 1e2
    admm_lam: float = 1e-6
    admm_epsilon: float = 1e-7
    stage_pre: StageHyper = field(default_factory=lambda: StageHyper(1e-4, 4, 8))
    stage_latent: StageHyper = field(default_factory=lambda: StageHyper(1e-5, 1, 8))
    stage_global: StageHyper = field(default_factory=lambda: StageHyper(1e-6, 1, 8))
    cosine_schedule: bool = True
    verbose: bool = False

    @property
    def np_compute_dtype(self) -> np.dtype:
        """The numpy scalar dtype for solver entry points."""
        return np.dtype(self.compute_dtype)

    def __post_init__(self) -> None:
        if self.compute_dtype not in _COMPUTE_DTYPES:
            raise ValueError(
                f"compute_dtype must be one of {_COMPUTE_DTYPES} (float32 is opt-in "
                f"only, never a default), got {self.compute_dtype!r}"
            )
        if self.scale_dtype not in _SCALE_DTYPES:
            raise ValueError(
                f"scale_dtype must be one of {_SCALE_DTYPES}, got {self.scale_dtype!r}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1], got {self.gamma}")
        if self.tau is not None and self.tau <= 0.0:
            raise ValueError(f"tau must be positive or None, got {self.tau}")
        if self.divergence_band_factor <= 1.0:
            raise ValueError(
                f"divergence_band_factor must be > 1.0, got {self.divergence_band_factor}"
            )
        if self.admm_max_iter < 1:
            raise ValueError(f"admm_max_iter must be >= 1, got {self.admm_max_iter}")
        if self.rho_start <= 0.0:
            raise ValueError(f"rho_start must be positive, got {self.rho_start}")
        if self.rho_end < self.rho_start:
            raise ValueError(f"rho_end ({self.rho_end}) must be >= rho_start ({self.rho_start})")
        if self.admm_lam < 0.0:
            raise ValueError(f"admm_lam must be >= 0, got {self.admm_lam}")
        if self.admm_epsilon <= 0.0:
            raise ValueError(f"admm_epsilon must be positive, got {self.admm_epsilon}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
