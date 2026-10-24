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
from typing import Any, Literal

import numpy as np

from ..qk_report import COLLAPSE_LINE

__all__ = ["PTQConfig", "StageHyper", "PrefixProbeAbort"]

_COMPUTE_DTYPES = ("float64", "float32")
_SOLVER_BACKENDS = ("torch", "numpy")
_SCALE_DTYPES = ("float16", "float32")
# Root-cause ablation knobs (change qk-root-cause-ablation): each isolates ONE
# constraint so a failing run says which stage lies. Unknown values are
# validation errors, never silent defaults (D-6).
_FACTOR_MODES = ("binary", "fp")
_ABLATIONS = ("none", "untuned-signs")
_LAYER_SCOPES = ("all", "self_attn", "mlp")


class PrefixProbeAbort(RuntimeError):
    """Raised by a ``progress_probe`` to halt the pipeline mid-run.

    Carries the evidence the run report needs (change ``qk-fail-fast-loop``):
    the aborting block's module name and 1-based depth, the quantized
    prefix's perplexity, the fp-prefix baseline at the SAME depth, and the
    ratio between them.

    ``line`` is the derived ``qk_report.COLLAPSE_LINE`` and is NOT a
    constructor parameter — a per-run threshold override is the
    green-washing escape hatch the probe family of changes exists to forbid,
    so there is deliberately no argument by which it can be re-pinned.
    """

    def __init__(
        self,
        *,
        block: str,
        depth: int,
        prefix_ppl: Any,
        fp_prefix_ppl: Any,
        ratio: Any,
    ) -> None:
        self.block = block
        self.depth = depth
        self.prefix_ppl = prefix_ppl
        self.fp_prefix_ppl = fp_prefix_ppl
        self.ratio = ratio
        self.line = COLLAPSE_LINE
        super().__init__(
            f"prefix probe aborted at block {block!r} (depth {depth}): "
            f"prefix perplexity {prefix_ppl!r} vs fp prefix {fp_prefix_ppl!r} "
            f"is ratio {ratio!r}, over the collapse line {COLLAPSE_LINE}"
        )


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
            regimes and solver basins. Default 10000.0 passes that range
            with >=5x margin on both sides and trips the measured block-0
            divergence (6.3e4) with ~6x margin.
        admm_max_iter: K, the ADMM iteration cap ([NQ] App. C: 400).
        rho_start / rho_end: linear penalty schedule endpoints over K
            iterations. [NQ] App. C :3008 prescribes the linear SCHEDULE
            SHAPE only -- the endpoints are an implementation choice, not a
            paper-pinned value. They are calibrated for the NORMALIZED
            regime: the target is divided by its own Frobenius norm at ADMM
            entry (per [NQ] Step 2), so the data term is O(1) and an endpoint
            of 1e2 lets the consensus term dominate it 100x, crushing the
            latents onto the binary proxy before the continuous fit
            converges (measured: converged=False at 400/400 with the warm
            start destroyed, 0.84 -> 0.32 cos). 1e-1 caps the consensus term
            at <=10% of the data term.
        admm_lam: ridge lambda of Eq. 4 (assumption A-3: 1e-6).
        admm_epsilon: consensus-residual early-stop tolerance (assumption
            A-4: 1e-7, relative form of design section 2).
        stage_pre: TuneFP hyperparameters ([NQ] App. C).
        stage_latent: TuneLatentSTE hyperparameters ([NQ] App. C).
        stage_global: TuneScalesKD hyperparameters ([NQ] App. C).
        cosine_schedule: cosine lr schedule per stage ([NQ] App. C).
        progress_probe: observer hook (change qk-fail-fast-loop): called after
            each block is written back into the host as ``(model, depth,
            name)``; may raise ``PrefixProbeAbort`` to halt the run. Default
            ``None`` (unset probe changes nothing); excluded from the
            checkpoint fingerprint.
        verbose: progress flag; no callback protocol in v0.1 (design section
            6).
    """

    rank: int = 8
    compute_dtype: str = "float64"
    solver_backend: str = "torch"
    scale_dtype: str = "float32"
    seed: int = 0
    temperature: float = 1.0
    tau: float | None = None
    gamma: float = 0.2
    divergence_band_factor: float = 10000.0
    polish_rounds: int = 4
    admm_max_iter: int = 400
    # SVD path (ablation-measured): plain truncated SVD warm start
    # through the same balance+polish lands within 0.003 cos of the full
    # 400-iteration LB-ADMM (measured on production layers) while ADMM
    # spends 66-83 s/layer at rank 512. Off = SVD init, balance, polish.
    solver: Literal["admm", "svd"] = "admm"
    rho_start: float = 1e-2
    rho_end: float = 1e-1
    admm_lam: float = 1e-6
    admm_epsilon: float = 1e-7
    stage_pre: StageHyper = field(default_factory=lambda: StageHyper(1e-4, 4, 8))
    stage_latent: StageHyper = field(default_factory=lambda: StageHyper(1e-5, 1, 8))
    stage_global: StageHyper = field(default_factory=lambda: StageHyper(1e-6, 1, 8))
    cosine_schedule: bool = True
    # --- root-cause ablation knobs (change qk-root-cause-ablation) ---------
    # MEASUREMENT knobs, never landing knobs: a run carrying a non-default
    # value is an ablation, and its verdict is data, not a release candidate.
    # All three are solver state — they join the checkpoint fingerprint, so a
    # resumed fp run can never resume as a binary run (D-6).
    factor_mode: Literal["binary", "fp"] = "binary"
    ablation: Literal["none", "untuned-signs"] = "none"
    layer_scope: Literal["all", "self_attn", "mlp"] = "all"
    # --- paper-faithful run flag (change qk-fix-tunefp-teacher) ------------
    # A run that CLAIMS paper fidelity fails closed on any
    # ``tune_fp_vacuous_teacher`` event instead of continuing with pristine
    # weights: under the [NQ] Figure 3 placement the operative TuneFP call
    # always has factorized in-block layers, so a zero first-step gradient is
    # a construction defect, never legitimate nothing-to-compensate. Default
    # False keeps the diagnostic semantics (event recorded, stage skipped).
    # Solver state: it joins the checkpoint fingerprint.
    paper_faithful: bool = False
    # --- bounded discrete flip pass (change qk-fix-ste-sign-adaptation) ----
    # Fallback 3 of the mechanism ladder: proposals per factor per linear,
    # accepted only on a decrease of the block-reconstruction objective. The
    # paper has no discrete flip pass, so any nonzero value is an EXPLICIT
    # DEVIATION and is recorded as such in the run record. 0 disables the
    # pass, in which case a linear whose continuous refinement produced no
    # sign crossing is declared scale-only under the greppable re-scope
    # marker — never a silent zero-flip completion.
    ste_flip_pass_budget: int = 0
    # Observer hook (change qk-fail-fast-loop): called by the sequential
    # block reconstruction after each block is written back into the host
    # model, as ``probe(model, depth, name)`` with the live host, the 1-based
    # count of quantized blocks now live, and the block's module name. May
    # raise ``PrefixProbeAbort`` to halt the run. Default ``None``: an unset
    # probe changes nothing. Excluded from the checkpoint fingerprint — an
    # observer is not solver state (compare=False keeps it out of equality
    # too).
    progress_probe: Any = field(default=None, compare=False, repr=False)
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
        if self.solver_backend not in _SOLVER_BACKENDS:
            raise ValueError(
                f"solver_backend must be one of {_SOLVER_BACKENDS}, got "
                f"{self.solver_backend!r}. Production execution is the accelerated "
                f"backend; 'numpy' is the parity oracle and is reachable only by "
                f"asking for it by name."
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
        if self.progress_probe is not None and not callable(self.progress_probe):
            raise TypeError(
                f"progress_probe must be callable or None, got {type(self.progress_probe).__name__}"
            )
        if self.factor_mode not in _FACTOR_MODES:
            raise ValueError(
                f"factor_mode must be one of {_FACTOR_MODES}, got {self.factor_mode!r}"
            )
        if self.ablation not in _ABLATIONS:
            raise ValueError(f"ablation must be one of {_ABLATIONS}, got {self.ablation!r}")
        if self.layer_scope not in _LAYER_SCOPES:
            raise ValueError(
                f"layer_scope must be one of {_LAYER_SCOPES}, got {self.layer_scope!r}"
            )
        if self.factor_mode == "fp" and self.ablation != "none":
            raise ValueError(
                "ablation='untuned-signs' is a BINARY ablation (it exports signs of the "
                f"balance latents); it cannot combine with factor_mode={self.factor_mode!r}"
            )
        if not isinstance(self.paper_faithful, bool):
            raise ValueError(f"paper_faithful must be a bool, got {self.paper_faithful!r}")
        if self.ste_flip_pass_budget < 0:
            raise ValueError(
                "ste_flip_pass_budget must be >= 0 (0 disables the discrete flip "
                f"pass), got {self.ste_flip_pass_budget}"
            )
