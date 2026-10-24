"""Lane 0: the decisive preconditioner-divergence probe (measurement-only).

`qk-nanoquant-alignment-lane0`. Hypothesis under test: the preconditioner
divergence is the dominant accuracy term, and a healthy preconditioner moves the
post-ADMM pre-polish reconstruction cosine out of the measured ~0.02-0.17 band
with NO other change.

For each layer in the pinned ``PROBE_SET`` two preconditioner variants are built:

  arm A (shipped)  ``collect_stats`` + ``robust_diag``, folded raw --
                   ``w_tilde = (d_out[:, None] * w) * d_in[None, :]``, exactly
                   the construction at ``block_recon.py:1349-1350``.
  arm B (intended) ``W~ = D~_out W D~_in`` with the diagonals the square root of
                   robust second-moment statistics, per-token norms clipped at
                   the robust 99.9th percentile before squaring, the gradient
                   side from a real cross-entropy backward through the head,
                   and the paper's clip-and-shrinkage robustness on top.

Both arms then feed the SHIPPED ``lb_admm`` + ``magnitude_balance`` entry points
under the same production-default ``PTQConfig``. The only per-arm variation is
``(D~_in, D~_out, W~)``. Per layer per arm the report records the four decisive
quantities: post-ADMM pre-polish recon cos against the ORIGINAL weight, the full
consensus residual trajectory, ``n_iter``, and the ``min_eigs`` floor.

Lands nothing in the artifact path: no export, no polish, no STE, no verdict band.

Run:
    uv run python tools/lane0_precond_probe.py --smoke        # synthetic, fast
    uv run python tools/lane0_precond_probe.py --out artifacts/lane0_precond_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import glob
import json
import os
import pathlib
import threading
import time
from typing import Any

import numpy as np

from quantibbler.ptq.calibration import collect_stats, robust_diag
from quantibbler.ptq.config import PTQConfig
from quantibbler.ptq.lb_admm import lb_admm, magnitude_balance

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-0.6B"

# The production solver geometry: the band this lane exists to move was measured
# at rank 512, so the probe runs there too. `PTQConfig`'s dataclass default (8)
# is a smoke-scale default, not the production one.
PRODUCTION_RANK = 512

# The measured pre-polish band (repo measurement, recorded at block_recon.py in
# the greedy-polish rationale -- NOT a paper number).
BAND_LOW = 0.02
BAND_TOP = 0.17

# Informed implementation of the paper's robust tau selection ([NQ] Eq. 3 pins
# clip-to-tau_max + Ledoit-Wolf shrinkage; the percentile is ours, recorded per
# row so a non-domination verdict can re-open it as an axis).
PERCENTILE = 99.9
GAMMA = 0.2
# Protocol geometry, in SEQUENCES ([NQ] Sec 3.4 / App. C). The loader counts
# packed batches, so build_real_arms converts through the driver's own
# packed_chunks rather than passing these straight to the tokenizer.
ATTN_IMPL = os.environ.get("QK_ATTN_IMPL", "sdpa")
# The row-chunking production gives collect_stats on the real-model path
# (block_recon.py:1269). Mirrored rather than restated: the harness's job is to
# measure the shipped construction, not a reference configuration of it.
SHIPPED_CHUNK_ROWS = 4
SEQ_LEN = 2048
CALIBRATION_SEQS = 128

# The pinned probe set: real production layers, attention and MLP, four distinct
# blocks. A named constant with no CLI override -- per-run ad-hoc set edits are
# impossible by construction (the set is the evidence, so it is not a flag).
# Coverage: attention and MLP at each block, spanning shallow (0/1), middle
# (14/15) and the deepest pair (26/27) -- the hypothesis is that preconditioner
# health degrades with depth, so the deep end must be probed, not assumed.
# Verified present in the offline HF cache (one shard, all eight keys).
#
# DELIBERATE ASYMMETRY in the middle span: block 14 carries attention only
# (q_proj) and block 15 MLP only (down_proj). Both families ARE probed in that
# span; the split is by design, not an accidental skip of block 14's down_proj.
# The shallow and deep pairs each carry both families in full.
PROBE_SET: tuple[tuple[str, int, str], ...] = (
    (MODEL_ID, 0, "self_attn.q_proj"),
    (MODEL_ID, 0, "mlp.down_proj"),
    (MODEL_ID, 1, "self_attn.o_proj"),
    (MODEL_ID, 1, "mlp.gate_proj"),
    (MODEL_ID, 14, "self_attn.q_proj"),
    (MODEL_ID, 15, "mlp.down_proj"),
    (MODEL_ID, 26, "self_attn.q_proj"),
    (MODEL_ID, 26, "mlp.down_proj"),
    (MODEL_ID, 27, "self_attn.o_proj"),
    (MODEL_ID, 27, "mlp.gate_proj"),
)

REQUIRED_ROW_FIELDS = (
    "model",
    "block_idx",
    "layer_name",
    "arm",
    "recon_cos",
    "residuals",
    "n_iter",
    "min_eigs_floor",
    "stage",
)


def production_config() -> PTQConfig:
    """The production solver config: default numerics at the production rank."""
    return PTQConfig(rank=PRODUCTION_RANK)


# --- statistics construction ---------------------------------------------------


def clip_token_norms(x: Any, *, percentile: float = PERCENTILE) -> tuple[np.ndarray, float]:
    """Clip per-token vector norms at the robust percentile of the batch.

    The clip happens BEFORE squaring, so a heavy-tailed token cannot dominate a
    feature's second moment. Returns ``(clipped, applied_percentile)`` -- the
    applied percentile is recorded in the report because it is our informed
    choice, not a paper constant.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1)
    cap = float(np.percentile(norms, percentile))
    scale = np.ones_like(norms)
    over = norms > cap
    scale[over] = cap / norms[over]
    return arr * scale[:, None], float(percentile)


def robust_clip_shrink(raw: np.ndarray, *, gamma: float = GAMMA) -> np.ndarray:
    """Clip at the per-vector ceiling then shrink toward the mean ([NQ] Eq. 3)."""
    vec = np.asarray(raw, dtype=np.float64)
    clipped = np.minimum(vec, float(np.max(vec)) if vec.size else 0.0)
    if gamma <= 0.0:
        return clipped
    return (1.0 - gamma) * clipped + gamma * float(np.mean(clipped))


def fold_intended(
    w: np.ndarray,
    *,
    d_out_raw: np.ndarray,
    d_in_raw: np.ndarray,
    gamma: float = GAMMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``W~ = D~_out W D~_in`` with the sqrt fold applied once per side.

    Returns ``(w_tilde, d_out, d_in)`` where the diagonals are the square roots
    of the robust clipped-and-shrunk second-moment statistics. The SAME
    diagonals are what ``magnitude_balance`` unfolds with.
    """
    d_out = np.sqrt(robust_clip_shrink(d_out_raw, gamma=gamma))
    d_in = np.sqrt(robust_clip_shrink(d_in_raw, gamma=gamma))
    w_tilde = (d_out[:, None] * np.asarray(w, dtype=np.float64)) * d_in[None, :]
    return w_tilde, d_out, d_in


def uniform_cotangent_statistic(linear: Any, x: Any) -> np.ndarray:
    """The shipped gradient-side statistic: mean square under a ones-cotangent."""
    import torch

    x_t = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    out = linear(x_t)
    linear.zero_grad(set_to_none=True)
    out.backward(torch.ones_like(out))
    grad = linear.weight.grad.detach().numpy()
    linear.zero_grad(set_to_none=True)
    return (grad**2).mean(axis=0)


def ce_gradient_statistic(linear: Any, x: Any, labels: Any) -> np.ndarray:
    """Per-feature mean square of the REAL cross-entropy weight gradient.

    ``labels`` index the linear's own output (the probe measures a layer in
    isolation, so the head is the linear itself -- the CE loss the paper's
    App. C :3011 describes, not a uniform-ones cotangent).
    """
    import torch

    x_t = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    y_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    logits = linear(x_t)
    loss = torch.nn.functional.cross_entropy(logits, y_t)
    linear.zero_grad(set_to_none=True)
    loss.backward()
    grad = linear.weight.grad.detach().numpy()
    linear.zero_grad(set_to_none=True)
    return (grad**2).mean(axis=0)


# --- measurement ---------------------------------------------------------------


def recon_cos(w: np.ndarray, product: np.ndarray) -> float:
    """``cos(W, U V^T)`` against the ORIGINAL weight (the solver-probe convention)."""
    a = np.asarray(w, dtype=np.float64).ravel()
    b = np.asarray(product, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def assert_row_complete(row: dict) -> None:
    """Fail closed on a partial row: a missing quantity is a failed run."""
    missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"lane-0 row incomplete, missing {missing}")
    if not row["residuals"]:
        raise ValueError("lane-0 row carries an empty residual trajectory")
    if row["n_iter"] != len(row["residuals"]):
        raise ValueError(
            f"lane-0 row truncated: n_iter={row['n_iter']} but {len(row['residuals'])} residuals"
        )


def _solve(
    w: np.ndarray,
    *,
    d_out: np.ndarray,
    d_in: np.ndarray,
    cfg: PTQConfig,
    device: str | None = None,
) -> dict:
    """One arm's solve: shipped solver pair, four quantities out.

    With ``device`` set, the target and diagonals move to the accelerator
    BEFORE the solver runs — a decisive fire labelled GPU must compute on the
    GPU (fail-closed at the stats stage, so reaching here means the device
    exists). Without ``device`` the path is the original, byte-for-byte.
    """
    dt = np.dtype(cfg.compute_dtype)
    if device is not None:
        import torch as _torch

        _dt = getattr(_torch, cfg.compute_dtype)  # float32 on the GPU fire, per cfg
        w_t = _torch.as_tensor(w, dtype=_dt).to(device)
        d_out_t = _torch.as_tensor(np.asarray(d_out), dtype=_dt).to(device)
        d_in_t = _torch.as_tensor(np.asarray(d_in), dtype=_dt).to(device)
        w_tilde = (d_out_t[:, None] * w_t) * d_in_t[None, :]
        res = lb_admm(w_tilde, cfg)
        bal = magnitude_balance(res.P_u, res.P_v, d_out_t, d_in_t, target=w_t)
        product = bal.U @ bal.V.T
        trace = res.trace
        recon = float((_torch.sum(w_t * product)) / (_torch.norm(w_t) * _torch.norm(product)))
        return {
            "recon_cos": recon,
            "residuals": [float(r) for r in trace.residuals],
            "n_iter": int(res.n_iter),
            "min_eigs_floor": float(min(trace.min_eigs)) if trace.min_eigs else float("nan"),
            "n_jitter": int(trace.n_jitter),
            "converged": bool(res.converged),
            "cfg_rank": int(cfg.rank),
            "solver": "admm",
            "compute_dtype": cfg.compute_dtype,
        }
    w_tilde = ((np.asarray(d_out)[:, None] * w) * np.asarray(d_in)[None, :]).astype(dt)
    res = lb_admm(w_tilde, cfg)
    bal = magnitude_balance(res.P_u, res.P_v, d_out, d_in, target=w)
    product = bal.U @ bal.V.T
    trace = res.trace
    return {
        "recon_cos": recon_cos(w, product),
        "residuals": [float(r) for r in trace.residuals],
        "n_iter": int(res.n_iter),
        "min_eigs_floor": float(min(trace.min_eigs)) if trace.min_eigs else float("nan"),
        "n_jitter": int(trace.n_jitter),
        "converged": bool(res.converged),
        "cfg_rank": int(cfg.rank),
        "solver": "admm",
        "compute_dtype": cfg.compute_dtype,
    }


def _synthetic_target(dout: int = 1024, din: int = 1024, seed: int = 0) -> dict:
    """A planted rank-16 + noise target with its nominal statistics."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((dout, 16)) @ rng.standard_normal((16, din))
    w = w + 0.05 * rng.standard_normal((dout, din))
    return {
        "w": np.ascontiguousarray(w, dtype=np.float64),
        "d_in_raw": rng.uniform(0.5, 4.0, size=din),
        "d_out_raw": rng.uniform(0.5, 4.0, size=dout),
    }


def run_arms(
    probe_set: tuple[tuple[str, int, str], ...],
    *,
    seed: int = 0,
    cfg: PTQConfig | None = None,
    target_provider: Any = None,
    stats: dict | None = None,
    device: str | None = None,
) -> list[dict]:
    """Both arms over the pinned set, in set order, one row pair per layer.

    ``stats`` carries ``build_real_arms`` output — the real shipped/intended
    statistics keyed by ``(block_idx, layer_name)``. When it is None the
    provider's nominal statistics drive both arms (the smoke path).
    """
    cfg = cfg or production_config()
    provider = target_provider or (
        lambda _model, _block, _layer, _seed: _synthetic_target(seed=_seed)
    )
    rows: list[dict] = []
    for model, block_idx, layer_name in probe_set:
        target = provider(model, block_idx, layer_name, seed)
        w = target["w"]
        if stats is not None:
            key = (int(block_idx), layer_name)
            arm_a = stats["shipped"][key]
            arm_b = stats["intended"][key]
            # arm A: shipped robust_diag + shipped raw fold. tau/gamma come from
            # the config, as production does (block_recon.py:1349): the values
            # coincide today only because PTQConfig's defaults are the function's
            # defaults, so setting either knob would silently leave the shipped
            # arm measuring a preconditioner the pipeline never uses.
            d_in_a, d_out_a = robust_diag(
                arm_a["d_in_raw"], arm_a["d_out_raw"], tau=cfg.tau, gamma=cfg.gamma
            )
            # arm B: intended (clip -> sqrt fold, once per side)
            _, d_out_b, d_in_b = fold_intended(
                w, d_out_raw=arm_b["d_out_raw"], d_in_raw=arm_b["d_in_raw"]
            )
            percentile = arm_b.get("percentile", PERCENTILE)
            grad_fallback = arm_b.get("grad_fallback")
        else:
            # arm A: shipped robust_diag + shipped raw fold (config-threaded, as above).
            d_in_a, d_out_a = robust_diag(
                target["d_in_raw"], target["d_out_raw"], tau=cfg.tau, gamma=cfg.gamma
            )
            # arm B: intended (clip -> sqrt fold, once per side)
            _, d_out_b, d_in_b = fold_intended(
                w, d_out_raw=target["d_out_raw"], d_in_raw=target["d_in_raw"]
            )
            percentile = PERCENTILE
            grad_fallback = None
        for arm, (d_out, d_in) in (("A", (d_out_a, d_in_a)), ("B", (d_out_b, d_in_b))):
            measured = _solve(w, d_out=d_out, d_in=d_in, cfg=cfg, device=device)
            row = {
                "model": model,
                "block_idx": int(block_idx),
                "layer_name": layer_name,
                "arm": arm,
                "stage": "post_admm_pre_polish",
                "polish_invoked": False,
                "ste_invoked": False,
                "percentile": percentile if arm == "B" else None,
                "grad_fallback": grad_fallback if arm == "B" else None,
                "fold": "intended_sqrt" if arm == "B" else "shipped_raw",
                **measured,
            }
            assert_row_complete(row)
            rows.append(row)
    return rows


def smoke_rows(*, seed: int = 0, cfg: PTQConfig | None = None) -> list[dict]:
    """Two complete rows (one per arm) on the synthetic planted target.

    Smoke-scale by default (the ``PTQConfig`` dataclass rank): the decisive run
    uses ``production_config()`` at rank 512, which is far too slow for a test.
    The synthetic target is production-shaped (1024x1024), so a rank-512 smoke
    is possible when one is wanted.
    """
    return run_arms((("synthetic", 0, "planted-rank16"),), seed=seed, cfg=cfg or PTQConfig())


# --- report and verdict --------------------------------------------------------


def build_report(rows: list[dict], *, probe_set: Any) -> dict:
    """quantize_report-style JSON: strict-parseable, provenance-complete, no band."""
    return {
        "lane": "qk-nanoquant-alignment-lane0",
        "artifact_exported": False,
        "kind": "decisive_preconditioner_probe",
        "probe_set": [list(e) for e in probe_set],
        "band": {"low": BAND_LOW, "top": BAND_TOP},
        "percentile": PERCENTILE,
        "gamma": GAMMA,
        "solver": {"name": "admm", "rank": PRODUCTION_RANK},
        "provenance": {
            "tool": "tools/lane0_precond_probe.py",
            "model": MODEL_ID,
            "compute_dtype": production_config().compute_dtype,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_digest": os.environ.get("QUANTIBBLER_IMAGE_DIGEST", "unknown"),
        },
        "rows": rows,
    }


def build_verdict(rows: list[dict], *, band_top: float = BAND_TOP) -> dict:
    """Winner per probed layer, and the domination rule's outcome.

    Domination (design.md): arm B strictly wins a majority of probed layers AND
    arm B's per-layer mean recon cos lies above the top of the measured band.
    """
    by_layer: dict[str, dict[str, float]] = {}
    for row in rows:
        by_layer.setdefault(row["layer_name"], {})[row["arm"]] = float(row["recon_cos"])
    per_layer: dict[str, dict] = {}
    b_wins = 0
    for layer_name, arms in by_layer.items():
        a, b = arms.get("A"), arms.get("B")
        tie = a is not None and b is not None and a == b
        winner = "B" if (a is not None and b is not None and b > a) else ("not-won" if tie else "A")
        if winner == "B":
            b_wins += 1
        per_layer[layer_name] = {
            "arm_a_cos": a,
            "arm_b_cos": b,
            "winner": winner,
            "tie": bool(tie),
            "in_band": a is not None and BAND_LOW <= a <= band_top,
        }
    b_mean = float(np.mean([v["arm_b_cos"] for v in per_layer.values() if v["arm_b_cos"]]))
    majority = b_wins > len(per_layer) / 2
    above_band = b_mean > band_top
    dominates = bool(majority and above_band)
    gate = "OPEN" if dominates else "STOP"
    if dominates:
        paragraph = (
            f"Intended preconditioner DOMINATES: arm B strictly wins {b_wins} of "
            f"{len(per_layer)} probed layers and its mean recon cos {b_mean:.4f} lies "
            f"above the measured band top {band_top}. Lanes 1-3 gate: OPEN."
        )
    else:
        paragraph = (
            f"Hypothesis FALSIFIED: arm B strictly wins {b_wins} of {len(per_layer)} "
            f"probed layers (majority={majority}) and its mean recon cos {b_mean:.4f} "
            f"against the band top {band_top} (above_band={above_band}). "
            "Lanes 1-3 BLOCKED: STOP and re-analyze before landing anything."
        )
    return {
        "per_layer": per_layer,
        "arm_b_wins": b_wins,
        "arm_b_mean_cos": b_mean,
        "majority": bool(majority),
        "above_band": bool(above_band),
        "dominates": dominates,
        "hypothesis": "supported" if dominates else "falsified",
        "lanes_1_3_gate": gate,
        "paragraph": paragraph,
    }


# --- real-model path -----------------------------------------------------------


def _as_hub_root(root: pathlib.Path) -> pathlib.Path:
    """Normalize a cache root to the directory that actually holds the shards.

    A usable root is a *hub* directory -- the one containing `models--<org>--<name>`.
    `HF_HOME` is the directory *above* it, so the `/hf` mount has to become
    `/hf/hub`; without that the shard glob matches nothing and the load reports
    "not found in any offline HF cache root" while the weights are sitting right
    there. Measured: that is how the decisive fire died, and the witness named it.

    Already-a-hub roots are untouched, because `~/.cache/huggingface/hub/hub`
    does not exist. Normalizing here rather than at the call site keeps every
    consumer honest instead of each one remembering.
    """
    return root / "hub" if (root / "hub").is_dir() else root


def _hf_cache_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        roots.append(pathlib.Path(env))
    roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(pathlib.Path("/hf"))
    return [r for r in (_as_hub_root(c) for c in roots) if r.exists()]


def load_layer_weight(model_id: str, block_idx: int, layer_name: str) -> tuple[np.ndarray, str]:
    """A production layer's fp weight straight from the offline HF cache."""
    from safetensors import safe_open

    key = f"model.layers.{block_idx}.{layer_name}.weight"
    for root in _hf_cache_roots():
        pattern = str(
            root / f"models--{model_id.replace('/', '--')}" / "snapshots" / "*" / "*.safetensors"
        )
        for path in sorted(glob.glob(pattern)):
            with safe_open(path, framework="pt") as handle:
                # `safe_open` is not iterable: `key in handle` raises TypeError.
                # `keys()` is the API. This sat one line below the hub/ offset and
                # was unreachable until that was fixed -- the glob never matched,
                # so this line had never executed. Same class, one line deeper,
                # and the end-to-end fixture is what would have caught both.
                #
                # `noqa` because SIM118's autofix is to *remove* `.keys()` -- the
                # rule assumes a mapping, and `safe_open` is precisely the object
                # that is not one. Applying it silently restores the TypeError, so
                # this is the rare case where the linter's fix is the bug.
                if key in handle.keys():  # noqa: SIM118
                    weight = handle.get_slice(key)[:, :].float().numpy().astype(np.float64)
                    return np.ascontiguousarray(weight), pathlib.Path(path).name
    raise FileNotFoundError(f"{key} not found in any offline HF cache root: {_hf_cache_roots()}")


def _peak_memory_bytes() -> int:
    """Peak reserved bytes on the accelerator; 0 where there is none.

    ``reserved`` rather than ``allocated``: the caching allocator never returns
    freed blocks on gfx1151, so reserved is the number that actually decides
    whether the fence holds.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is present in this lane
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_reserved())


def _empty_cache() -> None:
    """Release the accelerator's cached blocks after a chunk.

    The ROCm caching allocator on gfx1151 does not return freed blocks, so
    reserved memory only climbs across chunks unless the pool is drained. On CPU
    this is a no-op.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is present in this lane
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _checkpointed_blocks(trunk: Any, *, reentrant: bool = True) -> Any:
    """Gradient-checkpoint every decoder block for the duration of the context.

    The trunk activation graph, not the head, is the thing that is actually
    huge: two host deaths from this code path falsified the ~9-10 GB arithmetic,
    and the surviving explanation is attention materialising per-head score
    matrices for backward (28 layers x 8 heads x 2048^2), which checkpointing
    bounds regardless of whether SDPA takes its fused path.

    Implemented by wrapping each block's bound ``forward``, NOT by re-implementing
    the decoder's call sequence. That matters twice over: the model keeps
    computing its own rotary embeddings, masks and cache positions exactly as
    before, and the attention implementation is left entirely alone -- so what
    the container runs is what the fixture measured. Recomputing activations in
    backward is the same kernels on the same inputs, which is why the statistic
    stays bitwise identical (same property the head-chunk fixture proves).

    Raises if the trunk exposes no blocks: silently not checkpointing is the
    failure mode this exists to prevent, so it fails closed instead.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt  # noqa: PLC0415

    layers = list(getattr(trunk, "layers", None) or [])
    if not layers:
        raise ValueError(
            "trunk checkpointing requires a decoder exposing .layers; refusing to "
            "silently run with an unbounded activation graph"
        )
    patched: list[tuple[Any, Any]] = []
    for layer in layers:
        original = layer.forward

        def _make(original: Any = original) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                # kwargs are bound into a partial rather than handed to
                # checkpoint: reentrant checkpointing takes positional inputs
                # only, and the model's own call passes use_cache as a keyword.
                return _ckpt(functools.partial(original, **kwargs), *args, use_reentrant=reentrant)

            return _wrapped

        layer.forward = _make()
        patched.append((layer, original))
    try:
        yield len(layers)
    finally:
        for layer, original in patched:
            layer.forward = original


def collect_intended_statistics(
    model: Any,
    calibration_batches: Any,
    *,
    wanted: tuple[tuple[int, str], ...],
    percentile: float = PERCENTILE,
    chunk_rows: int | None = None,
    chunk_tokens: int | None = None,
    checkpoint_blocks: bool = True,
    gate_on_grad: bool = True,
    reentrant: bool = True,
    diagnostics: dict[str, Any] | None = None,
    telemetry: MemTelemetry | None = None,
) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Per-linear ``(d_in_raw, d_out_raw)`` the paper-intended way.

    Mirrors the shipped ``collect_stats`` structure with exactly the two
    divergences this lane measures removed:

    * the activation side clips per-token norms at the robust percentile BEFORE
      squaring, so a heavy-tailed token cannot dominate a feature's moment;
    * the gradient side is a real cross-entropy backward of the fp teacher's
      logits against the calibration targets (next-token), not the shipped
      ``ones_like`` uniform cotangent (``calibration.py:245``).

    ``wanted`` names ``(block_idx, layer_name)`` pairs; matching is by module
    suffix plus block index, so the harness never depends on the model's exact
    module path. Returns one entry per requested pair; a pair with no reachable
    gradient records ``grad_fallback`` instead of silently substituting a
    statistic.

    ``chunk_tokens`` selects the memory-bounded gradient path: the trunk runs
    ONCE on the full batch (causal context intact) and only the head -- logits,
    CE, backward -- is chunked along the flattened token axis. ``None`` keeps the
    original whole-batch path, which the equivalence fixture compares against.
    """
    import torch

    def _matches(module_name: str) -> tuple[int, str] | None:
        for block_idx, layer_name in wanted:
            if module_name.endswith(f"layers.{block_idx}.{layer_name}"):
                return (block_idx, layer_name)
        return None

    linears = [
        (name, m, key)
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and (key := _matches(name)) is not None
    ]
    if not linears:
        raise ValueError("collect_intended_statistics matched no probed linears")

    device = next(iter(model.parameters())).device
    calibration_batches = [b.to(device) for b in calibration_batches]

    sum_sq_in = {key: None for _, _, key in linears}
    sum_sq_out = {key: None for _, _, key in linears}
    rows = {key: 0 for _, _, key in linears}
    handles = []

    def _clip_and_accumulate(tensor: Any) -> np.ndarray:
        arr = tensor.detach().reshape(-1, tensor.shape[-1]).float().cpu().numpy()
        clipped, _ = clip_token_norms(arr, percentile=percentile)
        return (clipped**2).sum(axis=0), clipped.shape[0]

    for _, module, key in linears:

        def _pre_hook(mod, inp, key=key):  # noqa: ANN001
            # Gate on grad-enabled state: reentrant checkpointing runs the first
            # pass under no_grad and recomputes under grad, so counting only the
            # grad-enabled pass lands every row in sum_sq EXACTLY ONCE. Without
            # the gate both passes count, sum_sq doubles, and only the ratio
            # saves the mean -- correct by accident, and silently wrong for
            # anyone who later reads the raw sum.
            if gate_on_grad and not torch.is_grad_enabled():
                return
            acc, n = _clip_and_accumulate(inp[0])
            sum_sq_in[key] = acc if sum_sq_in[key] is None else sum_sq_in[key] + acc
            rows[key] += n

        handles.append(module.register_forward_pre_hook(_pre_hook))

    if checkpoint_blocks and chunk_tokens is None:
        # Fail closed rather than measure a different gradient. The whole-batch
        # head with checkpointed blocks diverges from every configuration the
        # fire executes: d_out_raw moves ~0.8% while raw sums and row counts stay
        # bitwise identical (803c3fc). Three orders of magnitude above FP
        # reassociation noise, so it is a different gradient being measured, not
        # rounding. No launch path runs it; this guard keeps it that way.
        raise ValueError(
            "whole-batch head + checkpointed blocks is not a supported configuration "
            "(diverges ~0.8% in d_out_raw from every path the fire runs; see commit "
            "803c3fc). Pass chunk_tokens, or set checkpoint_blocks=False for the "
            "whole-batch reference."
        )

    grads: dict[tuple[int, str], np.ndarray | None] = {key: None for _, _, key in linears}
    for _, module, key in linears:

        def _post_hook(mod, inp, out, key=key):  # noqa: ANN001
            if gate_on_grad and not torch.is_grad_enabled():
                return
            acc, n = _clip_and_accumulate(out)
            sum_sq_out[key] = acc if sum_sq_out[key] is None else sum_sq_out[key] + acc
            if out.requires_grad:
                out.register_hook(
                    lambda g, key=key: grads.__setitem__(
                        key,
                        (g.detach().reshape(-1, g.shape[-1]).float().cpu().numpy() ** 2).sum(axis=0)
                        if grads[key] is None
                        else grads[key]
                        + (g.detach().reshape(-1, g.shape[-1]).float().cpu().numpy() ** 2).sum(
                            axis=0
                        ),
                    )
                )

        handles.append(module.register_forward_hook(_post_hook))

    grad_rows = 0
    peak_bytes = 0
    with torch.enable_grad():
        for batch_idx, batch in enumerate(calibration_batches):
            if telemetry is not None:
                telemetry.phase = f"batch{batch_idx + 1}/{len(calibration_batches)}"
            model.zero_grad(set_to_none=True)
            if chunk_tokens is None:
                trunk = model.get_decoder() if hasattr(model, "get_decoder") else model.model
                if checkpoint_blocks:
                    with _checkpointed_blocks(trunk, reentrant=reentrant):
                        logits = model(batch)
                else:
                    logits = model(batch)
                if not isinstance(logits, torch.Tensor):
                    logits = logits[0]
                # Real CE against the calibration targets: next-token prediction.
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = batch[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.float(), shift_labels, ignore_index=-100
                )
                loss.backward()
                del logits, shift_logits, loss
            else:
                # Head-only chunking -- and the BACKWARD is not chunked.
                #
                # The trunk runs once over the full batch so causal context is
                # intact, then the head's forward runs per chunk under
                # torch.utils.checkpoint, so only one chunk's logits exist at a
                # time (checkpointing discards them and recomputes in backward).
                # The chunk losses are summed into a single scalar and ONE
                # backward runs from it, so the hooks fire once on the TOTAL
                # gradient -- which is the quantity the statistic is defined as.
                #
                # Per-chunk backward is FORBIDDEN, and the equivalence fixture
                # pins why: the hook sits on trunk outputs, attention couples
                # rows, so each chunk's backward carries cross-row contributions
                # and squared partials do not sum. Measured on the fixture, that
                # shape lands ~9% off the whole-batch statistic.
                from torch.utils.checkpoint import checkpoint as _ckpt  # noqa: PLC0415

                trunk = model.get_decoder() if hasattr(model, "get_decoder") else model.model
                head = model.get_output_embeddings()
                if checkpoint_blocks:
                    with _checkpointed_blocks(trunk, reentrant=reentrant):
                        hidden = trunk(batch, use_cache=False)[0]
                else:
                    hidden = trunk(batch, use_cache=False)[0]
                shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
                shift_labels = batch[:, 1:].reshape(-1)
                n_b = int(shift_labels.shape[0])

                def _ce_sum(hs: Any, ls: Any, head: Any = None) -> Any:
                    # .float() on the logits mirrors the whole-batch path exactly:
                    # the shipped line casts before CE, and matching the statistic
                    # bitwise means matching that cast, not just the algebra.
                    return torch.nn.functional.cross_entropy(
                        head(hs).float(), ls, ignore_index=-100, reduction="sum"
                    )

                spans = [
                    (lo, min(lo + chunk_tokens, shift_hidden.shape[0]))
                    for lo in range(0, shift_hidden.shape[0], chunk_tokens)
                ]
                total = sum(
                    # The HEAD keeps the non-reentrant flavour deliberately.
                    # Reentrant here moved the captured gradient statistic by
                    # ~1.6% relative (measured on the equivalence fixture), while
                    # the head's own accumulators are never captured -- nothing
                    # in `wanted` lives in the head -- so reentrancy buys the
                    # statistic nothing here and costs exactness. The BLOCKS are
                    # reentrant, which is where the doubling actually mattered.
                    _ckpt(
                        functools.partial(_ce_sum, head=head),
                        shift_hidden[lo:hi],
                        shift_labels[lo:hi],
                        use_reentrant=False,
                    )
                    for lo, hi in spans
                )
                # The mean gauge is applied to the LOSS, not to the squared
                # gradients. One backward from the mean-loss-equivalent scalar
                # reproduces the shipped path's gradient exactly, so nothing
                # downstream is rescaled -- no 1/N^2 factor anywhere.
                (total / n_b).backward()
                del hidden, shift_hidden, total
                _empty_cache()
            grad_rows += int(shift_labels.shape[0])
            # Read the peak BEFORE releasing, or the release would censor the
            # very number the fence is decided on.
            peak_bytes = max(peak_bytes, _peak_memory_bytes())
            # Boundary release: this batch's graph is dead, but the allocator
            # still caches its blocks and the collector has not run. Both have
            # to be told, every batch, for the memory to actually leave.
            release_device_memory(device)
    for handle in handles:
        handle.remove()

    if diagnostics is not None:
        diagnostics["chunk_tokens"] = chunk_tokens
        diagnostics["peak_bytes"] = peak_bytes
        diagnostics["grad_rows"] = grad_rows
        diagnostics["batches"] = len(calibration_batches)
        diagnostics["device"] = str(device)

    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for key in sum_sq_in:
        n_in = max(rows[key], 1)
        n_out = max(rows[key], 1)
        entry: dict[str, Any] = {
            "d_in_raw": np.asarray(sum_sq_in[key], dtype=np.float64) / n_in,
            "d_out_raw": np.asarray(sum_sq_out[key], dtype=np.float64) / n_out,
            "percentile": percentile,
            "grad_fallback": None,
            "grad_rows": grad_rows,
            # Raw accumulators, exposed so the fixture asserts identity on the
            # SUMS rather than only on the means: a doubled numerator with a
            # doubled denominator passes a mean check and fails this one.
            "sum_sq_in": np.asarray(sum_sq_in[key], dtype=np.float64),
            "sum_sq_out": np.asarray(sum_sq_out[key], dtype=np.float64),
            "rows": rows[key],
        }
        if grads[key] is None:
            # No reachable gradient: fall back to the activation-only statistic
            # and RECORD it -- the row is not allowed to pretend otherwise.
            entry["grad_fallback"] = "activation_only_no_reachable_gradient"
            entry["d_out_raw"] = entry["d_in_raw"]
        else:
            entry["d_out_raw"] = np.asarray(grads[key], dtype=np.float64) / max(grad_rows, 1)
        out[key] = entry
    return out


def _is_cuda(device: Any) -> bool:
    """True when `device` names the accelerator, for a str or a torch.device.

    The collector is handed a `torch.device` and the launcher a string, so a
    helper that only understands one of them fails on the other.
    """
    if device is None:
        return False
    import torch

    try:
        resolved = device if isinstance(device, torch.device) else torch.device(device)
    except (TypeError, RuntimeError):
        return str(device).startswith("cuda")
    return resolved.type == "cuda"


def release_device_memory(device: Any) -> None:
    """Return device memory to the OS at a boundary.

    Two things hold memory here and only one of them is the allocator: the
    caching allocator keeps freed blocks, and the autograd graph keeps tensors
    alive until it is dropped. Collecting first and emptying second covers
    both; emptying alone frees nothing Python still references, and collecting
    alone returns nothing to the driver.
    """
    gc.collect()
    if _is_cuda(device):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemTelemetry:
    """Device-memory samples every `interval` seconds, flushed line by line.

    The flush is the whole point: a run that dies does not get to close its
    files, so the last line on disk is the last sample actually taken. `phase`
    is written from the main thread, so that final line names the stage that
    was running when the box went -- the one question three dead runs have
    left open.
    """

    def __init__(self, path: pathlib.Path, device: Any, interval: float = 2.0) -> None:
        self.path = path
        self.device = device
        self.interval = interval
        self.phase = "startup"
        # The host-visible peak. torch's caching-allocator counters read zero
        # once PYTORCH_NO_CUDA_MEMORY_CACHING is on -- so the allocator's
        # max_memory_reserved is NOT the acceptance number for this lane, and
        # reading it as one would clear a run that peaked at 50GB.
        self.peak_used = 0
        self.peak_phase = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write_sample(self, reserved: int, allocated: int, used: int, free: int) -> None:
        """Append one flushed line.

        Separated from sampling so the record's format and its durability are
        testable without an accelerator -- the flush is the property that
        matters, and it is the one a dead run depends on.
        """
        with self.path.open("a") as handle:
            handle.write(f"{time.time():.3f},{self.phase},{reserved},{allocated},{used},{free}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def sample(self) -> None:
        if not _is_cuda(self.device):
            return
        import torch

        if not torch.cuda.is_available():
            return
        free_b, total_b = torch.cuda.mem_get_info()
        used = int(total_b - free_b)
        if used > self.peak_used:
            self.peak_used = used
            self.peak_phase = self.phase
        self.write_sample(
            int(torch.cuda.memory_reserved()),
            int(torch.cuda.memory_allocated()),
            used,
            int(free_b),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            # Telemetry must never be the reason a run dies: a sampling failure
            # is dropped so the run continues, and the gap in the trace shows it.
            with contextlib.suppress(Exception):
                self.sample()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("t,phase,reserved,allocated,used,free\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        self.sample()


def rehearsal_sequences(sequences: int, max_batches: int | None) -> int:
    """The sequence count the construction will actually BUILD.

    A bounded rehearsal reduces the count at the boundary it is converted at,
    rather than building the full protocol and slicing the batches down: a
    sliced construction leaves the guard comparing a reduced row count against
    the protocol count, which is exactly how green-1 refused. The packed batch
    size comes from the driver so the two boundaries cannot drift apart.
    """
    if max_batches is None:
        return sequences
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from quantize_hf_model import BATCH  # noqa: PLC0415

    return max_batches * BATCH


def build_real_arms(
    probe_set: tuple[tuple[str, int, str], ...] = PROBE_SET,
    *,
    sequences: int = CALIBRATION_SEQS,
    protocol_sequences: int = CALIBRATION_SEQS,
    seq_len: int = SEQ_LEN,
    seed: int = 0,
    percentile: float = PERCENTILE,
    device: str | None = None,
    chunk_tokens: int | None = None,
    max_batches: int | None = None,
    diagnostics: dict[str, Any] | None = None,
    telemetry: MemTelemetry | None = None,
) -> dict[str, dict[tuple[int, str], dict[str, Any]]]:
    """Real statistics for both arms, from the real fp teacher at true geometry.

    GEOMETRY CONTRACT: ``tokenize_chunks`` counts PACKED BATCHES, not sequences,
    so the knob is converted through the driver's own ``packed_chunks`` -- the
    same fail-closed boundary ``quantize_hf_model.py:930`` uses. Calling the
    loader directly with ``sequences`` would run this fire at 8x geometry while
    every row still self-labelled "128 sequences": exactly the defect Spec G
    removed, reintroduced by a second caller. The conversion is therefore
    imported, not re-derived.

    Returns ``{"shipped": {...}, "intended": {...}}`` keyed by
    ``(block_idx, layer_name)``, each carrying ``d_in_raw``, ``d_out_raw``,
    ``percentile`` and ``grad_fallback``.
    """
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from quantize_hf_model import (  # noqa: PLC0415
        find_wikitext_shards,
        load_wikitext_text,
        packed_chunks,
        tokenize_chunks,
    )

    if telemetry is not None:
        telemetry.phase = "tokenize"
    text = load_wikitext_text(find_wikitext_shards("train"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Rehearsal bound: exercise the memory ceiling on a few batches instead of
    # all sixteen. The statistic is not being measured here, only the peak --
    # and the peak is a per-batch property, so a bounded run still reproduces a
    # per-batch defect. The reduction happens BEFORE the loader, so what is
    # built is what the guard checks.
    built_sequences = rehearsal_sequences(sequences, max_batches)
    n_batches = packed_chunks(built_sequences, "--calibration-seqs")
    batches = tokenize_chunks(tokenizer, text, seq_len, n_batches, seed)
    resolved = sum(int(b.shape[0]) for b in batches)
    if resolved != built_sequences:
        raise ValueError(
            f"calibration geometry resolved to {resolved} sequences, expected {built_sequences}"
        )
    if diagnostics is not None:
        # Provenance: a rehearsal's geometry must be readable from the record,
        # or "sequences=8" gets read as a protocol claim instead of a bound.
        diagnostics["rehearsal"] = max_batches is not None
        diagnostics["sequences"] = resolved
        diagnostics["protocol_sequences"] = protocol_sequences
        diagnostics["packed_batches"] = len(batches)
        diagnostics["attn_implementation"] = ATTN_IMPL

    if telemetry is not None:
        telemetry.phase = "model_load"
    # Pinned, not auto: which attention kernel ran is part of what the peak
    # means, and `auto` resolves differently per environment. QK_ATTN_IMPL
    # makes the alternative testable without a code change.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
    ).eval()
    del text, tokenizer
    # Fail closed, never silently fall back: a "GPU" run that quietly computed on
    # CPU would report GPU timings for CPU work. Both collectors move the batches
    # to the model's parameter device themselves, so moving the model is the
    # whole device story here.
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device={device!r} requested but no accelerator is available")
        model = model.to(device)
    wanted = tuple((block, name) for _, block, name in probe_set)
    trunk = model.get_decoder() if hasattr(model, "get_decoder") else model.model

    # Arm A: the shipped construction, in the shape production runs it.
    #
    # Production feeds collect_stats the CAPTURED BLOCK INPUTS and runs it on one
    # block (block_recon.py:1261/1269), so the 151936-way head is never in that
    # graph. This used to call collect_stats(model, batches) -- the whole model,
    # head included -- under a comment claiming it was "verbatim -- same call the
    # pipeline makes". Measured on the instrumented rehearsal: 50.1 GB peak, in
    # Arm A, of which ~10 GB was fp32 logits at (8, 2048, 151936) and ~20 GB a
    # fp64 cast of the same, per batch. That head is not in production's path at
    # all. The statistic is unchanged; what changed is that we no longer pay for
    # a shape the pipeline never runs.
    block_indices = sorted({block for block, _ in wanted})

    # The capture runs the MODEL's own forward, in row chunks, and only reads the
    # block inputs off a hook. Calling the decoder directly would mean
    # reproducing position_embeddings and cache_position by hand -- the trap the
    # checkpoint wrapper already documents -- and running the model unchunked
    # would put (8, 2048, 151936) of logits in memory to capture activations that
    # never needed them. Row-chunking is exact: attention is causal and per
    # sequence, so splitting rows cannot change a block's inputs.
    from quantibbler.ptq.block_recon import CAPTURE_CHUNK_ROWS  # noqa: PLC0415

    collected: dict[int, list[Any]] = {idx: [] for idx in block_indices}

    # The collector binds its index at REGISTRATION, through a closure. A default
    # argument cannot do it: register_forward_pre_hook calls the hook with
    # (module, inputs) and nothing else, so `idx` took its default on every
    # firing -- all ten probed blocks appended into collected[0], the other nine
    # keys stayed empty, and the first non-zero probe index crashed on None.
    # The counterfactual is worse than the crash: block 0's statistic would have
    # been computed on an interleaved cat of ten different layers' activations,
    # a confident number measuring no layer that exists.
    def _collector_for(index: int) -> Any:
        def _collect(mod, inp) -> None:  # noqa: ANN001
            collected[index].append(inp[0].detach())

        return _collect

    handles = [
        trunk.layers[idx].register_forward_pre_hook(_collector_for(idx)) for idx in block_indices
    ]
    try:
        if telemetry is not None:
            telemetry.phase = "capture"
        with torch.no_grad():
            for batch in batches:
                moved = batch.to(device) if device is not None else batch
                for lo in range(0, moved.shape[0], CAPTURE_CHUNK_ROWS):
                    model(moved[lo : lo + CAPTURE_CHUNK_ROWS])
                    release_device_memory(device)
    finally:
        for handle in handles:
            handle.remove()
    captured = {
        idx: (torch.cat(chunks, dim=0) if chunks else None) for idx, chunks in collected.items()
    }
    release_device_memory(device)

    if telemetry is not None:
        telemetry.phase = "stats_shipped"
    shipped: dict[tuple[int, str], dict[str, Any]] = {}
    # LayerBlock is production's own block adapter (quantize_hf_model.py:141):
    # modern decoder layers do not compute rotary themselves, so a bare layer
    # cannot be called on activations at all. Imported, not rebuilt -- a hand-rolled
    # adapter here would be the sixth instance of this same defect.
    from quantize_hf_model import LayerBlock  # noqa: PLC0415

    for idx in block_indices:
        block = LayerBlock(trunk.layers[idx], trunk.rotary_emb)
        book = collect_stats(block, [captured[idx]], chunk_rows=SHIPPED_CHUNK_ROWS)
        for entry in book:  # LayerStatsBook is iterable; _entries is private
            for block, name in wanted:
                if block != idx:
                    continue
                # Names come back relative to the block that was passed in.
                if entry.name == name or entry.name.endswith(f".{name}"):
                    shipped[(block, name)] = {
                        "d_in_raw": np.asarray(entry.d_in_raw, dtype=np.float64),
                        "d_out_raw": np.asarray(entry.d_out_raw, dtype=np.float64),
                        "percentile": None,
                        "grad_fallback": None,
                    }
        release_device_memory(device)
    del captured

    # Arm B: the intended construction -- real CE backward, p99.9 clip first.
    # chunk_tokens selects the memory-bounded head-chunked path (see
    # collect_intended_statistics); None keeps the whole-batch reference.
    if telemetry is not None:
        telemetry.phase = "stats_intended"
    intended = collect_intended_statistics(
        model,
        batches,
        wanted=wanted,
        percentile=percentile,
        chunk_tokens=chunk_tokens,
        # Blocks are checkpointed exactly when the head is chunked: the
        # whole-batch reference stays uncheckpointed, which is both the only
        # supported combination and the naive baseline it is meant to be.
        checkpoint_blocks=chunk_tokens is not None,
        diagnostics=diagnostics,
        telemetry=telemetry,
    )

    missing = [key for key in wanted if key not in shipped or key not in intended]
    if missing:
        raise ValueError(f"statistics missing for probed layers: {missing}")
    return {"shipped": shipped, "intended": intended}


def real_target(model: str, block_idx: int, layer_name: str, seed: int) -> dict:
    """Real production weight plus nominal statistics for the probe."""
    weight, _shard = load_layer_weight(model, block_idx, layer_name)
    rng = np.random.default_rng(seed)
    dout, din = weight.shape
    return {
        "w": weight,
        "d_in_raw": rng.uniform(0.5, 4.0, size=din),
        "d_out_raw": rng.uniform(0.5, 4.0, size=dout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="synthetic planted target only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jsonl", type=pathlib.Path, default=None, help="row stream, one JSON per line"
    )
    parser.add_argument("--out", type=pathlib.Path, default=None, help="report JSON path")
    parser.add_argument("--real", action="store_true", help="use the pinned production layers")
    parser.add_argument(
        "--device",
        default=None,
        help="accelerator device (the decisive run defaults to cuda, fail-closed)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=256,
        help="head-chunk size for the gradient statistic (0 disables chunking)",
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="statistics pass ONLY, no solves: the pre-fire memory rehearsal",
    )
    parser.add_argument(
        "--rehearsal-batches",
        type=int,
        default=4,
        help="batches the rehearsal feeds the statistics pass (peak is per-batch)",
    )
    args = parser.parse_args()

    probe_set = (("synthetic", 0, "planted-rank16"),) if args.smoke else PROBE_SET
    provider = None if args.smoke else real_target
    # --smoke is the fast path (smoke rank); the decisive run is production rank.
    cfg = PTQConfig() if args.smoke else production_config()
    stats = None
    device = None
    if not args.smoke:
        # The decisive fire is GPU-mandated (user directive): torch backend on
        # the accelerator, fp32 as the earning dtype — fail-closed, never a
        # quiet CPU fallback. compute_dtype is recorded per row and in
        # launch.env; the tolerance class covers fp32-vs-fp64 movement.
        device = args.device or "cuda"
        cfg = PTQConfig(rank=PRODUCTION_RANK, compute_dtype="float32")
        chunk_tokens = args.chunk_tokens or None
        if args.rehearsal:
            # Pre-fire rehearsal: the statistics pass and nothing else. This is
            # the phase that killed the box, so it is the phase that gets
            # measured -- inside the fence, before any solve starts. The peak is
            # a per-batch property, so a bounded batch count still reproduces a
            # per-batch defect.
            import torch  # noqa: PLC0415

            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            diag: dict[str, Any] = {}
            # Telemetry starts before the model is loaded, so the trace covers
            # the load and the tokenize pass as well as the statistics: the
            # drain is fast enough that a trace starting at the stats pass
            # would open after the interesting part.
            telemetry = (
                MemTelemetry(args.out.parent / "rehearsal-mem.csv", device)
                if args.out is not None
                else None
            )
            if telemetry is not None:
                telemetry.start()
            try:
                build_real_arms(
                    probe_set,
                    device=device,
                    chunk_tokens=chunk_tokens,
                    max_batches=args.rehearsal_batches,
                    diagnostics=diag,
                    telemetry=telemetry,
                )
            finally:
                if telemetry is not None:
                    telemetry.stop()
            peak_bytes = int(diag.get("peak_bytes", 0))
            record = {
                "phase": "rehearsal",
                "rehearsal": True,
                "sequences": diag.get("sequences"),
                "protocol_sequences": diag.get("protocol_sequences"),
                "packed_batches": diag.get("packed_batches"),
                "chunk_tokens": chunk_tokens,
                "rehearsal_batches": args.rehearsal_batches,
                "device": device,
                "grad_rows": diag.get("grad_rows"),
                "peak_bytes": peak_bytes,
                "peak_gb": round(peak_bytes / 1e9, 3),
                "cuda_available": bool(torch.cuda.is_available()),
                "attn_implementation": diag.get("attn_implementation"),
                "mem_csv": "rehearsal-mem.csv" if args.out is not None else None,
                "host_peak_gb": round(telemetry.peak_used / 1e9, 3) if telemetry else None,
                "host_peak_phase": telemetry.peak_phase if telemetry else None,
                "no_cuda_memory_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
            }
            print(f"[lane0] REHEARSAL {json.dumps(record, sort_keys=True)}")
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                (args.out.parent / "rehearsal.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n"
                )
                env_path = args.out.parent / "launch.env"
                with env_path.open("a") as handle:
                    handle.write(
                        f"rehearsal_peak_gb={record['peak_gb']}\n"
                        f"rehearsal_chunk_tokens={chunk_tokens}\n"
                        f"rehearsal_batches={args.rehearsal_batches}\n"
                        f"rehearsal={record['rehearsal']}\n"
                        f"rehearsal_sequences={record['sequences']}\n"
                        f"protocol_sequences={record['protocol_sequences']}\n"
                        f"host_peak_gb={record['host_peak_gb']}\n"
                        f"host_peak_phase={record['host_peak_phase']}\n"
                    )
            return 0
        stats = build_real_arms(probe_set, device=device, chunk_tokens=chunk_tokens)
    rows = run_arms(
        probe_set,
        seed=args.seed,
        target_provider=provider,
        cfg=cfg,
        stats=stats,
        device=device,
    )

    for row in rows:
        line = json.dumps(row)
        print(f"[lane0] {row['arm']} {row['layer_name']} cos={row['recon_cos']:.6f} {line}")
        if args.jsonl is not None:
            with args.jsonl.open("a") as handle:
                handle.write(line + "\n")

    report = build_report(rows, probe_set=probe_set)
    verdict = build_verdict(rows)
    report["verdict"] = verdict
    print(f"[lane0] {verdict['paragraph']}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
