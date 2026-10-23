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
import glob
import json
import os
import pathlib
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


def fold_shipped(
    w: np.ndarray, d_out: np.ndarray, d_in: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shipped fold, verbatim from ``block_recon.py:1350``: no sqrt."""
    w_tilde = (np.asarray(d_out)[:, None] * np.asarray(w, dtype=np.float64)) * np.asarray(d_in)[
        None, :
    ]
    return w_tilde, np.asarray(d_out), np.asarray(d_in)


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
            # arm A: shipped robust_diag + shipped raw fold
            d_in_a, d_out_a = robust_diag(arm_a["d_in_raw"], arm_a["d_out_raw"])
            # arm B: intended (clip -> sqrt fold, once per side)
            _, d_out_b, d_in_b = fold_intended(
                w, d_out_raw=arm_b["d_out_raw"], d_in_raw=arm_b["d_in_raw"]
            )
            percentile = arm_b.get("percentile", PERCENTILE)
            grad_fallback = arm_b.get("grad_fallback")
        else:
            # arm A: shipped robust_diag + shipped raw fold
            d_in_a, d_out_a = robust_diag(target["d_in_raw"], target["d_out_raw"])
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


def _hf_cache_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        roots.append(pathlib.Path(env))
    roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(pathlib.Path("/hf"))
    return [r for r in roots if r.exists()]


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
                if key in handle:
                    weight = handle.get_slice(key)[:, :].float().numpy().astype(np.float64)
                    return np.ascontiguousarray(weight), pathlib.Path(path).name
    raise FileNotFoundError(f"{key} not found in any offline HF cache root: {_hf_cache_roots()}")


def collect_intended_statistics(
    model: Any,
    calibration_batches: Any,
    *,
    wanted: tuple[tuple[int, str], ...],
    percentile: float = PERCENTILE,
    chunk_rows: int | None = None,
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
            acc, n = _clip_and_accumulate(inp[0])
            sum_sq_in[key] = acc if sum_sq_in[key] is None else sum_sq_in[key] + acc
            rows[key] += n

        handles.append(module.register_forward_pre_hook(_pre_hook))

    grads: dict[tuple[int, str], np.ndarray | None] = {key: None for _, _, key in linears}
    for _, module, key in linears:

        def _post_hook(mod, inp, out, key=key):  # noqa: ANN001
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
    with torch.enable_grad():
        for batch in calibration_batches:
            model.zero_grad(set_to_none=True)
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
            grad_rows += int(shift_labels.shape[0])
    for handle in handles:
        handle.remove()

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


def build_real_arms(
    probe_set: tuple[tuple[str, int, str], ...] = PROBE_SET,
    *,
    sequences: int = CALIBRATION_SEQS,
    seq_len: int = SEQ_LEN,
    seed: int = 0,
    percentile: float = PERCENTILE,
    device: str | None = None,
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

    text = load_wikitext_text(find_wikitext_shards("train"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    n_batches = packed_chunks(sequences, "--calibration-seqs")
    batches = tokenize_chunks(tokenizer, text, seq_len, n_batches, seed)
    resolved = sum(int(b.shape[0]) for b in batches)
    if resolved != sequences:
        raise ValueError(
            f"calibration geometry resolved to {resolved} sequences, expected {sequences}"
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    # Fail closed, never silently fall back: a "GPU" run that quietly computed on
    # CPU would report GPU timings for CPU work. Both collectors move the batches
    # to the model's parameter device themselves, so moving the model is the
    # whole device story here.
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device={device!r} requested but no accelerator is available")
        model = model.to(device)
    wanted = tuple((block, name) for _, block, name in probe_set)

    # Arm A: the shipped construction, verbatim -- same call the pipeline makes.
    book = collect_stats(model, batches)
    shipped: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in book:  # LayerStatsBook is iterable; _entries is private
        for block, name in wanted:
            if entry.name.endswith(f"layers.{block}.{name}"):
                shipped[(block, name)] = {
                    "d_in_raw": np.asarray(entry.d_in_raw, dtype=np.float64),
                    "d_out_raw": np.asarray(entry.d_out_raw, dtype=np.float64),
                    "percentile": None,
                    "grad_fallback": None,
                }

    # Arm B: the intended construction -- real CE backward, p99.9 clip first.
    intended = collect_intended_statistics(model, batches, wanted=wanted, percentile=percentile)

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
        stats = build_real_arms(probe_set, device=device)
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
