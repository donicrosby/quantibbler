"""Quantized tiled-attention NumPy reference — the executable specification.

Change: ``qk-sage-attention``, task 1.2. Implements [SA] §4 (SageAttention, the
INT8 attention scheme) on top of the tiled loop of
:mod:`quantibbler.kernels.attention_tiled` and is the oracle the device INT8
kernel is measured against. It reuses that module's shape contract (``_as_4d``,
``_is_positive_int``, ``_DEFAULT_BLOCK``) so the two faces cannot drift on the
pinned engine conventions; the docstring there is the reference for the
``(B, n_heads, S, d)`` layout, GQA grouping, ``1/√d`` softmax scale, causal
blocking and additive-bias contract.

The scheme ([SA] §4)
--------------------
- **γ(K) = K − mean(K)** ([SA] §4.2, Eq. 6): K's channel-wise outliers are a
  per-key bias shared by all tokens, so subtracting the mean removes them
  *without loss* — ``σ(q(K − mean(K))ᵀ) = σ(qKᵀ)`` for any query row, because
  the subtracted term is a constant per query. **Per K-block** is enough: the
  mean is a constant *within* the tile currently in flight, so the kernel never
  needs a whole-cache pass (the derived property recorded in this change's
  ``design.md``). The reference therefore also *unsmooths* — it adds the block's
  shift back into the scores — which is what keeps ``m``/``ℓ`` comparable across
  key blocks and is exactly the "dequantized-then-unsmoothed score path" the
  ``gamma-shift-invariance`` scenario is written against.
- **Granularity and dtypes** ([SA] §4.3, Table 6 SAGEAttn-vT row with an f32
  accumulator per Table 4): Q and K per-token INT8 (per-channel is structurally
  impossible at the inner axis of QKᵀ), P̃ per-block at the **static** scale
  ``1/127`` (P̃ = exp(S − rowmax) has row max 1 by construction), V per-channel
  INT8 (it carries the channel-wise outlier), statistics and accumulator in
  float64 here — the reference's arithmetic is f64 while the kernel's is f32,
  which is what the parity budget is for.
- **Structure** is [SA] Eq. 4/5 verbatim: ``(δ_Q, Q̂) = ψ_Q(Q/√d)`` (the 1/√d is
  folded into Q's quantizer, [SA] §4.6), ``(δ_K, K̂) = ψ_K(γ(K))``,
  ``S = ψ⁻¹_{δ_Qδ_K}(Q̂K̂ᵀ) = δ_Q δ_K (Q̂K̂ᵀ)``, and
  ``O ← diag(exp(m′−m))O + ψ⁻¹_{δ_Pδ_V}(P̂V̂)``. The matmuls are integer
  accumulations scaled *after* the dot (the INT8 dot is [SA]'s whole point), and
  the online softmax statistics stay full precision, as Eq. 5 states.

Deliberate choices, named rather than silent
--------------------------------------------
- **P̃ is quantized after the running-max rescaling** (``exp(S − m_new)``, the
  FlashAttention form the tiled reference uses), not against the block's own
  rowmax. Either way ``|P̃| ≤ 1``, so the paper's static ``1/127`` scale still
  covers the range; this keeps the reference on the same loop shape as the
  kernel.
- **ℓ uses the full-precision P̃** while the ``P̂V̂`` product uses the quantized
  one — that is [SA] Eq. 5 literally (``(m′, P) = σ̃(m, S)`` full precision,
  only the product quantized).
- **V is quantized per channel *within the tile*** (one scale per channel of the
  V-block the inner loop has loaded), not over the whole cache: a tile-local
  scale is the only form the kernel can compute without a whole-cache pass, and
  the scale's axis (the outer axis of P̃Vᵀ) is what [SA] §4.3 reasons about.
- **A zero-range row or channel keeps a positive, defined scale** (``δ = 1.0``,
  all-zero codes) rather than dividing by zero: its value cannot change any
  output (zero codes dequantize to zero for *any* δ), while leaving δ undefined
  would poison the tile. This is load-bearing at ragged tails and at
  ``block_size=1``, where a one-token K-block has ``mean(K) = K`` hence
  ``γ(K) ≡ 0``; there the shift added back restores the exact keys, so the score
  path is exact rather than merely small.
- **Quantizer outputs are exposed** (``SageStats``/``SageBlockStats``: scales,
  smoothed K, int8 tiles, and the score path the softmax actually consumed), as
  the ``qk-kernels`` delta requires, so probes can assert each stage against the
  paper's algebra instead of trusting the end-to-end number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .attention_tiled import _DEFAULT_BLOCK, _as_4d, _is_positive_int

__all__ = [
    "INT8_MAX",
    "PTILDE_STATIC_SCALE",
    "SageBlockStats",
    "SageStats",
    "attention_sage",
    "dequantize_int8",
    "quantize_p_tilde",
    "quantize_per_channel",
    "quantize_per_token",
    "smooth_k_block",
]

#: INT8 symmetric range, the quantizers' full-scale ([SA] §3.2: ``δ = max|A|/127``).
INT8_MAX = 127
#: [SA] §4.3's static scale for P̃: ``exp(S − rowmax)`` has row max 1 by construction.
PTILDE_STATIC_SCALE = 1.0 / 127.0


@dataclass(eq=False)
class SageBlockStats:
    """Quantizer state and score path for one (query block, key block) tile.

    Everything here is a *record of what ran*: the probes assert the algebra
    against these rather than against a re-derivation.
    """

    i0: int
    i1: int
    j0: int
    j1: int
    #: γ's shift, per kv head and channel: ``(B, n_kv, 1, d)``.
    k_mean: np.ndarray
    #: γ(K) before quantization: ``(B, n_kv, kj, d)``.
    k_smoothed: np.ndarray
    #: ψ_K's output and per-token scale: ``int8 (B, n_kv, kj, d)``, ``(B, n_kv, kj, 1)``.
    k_hat: np.ndarray
    k_scale: np.ndarray
    #: ψ⁻¹(ψ(γ(K))) and the same with the shift added back.
    k_dequant: np.ndarray
    k_unsmoothed: np.ndarray
    #: The scores the softmax consumed: ``(B, n_heads, qi, kj)``, causal-masked and biased.
    scores_unsmoothed: np.ndarray
    #: ψ_P's output at the static scale: ``int8 (B, n_heads, qi, kj)``.
    p_hat: np.ndarray
    #: ψ_V's output and per-channel scale: ``int8 (B, n_kv, kj, d)``, ``(B, n_kv, 1, d)``.
    v_hat: np.ndarray
    v_scale: np.ndarray
    p_scale: float = PTILDE_STATIC_SCALE


@dataclass(eq=False)
class SageStats:
    """The reference's quantizer outputs, exposed for probes ([SA] §4.3 axes)."""

    #: ψ_Q's output and per-token scale: ``int8 (B, n_heads, S, d)``, ``(B, n_heads, S, 1)``.
    q_hat: np.ndarray
    q_scale: np.ndarray
    q_dequant: np.ndarray
    blocks: tuple[SageBlockStats, ...]


def smooth_k_block(k_block: object) -> tuple[np.ndarray, np.ndarray]:
    """γ per K-block: ``(K − mean(K), mean(K))`` ([SA] §4.2, Eq. 6).

    The mean is over the tokens **of this block** — a constant shift per query
    row, which softmax is invariant to, so any K-block in flight is a valid
    place to take it.
    """
    kb = np.asarray(k_block, dtype=np.float64)
    mean = kb.mean(axis=-2, keepdims=True)
    return kb - mean, mean


def _dynamic_scale(peak: np.ndarray) -> np.ndarray:
    """[SA] §3.2's ``δ = max|A|/127``, with the zero-range convention.

    A row (or channel) whose peak is zero quantizes to all-zero codes and back to
    exactly zero **for any** scale, so the scale's value cannot affect a single
    output — but it must be *defined* and strictly positive for the tile to be a
    tile at all (the ``quantized-budget`` scenario asserts exactly that). The
    reference therefore fixes ``δ = 1.0`` there instead of dividing by zero. This
    is not a corner-case detail: a one-token K-block has ``γ(K) ≡ 0`` by
    construction, and a token equal to its block's mean is zero range in any
    block size — in both cases the shift added back on the unsmoothed path
    restores the exact keys.
    """
    return np.where(peak > 0.0, peak / INT8_MAX, 1.0)


def quantize_per_token(x: object) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-token INT8 quantizer ψ ([SA] §3.2: ``δ = max|A|/127``).

    One scale per token row (shared across the row's channels — the outer axis
    of QKᵀ, the only axis [SA] §4.3 makes available for Q and K).
    """
    xf = np.asarray(x, dtype=np.float64)
    delta = _dynamic_scale(np.max(np.abs(xf), axis=-1, keepdims=True))
    hat = np.clip(np.rint(xf / delta), -INT8_MAX, INT8_MAX)
    return hat.astype(np.int8), delta


def quantize_per_channel(x: object) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel INT8 quantizer ψ for V ([SA] §3.2, §4.3).

    The channel axis is the last one; the scale is taken over the token axis
    (``-2``), i.e. one scale per channel of the tile — the axis P̃Vᵀ's
    dequantization requires. Zero-range channels follow the same convention as
    zero-range rows (see :func:`_dynamic_scale`): all-zero codes, unit scale.
    """
    xf = np.asarray(x, dtype=np.float64)
    delta = _dynamic_scale(np.max(np.abs(xf), axis=-2, keepdims=True))
    hat = np.clip(np.rint(xf / delta), -INT8_MAX, INT8_MAX)
    return hat.astype(np.int8), delta


def quantize_p_tilde(p: object) -> tuple[np.ndarray, float]:
    """ψ_P: per-block INT8 at the static scale ``1/127`` ([SA] §4.3 reason 2).

    ``P̃ = exp(S − rowmax)`` has row max 1, so a single static scale covers every
    block and the paper records its accuracy as equal to per-token quantization.
    """
    pf = np.asarray(p, dtype=np.float64)
    hat = np.clip(np.rint(pf / PTILDE_STATIC_SCALE), 0, INT8_MAX)
    return hat.astype(np.int8), PTILDE_STATIC_SCALE


def dequantize_int8(hat: object, scale: object) -> np.ndarray:
    """ψ⁻¹: element-wise rescale ([SA] §3.2: ``ψ⁻¹_δ(Â) = δÂ``)."""
    return np.asarray(hat, dtype=np.float64) * np.asarray(scale, dtype=np.float64)


def attention_sage(
    q: object,
    k: object,
    v: object,
    *,
    causal: bool = False,
    mask: object | None = None,
    block_size: int = _DEFAULT_BLOCK,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, SageStats]:
    """Quantized tiled attention forward ([SA] §4 on [FA] §3.1 Algorithm 1).

    Returns the attention output ``(B, n_heads, S, val_dim)``; with
    ``return_stats=True`` also returns the :class:`SageStats` probe record.
    """
    if not _is_positive_int(block_size):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    block = int(block_size)

    q = _as_4d("q", q)
    k = _as_4d("k", k)
    v = _as_4d("v", v)
    b, n_heads, seq_q, d = q.shape
    if k.shape[0] != b or v.shape[0] != b:
        raise ValueError(f"batch mismatch: q {q.shape}, k {k.shape}, v {v.shape}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"k/v head-count mismatch: k {k.shape[1]}, v {v.shape[1]}")
    n_kv = int(k.shape[1])
    if k.shape[2] != seq_q or v.shape[2] != seq_q:
        raise ValueError(
            f"k/v sequence length must match q ({seq_q}), got k {k.shape[2]}, v {v.shape[2]}"
        )
    if k.shape[3] != d:
        raise ValueError(f"head-dim mismatch between q and k: q d={d}, k {k.shape[3]}")
    val_dim = int(v.shape[3])
    if n_kv <= 0 or n_heads % n_kv != 0:
        raise ValueError(f"n_kv_heads must divide n_heads: n_heads={n_heads}, n_kv={n_kv}")

    bias = None
    if mask is not None:
        bias = np.asarray(mask, dtype=np.float64)
        try:
            np.broadcast_shapes(bias.shape, (b, n_heads, seq_q, seq_q))
        except ValueError as exc:
            raise ValueError(
                f"mask shape {bias.shape} is not broadcastable to {(b, n_heads, seq_q, seq_q)}"
            ) from exc

    # [SA] §4.6: the 1/√d coefficient is folded into Q's quantizer rather than
    # left in the attention loop, so ψ_Q takes Q/√d (Eq. 4).
    q_hat, q_scale = quantize_per_token(q * (1.0 / math.sqrt(d)))
    q_dequant = dequantize_int8(q_hat, q_scale)

    group = n_heads // n_kv
    kv_index = np.repeat(np.arange(n_kv), group)

    out = np.zeros((b, n_heads, seq_q, val_dim), dtype=np.float64)
    blocks: list[SageBlockStats] = []

    for i0 in range(0, seq_q, block):
        i1 = min(i0 + block, seq_q)
        rows = np.arange(i0, i1)
        q_tile_hat = q_hat[:, :, i0:i1, :]
        q_tile_scale = q_scale[:, :, i0:i1, :]
        q_tile_dequant = q_dequant[:, :, i0:i1, :]
        qi = i1 - i0
        m = np.full((b, n_heads, qi, 1), -np.inf, dtype=np.float64)
        ell = np.zeros((b, n_heads, qi, 1), dtype=np.float64)
        acc = np.zeros((b, n_heads, qi, val_dim), dtype=np.float64)

        j_stop = min(i1, seq_q) if causal else seq_q
        for j0 in range(0, j_stop, block):
            j1 = min(j0 + block, seq_q)
            cols = np.arange(j0, j1)

            # γ per K-block, then ψ_K per token on the smoothed tile ([SA] §4.2, §4.3).
            k_kv = k[:, :, j0:j1, :]
            k_smoothed, k_mean = smooth_k_block(k_kv)
            k_hat_kv, k_scale_kv = quantize_per_token(k_smoothed)
            k_dequant_kv = dequantize_int8(k_hat_kv, k_scale_kv)
            k_unsmoothed_kv = k_dequant_kv + k_mean

            # GQA: quantization happens once per kv head, on the tile the inner
            # loop loads; the scores live in query-head space.
            k_hat = k_hat_kv[:, kv_index]
            k_scale = k_scale_kv[:, kv_index]
            k_mean_h = k_mean[:, kv_index]

            # S = ψ⁻¹_{δ_Q δ_K}(Q̂K̂ᵀ) — the integer dot first ([SA] Eq. 5), then
            # the two scales; the shift is added back so m/ℓ stay comparable
            # across key blocks (γ is loss-free, [SA] §4.2).
            dot = q_tile_hat.astype(np.float64) @ k_hat.astype(np.float64).swapaxes(-1, -2)
            # δ_Q multiplies along the query axis (qi, 1); δ_K along the key axis
            # (1, kj) — the scale vectors are stored row-wise with the tiles they
            # belong to, so K's is transposed for the score layout.
            scores = dot * q_tile_scale * k_scale.swapaxes(
                -1, -2
            ) + q_tile_dequant @ k_mean_h.swapaxes(-1, -2)
            if causal:
                scores = np.where(cols[None, :] > rows[:, None], -np.inf, scores)
            if bias is not None:
                scores = scores + bias[..., i0:i1, j0:j1]

            m_new = np.maximum(m, np.max(scores, axis=-1, keepdims=True))
            # A row with no admissible score keeps m = -inf and ℓ = 0 exactly (the
            # tiled reference's contract: the output's NaN is the same 0/0).
            empty = np.isneginf(m_new)
            with np.errstate(invalid="ignore"):
                alpha = np.where(empty, 0.0, np.exp(m - m_new))
                p = np.where(empty, 0.0, np.exp(scores - m_new))
            ell = alpha * ell + np.sum(p, axis=-1, keepdims=True)

            # ψ_P (static 1/127) and ψ_V (per channel); O = diag(exp(m'-m))O + ψ⁻¹_{δ_Pδ_V}(P̂V̂).
            p_hat, p_scale = quantize_p_tilde(p)
            v_hat_kv, v_scale_kv = quantize_per_channel(v[:, :, j0:j1, :])
            pv = p_hat.astype(np.float64) @ v_hat_kv[:, kv_index].astype(np.float64)
            contribution = pv * p_scale * v_scale_kv[:, kv_index]
            acc = alpha * acc + np.where(empty, 0.0, contribution)
            m = m_new

            blocks.append(
                SageBlockStats(
                    i0=i0,
                    i1=i1,
                    j0=j0,
                    j1=j1,
                    k_mean=k_mean,
                    k_smoothed=k_smoothed,
                    k_hat=k_hat_kv,
                    k_scale=k_scale_kv,
                    k_dequant=k_dequant_kv,
                    k_unsmoothed=k_unsmoothed_kv,
                    scores_unsmoothed=scores.copy(),
                    p_hat=p_hat,
                    v_hat=v_hat_kv,
                    v_scale=v_scale_kv,
                )
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, :, i0:i1, :] = acc / ell

    if not return_stats:
        return out
    return out, SageStats(q_hat=q_hat, q_scale=q_scale, q_dequant=q_dequant, blocks=tuple(blocks))
