"""Tiled exact-attention NumPy reference — the executable specification.

Change: ``qk-flash-attention``, task 1.2. Implements [FA] §3.1 Algorithm 1
(tiled FlashAttention forward): outer loop over query blocks, inner loop over
key/value blocks, with the running statistics ``m`` (row maximum) and ``ℓ``
(row sum of exponentials) and the rescaling identity

    O ← diag(exp(m_old − m_new)) · O + exp(S_ij − m_new) · V_j

so the softmax is never materialized over the full row. Float64 throughout:
this module is the *oracle* the GPU face is measured against (parity budget
1e-5 widened from f32), and the tolerance gates only work if the reference is
exact to round-off rather than merely close.

Conventions (pinned engine conventions, qk-server spec)
------------------------------------------------------
- ``q``: ``(B, n_heads, S, d)`` post-RoPE queries; ``k``:
  ``(B, n_kv_heads, S, d)`` post-RoPE keys; ``v``:
  ``(B, n_kv_heads, S, val_dim)`` post-norm values. ``val_dim`` may differ
  from ``d`` (the substrate exposes them separately and a value projection
  need not be square): the softmax scale is ``1/√d`` — the *query/key*
  dimension — and the output carries ``val_dim``.
- GQA: ``n_heads // n_kv_heads`` query heads share one kv head in
  consecutive groups (``repeat_interleave``); head ``h`` reads kv head
  ``h // group``. ``n_heads == n_kv_heads`` is the dense MHA case.
- Scores are scaled by ``1/√d`` (no softmax-scale field exists in the
  substrate, so the scale is applied to Q·Kᵀ here, exactly as the GPU face
  pre-scales Q).
- ``causal=True`` blocks strictly-upper positions (``j > i``): they are
  removed before the row maximum, so they contribute *exactly* zero rather
  than a small exponential.
- ``mask`` is an additive float64 bias broadcastable to ``(B, H, S, S)``,
  added after the causal blocking and before the softmax (the substrate's
  ``masked``/per-tensor mask contract).

Ragged tails: the sequence length need not be a multiple of ``block_size``;
the final query and key blocks are truncated, and any block size produces the
same result to round-off (that equivalence is what ``tests/test_attention_tiled.py``
verifies across blocks 1, 3, 4, 8, 64, 128).

Contract edges: every row of the causal/normal case has at least one finite
score, so ``ℓ > 0``. A mask that removes *every* score in a row leaves
``m = -inf, ℓ = 0`` and the output is NaN, deliberately matching the dense
formula's ``0/0`` rather than inventing a fallback the GPU face would not
reproduce; ``return_lse`` reports ``-inf`` for that row. Both behaviours are
pinned by the fully-masked-row scenario, which is checked as a NaN *pattern*
match rather than through the finite-output roundoff gate.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["attention_tiled"]

_DEFAULT_BLOCK = 64


def _is_positive_int(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) > 0
    return False


def _as_4d(name: str, x: object) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError(f"{name} must be 4-D (B, H, S, D) in float64, got shape {a.shape}")
    return a


def attention_tiled(
    q: object,
    k: object,
    v: object,
    *,
    causal: bool = False,
    mask: object | None = None,
    block_size: int = _DEFAULT_BLOCK,
    return_lse: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Tiled attention forward ([FA] §3.1 Algorithm 1), float64, shape-exact.

    Returns the attention output ``(B, n_heads, S, val_dim)``; with
    ``return_lse=True`` also returns the log-sum-exp ``(B, n_heads, S)`` of
    the masked scores — the running ``m + log(ℓ)`` statistic, which the GPU
    face carries for backward and which the parity battery compares.
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

    group = n_heads // n_kv
    # kv index per query head (repeat_interleave); a gather, never an N² copy.
    kv_index = np.repeat(np.arange(n_kv), group)
    scale = 1.0 / math.sqrt(d)

    out = np.zeros((b, n_heads, seq_q, val_dim), dtype=np.float64)
    lse = np.full((b, n_heads, seq_q), -np.inf, dtype=np.float64)

    for i0 in range(0, seq_q, block):
        i1 = min(i0 + block, seq_q)
        rows = np.arange(i0, i1)
        q_tile = q[:, :, i0:i1, :]
        m = np.full((b, n_heads, i1 - i0, 1), -np.inf, dtype=np.float64)
        ell = np.zeros((b, n_heads, i1 - i0, 1), dtype=np.float64)
        acc = np.zeros((b, n_heads, i1 - i0, val_dim), dtype=np.float64)

        # Causal: every key block starting at or after i1 is entirely masked
        # out, so it is skipped rather than exponentiated to zero.
        j_stop = min(i1, seq_q) if causal else seq_q
        for j0 in range(0, j_stop, block):
            j1 = min(j0 + block, seq_q)
            cols = np.arange(j0, j1)
            k_tile = k[:, kv_index, j0:j1, :]
            v_tile = v[:, kv_index, j0:j1, :]

            scores = (q_tile @ k_tile.swapaxes(-1, -2)) * scale
            if causal:
                blocked = cols[None, :] > rows[:, None]
                scores = np.where(blocked, -np.inf, scores)
            if bias is not None:
                scores = scores + bias[..., i0:i1, j0:j1]

            m_new = np.maximum(m, np.max(scores, axis=-1, keepdims=True))
            # A row whose every score is masked keeps m = -inf, and
            # exp(-inf - (-inf)) is NaN — which would poison the *statistic*
            # as well as the output. The fully-masked-row scenario fixes the
            # statistic instead: such a row has m = -inf and l = 0 exactly
            # (an empty sum), so the output's NaN comes from the same 0/0 the
            # dense formula produces rather than from NaN propagation, and
            # ``return_lse`` reports -inf rather than NaN.
            empty = np.isneginf(m_new)
            with np.errstate(invalid="ignore"):
                alpha = np.where(empty, 0.0, np.exp(m - m_new))
                weights = np.where(empty, 0.0, np.exp(scores - m_new))
            ell = alpha * ell + np.sum(weights, axis=-1, keepdims=True)
            acc = alpha * acc + weights @ v_tile
            m = m_new

        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, :, i0:i1, :] = acc / ell
            lse[:, :, i0:i1] = (m + np.log(ell))[..., 0]

    if return_lse:
        return out, lse
    return out
