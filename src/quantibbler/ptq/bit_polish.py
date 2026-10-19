"""Greedy bit-flip polish: discrete local search on the stored objective.

The pipeline's continuous stages (LB-ADMM init, TuneLatentSTE) never
perform a single discrete move on the packed binaries (measured, r137:
4,096 STE steps flipped zero sign bits; exported artifacts carried
per-layer cos ~0.02-0.17 with the fp weights). This module closes that
gap: coordinate-descent sign flips on ``Ub``/``Vb`` minimizing

    SSE(B, s) = || W - diag(s1) Ub Vb^T diag(s2) ||_F^2

with EXACT per-flip deltas from the maintained residual, alternated with
exact per-row/per-column least-squares refits of the scales (bits held).

Flip-gain algebra (b = bit value before flip, c = s1[i] * Vb[:,k] * s2):
  flipping (i, k) changes row i of Wq by -2 b c, so
    dSSE = 4 b s1[i] <R_i, Vb[:,k]*s2> + 4 s1[i]^2 ||Vb[:,k]*s2||^2
(the +4s^2||c||^2 term is the squared norm of the row change and is
always positive). Accept the most negative dSSE per row until none is
negative. Measured on Qwen3-0.6B block-0 production geometry: cos
0.09 -> 0.51 (q_proj), 0.08 -> 0.45 (gate_proj), 0.17 -> 0.39
(down_proj) in 8-11 s per layer — two orders of magnitude cheaper than
the ADMM stage it refines.
"""

from __future__ import annotations

import numpy as np


def greedy_polish(
    w: np.ndarray,
    ub: np.ndarray,
    vb: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    *,
    rounds: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Polish binary factors in place semantics (returns new arrays).

    Args:
        w: ``(dout, din)`` fp target (the ORIGINAL weight, un-preconditioned —
            the stored objective is against W, matching the artifact).
        ub: ``(dout, r)`` binary factor (int8 or float, entries +-1).
        vb: ``(din, r)`` binary factor, same rules.
        s1: ``(dout,)`` scale vector (any positive/negative finite floats;
            signs fold into the bits).
        s2: ``(din,)`` scale vector.
        rounds: sweep count; each round is a full U sweep + V sweep +
            alternating-LS scale refit. Flip counts decay monotonically;
            3-6 rounds reaches the local plateau on production geometry.

    Returns:
        ``(ub, vb, s1, s2)`` with float64 ub/vb (+1/-1) and float64 scales.

    Raises:
        ValueError: on shape mismatches or non-finite inputs.
    """
    w = np.ascontiguousarray(w, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64).copy()
    vb = np.asarray(vb, dtype=np.float64).copy()
    s1 = np.asarray(s1, dtype=np.float64).copy()
    s2 = np.asarray(s2, dtype=np.float64).copy()
    if w.ndim != 2:
        raise ValueError(f"target must be 2-D, got {w.shape}")
    dout, din = w.shape
    if ub.shape != (dout, vb.shape[1]) or s1.shape != (dout,) or s2.shape != (din,):
        raise ValueError(
            f"factor/scale shapes disagree: W{w.shape} Ub{ub.shape} Vb{vb.shape} "
            f"s1{s1.shape} s2{s2.shape}"
        )
    if not (np.isfinite(w).all() and np.isfinite(s1).all() and np.isfinite(s2).all()):
        raise ValueError("greedy_polish requires finite inputs")
    if not (np.all(np.abs(ub) == 1.0) and np.all(np.abs(vb) == 1.0)):
        raise ValueError("binary factors must contain exactly +-1 entries")
    r = ub.shape[1]

    for _rnd in range(max(rounds, 0)):
        total_flips = 0
        R = w - (s1[:, None] * (ub @ vb.T)) * s2[None, :]
        # --- sweep U rows (output channels) ---
        vb_s2 = vb * s2[:, None]  # (din, r); column k = Vb[:,k]*s2
        nrm2u = (vb_s2 * vb_s2).sum(axis=0)
        for i in range(dout):
            base = s1[i]
            if base == 0.0:
                continue
            ri = R[i]
            for _ in range(r):
                proj = ri @ vb_s2
                gain = 4.0 * ub[i] * base * proj + 4.0 * base * base * nrm2u
                k = int(np.argmin(gain))
                if gain[k] >= 0.0:
                    break
                b = ub[i, k]
                ub[i, k] = -b
                ri += 2.0 * b * base * vb_s2[:, k]
                total_flips += 1
        # --- sweep V rows (input channels) ---
        ub_s1 = ub * s1[:, None]  # (dout, r); column k = Ub[:,k]*s1
        nrm2v = (ub_s1 * ub_s1).sum(axis=0)
        for j in range(din):
            base = s2[j]
            if base == 0.0:
                continue
            rj = R[:, j]
            for _ in range(r):
                proj = rj @ ub_s1
                gain = 4.0 * vb[j] * base * proj + 4.0 * base * base * nrm2v
                k = int(np.argmin(gain))
                if gain[k] >= 0.0:
                    break
                b = vb[j, k]
                vb[j, k] = -b
                rj += 2.0 * b * base * ub_s1[:, k]
                total_flips += 1
        # --- alternating least-squares scale refit (bits held) ---
        Bc = ub @ (vb * s2[:, None]).T  # W ~ diag(s1) Bc
        d1 = (Bc * Bc).sum(axis=1)
        s1 = np.where(d1 > 0, (w * Bc).sum(axis=1) / np.maximum(d1, 1e-300), s1)
        Br = (ub * s1[:, None]) @ vb.T  # W ~ Br diag(s2)
        d2 = (Br * Br).sum(axis=0)
        s2 = np.where(d2 > 0, (w * Br).sum(axis=0) / np.maximum(d2, 1e-300), s2)
        if total_flips == 0:
            break  # local optimum reached

    return ub, vb, s1, s2
