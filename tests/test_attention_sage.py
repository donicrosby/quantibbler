"""Quantized tiled-attention NumPy reference (change ``qk-sage-attention``).

Scenario IDs from ``openspec/changes/qk-sage-attention/specs/qk-kernels/spec.md``:
``gamma-shift-invariance`` and ``quantized-budget``. The module under test is
``quantibbler.kernels.attention_sage``: the [SA] §4 scheme (INT8 Q/K/P̃/V on top
of the [FA] §3.1 tiled loop), which is the oracle the device INT8 kernel is
measured against.

What each half of the battery is for
------------------------------------
``gamma-shift-invariance`` — γ(K) = K − mean(K) ([SA] §4.2, Eq. 6) is
*loss-free*: for any constant shift, σ(q(K − c)ᵀ) = σ(qKᵀ), so the reference is
allowed to smooth per K-block (that is the algebra that keeps the kernel from
ever needing a whole-cache pass) and must undo the shift exactly. The scenario
states the equality at f64 round-off (≤ 1e-12 relative) — which is only
attainable when the quantizer is itself lossless, so the battery has two parts:

* ``..._lossless_battery`` — a fixture built on the INT8 grid (each K-block's
  column sums are exactly zero after a shared per-channel bias is added, and
  every row peaks at exactly ±127, so δ_K = 127/127 = 1.0 and ψ⁻¹(ψ(γ(K))) is
  BIT-identical to γ(K)). That is the fixture on which the scenario's ≤ 1e-12
  clause is literally true, and it is a sharp test: a wrong mean, a missing
  shift-back, or a block boundary that ignores the ragged tail lands at O(1).
* ``..._int8_only`` — general fixtures, where the same path cannot be exact.
  There the assertion is the scenario's own attribution: the unsmoothed path is
  *exactly* the path of the dequantized operands (δ_Q δ_K (Q̂K̂ᵀ) + q̃·meanᵀ, per
  [SA] Eq. 5), so the deviation from the exact scores is the operands' INT8
  representation error and nothing else — γ and the shift contribute zero.

``quantized-budget`` — [SA] §4.3's own accuracy metrics (Cosine Sim, Relative
L1, RMSE) against the EXACT tiled reference, plus the strictly-positive scale /
no-zero-range-token clause. [SA] Table 3 is why the gate is on our activations
and not on the paper's tables: worst-case INT8 layers degrade to 56–79% cos-sim
when the outlier handling is wrong.

Battery: the [FA] fixture shapes (seq 8/10/64, d 4/128, heads (2,1)/(8,2)) with
the same block sizes the tiled battery uses, ``1`` included. ``block_size=1`` is
where per-block γ is *trivially* loss-free: a one-token K-block has
mean(K_block) = K_block, hence γ(K) ≡ 0, hence a zero-range row whose scale the
reference defines as ``1.0`` with all-zero codes (see ``_dynamic_scale``). The
shift added back then restores K exactly, so even that block size has an exact
score path — which is why the strictly-positive-scale clause of
``quantized-budget`` is satisfiable for every fixture rather than only for
convenient ones.

Lean tier: NumPy only (no torch), like the tiled battery it extends.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from quantibbler.kernels.attention_sage import (
    INT8_MAX,
    PTILDE_STATIC_SCALE,
    attention_sage,
    dequantize_int8,
    quantize_p_tilde,
    quantize_per_channel,
    quantize_per_token,
    smooth_k_block,
)
from quantibbler.kernels.attention_tiled import attention_tiled

RTOL = 1e-12

# (seq, d, n_heads, n_kv_heads) — the [FA] fixture battery, unchanged.
SHAPES = [
    (8, 4, 2, 1),
    (8, 128, 8, 2),
    (10, 4, 8, 2),
    (10, 128, 2, 1),
    (64, 4, 2, 1),
    (64, 128, 8, 2),
]

# The tiled battery's block sizes, unchanged: 1 and 3 divide none of {8, 10, 64}
# (ragged tails on every shape), 64/128 cover single-block and larger-than-seq.
BLOCK_SIZES = [1, 3, 4, 8, 64, 128]

# (shape, block) pairs for the general battery. ``block_size=1`` is dropped on the
# long shapes only: at seq=64 a case is 2080 (query block, key block) tiles of
# per-block assertions — minutes under co-tenant load — for a branch (γ ≡ 0 and
# so a zero-range K row) that the short shapes and
# ``test_attention_sage_contract_edges`` already pin symbol-for-symbol. Every
# other block size runs on every shape, ragged tails included.
BATTERY = [
    (shape, block)
    for shape in SHAPES
    for block in BLOCK_SIZES
    if not (block == 1 and shape[0] > 16)
]
BATTERY_IDS = [f"s{s[0]}-d{s[1]}-h{s[2]}-kv{s[3]}-b{b}" for s, b in BATTERY]

# [SA] §4.3 gate numbers, adopted verbatim from the spec delta.
COS_SIM_MIN = 0.999
RELATIVE_L1_MAX = 0.02


def _fixtures(seq: int, d: int, n_heads: int, n_kv: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    b = 2
    q = rng.standard_normal((b, n_heads, seq, d))
    k = rng.standard_normal((b, n_kv, seq, d))
    v = rng.standard_normal((b, n_kv, seq, d))
    return q, k, v


def _repeat_kv(x: np.ndarray, n_heads: int) -> np.ndarray:
    """GQA broadcast: repeat each kv head consecutively (``repeat_interleave``)."""
    return np.repeat(x, n_heads // x.shape[1], axis=1)


def _additive_mask(seq: int, seed: int) -> np.ndarray:
    """Broadcastable additive bias: (1, 1, seq, seq), both signs, no −inf."""
    return np.random.default_rng(seed).standard_normal((1, 1, seq, seq)) * 0.5


def _scores_block(
    q_op_block: np.ndarray,
    k_op_block: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    causal: bool,
    mask_block: np.ndarray | None,
) -> np.ndarray:
    """The block's softmax argument, built from explicit quantizer operands.

    Both operands are already sliced to the block under test (``q_op_block``
    ``(B, n_heads, qi, d)``, ``k_op_block`` ``(B, n_kv, kj, d)``) and already
    dequantized — ``q_op_block`` therefore carries [SA] §4.6's folded 1/√d, so
    this is exactly ``q_op @ k_opᵀ`` plus the block's masking. The reference's
    exposed score path must be this, and nothing else.
    """
    k_rep = _repeat_kv(k_op_block, q_op_block.shape[1])
    s = q_op_block @ k_rep.swapaxes(-1, -2)
    if causal:
        s = np.where(cols[None, :] > rows[:, None], -np.inf, s)
    if mask_block is not None:
        s = s + mask_block
    return s


def _assert_roundoff(a: np.ndarray, b: np.ndarray, rtol: float = RTOL) -> None:
    """Round-off agreement on the entries both sides define (finite) entries.

    Same two-part shape the tiled battery uses: an infinity-norm relative bound,
    plus a per-element bound restricted to the band within ``1e-2`` of the row
    max where the relative measure is well-conditioned. The ``−inf`` pattern of
    the causal/masked block rows is asserted separately, so a path that masked
    the wrong positions cannot pass by comparing only the shared finite part.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), (
        "score paths disagree about which entries are defined: "
        f"{int(np.isfinite(a).sum())} finite vs {int(np.isfinite(b).sum())}"
    )
    a = a[np.isfinite(b)]
    b = b[np.isfinite(b)]
    norm = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    assert np.max(np.abs(a - b)) <= rtol * norm, (
        f"differs beyond inf-norm rel {rtol}: {np.max(np.abs(a - b)) / norm:.3e} (norm {norm:.3e})"
    )
    row_max = np.max(np.abs(b), axis=-1, keepdims=True)
    strong = np.abs(b) >= 1e-2 * row_max
    if strong.any():
        rel = np.abs(a - b)[strong] / np.abs(b)[strong]
        assert np.all(rel <= rtol), (
            f"differs beyond per-element rel {rtol} on well-conditioned entries: "
            f"max {np.max(rel):.3e}"
        )


def _cos_sim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """[SA] §4.3: ``Σ OO' / (√ΣO² √ΣO'²)`` over the flattened outputs."""
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _relative_l1(reference: np.ndarray, candidate: np.ndarray) -> float:
    """[SA] §4.3: ``Σ|O − O'| / Σ|O|``."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(a)))


def _rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """[SA] §4.3: ``√((1/n)Σ(O − O')²)``."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Lossless fixture: γ(K) exactly on the INT8 grid
# ---------------------------------------------------------------------------


def _lossless_block(rows: int, d: int, flip: np.ndarray) -> np.ndarray:
    """Integer block, zero column sums, every row peaking at exactly ±127.

    [SA] §4.2's motivating shape is inverted on purpose: instead of a large
    shared bias plus a small signal, this is a small signal that γ leaves
    untouched, so ψ⁻¹(ψ(γ(K))) is bit-identical to γ(K) and the scenario's
    ≤ 1e-12 clause becomes attainable. ``rows % 2 == 0`` pairs (+127, −127) in
    column 0; an odd block spends one extra column on the leftover row. A
    one-token block has no zero-sum solution *and needs none*: its γ is
    identically zero, so any ±127 row is exactly restorable by the shift.
    """
    z = np.zeros((rows, d), dtype=np.float64)
    if rows == 1:
        z[0, 0] = 127.0
        return z
    paired = rows - (rows % 2)
    z[0:paired:2, 0] = 127.0 * flip
    z[1:paired:2, 0] = -127.0 * flip
    if paired != rows:
        z[rows - 1, 1] = 127.0
        z[0, 1] = -127.0
    return z


def _lossless_k(seq: int, d: int, n_kv: int, block: int, seed: int, bias: np.ndarray) -> np.ndarray:
    """``(2, n_kv, seq, d)`` keys whose γ is exactly representable per block.

    Each key block carries a large shared bias (the per-channel outlier [SA] §4.2
    describes) *off by a block-dependent multiple*, so the per-block means differ
    from one another and from the whole-cache mean: a γ that smoothed against the
    whole cache would leave a large, non-grid residual and fail the losslessness
    assertion instead of passing this battery vacuously.
    """
    rng = np.random.default_rng(seed)
    per_block = []
    for j0 in range(0, seq, block):
        rows = min(block, seq - j0)
        flip = rng.integers(0, 2, size=(rows - rows % 2) // 2) * 2 - 1
        block_signal = _lossless_block(rows, d, flip.astype(np.float64))
        per_block.append(block_signal + (bias + 64.0 * (j0 // block + 1))[None, :])
    z = np.concatenate(per_block, axis=0)
    return np.broadcast_to(z, (2, n_kv, seq, d)).copy()


def _grid_bias(d: int) -> np.ndarray:
    """A large per-channel bias, integer-valued so every f64 step is exact."""
    return 512.0 * np.arange(1, d + 1, dtype=np.float64)


def _on_grid(d: int, seq: int, n_heads: int, seed: int) -> np.ndarray:
    """``(2, n_heads, seq, d)`` queries that quantize losslessly after 1/√d.

    ``q = √d · z`` with ``z`` integer and per-row peak 127 makes ``q/√d`` land
    on the same grid, so Q's quantizer contributes only the division's round-off
    (exactly zero for d = 4).
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(-INT8_MAX, INT8_MAX, size=(2, n_heads, seq, d)).astype(np.float64)
    # Force a ±127 peak per row so δ_Q = 127/127 is exact.
    z[:, :, :, 0] = 127.0 * np.where(rng.integers(0, 2, size=(2, n_heads, seq)) == 1, 1.0, -1.0)
    return z * math.sqrt(d)


# (seq, block) pairs for the lossless battery: every block size the general
# battery uses, including the ragged tails (8 % 3 = 2, 10 % 4 = 2, 10 % 8 = 2) and
# the single-token blocks where γ ≡ 0 by construction.
LOSSLESS_PAIRS = [
    (8, 1),
    (8, 3),
    (8, 8),
    (8, 128),
    (10, 1),
    (10, 4),
    (10, 8),
    (64, 3),
    (64, 8),
    (64, 128),
]


# ---------------------------------------------------------------------------
# gamma-shift-invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "addmask"])
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
@pytest.mark.parametrize("d", [4, 128], ids=["d4", "d128"])
@pytest.mark.parametrize("seq,block", LOSSLESS_PAIRS)
def test_scn_gamma_shift_invariance_lossless_battery(seq, block, d, causal, masked):
    """gamma-shift-invariance: the dequantized-then-unsmoothed score path is
    exact, because γ is loss-free ([SA] §4.2, Eq. 6) and the fixture's γ(K)
    lies on the INT8 grid."""
    n_heads, n_kv = 2, 1
    bias = _grid_bias(d)
    q = _on_grid(d, seq, n_heads, seed=1100 + seq)
    k = _lossless_k(seq, d, n_kv, block, seed=2200 + seq, bias=bias)
    v = _fixtures(seq, d, n_heads, n_kv, seed=3300 + seq)[2]
    mask = _additive_mask(seq, seed=7 * seq) if masked else None

    out, stats = attention_sage(
        q, k, v, causal=causal, mask=mask, block_size=block, return_stats=True
    )
    assert out.shape == (2, n_heads, seq, d)

    scale = 1.0 / math.sqrt(d)
    k_rep = _repeat_kv(k, n_heads)
    seen_blocks = set()
    for st in stats.blocks:
        rows = np.arange(st.i0, st.i1)
        cols = np.arange(st.j0, st.j1)
        # (a) γ is loss-free on this fixture: the dequantized smoothed K is
        #     bit-identical to γ(K), so no quantization error is in play at all.
        assert np.array_equal(st.k_dequant, st.k_smoothed), (
            f"fixture is not lossless at block {st.j0}:{st.j1} — "
            f"max|ψ⁻¹(ψ(γ(K))) − γ(K)| = {np.max(np.abs(st.k_dequant - st.k_smoothed)):.3e}"
        )
        assert np.all(np.isfinite(st.k_smoothed))
        if st.j1 - st.j0 > 1:
            assert np.any(st.k_smoothed != 0.0), "the smoothed block is identically zero"

        # (b) the dequantized-then-unsmoothed score path equals the exact scores.
        unsmoothed = st.k_dequant + st.k_mean
        assert np.array_equal(unsmoothed, k[:, :, st.j0 : st.j1, :]), (
            "unsmoothing the dequantized K did not restore the exact keys"
        )
        exact = _scores_block(
            q[:, :, st.i0 : st.i1, :] * scale,
            k_rep[:, :, st.j0 : st.j1, :],
            rows,
            cols,
            causal=causal,
            mask_block=None if mask is None else mask[..., st.i0 : st.i1, st.j0 : st.j1],
        )
        _assert_roundoff(st.scores_unsmoothed, exact)
        seen_blocks.add((st.i0, st.i1, st.j0, st.j1))

    # Non-vacuity: every (query block, key block) pair the tiling implies ran,
    # and the per-block means are genuinely per-block (a whole-cache mean would
    # contradict [SA] §4.2's block-local form).
    expected = 0
    for i0 in range(0, seq, block):
        i1 = min(i0 + block, seq)
        expected += len(range(0, min(i1, seq) if causal else seq, block))
    assert len(seen_blocks) == expected
    means = [st.k_mean for st in stats.blocks if st.i0 == 0]
    if len(means) > 1:
        assert any(not np.allclose(means[0], m) for m in means[1:]), (
            "per-block means are identical across key blocks — the fixture would not notice "
            "a whole-cache mean"
        )

    # Non-vacuity of γ itself: quantizing the UNSMOOTHED K block destroys it.
    raw_hat, raw_delta = quantize_per_token(k[:, :, 0 : min(block, seq), :])
    raw_err = np.max(np.abs(dequantize_int8(raw_hat, raw_delta) - k[:, :, 0 : min(block, seq), :]))
    assert raw_err > 1.0, f"fixture is vacuous: the unsmoothed K quantizes cleanly ({raw_err:.3e})"


@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
@pytest.mark.parametrize("shape,block", BATTERY, ids=BATTERY_IDS)
def test_scn_gamma_shift_invariance_int8_only(shape, block, causal):
    """gamma-shift-invariance on general fixtures: the unsmoothed path IS the
    dequantized-operand path, so the deviation from the exact scores is INT8
    representation error alone (γ and the shift contribute exactly nothing)."""
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=1000 + seq * 7 + d)
    _, stats = attention_sage(q, k, v, causal=causal, block_size=block, return_stats=True)

    scale = 1.0 / math.sqrt(d)
    k_rep = _repeat_kv(k, n_heads)
    for st in stats.blocks:
        rows = np.arange(st.i0, st.i1)
        cols = np.arange(st.j0, st.j1)
        # (i) the reference's score path is the paper's algebra, rebuilt here
        #     from the exposed int8 tiles and scales: δ_Q δ_K (Q̂K̂ᵀ) + q̃·meanᵀ.
        rebuilt = _scores_block(
            stats.q_dequant[:, :, st.i0 : st.i1, :],
            st.k_unsmoothed,
            rows,
            cols,
            causal=causal,
            mask_block=None,
        )
        _assert_roundoff(st.scores_unsmoothed, rebuilt)

        # (ii) …therefore the deviation from the exact scores equals the
        #      deviation of the dequantized operands from the exact operands.
        #      The deviation is a difference of two O(1) score tensors, so the
        #      bound is absolute against the score magnitude (a relative measure
        #      on the cancelled difference is ill-conditioned by construction).
        exact = _scores_block(
            q[:, :, st.i0 : st.i1, :] * scale,
            k_rep[:, :, st.j0 : st.j1, :],
            rows,
            cols,
            causal=causal,
            mask_block=None,
        )
        operands = _scores_block(
            stats.q_dequant[:, :, st.i0 : st.i1, :],
            st.k_unsmoothed,
            rows,
            cols,
            causal=causal,
            mask_block=None,
        )
        # Causal blocking writes −inf, and inf − inf is NaN, so the comparison
        # lives on the entries all three paths define (their −inf patterns agree).
        finite = np.isfinite(exact)
        assert np.array_equal(np.isfinite(operands), finite)
        assert np.array_equal(np.isfinite(st.scores_unsmoothed), finite)
        unsmot, ref = st.scores_unsmoothed[finite], exact[finite]
        deq = operands[finite]
        magnitude = max(float(np.max(np.abs(ref))), float(np.max(np.abs(deq))))
        residual = np.max(np.abs((unsmot - ref) - (deq - ref)))
        assert residual <= RTOL * magnitude, (
            f"the unsmoothed path's deviation from the exact scores is not the dequantized "
            f"operands' deviation: residual {residual:.3e} > {RTOL * magnitude:.3e} at "
            f"block i{st.i0}:{st.i1} j{st.j0}:{st.j1}"
        )

        # Per-block means, checked against the block itself (a whole-cache mean
        # would pass (i) and (ii) but fail here).
        np.testing.assert_allclose(
            st.k_mean, k[:, :, st.j0 : st.j1, :].mean(axis=-2, keepdims=True)
        )


def test_scn_gamma_shift_invariance_blocks_use_their_own_tokens():
    """gamma-shift-invariance, block-scope half: with more than one K-block the
    means must differ from each other and from the whole-cache mean — that is
    the property that lets the kernel skip a whole-cache pass."""
    seq, d, n_heads, n_kv = (64, 8, 2, 1)
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=777)
    _, stats = attention_sage(q, k, v, block_size=8, return_stats=True)
    means = [st.k_mean for st in stats.blocks if st.i0 == 0]
    assert len(means) == seq // 8
    whole = k.mean(axis=-2, keepdims=True)
    assert any(not np.allclose(m, means[0]) for m in means[1:])
    assert any(not np.allclose(m, whole) for m in means)


# ---------------------------------------------------------------------------
# quantized-budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "addmask"])
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
@pytest.mark.parametrize("shape,block", BATTERY, ids=BATTERY_IDS)
def test_scn_quantized_budget(shape, block, causal, masked):
    """quantized-budget: Cos Sim ≥ 0.999 and Relative L1 ≤ 0.02 against the
    exact tiled reference ([SA] §4.3 metrics), on the fixture battery."""
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=2000 + seq * 7 + d)
    mask = _additive_mask(seq, seed=11 * seq) if masked else None

    reference = attention_tiled(q, k, v, causal=causal, mask=mask, block_size=block)
    out = attention_sage(q, k, v, causal=causal, mask=mask, block_size=block)

    assert out.shape == reference.shape
    assert np.isfinite(out).all(), "quantized reference produced a non-finite output"
    cos = _cos_sim(reference, out)
    rel_l1 = _relative_l1(reference, out)
    rmse = _rmse(reference, out)
    where = f"shape={shape} block={block} causal={causal} masked={masked}"
    assert cos >= COS_SIM_MIN, f"cos {cos:.6f} < {COS_SIM_MIN} at {where}"
    assert rel_l1 <= RELATIVE_L1_MAX, f"relative L1 {rel_l1:.6f} > {RELATIVE_L1_MAX} at {where}"
    assert np.isfinite(rmse)


def test_scn_quantized_budget_scales_are_strictly_positive():
    """quantized-budget, scale clause: per-token Q/K scales are strictly
    positive and no token has zero range ([SA] §3.2's dynamic scale is undefined
    at zero range, so the reference refuses instead of dividing by zero)."""
    for shape, block in BATTERY:
        seq, d, n_heads, n_kv = shape
        q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=3000 + seq * 7 + d)
        _, stats = attention_sage(q, k, v, causal=True, block_size=block, return_stats=True)
        assert np.all(stats.q_scale > 0.0), "non-positive per-token Q scale"
        assert np.all(np.isfinite(stats.q_scale))
        assert np.all(np.isfinite(stats.q_hat.astype(np.float64) * stats.q_scale))
        for st in stats.blocks:
            assert np.all(st.k_scale > 0.0), f"non-positive per-token K scale in block {st.j0}"
            assert np.all(np.isfinite(st.k_scale))
            assert np.all(st.v_scale > 0.0), f"non-positive per-channel V scale in block {st.j0}"
            assert np.all(np.isfinite(st.v_scale))
            # Zero-range rows are the load-bearing case (a one-token K-block, or
            # a token equal to its block's mean): δ stays positive and defined
            # (1.0) with all-zero codes, which is exact once the shift is added.
            peak = np.max(np.abs(st.k_smoothed), axis=-1)
            zero = peak == 0.0
            if zero.any():
                assert np.all(st.k_scale[..., 0][zero] == 1.0)
                assert np.all(st.k_hat[zero] == 0), "a zero-range row emitted non-zero codes"
    assert PTILDE_STATIC_SCALE == 1.0 / 127.0


def test_scn_quantized_budget_granularities_are_the_paper_axes():
    """quantized-budget, probe half: the exposed quantizer outputs carry [SA]
    §4.3's granularity per tensor — Q/K per-token (outer axis of QKᵀ), P̃
    per-block at the static 1/127 scale, V per-channel (outer axis of P̃Vᵀ)."""
    seq, d, n_heads, n_kv = (8, 4, 4, 2)
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=9001)
    _, stats = attention_sage(q, k, v, block_size=3, return_stats=True)
    assert stats.q_scale.shape == (2, n_heads, seq, 1)
    assert stats.q_hat.dtype == np.int8
    for st in stats.blocks:
        kj = st.j1 - st.j0
        assert st.k_scale.shape == (2, n_kv, kj, 1)
        assert st.k_hat.dtype == np.int8
        assert st.k_smoothed.shape == (2, n_kv, kj, d)
        assert st.k_mean.shape == (2, n_kv, 1, d)
        assert st.v_scale.shape == (2, n_kv, 1, d)
        assert st.v_hat.dtype == np.int8
        assert st.p_hat.dtype == np.int8
        assert st.p_scale == PTILDE_STATIC_SCALE
        # P̃'s row max is 1 by construction, so the static scale covers it.
        assert np.min(st.p_hat) >= 0 and np.max(st.p_hat) <= INT8_MAX


def test_attention_sage_matches_the_oracle_on_the_lossless_fixture():
    """The lossless γ fixture isolates the rest of the path: with Q and γ(K)
    exact, the only remaining error is P̃/V quantization, so the output must sit
    inside the paper-class budget with room to spare."""
    seq, d, n_heads, n_kv, block = 64, 4, 2, 1, 8
    bias = _grid_bias(d)
    q = _on_grid(d, seq, n_heads, seed=4444)
    k = _lossless_k(seq, d, n_kv, block, seed=5555, bias=bias)
    v = _fixtures(seq, d, n_heads, n_kv, seed=6666)[2]
    reference = attention_tiled(q, k, v, causal=True, block_size=block)
    out = attention_sage(q, k, v, causal=True, block_size=block)
    assert _cos_sim(reference, out) >= COS_SIM_MIN
    assert _relative_l1(reference, out) <= RELATIVE_L1_MAX


# ---------------------------------------------------------------------------
# Captured-activation fixture hook
# ---------------------------------------------------------------------------

CAPTURED_ACTIVATIONS_ENV = "QUANTIBBLER_SAGE_ACTIVATIONS"
CAPTURED_ACTIVATIONS_DEFAULT = Path(__file__).parent / "fixtures" / "attention_sage_activations.npz"


@dataclass(eq=False)
class CapturedCase:
    """One captured (q, k, v) triple, as the budget scenario's extra input."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    causal: bool = False
    block_size: int = 64


def load_captured_activations(path: Path | str | None = None) -> tuple[list[CapturedCase], str]:
    """Resolve the captured-activation fixture.

    Returns ``(cases, source)``; ``source`` always names where the hook looked,
    so an absence is a recorded gap rather than a silently short battery. The
    file is a ``.npz`` of ``q``/``k``/``v`` (+ optional scalar ``causal`` and
    ``block_size``); ``QUANTIBBLER_SAGE_ACTIVATIONS`` overrides the default path.
    Deliberately NOT a ``pytest.skip``: the CI tiers pin their skip counts, and
    a quietly shortened battery is exactly the failure mode the scenario's
    "plus captured real activations" clause exists to prevent.
    """
    resolved = Path(
        path or os.environ.get(CAPTURED_ACTIVATIONS_ENV) or CAPTURED_ACTIVATIONS_DEFAULT
    )
    if not resolved.is_file():
        return [], (
            f"absent: no captured activations at {resolved}; the battery below is the "
            "fixture battery only (task 4.2 produces the captured records)"
        )
    with np.load(resolved) as data:
        case = CapturedCase(
            q=np.asarray(data["q"], dtype=np.float64),
            k=np.asarray(data["k"], dtype=np.float64),
            v=np.asarray(data["v"], dtype=np.float64),
            causal=bool(np.asarray(data["causal"]).item()) if "causal" in data else False,
            block_size=int(np.asarray(data["block_size"]).item()) if "block_size" in data else 64,
        )
    return [case], f"loaded from {resolved}"


def test_scn_quantized_budget_captured_activations_hook(tmp_path: Path):
    """quantized-budget's "plus captured real activations" half.

    No captured r137 activations are shipped in the repo yet (task 4.2 produces
    them, into the model directory) — so the hook must be *loud* about the
    absence instead of silently passing, and its present-branch must actually
    work. The present branch is exercised here on a synthesized capture, which
    is a hook test, not a quality claim: the quality claim is the battery above.
    """
    cases, source = load_captured_activations()
    assert source, "the hook must name where it looked"
    if not cases:
        assert "absent" in source, f"a missing capture must be named as absent, got {source!r}"

    path = tmp_path / "captured.npz"
    seq, d, n_heads, n_kv = 8, 4, 2, 1
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=8080)
    np.savez(path, q=q, k=k, v=v)
    cases, source = load_captured_activations(path)
    assert cases and str(path) in source
    for case in cases:
        reference = attention_tiled(
            case.q, case.k, case.v, causal=case.causal, block_size=case.block_size
        )
        out = attention_sage(case.q, case.k, case.v, causal=case.causal, block_size=case.block_size)
        assert _cos_sim(reference, out) >= COS_SIM_MIN
        assert _relative_l1(reference, out) <= RELATIVE_L1_MAX


# ---------------------------------------------------------------------------
# Contract edges
# ---------------------------------------------------------------------------


def test_attention_sage_contract_edges():
    """Contract edges are defined rather than silently degraded.

    The zero-range convention is the load-bearing one: [SA] §3.2's dynamic scale
    ``max|A|/127`` is 0/0 for a row whose peak is zero, and that row is *not* an
    exotic case — a one-token K-block has ``mean(K_block) = K_block``, hence
    ``γ(K) ≡ 0``, at every ragged tail of a sequence whose length misses the
    block boundary. The reference keeps δ = 1.0 with all-zero codes there (the
    value cannot move any output, since zero codes dequantize to zero for any δ)
    and the shift restores the keys exactly, so ``block_size=1`` is a working
    configuration whose score path is exact — not a refusal.
    """
    q, k, v = _fixtures(8, 4, 2, 1, seed=6001)
    with pytest.raises(ValueError, match="block_size"):
        attention_sage(q, k, v, block_size=0)
    with pytest.raises(ValueError):
        attention_sage(q, k, v[:, :, :4, :])
    with pytest.raises(ValueError):
        attention_sage(q, k, v, mask=np.ones((8, 8))[:, :4])
    with pytest.raises(ValueError):
        attention_sage(q, np.repeat(k, 3, axis=1), v)

    # Zero range keeps a defined, positive scale and zero codes.
    zh, zd = quantize_per_token(np.zeros((1, 1, 2, 4)))
    assert np.all(zh == 0) and np.all(zd == 1.0)
    vh, vd = quantize_per_channel(np.zeros((1, 1, 4, 2)))
    assert np.all(vh == 0) and np.all(vd == 1.0)

    # A one-token block is exactly that case for K, and stays exact. Non-causal
    # with block 1 means every (row, key) pair is its own tile: seq² of them.
    seq = q.shape[-2]
    out1, st1 = attention_sage(q, k, v, block_size=1, return_stats=True)
    assert len(st1.blocks) == seq * seq
    for st in st1.blocks:
        assert np.array_equal(st.k_smoothed, np.zeros_like(st.k_smoothed))
        assert np.array_equal(st.k_hat, np.zeros_like(st.k_hat))
        assert np.all(st.k_scale == 1.0)
        assert np.array_equal(st.k_unsmoothed, k[:, :, st.j0 : st.j1, :])

    # The tile-local P̃ and V quantization are exact at block 1 too (a one-token
    # P̃ row is 0 or 1; a one-token V tile puts every channel at full scale), so
    # the score path above is the whole story and the output error is Q's alone —
    # inside the same budget the battery asserts, not at round-off.
    reference = attention_tiled(q, k, v, block_size=1)
    assert _cos_sim(reference, out1) >= COS_SIM_MIN
    assert _relative_l1(reference, out1) <= RELATIVE_L1_MAX


def test_smooth_k_block_and_quantizers_are_the_paper_stages():
    """The exposed stages are individually assertable against [SA] §4.2/§3.2."""
    rng = np.random.default_rng(42)
    kb = rng.standard_normal((1, 2, 5, 8))
    smoothed, mean = smooth_k_block(kb)
    np.testing.assert_allclose(mean, kb.mean(axis=-2, keepdims=True))
    np.testing.assert_allclose(smoothed, kb - mean)
    # Shift invariance of the smoothed scores with the mean added back.
    q = rng.standard_normal((1, 2, 3, 8))
    np.testing.assert_allclose((q @ (smoothed + mean).swapaxes(-1, -2)), q @ kb.swapaxes(-1, -2))

    hat, delta = quantize_per_token(kb)
    assert hat.dtype == np.int8
    np.testing.assert_allclose(delta, np.max(np.abs(kb), axis=-1, keepdims=True) / INT8_MAX)
    assert np.max(np.abs(hat)) <= INT8_MAX
    assert np.max(np.abs(dequantize_int8(hat, delta) - kb)) <= np.max(delta) / 2.0

    vhat, vdelta = quantize_per_channel(kb)
    np.testing.assert_allclose(vdelta, np.max(np.abs(kb), axis=-2, keepdims=True) / INT8_MAX)
    assert np.max(np.abs(vhat)) <= INT8_MAX

    p = np.linspace(0.0, 1.0, 32).reshape(1, 1, 8, 4)
    phat, pdelta = quantize_p_tilde(p)
    assert pdelta == PTILDE_STATIC_SCALE
    assert np.max(phat) == INT8_MAX
    assert np.max(np.abs(dequantize_int8(phat, pdelta) - p)) <= PTILDE_STATIC_SCALE / 2.0
