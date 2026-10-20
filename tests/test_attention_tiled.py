"""Tiled exact-attention NumPy reference (change ``qk-flash-attention``).

Scenario IDs from ``openspec/changes/qk-flash-attention/specs/qk-kernels/spec.md``:
``ref-matches-formula`` and ``ref-causal-mask``. The oracle is the *dense*
float64 formula ``softmax(QKᵀ/√d + mask)·V``; the module under test is [FA]
§3.1 Algorithm 1 — blockwise online softmax with running statistics (m, ℓ)
and the ``exp(m_old − m_new)`` rescaling identity — so agreement between the
two is the evidence that the tiling, the rescaling and the masking are all
correct. Every fixture is run at block sizes that do (and do not) divide the
sequence length, because a ragged tail is where tiled attention usually
breaks.

Tolerances: the house parity gate is float64 at relative 1e-12. For
attention that literal *per-element* form is unattainable — softmax outputs
cancel, so an entry can be 1e-4 while its row norm is O(1), and the absolute
round-off error (one ulp ≈ 1e-15) then reads as ~3e-12 relative. The gate
below is therefore two-part and stated in `_assert_roundoff`: an
infinity-norm relative bound at 1e-12, plus the per-element bound at 1e-12
restricted to entries within 1e-2 of the row-max magnitude, where the
relative measure is well-conditioned. Measured on this battery the two parts
come in at 3.3e-16 and 6.2e-15 — three orders inside the gate — while any
masking, GQA or rescaling bug lands at O(1) and is caught.

Lean tier: NumPy only — no torch — so this runs in every environment,
including the extras-free one.

GQA convention (pinned engine convention, qk-server spec): ``n_heads`` query
heads share ``n_kv_heads`` key/value heads in consecutive groups of
``n_heads // n_kv_heads`` (``repeat_interleave``), head ``h`` reading kv head
``h // group``.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantibbler.kernels.attention_tiled import attention_tiled

RTOL = 1e-12

# (seq, d, n_heads, n_kv_heads) — the spec's fixture battery.
SHAPES = [
    (8, 4, 2, 1),
    (8, 128, 8, 2),
    (10, 4, 8, 2),
    (10, 128, 2, 1),
    (64, 4, 2, 1),
    (64, 128, 8, 2),
]

# 1 and 3 divide none of {8, 10, 64}: ragged tails on every shape; 64/128
# cover the single-block and larger-than-sequence cases.
BLOCK_SIZES = [1, 3, 4, 8, 64, 128]


def _fixtures(seq: int, d: int, n_heads: int, n_kv: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    b = 2
    q = rng.standard_normal((b, n_heads, seq, d))
    k = rng.standard_normal((b, n_kv, seq, d))
    v = rng.standard_normal((b, n_kv, seq, d))
    return q, k, v


def _repeat_kv(x: np.ndarray, n_heads: int) -> np.ndarray:
    """GQA broadcast: repeat each kv head consecutively (``repeat_interleave``)."""
    group = n_heads // x.shape[1]
    return np.repeat(x, group, axis=1)


def _dense(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool, mask=None) -> np.ndarray:
    """The direct float64 formula the tiled reference must reproduce."""
    k = _repeat_kv(k, q.shape[1])
    v = _repeat_kv(v, q.shape[1])
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    seq = q.shape[-2]
    if causal:
        blocked = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(blocked, -np.inf, scores)
    if mask is not None:
        scores = scores + mask
    peak = np.max(scores, axis=-1, keepdims=True)
    # A fully-masked row has peak = -inf, so this exponentiation is the dense
    # 0/0 the reference scenario compares against — deliberate, not a defect.
    with np.errstate(invalid="ignore"):
        weights = np.exp(scores - peak)
    return (weights @ v) / np.sum(weights, axis=-1, keepdims=True)


def _assert_roundoff(a: np.ndarray, b: np.ndarray, rtol: float = RTOL) -> None:
    """Round-off agreement, measured the only way an attention output allows.

    ``(1)`` infinity-norm relative: ``max|a-b| <= rtol * max(|a|,|b|)``.
    ``(2)`` per-element relative ``<= rtol`` for entries within ``1e-2`` of
    the row-max magnitude — the band where the relative measure is not
    dominated by cancellation. Part (1) bounds the absolute error against the
    tensor scale; part (2) is the house-shaped per-element check on the
    entries that can carry it.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), "non-finite attention output"
    norm = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    assert np.max(np.abs(a - b)) <= rtol * norm, (
        f"tiled reference differs beyond inf-norm rel {rtol}: "
        f"{np.max(np.abs(a - b)) / norm:.3e} (norm {norm:.3e})"
    )
    row_max = np.max(np.abs(b), axis=-1, keepdims=True)
    strong = np.abs(b) >= 1e-2 * row_max
    if strong.any():
        rel = np.abs(a - b)[strong] / np.abs(b)[strong]
        assert np.all(rel <= rtol), (
            f"tiled reference differs beyond per-element rel {rtol} on "
            f"well-conditioned entries: max {np.max(rel):.3e}"
        )


def _additive_mask(seq: int, seed: int) -> np.ndarray:
    """Broadcastable additive bias: (1, 1, seq, seq), both signs, no −inf."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 1, seq, seq)) * 0.5


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_ref_matches_formula(shape, causal, block_size):
    """ref-matches-formula: tiled output ≡ dense float64 formula at rel 1e-12."""
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=1000 + seq * 7 + d)
    out = attention_tiled(q, k, v, causal=causal, block_size=block_size)
    _assert_roundoff(out, _dense(q, k, v, causal=causal))


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ref_matches_formula_additive_mask(shape, block_size):
    """ref-matches-formula, optional mask: additive bias enters pre-softmax."""
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=2000 + seq * 7 + d)
    mask = _additive_mask(seq, seed=7 * seq)
    out = attention_tiled(q, k, v, mask=mask, block_size=block_size)
    _assert_roundoff(out, _dense(q, k, v, causal=False, mask=mask))


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ref_causal_mask(shape, block_size):
    """ref-causal-mask: the tiled result equals the −inf-m asked dense formula."""
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=3000 + seq * 7 + d)
    out = attention_tiled(q, k, v, causal=True, block_size=block_size)
    blocked = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    dense = _dense(q, k, v, causal=False, mask=np.where(blocked, -np.inf, 0.0))
    _assert_roundoff(out, dense)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ref_causal_future_tokens_contribute_exactly_zero(shape, block_size):
    """ref-causal-mask, exact half: a masked key cannot move an earlier row.

    Move one key's value by a large amount and require every row *before* it
    to come back bit-identical: approximation would leak the change through a
    rescaling error, so exactness here is a stronger statement than the
    tolerance comparison and needs no tolerance to interpret.
    """
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=4000 + seq * 7 + d)
    base = attention_tiled(q, k, v, causal=True, block_size=block_size)
    for position in sorted({1, seq // 2, seq - 1}):
        moved_v = v.copy()
        moved_v[:, :, position, :] += 1.0e3
        moved = attention_tiled(q, k, moved_v, causal=True, block_size=block_size)
        assert np.array_equal(base[:, :, :position, :], moved[:, :, :position, :]), (
            f"causal masking leaked key {position} into rows < {position} "
            f"(seq={seq}, d={d}, block={block_size})"
        )
        assert not np.array_equal(base[:, :, position:, :], moved[:, :, position:, :]), (
            f"the probe key {position} changed nothing at all — the test would pass vacuously"
        )


@pytest.mark.parametrize("block_size", [1, 3, 8, 64])
@pytest.mark.parametrize("shape", SHAPES)
def test_ref_matches_formula_wide_dynamic_range(shape, block_size):
    """ref-matches-formula on numerically hard inputs: block maxima ~1e3 apart.

    This is the reason [FA] Algorithm 1 carries a *running* max rather than
    each block's own: with a non-monotone ``m`` the rescale exponentiates a
    positive difference and overflows float64. Verified to be a real
    discriminator rather than a decoration — replacing the running-max line
    with the block's own max produces 824 non-finite outputs on exactly this
    fixture, while the implementation here matches the dense oracle with
    inf-norm relative error 0.0.
    """
    seq, d, n_heads, n_kv = shape
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=7000 + seq * 7 + d)
    q = q * 2000.0
    k = k * 2000.0
    out = attention_tiled(q, k, v, causal=True, block_size=block_size)
    _assert_roundoff(out, _dense(q, k, v, causal=True))


def test_ref_returns_lse_consistent_with_dense():
    """``return_lse`` is the log-sum-exp of the masked scores, from the running (m, ℓ)."""
    seq, d, n_heads, n_kv = (10, 128, 8, 2)
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=5001)
    out, lse = attention_tiled(q, k, v, causal=True, block_size=3, return_lse=True)
    assert lse.shape == (2, n_heads, seq)
    scores = q @ _repeat_kv(k, n_heads).swapaxes(-1, -2) / np.sqrt(d)
    blocked = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    masked = np.where(blocked, -np.inf, scores)
    peak = np.max(masked, axis=-1, keepdims=True)
    ref_lse = np.squeeze(peak, axis=-1) + np.log(np.sum(np.exp(masked - peak), axis=-1))
    _assert_roundoff(lse, ref_lse)
    _assert_roundoff(out, _dense(q, k, v, causal=True))


def test_ref_val_dim_may_differ_from_head_dim():
    """``val_dim`` is free; the softmax scale stays ``1/√head_dim``.

    A value projection need not be square, and the substrate exposes the two
    dimensions separately — so the convention has to be pinned, not assumed:
    scaling by ``1/√val_dim`` would pass every square fixture.
    """
    seq, d, val_dim, n_heads, n_kv = 8, 4, 8, 2, 1
    q, k, _ = _fixtures(seq, d, n_heads, n_kv, seed=6002)
    rng = np.random.default_rng(6003)
    v = rng.standard_normal((2, n_kv, seq, val_dim))
    for causal in (False, True):
        out = attention_tiled(q, k, v, causal=causal, block_size=3)
        assert out.shape == (2, n_heads, seq, val_dim)
        _assert_roundoff(out, _dense(q, k, v, causal=causal))


def test_ref_fully_masked_row_yields_the_dense_nan_pattern():
    """Scenario: Fully-masked row yields the dense NaN pattern.

    Deliberately NOT routed through `_assert_roundoff`: that gate asserts
    all-finite by construction, so it can never carry this contract. The
    statistic is pinned too (``m = -inf, ℓ = 0``, hence ``lse = -inf``),
    because propagating NaN through ``exp(-inf - (-inf))`` would also put a
    NaN in the running statistic — and a fixture that only inspected the
    output could not tell the two apart.
    """
    seq, d, n_heads, n_kv = 8, 4, 2, 1
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=7001)
    bias = np.zeros((1, 1, seq, seq))
    bias[..., 3, :] = -np.inf  # row 3 has no admissible score at all

    out, lse = attention_tiled(q, k, v, mask=bias, block_size=3, return_lse=True)
    dense = _dense(q, k, v, causal=False, mask=bias)

    # Non-vacuous: the dense 0/0 must actually produce NaNs, in exactly one row.
    assert np.isnan(dense[..., 3, :]).all(), "fixture is vacuous: dense has no NaNs"
    assert not np.isnan(np.delete(dense, 3, axis=-2)).any(), "fixture masks more than row 3"

    assert np.array_equal(np.isnan(out), np.isnan(dense)), (
        "NaN pattern differs from the dense 0/0 formula: "
        f"device {int(np.isnan(out).sum())} NaN entries, dense {int(np.isnan(dense).sum())}"
    )
    # No invented fallback: the fully masked row is NaN, all other rows finite.
    assert np.isnan(out[..., 3, :]).all(), "fully masked row was given a fallback value"
    assert np.isfinite(np.delete(out, 3, axis=-2)).all(), "a partially masked row went non-finite"
    assert np.isneginf(lse[..., 3]).all(), "fully masked row's log-sum-exp is not -inf"
    assert np.isfinite(lse[..., :3]).all() and np.isfinite(lse[..., 4:]).all()


def test_ref_combined_causal_and_additive_mask():
    """Scenario: Combined causal and additive mask — both constraints apply.

    The additive bias sits on *allowed* (lower-triangle) positions, so a code
    path that quietly disables the mask when causal masking is on cannot pass:
    the oracle carries both, and the fixture additionally proves it is not
    vacuous by checking that dropping the bias changes the result.
    """
    seq, d, n_heads, n_kv = 10, 4, 8, 2
    q, k, v = _fixtures(seq, d, n_heads, n_kv, seed=7002)
    rng = np.random.default_rng(7002)
    lower = np.tril(np.ones((seq, seq), dtype=bool))
    # A strong negative bias on every allowed position: the distribution over
    # the surviving keys is nearly flat, which is not what causal-only gives.
    bias = np.where(lower, -50.0 * rng.random((seq, seq)), 0.0)[None, None]

    out = attention_tiled(q, k, v, causal=True, mask=bias, block_size=3)
    dense = _dense(q, k, v, causal=True, mask=bias)
    _assert_roundoff(out, dense)

    without_bias = attention_tiled(q, k, v, causal=True, block_size=3)
    assert not np.allclose(out, without_bias), (
        "fixture is vacuous: the additive mask changes nothing, so a causal path "
        "that ignored it would pass"
    )
    assert np.isfinite(out).all(), "combined masking produced a non-finite output"


def test_ref_rejects_bad_shapes_and_block_size():
    """Contract edges fail loudly rather than silently broadcasting."""
    q, k, v = _fixtures(8, 4, 2, 1, seed=6001)
    with pytest.raises(ValueError):
        attention_tiled(q, k, v, block_size=0)
    with pytest.raises(ValueError):
        attention_tiled(q, k, v[:, :, :4, :])  # seq mismatch k/v
    with pytest.raises(ValueError):
        attention_tiled(q, k, v, mask=np.ones((8, 8))[:, :4])
    with pytest.raises(ValueError):
        attention_tiled(q, np.repeat(k, 3, axis=1), v)  # n_kv > n_heads
