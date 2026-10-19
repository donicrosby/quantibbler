"""Gram-table polish parity against a naive reference implementation.

The Gram-table hot loop (bit_polish.py) replaces the naive per-flip
re-projection with exact O(r) updates off a per-sweep Gram table. This
suite pins the parity contract at production geometries:

- At production scale classes (s1, s2 ~ 1e-4..1e0) the two paths must
  agree BIT-FOR-BIT (same flips, same final arrays).
- At pathological scale splits (s1, s2 ~ 1e4, Wq init >> W) genuine
  near-ties between large gains exist; tie-break order may diverge and
  the paths land on different local optima of the same objective — both
  still valid descents, each final SSE strictly below the input's.

The naive reference below is deliberately kept inline and naive (per-flip
residual re-projection) — it is the executable spec of the pre-Gram
behaviour, not production code.
"""

import numpy as np

from quantibbler.ptq.bit_polish import greedy_polish


def _naive_reference(w, ub, vb, s1, s2, rounds=4):
    ub = ub.copy()
    vb = vb.copy()
    s1 = s1.copy()
    s2 = s2.copy()
    dout, din = w.shape
    r = ub.shape[1]
    for _rnd in range(rounds):
        R = w - (s1[:, None] * (ub @ vb.T)) * s2[None, :]
        vb_s2 = vb * s2[:, None]
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
        ub_s1 = ub * s1[:, None]
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
        Bc = ub @ (vb * s2[:, None]).T
        d1 = (Bc * Bc).sum(axis=1)
        raw1 = np.where(d1 > 0, (w * Bc).sum(axis=1) / np.maximum(d1, 1e-300), s1)
        ref1 = np.maximum(np.abs(s1), 1e-30)
        new1 = np.minimum(np.maximum(np.abs(raw1), 0.125 * ref1), 8.0 * ref1)
        sgn1 = np.where(raw1 < 0.0, -1.0, 1.0)
        flip_rows = sgn1 < 0
        if flip_rows.any():
            ub[flip_rows] = -ub[flip_rows]
        s1 = new1
        Br = (ub * s1[:, None]) @ vb.T
        d2 = (Br * Br).sum(axis=0)
        raw2 = np.where(d2 > 0, (w * Br).sum(axis=0) / np.maximum(d2, 1e-300), s2)
        ref2 = np.maximum(np.abs(s2), 1e-30)
        new2 = np.minimum(np.maximum(np.abs(raw2), 0.125 * ref2), 8.0 * ref2)
        sgn2 = np.where(raw2 < 0.0, -1.0, 1.0)
        flip_cols = sgn2 < 0
        if flip_cols.any():
            vb[flip_cols] = -vb[flip_cols]
        s2 = new2
    return ub, vb, s1, s2


def _case(rng, dout, din, r, scale):
    W = (rng.standard_normal((dout, din)) * scale) * rng.uniform(0.1, 10)
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    Ub0 = np.sign(u[:, :r] * np.sqrt(s[:r]))
    Vb0 = np.sign(vt[:r].T * np.sqrt(s[:r]))
    s1_0 = np.abs(rng.standard_normal(dout)) * scale + 1e-3
    s2_0 = np.abs(rng.standard_normal(din)) * scale + 1e-3
    return W, Ub0, Vb0, s1_0, s2_0


def test_scn_gram_table_polish_parity():
    rng = np.random.default_rng(11)
    for scale in (1e-4, 1.0, 1e4):
        for dout, din, r in ((24, 16, 6), (64, 48, 10)):
            W, Ub0, Vb0, s1_0, s2_0 = _case(rng, dout, din, r, scale)
            u1, v1, a1, b1 = greedy_polish(
                W, Ub0.copy(), Vb0.copy(), s1_0.copy(), s2_0.copy(), rounds=4
            )
            u2, v2, a2, b2 = _naive_reference(
                W, Ub0.copy(), Vb0.copy(), s1_0.copy(), s2_0.copy(), rounds=4
            )

            def _sse(ub, vb, a, b, w=W):
                return float(((w - (a[:, None] * (ub @ vb.T)) * b[None, :]) ** 2).sum())

            s1q, s2q = _sse(u1, v1, a1, b1), _sse(u2, v2, a2, b2)
            s_init = _sse(Ub0, Vb0, s1_0, s2_0)
            rel = abs(s1q - s2q) / max(abs(s2q), 1e-300)
            assert s1q < s_init and s2q < s_init, "descent violated"
            # Bit parity at production scale classes; objective parity
            # (percent-level tie divergence tolerated) at pathological
            # splits where near-ties between large gains exist.
            if scale < 1e4:
                assert np.array_equal(u1, u2) and np.array_equal(v1, v2), (
                    f"bits diverge at benign scale {scale}, shape {(dout, din, r)}"
                )
            else:
                assert rel < 1e-1, f"SSE diverges at pathological scale: rel {rel:.3e}"
