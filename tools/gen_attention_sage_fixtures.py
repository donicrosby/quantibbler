#!/usr/bin/env python
"""Generate the INT8 attention device-parity fixtures (change qk-sage-attention).

Emits ``crates/quantibbler-kernels/tests/fixtures/attention_sage.json``: for
each battery case, the f32 inputs the device face consumes plus THREE records
the Rust-side test (``tests/gpu_attention_sage.rs``) asserts against —

- ``faces.quantized-reference``: the output of
  ``quantibbler.kernels.attention_sage`` (the verification fixture at the
  declared dtype for the [SA] §4 scheme) — the device's parity target at the spec's
  ``rel <= 1e-4`` block-level f32 budget;
- ``faces.exact``: the output of ``quantibbler.kernels.attention_tiled``
  (the exact f64 tiled face) — the quality reference the quantized path is
  measured *against*;
- ``reference_metrics``: [SA] §4.3's Cosine Sim / Relative L1 / RMSE of
  ``quantized-reference`` vs ``exact`` on this very case, so the device's
  metrics can be checked to land inside the reference's own measured envelope
  rather than against a hand-written number.

Battery: the ``gen_attention_fixtures.py`` families (same seeds, so the two
fixture files stay comparable) plus a per-case ``block_size`` chosen to force
the multi-block online-softmax path (including ragged tails and the
``block_size=1`` zero-range-γ convention the reference's crystallization entry
names). Regenerate with ``uv run python tools/gen_attention_sage_fixtures.py``;
fixture drift is a reviewable diff, never a manual copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantibbler.kernels.attention_sage import attention_sage
from quantibbler.kernels.attention_tiled import attention_tiled

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "crates" / "quantibbler-kernels" / "tests" / "fixtures"
)

# (batch, n_heads, n_kv_heads, seq, head_dim, val_dim, causal, block_size)
BATTERY = [
    # one block, both masking states
    (1, 2, 1, 8, 4, 4, False, 8),
    # two query blocks x two key blocks, causal
    (1, 2, 1, 8, 4, 4, True, 4),
    # ragged: 10 tokens at block 3 -> a one-token tail (the γ == 0 convention)
    (1, 8, 2, 10, 4, 4, False, 3),
    (1, 8, 2, 10, 4, 4, True, 4),
    # 8 key blocks: the online-softmax rescale chain, exercised long
    (1, 2, 1, 64, 4, 4, True, 8),
    (2, 8, 2, 10, 4, 4, True, 3),
    # d == val == 128: the wide tile, GQA 2:1
    (1, 2, 1, 8, 128, 128, True, 3),
    # val_dim != head_dim: separates the two axes
    (1, 2, 1, 8, 4, 8, False, 4),
    (1, 2, 1, 8, 128, 32, True, 2),
    # block_size = 1: every key block is a single token, gamma(K) == 0, the
    # zero-range delta == 1 convention is the only thing keeping it finite.
    (1, 2, 1, 8, 4, 4, True, 1),
]


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """[SA] §4.3 accuracy metrics, verbatim (same formulas as the pytest tier)."""
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel_l1 = float(np.sum(np.abs(a - b)) / np.sum(np.abs(a)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {"cosine": cos, "relative_l1": rel_l1, "rmse": rmse}


def _case(batch, n_heads, n_kv, seq, head_dim, val_dim, causal, block, seed):
    """One battery case; inputs drawn f32-exactly so the fixture round-trips."""
    rng = np.random.default_rng(seed)
    # Same convention as gen_attention_fixtures.py: the device consumes f32, so
    # the f64 references must be computed on the *same* numbers.
    q = rng.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float32).astype(np.float64)
    k = rng.standard_normal((batch, n_kv, seq, head_dim)).astype(np.float32).astype(np.float64)
    v = rng.standard_normal((batch, n_kv, seq, val_dim)).astype(np.float32).astype(np.float64)

    exact = attention_tiled(q, k, v, causal=causal)
    quantized = attention_sage(q, k, v, causal=causal, block_size=block)
    return {
        "scenario": "device-parity-quantized",
        "shape": {
            "batch": batch,
            "n_heads": n_heads,
            "n_kv_heads": n_kv,
            "seq": seq,
            "head_dim": head_dim,
            "val_dim": val_dim,
        },
        "causal": causal,
        "block_size": block,
        "seed": seed,
        "inputs": {
            "query": q.reshape(-1).astype(np.float32).tolist(),
            "key": k.reshape(-1).astype(np.float32).tolist(),
            "value": v.reshape(-1).astype(np.float32).tolist(),
        },
        "faces": {
            "quantized-reference": {"out": quantized.reshape(-1).tolist()},
            "exact": {"out": exact.reshape(-1).tolist()},
        },
        "reference_metrics": _metrics(exact, quantized),
        "dtypes": {
            "query": "f32",
            "key": "f32",
            "value": "f32",
            "out": "f32",
            "reference": "f64",
        },
    }


def main() -> None:
    records = [_case(*case, seed=9100 + i) for i, case in enumerate(BATTERY)]
    path = FIXTURE_DIR / "attention_sage.json"
    # Compact separators: the payload is numeric arrays of full-precision floats,
    # so indentation is ~20% of the file and pushes it past the repo's 500 KB
    # large-file gate. The bytes on the wire are the same JSON values either way
    # (both readers parse it); only the whitespace differs. The sibling
    # ``gen_attention_fixtures.py`` keeps ``indent=1`` because its file fits.
    path.write_text(json.dumps(records, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(records)} cases)")
    for rec in records:
        shape = rec["shape"]
        m = rec["reference_metrics"]
        print(
            f"  batch={shape['batch']} heads={shape['n_heads']}/{shape['n_kv_heads']} "
            f"seq={shape['seq']} d={shape['head_dim']} val={shape['val_dim']} "
            f"causal={rec['causal']} block={rec['block_size']} "
            f"cos={m['cosine']:.6f} relL1={m['relative_l1']:.6f} rmse={m['rmse']:.3e}"
        )


if __name__ == "__main__":
    main()
