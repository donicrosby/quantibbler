#!/usr/bin/env python
"""Generate the GPU attention parity fixtures (change qk-flash-attention).

Emits ``crates/quantibbler-kernels/tests/fixtures/attention_tiled.json``: for
each battery case, the f32 inputs the device face consumes plus the
``numpy-reference`` output (the executable specification) widened to f64. The
Rust-side test (``tests/gpu_attention_parity.rs``) widens the device's f32
result back and compares at the spec's budget (rel 1e-5) — a device face is
never compared to itself.

Battery coverage versus the spec's fixture battery (seq 8/10/64, d 4/128,
heads (2,1)/(8,2)): every family is represented, with the d=128 case at
seq 8 and every ragged tail (seq 10 and 64 against block sizes 3 and 8 are
not block multiples) — the d=128 x seq=64 x 8-head corner is covered in the
Python reference battery (``tests/test_attention_tiled.py``), where a fixture
file is not paying rent, and is deliberately absent here rather than silently
shrunk.

Regenerate with ``uv run python tools/gen_attention_fixtures.py``; fixture
drift is a reviewable diff, never a manual copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantibbler.kernels.attention_tiled import attention_tiled

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "crates" / "quantibbler-kernels" / "tests" / "fixtures"
)

# (batch, n_heads, n_kv_heads, seq, head_dim, val_dim, causal)
BATTERY = [
    (1, 2, 1, 8, 4, 4, False),
    (1, 2, 1, 8, 4, 4, True),
    (1, 8, 2, 10, 4, 4, False),
    (1, 8, 2, 10, 4, 4, True),
    (1, 2, 1, 64, 4, 4, True),
    (2, 8, 2, 10, 4, 4, True),
    (1, 2, 1, 8, 128, 128, True),
    # head_dim != val_dim: separates 1/sqrt(head_dim) from 1/sqrt(val_dim) in
    # whichever face applies the softmax scale, so the two conventions cannot
    # hide behind each other.
    (1, 2, 1, 8, 4, 8, False),
    (1, 2, 1, 8, 128, 32, True),
]


def _case(batch, n_heads, n_kv, seq, head_dim, val_dim, causal, seed):
    """One battery case, inputs drawn f32-exactly so the fixture round-trips."""
    rng = np.random.default_rng(seed)
    # Draw, then round through f32 and back: the device consumes f32, so the
    # f64 reference must be computed on the *same* numbers, not on f64 draws
    # that the device could never have received.
    q = rng.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float32).astype(np.float64)
    k = rng.standard_normal((batch, n_kv, seq, head_dim)).astype(np.float32).astype(np.float64)
    v = rng.standard_normal((batch, n_kv, seq, val_dim)).astype(np.float32).astype(np.float64)
    out = attention_tiled(q, k, v, causal=causal)
    return {
        "scenario": "gpu-parity-budget",
        "shape": {
            "batch": batch,
            "n_heads": n_heads,
            "n_kv_heads": n_kv,
            "seq": seq,
            "head_dim": head_dim,
            "val_dim": val_dim,
        },
        "causal": causal,
        "seed": seed,
        "inputs": {
            "query": q.reshape(-1).astype(np.float32).tolist(),
            "key": k.reshape(-1).astype(np.float32).tolist(),
            "value": v.reshape(-1).astype(np.float32).tolist(),
        },
        "faces": {"numpy-reference": {"out": out.reshape(-1).tolist()}},
        "dtypes": {"query": "f32", "key": "f32", "value": "f32", "out": "f32", "reference": "f64"},
    }


def main() -> None:
    records = [_case(*case, seed=9100 + i) for i, case in enumerate(BATTERY)]
    path = FIXTURE_DIR / "attention_tiled.json"
    path.write_text(json.dumps(records, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(records)} cases)")
    for rec in records:
        shape = rec["shape"]
        print(
            f"  batch={shape['batch']} heads={shape['n_heads']}/{shape['n_kv_heads']} "
            f"seq={shape['seq']} d={shape['head_dim']} causal={rec['causal']}"
        )


if __name__ == "__main__":
    main()
