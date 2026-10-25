#!/usr/bin/env python
"""Capture real attention activations for the gate tool (change qk-sage-attention, task 4.3).

The gate tool's JSON path consumes the committed *fixture battery* (i.i.d.
Gaussian inputs), which this change measured to over-state the INT8 face by
orders of magnitude against real activations ([SA] Table 3's warning landing on
us). This script produces the other evidence class: the operands the model
actually hands its attention face on a held-out eval set, captured from the
healthy base model whose outlier statistics the gate exists to be robust to. A
DEAD qpk1 artifact is not a capture source — a gate measured on a quantization
that failed measures nothing, and the change's design.md records which artifacts
those are and why. Capturing from a *served* qpk1 artifact would need a capture
surface in the Rust engine — a separate change, out of scope here.

How it captures
---------------
The exact and quantized NumPy faces are registered as attention
implementations (``ALL_ATTENTION_FUNCTIONS.register``), and the model is loaded
under one of their names, so ``Qwen3Attention.forward`` calls them with exactly
the tensors they document — post-``q_norm``/``k_norm``, post-RoPE
``query_states``/``key_states``/``value_states``, GQA **un-repeated**, in the
``(B, n_heads, S, d)`` / ``(B, n_kv_heads, S, d)`` layout. Every captured tensor
is therefore a REAL activation. The capture is a property of the *inputs*, not
of the face, so the operands captured are the same whichever registered face the
forward runs (``--capture-face`` selects which one runs; the default is the
quantized face the gate is about, so a capture also exercises it end to end).

The swap is validated before any number is trusted: for every captured case the
exact face is compared against HF's own eager-attention semantics on the SAME
captured operands, and the worst relative deviation is recorded in the capture's
metadata (measured on the Qwen3-0.6B run: 8.5e-7, i.e. the harness noise floor).

Output
------
``tests/fixtures/attention_sage_activations.npz`` by default — the versioned,
checksummed f32 format owned by ``tools/attention_activations.py``, which both
this script and the gate tool read/write. Nothing is written to the working
tree unless ``--out`` says so, and the default target is a deliberate one:

**Large-file gate.** A real-activation capture is deliberately outside the
repo's 500 KB ``check-added-large-files`` gate: one seq-256 case at 16 heads /
128 dim is ~4 MB of f32 operands, so a real battery is orders past it. That
exemption is deliberate (the capture is a numeric record, not a fixture to be
shrunk) and is named in the commit that lands this path.

Run it (the model and wikitext-2 ride the local HF cache; the driver's
``HF_HUB_CACHE`` contract applies and is passed on to HF)::

    HF_HUB_CACHE=/opt/data/hf-cache/hub uv run --no-sync python \\
        tools/capture_attention_activations.py \\
        --model Qwen/Qwen3-0.6B --layers all --capture-windows 3 \\
        --seq-len 256 --block-size 64

Then mint the record from the capture (this tree cannot measure the eval-set
PPL clause; the gate admits a record that declares none, at WARN)::

    uv run --no-sync python tools/gen_attention_gate_record.py \\
        --model-dir <model dir> --activations tests/fixtures/attention_sage_activations.npz
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import random
import sys
import time
from pathlib import Path

# Fail closed on the network before transformers/HF get a chance to phone home,
# and keep this box's memory/thread pins identical to the driver's: a capture
# that silently downloads a different model, or that saturates a shared box, is
# not the record this change is auditing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from attention_activations import CapturedCase, write_capture  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

from quantibbler.kernels.attention_sage import attention_sage  # noqa: E402
from quantibbler.kernels.attention_tiled import attention_tiled  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "attention_sage_activations.npz"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
EXACT_FACE = "sage_exact"
QUANT_FACE = "sage_quant"

# Capture state. Module-level because the attention interface call carries no
# channel of its own (the runbook's mechanics note), and flagged off between
# passes so a later forward cannot overwrite the capture.
CAPTURE_ENABLED = False
#: Layers to capture; ``None`` means every layer. Set from ``--layers`` so the
#: flag selects the battery's layers rather than merely labelling it.
CAPTURE_LAYERS: set[int] | None = None
#: The window index of the forward currently in flight — set by the driver, read
#: by the interface, so a case's coordinates are (layer, window).
CURRENT_WINDOW = [0]
CAPTURED: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _host(tensor: torch.Tensor) -> np.ndarray:
    """Device/host hop: f64, the dtype both reference faces consume."""
    return tensor.detach().to(torch.float64).cpu().numpy()


def _torch_eager_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scaling: float
) -> np.ndarray:
    """HF ``eager_attention_forward`` semantics on ``(B, H, S, d)`` / ``(B, Hkv, S, d)``."""
    qq = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32))
    kk = torch.from_numpy(np.ascontiguousarray(k, dtype=np.float32))
    vv = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    repeat = q.shape[1] // k.shape[1]
    if repeat > 1:
        kk = kk.repeat_interleave(repeat, dim=1)
        vv = vv.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(qq, kk.transpose(-1, -2)) * scaling
    seq = q.shape[-2]
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(attn, vv).numpy().astype(np.float64)


def _make_interface(face: str, block_size: int):
    """HF's ``attention_interface`` for one registered face.

    Returns ``(attn_output_channels_last, attn_weights)``: both faces produce
    ``(B, H, S, d)``, so the output is transposed back to HF's ``(B, S, H, d)``.
    """

    def fn(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **_kw):
        layer = int(getattr(module, "layer_idx", -1))
        q, k, v = _host(query), _host(key), _host(value)
        if CAPTURE_ENABLED and (CAPTURE_LAYERS is None or layer in CAPTURE_LAYERS):
            CAPTURED[(layer, CURRENT_WINDOW[0])] = (q, k, v)
        if face == QUANT_FACE:
            out = attention_sage(q, k, v, causal=True, block_size=block_size)
        else:
            out = attention_tiled(q, k, v, causal=True)
        host = torch.from_numpy(np.ascontiguousarray(out)).to(query.dtype).to(query.device)
        return host.transpose(1, 2).contiguous(), None

    return fn


def _wikitext_shards(hub: Path, split: str) -> list[Path]:
    roots = sorted(hub.glob("datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1"))
    if not roots:
        raise SystemExit(f"wikitext-2-raw-v1 not cached under {hub} (set HF_HUB_CACHE)")
    shards = sorted(roots[0].glob(f"{split}-*.parquet"))
    if not shards:
        raise SystemExit(f"no '{split}-*.parquet' under {roots[0]}")
    return shards


def _load_text(shards: list[Path]) -> str:
    parts: list[str] = []
    for shard in shards:
        column = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
        parts.extend(line for line in column if line.strip())
    return "\n".join(parts)


def _windows(tokenizer, text: str, seq_len: int, count: int, seed: int) -> list[torch.Tensor]:
    """``count`` deterministic windows of ``seq_len`` tokens (block-shuffled, the driver's rule)."""
    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    stream = torch.tensor(ids, dtype=torch.long)
    blocks = int(stream.shape[0] // seq_len)
    if blocks < count:
        raise SystemExit(f"wikitext yields {blocks} blocks of {seq_len}; need {count}")
    order = list(range(blocks))
    random.Random(seed).shuffle(order)
    reshaped = stream[: blocks * seq_len].view(blocks, seq_len)
    return [reshaped[i : i + 1] for i in order[:count]]


def _hub_cache() -> Path:
    """The HF hub cache, resolved by the driver's own contract."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache/huggingface"
    return base / "hub"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base model id or local snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="capture target (.npz)")
    parser.add_argument("--split", default="test", help="wikitext-2 split")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--capture-windows",
        type=int,
        default=3,
        help="eval windows to capture; every --layers layer contributes one case per window",
    )
    parser.add_argument("--layers", default="all", help="'all' or a comma-separated layer list")
    parser.add_argument("--block-size", type=int, default=64, help="the face's block granularity")
    parser.add_argument(
        "--capture-face",
        choices=(QUANT_FACE, EXACT_FACE),
        default=QUANT_FACE,
        help="registered face the capturing forward runs (the capture is of the inputs)",
    )
    parser.add_argument("--validate-swap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1397)
    args = parser.parse_args(argv)

    hub = _hub_cache()

    started = time.time()
    ALL_ATTENTION_FUNCTIONS.register(EXACT_FACE, _make_interface(EXACT_FACE, args.block_size))
    ALL_ATTENTION_FUNCTIONS.register(QUANT_FACE, _make_interface(QUANT_FACE, args.block_size))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.capture_face
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    layers = (
        list(range(int(model.config.num_hidden_layers)))
        if args.layers.strip().lower() == "all"
        else sorted({int(x) for x in args.layers.split(",") if x.strip()})
    )
    print(
        f"[load] {time.time() - started:.1f}s model={args.model} "
        f"attn={model.config._attn_implementation} layers={len(layers)} "
        f"heads={model.config.num_attention_heads}/{model.config.num_key_value_heads} "
        f"d={getattr(model.config, 'head_dim', None)}"
    )

    shards = _wikitext_shards(hub, args.split)
    text = _load_text(shards)
    windows = _windows(tokenizer, text, args.seq_len, args.capture_windows, args.seed)
    print(
        f"[data] split={args.split} shards={len(shards)} windows={len(windows)} "
        f"seq_len={args.seq_len} tokens={len(windows) * args.seq_len}"
    )

    global CAPTURE_ENABLED, CAPTURE_LAYERS
    CAPTURE_ENABLED = True
    CAPTURE_LAYERS = set(layers)
    for index, window in enumerate(windows):
        CURRENT_WINDOW[0] = index
        with torch.no_grad():
            model(window)
        captured = sum(1 for _, win in CAPTURED if win == index)
        print(
            f"[capture] window {index}: {captured} layers captured, "
            f"{time.time() - started:.1f}s elapsed"
        )
    CAPTURE_ENABLED = False
    if not CAPTURED:
        raise SystemExit("no cases captured — check --layers and --capture-windows")

    cases: list[CapturedCase] = []
    worst_swap = 0.0
    for (layer, window_index), (q, k, v) in sorted(CAPTURED.items()):
        if args.validate_swap:
            exact = attention_tiled(q, k, v, causal=True)
            eager = _torch_eager_attention(q, k, v, float(q.shape[-1]) ** -0.5)
            worst_swap = max(
                worst_swap, float(np.linalg.norm(eager - exact) / np.linalg.norm(exact))
            )
        cases.append(
            CapturedCase(
                q=q,
                k=k,
                v=v,
                causal=True,
                block_size=args.block_size,
                layer=layer,
                window=window_index,
            )
        )

    metadata = {
        "instrument": "tools/capture_attention_activations.py",
        "captured_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "model_class": type(model).__name__,
        "model_config": model.config.to_dict(),
        "split": args.split,
        "seq_len": args.seq_len,
        "windows": len(windows),
        "layers": layers,
        "block_size": args.block_size,
        "capture_face": args.capture_face,
        "seed": args.seed,
        "dtype": "bfloat16",
        "swap_validation": {
            "enabled": args.validate_swap,
            "worst_exact_vs_hf_eager_relative": worst_swap if args.validate_swap else None,
        },
        "interpreter": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
        },
        "hf_hub_cache": str(hub),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    capture = write_capture(
        args.out,
        cases,
        capture_model=args.model,
        metadata=metadata,
    )
    out = capture.path
    kib = out.stat().st_size / 1024
    print(
        f"\n[wrote] {out} ({kib:,.1f} KiB, {len(capture.cases)} cases, "
        f"format_version={capture.format_version})"
    )
    print(f"[kind] {capture.activations_kind}   [checksum] {capture.checksum}")
    if args.validate_swap:
        print(f"[swap] worst exact-vs-HF-eager relative {worst_swap:.3e} (harness noise floor)")
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:  # an --out outside the repo (a scratch capture)
        shown = out
    print(
        "[gate] mint the record with:\n"
        f"    uv run --no-sync python tools/gen_attention_gate_record.py "
        f"--model-dir <model dir> --activations {shown}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
