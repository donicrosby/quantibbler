#!/usr/bin/env python
"""Generate the qk-attention parity fixture (tasks 2.1/2.2, scenario E-2.2).

Builds a tiny loader-valid Qwen3 (2 blocks, d_model 64, 4 query heads over
2 KV heads x head_dim 32), quantizes it through the REAL driver path
(``quantize_hf_model`` plumbing: CausalLMLogits + quantize_model), exports
the qk-server directory into tests/fixtures/parity-tiny/, and records the
golden reference the Rust engine must reproduce: the post-quantization HF
forward (quantize_model wrote the artifact back into the live model) at
fp32, teacher-forced on fixed token ids, all positions.

The Rust side (crates/quantibbler-server engine parity test) loads the same
directory via quantibbler-qpk1 and rebuilds the forward from packed
bitplanes + norms.bin + e1.f32.bin; max |Δlogit| budget 1e-2 (E-2.2).

Deterministic (fixed torch seed, fixed ids); regenerating is idempotent
and fixture drift is a reviewable diff.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "parity-tiny"

# Loader-valid GQA geometry: n_kv * head_dim == d_model, n_heads % n_kv == 0.
LAYERS, D_MODEL, N_HEADS, N_KV, HEAD_DIM, INTER, VOCAB = 2, 64, 4, 2, 32, 128, 128
INPUT_IDS = [5, 17, 3, 99, 42, 7, 0, 64]
SEQ_LEN = 32


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "quantize_hf_model", REPO_ROOT / "tools" / "quantize_hf_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    qhf = _load_driver()
    from quantibbler.ptq.api import quantize_model
    from quantibbler.ptq.config import PTQConfig, StageHyper

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=D_MODEL,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        vocab_size=VOCAB,
        head_dim=HEAD_DIM,
        max_position_embeddings=256,
    )
    causal = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    qhf.rewire_trunk_embedding(causal)
    model = qhf.CausalLMLogits(causal)

    torch.manual_seed(1)
    batches = [torch.randint(0, VOCAB, (4, SEQ_LEN)) for _ in range(2)]
    hyper = PTQConfig(
        seed=0,
        rank=8,
        compute_dtype="float64",
        scale_dtype="float16",
        stage_pre=StageHyper(1e-4, 4, 0),
        stage_latent=StageHyper(2e-3, 4, 3),
        stage_global=StageHyper(1e-3, 4, 4),
    )
    qm = quantize_model(model, batches, hyper)

    # Export the quantized artifact (tokenizer is irrelevant to the parity
    # forward; a stub keeps the export path honest without hub lookups).
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    qhf.export_model_directory(
        FIXTURE_DIR,
        qm,
        embedding_f64=causal.get_input_embeddings().weight.detach().to(torch.float64).cpu().numpy(),
        model_id="parity-tiny",
        eos_token_id=0,
        bos_token_id=1,
        tokenizer=type("T", (), {"save_pretrained": staticmethod(lambda d: None)})(),
        config_f64={},
        geometry=qhf._export_geometry(causal),
        norm_weights=qhf._export_norm_weights(causal),
    )

    # Golden reference: post-quantization HF forward at fp32 (the artifact
    # was written back into `causal` by quantize_model), teacher-forced.
    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        logits = causal(ids).logits.to(torch.float32)[0]
    goldens = {
        "input_ids": INPUT_IDS,
        "geometry": qhf._export_geometry(causal),
        # [seq, vocab] fp32, rounded to 1e-6 for a stable reviewable diff.
        "logits": [[round(float(v), 6) for v in row] for row in logits.tolist()],
    }
    (FIXTURE_DIR / "goldens.json").write_text(
        json.dumps(goldens, indent=1) + "\n", encoding="utf-8"
    )
    n = logits.shape[1]
    print(f"wrote {FIXTURE_DIR} (logits [{logits.shape[0]}x{n}])")


if __name__ == "__main__":
    main()
