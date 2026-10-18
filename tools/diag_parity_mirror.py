#!/usr/bin/env python
"""Diagnostic: mirror of the Rust engine's transformer (same op
order, same f32) diffed against HF stage hooks on the parity-tiny artifact.
Localizes the E-2.2 divergence: if the mirror matches HF, the Rust port has
a code bug; if the mirror diverges the same way, the ALGORITHM misreads the
HF convention."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np
import torch

INPUT_IDS = [5, 17, 3, 99, 42, 7, 0, 64]
LAYERS, D_MODEL, N_HEADS, N_KV, HEAD_DIM, INTER, VOCAB = 2, 64, 4, 2, 32, 128, 128


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "quantize_hf_model", REPO_ROOT / "tools" / "quantize_hf_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model_and_artifact():
    """Same path as gen_parity_fixture: real driver PTQ on a tiny Qwen3."""
    qhf = _load_driver()
    from transformers import Qwen3Config, Qwen3ForCausalLM

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
    batches = [torch.randint(0, VOCAB, (4, 32)) for _ in range(2)]
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
    return causal, qm, qhf


def mirror_forward(causal, ids):
    """The Rust engine's algorithm, verbatim, in torch fp32."""
    eps = float(causal.config.rms_norm_eps)
    theta = 10000.0

    def rms(x, w):
        ss = (x * x).sum(-1, keepdim=True) / x.shape[-1]
        return x * (1.0 / torch.sqrt(ss + eps)) * w

    def rope(v, pos):  # rotate_half, per head, f32
        b = v.shape[0] if v.dim() > 1 else 1
        half = HEAD_DIM // 2
        inv = theta ** (-2.0 * torch.arange(half, dtype=torch.float64) / HEAD_DIM)
        ang = pos * inv
        s, c = torch.sin(ang).float(), torch.cos(ang).float()
        v = v.reshape(b, HEAD_DIM)
        a, h2 = v[:, :half], v[:, half:]
        out = torch.cat([a * c - h2 * s, h2 * c + a * s], -1)
        return out.view(-1)

    def attn_block(x, li, pos, cache_li):
        layer = causal.model.layers[li]
        normed = rms(x, layer.input_layernorm.weight.float())
        q = (normed @ layer.self_attn.q_proj.weight.float().T).view(N_HEADS, HEAD_DIM)
        k = (normed @ layer.self_attn.k_proj.weight.float().T).view(N_KV, HEAD_DIM)
        v = (normed @ layer.self_attn.v_proj.weight.float().T).view(N_KV, HEAD_DIM)
        q = rms(q, layer.self_attn.q_norm.weight.float())
        k = rms(k, layer.self_attn.k_norm.weight.float())
        q = torch.stack([rope(q[h], pos) for h in range(N_HEADS)])
        k = torch.stack([rope(k[h], pos) for h in range(N_KV)])
        cache_li[0].append(k)
        cache_li[1].append(v)
        groups = N_HEADS // N_KV
        outs = []
        for h in range(N_HEADS):
            kv = h // groups
            # single query position attends over cached 0..pos
            cache_k, cache_v = kv_cache[li]
            s_all = torch.stack(
                [torch.dot(cache_k[p][kv], q[h]) / np.sqrt(HEAD_DIM) for p in range(pos + 1)]
            )
            w = torch.softmax(s_all, -1)
            outs.append(
                torch.einsum(
                    "p,ph->h",
                    w,
                    torch.stack([cache_v[p][kv] for p in range(pos + 1)]),
                )
            )
        ctx = torch.cat(outs)
        return ctx @ layer.self_attn.o_proj.weight.float().T + x

    def mlp_block(x, li):
        layer = causal.model.layers[li]
        normed = rms(x, layer.post_attention_layernorm.weight.float())
        gate = normed @ layer.mlp.gate_proj.weight.float().T
        up = normed @ layer.mlp.up_proj.weight.float().T
        down = (torch.nn.functional.silu(gate) * up) @ layer.mlp.down_proj.weight.float().T
        return down + x

    h = causal.get_input_embeddings().weight.float()[ids[0]]
    kv_cache = [[[], []] for _ in range(LAYERS)]  # per layer: [k_rows, v_rows]
    for pos in range(h.shape[0]):
        x = h[pos]
        for li in range(LAYERS):
            x = attn_block(x, li, pos, kv_cache[li])
            x = mlp_block(x, li)
        x = rms(x, causal.model.norm.weight.float())
        h[pos] = x
    logits = h @ causal.lm_head.weight.float().T
    return logits


def stage_probe(causal):
    """pos-0 block-0 intermediates, printed in the Rust probe's units."""
    eps = float(causal.config.rms_norm_eps)
    layer = causal.model.layers[0]
    x0 = causal.get_input_embeddings().weight.float()[INPUT_IDS[0]]
    ss = (x0 * x0).sum() / x0.shape[-1]
    normed = x0 / torch.sqrt(ss + eps) * layer.input_layernorm.weight.float()
    print("normed", [round(v, 7) for v in normed[:4].tolist()])
    q = normed @ layer.self_attn.q_proj.weight.float().T
    print("q", [round(v, 9) for v in q[:4].tolist()])
    # q-norm over head_dim on head 0
    qh = q.view(N_HEADS, HEAD_DIM)[0]
    ss = (qh * qh).sum() / HEAD_DIM
    qn = qh / torch.sqrt(ss + eps) * layer.self_attn.q_norm.weight.float()
    print("q_rot", [round(v, 7) for v in qn[:4].tolist()])


def main():
    causal, _, _ = build_model_and_artifact()
    # Engine-side math: f16-scale-exact fp32 products. The HF model holds
    # bf16-rounded W_q; compare BOTH against the mirror.
    causal_f32 = causal.to(torch.float32)
    ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        golden = causal_f32(ids).logits[0]
    stage_probe(causal_f32)
    mirror = mirror_forward(causal_f32, ids)
    d = (golden - mirror).abs().max().item()
    print(f"mirror vs HF-f32 max|delta| = {d:.6f}")


if __name__ == "__main__":
    main()
