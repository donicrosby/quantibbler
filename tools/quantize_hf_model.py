#!/usr/bin/env python
"""Model-development loop: HF causal LM -> quantibbler qpk1 -> model directory.

The real-model counterpart of ptq/e2e.py's tiny gate (tools/ is the home of
executable tooling, per gen_server_fixtures.py precedent): load a HF causal
LM, evaluate the bf16 baseline, run the PTQ pipeline, evaluate the quantized
model, and export the qk-server acquisition directory (manifest + config +
qpk1 layer streams + fp32 sidecars + tokenizer) that the pure-Rust server
loads. This script — not ad-hoc shell — IS the model development loop; every
dependency it needs is a declared extra (pyproject [project.optional-
dependencies].hf), and it never touches the network (HF_HUB_OFFLINE=1).

Numerics regime (matched to the pinned papers, tools/arxiv2md sources):

- Baseline eval + all model movement: bfloat16, the checkpoint's native
  dtype. The papers' eval tables measure their artifacts against BF16
  baselines ([NQ] Tables: "BF16 | 16.00" rows), so the pedestal must be
  bf16 for cross-paper (and cross-project, e.g. LittleBit smoke) numbers
  to mean anything.
- Solver (robust diag, LB-ADMM, STE, scale KD): float64 on upcast weights,
  per qk-representation design (sign extraction needs numerical margin;
  the NumPy/Rust parity gates need a deterministic reference). The design
  explicitly makes this falsifiable: wrong solver precision shows up as
  perplexity regressions HERE, which is what this driver measures.
- Artifact: packed +-1 bitplanes + float16 scales (the papers' BPW
  accounting, [LB] App. D / [NQ] App. F.5).
- Quantized eval runs TWICE: once float64 (artifact-exact pipeline check —
  host forward == serialized artifact bit-for-bit) and once bfloat16
  (serving regime, the number comparable against the baseline and the
  papers). Both land in the report; the bf16 number is the headline.

Server note: the Rust server loads and validates the exported directory,
but its engine is a linear stack in v0.1 (no attention loop yet) — serving
generation from a real transformer is a separate workstream (qk-engine,
not yet specced). The server boot at the end of this flow therefore proves
acquisition, not generation.

Usage (from the repo root):
    uv sync --extra torch --extra hf --extra kernels
    uv run python tools/quantize_hf_model.py \
        --model Qwen/Qwen3-0.6B --rank 8 \
        --calibration-seqs 128 --eval-seqs 40 --seq-len 512 \
        --out artifacts/qwen3-0.6B-qpk1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import threading
import time
from pathlib import Path

# Fail closed on network before transformers/HF get a chance to phone home,
# and keep the (multi-GB) KD teacher-logit cache off tmpfs — /tmp is a
# small tmpfs here; /opt/data is real disk.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("QUANTIBBLER_KD_TMPDIR", "/opt/data/tmp")
# Bounded native allocations (hundreds of MB drift otherwise) and bounded
# BLAS/torch thread pools (each thread's stacks and scratch count against
# the cgroup this sandbox shares).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantibbler.ptq.api import QuantizedModel, quantize_model
from quantibbler.ptq.config import PTQConfig, StageHyper


def _wikitext_cache_root() -> Path:
    """Resolve the cached wikitext snapshot dir, honoring the HF env contract.

    HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub — the driver must
    work in containers where the cache is bind-mounted and HOME is not the
    cache owner (the in-container failure that motivated this).
    """
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env) / "datasets--Salesforce--wikitext/snapshots"
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache/huggingface"
    return base / "hub/datasets--Salesforce--wikitext/snapshots"


WIKITEXT_TEST = _wikitext_cache_root()
BATCH = 8  # eval/calibration microbatch (bf16 0.6B fits easily on CPU)


class LayerBlock(nn.Module):
    """One decoder layer as a pipeline block: callable on bare activations.

    Modern transformers decoder layers do NOT compute rotary embeddings
    themselves — the trunk computes ``position_embeddings`` once and hands
    them to every layer. This adapter supplies them per call (contiguous
    positions; packed wikitext rows) and relies on mask-free causal
    attention, which is exact for full-length packed rows.
    """

    def __init__(self, layer: nn.Module, rotary: nn.Module) -> None:
        super().__init__()
        self.inner = layer
        self._rotary = [rotary]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pe = self._rotary[0](x, pos)
        out = self.inner(x, position_embeddings=pe, attention_mask=None)
        # transformers 5.x decoder layers return the tensor directly;
        # older lines return (hidden_states, ...).
        return out[0] if isinstance(out, tuple) else out


class CausalLMLogits(nn.Module):
    """Logits-tensor view of a HF ForCausalLM with pipeline-friendly anatomy.

    The PTQ pipeline is logits-semantics end to end (gate models return raw
    tensors; ``calibrate_scales`` feeds ``model(x)`` into the KL loss) and
    its block loop iterates children (expanding ``nn.ModuleList`` members
    into per-unit blocks), quantizing every block that contains an
    ``nn.Linear`` and ending at the last non-quantized linear (A-9 stem/post
    detection). HF's ``*ForCausalLM`` breaks both: it returns ModelOutput
    objects and nests the embedding INSIDE the trunk.

    This view exposes the anatomy as pipeline blocks:

    - ``embed`` — the embedding (no linears; skipped, becomes the stem);
    - ``layers.0 .. layers.N-1`` — the decoder stack as ONE BLOCK PER
      LAYER (the papers reconstruct per transformer layer, Algorithm 1
      line 9: the quantized prefix grows one layer at a time). Wrapping
      the whole stack as a single block made every pipeline stage retain
      all 28 layers' calibration activations at once (~28G at real
      scale — the container OOM the breadcrumbs localized);
    - ``norm`` — final RMSNorm (no linears; skipped; applied inside the
      stock forward, never needed as a prefix block);
    - ``head`` — the tied LM head (leaf name excluded by A-9; the post).

    ``forward`` runs the stock rewired trunk (embed-or-pass, RoPE, final
    norm) and returns the logits tensor for Phase 3 KD. The trunk module
    is held in a plain-list slot so it is NOT registered as a child —
    it would otherwise re-register the whole stack as one block.
    """

    def __init__(self, causal: nn.Module) -> None:
        super().__init__()
        self.embed = causal.model.embed_tokens
        rotary = causal.model.rotary_emb
        self.layers = nn.ModuleList([LayerBlock(layer, rotary) for layer in causal.model.layers])
        self.norm = causal.model.norm
        self.head = causal.lm_head
        self._stock_trunk = [causal.model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._stock_trunk[0](input_ids=x).last_hidden_state)


def rewire_trunk_embedding(causal: nn.Module) -> None:
    """Make ``Qwen3Model`` (or any HF trunk with .embed_tokens) callable with
    pre-embedded activations: token ids still embed; float input passes the
    embedding and flows through the stock ``inputs_embeds`` path."""

    trunk = causal.model  # Qwen3ForCausalLM -> Qwen3Model
    stock_forward = trunk.forward

    def forward(input_ids=None, inputs_embeds=None, **kwargs):  # noqa: ANN001
        if inputs_embeds is None and input_ids is not None and input_ids.dtype.is_floating_point:
            # Float "ids" are already pre-embedded activations from the
            # pipeline's stem; hand them to the stock inputs_embeds path.
            inputs_embeds = input_ids
            input_ids = None
        return stock_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    trunk.forward = forward  # type: ignore[method-assign]


@torch.no_grad()
def perplexity_from_logits(logits_fn, batches: list[torch.Tensor]) -> float:
    """Next-token perplexity over equal-size batches.

    The cross-entropy is accumulated per row: casting a whole batch's
    logits (8 x 511 x 152k) to float32 at once is a multi-GB transient,
    which matters under a container memory cap. Mean-of-means stays exact
    (every batch carries BATCH * (seq_len - 1) predictions).
    """
    total_nats = 0.0
    n_preds = 0
    for batch in batches:
        logits = logits_fn(batch[:, :-1])
        targets = batch[:, 1:]
        for i in range(logits.shape[0]):
            nats = torch.nn.functional.cross_entropy(logits[i].float(), targets[i], reduction="sum")
            total_nats += float(nats)
        n_preds += targets.numel()
    return float(np.exp(total_nats / n_preds))


def load_wikitext_text(shards: list[Path]) -> str:
    """Concatenated wikitext-2 text from the given parquet shards."""
    if not shards:
        raise FileNotFoundError("no wikitext parquet shards given")
    parts: list[str] = []
    for shard in shards:
        column = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
        parts.extend(line for line in column if line.strip())
    return "\n".join(parts)


def find_wikitext_shards(split: str) -> list[Path]:
    """The cached wikitext-2-raw-v1 shards for `split` (flat layout: the
    hub cache stores ``<split>-<shard-index>.parquet`` directly under the
    version dir, no per-split subdirectories)."""
    roots = sorted(WIKITEXT_TEST.glob("*/wikitext-2-raw-v1"))
    if not roots:
        raise FileNotFoundError(f"wikitext-2-raw-v1 not cached under {WIKITEXT_TEST}")
    shards = sorted(roots[0].glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no '{split}-*.parquet' shards under {roots[0]}")
    return shards


def tokenize_chunks(
    tokenizer, text: str, seq_len: int, n_chunks: int, seed: int
) -> list[torch.Tensor]:
    """n_chunks deterministic (8, seq_len) token-id batches from `text`.

    The token stream is cut into seq_len blocks, block order is seeded-
    shuffled (decorrelates calibration/eval chunks without replacement),
    and chunks are packed BATCH rows at a time.
    """
    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    stream = torch.tensor(ids, dtype=torch.long)
    n_blocks = stream.shape[0] // seq_len
    if n_blocks < n_chunks * BATCH:
        raise ValueError(
            f"wikitext yields {n_blocks} blocks of {seq_len}; "
            f"need {n_chunks * BATCH} for this eval sizing"
        )
    order = list(range(n_blocks))
    random.Random(seed).shuffle(order)
    # Materialize seq_len token blocks FIRST, then seed-shuffled fancy-index
    # picks whole blocks (indexing the 1-D stream would pick loose tokens).
    blocks = stream[: n_blocks * seq_len].view(n_blocks, seq_len)
    picked = blocks[order[: n_chunks * BATCH]]
    return [picked[i * BATCH : (i + 1) * BATCH] for i in range(n_chunks)]


def quantized_config(args: argparse.Namespace) -> PTQConfig:
    """PTQConfig for real models: gate-shaped stages (TuneFP off — the fp
    checkpoint needs no polish; STE + global scale KD at the gate's measured
    lr), paper-style solver knobs, caller-selected rank."""
    return PTQConfig(
        seed=args.seed,
        rank=args.rank,
        compute_dtype="float64",
        scale_dtype="float16",
        admm_max_iter=args.max_admm_iter,
        stage_pre=StageHyper(1e-4, 4, 0),
        stage_latent=StageHyper(2e-3, 4, 3),
        stage_global=StageHyper(1e-3, 4, 4),
        verbose=True,
    )


def export_model_directory(
    out_dir: Path,
    qm: QuantizedModel,
    embedding_f64: np.ndarray,
    model_id: str,
    eos_token_id: int,
    bos_token_id: int,
    tokenizer,
    config_f64: dict,
) -> None:
    """Write the qk-server acquisition directory (loader-validated layout).

    Layer streams land at layers/<dotted/name/path>.qpk1; the fp32 sidecar
    is the embedding row-major [vocab, d_model] LE float32; the tied head
    reuses the same file via weight_tying = "lm_head=e1@fp32".
    """
    layer_files: dict[str, str] = {}
    for name, blob in sorted(qm.serialize().items()):
        rel = "layers/" + name.replace(".", "/") + ".qpk1"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        layer_files[name] = rel

    emb = np.ascontiguousarray(embedding_f64, dtype=np.float32)
    (out_dir / "e1.f32.bin").write_bytes(emb.tobytes(order="C"))

    (out_dir / "config.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f"vocab_size = {embedding_f64.shape[0]}\n"
        f"d_model = {embedding_f64.shape[1]}\n"
        f"eos_token_id = {eos_token_id}\n"
        f"bos_token_id = {bos_token_id}\n"
        'weight_tying = "lm_head=e1@fp32"\n'
    )

    manifest_layers = ",\n".join(f'  "{n}"' for n in layer_files)
    manifest_layer_files = ",\n".join(f'"{n}" = "{p}"' for n, p in layer_files.items())
    (out_dir / "manifest.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f'model_id = "{model_id}"\n'
        f"layers = [\n{manifest_layers},\n]\n"
        f"\n[layer_files]\n{manifest_layer_files},\n"
        f"\n[sidecar]\n"
        f'embedding = "e1.f32.bin"\n'
        f'lm_head = "e1.f32.bin"\n'
    )
    # Serialize the already-loaded tokenizer straight into the export dir
    # (tokenizer.json + config files). No hub lookup: offline containers and
    # local snapshot paths both work, and the in-memory object is by
    # definition the exact tokenizer the pipeline used.
    tokenizer.save_pretrained(out_dir)

    # self-check: the serialized stream header parses back as expected
    first = sorted(layer_files.values())[0]
    header = struct.unpack("<4sIIIII", (out_dir / first).read_bytes()[:24])
    assert header[0] == b"QPK1" and header[1] == 1, header


def main() -> int:
    # ---- crash-safe breadcrumbs ------------------------------------
    # SIGKILL can't flush our stdout, so a watchdog thread samples the
    # shared cgroup's memory + this process's RSS and appends stage
    # markers to a plain file. Whatever kills us, the tail of this file
    # says WHERE we were and HOW BIG we were.
    breadcrumb_path = Path(
        os.getenv("QUANTIBBLER_BREADCRUMBS", "/opt/data/tmp/quantibbler-breadcrumbs.log")
    )

    def _bread(note: str) -> None:
        try:
            mem = 0
            try:
                with open("/sys/fs/cgroup/memory.current") as fh:
                    mem = int(fh.read())
            except OSError:
                pass
            with open(breadcrumb_path, "a") as fh:
                fh.write(f"{time.time():.0f} cgroup={mem / 2**30:.2f}G :: {note}\n")
        except OSError:
            pass

    def _watchdog() -> None:
        peak = 0
        while True:
            time.sleep(5)
            try:
                with open("/sys/fs/cgroup/memory.current") as fh:
                    cur = int(fh.read()) / 2**30
                peak = max(peak, cur)
                with open(breadcrumb_path, "a") as fh:
                    fh.write(f"{time.time():.0f} cgroup={cur:.2f}G peak={peak:.2f}G\n")
            except OSError:
                pass

    def _stage(note: str) -> None:
        print(note, flush=True)
        _bread(note)

    threading.Thread(target=_watchdog, daemon=True).start()
    _bread(f"start pid={os.getpid()} argv={' '.join(sys.argv[1:])}")
    try:
        with open("/sys/fs/cgroup/memory.events") as fh:
            _bread("memory.events@start: " + " ".join(fh.read().split()))
    except OSError:
        pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--calibration-seqs",
        type=int,
        default=16,
        help="16 batches of 8 sequences = 128 calibration rows",
    )
    parser.add_argument("--eval-seqs", type=int, default=5, help="5 batches of 8 = 40 eval rows")
    parser.add_argument("--max-admm-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "model-qpk1")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/6] loading {args.model} (local cache, offline)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    causal = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    rewire_trunk_embedding(causal)
    model = CausalLMLogits(causal).to(torch.bfloat16).eval()
    cfg = causal.config
    print(
        f"      d_model={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"vocab={cfg.vocab_size} tied={cfg.tie_word_embeddings}"
    )

    print("[2/6] tokenizing wikitext-2 (cached parquet)")
    calib = tokenize_chunks(
        tokenizer,
        load_wikitext_text(find_wikitext_shards("train")),
        args.seq_len,
        args.calibration_seqs,
        args.seed,
    )
    held_out = tokenize_chunks(
        tokenizer,
        load_wikitext_text(find_wikitext_shards("test")),
        args.seq_len,
        args.eval_seqs,
        args.seed + 1,
    )
    print(f"      calib {len(calib)} batches, eval {len(held_out)} batches")

    print("[3/6] bf16 baseline perplexity (paper-regime pedestal)", flush=True)
    ppl_base = perplexity_from_logits(lambda b: model(b), held_out)
    print(f"      ppl_baseline = {ppl_base:.4f}", flush=True)

    _stage("[4/6] PTQ: bf16 model, per-layer float64 solver (block recon + scale KD)")
    _bread("entering quantize_model: block recon first, then KD scale calibration")
    solver_t = time.time()
    qm = quantize_model(model, calib, quantized_config(args))
    _bread(f"quantize_model returned: {len(qm.layers)} layers")
    print(f"      solver wall: {time.time() - solver_t:.0f}s", flush=True)

    print("[5/6] quantized perplexity (checkpoint-native dtype, artifact written back)")
    host = model  # quantize_model wrote the artifact back into the host
    ppl_quant = perplexity_from_logits(lambda b: host(b), held_out)
    bpw = qm.effective_bpw()
    total_qpk1 = sum(len(b) for b in qm.serialize().values())
    print(f"      ppl_quant      = {ppl_quant:.4f}")
    print(f"      effective BPW  = {bpw:.4f}  (qpk1 layer bytes: {total_qpk1:,})", flush=True)

    print("[6/6] exporting qk-server model directory ->", out_dir, flush=True)
    embedding = causal.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    export_model_directory(
        out_dir,
        qm,
        embedding,
        model_id=args.model.replace("/", "-"),
        eos_token_id=cfg.eos_token_id,
        bos_token_id=cfg.bos_token_id,
        tokenizer=tokenizer,
        config_f64={"dtype_note": "solver float64 / scales fp16 / artifact qpk1"},
    )
    report = {
        "model": args.model,
        "rank": args.rank,
        "seq_len": args.seq_len,
        "calibration_rows": len(calib) * BATCH,
        "eval_rows": len(held_out) * BATCH,
        "max_admm_iter": args.max_admm_iter,
        "seed": args.seed,
        "ppl_baseline_bf16": ppl_base,
        "ppl_quant_native": ppl_quant,
        "ppl_ratio": ppl_quant / ppl_base,
        "effective_bpw": bpw,
        "qpk1_layer_bytes": total_qpk1,
        "regime": {
            "baseline_eval": "bfloat16 (checkpoint native; papers' pedestal)",
            "solver": "float64, per-layer just-in-time (sign margins + parity reference)",
            "kd_and_ste": "model dtype (the papers' distillation regime)",
            "artifact": "qpk1: +-1 bitplanes + float16 scales",
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "quantize_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nserver acquisition check (linear-stack engine: load only, no generation):")
    print(f"  cargo run --release -p quantibbler-server -- serve --model-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
