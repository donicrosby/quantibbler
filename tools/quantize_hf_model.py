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
import hashlib
import os
import random
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Fail closed on network before transformers/HF get a chance to phone home,
# and keep the (multi-GB) KD teacher-logit cache off tmpfs — /tmp is a
# small tmpfs here; /opt/data is real disk. That preference applies ONLY where
# the directory exists: a default that cannot exist off-sandbox is what turned
# the first full-python CI run red (a runner has no /opt/data, and
# `tempfile.TemporaryDirectory` does not create parents), so every other host
# gets the platform scratch dir instead of a path that is guaranteed missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_SANDBOX_SCRATCH = Path("/opt/data/tmp")
os.environ.setdefault(
    "QUANTIBBLER_KD_TMPDIR",
    str(_SANDBOX_SCRATCH) if _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir(),
)
# Bounded native allocations (hundreds of MB drift otherwise) and bounded
# BLAS/torch thread pools (each thread's stacks and scratch count against
# the cgroup this sandbox shares).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# numpy's scipy-openblas is PTHREADS-backed: OMP/MKL pins do NOT size its
# pool, which defaults to visible-CPU count. Measured 2026-10-18 (r64
# slab): unpinned 235.5s vs pinned-to-optimum 3.98s — a 59x thread-thrash
# artifact, not algorithm scaling. Pin it with the others.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantibbler.ptq.api import QuantizedModel, quantize_model
from quantibbler.ptq.block_recon import VacuousTeacherError
from quantibbler.ptq.config import PrefixProbeAbort, PTQConfig, StageHyper
from quantibbler.qk_report import (
    COLLAPSE_LINE,
    EVAL_EVALUATOR,
    FAILED,
    MIN_PROBE_DEPTH,
    OK_LINE,
    abort_report,
    dump_report,
    evaluation_record,
    fidelity_verdict,
    geometry_deviations,
    probe_aborts,
    probe_ratio,
    provenance,
    stage_failure_report,
    stage_map,
    verdict,
)
from quantibbler.qk_report import (
    exit_code as verdict_exit_code,
)


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
        # Inputs arrive tokenized on the CPU; the weights may be on an accelerator.
        # The device is read off the module's own parameters rather than threaded
        # down from the driver: a second source of truth can disagree with where
        # the weights actually are, and the disagreement surfaces as a
        # device-mismatch error several frames from its cause. Every move here is
        # a no-op on a CPU tensor, so the reference path is unchanged rather than
        # merely re-tested.
        _dev = next(causal.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(_dev)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(_dev)
        kwargs = {k: (v.to(_dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
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


def _tokenize_blocks(
    tokenizer, text: str, seq_len: int, n_chunks: int, seed: int
) -> tuple[list[torch.Tensor], dict]:
    """``(batches, selection record)``: the chunks AND how they were chosen.

    The record is what makes an evaluation auditable (source tokens, complete
    blocks, and the ordered block indices actually picked); returning it from
    the same pass that builds the batches keeps the record from drifting to a
    second, separately-seeded selection.
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
    picked_index = order[: n_chunks * BATCH]
    picked = blocks[picked_index]
    batches = [picked[i * BATCH : (i + 1) * BATCH] for i in range(n_chunks)]
    return batches, {
        "source_tokens": int(stream.shape[0]),
        "full_blocks": int(n_blocks),
        "picked_index": picked_index,
    }


def tokenize_chunks(
    tokenizer, text: str, seq_len: int, n_chunks: int, seed: int
) -> list[torch.Tensor]:
    """n_chunks deterministic (8, seq_len) token-id batches from `text`.

    The token stream is cut into seq_len blocks, block order is seeded-
    shuffled (decorrelates calibration/eval chunks without replacement),
    and chunks are packed BATCH rows at a time.
    """
    return _tokenize_blocks(tokenizer, text, seq_len, n_chunks, seed)[0]


def _positive_int(text: str) -> int:
    """argparse type for the probe knobs: a non-positive value is a parse error.

    Rejected at parsing, not clamped downstream: a silently-accepted
    ``--probe-interval 0`` would disable the very probe the run asked for,
    and the operator would read the absence of probe lines as "healthy".
    """
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def quantized_config(
    args: argparse.Namespace,
    progress_probe: Any = None,
) -> PTQConfig:
    """PTQConfig for real models: gate-shaped stages (TuneFP off — the fp
    checkpoint needs no polish; STE + global scale KD at the gate's measured
    lr), paper-style solver knobs, caller-selected rank, and the per-block
    progress probe when the caller wired one in (None = no probe, the
    pipeline path is unchanged).

    ``--paper-faithful`` switches the stage budgets to [NQ] App. C verbatim
    (T_pre/T_post/T_glob = 8 epochs at 1e-4/1e-5/1e-6) and sets
    ``paper_faithful`` on the config, which makes the TuneFP vacuity guard
    FAIL the run instead of continuing with pristine weights. That flag is
    the whole difference between "we ran something" and "we ran the paper's
    method": the gate-shaped override turns TuneFP into a single step.

    scale_dtype follows --scale-dtype (default float32): v5/v6 proved the
    hard-coded float16 overflows late-block s2 (>65,504 -> inf, 54/196
    layers). float16 remains available for BPW-tight artifacts whose
    post-polish scales are verified bounded; the storage cast now raises
    instead of silently emitting inf.
    """
    if getattr(args, "paper_faithful", False):
        stage_pre = StageHyper(1e-4, 4, 8)
        stage_latent = StageHyper(1e-5, 1, 8)
        stage_global = StageHyper(1e-6, 1, 8)
    else:
        stage_pre = StageHyper(1e-4, 4, 0)
        stage_latent = StageHyper(2e-3, 4, 3)
        stage_global = StageHyper(1e-3, 4, 4)
    return PTQConfig(
        seed=args.seed,
        rank=args.rank,
        compute_dtype=args.compute_dtype,
        scale_dtype=args.scale_dtype,
        admm_max_iter=args.max_admm_iter,
        solver=args.solver,
        stage_pre=stage_pre,
        stage_latent=stage_latent,
        stage_global=stage_global,
        verbose=True,
        progress_probe=progress_probe,
        # Root-cause ablation knobs (qk-root-cause-ablation): solver state, so
        # they ride the checkpoint fingerprint and a resumed fp run can never
        # resume as a binary run.
        factor_mode=args.factor_mode,
        ablation=args.ablation,
        layer_scope=args.layer_scope,
        # Paper-faithful claim + the bounded discrete flip pass
        # (qk-fix-tunefp-teacher / qk-fix-ste-sign-adaptation): both are
        # solver state and both are recorded in the run report.
        paper_faithful=bool(getattr(args, "paper_faithful", False)),
        ste_flip_pass_budget=int(getattr(args, "ste_flip_pass_budget", 0)),
    )


def _reportable_stage_payload(value: Any) -> Any:
    """Strip non-report material (tensors, modules) from a stage record.

    Tensors are not report material — the numbers they carried are, and those
    are recorded separately — so they collapse to a type marker. Everything
    else passes through for ``dump_report``'s own sanitizer.
    """
    if isinstance(value, dict):
        return {k: _reportable_stage_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_reportable_stage_payload(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def _stage_recorder(sink: list) -> Any:
    """The pipeline's stage recorder, appending JSON-safe records to ``sink``."""

    def _record(event: str, **info: Any) -> None:
        sink.append((event, _reportable_stage_payload(info)))

    return _record


def _run_identity(
    args: argparse.Namespace,
    prov: dict,
    *,
    tune_seq_len: int,
    calibration_rows: int,
    eval_rows: int,
    ppl_baseline_bf16: Any,
    evaluation: dict,
) -> dict:
    """Report identity fields shared by every exit path.

    Built once so a failed run is as self-describing as a finished one — the
    probe-abort, stage-failure, and completion reports all carry the same
    identity. Lives at module level (not nested in ``main``) because the
    wiring pin reads this literal from source, and because a nested helper's
    own ``return`` would red the one-return pin on ``main``.
    """
    return {
        "model": args.model,
        "rank": args.rank,
        "solver": prov["solver"],
        "scale_dtype": prov["scale_dtype"],
        "factor_mode": args.factor_mode,
        "ablation": args.ablation,
        "layer_scope": args.layer_scope,
        "seq_len": args.seq_len,
        "tune_seq_len": tune_seq_len,
        "calibration_rows": calibration_rows,
        "eval_rows": eval_rows,
        "max_admm_iter": args.max_admm_iter,
        "seed": args.seed,
        "ppl_baseline_bf16": ppl_baseline_bf16,
        "paper_faithful": bool(args.paper_faithful),
        "ste_flip_pass_budget": int(args.ste_flip_pass_budget),
        # Geometry deviations from [NQ]'s protocol (sliced tune windows,
        # non-protocol row counts): a stage map that is complete while the
        # geometry is not the protocol's would make "paper-faithful" a green
        # label over a deviant artifact.
        "runtime_deviations": geometry_deviations(args.seq_len, tune_seq_len, calibration_rows),
        # The evaluation verdict, on every exit path: what the pedestal covered
        # and which harness produced it. Sits beside the method verdict rather
        # than inside it — see `evaluation_record`.
        "evaluation": evaluation,
        "regime": {
            "baseline_eval": "bfloat16 (checkpoint native; papers' pedestal)",
            "solver": "float64, per-layer just-in-time (sign margins + parity reference)",
            "kd_and_ste": "model dtype (the papers' distillation regime)",
            "artifact": prov["regime_artifact"],
        },
    }


def export_model_directory(
    out_dir: Path,
    qm: QuantizedModel,
    embedding_f64: np.ndarray,
    model_id: str,
    eos_token_id: int,
    bos_token_id: int,
    tokenizer,
    config_f64: dict,
    geometry: dict,
    inventory: list[str],
    norm_weights: np.ndarray,
    lm_head_f64: np.ndarray | None,
    tied: bool = True,
) -> None:
    """Write the qk-server acquisition directory (loader-validated layout).

    Layer streams land at layers/<dotted/name/path>.qpk1; the fp32 sidecar
    is the embedding row-major [vocab, d_model] LE float32; a tied head
    reuses the embedding file via weight_tying = "lm_head=e1@fp32", an
    UNTIED head exports its own matrix (weight_tying =
    "lm_head=head.f32.bin@fp32"). config.toml declares the norm inventory
    (norm_slots) the sidecar is laid out by.
    """
    layer_files: dict[str, str] = {}
    for name, blob in sorted(qm.serialize().items()):
        rel = "layers/" + name.replace(".", "/") + ".qpk1"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        layer_files[name] = rel

    if tied:
        assert lm_head_f64 is None, "tied export must not pass a head matrix"
    emb = np.ascontiguousarray(embedding_f64, dtype=np.float32)
    (out_dir / "e1.f32.bin").write_bytes(emb.tobytes(order="C"))
    if not tied:
        assert lm_head_f64 is not None, "untied export requires the head matrix"
        head = np.ascontiguousarray(lm_head_f64, dtype=np.float32)
        (out_dir / "head.f32.bin").write_bytes(head.tobytes(order="C"))

    (out_dir / "config.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f"vocab_size = {embedding_f64.shape[0]}\n"
        f"d_model = {embedding_f64.shape[1]}\n"
        f"eos_token_id = {eos_token_id}\n"
        f"bos_token_id = {bos_token_id}\n"
        + (
            'weight_tying = "lm_head=e1@fp32"\n'
            if tied
            else 'weight_tying = "lm_head=head.f32.bin@fp32"\n'
        )
        + f"n_layers = {geometry['n_layers']}\n"
        f"n_heads = {geometry['n_heads']}\n"
        f"n_kv_heads = {geometry['n_kv_heads']}\n"
        f"head_dim = {geometry['head_dim']}\n"
        f"rms_norm_eps = {geometry['rms_norm_eps']}\n"
        f"rope_theta = {geometry['rope_theta']}\n"
        "norm_slots = [" + ", ".join(f'"{s}"' for s in inventory) + "]\n"
    )

    # Norm sidecar: fp32 LE — per block, each declared inventory slot's
    # vector in order, then the model-final norm. The loader requires the
    # inventory-derived count (qk-attention N-1); asserted below so a
    # miscount fails here, not at server load.
    norms = np.ascontiguousarray(norm_weights, dtype=np.float32)
    (out_dir / "norms.bin").write_bytes(norms.tobytes(order="C"))

    manifest_layers = ",\n".join(f'  "{n}"' for n in layer_files)
    manifest_layer_files = "\n".join(f'"{n}" = "{p}"' for n, p in layer_files.items())
    (out_dir / "manifest.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f'model_id = "{model_id}"\n'
        f"layers = [\n{manifest_layers},\n]\n"
        f"\n[layer_files]\n{manifest_layer_files}\n"
        f"\n[sidecar]\n"
        f'embedding = "e1.f32.bin"\n'
        + ('lm_head = "e1.f32.bin"\n' if tied else 'lm_head = "head.f32.bin"\n')
        + 'norms = "norms.bin"\n'
    )
    # Serialize the already-loaded tokenizer straight into the export dir
    # (tokenizer.json + config files). No hub lookup: offline containers and
    # local snapshot paths both work, and the in-memory object is by
    # definition the exact tokenizer the pipeline used.
    tokenizer.save_pretrained(out_dir)

    # Self-check the loader-facing contract (cheap, byte-level — this is the
    # tripwire that catches a half-updated export before 30 GPU-minutes are
    # spent): layer stream header parses, and the norm sidecar carries
    # exactly the inventory-derived float count (qk-attention N-1.0).
    first = sorted(layer_files.values())[0]
    header = struct.unpack("<4sIIIII", (out_dir / first).read_bytes()[:24])
    assert header[0] == b"QPK1" and header[1] == 1, header
    # d_model as the loader computes it: the embedding column count (the
    # GQA rule n_kv*head_dim == d_model is enforced server-side; heads/width
    # need not equal it per-axis, so derive the count from the same place).
    d_model = embedding_f64.shape[1]
    width = {"d_model": d_model, "head_dim": geometry["head_dim"]}
    per_block = sum(width[_NORM_SLOT_WIDTH[s]] for s in inventory)
    expected_norm_floats = per_block * geometry["n_layers"] + d_model
    assert norms.size == expected_norm_floats, (
        f"norms.bin has {norms.size} floats; inventory {inventory} at "
        f"d_model {d_model}, head_dim {geometry['head_dim']} requires "
        f"{per_block} x {geometry['n_layers']} + {d_model} = {expected_norm_floats}"
    )
    assert (out_dir / "norms.bin").stat().st_size == expected_norm_floats * 4


def _export_geometry(causal) -> dict:
    """Attention geometry + rope theta off the live HF config.

    This transformers generation folds theta into ``rope_parameters``
    (older versions carried a flat ``rope_theta`` attribute).
    """
    hcfg = causal.config
    return {
        "n_layers": hcfg.num_hidden_layers,
        "n_heads": hcfg.num_attention_heads,
        "n_kv_heads": hcfg.num_key_value_heads,
        "head_dim": getattr(hcfg, "head_dim", None) or hcfg.hidden_size // hcfg.num_attention_heads,
        "rms_norm_eps": hcfg.rms_norm_eps,
        "rope_theta": float(
            getattr(hcfg, "rope_theta", None)
            or (hcfg.rope_parameters or {}).get("rope_theta", 1_000_000.0)
        ),
    }


def _export_norm_inventory(causal) -> list[str]:
    """Declared per-block norm inventory (qk-attention N-1.0), read off the
    live model's modules — a family lookup table would silently mislabel a
    lookalike config; presence of the module is the truth. Order is the
    sidecar's per-block layout order and must match the engine's slot
    semantics (qk-attention E-2.0)."""
    layer = causal.model.layers[0]
    inventory = ["input", "post_attention"]
    attn = layer.self_attn
    if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
        inventory += ["q_norm", "k_norm"]
    if hasattr(layer, "post_feedforward_layernorm"):
        inventory += ["post_feedforward"]
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "final_norm"):
        raise NotImplementedError(
            "post-feedforward norm module shape not in the inventory "
            "vocabulary; extend NormSlot + this resolver together"
        )
    return inventory


_NORM_SLOT_MODULES = {
    "input": lambda layer: layer.input_layernorm.weight,
    "post_attention": lambda layer: layer.post_attention_layernorm.weight,
    "q_norm": lambda layer: layer.self_attn.q_norm.weight,
    "k_norm": lambda layer: layer.self_attn.k_norm.weight,
    "post_feedforward": lambda layer: layer.post_feedforward_layernorm.weight,
}

_NORM_SLOT_WIDTH = {
    "input": "d_model",
    "post_attention": "d_model",
    "post_feedforward": "d_model",
    "q_norm": "head_dim",
    "k_norm": "head_dim",
}


def _export_norm_weights(causal, inventory) -> np.ndarray:
    """Per-block norm vectors in declared inventory order, then the
    trunk-final norm — the sidecar layout (qk-attention N-1.0). q_norm /
    k_norm ride head_dim-wide vectors (one per projection, broadcast
    across heads)."""
    parts = []
    for layer in causal.model.layers:
        for slot in inventory:
            w = _NORM_SLOT_MODULES[slot](layer)
            parts.append(w.detach().to(torch.float32).reshape(-1))
    parts.append(causal.model.norm.weight.detach().to(torch.float32))
    return torch.cat(parts).cpu().numpy()


def main() -> int:
    # ---- crash-safe breadcrumbs ------------------------------------
    # SIGKILL can't flush our stdout, so a watchdog thread samples the
    # shared cgroup's memory + this process's RSS and appends stage
    # markers to a plain file. Whatever kills us, the tail of this file
    # says WHERE we were and HOW BIG we were.
    breadcrumb_path = Path(
        os.getenv(
            "QUANTIBBLER_BREADCRUMBS",
            str(Path(os.environ["QUANTIBBLER_KD_TMPDIR"]) / "quantibbler-breadcrumbs.log"),
        )
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
        "--tune-seq-len",
        type=_positive_int,
        default=None,
        help="per-block reconstruction window length: the tune stage's "
        "memory bound. Unset means --seq-len (the pre-flag behavior). "
        "--seq-len stays the EVALUATION geometry, so this never changes "
        "what the reported perplexity is measured over",
    )
    parser.add_argument(
        "--calibration-seqs",
        type=int,
        default=16,
        help="16 batches of 8 sequences = 128 calibration rows",
    )
    parser.add_argument("--eval-seqs", type=int, default=5, help="5 batches of 8 = 40 eval rows")
    parser.add_argument(
        "--probe-interval",
        type=_positive_int,
        default=1,
        help="run the per-block progress probe every N blocks from depth 2 "
        "(default 1 = every block boundary); non-positive is a parse error",
    )
    parser.add_argument(
        "--probe-batches",
        type=_positive_int,
        default=2,
        help="eval batches in the probe slice, taken as the head of the "
        "held-out set (default 2); non-positive is a parse error",
    )
    parser.add_argument("--max-admm-iter", type=int, default=400)
    parser.add_argument(
        "--solver",
        choices=("admm", "svd"),
        default="admm",
        help="warm-start source: 400-iter LB-ADMM or plain truncated SVD "
        "(ablation: within 0.003 cos, orders faster at high rank)",
    )
    parser.add_argument(
        "--factor-mode",
        choices=("binary", "fp"),
        default="binary",
        help="root-cause ablation: `fp` removes the binary constraint entirely "
        "(continuous latents; no sign, no STE, no bit-flip polish). This is a "
        "MEASUREMENT — it packs nothing and never exports an artifact",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "untuned-signs"),
        default="none",
        help="binary ablation: export sign() of the balance latents with the "
        "Eq. 8 scale fold and ZERO refinement (no polish, no TuneLatentSTE) — "
        "it separates 'the bits cannot hold it' from 'the optimizer is not "
        "finding it'",
    )
    parser.add_argument(
        "--layer-scope",
        choices=("all", "self_attn", "mlp"),
        default="all",
        help="factorize only the linears whose module path contains the named "
        "component; every other linear stays full precision",
    )
    parser.add_argument(
        "--scale-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="qpk1 scale storage dtype (float16 overflows to inf when any "
        "scale exceeds 65,504 — the cast now raises instead of shipping it)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts") / "ptq-checkpoints",
        help="per-block solve checkpoints (fingerprint-guarded); enables "
        "crash resume of the dominant-cost stage",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="disable solve checkpointing entirely",
    )
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help="claim paper fidelity: [NQ] App. C stage budgets verbatim "
        "(T_pre/T_post/T_glob = 8 epochs at 1e-4/1e-5/1e-6), and any TuneFP "
        "vacuity event FAILS the run instead of continuing with pristine "
        "weights (qk-fix-tunefp-teacher A3). The report's stage_map names "
        "every deviation that remains",
    )
    parser.add_argument(
        "--ste-flip-pass-budget",
        type=int,
        default=0,
        help="bounded discrete sign-flip proposals per factor per linear "
        "(qk-fix-ste-sign-adaptation fallback 3). 0 disables the pass; any "
        "nonzero value is an EXPLICIT DEVIATION from the paper and is "
        "recorded as such in the run report. INERT under --paper-faithful: "
        "the paper's path is sign+pack on whatever signs STE produced "
        "(Algorithm 1 lines 20-23), so the pass never fires and the record "
        "carries `declared: paper_ste` with the measured crossings",
    )
    parser.add_argument(
        "--compute-dtype",
        default="float64",
        choices=("float64", "float32"),
        help="solver compute dtype for the per-layer factorization (consumed by "
        "lb_admm and block_recon). float64 is the default and the only value a "
        "faithful run gets for free; float32 is an opt-in accelerator -- ~2x on "
        "the BLAS-bound ADMM solve -- whose Cholesky stability on real weights is "
        "characterized by the layers.0 A/B, not assumed. It is recorded in "
        "launch.env and in every solver-probe row, so an fp32 artifact is "
        "identifiable from its own directory rather than from someone's "
        "scrollback: two arms whose rows are indistinguishable are the same "
        "provenance gap as an unlabelled rho_end.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="where the torch stages run (capture, TuneFP, STE, eval). 'auto' takes "
        "the accelerator when torch can see one and falls back to CPU otherwise, so "
        "a CPU run is unchanged and a GPU run is one flag. The per-linear SOLVER "
        "stays NumPy/CPU either way: its weights already hand off via .cpu().numpy() "
        "and back, roughly 200 crossings of small matrices across the whole run, "
        "which costs seconds rather than hours -- so moving the autograd stages does "
        "not require porting the solver. Timed on the CPU path: the torch stages are "
        "about 69%% of wall, which is what this moves.",
    )
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
    # Resolve ONCE, from the same predicate torch uses: a second source of truth
    # for "which device" can disagree with where the parameters actually are, and
    # the mismatch then surfaces frames away from its cause.
    _device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    causal.to(_device)
    os.environ["QUANTIBBLER_DEVICE"] = _device
    print(
        f"      device={_device} (requested {args.device}; "
        f"torch.cuda.is_available()={torch.cuda.is_available()})",
        flush=True,
    )
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
    held_out, eval_selection = _tokenize_blocks(
        tokenizer,
        load_wikitext_text(find_wikitext_shards("test")),
        args.seq_len,
        args.eval_seqs,
        args.seed + 1,
    )
    print(f"      calib {len(calib)} batches, eval {len(held_out)} batches")
    # The evaluation verdict (qk-fix-tunefp-teacher: "the eval pedestal is the
    # one geometry the label can't see"). `eval_rows` records the numerator;
    # this record states what it is a numerator OF, so the artifact can prove
    # its own coverage and the evaluator is named rather than inferred. It is
    # recorded beside the method verdict, never folded into it: comparability
    # is a harness fact, method fidelity is a protocol fact.
    evaluation = evaluation_record(
        evaluator=EVAL_EVALUATOR,
        source_tokens=eval_selection["source_tokens"],
        seq_len=args.seq_len,
        selected_blocks=len(eval_selection["picked_index"]),
        batch=BATCH,
        selection_seed=args.seed + 1,
        selection_fingerprint=hashlib.sha256(
            ",".join(str(i) for i in eval_selection["picked_index"]).encode()
        ).hexdigest()[:16],
    )
    print(
        f"      eval coverage: {evaluation['selected_blocks']}/"
        f"{evaluation['full_blocks']} blocks, "
        f"{evaluation['coverage_of_source_tokens']:.3%} of source tokens "
        f"(paper_table_comparable={evaluation['paper_table_comparable']})",
        flush=True,
    )
    # PREFLIGHT, before any quantization work: a faithful claim on this
    # evaluator must use its maximal local window. Anything smaller is a
    # sample presented as the split, and hours of PTQ should not be spent to
    # discover that at report time. Raised as a ValueError like the sizer's
    # overshoot refusal beside it (a usage error, not a run verdict): the
    # driver's single exit path belongs to the gate's verdict, so a config
    # refusal fails loudly here instead of implying a completed run.
    if args.paper_faithful and args.eval_seqs != evaluation["full_local_window_batches"]:
        raise ValueError(
            f"--paper-faithful requires the maximal local evaluation window: "
            f"--eval-seqs {evaluation['full_local_window_batches']} "
            f"(wikitext test at seq {args.seq_len} yields "
            f"{evaluation['full_blocks']} blocks = "
            f"{evaluation['full_local_window_batches']} batches of {BATCH}); "
            f"got {args.eval_seqs} ({evaluation['selected_blocks']} blocks, "
            f"{evaluation['coverage_of_source_tokens']:.3%} of source tokens). "
            "Diagnostics may use a smaller window and will record it; a "
            "faithful claim may not."
        )

    # --- independent tune geometry (qk-nq-recreation-bar) ----------------
    # `--seq-len` is the EVALUATION protocol (WT2 at 2048 for the bar) and
    # `--tune-seq-len` bounds the per-block reconstruction windows. The
    # windows are SLICED from the same calibration rows rather than
    # re-tokenized at the shorter length: sampling 512-token rows would
    # silently drop three quarters of [NQ]'s 128 x 2048 calibration tokens,
    # which changes the method, not just the memory budget. Slicing bounds
    # the retained N^2 attention node - the thing that actually died at seq
    # 2048 - and keeps every calibration token. `calibration_rows` is
    # counted BEFORE the slice because it counts sequences, and slicing
    # adds none.
    tune_seq_len = args.tune_seq_len or args.seq_len
    calibration_sequences = len(calib) * BATCH
    if tune_seq_len != args.seq_len:
        calib = [window for batch in calib for window in batch.split(tune_seq_len, dim=1)]
        print(
            f"      tune geometry: eval --seq-len {args.seq_len} kept; "
            f"calib sliced to {len(calib)} windows of {tune_seq_len}",
            flush=True,
        )

    print("[3/6] bf16 baseline perplexity (paper-regime pedestal)", flush=True)
    ppl_base = perplexity_from_logits(lambda b: model(b), held_out)
    print(f"      ppl_baseline = {ppl_base:.4f}", flush=True)

    # --- per-block progress probe (qk-fail-fast-loop) -------------------
    # The run dies in minutes instead of after the full block sweep. The fp
    # baseline is measured ONCE, here, before quantization: the head slice of
    # the SAME held-out batches the end-of-run gate uses, in the same order,
    # under the same seed discipline — never a separately generated dataset.
    # Every depth's ratio therefore divides by the identical pedestal, so the
    # only difference between the quantized prefix and its baseline is
    # quantization damage.
    probe_slice = held_out[: args.probe_batches]
    fp_probe_ppl = perplexity_from_logits(lambda b: model(b), probe_slice)
    print(
        f"      probe: {len(probe_slice)}-batch slice of the held-out eval, "
        f"fp prefix baseline = {fp_probe_ppl:.4f}",
        flush=True,
    )
    probe_history: list[dict] = []

    def _probe(host_model: Any, depth: int, name: str) -> None:  # noqa: ANN001
        """Boundary check: measure the live prefix, record it, abort on collapse.

        Called by the pipeline after each block's write-back. ``--probe-interval``
        lives here, not in the pipeline: the pipeline contract stays "one call
        per processed block" with no interval knowledge. The abort threshold is
        the shared derived ``COLLAPSE_LINE`` via ``probe_aborts`` — there is no
        argument by which this call site could re-pin it.
        """
        if depth >= MIN_PROBE_DEPTH and not (depth - MIN_PROBE_DEPTH) % args.probe_interval:
            prefix_ppl = perplexity_from_logits(lambda b: host_model(b), probe_slice)
            ratio = probe_ratio(prefix_ppl, fp_probe_ppl)
            probe_history.append(
                {
                    "block": name,
                    "depth": depth,
                    "prefix_ppl": prefix_ppl,
                    "fp_prefix_ppl": fp_probe_ppl,
                    "ratio": ratio,
                }
            )
            print(
                f"      probe depth {depth:>2} ({name}): prefix_ppl={prefix_ppl:.4f} "
                f"fp={fp_probe_ppl:.4f} ratio={ratio!r} line={COLLAPSE_LINE}",
                flush=True,
            )
            if probe_aborts(depth, prefix_ppl, fp_probe_ppl):
                raise PrefixProbeAbort(
                    block=name,
                    depth=depth,
                    prefix_ppl=prefix_ppl,
                    fp_prefix_ppl=fp_probe_ppl,
                    ratio=ratio,
                )

    _stage("[4/6] PTQ: bf16 model, per-layer float64 solver (block recon + scale KD)")
    _bread("entering quantize_model: block recon first, then KD scale calibration")
    solver_t = time.time()
    aborted: PrefixProbeAbort | None = None
    stage_failed: VacuousTeacherError | None = None
    ppl_ratio: float | None = None
    # provenance is computed before the pipeline so the abort report can
    # carry the same identity fields as the completion report: an aborted
    # run must be as self-describing as a finished one.
    # The effective rho endpoint comes from QK_RHO_END or the config default --
    # resolved from the same helper the solver calls, so the report cannot
    # disagree with the run it describes.
    from quantibbler.ptq.config import PTQConfig as _PTQConfig
    from quantibbler.ptq.lb_admm import rho_end_provenance as _rho_prov

    _rho_end_value, _rho_end_source = _rho_prov(_PTQConfig().rho_end)
    prov = provenance(
        args.solver,
        args.scale_dtype,
        rho_end=_rho_end_value,
        rho_end_source=_rho_end_source,
    )

    # --- stage record (qk-fix-tunefp-teacher / qk-fix-ste-sign-adaptation) --
    # The pipeline reports every stage through this recorder; the JSON-safe
    # part of each record lands in the run report as `stages`, and
    # `stage_map` resolves it against [NQ] Algorithm 1 lines 0-28. Tensors are
    # dropped (they are not report material; the numbers they carried are).
    stage_events: list[tuple[str, dict]] = []
    _record = _stage_recorder(stage_events)
    # Identity fields shared by every report path, built once (see
    # `_run_identity`: the probe-abort and stage-failure reports must be as
    # self-describing as a completion report).
    identity = _run_identity(
        args,
        prov,
        tune_seq_len=tune_seq_len,
        calibration_rows=calibration_sequences,
        eval_rows=len(held_out) * BATCH,
        ppl_baseline_bf16=ppl_base,
        evaluation=evaluation,
    )

    try:
        qm = quantize_model(
            model,
            calib,
            quantized_config(args, progress_probe=_probe),
            checkpoint_dir=None if args.no_checkpoints else args.checkpoint_dir,
            recorder=_record,
        )
        _bread(f"quantize_model returned: {len(qm.layers)} layers")
        print(f"      solver wall: {time.time() - solver_t:.0f}s", flush=True)

        print("[5/6] quantized perplexity (checkpoint-native dtype, artifact written back)")
        host = model  # quantize_model wrote the artifact back into the host
        ppl_quant = perplexity_from_logits(lambda b: host(b), held_out)
        # A measurement has no packed artifact: BPW and qpk1 bytes are null,
        # never 0 — a consumer must not read "0.0000 bpp" as a compression win.
        measured_artifact = args.factor_mode != "fp"
        bpw = qm.effective_bpw() if measured_artifact else None
        total_qpk1 = sum(len(b) for b in qm.serialize().values()) if measured_artifact else None
        print(f"      ppl_quant      = {ppl_quant:.4f}")
        if measured_artifact:
            print(
                f"      effective BPW  = {bpw:.4f}  (qpk1 layer bytes: {total_qpk1:,})",
                flush=True,
            )
        else:
            print(
                "      effective BPW  = n/a — factor_mode='fp' is a measurement and packs nothing",
                flush=True,
            )

        # [6/6] export — D-5 driver backstop: behind extract_and_pack's
        # conversion-site refusal, an fp measurement is never handed to the
        # export path at all, so no empty manifest can stand in for an artifact.
        artifact_exported = False
        if measured_artifact:
            print("[6/6] exporting qk-server model directory ->", out_dir, flush=True)
            embedding_f64 = (
                causal.get_input_embeddings().weight.detach().to(torch.float64).cpu().numpy()
            )
            # Attention geometry + norm inventory + norm weights straight from the
            # live model (never hardcoded): the exported artifact must be
            # self-describing for the server's transformer forward (qk-attention
            # N-1/N-2/E-2.0). The inventory IS the layout contract — sidecar order
            # here must match the engine's slot semantics exactly. Head tying is
            # the MODEL's property, not an export choice (Qwen3 ships untied).
            tied = bool(getattr(causal.config, "tie_word_embeddings", False))
            lm_head_f64 = (
                causal.get_output_embeddings().weight.detach().to(torch.float64).cpu().numpy()
                if not tied
                else None
            )
            geometry = _export_geometry(causal)
            inventory = _export_norm_inventory(causal)
            norm_weights = _export_norm_weights(causal, inventory)

            export_model_directory(
                out_dir,
                qm,
                embedding_f64=embedding_f64,
                model_id=args.model.replace("/", "-"),
                eos_token_id=cfg.eos_token_id,
                bos_token_id=cfg.bos_token_id,
                tokenizer=tokenizer,
                config_f64={"dtype_note": prov["dtype_note"]},
                geometry=geometry,
                inventory=inventory,
                norm_weights=norm_weights,
                lm_head_f64=lm_head_f64,
                tied=tied,
            )
            artifact_exported = True
        else:
            print(
                "[6/6] export SKIPPED: factor_mode='fp' is a measurement — no artifact is written",
                flush=True,
            )
        ppl_ratio = ppl_quant / ppl_base
        status = verdict(ppl_ratio)
        # Phase 3 (Algorithm 1 lines 26-27) runs inside quantize_model; the
        # driver records its execution here so the stage map covers every
        # stage of lines 0-28 without threading a recorder into scale_calib.
        stage_events.append(
            (
                "tune_scales_kd",
                {"factor_mode": args.factor_mode, "artifact": measured_artifact},
            )
        )
        # The fidelity label is a COMPUTED verdict over the run's own record
        # (qk-fix-tunefp-teacher, scenario `label_flips_on_runtime_deviations` /
        # `label_flips_on_mechanism_deviations`): the flag AND an empty
        # deviation list AND no mechanism-class deviation. Every condition is
        # recorded beside it, so a reader can see which one denied the claim.
        faithful_label, fidelity_conditions = fidelity_verdict(
            prov["solver"],
            identity["runtime_deviations"],
            stage_events,
            requested=bool(args.paper_faithful),
        )
        report = {
            "model": args.model,
            "rank": args.rank,
            "solver": prov["solver"],
            "scale_dtype": prov["scale_dtype"],
            "seq_len": args.seq_len,
            "tune_seq_len": tune_seq_len,
            "calibration_rows": calibration_sequences,
            "eval_rows": len(held_out) * BATCH,
            "max_admm_iter": args.max_admm_iter,
            "seed": args.seed,
            # Ablation provenance (D-5): the knobs are ALWAYS named, so a
            # measurement can never be mistaken for a landing in the archive.
            "factor_mode": args.factor_mode,
            "ablation": args.ablation,
            "layer_scope": args.layer_scope,
            "ppl_baseline_bf16": ppl_base,
            "ppl_quant_native": ppl_quant,
            "ppl_ratio": ppl_ratio,
            "effective_bpw": bpw,
            "qpk1_layer_bytes": total_qpk1,
            "regime": {
                "baseline_eval": "bfloat16 (checkpoint native; papers' pedestal)",
                "solver": "float64, per-layer just-in-time (sign margins + parity reference)",
                "kd_and_ste": "model dtype (the papers' distillation regime)",
                "artifact": prov["regime_artifact"],
            },
            # The probe trajectory, recorded through the same dump_report path
            # as every other report value: on a run that survives, this IS the
            # probe's calibration data (where a healthy depth profile sits
            # relative to the abort line) instead of an unauditable near-miss.
            "probe": {
                "interval": args.probe_interval,
                "batches": args.probe_batches,
                "fp_prefix_ppl": fp_probe_ppl,
                "history": probe_history,
            },
            "probe_abort": None,  # no abort: the key exists on both paths
            "artifact_exported": artifact_exported,
            # Stage record + Algorithm 1 map (qk-fix-tunefp-teacher): every
            # executed stage resolved against lines 0-28 with its deviations
            # named, so "paper-faithful" is auditable from the report alone.
            # `paper_faithful` is the COMPUTED verdict (flag AND no deviation),
            # with its condition record beside it.
            "paper_faithful": faithful_label,
            "fidelity_conditions": fidelity_conditions,
            # Beside the method verdict, never inside it: evaluation
            # comparability is a harness fact (coverage, windowing, evaluator),
            # not a property of the quantization method or calibration
            # protocol. See `qk_report.evaluation_record`.
            "evaluation": identity["evaluation"],
            "ste_flip_pass_budget": int(args.ste_flip_pass_budget),
            "runtime_deviations": identity["runtime_deviations"],
            "stages": [{"event": event, **info} for event, info in stage_events],
            "stage_map": stage_map(stage_events),
            "status": status,
            "wall_seconds": round(time.time() - t0, 1),
        }

    except PrefixProbeAbort as exc:
        aborted = exc
        print(
            f"\n      PROBE ABORT at depth {exc.depth} ({exc.block}): "
            f"prefix_ppl={exc.prefix_ppl!r} fp_prefix_ppl={exc.fp_prefix_ppl!r} "
            f"ratio={exc.ratio!r} line={exc.line}",
            flush=True,
        )
        # Built by qk_report.abort_report — the catch→report→status logic lives
        # in a torch-free module so it has lean suite coverage instead of
        # resting on one manual real-model run. The builder emits the same key
        # set as the completion report (unmeasured end-of-run numerics as null,
        # artifact_exported false, status failed), so strict consumers never
        # KeyError on either shape.
        report = abort_report(
            exc,
            {
                **identity,
                "probe": {
                    "interval": args.probe_interval,
                    "batches": args.probe_batches,
                    "fp_prefix_ppl": fp_probe_ppl,
                },
                "stages": [{"event": event, **info} for event, info in stage_events],
                "stage_map": stage_map(stage_events),
                "wall_seconds": round(time.time() - t0, 1),
            },
            probe_history,
        )
    except VacuousTeacherError as exc:
        # Fail-closed stage gate (qk-fix-tunefp-teacher A3): a run that
        # CLAIMS paper fidelity must not continue on a TuneFP pass that
        # executed nothing. The report is written through the same
        # dump_report path as every other shape, with `paper_faithful: false`
        # and the two norms on record — a failed run leaves forensics, never
        # a traceback and no evidence.
        stage_failed = exc
        print(
            f"\n      TUNEFP VACUOUS TEACHER at block {exc.block!r} "
            f"({exc.reason}): grad_norm={exc.gradient_norm!r} "
            f"initial_residual={exc.initial_residual_norm!r} — a run claiming "
            "paper fidelity fails here",
            flush=True,
        )
        failure_identity = dict(identity)
        failure_identity["probe"] = {
            "interval": args.probe_interval,
            "batches": args.probe_batches,
            "fp_prefix_ppl": fp_probe_ppl,
            "history": probe_history,
        }
        failure_identity["probe_abort"] = None
        failure_identity["wall_seconds"] = round(time.time() - t0, 1)
        report = stage_failure_report(
            exc,
            failure_identity,
            [{"event": event, **info} for event, info in stage_events],
        )
    payload, faithful = dump_report(report)
    if not faithful or aborted is not None or stage_failed is not None:
        # Three routes to a failed process, and all must fail: the report
        # could not be serialized as measured (an unrecognized value type
        # reached the encoder), the progress probe aborted the run, or a
        # fail-closed stage gate tripped — each before the end-of-run gate
        # produced a verdict. A report and an exit code that disagree is
        # exactly the drift this gate exists to stop; both failure reports
        # are written with status FAILED, so the two agree by construction.
        status = FAILED
    (out_dir / "quantize_report.json").write_text(payload)
    print(payload)
    print(
        f"\nE2E quality verdict: {status.upper()}  "
        f"(ppl_ratio={ppl_ratio!r}; ok <= {OK_LINE} < degraded <= "
        f"{COLLAPSE_LINE} < failed)"
    )
    if aborted is None:
        print("\nserver acquisition check (linear-stack engine: load only, no generation):")
        print(f"  cargo run --release -p quantibbler-server -- serve --model-dir {out_dir}")
    elif aborted is not None:
        print("\nno artifact exported: the probe aborted the run, so there is")
        print("nothing to serve — the report above is the whole forensic trace.")
    else:
        print(
            "\nno artifact exported: a fail-closed stage gate tripped "
            f"({stage_failed.stage} at block {stage_failed.block!r}), so the run "
            "stopped before the end-of-run gate; the report above carries the "
            "stage record and the Algorithm 1 map."
        )
    # Fail closed: on a completed run the artifact and report above are written
    # either way, so a failed run leaves a forensically complete artifact and a
    # nonzero exit; on an aborted or stage-failed run the report alone is the
    # forensic trace.
    return verdict_exit_code(status)


if __name__ == "__main__":
    raise SystemExit(main())
