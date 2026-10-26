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

Usage (from the repo root) — the pipeline's deps are the product's own, so a
plain sync is enough (no extras):
    uv sync
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
import subprocess
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


def _scratch_root() -> str:
    """The host scratch root: real disk where it exists, platform temp otherwise.

    Named once because the KD teacher-logit cache AND the per-run checkpoint
    directory both need to be off a small tmpfs: two expressions of "the scratch
    root" is how they drift apart.
    """
    return str(_SANDBOX_SCRATCH) if _SANDBOX_SCRATCH.is_dir() else tempfile.gettempdir()


os.environ.setdefault("QUANTIBBLER_KD_TMPDIR", _scratch_root())
# Bounded native allocations (hundreds of MB drift otherwise) and bounded
# BLAS/torch thread pools (each thread's stacks and scratch count against
# the cgroup this sandbox shares).
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# numpy's scipy-openblas is PTHREADS-backed: OMP/MKL pins do NOT size its
# pool, which defaults to visible-CPU count. Measured: unpinned 235.5s vs
# pinned-to-optimum 3.98s — a 59x thread-thrash artifact, not algorithm
# scaling. Pin it with the others.
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
from quantibbler.ptq.checkpoints import append_stage_event
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
from quantibbler.qpk1 import REPRESENTATION_VERSION as qpk1_version
from quantibbler.remainder_dtype import REMAINDER_DTYPES
from quantibbler.remainder_dtype import encode as encode_remainder
from quantibbler.remainder_dtype import itemsize as remainder_itemsize
from quantibbler.serialization import LaunchAttributionRecord


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
        # Scoped default device for the stock forward. The arguments are moved
        # above, but the stock HF forward also CREATES tensors -- the rotary
        # arange / inv_freq class of temporaries -- and in this transformers
        # version those are born on the default device, which is CPU, giving
        # "found at least two devices, cuda:0 and cpu" inside the model rather
        # than at any call site we control. Creating them on the weights' device
        # is the version-agnostic fix; naming cache_position/position_ids would
        # only cover whichever temporaries this version happens to expose.
        # Scoped, not global: torch.set_default_device would also retarget the
        # NumPy solver stage, which must stay CPU. On CPU this context is a no-op.
        with torch.device(_dev):
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
        # `batch` is tokenized on the CPU; the logits come back on whatever
        # device the model ran on. Move the batch to match instead of assuming
        # they agree -- the target, and any mask derived from it, must be on the
        # logits' device. A no-op on CPU.
        batch = batch.to(logits.device)
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


def packed_chunks(sequences: int, knob: str) -> int:
    """``sequences`` -> the packed batches the loader wants, fail-closed.

    The data knobs count SEQUENCES ([NQ] Sec 3.4/App. C: 128 calibration
    sequences), but ``_tokenize_blocks`` packs ``BATCH`` rows per chunk, so its
    ``n_chunks`` parameter counts batches. Converting at the knob boundary — in
    one place, before the counter that reports the same unit — keeps the
    packed-batch structure an implementation detail of the tokenizer.

    Refuses a count that is not a whole number of batches rather than silently
    rounding it: a knob whose value is quietly changed is the defect this
    function exists to remove.
    """
    if sequences % BATCH:
        raise ValueError(
            f"{knob} counts sequences: {sequences} is not divisible by the "
            f"packed batch size {BATCH}"
        )
    return sequences // BATCH


def eval_window_sequences(evaluation: dict) -> int:
    """The maximal local eval window in the KNOB's unit (sequences).

    ``--eval-seqs`` counts sequences ({packed_chunks}'s contract), while
    ``evaluation_record`` keeps ``full_local_window_batches`` — a harness fact
    of the sizer, pinned in batches by the report's own tests. The
    paper-faithful gate is where the two units meet, so the conversion lives
    here in one named place: a gate that compared a sequence count against a
    batch count would pass a window eight times too small and call it maximal.
    """
    return evaluation["full_local_window_batches"] * BATCH


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
        # Root-cause ablation knobs: solver state, so they ride the checkpoint
        # fingerprint and a resumed fp run can never resume as a binary run.
        factor_mode=args.factor_mode,
        ablation=args.ablation,
        layer_scope=args.layer_scope,
        # Paper-faithful claim + the bounded discrete flip pass: both are
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


def _stage_recorder(sink: list, log_dir: Path | None = None) -> Any:
    """The pipeline's stage recorder, appending JSON-safe records to ``sink``.

    When ``log_dir`` is given (checkpointing enabled for this launch) the SAME
    record is also appended to the stage journal beside the checkpoints,
    once per event, as the event fires: a run that dies in the pipeline tail
    after solving every block can then be resumed with a record that still
    attests the stages it executed. Disabled checkpointing means no journal —
    the in-process record is the only one, exactly as before it existed.
    """

    def _record(event: str, **info: Any) -> None:
        payload = _reportable_stage_payload(info)
        sink.append((event, payload))
        if log_dir is not None:
            append_stage_event(log_dir, event, payload)

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
        # The artifact's OTHER storage dtype. A report naming only the scale
        # dtype under-identifies the bytes: two artifacts of the same run can
        # hold fp32 or bf16 remainders, and a reader cannot tell them apart.
        "remainder_dtype": args.remainder_dtype,
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
    geometry: dict,
    inventory: list[str],
    norm_weights: np.ndarray,
    lm_head_f64: np.ndarray | None,
    tied: bool = True,
    remainder_dtype: str = "fp32",
) -> None:
    """Write the qk-server acquisition directory (loader-validated layout).

    Layer streams land at layers/<dotted/name/path>.qpk1; the remainder
    sidecar is the embedding row-major [vocab, d_model] LE at the declared
    remainder dtype; a tied head reuses the embedding file via weight_tying =
    "lm_head=e1@<dtype>", an UNTIED head exports its own matrix (weight_tying
    = "lm_head=head.<suffix>.bin@<dtype>"). config.toml declares the norm
    inventory (norm_slots) the sidecar is laid out by.

    ``remainder_dtype`` is the storage dtype of every remainder sidecar
    (embedding, head, norms) and is DECLARED through the manifest's
    ``[sidecar] dtype`` key, because the loader reads the declaration and never
    guesses a dtype from a byte length. The ``fp32`` default is the pre-option
    remainder FILES byte-for-byte (same names, same bytes) — the manifest gains
    the declaration line; ``bf16`` is opt-in and lossy (8 mantissa bits).
    """
    if remainder_dtype not in REMAINDER_DTYPES:
        raise ValueError(
            f"remainder_dtype must be one of {REMAINDER_DTYPES}, got {remainder_dtype!r}"
        )
    # fp32 keeps the shipped spelling (.f32.bin / norms.bin); any other dtype
    # names itself in the file names too — a convention echo, never the
    # authority (the `[sidecar] dtype` key is).
    emb_name = "e1.f32.bin" if remainder_dtype == "fp32" else "e1.bf16.bin"
    head_name = "head.f32.bin" if remainder_dtype == "fp32" else "head.bf16.bin"
    norms_name = "norms.bin" if remainder_dtype == "fp32" else "norms.bf16.bin"
    layer_files: dict[str, str] = {}
    for name, blob in sorted(qm.serialize().items()):
        rel = "layers/" + name.replace(".", "/") + ".qpk1"
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        layer_files[name] = rel

    if tied:
        assert lm_head_f64 is None, "tied export must not pass a head matrix"
    (out_dir / emb_name).write_bytes(encode_remainder(embedding_f64, remainder_dtype))
    if not tied:
        assert lm_head_f64 is not None, "untied export requires the head matrix"
        (out_dir / head_name).write_bytes(encode_remainder(lm_head_f64, remainder_dtype))

    (out_dir / "config.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f"vocab_size = {embedding_f64.shape[0]}\n"
        f"d_model = {embedding_f64.shape[1]}\n"
        f"eos_token_id = {eos_token_id}\n"
        f"bos_token_id = {bos_token_id}\n"
        # `e1` is the tied SENTINEL (it names the embedding sidecar, not a
        # file); an untied head names its own matrix. The dtype token is part
        # of the declaration the loader validates.
        + (
            f'weight_tying = "lm_head=e1@{remainder_dtype}"\n'
            if tied
            else f'weight_tying = "lm_head={head_name}@{remainder_dtype}"\n'
        )
        + f"n_layers = {geometry['n_layers']}\n"
        f"n_heads = {geometry['n_heads']}\n"
        f"n_kv_heads = {geometry['n_kv_heads']}\n"
        f"head_dim = {geometry['head_dim']}\n"
        f"rms_norm_eps = {geometry['rms_norm_eps']}\n"
        f"rope_theta = {geometry['rope_theta']}\n"
        "norm_slots = [" + ", ".join(f'"{s}"' for s in inventory) + "]\n"
    )

    # Norm sidecar at the declared dtype — per block, each declared inventory
    # slot's vector in order, then the model-final norm. The loader requires
    # the inventory-derived count; asserted below so a
    # miscount fails here, not at server load.
    (out_dir / norms_name).write_bytes(encode_remainder(norm_weights, remainder_dtype))

    manifest_layers = ",\n".join(f'  "{n}"' for n in layer_files)
    manifest_layer_files = "\n".join(f'"{n}" = "{p}"' for n, p in layer_files.items())
    (out_dir / "manifest.toml").write_text(
        f"# generated by tools/quantize_hf_model.py for {model_id}\n"
        f'model_id = "{model_id}"\n'
        f"layers = [\n{manifest_layers},\n]\n"
        f"\n[layer_files]\n{manifest_layer_files}\n"
        f"\n[sidecar]\n"
        # The dtype the three files below hold, declared ONCE for the set: the
        # loader reads THIS key (an absent key means fp32) and never infers a
        # width from a byte length.
        f'dtype = "{remainder_dtype}"\n'
        f'embedding = "{emb_name}"\n'
        + (f'lm_head = "{emb_name}"\n' if tied else f'lm_head = "{head_name}"\n')
        + f'norms = "{norms_name}"\n'
    )
    # Serialize the already-loaded tokenizer straight into the export dir
    # (tokenizer.json + config files). No hub lookup: offline containers and
    # local snapshot paths both work, and the in-memory object is by
    # definition the exact tokenizer the pipeline used.
    tokenizer.save_pretrained(out_dir)

    # Self-check the loader-facing contract (cheap, byte-level — this is the
    # tripwire that catches a half-updated export before 30 GPU-minutes are
    # spent): layer stream header parses, and the norm sidecar carries
    # exactly the inventory-derived float count.
    first = sorted(layer_files.values())[0]
    header = struct.unpack("<4sIIIII", (out_dir / first).read_bytes()[:24])
    # The exporter writes the CURRENT representation version (bias-carrying);
    # a stream that claims otherwise means the layer writer and this check
    # have drifted apart.
    assert header[0] == b"QPK1" and header[1] == qpk1_version, header
    # d_model as the loader computes it: the embedding column count (the
    # GQA rule n_kv*head_dim == d_model is enforced server-side; heads/width
    # need not equal it per-axis, so derive the count from the same place).
    d_model = embedding_f64.shape[1]
    width = {"d_model": d_model, "head_dim": geometry["head_dim"]}
    per_block = sum(width[_NORM_SLOT_WIDTH[s]] for s in inventory)
    expected_norm_floats = per_block * geometry["n_layers"] + d_model
    norm_count = np.asarray(norm_weights).size
    assert norm_count == expected_norm_floats, (
        f"{norms_name} has {norm_count} floats; inventory {inventory} at "
        f"d_model {d_model}, head_dim {geometry['head_dim']} requires "
        f"{per_block} x {geometry['n_layers']} + {d_model} = {expected_norm_floats}"
    )
    assert (out_dir / norms_name).stat().st_size == expected_norm_floats * remainder_itemsize(
        remainder_dtype
    ), f"{norms_name} does not carry {expected_norm_floats} @{remainder_dtype} elements"
    # Every declared entry must name a file the export actually wrote: a
    # declaration pointing at nothing is exactly the drift this tripwire is for.
    for key, name in (("embedding", emb_name), ("norms", norms_name)):
        assert (out_dir / name).is_file(), f"[sidecar] {key} declares {name}, which was not written"


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


def _resolve_rev() -> str:
    """The checkout revision this driver file belongs to, or literal 'unknown'.

    The launch line names the run's CODE identity, so the revision is read from
    the driver's OWN location rather than the cwd:
    a fire launched from elsewhere still belongs to the tree its code came from,
    and that is what a reader of a surviving trail needs. A copy placed outside
    any checkout (CI smoke harnesses, bare scripts) has no revision to name and
    MUST say so literally — never a crash, never a fabricated value.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


#: Environment variables that can carry the launch-owner stamp, in precedence
#: order. The label-named spellings are the SAME evidence under another
#: spelling, never a second contract: docker never shows a container its own
#: labels, so a direct `docker run` can only carry the owner as an environment
#: variable -- the labels alone are not visible from inside the container.
_LAUNCH_OWNER_VARS = ("QKB_LAUNCH_OWNER", "QKB_LAUNCH_LABEL_OWNER", "qkb_launch_owner")
_LAUNCH_CARD_VARS = ("QKB_LAUNCH_CARD", "QKB_LAUNCH_LABEL_CARD", "qkb_launch_card")


def _launch_owner() -> tuple[str, str]:
    """Who owns this run -- or a refusal, before anything is written.

    The launch line is the trail's FIRST line, so the owner it records has to be
    resolved before it is written: a run that cannot name its owner is refused
    here rather than recorded as an anonymous one. This lives outside ``main()``
    on purpose, like ``_resolve_device``: ``main()`` has exactly one exit, the
    verdict path, and that invariant is pinned (``_driver_wiring_findings``
    flags any ``raise SystemExit`` inside it) -- the refusal is a pre-flight, and
    it is named as one rather than folded into the pin's exception list.

    The owner is required; the owning card id is recorded as literal ``unset``
    when absent, because a run that legitimately has no board card (a CI smoke
    leg, a local diagnostic) still has an owner. The wrapper -- where a board
    fire is born -- requires both.
    """

    def first(names: tuple[str, ...]) -> str:
        for name in names:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    owner = first(_LAUNCH_OWNER_VARS)
    if not owner:
        raise SystemExit(
            "error: launch ownership missing -- refusing to start an unattributable "
            "run. Set QKB_LAUNCH_OWNER=<who is launching> (and QKB_LAUNCH_CARD="
            "<kanban card id> for a board fire). The label spellings "
            "QKB_LAUNCH_LABEL_OWNER/QKB_LAUNCH_LABEL_CARD/qkb_launch_owner/"
            "qkb_launch_card carry the same evidence. tools/run-ptq-container.sh "
            "refuses the same launch on the host and stamps both as container "
            "labels beside qkb_launch_id."
        )
    return owner, first(_LAUNCH_CARD_VARS) or "unset"


def _launch_record(owner: str, card: str, out_dir: Path) -> LaunchAttributionRecord:
    """The trail's first line, as the record that owns its field set.

    One construction site, so the launch line and the record cannot describe
    different runs: the process (pid), the exact argv, the code (the revision of
    the tree this file belongs to, ``unknown`` when unresolvable), the resolved
    output target, and the owner pair. ``at`` is the epoch the crumb line leads
    with — the stall watchdog keys liveness off the line's first field, so the
    record carries the timestamp the line opens with rather than a second clock
    reading that could disagree with it.
    """
    return LaunchAttributionRecord(
        at=time.time(),
        pid=os.getpid(),
        argv=" ".join(sys.argv[1:]),
        rev=_resolve_rev(),
        out=str(out_dir),
        owner=owner,
        card=card,
    )


def _resolve_device(requested: str) -> str:
    """Where the torch stages run -- or a refusal, before anything is loaded.

    'auto' means the accelerator or refuse. It deliberately does not fall back:
    `:latest` carries CUDA-wheel torch, so on this AMD host
    `torch.cuda.is_available()` is False and 'auto' resolved to cpu without a
    word -- a production fire spent 21 hours on one socket while its own launch
    record printed `device=cpu`. An explicit 'cpu' is still honest and still
    available; arriving there by accident is not. Same rule as the solver
    backend: the non-accelerator path is reachable by explicit name only.

    This lives outside `main()` on purpose. `main()` has exactly one exit, the
    verdict path, and that invariant is pinned; a pre-flight refusal is a
    different thing and is named as one rather than folded into the pin's
    exception list.
    """
    if requested == "auto" and not torch.cuda.is_available():
        raise SystemExit(
            "error: --device auto found no accelerator "
            "(torch.cuda.is_available()=False). This host is AMD/ROCm: the container "
            "needs GPU=1 and a rocm-tagged image "
            "(IMAGE=quantibbler-ptq:rocm-official), or pass --device cpu to ask for "
            "a CPU run explicitly. Refusing to run the autograd stages on CPU by "
            "default."
        )
    return "cuda" if requested == "auto" else requested


def _checkpoint_launcher_dir(checkpoint_dir: Path | None) -> Path:
    """The solve-checkpoint directory THIS launch will use.

    An explicitly passed directory is the resumable path and is used exactly as
    given. Without one the launch gets a FRESH per-run directory: a shared,
    stable, mutable default is the amplifier of a measured defect (two models
    launched with no flags collided in `artifacts/ptq-checkpoints`, and model B's
    resume then restored model A's block archives under an identical PTQConfig --
    qk-fp-model-identity), so the default removes the collision instead of
    reporting it. Resumption is opt-in: name the directory a launch printed.

    Per-run directories land under `_scratch_root()` for the same reason the KD
    cache does -- real disk where it exists, so block archives and fp-stream
    targets are not written into a small tmpfs.
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    return Path(tempfile.mkdtemp(prefix="ptq-checkpoints-", dir=_scratch_root()))


def main() -> int:
    # ---- crash-safe breadcrumbs ------------------------------------
    # SIGKILL can't flush our stdout, so a watchdog thread samples the
    # shared cgroup's memory + this process's RSS and appends stage
    # markers to a plain file. Whatever kills us, the tail of this file
    # says WHERE we were and HOW BIG we were.
    # The DEFAULTED trail is per-launch (`breadcrumbs-<pid>.log`) so two
    # concurrent sandbox runs write two files instead of interleaving into one;
    # an explicit QUANTIBBLER_BREADCRUMBS is honored EXACTLY, because the
    # container is single-run and that path is a contract (the wrapper's run
    # headers and the forensics read path point at that one file).
    breadcrumb_path = Path(
        os.getenv(
            "QUANTIBBLER_BREADCRUMBS",
            str(Path(os.environ["QUANTIBBLER_KD_TMPDIR"]) / f"breadcrumbs-{os.getpid()}.log"),
        )
    )

    def _bread(note: str, *, at: float | None = None) -> None:
        # ``at`` lets the launch record supply the epoch its own ``at`` field
        # already holds, so the line's leading field and the record cannot drift
        # apart across two clock readings. Every other crumb stamps itself.
        try:
            mem = 0
            try:
                with open("/sys/fs/cgroup/memory.current") as fh:
                    mem = int(fh.read())
            except OSError:
                pass
            stamp = time.time() if at is None else at
            with open(breadcrumb_path, "a") as fh:
                fh.write(f"{stamp:.0f} cgroup={mem / 2**30:.2f}G :: {note}\n")
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
        default=128,
        help="128 calibration SEQUENCES ([NQ] Sec 3.4/App. C), packed "
        f"{BATCH} rows per batch (16 batches); the count must be a multiple of {BATCH}",
    )
    parser.add_argument(
        "--eval-seqs",
        type=int,
        default=40,
        help=f"40 held-out evaluation SEQUENCES, packed {BATCH} rows per batch "
        f"({40 // BATCH} batches); the count must be a multiple of {BATCH}",
    )
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
        "--remainder-dtype",
        choices=REMAINDER_DTYPES,
        default="fp32",
        help="storage dtype of the exported artifact's remainder sidecars "
        "(embedding, lm_head, norms), declared in the manifest — the only thing "
        "the loader reads to size them. fp32 is the default and reproduces every "
        "pre-existing artifact byte-for-byte; bf16 halves those bytes and is "
        "lossy (8 mantissa bits)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="per-block solve checkpoints (fingerprint-guarded) and the ONE way "
        "to opt into crash resume of the dominant-cost stage: name the directory "
        "a previous launch printed. Default: a FRESH per-run directory — a shared "
        "stable default is how one model's block archives get restored into "
        "another model's solve",
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
        "the accelerator or REFUSES -- it never falls back to CPU, because a silent "
        "fallback is how a production fire ran 21 hours on one socket while its own "
        "launch record printed device=cpu and nobody read it. A CPU run is still "
        "available by explicit name: --device cpu. The per-linear SOLVER "
        "stays NumPy/CPU either way: its weights already hand off via .cpu().numpy() "
        "and back, roughly 200 crossings of small matrices across the whole run, "
        "which costs seconds rather than hours -- so moving the autograd stages does "
        "not require porting the solver. Timed on the CPU path: the torch stages are "
        "about 69%% of wall, which is what this moves.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "model-qpk1")
    args = parser.parse_args()

    # Ownership BEFORE anything is written: a run that cannot name its owner is
    # refused, and a refused run leaves no trail line claiming to be a launch.
    owner, card = _launch_owner()

    # The launch line is the trail's FIRST line and the only record a corpse is
    # guaranteed to have written: it names the process (pid), the exact argv, the
    # code (rev), the output target (out) and who owns the run (owner, card), so
    # a surviving trail self-attributes without kernel records or container
    # metadata. `out` is only known after the
    # parse, and nothing may reach the file before this line — the watchdog
    # thread therefore starts HERE, below it, rather than above it, and the
    # memory.events crumb follows it.
    #
    # Every identity field is read off the launch record, so the line and the
    # record cannot describe different runs, and `at` is the record's own epoch:
    # the line's leading field is the timestamp the record attests.
    out_dir: Path = args.out.resolve()
    threading.Thread(target=_watchdog, daemon=True).start()
    launch = _launch_record(owner, card, out_dir)
    _bread(
        f"start pid={launch.pid} argv={launch.argv} rev={launch.rev} "
        f"out={launch.out} owner={launch.owner} card={launch.card}",
        at=launch.at,
    )
    try:
        with open("/sys/fs/cgroup/memory.events") as fh:
            _bread("memory.events@start: " + " ".join(fh.read().split()))
    except OSError:
        pass

    _device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/6] loading {args.model} (local cache, offline)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    causal = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    # Resolve ONCE, from the same predicate torch uses: a second source of truth
    # for "which device" can disagree with where the parameters actually are, and
    # the mismatch then surfaces frames away from its cause.
    causal.to(_device)
    from quantibbler.ptq.device_audit import install_device_auditor

    if install_device_auditor():
        print("      device auditor ON (fail-closed; names the call site)", flush=True)
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
        packed_chunks(args.calibration_seqs, "--calibration-seqs"),
        args.seed,
    )
    held_out, eval_selection = _tokenize_blocks(
        tokenizer,
        load_wikitext_text(find_wikitext_shards("test")),
        args.seq_len,
        packed_chunks(args.eval_seqs, "--eval-seqs"),
        args.seed + 1,
    )
    print(f"      calib {len(calib)} batches, eval {len(held_out)} batches")
    # The evaluation verdict ("the eval pedestal is the one geometry the label
    # can't see"). `eval_rows` records the numerator;
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
    # The knob counts sequences; `evaluation` records the window in batches (a
    # harness fact, kept). The comparison is made in the knob's unit, so the
    # gate's demanded value and the recorded one are the same kind of number.
    full_window_sequences = eval_window_sequences(evaluation)
    if args.paper_faithful and args.eval_seqs != full_window_sequences:
        raise ValueError(
            f"--paper-faithful requires the maximal local evaluation window: "
            f"--eval-seqs {full_window_sequences} sequences "
            f"(wikitext test at seq {args.seq_len} yields "
            f"{evaluation['full_blocks']} blocks = "
            f"{evaluation['full_local_window_batches']} batches of {BATCH}); "
            f"got {args.eval_seqs} sequences ({evaluation['selected_blocks']} blocks, "
            f"{evaluation['coverage_of_source_tokens']:.3%} of source tokens). "
            "Diagnostics may use a smaller window and will record it; a "
            "faithful claim may not."
        )

    # --- independent tune geometry ---------------------------------------
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
    ppl_base = perplexity_from_logits(lambda b: model(b.to(_device)), held_out)
    print(f"      ppl_baseline = {ppl_base:.4f}", flush=True)

    # --- per-block progress probe ----------------------------------------
    # The run dies in minutes instead of after the full block sweep. The fp
    # baseline is measured ONCE, here, before quantization: the head slice of
    # the SAME held-out batches the end-of-run gate uses, in the same order,
    # under the same seed discipline — never a separately generated dataset.
    # Every depth's ratio therefore divides by the identical pedestal, so the
    # only difference between the quantized prefix and its baseline is
    # quantization damage.
    probe_slice = held_out[: args.probe_batches]
    fp_probe_ppl = perplexity_from_logits(lambda b: model(b.to(_device)), probe_slice)
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
            prefix_ppl = perplexity_from_logits(lambda b: host_model(b.to(_device)), probe_slice)
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
    # disagree with the run it describes. The alias deliberately avoids the
    # existing local ``_resolve_device`` (the torch-STAGE device resolver above):
    # shadowing it made every driver run die on UnboundLocalError before the
    # pipeline started.
    from quantibbler.ptq.config import PTQConfig as _PTQConfig
    from quantibbler.ptq.lb_admm import admm_reg_provenance as _reg_prov
    from quantibbler.ptq.lb_admm import rho_end_provenance as _rho_prov
    from quantibbler.ptq.solver_backend import (
        resolve_device as _solver_resolve_device,
    )
    from quantibbler.ptq.solver_backend import (
        solver_backend_provenance as _backend_prov,
    )

    _cfg_defaults = _PTQConfig()
    _rho_end_value, _rho_end_source = _rho_prov(_cfg_defaults.rho_end)
    # The stabilizer floor is solve-time env-overridable, resolved from
    # the same helper the solver calls so the report cannot disagree with the run.
    _reg_value, _reg_source = _reg_prov(_cfg_defaults.admm_reg)
    # The arm and the device it actually ran on, resolved the
    # same way the solver resolves them (SOLVER_BACKEND knob, explicit device env).
    _backend_value, _backend_source = _backend_prov(_cfg_defaults.solver_backend)
    _solver_device = _solver_resolve_device(
        _backend_value, device=os.environ.get("QUANTIBBLER_DEVICE")
    )
    prov = provenance(
        args.solver,
        args.scale_dtype,
        remainder_dtype=args.remainder_dtype,
        rho_end=_rho_end_value,
        rho_end_source=_rho_end_source,
        # rho_start has no env override: it is always the config default.
        rho_start=_cfg_defaults.rho_start,
        rho_start_source="config",
        admm_reg=_reg_value,
        admm_reg_source=_reg_source,
        solver_backend=_backend_value,
        solver_backend_source=_backend_source,
        solver_device=_solver_device,
    )

    # --- stage record ------------------------------------------------------
    # The pipeline reports every stage through this recorder; the JSON-safe
    # part of each record lands in the run report as `stages`, and
    # `stage_map` resolves it against [NQ] Algorithm 1 lines 0-28. Tensors are
    # dropped (they are not report material; the numbers they carried are).
    # With checkpointing enabled the same record is appended, per event, to the
    # stage journal beside the checkpoints, so a run resumed after a crash in
    # the pipeline tail still carries the stages that actually fired.
    stage_events: list[tuple[str, dict]] = []
    if args.no_checkpoints:
        _ckpt_dir = None
    else:
        _ckpt_dir = _checkpoint_launcher_dir(args.checkpoint_dir)
        if args.checkpoint_dir is None:
            # Named, not silent: a crash-resume needs this path, and the trail is
            # the only place a corpse is guaranteed to have written it.
            print(
                f"[checkpoint] per-run checkpoint dir: {_ckpt_dir} "
                "(pass --checkpoint-dir <dir> to make a launch resumable)",
                flush=True,
            )
    _record = _stage_recorder(stage_events, _ckpt_dir)
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
            checkpoint_dir=_ckpt_dir,
            recorder=_record,
            stage_sink=stage_events,
        )
        _bread(f"quantize_model returned: {len(qm.layers)} layers")
        print(f"      solver wall: {time.time() - solver_t:.0f}s", flush=True)

        print("[5/6] quantized perplexity (checkpoint-native dtype, artifact written back)")
        host = model  # quantize_model wrote the artifact back into the host
        ppl_quant = perplexity_from_logits(lambda b: host(b.to(_device)), held_out)
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
            # self-describing for the server's transformer forward. The inventory
            # IS the layout contract — sidecar order
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
                geometry=geometry,
                inventory=inventory,
                norm_weights=norm_weights,
                lm_head_f64=lm_head_f64,
                tied=tied,
                remainder_dtype=args.remainder_dtype,
            )
            artifact_exported = True
        else:
            print(
                "[6/6] export SKIPPED: factor_mode='fp' is a measurement — no artifact is written",
                flush=True,
            )
        ppl_ratio = ppl_quant / ppl_base
        status = verdict(ppl_ratio)
        # Phase 3 (Algorithm 1 lines 26-27) runs inside quantize_model, and a
        # real fire now records its own row (device + elapsed_s) from the
        # calibrate_scales call site in api.py. Only the factor_mode='fp'
        # MEASUREMENT, which skips the stage entirely, is recorded here, so the
        # stage map still names the skip without a second row for a fire that
        # already measured itself. Through the recorder (not a bare append) so
        # the record and the journal beside the checkpoints carry the same
        # event set.
        if not measured_artifact:
            _record(
                "tune_scales_kd",
                factor_mode=args.factor_mode,
                artifact=measured_artifact,
            )
        # The fidelity label is a COMPUTED verdict over the run's own record:
        # the flag AND an empty deviation list AND no mechanism-class deviation.
        # Every condition is recorded beside it, so a reader can see which one
        # denied the claim.
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
            "remainder_dtype": args.remainder_dtype,
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
            # Stage record + Algorithm 1 map: every
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
        # Fail-closed stage gate: a run that
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
