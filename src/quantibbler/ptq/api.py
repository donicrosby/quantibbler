"""Public PTQ API (tasks 6.1-6.4): quantize_model, QuantizedModel, and the
qpk1 serialize/reload round trip.

``quantize_model`` consumes a caller-constructed fp model and caller-supplied
calibration batches (no-network rule: this package imports no network
library — audited by test_scn_entry_point_signature) and returns a
``QuantizedModel`` whose per-layer packed binaries + float scales round-trip
through the ``qk-representation`` qpk1 container byte-exactly (D4). Forward
passes run through the qk-kernels parity-gated dispatch; the effective-BPW
accounting is computed from serialized bytes only (D5).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .block_recon import QuantizedLayerState, _quantizable_names, reconstruct_model
from .config import PTQConfig

__all__ = ["QuantizedModel", "quantize_model"]


def quantize_model(
    model: nn.Module,
    calibration_data,
    config: PTQConfig,
    *,
    checkpoint_dir=None,
    recorder=None,
) -> QuantizedModel:
    """Run the NanoQuant PTQ pipeline (Phases 1-3) and return the artifact.

    The input ``model`` is left with its factorized layers' frozen weights
    written back (the quantized weights), so it can be used directly after
    the call; the returned :class:`QuantizedModel` is the serializable view.
    Phase 3's teacher is the caller's model in its pre-reconstruction state:
    a structural deep copy is taken up front so seeded models reproduce the
    fp teacher exactly. That same frozen copy is the pristine-pinned TuneFP
    teacher source for Phase 2 (qk-fix-tunefp-teacher A1) — the pipeline
    never reads teacher weights off live, already-written-back module state.

    ``checkpoint_dir`` (optional): enables per-block solve checkpointing
    and crash resume — see :mod:`quantibbler.ptq.checkpoints`. Resumed
    runs are byte-identical to fresh runs (deterministic solve;
    fingerprint-guarded against stale configs/code).

    ``recorder`` (optional): ``recorder(event, **info)``, forwarded to the
    block loop so the run artifact can carry the stage records (teacher
    provenance, TuneFP effect, per-linear STE flip counts, scale movement,
    the TuneFP vacuity guard event).
    """
    from .scale_calib import calibrate_scales

    batches = list(calibration_data)
    if not batches:
        raise ValueError("quantize_model needs calibration data")

    quant_names = _quantizable_names(model)
    if not quant_names:
        raise ValueError("model has no quantizable nn.Linear layers")

    # Phase 1 global statistics are NOT computed here: the per-block
    # statistics passes at reconstruction time are what feed the
    # preconditioners (see design.md section 11). collect_stats remains
    # public for the QAT lane's global-pass use.

    # fp teacher, structurally frozen before any block is reconstructed
    # (Phase 3's cached logits must come from the pre-quantization model,
    # and Phase 2's TuneFP teacher is read from this same frozen copy).
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Phase 2: block-wise reconstruction (Algorithm 1 lines 8-24). Each
    # BlockReconResult carries the per-layer artifacts in declaration order.
    results = reconstruct_model(
        model,
        batches,
        config,
        checkpoint_dir=checkpoint_dir,
        recorder=recorder,
        teacher_model=teacher,
    )
    layers: dict[str, QuantizedLayerState] = {}
    for result in results:
        for st in result.layers:
            layers[st.name] = st

    # fp mode is a MEASUREMENT (qk-root-cause-ablation D-5): it makes no bits at
    # all, so there are no layer states to account for and nothing to calibrate.
    # Scale calibration serves a packed artifact, while the fp rung measures
    # HOST-WEIGHT reconstruction fidelity — that skip is the ablation boundary
    # and is design intent, not an accident of the plumbing.
    if config.factor_mode != "fp":
        missing = quant_names - set(layers)
        if missing:
            raise RuntimeError(
                f"block reconstruction skipped quantizable layers: {sorted(missing)}"
            )
        if set(layers) - quant_names:
            raise RuntimeError(
                f"block reconstruction produced out-of-scope layers: "
                f"{sorted(set(layers) - quant_names)}"
            )

        # Phase 3: global scale calibration — only s1/s2 train, packed binaries
        # are bit-frozen ([NQ] Sec 3.3, Algorithm 1 lines 26-27). The tuned
        # scales are then written back into the host model's frozen weights so
        # the host forward equals the artifact forward bit-for-bit.
        calibrate_scales(model, layers, teacher, batches, config)

    # Finalize the scales at their storage precision: the qpk1 container's
    # scale values ARE the canonical artifact scales, so rounding to the
    # storage dtype happens once, here — not implicitly inside serialize().
    # Only then is the round trip exactly lossless (bit-for-bit float64
    # scales, forward equality at float64 round-off, D4).
    scale_dt = np.float16 if config.scale_dtype == "float16" else np.float32
    for st in layers.values():
        # Storage-cast finiteness (r137-v5 post-mortem): KD's f64 scales
        # were all finite (armor never fired), then astype(float16) sent
        # 54/196 layers' s2 to inf — anything above 65,504 overflows and
        # the cast is silent. Default storage is now float32 (tag 2,
        # +~0.03 BPW at r137); if float16 is ever opted back in, an
        # overflow must die HERE with a message naming the real stage,
        # not downstream at the export gate.
        s1_cast = st.s1.astype(scale_dt).astype(np.float64)
        s2_cast = st.s2.astype(scale_dt).astype(np.float64)
        if not (np.isfinite(s1_cast).all() and np.isfinite(s2_cast).all()):
            from ..qpk1 import Qpk1Error

            raise Qpk1Error(
                f"scale storage cast to {scale_dt} overflowed to non-finite "
                f"(layer {st.name!r}); raise scale_dtype or clamp the stage "
                f"that inflated the scales"
            )
        st.s1 = s1_cast
        st.s2 = s2_cast
    _write_scales_to_host(model, layers)

    stem, post = _split_stem_post(model)
    return QuantizedModel(
        layers=layers,
        config=config,
        source_model=model,
        source_info={"stem": stem, "post": post},
    )


def _write_scales_to_host(model: nn.Module, layers: dict[str, QuantizedLayerState]) -> None:
    """Rebuild each quantized linear's frozen weight from the final scales.

    Phase 3 tunes the scale vectors held in the layer artifacts; until this
    write-back the host model still carries the Phase-2 weights, so the host
    forward would disagree with the serialized artifact after reload.
    """
    modules = dict(model.named_modules())
    for name, st in layers.items():
        linear = modules.get(name)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"layer {name!r} not found in the host model")
        with torch.no_grad():
            w_q = (st.s1[:, None] * st.Ub.astype(np.float64)) @ (
                st.s2[:, None] * st.Vb.astype(np.float64)
            ).T
            linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(w_q)))
            if linear.bias is not None and st.bias is not None:
                linear.bias.copy_(torch.from_numpy(np.ascontiguousarray(st.bias)))


def _split_stem_post(model: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Record the (fp) modules before the first and after the last block.

    Used by reloaded artifacts that have no host model: a stem (embedding)
    maps token ids into the blocks' input space, the post module (LM head)
    produces logits from the blocks' output. ``(None, None)`` when the model
    exposes no such leading/trailing non-block modules.
    """
    children = list(model.named_children())
    if not children:
        return None, None
    stem: nn.Module | None = None
    if isinstance(children[0][1], nn.Embedding):
        stem = nn.Sequential(children[0][1], nn.Flatten(start_dim=2))
    post: nn.Module | None = None
    last_name, last_mod = children[-1]
    if isinstance(last_mod, nn.Linear) and last_name not in _quantizable_names(model):
        post = last_mod
    return stem, post


class QuantizedModel:
    """A quantized model: packed binaries + float scales per layer.

    Serializes to / reloads from the qpk1 container (one stream per layer,
    host module paths as keys) and runs forwards against a host model. A
    reloaded artifact (no host model attached) forwards on its own through
    the recorded fp stem embedding + post-block head.
    """

    def __init__(
        self,
        layers: dict[str, QuantizedLayerState],
        config: PTQConfig,
        source_model: nn.Module | None = None,
        source_info: dict | None = None,
    ) -> None:
        self.layers = layers
        self.config = config
        self.source_model = source_model
        self.source_info = dict(source_info or {})
        self._stem: nn.Module | None = self.source_info.get("stem")
        self._post: nn.Module | None = self.source_info.get("post")

    # -- qpk1 round trip -------------------------------------------------

    def serialize(self) -> dict[str, bytes]:
        """Serialize every layer to its canonical qpk1 byte stream.

        Raises a :class:`Qpk1Error` subclass if ANY layer's scales are
        non-finite — the v4 export gate: a run whose KD stage poisoned
        72/196 layers once shipped NaN scales to disk silently, and the
        container still exited 0 (ppl was reported NaN downstream). No
        artifact with non-finite scales leaves the pipeline again.
        """
        import numpy as _np

        from .. import qpk1
        from ..qpk1 import Qpk1Error

        bad = sorted(
            name
            for name, st in self.layers.items()
            if not (
                _np.isfinite(_np.asarray(st.s1, _np.float64)).all()
                and _np.isfinite(_np.asarray(st.s2, _np.float64)).all()
            )
        )
        if bad:
            raise Qpk1Error(
                f"export gate: {len(bad)}/{len(self.layers)} layers carry "
                f"non-finite scales (first: {bad[0]}); refusing to serialize"
            )
        dtype = np.float16 if self.config.scale_dtype == "float16" else np.float32
        return {
            name: qpk1.serialize(st.Ub, st.Vb, st.s1, st.s2, dtype=dtype)
            for name, st in self.layers.items()
        }

    @classmethod
    def deserialize(
        cls, blobs: dict[str, bytes], config: PTQConfig | None = None
    ) -> QuantizedModel:
        """Reload from qpk1 byte streams (fail-closed reader, D8).

        Bias vectors are not part of the qpk1 container; attach the artifact
        to a host model (:meth:`attach_to`) to recover a full forward state,
        or forward through the recorded fp stem + post-head of the run that
        produced the artifact.
        """
        from .. import qpk1

        layers: dict[str, QuantizedLayerState] = {}
        for name, blob in blobs.items():
            layer = qpk1.deserialize(blob)
            layers[name] = QuantizedLayerState(
                name,
                layer.Ub,
                layer.Vb,
                np.asarray(layer.s1, dtype=np.float64),
                np.asarray(layer.s2, dtype=np.float64),
                None,
            )
        return cls(layers, config or PTQConfig())

    def attach_to(self, host: nn.Module) -> None:
        """Write the packed bits + scales into a host model's linears.

        Layer names are the host's module paths (the names quantize_model
        recorded). After this call the host carries exactly the state the
        artifact describes and the artifact forwards through the host (the
        host's fp stem/head complete the network around the quantized
        blocks, per A-9).
        """
        modules = dict(host.named_modules())
        for name, st in self.layers.items():
            linear = modules.get(name)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"layer {name!r} not found in the host model")
            w_q = (st.s1[:, None] * st.Ub.astype(np.float64)) @ (
                st.s2[:, None] * st.Vb.astype(np.float64)
            ).T
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(w_q)))
                if linear.bias is not None and st.bias is not None:
                    # The container carries no bias field; when the artifact
                    # has one (a quantize_model product) it is authoritative,
                    # otherwise the host's own bias stands (it is host state
                    # by construction — see block_recon reconstruct_block).
                    linear.bias.copy_(torch.from_numpy(np.ascontiguousarray(st.bias)))
        self.source_model = host

    def effective_bpw(self) -> float:
        """Effective BPW from serialized bytes only (D5).

        ``sum(len(blob)) * 8 / sum(dout * din)`` — header, scales and
        padding all included; never derived from parameter counts.
        """
        blobs = self.serialize()
        total_bytes = sum(len(b) for b in blobs.values())
        total_entries = sum(st.Ub.shape[0] * st.Vb.shape[0] for st in self.layers.values())
        return 8.0 * total_bytes / total_entries

    # -- forward ----------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Frozen-binary forward on the recorded/host model.

        With a host model (from quantize_model, or :meth:`attach_to`), the
        host's modules carry the artifact's factorized weights, so the host
        forward IS the quantized forward — the same dispatch surface the
        pipeline used (qk-kernels parity-gated). A bare reloaded artifact
        forwards through the recorded fp stem + post-head.
        """
        if self.source_model is not None:
            self.source_model.eval()
            with torch.no_grad():
                return self.source_model(tokens)
        if self._stem is not None and self._post is not None:
            self._stem.eval()
            self._post.eval()
            with torch.no_grad():
                return self._post(self._stem(tokens))
        raise ValueError(
            "no host model attached and no recorded stem/post; reload through "
            "quantize_model or attach_to a host model"
        )
