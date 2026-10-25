"""qk-fix-fp-resume-guard: empty or partial archives never pass resume.

Scenarios under test: ``empty_archive_is_never_written``,
``empty_archive_is_never_counted_done``,
``restore_asserts_expected_equals_present``.

The measured defect: an fp-mode run wrote a 22-byte zero-key npz (zip
end-of-central-directory), ``completed_blocks`` counted it as done, and the
next resume skipped that block's solve with nothing restored — so the
quantized prefix advanced on pristine weights while every gate stayed
green. The guard is now driven by what the LIVE block is OWED, never by
what an archive happens to contain.
"""

import numpy as np
import pytest
import torch
from torch import nn

from quantibbler.ptq.block_recon import QuantizedLayerState
from quantibbler.ptq.checkpoints import (
    apply_layer_states,
    completed_blocks,
    load_block_checkpoint,
    save_block_checkpoint,
)


def _state(name: str, dout: int = 4, din: int = 4, rank: int = 2, seed: int = 0):
    gen = np.random.default_rng(seed)
    ub = np.where(gen.random((dout, rank)) < 0.5, -1, 1).astype(np.int8)
    vb = np.where(gen.random((din, rank)) < 0.5, -1, 1).astype(np.int8)
    return QuantizedLayerState(
        name,
        ub,
        vb,
        gen.random(dout) + 0.5,
        gen.random(din) + 0.5,
        gen.random(dout),
    )


class _Block(nn.Module):
    """Two nested linears under ``self_attn`` / ``mlp`` (the Qwen3 shape)."""

    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(d, d, dtype=torch.float64)})
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, d, dtype=torch.float64)})
        with torch.no_grad():
            gen = torch.Generator().manual_seed(0)
            for m in (self.self_attn["q_proj"], self.mlp["gate_proj"]):
                m.weight.copy_(torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.3)
                m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp["gate_proj"](torch.tanh(self.self_attn["q_proj"](x)))


def _expected_weight(st) -> torch.Tensor:
    ub = torch.from_numpy(np.ascontiguousarray(st.Ub, dtype=np.float64))
    vb = torch.from_numpy(np.ascontiguousarray(st.Vb, dtype=np.float64))
    s1 = torch.from_numpy(np.ascontiguousarray(st.s1, dtype=np.float64))
    s2 = torch.from_numpy(np.ascontiguousarray(st.s2, dtype=np.float64))
    return (s1.unsqueeze(1) * ub) @ (s2.unsqueeze(1) * vb).T


def test_scn_empty_archive_is_never_written(tmp_path):
    """A block that is owed layers and produced none raises; no file lands."""
    with pytest.raises(RuntimeError, match="empty block checkpoint"):
        save_block_checkpoint(tmp_path, 0, [])
    assert list(tmp_path.glob("block-*.npz")) == []
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_scn_empty_archive_is_never_counted_done(tmp_path):
    """A hand-copied zero-key archive cannot masquerade as a finished block."""
    # The measured artifact: np.savez with no arrays = 22-byte empty zip.
    with open(tmp_path / "block-1.npz", "wb") as fh:
        np.savez(fh)
    assert (tmp_path / "block-1.npz").stat().st_size == 22

    good = [_state("self_attn.q_proj"), _state("mlp.gate_proj", seed=1)]
    save_block_checkpoint(tmp_path, 3, good)

    assert completed_blocks(tmp_path) == [3], "the empty archive was counted as a completed block"


def test_scn_restore_asserts_expected_equals_present(tmp_path):
    """A truncated archive raises naming the missing layer, before any write."""
    good = [_state("self_attn.q_proj"), _state("mlp.gate_proj", seed=1)]
    save_block_checkpoint(tmp_path, 0, good)

    # Truncate the archive to one of its two linears.
    truncated = [_state("self_attn.q_proj")]
    with open(tmp_path / "block-1.npz", "wb") as fh:
        arrays = {}
        for st in truncated:
            arrays[f"{st.name}|Ub"] = st.Ub
            arrays[f"{st.name}|Vb"] = st.Vb
            arrays[f"{st.name}|s1"] = st.s1
            arrays[f"{st.name}|s2"] = st.s2
        np.savez(fh, **arrays)

    block = _Block()
    before = {
        n: m.weight.detach().clone() for n, m in block.named_modules() if isinstance(m, nn.Linear)
    }
    layers = load_block_checkpoint(tmp_path, 1, QuantizedLayerState)
    with pytest.raises(RuntimeError, match=r"missing=\['mlp\.gate_proj'\]"):
        apply_layer_states(
            block, "block0", layers, expected_names={"self_attn.q_proj", "mlp.gate_proj"}
        )
    after = {n: m.weight.detach() for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    assert all(torch.equal(before[n], after[n]) for n in before), (
        "the guard fired AFTER a partial write-back"
    )

    # An empty archive restores nothing, and the expected-vs-present guard
    # says so by name.
    with open(tmp_path / "block-2.npz", "wb") as fh:
        np.savez(fh)
    empty = load_block_checkpoint(tmp_path, 2, QuantizedLayerState)
    assert empty == []
    with pytest.raises(RuntimeError, match="is incomplete"):
        apply_layer_states(block, "block0", empty, expected_names={"self_attn.q_proj"})

    # Unexpected names are reported too (a hand-edited archive cannot pass).
    with pytest.raises(RuntimeError, match="unexpected="):
        apply_layer_states(block, "block0", good, expected_names={"self_attn.q_proj"})


def test_scn_good_archive_restores_bit_exactly(tmp_path):
    """A complete archive restores exactly the weights reconstruct_block wrote."""
    layers = [_state("self_attn.q_proj"), _state("mlp.gate_proj", seed=1)]
    save_block_checkpoint(tmp_path, 0, layers)
    reloaded = load_block_checkpoint(tmp_path, 0, QuantizedLayerState)
    assert [st.name for st in reloaded] == ["mlp.gate_proj", "self_attn.q_proj"]

    block = _Block()
    apply_layer_states(
        block,
        "block0",
        reloaded,
        expected_names={"self_attn.q_proj", "mlp.gate_proj"},
    )
    by_name = {st.name: st for st in reloaded}
    for rel, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        want = _expected_weight(by_name[rel])
        assert torch.equal(module.weight, want.to(module.weight.dtype))
        assert torch.equal(
            module.bias,
            torch.from_numpy(np.ascontiguousarray(by_name[rel].bias)).to(module.bias.dtype),
        )


def test_scn_rung1_archive_format_still_loads(tmp_path):
    """The pre-guard archive format (name|Ub / |Vb / |s1 / |s2) still loads."""
    path = tmp_path / "block-7.npz"
    gen = np.random.default_rng(11)
    ub = np.where(gen.random((4, 2)) < 0.5, -1, 1).astype(np.int8)
    vb = np.where(gen.random((4, 2)) < 0.5, -1, 1).astype(np.int8)
    s1 = gen.random(4) + 0.5
    s2 = gen.random(4) + 0.5
    with open(path, "wb") as fh:
        np.savez(
            fh,
            **{
                "self_attn.q_proj|Ub": ub,
                "self_attn.q_proj|Vb": vb,
                "self_attn.q_proj|s1": s1,
                "self_attn.q_proj|s2": s2,
            },
        )
    assert completed_blocks(tmp_path) == [7]
    layers = load_block_checkpoint(tmp_path, 7, QuantizedLayerState)
    assert len(layers) == 1
    st = layers[0]
    assert np.array_equal(st.Ub, ub) and np.array_equal(st.Vb, vb)
    assert st.bias is None  # the legacy format carries no bias field

    block = _Block()
    apply_layer_states(block, "block0", layers, expected_names={"self_attn.q_proj"})
    assert torch.equal(block.self_attn["q_proj"].weight, _expected_weight(st))


def test_empty_archive_never_skips_a_solve_end_to_end(tmp_path):
    """A stray empty archive does not let a real run skip that block's solve.

    Integration form of the defect: seed the checkpoint dir with a
    fingerprint and one hand-copied 22-byte archive, then quantize. The
    block must be SOLVED (the artifact equals a fresh run's), not silently
    restored from nothing.
    """
    from test_api import _calib, _gate_cfg, _TinyApiModel

    from quantibbler.ptq.api import quantize_model
    from quantibbler.ptq.checkpoints import write_fingerprint

    cfg = _gate_cfg()
    torch.manual_seed(0)
    batches = _calib(seed=7)
    fresh = quantize_model(_TinyApiModel(seed=3), batches, cfg).serialize()

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    # The fingerprint now binds the model identity too, so
    # it is written for the model that will resume it.
    write_fingerprint(ckpt_dir, cfg, _TinyApiModel(seed=3))
    with open(ckpt_dir / "block-1.npz", "wb") as fh:
        np.savez(fh)
    assert completed_blocks(ckpt_dir) == []

    resumed = quantize_model(_TinyApiModel(seed=3), batches, cfg, checkpoint_dir=ckpt_dir)
    assert resumed.serialize() == fresh


def test_yb_only_archive_is_not_counted_done(tmp_path):
    """A y_b-only archive carries no quantized-layer keys: never a done block.

    The reserved teacher-target key (``_YB_KEY``) carries no ``|`` separator,
    so it derives no layer name — the scan-side derivation must exclude it
    exactly as ``load_block_checkpoint`` does. Otherwise a y_b-only archive
    masquerades as a finished block and the next resume SKIPS that solve with
    nothing restored: the same silent-prefix-advance shape as the zero-key
    defect, one reserved key wide.
    """
    with open(tmp_path / "block-1.npz", "wb") as fh:
        np.savez(fh, y_b=np.zeros((2, 3), dtype=np.float32))
    assert completed_blocks(tmp_path) == []


def test_expected_names_cannot_be_omitted(tmp_path):
    """The expected-vs-present guard has no disarm-by-omission default.

    A future caller that cannot name what the live block is owed has no
    business restoring into it: the parameter is required, so omitting it is
    a TypeError at the call, never a silently disarmed guard.
    """
    block = _Block()
    with pytest.raises(TypeError):
        apply_layer_states(block, "block0", [])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_yb_low_precision_survives_the_numpy_hop(tmp_path, dtype):
    """bf16/f16 teacher targets must survive the ``.numpy()`` serialization hop.

    NumPy has no bfloat16 dtype and torch refuses ``.numpy()`` on bf16
    (``unsupported ScalarType BFloat16``), so a real bf16 run crashed at block 0
    with default checkpoints -- a subprocess-only failure until a complete env
    ran the driver tier. The lift to float32 is EXACT (bf16 8-bit and f16 10-bit
    mantissas are subsets of f32), so the stored array must equal the widened
    input bit-for-bit; fresh and resumed runs widen identically, which is what
    keeps the byte-identical resume contract.
    """
    y = torch.randn(3, 5, dtype=dtype)
    save_block_checkpoint(tmp_path, 0, [_state("m.0")], y_b=y)

    with np.load(tmp_path / "block-0.npz") as npz:
        assert "y_b" in npz.files, "the teacher target did not reach the archive"
        stored = npz["y_b"]

    assert stored.dtype == np.float32
    assert np.array_equal(stored, y.float().numpy()), (
        "the bf16/f16 lift at the .numpy() hop is not exact"
    )

    # Roundtrip through the resume path must recover the same widened target.
    with np.load(tmp_path / "block-0.npz") as npz:
        assert np.array_equal(npz["y_b"], y.float().numpy())
