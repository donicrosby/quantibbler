"""qk-nanoquant-alignment-lane0 — the decisive probe's contracts, scenario by scenario.

RED-FIRST RECORD (pre-implementation): every test below fails at collection with
``ModuleNotFoundError: tools.lane0_precond_probe`` (or an attribute error on the
absent symbols), because the harness does not exist yet. That is the recorded red
for this lane's Phase 1.

The lane is measurement-only: it builds two preconditioner variants for a pinned
set of real production layers and runs BOTH through the shipped ``lb_admm`` +
``magnitude_balance``. Nothing here reaches the export path, and no test asserts a
verdict band because the lane emits none.

Test names carry the scenario ids from
``openspec/changes/qk-nanoquant-alignment-lane0/specs/qk-ptq/spec.md``.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

# Fixtures are imported from the sibling driver test, the established pattern in
# this tier (see test_driver_stage_map.py): one locally-built tiny Qwen3 and one
# locally-built parquet corpus, shared rather than rebuilt per module.
from test_driver_tune_seq_len import offline_corpus, tiny_model  # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "lane0_precond_probe.py"

SHIPPED_MODULES = (
    "src/quantibbler/ptq/calibration.py",
    "src/quantibbler/ptq/lb_admm.py",
    "src/quantibbler/ptq/block_recon.py",
    "src/quantibbler/ptq/config.py",
)


def _load_tool():
    """The harness as a module; ``main()``-guarded, so importing is inert."""
    spec = importlib.util.spec_from_file_location("lane0_precond_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def smoke_rows(tool):
    """Two complete rows (one per arm) on the synthetic planted target."""
    return tool.smoke_rows()


# --- measurement_only_no_export ------------------------------------------------


def test_measurement_only_no_export(tool, smoke_rows):
    """No lane path reaches an export site, and the report carries no band/BPW."""
    report = tool.build_report(smoke_rows, probe_set=[("synthetic", 0, "planted-rank16")])
    assert report["artifact_exported"] is False
    assert "status" not in report
    assert "bpw" not in json.dumps(report).lower()
    source = TOOL.read_text()
    for forbidden in ("extract_and_pack", "qpk1", "greedy_polish", "TuneLatentSTE"):
        assert forbidden not in source, f"lane-0 tool must not reference {forbidden}"


def test_shipped_pipeline_modules_untouched():
    """The four shipped modules carry no diff against HEAD."""
    proc = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", *SHIPPED_MODULES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stdout.strip() == "", f"shipped modules were modified:\n{proc.stdout}"


# --- fixed_probe_set_identity_recorded / both_arms_identical_probe_set ----------


def test_fixed_probe_set_identity_recorded(tool):
    """PROBE_SET is a pinned constant: entries carry model, block, layer."""
    probe_set = tool.PROBE_SET
    assert len(probe_set) >= 2
    for entry in probe_set:
        assert len(entry) == 3
        model, block_idx, layer_name = entry
        assert isinstance(model, str) and model
        assert isinstance(block_idx, int) and block_idx >= 0
        assert isinstance(layer_name, str) and layer_name
    # attention and MLP, at least two distinct blocks (design constraint).
    assert any("self_attn" in e[2] for e in probe_set)
    assert any("mlp" in e[2] for e in probe_set)
    assert len({e[1] for e in probe_set}) >= 2
    # no CLI override exists for the set
    help_proc = subprocess.run(
        [sys.executable, str(TOOL), "--help"], capture_output=True, text=True, check=False
    )
    assert help_proc.returncode == 0
    assert "--probe-set" not in help_proc.stdout
    assert "--layers" not in help_proc.stdout


def test_both_arms_identical_probe_set(tool, smoke_rows):
    """Both arms consume the same set in the same order, one row pair per layer."""
    # Explicitly smoke-scale: run_arms defaults to production_config() (rank 512),
    # which is the FIRE's setting and takes minutes per solve. A test that omits
    # cfg silently buys a rank-512 solve pair and hangs the suite.
    rows = tool.run_arms(
        [("synthetic", 0, "planted-rank16")],
        seed=0,
        cfg=tool.PTQConfig(
            solver_backend="numpy",
        ),
    )
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row["layer_name"], []).append(row["arm"])
    for layer_name, arms in by_layer.items():
        assert sorted(arms) == ["A", "B"], f"{layer_name} did not get both arms: {arms}"


# --- sqrt_fold_once_per_side ---------------------------------------------------


def test_sqrt_fold_once_per_side(tool):
    """W~ = D~_out W D~_in, entries the sqrt of the robust statistics, once per side."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 8))
    d_in_raw = np.full(8, 4.0)
    d_out_raw = np.full(16, 9.0)
    folded, d_out, d_in = tool.fold_intended(w, d_out_raw=d_out_raw, d_in_raw=d_in_raw, gamma=0.0)
    # sqrt applied once per side, against the clip-only (gamma=0) statistics
    np.testing.assert_allclose(d_out, np.sqrt(d_out_raw))
    np.testing.assert_allclose(d_in, np.sqrt(d_in_raw))
    np.testing.assert_allclose(folded, d_out[:, None] * w * d_in[None, :], rtol=1e-12)


# --- gradient_side_real_ce_backward --------------------------------------------


def test_gradient_side_real_ce_backward(tool):
    """The gradient statistic is the mean square of a real CE weight gradient.

    A planted linear + planted labels: the intended gradient statistic must equal
    the mean-square of the CE-loss weight gradient, and must NOT equal the
    uniform-cotangent value the shipped path produces.
    """
    import torch

    torch.manual_seed(0)
    linear = torch.nn.Linear(6, 4, bias=False, dtype=torch.float64)
    x = torch.randn(5, 6, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 0])
    ce_stat = tool.ce_gradient_statistic(linear, x, labels)
    loss = torch.nn.functional.cross_entropy(linear(x), labels)
    linear.zero_grad()
    loss.backward()
    planted = (linear.weight.grad**2).mean(dim=0).numpy()
    np.testing.assert_allclose(ce_stat, planted, rtol=1e-10)
    # and it is not the uniform-cotangent statistic
    uniform = tool.uniform_cotangent_statistic(linear, x)
    assert not np.allclose(ce_stat, uniform, rtol=1e-6)


def test_intended_pass_collects_real_ce_gradients(tool, tiny_model):  # noqa: F811
    """The full-model intended pass reaches probed linears and records fallbacks.

    Runs the real collector over the tiny fixture: every requested (block, layer)
    comes back with both diagonals, the applied percentile, and an explicit
    grad_fallback field (None when the CE gradient reached the layer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_model))
    model = AutoModelForCausalLM.from_pretrained(str(tiny_model), dtype=torch.float32).eval()
    ids = tok("the quick brown fox jumps over the lazy dog . " * 8)["input_ids"]
    ids = (ids * 8)[: 4 * 32]  # the fixture corpus is tiny; tile to a full batch
    batch = torch.tensor(ids).view(4, 32)
    wanted = ((0, "mlp.down_proj"),)
    stats = tool.collect_intended_statistics(
        model, [batch], wanted=wanted, percentile=99.9, checkpoint_blocks=False
    )
    assert set(stats) == set(wanted)
    entry = stats[(0, "mlp.down_proj")]
    assert entry["percentile"] == 99.9
    assert "grad_fallback" in entry
    assert entry["d_in_raw"].ndim == 1 and entry["d_out_raw"].ndim == 1
    assert np.all(np.isfinite(entry["d_in_raw"])) and np.all(np.isfinite(entry["d_out_raw"]))
    assert entry["d_in_raw"].shape == (model.model.layers[0].mlp.down_proj.weight.shape[1],)


def _tiny_batch(tiny_model, rows=2, cols=32):  # noqa: F811
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_model))
    ids = tok("the quick brown fox jumps over the lazy dog . " * 8)["input_ids"]
    ids = (ids * 8)[: rows * cols]
    return torch.tensor(ids).view(rows, cols)


def _tiny_fp64_model(tiny_model):  # noqa: F811
    import torch
    from transformers import AutoModelForCausalLM

    # Pin the attention implementation the container will actually run: the
    # checkpointing bound must hold for the real path, not for a default that
    # might differ between sandbox and image.
    return AutoModelForCausalLM.from_pretrained(
        str(tiny_model), dtype=torch.float64, attn_implementation="sdpa"
    ).eval()


def test_chunked_ce_preserves_statistic_exactly(tool, tiny_model):  # noqa: F811
    """Chunked head forward must reproduce the whole-batch statistic BITWISE.

    Chunking is additive over rows only if the backward is NOT chunked. The hook
    sits on trunk outputs, where attention couples rows, so per-chunk backward
    would capture partial gradients and squared partials do not sum. The
    mandated shape -- checkpointed head chunks, summed losses, mean gauge on the
    loss, ONE backward -- is an identity, so the bar is equality, not tolerance.
    """
    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)
    wanted = ((0, "mlp.down_proj"),)
    key = (0, "mlp.down_proj")

    def _stats(**kw):
        return tool.collect_intended_statistics(
            model, [batch], wanted=wanted, percentile=99.9, **kw
        )[key]

    reference = _stats(chunk_tokens=None, checkpoint_blocks=False)
    chunked_only = _stats(chunk_tokens=8, checkpoint_blocks=False)
    shipped = _stats(chunk_tokens=8, checkpoint_blocks=True)

    # Two pairwise identities, each isolating one change on a path we actually
    # run: head chunking alone, then block checkpointing on top of it.
    for label, candidate in (("head chunking", chunked_only), ("block checkpointing", shipped)):
        for field in ("d_in_raw", "d_out_raw", "sum_sq_in", "sum_sq_out"):
            a, b = reference[field], candidate[field]
            assert np.array_equal(a, b), (
                f"{field} moved under {label}: max abs diff {np.abs(a - b).max():.3e}"
            )
        assert reference["rows"] == candidate["rows"]
        assert reference["grad_rows"] == candidate["grad_rows"]


def test_per_chunk_backward_does_not_preserve_the_statistic(tool, tiny_model):  # noqa: F811
    """The forbidden shape is RED: squared per-chunk partials do not sum.

    This is the fixture bug-hunter asked for, kept as an explicit red case so the
    ban is pinned by measurement rather than by prose. If someone reintroduces
    per-chunk backward, this test still passes (it asserts the difference) while
    the identity test above goes red -- the pair together is the guard.
    """
    import torch
    import torch.nn.functional as F

    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)
    head = model.get_output_embeddings()
    hidden = model.model(batch, use_cache=False)[0]
    hh = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    ll = batch[:, 1:].reshape(-1)

    captured: dict[str, torch.Tensor] = {}
    target = next(m for n, m in model.named_modules() if n.endswith("mlp.down_proj"))

    def _hook(mod, inp, out):  # noqa: ANN001
        if out.requires_grad:
            out.register_hook(
                lambda g: captured.__setitem__(
                    "total", (g.detach().reshape(-1, g.shape[-1]) ** 2).sum(0)
                )
            )

    handle = target.register_forward_hook(_hook)

    # (a) ONE backward from the summed loss: square of the TOTAL gradient.
    model.zero_grad(set_to_none=True)
    hidden = model.model(batch, use_cache=False)[0]
    hh = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    total_loss = F.cross_entropy(head(hh), ll, reduction="sum")
    total_loss.backward()
    square_of_total = captured["total"].clone()

    # (b) per-chunk backward: sum of squared PARTIAL gradients.
    model.zero_grad(set_to_none=True)
    captured.clear()
    hidden = model.model(batch, use_cache=False)[0]
    hh = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    acc = None
    for lo in range(0, hh.shape[0], 8):
        hi = min(lo + 8, hh.shape[0])
        part = F.cross_entropy(head(hh[lo:hi]), ll[lo:hi], reduction="sum")
        part.backward(retain_graph=True)
        acc = captured["total"].clone() if acc is None else acc + captured["total"]
    handle.remove()

    assert acc is not None
    rel = float((acc - square_of_total).abs().max() / square_of_total.abs().max())
    assert rel > 1e-6, (
        f"per-chunk backward matched the whole-batch statistic (rel {rel:.3e}); "
        "the ban on it would be unnecessary"
    )


def test_trunk_checkpointing_preserves_statistic_exactly(tool, tiny_model):  # noqa: F811
    """Checkpointing the trunk blocks must not move the statistic at all.

    Recompute in backward runs the same kernels on the same inputs, so the bar is
    identity, not tolerance -- the same bar that caught the fp32-cast trap in the
    head-chunk fixture.
    """
    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)
    wanted = ((0, "mlp.down_proj"),)

    plain = tool.collect_intended_statistics(
        model, [batch], wanted=wanted, percentile=99.9, chunk_tokens=8, checkpoint_blocks=False
    )
    ckpt = tool.collect_intended_statistics(
        model, [batch], wanted=wanted, percentile=99.9, chunk_tokens=8, checkpoint_blocks=True
    )

    key = (0, "mlp.down_proj")
    for field in ("d_in_raw", "d_out_raw"):
        a, b = plain[key][field], ckpt[key][field]
        assert np.array_equal(a, b), (
            f"{field} moved under trunk checkpointing: max abs diff {np.abs(a - b).max():.3e}"
        )
    # Raw-sum identity, not just mean identity. A doubled numerator over a
    # doubled denominator rounds to the same mean, so a mean-equality bar cannot
    # see the doubling at all -- these assertions are the ones that can.
    for field in ("sum_sq_in", "sum_sq_out"):
        a, b = plain[key][field], ckpt[key][field]
        assert np.array_equal(a, b), (
            f"{field} differs under checkpointing: max abs diff {np.abs(a - b).max():.3e}"
        )
    assert plain[key]["rows"] == ckpt[key]["rows"]


def _raw_sums(tool, model, batch, **kwargs):
    stats = tool.collect_intended_statistics(
        model, [batch], wanted=((0, "mlp.down_proj"),), percentile=99.9, chunk_tokens=8, **kwargs
    )
    return stats[(0, "mlp.down_proj")]


def test_ungated_accumulator_doubles_the_raw_sum(tool, tiny_model):  # noqa: F811
    """RED: drop the grad-enabled gate and the no-grad first pass counts too.

    The mean is unchanged by the doubling -- 2S/2R rounds identically to S/R --
    so this test also demonstrates why the previous mean-equality bar could not
    see the defect the raw-sum bar catches.
    """
    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)

    gated = _raw_sums(tool, model, batch, gate_on_grad=True)
    ungated = _raw_sums(tool, model, batch, gate_on_grad=False)

    assert not np.array_equal(gated["sum_sq_in"], ungated["sum_sq_in"])
    assert ungated["rows"] / gated["rows"] == pytest.approx(2.0)
    assert np.array_equal(gated["d_in_raw"], ungated["d_in_raw"])


def test_non_reentrant_checkpoint_doubles_the_raw_sum(tool, tiny_model):  # noqa: F811
    """RED: non-reentrant checkpointing runs the first pass grad-enabled.

    With both passes counted, sum_sq doubles while the mean stays put -- the
    trap that made bitwise preservation rest on ratio arithmetic.
    """
    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)

    reentrant = _raw_sums(tool, model, batch, reentrant=True)
    non_reentrant = _raw_sums(tool, model, batch, reentrant=False)

    assert not np.array_equal(reentrant["sum_sq_in"], non_reentrant["sum_sq_in"])
    assert non_reentrant["rows"] / reentrant["rows"] == pytest.approx(2.0)


def test_every_trunk_block_is_gradient_checkpointed(tool, tiny_model):  # noqa: F811
    """RED case for a block left uncheckpointed.

    Checkpointing is proven by recomputation: an instrumented submodule inside
    each block fires once on the forward and once more when backward recomputes
    it. A block left unwrapped fires exactly once, so any omission shows up as a
    count of 1 and the test goes red -- which is the point, since "some blocks
    checkpointed" would bound nothing.
    """

    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)
    trunk = model.model
    wanted = ((0, "mlp.down_proj"),)

    def _count_attn_calls() -> dict[int, int]:
        counts: dict[int, int] = {}
        handles = [
            layer.self_attn.register_forward_hook(
                lambda m, i, o, idx=idx: counts.__setitem__(idx, counts.get(idx, 0) + 1)
            )
            for idx, layer in enumerate(trunk.layers)
        ]
        return counts, handles

    counts, handles = _count_attn_calls()
    tool.collect_intended_statistics(
        model, [batch], wanted=wanted, percentile=99.9, chunk_tokens=8, checkpoint_blocks=True
    )
    for h in handles:
        h.remove()
    assert set(counts) == set(range(len(trunk.layers))), counts
    assert all(n == 2 for n in counts.values()), f"not every block recomputed: {counts}"

    counts, handles = _count_attn_calls()
    tool.collect_intended_statistics(
        model, [batch], wanted=wanted, percentile=99.9, chunk_tokens=8, checkpoint_blocks=False
    )
    for h in handles:
        h.remove()
    assert all(n == 1 for n in counts.values()), f"unexpected recompute: {counts}"


def test_checkpointing_fails_closed_without_blocks(tool):
    """A trunk with no .layers must raise, never silently run unbounded."""
    import torch

    class _Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 4)

    with (
        pytest.raises(ValueError, match="requires a decoder exposing .layers"),
        tool._checkpointed_blocks(_Bare()),
    ):
        pass


# --- per_token_norm_clip_999 ---------------------------------------------------


def test_per_token_norm_clip_999(tool):
    """A per-token norm clip at the robust percentile caps an injected outlier."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 4)) * 0.1
    x[0] = 500.0  # injected outlier token
    clipped, applied = tool.clip_token_norms(x, percentile=99.9)
    assert applied == 99.9
    # the outlier token is capped; the bulk is untouched
    assert np.linalg.norm(clipped[0]) < np.linalg.norm(x[0])
    assert np.linalg.norm(clipped[0]) <= np.percentile(np.linalg.norm(x, axis=1), 99.9) * (1 + 1e-9)
    np.testing.assert_allclose(clipped[1:], x[1:], rtol=1e-12)


# --- same_solver_pair_both_arms / pre_polish_stage_boundary --------------------


def test_same_solver_pair_both_arms(tool, smoke_rows):
    """Both arms call the shipped solver pair under the same production config."""
    from quantibbler.ptq.config import PTQConfig

    cfg = tool.production_config()
    assert isinstance(cfg, PTQConfig)
    # Production geometry, not the dataclass smoke default: the band this lane
    # exists to move was measured at rank 512.
    assert cfg.rank == tool.PRODUCTION_RANK == 512
    default = PTQConfig(
        solver_backend="numpy",
    )
    assert (cfg.compute_dtype, cfg.admm_max_iter) == (
        default.compute_dtype,
        default.admm_max_iter,
    )
    # the arms share ONE config -- that identity is the scenario's point
    arms = {row["arm"]: row for row in smoke_rows}
    assert arms["A"]["cfg_rank"] == arms["B"]["cfg_rank"]
    assert arms["A"]["solver"] == arms["B"]["solver"] == "admm"
    assert arms["A"]["compute_dtype"] == arms["B"]["compute_dtype"] == cfg.compute_dtype


def test_pre_polish_stage_boundary(tool, smoke_rows):
    """Every row is the post-balance pre-polish state: no polish, no STE."""
    for row in smoke_rows:
        assert row["stage"] == "post_admm_pre_polish"
        assert row["polish_invoked"] is False
        assert row["ste_invoked"] is False


# --- four_quantities_per_layer_per_arm / recon_cos_against_original_weight -----


def test_four_quantities_per_layer_per_arm(tool, smoke_rows):
    """cos, the full residual trajectory, n_iter, and the min_eigs floor."""
    for row in smoke_rows:
        assert isinstance(row["recon_cos"], float) and np.isfinite(row["recon_cos"])
        assert isinstance(row["residuals"], list) and row["residuals"]
        assert isinstance(row["n_iter"], int) and row["n_iter"] > 0
        assert len(row["residuals"]) == row["n_iter"]
        assert isinstance(row["min_eigs_floor"], float) and np.isfinite(row["min_eigs_floor"])
        assert row["model"] and isinstance(row["block_idx"], int) and row["layer_name"]
    # the applied percentile is an arm-B record (it is arm B's construction)
    assert {r["arm"]: r["percentile"] for r in smoke_rows} == {"A": None, "B": 99.9}


def test_four_quantities_fail_closed(tool):
    """A truncated quantity fails the run instead of writing a partial row."""
    with pytest.raises(ValueError):
        tool.assert_row_complete({"arm": "A", "recon_cos": 0.1})


def test_recon_cos_against_original_weight(tool):
    """The cos is against the ORIGINAL weight, matching the cos_raw convention."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 16))
    product = w @ np.eye(16)  # a perfect reconstruction of w
    assert tool.recon_cos(w, product) == pytest.approx(1.0, abs=1e-9)
    assert tool.recon_cos(w, -product) == pytest.approx(-1.0, abs=1e-9)
    assert tool.recon_cos(w, np.zeros_like(w)) == 0.0


# --- winner_named_per_probed_layer / falsified_hypothesis_gates_lanes ----------


def test_winner_named_per_probed_layer(tool):
    """A layer arm B does not strictly win is recorded as not-won, never omitted."""
    rows = [
        {"layer_name": "a", "arm": "A", "recon_cos": 0.10},
        {"layer_name": "a", "arm": "B", "recon_cos": 0.50},
        {"layer_name": "b", "arm": "A", "recon_cos": 0.30},
        {"layer_name": "b", "arm": "B", "recon_cos": 0.30},
    ]
    verdict = tool.build_verdict(rows, band_top=0.17)
    assert verdict["per_layer"]["a"]["winner"] == "B"
    assert verdict["per_layer"]["b"]["winner"] == "not-won"
    assert verdict["per_layer"]["b"]["tie"] is True


def test_falsified_hypothesis_gates_lanes(tool):
    """A non-domination verdict carries the Lanes 1-3 STOP gate."""
    rows = [
        {"layer_name": "a", "arm": "A", "recon_cos": 0.50},
        {"layer_name": "a", "arm": "B", "recon_cos": 0.20},
        {"layer_name": "b", "arm": "A", "recon_cos": 0.40},
        {"layer_name": "b", "arm": "B", "recon_cos": 0.10},
    ]
    verdict = tool.build_verdict(rows, band_top=0.17)
    assert verdict["dominates"] is False
    assert verdict["hypothesis"] == "falsified"
    assert verdict["lanes_1_3_gate"] == "STOP"
    assert "BLOCKED" in verdict["paragraph"]


def test_domination_rule_needs_majority_and_band_exit(tool):
    """Domination = strict majority of layers AND mean cos above the band top."""
    # majority but mean still inside the band -> not dominating
    inside = [
        {"layer_name": "a", "arm": "A", "recon_cos": 0.10},
        {"layer_name": "a", "arm": "B", "recon_cos": 0.16},
        {"layer_name": "b", "arm": "A", "recon_cos": 0.10},
        {"layer_name": "b", "arm": "B", "recon_cos": 0.15},
    ]
    assert tool.build_verdict(inside, band_top=0.17)["dominates"] is False
    # majority and above the band -> dominating
    outside = [
        {"layer_name": "a", "arm": "A", "recon_cos": 0.10},
        {"layer_name": "a", "arm": "B", "recon_cos": 0.60},
        {"layer_name": "b", "arm": "A", "recon_cos": 0.10},
        {"layer_name": "b", "arm": "B", "recon_cos": 0.55},
    ]
    verdict = tool.build_verdict(outside, band_top=0.17)
    assert verdict["dominates"] is True
    assert verdict["hypothesis"] == "supported"
    assert verdict["lanes_1_3_gate"] == "OPEN"


def test_whole_batch_head_with_checkpointed_blocks_fails_closed(tool, tiny_model):  # noqa: F811
    """The anomalous combination is fenced, not merely avoided.

    Whole-batch head + checkpointed blocks measures a different gradient from
    every path the fire runs. Failing closed keeps the anomaly a named edge
    instead of a landmine someone re-derives later.
    """
    model = _tiny_fp64_model(tiny_model)
    batch = _tiny_batch(tiny_model)
    wanted = ((0, "mlp.down_proj"),)
    with pytest.raises(ValueError, match="803c3fc"):
        tool.collect_intended_statistics(
            model,
            [batch],
            wanted=wanted,
            percentile=99.9,
            chunk_tokens=None,
            checkpoint_blocks=True,
        )
    # Both supported shapes still run.
    for kwargs in (
        {"chunk_tokens": 8, "checkpoint_blocks": True},
        {"chunk_tokens": None, "checkpoint_blocks": False},
    ):
        assert tool.collect_intended_statistics(
            model, [batch], wanted=wanted, percentile=99.9, **kwargs
        )


class _FixedTokenizer:
    """A tokenizer whose stream length is known, so shapes are deterministic."""

    def __init__(self, n_tokens: int, vocab: int | None = None) -> None:
        if vocab is None:
            self._ids = list(range(1, n_tokens + 1))
        else:
            self._ids = [i % vocab for i in range(n_tokens)]

    def __call__(self, text: str, return_tensors=None, add_special_tokens=False):  # noqa: ANN001, ANN201, ARG002
        return {"input_ids": self._ids}


def _driver():
    import sys

    tools = str(pathlib.Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import quantize_hf_model

    return quantize_hf_model


def test_rehearsal_geometry_is_the_count_that_gets_built(tool):
    """The guard must compare against what was BUILT, not the protocol count.

    Green-1 refused here: the rehearsal built sixteen batches, sliced to one,
    then checked the eight surviving rows against the protocol's 128.
    """
    assert tool.rehearsal_sequences(128, None) == 128
    assert tool.rehearsal_sequences(128, 1) == 8
    assert tool.rehearsal_sequences(128, 16) == 128


def test_rehearsal_builds_the_shapes_the_guard_checks(tool):
    """Shape assertion (the audit point): what the reduced geometry constructs.

    Runs the driver's real packing code over a fixed token stream, so this
    asserts the loader's arithmetic rather than the corpus, and asserts the
    exact quantity the guard compares.
    """
    q = _driver()
    # Enough blocks for the largest case (16 packed batches of 8).
    tokenizer = _FixedTokenizer(128 * 2048)
    for max_batches, expected_batches in ((1, 1), (2, 2), (None, 16)):
        sequences = tool.rehearsal_sequences(128, max_batches)
        n_batches = q.packed_chunks(sequences, "--calibration-seqs")
        batches = q.tokenize_chunks(tokenizer, "text", 2048, n_batches, 0)
        assert len(batches) == expected_batches, (max_batches, len(batches))
        assert all(tuple(b.shape) == (q.BATCH, 2048) for b in batches), max_batches
        # This is precisely the comparison the guard makes.
        assert sum(int(b.shape[0]) for b in batches) == sequences, max_batches


def test_build_then_slice_would_still_refuse(tool):
    """The green-1 refusal, reproduced: the reduction must happen before the loader.

    Building at protocol count and slicing afterwards leaves the guard comparing
    the eight surviving rows against the protocol's 128 -- the identical error
    the rehearsal died on.
    """
    q = _driver()
    tokenizer = _FixedTokenizer(128 * 2048)
    sliced = q.tokenize_chunks(
        tokenizer, "text", 2048, q.packed_chunks(128, "--calibration-seqs"), 0
    )[:1]
    resolved = sum(int(b.shape[0]) for b in sliced)
    assert resolved == 8
    assert resolved != 128, "the guard would have accepted a sliced construction"


def test_is_cuda_accepts_both_device_forms(tool):
    """The collector passes a torch.device and the launcher a string."""
    import torch

    assert tool._is_cuda("cuda")
    assert tool._is_cuda("cuda:0")
    assert tool._is_cuda(torch.device("cuda"))
    assert not tool._is_cuda("cpu")
    assert not tool._is_cuda(torch.device("cpu"))
    assert not tool._is_cuda(None)


def test_release_is_safe_on_cpu_and_still_collects(tool):
    """The release must never raise where there is no accelerator to release."""
    tool.release_device_memory(None)
    tool.release_device_memory("cpu")
    import torch

    tool.release_device_memory(torch.device("cpu"))


def test_telemetry_line_is_on_disk_before_the_run_ends(tool, tmp_path):
    """Flush-per-line: a run that dies does not get to close its files.

    Asserted by reading the file back after a single sample, with no stop()
    in between -- which is exactly the state a killed run leaves behind.
    """
    csv = tmp_path / "rehearsal-mem.csv"
    telemetry = tool.MemTelemetry(csv, "cpu")
    telemetry.start()
    telemetry.phase = "batch1/1"
    telemetry.write_sample(100, 50, 75, 25)
    lines = csv.read_text().splitlines()
    assert lines[0] == "t,phase,reserved,allocated,used,free"
    assert lines[1].endswith(",batch1/1,100,50,75,25"), lines
    telemetry.phase = "batch2/2"
    telemetry.write_sample(200, 150, 175, 125)
    assert csv.read_text().splitlines()[2].endswith(",batch2/2,200,150,175,125")


def test_telemetry_stop_takes_a_final_sample_with_the_live_phase(tool, tmp_path):
    """The last line names the stage that was running when the run ended.

    Spied rather than read back, because on a CPU lane `sample` correctly
    writes nothing -- the property is that stop() takes one, with the phase
    that was live at the time.
    """
    calls: list[str] = []

    class _Spy(tool.MemTelemetry):
        def sample(self) -> None:
            calls.append(self.phase)
            super().sample()

    telemetry = _Spy(tmp_path / "rehearsal-mem.csv", "cpu")
    telemetry.start()
    telemetry.phase = "stats_intended"
    telemetry.stop()
    assert calls[-1] == "stats_intended", calls


def test_build_real_arms_runs_end_to_end_on_a_stubbed_corpus(tool, tiny_model, monkeypatch):  # noqa: F811
    """The only end-to-end exercise of build_real_arms, and why it has to exist.

    A `del` of a name still used further down is invisible to every other test
    here -- none of them call this function -- and it is fatal at fire time.
    Ruff caught exactly one such bug; this test is what makes the next one fail
    in the suite instead of in the container.
    """
    q = _driver()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = _tiny_fp64_model(tiny_model)
    vocab = int(model.config.vocab_size)
    monkeypatch.setattr(q, "find_wikitext_shards", lambda split: ["stub"])  # noqa: ARG005
    monkeypatch.setattr(q, "load_wikitext_text", lambda shards: "stub text")  # noqa: ARG005

    def _stub_tokenizer(*a, **k):  # noqa: ARG001
        return _FixedTokenizer(8 * 2048, vocab)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(_stub_tokenizer))
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *a, **k: model),  # noqa: ARG005
    )

    diagnostics: dict = {}
    arms = tool.build_real_arms(
        (("Qwen/Qwen3-0.6B", 0, "mlp.down_proj"),),
        device=None,
        chunk_tokens=8,
        max_batches=1,
        diagnostics=diagnostics,
    )
    assert set(arms) == {"shipped", "intended"}
    assert diagnostics["sequences"] == 8
    assert diagnostics["protocol_sequences"] == 128
    assert diagnostics["packed_batches"] == 1
