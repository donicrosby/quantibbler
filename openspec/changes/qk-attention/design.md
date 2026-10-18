# qk-attention — Design

## Implementation crystallization

### 2026-10-18 — Engine transformer forward + head-tying contract fix (chunk B)

- **2.2/2.3 landed:** RMSNorm (f32, config eps), RoPE (HF rotate_half
  convention — pairs `(i, i+half)`, NOT interleaved), per-head q/k RMSNorm
  (declared inventory drives application, E-2.3), GQA causal attention with
  an append-only per-block KV cache, SwiGLU MLP, residuals, final norm +
  head. `forward_prefill` returns ALL positions' `[seq x vocab]` logits
  (the parity vehicle); serving `decode_start` trunk-prefills all but the
  last prompt position and head-projects only the last. E-2.2 parity GREEN:
  max |Δlogit| = 4.1e-5 against the fp32-of-artifact goldens (budget 1e-2).
- **Golden-reference regime corrected:** the parities measure engine-vs-HF
  on the EXACT artifact math — f16 scales widen losslessly to fp32 — by
  casting the artifact-written-back model to fp32 for the golden forward.
  (A bf16 golden re-rounds the scales and would bake dtype noise into the
  budget; the committed fixture had exactly that defect.)
- **Found by the parity gate, root-caused by a Python mirror of the Rust
  algorithm (tools/diag_parity_mirror.py, mirror-vs-HF ≈ 4e-6):** the
  exporter hard-coded `weight_tying = "lm_head=e1@fp32"` — but Qwen3
  ships `tie_word_embeddings = false`, so every logit was computed through
  the embedding instead of the real head (Δ ≈ 1.43, position 0 included,
  which exonerated attention/RoPE early). Fixes: the exporter now exports
  the model's ACTUAL tying (real `head.f32.bin` when untied), and the
  loader's weight_tying grammar is the general form
  `lm_head=<path>@fp32` (tied sentinel still accepted; unknown forms fail
  closed as before). No archived spec pinned the old sentinel.
- **server-model fixture is now a real transformer block:** seven
  `layers.0.inner.*` projections (q/k/v/o/gate/up/down) at the same
  4x4x4 geometry, so load-res (manifest-name → block resolution) is
  exercised by the acquisition suite; corruption variants target
  `layers/q_proj.qpk1`. The old single `block0.mlp.fc` fixture would have
  been rejected by the resolving engine.
- **Engine tests un-ignored:** `engine_permit_bounds_and_generates` (adds
  an E-2.1 determinism double-run; EOS made unreachable via `u32::MAX`) and
  `generate_stops_at_eos` (EOS = the fixture's own first greedy token —
  self-consistent, no layout assumptions) run against the transformer
  fixture path.
- RoPE convention lesson: HF's Llama/Qwen line is rotate_half; the
  interleaved-pairs reading moved Δ by <1e-5 and was NOT the bug — the
  position-0 probe (attention output = v₀ exactly at pos 0) is what
  isolated the head.

### 2026-10-18 — Norm inventory is the layout contract (chunk A1)

- **`norm_slots` is the single source of truth for norm-sidecar layout**
  across exporter, loader, engine, and spec (N-1.0/N-1.3/E-2.0/E-2.3).
  The earlier in-flight draft hard-coded the Qwen3 count
  `(2n+1)·d_model + 2n·head_dim` in the engine and spec while the loader
  had already gone inventory-driven — the two consumers rejected each
  other's artifacts. All family formulas are now gone; Qwen3's q/k-norm
  convention is one declared inventory (`["input", "post_attention",
  "q_norm", "k_norm"]`), Llama-class and Gemma-style inventories are
  config, not code paths. `cross_inventory_sidecar_count_is_rejected`
  proves the sidecar is validated against its OWN inventory's count.
- **Exporter inventory comes from module introspection, not a family
  table** (`_export_norm_inventory`: `hasattr` probes for
  `self_attn.q_norm`/`k_norm` and `post_feedforward_layernorm`). A
  lookalike config that silently mislabels itself is worse than a loud
  `NotImplementedError`, which is what an unrecognized post-ff norm
  module shape raises.
- **`NormSlot::from_str` moved to the std `FromStr` trait** (clippy
  `should_implement_trait`); the loader's parse site uses
  `str::parse::<NormSlot>()` behind the already-validated inventory.
- **Engine `ModelWeights` now carries `norm_slots`** and derives the
  sidecar length from it (`inventory_block_floats`); the field is
  `#[allow(dead_code)]` until tasks 2.2/2.3 wire the forward to it
  (E-2.3: inventory presence drives the forward path).
- **Parity fixture ships a real minimal tokenizer.json.** The old no-op
  `save_pretrained` stub made `model::load` fail-closed on the missing
  file BEFORE the forward budget — a RED test failing for the wrong
  reason. The fixture now serializes a deterministic WordLevel tokenizer;
  the RED gate was re-observed failing at
  `unimplemented!("transformer prefill …")` (spec E-2.2), not at
  acquisition. Same run asserts `loaded norm_slots == golden norm_slots`
  (E-2.3 hook).
- **`gen_server_fixtures.GEOMETRY_TOML` carries a valid `norm_slots`**
  so corrupted-variant configs that reuse it fail on the field they
  corrupt, never on a missing inventory (rejection-class hygiene).
- **Spec delta restructured** under openspec --strict's 500-char
  requirement limit: N-1 split into "declare its norm inventory" +
  "validate the norm sidecar fail-closed", with slot vocabulary/widths
  and cross-inventory behavior in scenarios.
- **Task 2.1 ticked**: the failing-parity-test RED state was literally
  observed this session (post-A1), at the engine's unimplemented prefill
  — the test fails for the budget reason the task mandates, with
  acquisition and inventory cross-checks passing ahead of it.
- Deferred (user-raised, 2026-10-18): single-file qpk1 container
  (GGUF-style) as a future change; directory contract stays canonical
  until the engine forward lands and the first artifact ships.
