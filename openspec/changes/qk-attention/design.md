# qk-attention — Design

## Implementation crystallization

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
