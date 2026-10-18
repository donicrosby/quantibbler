## ADDED Requirements

### Requirement: The model directory SHALL declare its norm inventory

The exported model directory SHALL declare its per-block norm inventory
in `config.toml` as an ordered `norm_slots` array. The sidecar SHALL
hold, per block, each declared slot's vector in inventory order, then
the model-final norm, as fp32 little-endian. The total float count SHALL
be derived from the declared inventory, never from any model family's
formula.

#### Scenario: N-1.0 — Slot vocabulary, widths, and layout
- **WHEN** an inventory is declared
- **THEN** slot names come from the closed set `input`,
  `post_attention`, `q_norm`, `k_norm`, `post_feedforward`, with no
  repeats and at least one slot; `input`, `post_attention`, and
  `post_feedforward` are `d_model` floats each; `q_norm` and `k_norm`
  are `head_dim` floats each (one shared vector per projection,
  broadcast across heads — Qwen3's post-projection Q/K RMSNorm
  convention); per-block vectors follow inventory order, then the final
  norm's `d_model` floats

### Requirement: The loader SHALL validate the norm sidecar fail-closed

The loader SHALL validate the sidecar byte length against the
inventory-derived count and reject the directory fail-closed (per
`qk-server` SR-1.2 semantics) on any inventory violation (absent,
unknown-named, repeated) or sidecar violation (missing, truncated,
miscounted).

#### Scenario: N-1.1 — Load succeeds with a complete sidecar
- **WHEN** the server loads a model directory whose sidecar matches
  `n_layers` blocks at `d_model` each
- **THEN** loading completes and the engine reports ready

#### Scenario: N-1.2 — Missing or malformed sidecar fails closed
- **WHEN** the sidecar is absent, truncated, or wrong-count
- **THEN** the server never reports ready and exits non-zero naming the
  failing artifact

#### Scenario: N-1.3 — A second declared inventory of the same geometry loads
- **WHEN** two directories declare identical geometry but different
  inventories (e.g. `["input", "post_attention"]` vs
  `["input", "post_attention", "q_norm", "k_norm"]` vs a Gemma-style
  `["input", "post_attention", "post_feedforward"]`)
- **THEN** each loads iff its sidecar byte length equals its own
  inventory-derived count, and each is rejected against the other's
  count

### Requirement: The model config SHALL declare validated attention geometry

`config.toml` SHALL declare `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`,
`rms_norm_eps`, and `rope_theta`. The loader SHALL reject configurations
where `n_kv_heads * head_dim != d_model` or `n_heads % n_kv_heads != 0`
(GQA group consistency), and any missing field.

#### Scenario: N-2.1 — Consistent geometry loads
- **WHEN** the declared geometry satisfies the GQA consistency equations
- **THEN** the engine reports ready with that geometry

#### Scenario: N-2.2 — Inconsistent geometry is rejected
- **WHEN** a geometry field is missing or violates the consistency
  equations
- **THEN** the server exits non-zero naming the violation

### Requirement: The engine SHALL execute transformer-block forward semantics

Generation SHALL apply, per block in manifest order: each declared norm
slot applied at its semantics-defined point, packed factorized Q/K/V
projections, RoPE, GQA causal self-attention over a KV cache, output
projection, residual adds, gate/up projections with SwiGLU, down
projection — then final norm and LM head, in f32 with the kernels rlib's
f64 packed-matvec accumulation. Slot application points are normative
per Scenario E-2.0.

#### Scenario: E-2.0 — Slot semantics
- **WHEN** a declared inventory drives the forward
- **THEN** `input` = RMSNorm on the block input (pre-attention);
  `post_attention` = RMSNorm on the attention output (pre-MLP);
  `q_norm`/`k_norm` = RMSNorm over `head_dim` on Q/K after projection,
  before RoPE; `post_feedforward` = RMSNorm on the MLP output
  (Gemma-style sandwich, residual around each normed sublayer)

#### Scenario: E-2.1 — Greedy decode is deterministic and EOS-terminated
- **WHEN** a prompt is decoded greedily twice with the same inputs
- **THEN** both runs produce byte-identical token streams and terminate on
  EOS under the existing EOS/stop semantics

#### Scenario: E-2.2 — Block forward matches the HF reference on a tiny fixture
- **WHEN** the engine's forward runs against the HF reference forward on a
  tiny Qwen3 fixture with identical weights and fixed inputs
- **THEN** the maximum absolute logit divergence is within 1e-2

#### Scenario: E-2.3 — Inventory presence drives the forward path
- **WHEN** a model declares `q_norm`/`k_norm` its Q/K are normalized
  before RoPE, and when a model omits them its Q/K are NOT normalized;
  the same binary serves both without code changes
- **THEN** each directory's forward is driven by its declared inventory,
  and the engine rejects no inventory it declares (Qwen3 and
  Llama-class both serve)
