# qk-server delta: qk-bias-carry

## MODIFIED Requirements

### Requirement: The engine SHALL execute transformer-block forward semantics

Generation SHALL apply, per block in manifest order: each declared norm slot,
packed factorized Q/K/V projections, RoPE, GQA causal self-attention, output
projection, residual adds, gate/up projections with SwiGLU, down projection —
then final norm and LM head, in f32 with the kernels rlib's f64 packed-matvec
accumulation. Each packed projection SHALL add the bias its container
carries; a pre-bias stream contributes the zero map. Slot application points
are normative per Scenario E-2.0.

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

#### Scenario: E-2.4 — The forward applies the container bias
- **WHEN** two directories describe the same model, one with bias-carrying
  version-2 streams and one whose streams are pre-bias
- **THEN** the engine loads each bias at full width, a pre-bias stream
  contributes zeros, and the two forwards over identical token ids differ —
  an ignored bias would produce identical results.
