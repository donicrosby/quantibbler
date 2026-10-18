## ADDED Requirements

### Requirement: The model directory SHALL carry a norm weights sidecar

The exported model directory SHALL contain a norm sidecar holding every
transformer block's input-layernorm and post-attention-layernorm weights
plus the model-final norm — exactly `(2 x n_layers + 1) x d_model` fp32
little-endian floats. The loader SHALL validate its byte length against
that count and reject the directory fail-closed (per `qk-server` SR-1.2
semantics) when it is missing, truncated, or miscounted.

#### Scenario: N-1.1 — Load succeeds with a complete sidecar
- **WHEN** the server loads a model directory whose sidecar matches
  `n_layers` blocks at `d_model` each
- **THEN** loading completes and the engine reports ready

#### Scenario: N-1.2 — Missing or malformed sidecar fails closed
- **WHEN** the sidecar is absent, truncated, or wrong-count
- **THEN** the server never reports ready and exits non-zero naming the
  failing artifact

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

Generation SHALL apply, per block in manifest order: input RMSNorm, packed
factorized Q/K/V projections, RoPE, GQA causal self-attention over a KV
cache, output projection, residual add, post-attention RMSNorm, gate/up
projections with SwiGLU, down projection, residual add — then final norm
and LM head. Attention/norm/residual accumulation is f32; packed matvecs
keep the kernels rlib's f64 accumulation semantics.

#### Scenario: E-2.1 — Greedy decode is deterministic and EOS-terminated
- **WHEN** a prompt is decoded greedily twice with the same inputs
- **THEN** both runs produce byte-identical token streams and terminate on
  EOS under the existing EOS/stop semantics

#### Scenario: E-2.2 — Block forward matches the HF reference on a tiny fixture
- **WHEN** the engine's forward runs against the HF reference forward on a
  tiny Qwen3 fixture with identical weights and fixed inputs
- **THEN** the maximum absolute logit divergence is within 1e-2
