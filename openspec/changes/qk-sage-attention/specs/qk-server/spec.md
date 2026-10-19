# qk-sage-attention — Spec Delta (qk-server)

## ADDED Requirements

### Requirement: Sage attention face with measured quality gate

The server SHALL expose `gpu:sage` as a third attention face (opt-in env,
frozen at load, honest `/health` label) only for models whose shipped gate
record passes: Cosine Sim ≥ 0.999 and Relative L1 ≤ 0.02 vs the exact face
on captured real activations, plus a quantized-PPL delta within the bound
recorded at the gate run ([SA] §4.3 metrics; Table 3 worst-case
motivation). A model that fails the gate MUST refuse the face at startup
with a typed error — never serve silently degraded attention.

#### Scenario: Gate pass enables the face

- **WHEN** the opt-in is set and the model's captured gate record is within
  bounds,
- **THEN** the server starts with `"attention_face": "gpu:sage"` and the
  label stays fixed for the process lifetime.

#### Scenario: Gate fail refuses the face

- **WHEN** the opt-in is set and the gate record is missing or out of
  bounds,
- **THEN** startup aborts with a typed error naming the gate (it does NOT
  silently fall back — a silently degraded model is worse than a failed
  start).

#### Scenario: Soft fallback on device failure only

- **WHEN** the gate passes but device init fails,
- **THEN** the face falls back to the exact tiled face (or scalar) exactly
  per the frozen-decision pattern — quality is never the fallback trigger.

### Requirement: Quality gate provenance is shipped and reproducible

The gate record (activations source, per-layer metrics, PPL delta, date,
code commit) SHALL be shipped with the model directory and regenerable by a
committed tool against the pinned [SA] §4.3 metric definitions, so the
face's enabling evidence is auditable rather than asserted.

#### Scenario: Gate record regeneration matches

- **WHEN** the committed gate tool re-runs against the model directory and
  its captured activation fixture,
- **THEN** it reproduces the shipped record's metrics within f32
  round-off, and the server's startup gate accepts it.
