# qk-sage-attention — Spec Delta (qk-server)

## ADDED Requirements

### Requirement: Sage attention face SHALL be admitted only on a passing shipped gate record

The server SHALL expose `gpu:sage` as a third attention face (opt-in env,
frozen at load, honest `/health` label) only for models whose shipped gate
record passes the [SA] §4.3 metric clauses (Cosine Sim ≥ 0.999, Relative
L1 ≤ 0.02 vs the exact face). The gate SHALL distinguish evidence classes
by the record's `activations_kind`. A model that fails the gate MUST
refuse the face at startup with a typed error, never serving silently
degraded attention.

#### Scenario: Gate pass enables the face

- **WHEN** the opt-in is set and the model's shipped gate record is within
  the [SA] §4.3 metric bounds (Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02 vs
  the exact face),
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

#### Scenario: Fixture-battery evidence admits experimental status only

- **WHEN** the gate record's `activations_kind` is `"fixture-battery"` and
  the [SA] §4.3 metric clauses pass,
- **THEN** the face SHALL be admitted as `opt-in-experimental` with a WARN
  log naming the unmeasured real-activation quality, and SHALL NOT be
  promoted to a recommended status on that record alone.

#### Scenario: Real-activation evidence is the enabling class

- **WHEN** the gate record's `activations_kind` is a recognized
  real-activation source and the [SA] §4.3 metric clauses pass,
- **THEN** the face MAY be considered for status beyond experimental, and
  the log SHALL record the activation source and the clause values.

### Requirement: Gate record PPL clause is enforced when declared, admitted when absent

A shipped gate record that declares a PPL clause SHALL have its delta
within the bound recorded at that run; the server SHALL refuse the face on
violation. A record that declares no PPL clause SHALL be admitted on the
[SA] §4.3 metrics alone with the gap logged at WARN. The PPL delta bound
is a policy threshold (0.05, from the change's shipped-test convention),
not a paper-derived constant.

#### Scenario: PPL clause present and within bounds

- **WHEN** the gate record declares a `ppl` clause with `delta` and
  `bound` and `delta <= bound`,
- **THEN** the server admits the face on that clause, and the PPL delta
  value is recorded in the startup log.

#### Scenario: PPL clause present and out of bounds

- **WHEN** the gate record declares a `ppl` clause with `delta` and
  `bound` and `delta > bound`,
- **THEN** the server refuses the face with a typed error naming the PPL
  clause violation.

#### Scenario: PPL clause absent is admitted with a WARN

- **WHEN** the opt-in is set and the model's captured gate record declares no
  PPL clause,
- **THEN** the server admits the face on the [SA] §4.3 metrics and
  logs the absent clause at WARN level; no PPL value is invented (the
  clause's only producer is a real-model eval run requiring an explicit
  GO, and a fabricated number would be worse than the gap).

### Requirement: Quality gate provenance is shipped and reproducible

The gate record (activations source, per-layer metrics, PPL delta, date,
code commit) SHALL be shipped with the model directory and regenerable by a
committed tool against the pinned [SA] §4.3 metric definitions. The
`activations_kind` field SHALL name the evidence class
(`"fixture-battery"` or a recognized real-activation source), so the
face's enabling evidence is auditable rather than asserted.

#### Scenario: Gate record regeneration matches

- **WHEN** the committed gate tool re-runs against the model directory and
  its captured activation fixture,
- **THEN** it reproduces the shipped record's metrics within f32
  round-off, and the server's startup gate accepts it.

### Requirement: Serving parity across the quantized attention face

Generation through the `gpu:sage` face SHALL be indistinguishable from the
exact scalar face at the serving contract: identical token streams on the
fixture pipeline, with the final logits agreeing inside the face's OWN
quality gate ([SA] §4.3 — Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02). The
INT8 face is lossy by design, so the exact face's reassociation budget
(rel ≤ 1e-5) SHALL NOT be applied to it.

#### Scenario: Quantized-face token identity on the fixture pipeline

- **WHEN** the fixture transformer generates greedily through the exact
  scalar face and through the `gpu:sage` face over identical inputs,
- **THEN** the token streams are identical and the final logits agree at
  the [SA] §4.3 gate (Cosine Sim ≥ 0.999, Relative L1 ≤ 0.02).
