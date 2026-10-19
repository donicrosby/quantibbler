# qk-flash-attention — Spec Delta (qk-server)

## ADDED Requirements

### Requirement: Attention face dispatch (frozen decision, soft fallback)

The server SHALL select the attention face ONCE at model load —
`cpu:scalar` or `gpu:tiled` behind the `gpu` feature and an env opt-in —
using the same frozen-decision pattern as matvec dispatch: the face never
swaps mid-stream, and `/health` SHALL report the active face alongside
`matvec_face`. Device init failure SHALL be a soft fallback to `cpu:scalar`
(both faces compute the same attention within the parity budget).

#### Scenario: Face selection and honest health labeling

- **WHEN** the server starts with the opt-in enabled and a usable device,
- **THEN** `/health` reports `"attention_face": "gpu:tiled"`, the label
  stays fixed for the process lifetime, and with the opt-in unset it reports
  `cpu:scalar` with no device touched.

#### Scenario: Soft fallback on init failure

- **WHEN** the opt-in is set but device initialization fails,
- **THEN** the server serves on `cpu:scalar`, `/health` reports the CPU face
  (never a stale GPU label), and startup does not abort.

### Requirement: Serving parity across attention faces

Generation through the GPU attention face MUST be indistinguishable from the
scalar face at the serving contract: identical token streams on the fixture
pipeline, with logits agreement inside the reassociation budget (rel ≤ 1e-5;
bit-identity is not required — this face reassociates, per the 3.5
crystallization).

#### Scenario: Cross-face token identity on the fixture pipeline

- **WHEN** the fixture transformer generates greedily through each face over
  identical inputs,
- **THEN** the token streams are identical and final logits agree with
  relative error ≤ 1e-5.
