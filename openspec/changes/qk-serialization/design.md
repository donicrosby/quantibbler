# Design: qk-serialization

## 2026-10-25 — dependency decision: pydantic v2 (foundation card `t_0a8f5936`)

**Decision.** Add `pydantic>=2.14,<3` as a core dependency of the Python
compressor package (`src/quantibbler/`).

**Freshness (config.yaml rule, verified against the live registry, never from
memory).** `https://pypi.org/pypi/pydantic/json` → latest release `2.14.1` on
2026-10-25; `pydantic-core` latest `2.50.1`. pydantic is a 2.x package, so the
rule ("within ONE major version of the latest stable release") admits the
`>=2.14,<3` floor this change pins. `uv lock` resolved `pydantic 2.14.1` +
`pydantic-core 2.50.1` + `annotated-types 0.8.0` + `typing-inspection 0.4.4`.

**Build-graph check (config.yaml's C/C++ ban).** `pydantic` is pure Python;
the compiled half lives in `pydantic-core`, which ships as a **prebuilt Rust
wheel** (abi3). Rust is the sanctioned systems path in this repo ("If systems
speed is needed, it is Rust"), so this adds **no** compiled-from-C/C++ artifact
to the build graph. Nothing here needs a C compiler, and no optional extra or
test tooling gains one.

**Informed assumption (evidence hierarchy class 2 — no paper source).**

- *Assumption:* the resolved `pydantic-core` wheel for CPython 3.11-3.13 on
  linux-x86_64 exists and is a binary wheel, so no Rust toolchain is needed at
  install time.
- *Rationale:* PyPI's release files for `pydantic-core 2.50.1` list
  manylinux/musllinux wheels; `uv lock` resolved without a source build.
- *Wrong-assumption signature:* `uv sync`/CI falls back to an sdist and needs
  `cargo`. The install log would show `Building pydantic-core` instead of
  `Downloaded`; that is a build-graph change and must be re-ruled before it
  lands. Observed here: `Downloaded pydantic-core`, i.e. the assumption held.

## Model design

- **One model per persisted surface**, in one torch-free module
  (`quantibbler.serialization`) for the same reason `qk_report` and `qk_bar` are
  torch-free: the driver imports torch at module top, so anything kept there
  gets no lean-tier coverage.
- **Docstrings are citations, not descriptions.** Each model names the spec
  section / change doc that already owns its shape. The models are the
  executable form of those contracts; nothing here mints a new format.
- **`extra` is chosen per surface, by whether the writer emits a closed set:**
  - `extra="forbid"` — the acceptance-ledger row (a spec'd, enumerated field
    list) and the stage-journal entry (exactly `{event, info}`).
  - `extra="allow"` — the quantize report and the solver-probe row, whose
    writers emit keys beyond the modelled contract today. Dropping them would
    be a silent misparse; preserving them is `pydantic`'s `extra="allow"`, and
    the test pins that an unmodelled key survives a roundtrip.
- **Version policy.** Each model carries `schema_version` (default
  `SCHEMA_VERSION`). An *unknown* version refuses with `ArtifactSchemaError`
  naming the model and the version — never parsed as the current shape. An
  *absent* version reads as the current one, so the corpora already on disk
  (`artifacts/*/*.jsonl`, the stage journals) keep loading; that is the
  "old artifacts either load or refuse loudly" policy, and this change writes
  no producer, so no on-disk byte changes.
- **Single source of truth for vocabularies.** The ledger's `targeted_row`
  labels are read from `qk_bar.BAR_ROWS` at validation time (a second hand-kept
  list would be exactly the drift `qk_bar`'s docstring forbids), and the
  three-value verdict band is pinned equal to `qk_report`'s `OK`/`DEGRADED`/
  `FAILED` by a test.

## Discarded alternatives

- **`dataclasses` + hand-written `from_dict`/`to_json`.** That is the status
  quo with more code: no coercion rules, no shape citation, and every consumer
  re-derives the shape. Discarded — it doubles the surface it replaces.
- **JSON Schema documents + `jsonschema` validation.** Gives a checkable
  contract but no Python type, so writers still hand-assemble dicts and the
  schema is a second artifact that drifts from the code. Discarded for the same
  reason the module exists.
- **`msgspec`** (faster, also Rust-backed). Discarded: it is a 0.x package
  (fresher-floor pressure) and its model layer is thinner than pydantic's; no
  measured need justifies trading the larger ecosystem.
- **Attach `schema_version` by folding it into a digest / a side-car file.**
  Discarded on the standing ruling behind `config_fingerprint`: provenance is
  recorded *beside* the digest, never folded into it, and the version belongs
  on the artifact line itself where a reader meets it.

## Handover note for the adoption cards (byte-compatibility)

The models must not, by themselves, change an emitted artifact. Two consequences
the adoption cards (`t_b4a1383d`, `t_f0872f45`) must honour, recorded here so
they are not rediscovered:

- A model with optional fields dumps those fields as explicit `null`s unless the
  writer passes `exclude_none=True` (or `exclude_unset=True`). The report and the
  probe row are emitted to consumers today, so their writers must dump with
  `exclude_none=True` to stay byte-compatible — a roundtrip *through the model*
  is not the same claim as "the emitted bytes are unchanged".
- `extra="allow"` surfaces preserve unmodelled keys, so adopting the models on
  the report/probe writers does not silently drop a key a writer emits today.

## Open questions

- **OQ-1 — where do the *producer* adoption cards' spec deltas live?**
  Owner: planner. This change owns the models; the adoption cards
  (`t_b4a1383d` report/driver, `t_f0872f45` journal/probe) wire writers. If
  adoption forces a consumer-visible byte diff it is a spec-delta event for the
  owning change, not a silent fix here.
- **OQ-2 — the probe-row divergence.** `block_recon._record_solver_probe` does
  not carry `min_eigs` while `tools/lane3_reg_sweep.py` does (finding from
  `t_2522c684`). Resolving it is in the *adoption* card's scope; this model
  therefore makes `min_eigs` optional and models only the fields both writers
  always emit, so neither row is refused. NOT resolved here on purpose.

## Implementation crystallization

### 2026-10-25 — the report writer validates; it does not dump (adoption A, `t_b4a1383d`)

**Finding.** The handover note above ("their writers must dump with
`exclude_none=True` to stay byte-compatible") is wrong for the report, and
following it would have broken a different contract. `quantize_report.json`'s
key set is itself specified: the abort, stage-failure and completion paths must
carry IDENTICAL keys, with the unmeasured end-of-run numerics present as
explicit `null` (`qk-ptq`, scenario `report_single_shape_both_paths`). Dumping
from `QuantizeReport` with `exclude_none=True` drops exactly those nulls;
without it, the model adds a `null` for every declared-but-absent field
(`report_error` on a completion report) and reorders the keys into field
definition order.

**Ruling.** `dump_report` VALIDATES through `QuantizeReport` and serializes the
writer's own object with `schema_version` stamped in front. Concretely: same
keys, same value shapes, same relative order, `schema_version` the only added
key. A model rejection falls into the same minimal failed report an
unserializable value already produced (`faithful=False`, so the caller fails the
process), and that stub carries `schema_version` too — "every emitted
`quantize_report.json` names the shape it was written as" then has no exception,
which matters precisely because a reader that cannot decide has to refuse
loudly.

**Mechanism notes.** The import is function-level: `serialization` imports
`qk_report` (the band constants and the stage-journal shape table), so a
module-level import would be a cycle. Sanitization (`json_safe`) runs BEFORE
validation, so the model sees Python scalars — the driver's numbers come off
numpy arrays, and an `np.float32` is not the `float` the contract names, so
validating first would turn a healthy report into a rejection.

**Not adopted, and why.** The acceptance-ledger row has no producer to adopt:
`docs/acceptance-ledger.jsonl` is authored by a human at gate time and
`ledger-format.md` rule 2 states no pipeline code writes or reads it, so wiring
the row model is deferred to the first acceptance fire (an @user request), not
code that can be written now. `row_meets_bar` keeps its zero-production-caller
status — it is the ledger's predicate, never a verdict input.

**Superseded wording:** the bullet above beginning "A model with optional fields
dumps those fields as explicit `null`s unless the writer passes
`exclude_none=True`" — it holds for a writer that serializes FROM its model, and
neither the report writer nor (by the same key-set argument) the probe writer
may do that. Adoption B inherits this ruling.
