# Design: qk-artifact-dtype

Decisions D1–D12 below are the design-silent choices this change resolves;
each carries its evidence so the implementer and reviewer can audit the
choice instead of relitigating it. Decision D1 (the extension-vs-bump
ruling) is dated 2026-10-25, as the program brief requires. Grounded
facts: the remainder is
hardcoded fp32 in the loader (`read_f32_bin_exact`, loader.rs:237/241/272;
`checked_bytes` multiplies by 4, loader.rs:366), the `weight_tying` grammar
accepts exactly `lm_head=<path>@fp32` (loader.rs:552-561), and the manifest
`[sidecar]` block names the three remainder files (loader.rs:386-397). The
`ScaleDtype` enum (lib.rs:40) declares the per-layer stream scales; the
remainder has no equivalent.

## D1 — Extension, not a version bump (byte-layout evidence)

The remainder files carry NO self-describing header; their dtype lives only
in the text manifests (`config.toml` `weight_tying`, `manifest.toml`
`[sidecar]`). The declaration is therefore additive text, not a layout
change, and the decision is:

1. **Every artifact the current pipeline can produce carries an explicit
   `@fp32`.** The writer always emits a `weight_tying` line
   (quantize_hf_model.py:597-601); the grammar refuses anything else. The
   `@fp32` form keeps its exact meaning under the new grammar — current
   artifacts declare fp32 and load byte-identically.
2. **The loader's byte math is unchanged for fp32.** `checked_bytes` (x4)
   and `read_f32_bin_exact`'s exact-length stat gate remain the fp32 path;
   the new reader branches only on a declared bf16.
3. **A pre-change loader cannot silently misparse a bf16 artifact**, by two
   independent refusals: (a) the `weight_tying` grammar rejects the unknown
   `@bf16` suffix at config validation, before any file is read; (b) even
   with `weight_tying` unset, the exact-length check computes `floats x 4`
   while a bf16 file is `floats x 2` bytes — "expected exactly N bytes"
   refusal. Silent misparse is structurally impossible.
4. **A version bump would refuse every pre-change artifact.** Mandating the
   declaration everywhere would reject the v6 artifact and every committed
   fixture (none carry the new key), violating the standing constraint that
   v1/v2 artifacts remain readable. The bump alternative is therefore
   discarded on evidence, not preference.

Consequence: **absence of the new key is a defined legacy rule (fp32)**, not
a refusal. The brief's "unknown/undeclared → loud refusal" applies to
present-but-unknown declaration tokens and to cross-declaration disagreement
— the cases where a reader could otherwise guess — never to field absence,
whose only possible reading is the historical byte contract.

## D2 — Declaration surface

- `manifest.toml` `[sidecar]` gains an optional `dtype` key with values
  `fp32` and `bf16`. Absent = fp32 (D1 rule). Unknown value fails the TOML
  deserialize with the manifest named.
- `config.toml` `weight_tying` accepts `lm_head=<path>@{fp32,bf16}`; the
  tied sentinel form extends to `lm_head=e1@bf16`.
- **Agreement rule**: when `weight_tying` is present, it SHALL carry a
  dtype suffix (`@fp32` or `@bf16` — a suffix-less value refuses); the
  suffix SHALL equal the `[sidecar]` dtype; a mismatch is a named refusal
  naming both. An unset `weight_tying` with a bf16 `[sidecar]` is legal
  — the head file is read at the declared dtype like the other two.
- The new writer ALWAYS emits the `dtype` key (fp32 included): new artifacts
  are fully declared, absence remains the legacy reading for pre-change
  artifacts.

## D3 — One dtype for the whole remainder set

Embedding, lm_head (when untied), and the norm sidecar share one declared
dtype. The repack's scope (the whole remainder) and the brief's singular
"dtype" language support it; per-tensor dtypes would add grammar surface
with no product need (see discarded alternatives).

## D4 — bf16 encoding and cast semantics

- Storage: IEEE bfloat16 = the top 16 bits of IEEE float32 (1+8+7 bits),
  little-endian — 2 bytes per element.
- Cast: round-to-nearest-even. Per-element error |bf16(x) − x| ≤ 2^-8|x|
  for normals (half an ulp; relative ulp is 2^-7 for 8 significant bits);
  below bf16's subnormal range (min subnormal 2^-133) the result flushes to
  zero, so the bound carries an absolute floor: ≤ 2^-8|x| + 2^-134 (half
  the min-subnormal spacing).
- **The cast is a stage**: ONE shared conversion helper (used by both the
  exporter's bf16 path and the repack utility — DRY on a production
  primitive), with a post-cast finiteness gate: a non-finite source value is
  a named refusal, never silently shipped. The fp32 export path is untouched
  (byte-identical output).
- The helper converts through the runtime the pipeline already evaluates in
  (torch.bfloat16 — the numerics regime's model-side dtype), avoiding a new
  dependency.

## D5 — Loader internals

- New `RemainderDtype { Float32, Bf16 }` enum (itemsize 4 / 2) beside
  `ScaleDtype` in lib.rs.
- `read_f32_bin_exact` becomes dtype-parameterized: expected bytes =
  inventory-derived floats x declared itemsize; the stat length must match
  before any read; the refusal names the declared dtype and the expected
  byte count.
- `ModelDirectory` gains `remainder_dtype` (informational; values are still
  widened to f64 exactly — bf16→f32→f64 is exact, same as the existing
  f16/f32 widening).

## D6 — Writer

- `tools/quantize_hf_model.py` gains `--remainder-dtype {fp32,bf16}`
  (default fp32), mirrored on the export path (same plumbing shape as
  `--scale-dtype`).
- fp32 default: remainder file bytes identical to the pre-change writer
  (pinned by regression); the manifest gains the explicit `dtype = "fp32"`
  line (D2).
- bf16 opt-in: files named with the dtype token (`e1.bf16.bin`,
  `head.bf16.bin`, `norms.bf16.bin` — a convention echo, never the
  declaration authority), `[sidecar] dtype = "bf16"`, `weight_tying` suffix
  `@bf16` (agreement with D2).
- The export self-check (quantize_hf_model.py:658) validates the sidecar
  byte length at the DECLARED itemsize so a miscount fails at export, not at
  server load.

## D7 — Repack utility

- `tools/repack_remainder.py --in <dir> --out <dir>`: converts the remainder
  fp32→bf16 into a NEW directory.
- Refusals (all named): source declares a dtype other than fp32;
  non-finite remainder values.
- Non-remainder files (layer streams, tokenizer files) are copied
  byte-identically (asserted, not assumed).
- Output manifest: `[sidecar] dtype = "bf16"` + a `[repack]` table carrying
  `source_model_id`, `conversion = "remainder fp32 -> bf16"`, and
  `source_remainder_sha256` (sha256 over the source's embedding, head, and
  norms bytes in canonical order). Content identity, not wall-clock time:
  identical sources produce identical output bytes (determinism is the
  future `qk-artifact-blob` rule already honored here).
- Lossy-by-declaration: the `[repack]` record IS the lossy declaration.
- The tied head rewrites to the `lm_head=e1@bf16` sentinel; an untied head
  to `lm_head=<path>@bf16`.

## D8 — Report fidelity

`qk_report.provenance()` gains `remainder_dtype` (from the run's arguments,
default "fp32"); `regime_artifact` and `dtype_note` strings name it. Emitted
from arguments, never hardcoded — the same discipline that removed the
hardcoded float16-scale string. The `qk-serialization` change owns the
pydantic model adoption; the implementer reconciles this new field against
its landed models.

## D9 — Honest bf16 tolerance (do NOT borrow 1e-12)

The float64 1e-12 parity tolerance governs Rust-kernel-vs-NumPy comparisons;
it is meaningless for bf16, whose format floor is ~2^-3 relative per
element. Two bounds, both derived from the format:

- **Cast-level (exact math)**: |bf16(x) − x| ≤ 2^-8|x| + 2^-134 per element
  (D4). This is a theorem of RNE casting, tight; pinned elementwise in host
  tests.
- **Decode-level (accumulation-aware)**: a logit is a dot product of
  d_model terms through the norm chain, so the honest bound sums the
  artifact's OWN tensor magnitudes: |Δlogit| ≤ 2^-8 · (Σ_j |w_j x_j| +
  Σ_blocks c_b), where c_b bounds each block's norm-path contribution
  (≤ 2^-8 · sites · max|intermediate|). The acceptance test computes the
  bound from the loaded fp32 tensors — never from a magic constant. On the
  committed fixture (n_layers=1, d_model=4, vocab=32, 7 remainder tensors of
  O(1) magnitude) the formula instantiates to **2^-4 · max(1, max|logit|)**:
  per-element 2^-8 x ≤ 2^4 accumulation sites = 2^-4. Measured deltas are
  expected ~1e-3 — the bound carries ≥ 8x headroom while staying
  mathematically honest. The v6-artifact repack acceptance uses the general
  formula computed from that artifact's tensors.
- The decode comparison SHALL also assert the delta is NONZERO: if the
  loader silently read the bf16 file at the wrong itemsize the deltas would
  blow through the bound by orders of magnitude — the bound itself is the
  anti-vacuity guard, the nonzero assertion proves the variants differ at
  all.

## D10 — Fixtures

- `tools/gen_server_fixtures.py` gains a deterministic bf16 twin of the
  committed server-model fixture (same model, remainder in bf16, committed
  bytes). Cross-face byte identity: the Python writer's bf16 output equals
  the committed fixture bytes; the Rust loader's widened f64 values equal
  pinned literals in BOTH faces (the bias-carry v2-vector pattern).
- Refusal battery (Rust + host): unknown dtype tokens (`f8`, `fp16`, `b16`,
  empty), no-suffix weight_tying value, suffix/key mismatch (both directions), bf16-declared file at fp32
  byte length and the reverse, unset-tying bf16 acceptance.
- Backward-compat pins: the committed fp32 fixture loads with identical
  values before/after; a synthetic pre-change manifest (no `dtype` key)
  loads as fp32 (the D1 absence rule).

## D11 — Backward-compat pins

In addition to the fixture pins (D10): the change SHALL NOT alter the layer
streams (qk-representation untouched), the fp32 sidecar math, or any
existing artifact. The v6 fire artifacts are never read, copied, or written
by this change's committed test suite; a v6 repack acceptance, if the
orchestrator/@user selects that branch, runs out-of-suite against a
staged copy, never the live fire's mount.

## D12 — Non-goals (see proposal; the hard edges)

No layer-stream dtype change (kernels-dtype-generic owns genericity), no
other remainder dtypes, no bf16→fp32, no in-place repack, no re-fire, no
blob container.

## Discarded alternatives

- **Version bump refusing undeclared artifacts**: discarded — evidence D1.4
  (violates mandated v1/v2 readability).
- **Declaration in `config.toml` as a new `remainder_dtype` key instead of
  the manifest `[sidecar]`**: discarded — the `[sidecar]` block already
  names the three files, and the dtype belongs to the files' declaration;
  the `weight_tying` suffix remains the head's historical declaration site,
  so both sites already exist and the agreement rule keeps them
  single-sourced.
- **Per-tensor dtype declarations**: discarded — D3.
- **Dropping the `@` suffix from new artifacts**: discarded — the suffix's
  presence is what makes pre-change loaders fail closed on bf16 (D1.3a).
- **bf16 conversion via a new dependency (e.g. ml_dtypes)**: discarded — the
  pipeline already evaluates bf16 through torch; one shared helper on the
  existing runtime (D4).
- **Reusing the 1e-12 parity tolerance for bf16 comparisons**: discarded —
  D9; the format floor is ~2^-3 per element.

## Open Questions

None open. Every design-silent choice above is resolved with its evidence
(D1–D12). One execution-time branch is pre-resolved by the program brief:
the repack acceptance target is the v6 artifact if its directory is stable
and available at implementation time, else the committed fixture — the
fixture path is the standing alternative, and the choice is the
orchestrator's/@user's at implementation, not a spec gap.

## Implementation crystallization

### 2026-10-26 — the repack rewrites BOTH manifests; RDP-1.4 scopes byte-identity to the copies

Resolved during implementation. `tools/repack_remainder.py` rewrites
`manifest.toml` (the declaration and the `[repack]` record) AND `config.toml`
(`weight_tying`'s dtype token, D7). The second is forced, not chosen: leaving
`@fp32` beside bf16 bytes IS the cross-declaration disagreement qk-server
RD-1.3 refuses, so an unrewritten `config.toml` would make RDP-1.1 ("the output
directory loads") unsatisfiable. The requirement body already draws the line —
"layer streams and tokenizer files SHALL be copied byte-identically ... the
manifests rewritten" — so RDP-1.4's THEN ("every non-remainder file ...
byte-identical") reads the copied set, not the rewritten one. The tested
contract pins both halves: every copied file is byte-identical to its source
counterpart, and `config.toml` differs in exactly the `weight_tying` token.

Finding for the planner (doc wording, not a design change): RDP-1.4's THEN
would be unambiguous as "every file the repack copies".

### 2026-10-26 — the fp32 reader is generalized, not kept under its old name

D1.2 and D5 phrase the fp32 path as `read_f32_bin_exact` + `checked_bytes`
"remaining"; the implementation folds both into ONE dtype-parameterized
`read_remainder_exact(path, floats, dtype, config_path)` and deletes the two.
The byte math is exactly what D1.2 protects, and it is unchanged — floats x
declared itemsize, the same exact-length stat gate, the same refusal on any
mismatch (both directions) — but a reader that also reads bf16 must not keep
the name `read_f32_bin_exact`: the name is all a caller sees, and it would
promise a width the reader no longer assumes. D5's own wording ("becomes
dtype-parameterized") is the version kept.
