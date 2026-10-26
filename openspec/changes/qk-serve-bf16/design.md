# Design: qk-serve-bf16

## Context

The server currently loads directory-format artifacts through
`quantibbler_qpk1::loader::load` (directory traversal, 148 lines) and serves
at f32 precision throughout. The parent changes deliver:

- **qk-kernels-dtype-generic**: dtype-generic `packed_matvec` / `packed_matmul`
  with f32 accumulator for bf16 operands, bf16 backward tolerance 1e-2 of
  row term scale, PyO3 `*_dtype` entry points, `Dtype` tag resolution.
- **qk-artifact-blob**: single-file `.qpk` blob with `QKPB` magic, JSON
  header carrying `remainder_dtype`, 4K-aligned section table, binary dtype
  tags per section (`fp32_le=2`, `bf16_le=3`).

The server must consume both: load the blob, read the dtype, and route
serving computation through the dtype-generic kernel face.

## Decisions

### S1 — CLI knob: `--artifact <path>` over `--model-dir`

Chosen: a single `--artifact` flag accepting a path; the loader auto-detects
blob (`.qpk` file) vs directory. `--model-dir` becomes a deprecated alias.
Discarded: two separate flags (`--blob` / `--dir`) — the auto-detection is one
`is_file()` check, and two mutually-exclusive flags complicate the CLI parsing
and the help text. The user's intent is "serve this artifact"; the format is
the loader's concern.

### S2 — Mmap load path

Chosen: `memmap2::Mmap::map(&file)?` for the whole blob, then
`sliced[offset..][..len]` for each section — zero-copy where the kernel
page size permits (4K alignment is guaranteed by the blob spec B4).
Fallback: when `memmap2` is unavailable (a platform the crate doesn't target),
or when the file lives on a network filesystem that rejects mmap, fall back
to `std::fs::read` for the whole file with a WARN log naming the reason.
Discarded: always-read — loses the zero-copy benefit that was the whole
point of 4K alignment. Discarded: per-section mmap — the blob is one file;
mapping the whole file once and slicing is the simplest correct shape.

### S3 — Remainder dtype resolution

The blob header's `remainder_dtype` is REQUIRED (blob spec B2). On load the
server reads it and resolves a `ServingDtype` enum (`Fp32` / `Bf16`).
Missing → loud refusal (never defaulted). Unknown value → loud refusal
(the blob spec allows only `"fp32"` / `"bf16"`). The dtype is stored on
`ModelWeights` so every consumer reads one source of truth.

The section table's per-section dtype tags (`fp32_le=2`, `bf16_le=3`) are
the AUTHORITATIVE source for each blob section's byte width. The header's
`remainder_dtype` is a convenience declaration; the section table is what
the loader reads to size each read. A mismatch between header declaration
and section table (e.g. `remainder_dtype: "bf16"` but the `embedding`
section is tagged `fp32_le`) is a load-time refusal — the agreement rule
from qk-artifact-dtype D2 reproduced verbatim.

### S4 — Engine dtype dispatch

The engine currently hardcodes f32 for every float operation:
`PackedLayer.apply` accumulates in f64 then casts down to f32 for
scale/bias, norms are f32, attention is f32, the logits dot-product is f32.

The change: `ModelWeights` carries `serving_dtype: ServingDtype`. Where
`ServingDtype::Fp32`, every path is the current path (zero code change).
Where `ServingDtype::Bf16`:

- **Remainder tensors** (embedding, lm_head, norms) are loaded as bf16 bytes
  and stored as `Vec<half::bf16>` on `ModelWeights`. Conversion at load:
  `bf16::from_bits(u16::from_le_bytes([b0, b1]))`.
- **Packed matvecs** route through the dtype-aware entry point:
  `core::packed_matvec_dtype(packed, n_rows, n_bits, vec, Dtype::Bf16)`.
  The kernel accumulates in f32 and returns f32. The caller-side
  scale/bias application stays f32 — scales are f32 because this
  change's blobs and fixtures carry tag-2 (f32) scale streams
  (qk-representation SF-1.2: tag 1 = f16, tag 2 = f32; the stream's
  tag selects, nothing mandates)
- **Norm application** (RMSNorm): operands are bf16, computed in f32,
  result cast to bf16 — matching the kernel's accumulator convention.
- **Attention**: Q/K/V projections produce bf16 operands via the matvec
  path above; RoPE and score computation in f32 (matching the f64→f32
  downcast the fp32 path already does — no new precision loss).
- **Logits head**: dot-product over bf16 hidden × bf16 head rows,
  accumulated in f32, result f32 (sampler is argmax, so precision
  differences only matter at the token decision boundary — parity test
  covers this).
- **KV cache**: stored as f32 (the cache is a working buffer, not the
  artifact's dtype; fp32 keeps the softmax numerics stable).

The dispatch point: `PackedLayer.apply` takes `serving_dtype` and calls
the dtype-aware kernel entry point. The norm and attention paths read
`self.weights.serving_dtype` at their call sites.

### S5 — Chat template

`POST /v1/chat/completions` receives a `messages` array of
`[{role, content}]` objects. Before tokenization the server SHALL apply
the model's chat template (Qwen3) to produce a single prompt string.

Implementation: the template string (the genuine Qwen3 `chat_template`
from the HF export, provenance-pinned) is embedded in the fixture
tokenizer.json; the server reads it from the loaded tokenizer JSON via
serde_json (the `tokenizers` crate does not expose arbitrary metadata
fields — verified against the vendored 0.23.2 source: zero occurrences
of `chat_template`/`ChatTemplate`). Rendering SHALL use `minijinja`
3.0.3 (latest stable per the live crates.io registry 2026-10-26; pure
Rust with ZERO mandatory transitive deps — the C/C++ ban stays intact)
with a pinned render context: `messages`, `add_generation_prompt=true`,
`enable_thinking=false`. The rendered prompt string is then tokenized
with the loaded BPE.

Verification: a committed golden file generated once by the Python HF
stack with the SAME pinned kwargs (`chat_golden.json` containing the
rendered prompt string and the token IDs from the committed fixture
tokenizer.json) serves as the byte-level parity contract; the Rust test
asserts both rendered string and token IDs against the golden. If the
golden test later reddens due to template-field evolution, the golden
regeneration is a committed provenance-tracked operation.

Discarded: tokenizers-based template application — verified against the
vendored `tokenizers 0.23.2` source (zero matches for
`chat_template`/`ChatTemplate`; feature table has no `chat-template`
entry) and against the live registry (max_stable 0.23.2; the only newer
release is 1.0.0-rc.3 whose feature table still carries no chat-template
feature) — the mechanism does not exist in any published tokenizers
version. Minijinja is the only verifiable pure-Rust mechanism.

### S6 — Prefill timing

Non-streaming responses SHALL include a `timings` block carrying
`prompt_ms` (integer milliseconds, elapsed wall-clock from the start of
the FIRST prefill forward through the end of the LAST prefill forward).
The existing `Usage` struct gains a `timings` field with `prompt_ms`.
Tokenization and template application SHALL be excluded — equivalent to
llama.cpp's `timings.prompt_ms`, so PP t/s computed as
`prompt_tokens / (prompt_ms / 1000.0)` is identical across comparison
legs (t_72781a70 addendum).

Measurement: `Instant::now()` before the first prefill call in
`decode_start`, another after the prefill loop; the difference in
milliseconds is `prompt_ms`. Streaming requests do NOT report `timings`
(the SSE stream has no post-hoc metadata channel). The benchmark
program's `/v1/completions` path also gains the timing block.

Discarded: per-token timing — the llama-bench PP metric is
prefill-tokens / prompt_ms, which needs only the aggregate prompt time;
per-token detail is the bench program's concern, not the API surface.

### S7 — Parity contract

An fp32 artifact and the bf16 repack of the same model SHALL produce:

1. **Temperature-0 token-stream equivalence** on a fixed probe set (the
   committed test prompts). The first differing token is a parity failure.
2. **Final prefill logits are finite** on both legs and the worst/median
   elementwise absolute deviations |Δ| = |L_bf16 − L_fp32| are reported
   (no per-linear intermediate capture; per-linear backward error is owned
   by qk-kernels-dtype-generic). Per-linear backward-error gating (BS-3.2)
   stays a unit-level matvec dispatch test on a single layer.
3. **Must-differ guard**: the bf16-leg prefill logits are NOT bitwise
   identical to the fp32-leg logits. Bit-identity means native bf16
   serving never executed (widen-at-load fell through) — fail loudly. The
   fixture twin SHALL carry remainder tensors with values not exactly
   bf16-representable (provenance recorded in the fixture commit) so the
   guard is falsifiable.

### S8 — Directory load preserved

The existing `model::load(model_dir: &Path)` stays functional and follows
the sibling's declared-dtype widening contract (qk-artifact-dtype
RD-1/RD-2): a directory declaring bf16 is legal, its remainders widen
exactly to f64 at load, and the engine serves through the pre-change
fp32/f64 path. The directory path SHALL NOT select `ServingDtype::Bf16`;
native bf16 serving (this change's new kernel dispatch) is reachable only
through the blob loader. The blob loader is a new `load_blob(artifact:
&Path)` entry point; `model::load` dispatches to the right path based on
`is_file()`.

### S9 — Fixture pair

The committed test data gains three items for this change:

1. A **bf16 blob twin** of the server-model fixture: produced by
   `tools/repack_remainder.py` from the fp32 blob, committed with
   provenance (source blob sha256, repack command). The fixture remainder
   tensors SHALL carry values NOT exactly representable in bf16 (e.g.
   0.1f32) — provenance recorded — so the must-differ guard is
   falsifiable (BS-9.3).

2. The **genuine Qwen3 chat_template string** installed in the fixture
   tokenizer.json (provenance-pinned to the HF export, source revision
   recorded).

3. A **committed golden directory** containing `chat_golden.json` with
   the rendered prompt string and token IDs generated once by the Python
   HF stack from pinned probing messages (`[{role: "user", content:
   "hello"}]`), `add_generation_prompt=true`, `enable_thinking=false`,
   against the committed fixture tokenizer.json (post chat_template
   installation). The Rust chat-template test (BS-5.1) asserts both
   rendered string and token IDs against this golden.

Both blobs are committed; the parity test loads both and asserts token
equivalence (BS-9.1) plus the must-differ guard (BS-9.3).

## Discarded alternatives

- **Copy-on-load bf16→f32 widening**: discarded — the whole point is
  serving bf16 natively. Widening at load makes the artifact dtype
  declaration a lie and consumes 2x the working-set memory.
- **Per-tensor dtype unions (enum over f32/bf16 on every tensor)**:
  discarded — every remainder tensor shares the same declared dtype;
  an enum per tensor is noise and prevents unified code paths.
- **LLM-foundry-style chat-template crate**: discarded — the `tokenizers`
  crate does NOT contain `apply_chat_template` (verified: zero
  occurrences in the vendored 0.23.2 source; no such feature in any
  published version including 1.0.0-rc.3). Minijinja is the minimal
  pure-Rust mechanism for rendering one pinned template string; a
  full chat-template crate would be weightier and the template-drift
  risk is covered by the committed golden.
- **Timing in the SSE stream**: discarded — SSE has no trailer channel;
  a meta-chunk (`data: [DONE: ...]`) breaks OpenAI compatibility. The
  bench program measures wall-clock externally for streaming legs;
  the API surface carries timing only on non-streaming.

## Open Questions

| question | status | owner |
|---|---|---|
| Should the KV cache dtype be selectable (bf16 cache for bf16 artifacts)? | resolved — no; f32 cache keeps softmax stable; cache dtype is an independent performance decision, not a serving-contract requirement | this change |
| Should `/v1/models` advertise the serving dtype? | resolved — yes; the model entry gains a `dtype` field | this change |
| Should `enable_thinking` be request-configurable? | resolved — no; pinned false for this change; the committed golden uses pinned false and the surface does not expose the knob | this change |

## Informed assumptions

1. **bf16 operand f32-accumulate is enough for token-parity generation.**
   The kernel's measured bf16 backward error (1.86e-3 matvec worst at
   authoring time, ~5.4x inside the 1e-2 bound) is per-linear; the
   accumulated error across 28 layers with intervening norms and
   attention could compound. The parity test is the falsification
   mechanism: if temperature-0 tokens diverge, the assumption is wrong
   and a per-layer error trace becomes the diagnostic.

2. **memmap2 works on every platform the server targets (Linux x86-64,
   ARM64).** The crate is pure Rust with no C dependency and is already
   widely adopted. A platform where mmap is unavailable falls back to
   read; the fallback is a WARN, not a crash.

3. **Minijinja renders the Qwen3 template identically to Python Jinja2.**
   The committed golden file (Python HF stack, pinned
   `add_generation_prompt=true`, `enable_thinking=false`) is the
   byte-level proof: the Rust minijinja render and tokenization must
   match the golden. If minijinja diverges from Jinja2 semantics on the
   Qwen3 template, the golden test reddens and the mechanism must be
   re-evaluated. The template string is the genuine Qwen3 artifact
   (model data from the HF export), not implementation-sourced per the
   clean-room rule.

## Tolerance reference (from qk-kernels-dtype-generic)

- **float64 products**: rel 1e-12 + bit-identical accumulation — unchanged.
- **f32 operand products**: backward 1e-6 of row term scale Σ_k |v_k|.
- **bf16 operand products**: backward 1e-2 of row term scale.
- **Face parity** (rlib vs PyO3): exact, bit-identical.

## Implementation crystallization

### 2026-10-26 — loader/dtype/engine/prefill legs (coder, t_809b44f8, superseded card)

1. **The requirement names the PyO3 face's symbols; the rlib face has other
   names.** BS-3's text routes `PackedLayer.apply` through
   `packed_matvec_dtype` / `packed_matmul_dtype` with `Dtype::Bf16`. Those are
   the PyO3 (`lib.rs`) entry points. The server links the crate as an rlib and
   must not link libpython (`binary_purity`), so the equivalent rlib entry
   point is `dispatch::packed_matvec_dyn(packed, n_rows, n_bits,
   DenseOperand::Bf16(v))`, which is where the dtype is carried as data and
   `DenseOperand::dtype()` resolves to `Dtype::Bf16`. The engine calls that
   one; no second name is introduced for the same instantiation.
2. **The server owns the mmap; qpk1 owns the rules.** BS-1/Task 1 put the
   mmap-with-fallback in `model::load_blob`. Rather than re-deriving the
   section rules, `quantibbler_qpk1::loader::build_from_blob` became `pub`
   (one visibility change) so the server composes the phases: metadata prefix
   from the open file (BL-4.2 ordering intact — no map is requested until the
   header, padding, table and 4096-alignment assertions pass), then the byte
   source, then `validate_payload`, then the shared composition. `memmap2` is
   therefore a server dependency (same `0.9` line as qpk1's).
3. **S4's `Vec<half::bf16>` remainder storage is implemented as f32 storage
   holding exactly the declared bf16 values, plus one operand rounding at each
   bf16 call site.** The loader (qk-artifact-dtype RD-2) widens every remainder
   section to f64 exactly; for a bf16-declared section the widened value IS the
   bf16 value, so `from_layers` quantizing it to bf16-valued f32 changes no bit
   and a second copy of every tensor buys nothing. The observable contract —
   "bf16 operands, f32 accumulation" — is delivered at the operand boundary:
   the packed matvec rounds `diag(s2)·x` once and passes `DenseOperand::Bf16`,
   and the norms round their operand and cast the result back to bf16. The
   logits head needs no change: both operands are already bf16-valued f32
   words, and an f32 product of two bf16 words is exact (8 + 8 mantissa bits
   < 24), so accumulating in f32 sums exactly the bf16 products BS-4 names.
4. **The `.max(1)` on `prompt_ms` is a documented floor, measured.** BS-6.1
   requires a POSITIVE integer; the fixture's prefill is sub-millisecond, so
   `as_millis()` alone would report 0 and assert "no prefill happened". The
   value is `elapsed.as_nanos().div_ceil(1_000_000).max(1)` — ceil to the next
   millisecond, floored at 1.
5. **The bytes-before-map ordering makes BS-1.2 unimplementable as a live
   fault.** No mount is available to make `memmap2::Mmap::map` fail, so the
   mmap-error branch is exercised by the test-only hatch
   (`QKB_TEST_FORCE_READ_FALLBACK=1`) which takes the SAME read leg and logs
   the fallback reason at WARN. The hatch is read only by the loader and set
   only from `tests/common/mod.rs` (BS-1.3's own words).
6. **BS-2.4's refusal comes from the SECTION TABLE, and the fixture must move
   `weight_tying` with the header to reach it.** A crafted blob whose header
   says `bf16` while `weight_tying` still says `@fp32` is refused by the
   two-declarations rule FIRST (BL-3.1/BL-3.2 in the blob loader). The test
   therefore rewrites both, so the refusal under test is
   `read_remainder_section`'s header-vs-tag mismatch, naming the section and
   both spellings.
7. **`api_timings` bounds `prompt_ms` by `wall_ms + 1`, not `wall_ms`.** The
   millisecond ceiling (entry 4) can legitimately report 1 for a request that
   completed in under a millisecond; the strict inequality is unconstructible
   at that granularity, and the scenario's testable core — a positive integer
   that does not exceed the request — is what the test asserts.

### 2026-10-26 — chat-template leg (coder, t_2b7e43f3)

8. **`tokenizers` REJECTS any unknown top-level key in `tokenizer.json`, so the
   template must leave the document before the BPE is built.** Design S5 embeds
   the template in the artifact's `tokenizer.json` and reads it back with
   serde_json, but it does not say whether the crate can still parse the file
   the template was embedded in — measured against the vendored `tokenizers`
   0.23.2 (and its Python face, 0.23.3): a REAL HF `tokenizer.json` plus ONE
   extra top-level key fails outright ("expected `,` or `}`"; reproduced with a
   key of any type and in any position), because the newtype `Tokenizer`
   deserialization does not tolerate unknown fields.
   `model::tokenizer_from_json` therefore parses the document once, takes
   `chat_template` out of it, and hands the crate the same document MINUS that
   key (re-serializing only when the key is present, so an artifact without one
   is passed through byte-for-byte). The template still lives in
   `tokenizer.json` and is still read via serde_json; only the view the crate
   receives is narrowed. `tools/gen_chat_golden.py` performs the same removal so
   the Python HF stack can load the same fixture.
9. **minijinja ships no built-in string methods; the Qwen3 template needs
   six.** The genuine template calls `content.startswith(...)`,
   `.endswith(...)`, `.split(...)`, `.strip(...)`, `.lstrip(...)` and
   `.rstrip(...)` — Python `str` methods Jinja2 gets from Python and minijinja
   does not provide (`Value::_call_method` answers `UnknownMethod` for every
   non-object receiver). They are registered through
   `Environment::set_unknown_method_callback`, with Python's semantics (a
   `strip` argument is a CHARACTER SET, not a substring), and every other name
   stays refused — so a future template that needs an unimplemented method
   fails loudly rather than rendering something subtly different.
10. **The chat-template golden is `tests/fixtures/chat_golden.json`, generated
    by `tools/gen_chat_golden.py`.** It pins the rendered prompt string
    (transformers' `apply_chat_template`, cross-checked in the same run against
    a direct jinja2 render) and the token ids (the Python face of the same
    `tokenizers` the server links, `add_special_tokens=False` to mirror
    `Engine::encode`'s `tokenizer.encode(text, false)`). No timestamp is
    written, so regeneration is byte-stable and a diff is a real behavioural
    change. The template's provenance — `Qwen/Qwen3-0.6B`
    `tokenizer_config.json`:`chat_template`, revision
    `c1899de289a04d12100db370d81485cdf75e47ca` — is recorded in
    `tools/gen_server_fixtures.py` beside the string, and the fixture's
    remainder tensors keep the not-bf16-representable values the must-differ
    guard needs (`0.1`-class norms, `(i % 7 - 3) * 0.25` embedding). Committed
    digests: `server-model/tokenizer.json`
    `c4216adbb3fc8dd28c8a877530477cf2d7302328048561fc44933e706c6730ed`,
    `server-model.qpk`
    `5a72bfb0c90d529650e251104bd5d6c355769d69c9b1a62a56c5336775ed2ba7`,
    `server-model-bf16.qpk`
    `9ac15c5e17eb2155ac9b4c9c3f7dd79b06147553abe43164606317e801fcf1e2`,
    `chat_golden.json`
    `10e978ad59d43bb9b2f2e6385459f7ba2ad9aac1336762cd60a0698113304f43`.
    Two consecutive generator passes reproduce the blobs and the golden
    byte-for-byte (`cmp` identical).
11. **The fixture generator's PRE-EXISTING drift is not part of this change.**
    Regenerating `tests/fixtures/**` with the shipped
    `tools/gen_server_fixtures.py` also rewrites four paths this change does
    not own: `server-model/attention_gate.json` and
    `server-model-bf16/attention_gate.json` (deleted — another generator
    writes them), `server-model-bad/symlink_layer/layers/q_proj.qpk1` (an
    ABSOLUTE symlink resolved against the generating checkout, hence
    path-dependent by construction), and trailing-newline-only rewrites of
    `server-model-bad/malformed_tokenizer/tokenizer.json` and
    `server-model-bad/missing_sidecar_block/manifest.toml`. Those five paths
    were restored to their committed bytes; the fixture diff for this change is
    the tokenizer documents and the two blobs. Reported so whoever next needs a
    full regeneration starts from a known state.
