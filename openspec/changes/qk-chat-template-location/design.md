# Design: qk-chat-template-location

## Context

Ground A of the v6b sanity verdict (`SANITY-VERDICT-v6b-20261027.md`) is a
file-location gap, not missing content: the artifact ships the template in
`chat_template.jinja` (transformers 5.20.1) while the server reads only
`tokenizer.json`'s `chat_template` field. This design rules on the two
candidate fixes the verdict posed and pins the canonical contract.

## The ruling

**The chat template's canonical location in a quantibbler artifact is
`chat_template.jinja`.** The loader owns the read contract (precedence below);
the exporter owns the guarantee that the canonical file lands non-empty; and
the `tokenizer.json` injection is REJECTED. Rationale, per consideration:

- **Which side owns the contract long-term — the compressor's output IS the
  contract, and the "server reads what the compressor writes" invariant
  (`config.yaml` context) is what we re-align with, not invent against.**
  The compressor's `AutoTokenizer.save_pretrained()` already emits
  `chat_template.jinja` with no code change on our side. The fix makes the
  server read the file the compressor already writes.
- **What llama.cpp / Lemonade do (bench-compare parity).** This is decisive.
  (a) HF transformers 5.x: `chat_template.jinja` is the *recommended*
  standalone format; `save_pretrained()` writes it by default; the
  `tokenizer_config.json` `chat_template` field is a *load-only legacy* format
  that save *removes* to avoid duplication; `tokenizer.json` (the fast
  tokenizer) never carries a chat template at all (Hugging Face docs,
  "Writing a chat template / Storing and loading chat templates", and
  transformers v5 MIGRATION_GUIDE). (b) AMD Lemonade's own server had this
  exact bug and fixes it in an open PR — `lemonade-sdk/ryzenai-server` PR #7
  "fix: load chat template from chat_template.jinja when available" (open). (c)
  llama.cpp's converter reads the HF tokenizer's `chat_template` (the property
  that resolves `chat_template.jinja` → `tokenizer_config.json`) and stores it
  in GGUF `tokenizer.chat_template` metadata; its server reads it at runtime,
  never from `tokenizer.json`. **Parity therefore demands we read the SAME
  bytes HF/Lemonade/llama.cpp read — `chat_template.jinja` — not a bespoke
  field.** This also kills the prompt-token-parity risk directly: if we read a
  different location than Lemonade reads, a template that differs between the
  two locations yields silently different prompts.
- **Dual-source drift risk.** This is the deciding argument against the
  exporter-side injection (option (i)). Injecting `chat_template` into
  `tokenizer.json` WHILE `save_pretrained` also writes `chat_template.jinja`
  creates two copies of the same template. The exporter would have to keep them
  equal forever; a future refactor of either write path silently diverges them,
  and the server (JSON-primary) would then render different bytes than
  Lemonade (jinja-primary) — the exact silent-drift failure a benchmark parity
  harness cannot see but loses the comparison to. ONE canonical source
  (`chat_template.jinja`) with a deterministic read precedence eliminates the
  second copy. The `tokenizer_config.json` field is kept ONLY as a legacy
  load-only fallback (never written by us), mirroring HF's own "legacy
  load-only, don't write" posture.
- **bf16 / native serving.** Orthogonal. The template is a UTF-8 string read
  once at load into `LoadedModel.chat_template: Option<String>`; it is a
  metadata field, not a float operand, so dtype dispatch
  (BS-2/BS-3/BS-4) is untouched. No generator-side or kernel-side change.

## Loader read precedence (canonical → legacy)

Resolved once at load into `LoadedModel.chat_template`:

1. `chat_template.jinja` — canonical. Directory source: the file
   `<model-dir>/chat_template.jinja`. Blob source: the blob's raw
   `chat_template.jinja` section (new, `qk-artifact-blob` raw-tag durable).
   A blank file yields `None`.
2. `tokenizer_config.json`'s `chat_template` string field — legacy
   load-only (pre-transformers-5). Not written by us.
3. `tokenizer.json`'s `chat_template` field — DEPRECATED last resort, kept
   solely so the three already-committed fixtures keep loading until this
   change regenerates them. Removal is a follow-up once fixtures are re-pinned
   and the field is gone from `tests/fixtures/`.

An artifact with none of these yields `chat_template = None`, and
`/v1/chat/completions` refuses 400 (existing behaviour, BS-5.2) rather than
guessing a prompt format. The existing `tokenizer_from_json` extraction of
`chat_template` is removed in favour of a dedicated
`resolve_chat_template` that reads the precedence above; `tokenizer_from_json`
returns only the tokenizer but KEEPS its document-level strip of the
`chat_template` key — tokenizers 0.23.2 rejects any document carrying the key
(empirically verified), so the strip stays load-critical. Only template
*extraction* moves to `resolve_chat_template`.

## Exporter guarantee (fail-closed)

After `tokenizer.save_pretrained(out_dir)` in
`export_model_directory`:

- Read `(out_dir / "chat_template.jinja")`. It MUST exist, be non-empty after
  trim, and its content MUST equal the loaded tokenizer's `chat_template`
  string (the driver loaded `AutoTokenizer.from_pretrained(args.model)`, whose
  `.chat_template` is the normalized template). On any failure, raise
  (refusing the export — no artifact) with a diagnostic naming the file and the
  mismatch. This converts "the template is usually there" into "the artifact
  is provably self-described".

The exporter does NOT write `chat_template` into `tokenizer.json` or into
`tokenizer_config.json` — deliberate, to keep exactly one source.

## Blob container

`write_blob_from_directory` currently packs `tokenizer.*` files only.
`chat_template.jinja` (does not start with `tokenizer.`) is therefore dropped
today. This change adds it: when `source/chat_template.jinja` exists, emit a
`Section(name="chat_template.jinja", tag=TAG_RAW, ...)` in the canonical order
after the tokenizer sections. The qpk1 crate's `build_from_blob` reads that
optional section (raw → UTF-8, lossy-invalid → refused with a diagnostic) into
a new `ModelDirectory.chat_template_jinja: Option<String>`, and the server's
`compose` consults it first. Blobs produced before this change fall back
through the precedence (their `tokenizer_config.json`/`tokenizer.json`
sections, if any).

## Fixtures and golden

`gen_server_fixtures.py` writes the pinned Qwen3 `chat_template` (same
provenance it already pins — `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`) to `chat_template.jinja` instead
of injecting it into `tokenizer.json`. `gen_chat_golden.py` loads the template
from `tests/fixtures/server-model/chat_template.jinja` instead of the
`tokenizer.json` field; the golden's render string and token ids are unchanged
(the template bytes are identical, only relocated), so `chat_golden.json`
prompt and token ids are byte-stable (the `chat_template_source` provenance
field in the golden is updated to the sidecar path). The new blob fixtures pack `chat_template.jinja`.

## Real-artifact regression leg

The fixture-only hole is closed two ways: (a) the regenerated fixtures
structurally reproduce the real shape (template in `chat_template.jinja`,
absent from `tokenizer.json`) so the always-run suite catches the gap; (b) a
gated integration test (`QKB_REAL_ARTIFACT_DIR` env-var, SKIP when unset)
loads a REAL exported artifact directory — the v6b artifact at
`artifacts/qwen3-0.6B-r512-admm-paperfaithful-v6b-pertarget`, read-only — and
sends the pinned probe message `[{"role":"user","content":"hello"}]` through
`/v1/chat/completions`, asserting (1) non-400 and (2) the rendered prompt
equals HF's `AutoTokenizer.from_pretrained(<artifact>).apply_chat_template(...,
tokenize=False, add_generation_prompt=True)` with `enable_thinking` off
(oracle: `tools/gen_chat_golden.py` with `ENABLE_THINKING=False`,
deterministically rendering the same probe message).
The gate for this change is run WITH the env-var set so the real-artifact leg
actually executes, not merely skips.

## Discarded alternatives

- **(i) Exporter injects `chat_template` into `tokenizer.json`.** Rejected:
  bespoke location no upstream reader (HF/llama.cpp/Lemonade) consults, and it
  introduces a second copy that can drift from `chat_template.jinja` (dual-source).
- **Loader reads `tokenizer.json` only, no fallback.** Rejected: this is the
  shipped bug; it leaves every real artifact unserviceable.
- **Blob header carries the template string.** Rejected: a ~4 KB template does
  not belong in the small metadata header; a raw section is the format-native
  home and keeps the directory/blob representation identical.
- **Write the template into `tokenizer_config.json` instead.** Rejected: HF
  v5 actively removed that save path ("legacy load-only"); writing it would be
  exactly the backwards move Lemonade corrected.

## Open questions

None. The canonical-location question is ruled above on external evidence
(HF docs, Lemonade PR #7, llama.cpp GGUF metadata), and the blob-section
mechanism reuses the existing raw-tag path.
