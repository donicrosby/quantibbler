# Proposal: qk-chat-template-location

## Why

`POST /v1/chat/completions` 400s on every REAL exported artifact. Ground A of
the v6b sanity verdict (landed `a208a6e`,
`openspec/changes/qk-pertarget-cycle/SANITY-VERDICT-v6b-20261027.md`) measured
the cause: the server reads the model's Jinja2 chat template only from the
artifact's `tokenizer.json` `chat_template` field
(`crates/quantibbler-server/src/model/mod.rs:409-454`), but no real HF export
ever puts it there. transformers 5.20.1's `tokenizer.save_pretrained()` writes
the template to the standalone sidecar `chat_template.jinja`, and leaves no
`chat_template` field in either `tokenizer.json` or `tokenizer_config.json`
(verified against the on-disk artifact: 4168-byte `chat_template.jinja`
present; the field absent from both JSON files). The three committed server
fixtures were hand-made to inject the field into `tokenizer.json`, so the whole
surface is green while `no real fire artifact can serve chat` (verdict
quote). The template bytes ARE in the artifact — this is a file-location gap,
not missing content.

## What Changes

1. **Loader reads the canonical location.** The server resolves the chat
   template from `chat_template.jinja` (the transformers-5.x canonical sidecar
   — the file HF's documented precedence resolves as the authoritative source
   when present, and the location llama.cpp's
   converter and AMD Lemonade both read). The `tokenizer.json` `chat_template`
   read is DEMOTED to a deprecated last-resort fallback — it was a
   fixture-invented location no real export produces, and fixtures are
   regenerated in this same change so no committed fixture depends on it.
2. **Legacy fallback.** When `chat_template.jinja` is absent, the loader falls
   back to the `chat_template` field in `tokenizer_config.json` (the
   load-only pre-transformers-5 format), then to a `chat_template` field in
   `tokenizer.json` (deprecated last resort so the three already-committed
   fixtures keep loading until they are regenerated in this same change).
3. **Exporter guarantees the write.** `tools/quantize_hf_model.py` gains a
   fail-closed post-`save_pretrained` self-check: the artifact MUST contain a
   non-empty `chat_template.jinja` whose trimmed content equals the loaded
   tokenizer's `chat_template` string, else the export refuses and no artifact
   is emitted. The exporter does NOT inject `chat_template` into
   `tokenizer.json` — a second, bespoke copy is exactly the dual-source drift
   the benchmark cannot tolerate.
4. **Blob carries the same source.** `src/quantibbler/blob.py` packs
   `chat_template.jinja` as a raw section when present, and the qpk1 blob
   reader exposes it, so a blob artifact serves chat from the SAME canonical
   source (not a relocated or injected copy).
5. **Fixtures model reality.** `tools/gen_server_fixtures.py` writes the pinned
   Qwen3 template to `chat_template.jinja` instead of injecting it into
   `tokenizer.json`; `tools/gen_chat_golden.py` reads it from the sidecar. The
   regenerated fixtures reproduce the real-artifact shape (template in
   `chat_template.jinja`, absent from `tokenizer.json`).
6. **Real-artifact regression leg.** A gated test loads the on-disk v6b
   artifact directory — `chat_template.jinja` present, no `tokenizer.json`
   field — and serves a chat completion, asserting a non-400 response and
   prompt-token parity with HF's `apply_chat_template` on the same message.

## Impact

- **Invariant-touching: yes — the artifact's loader-facing chat-template
  contract.** The server now reads the template from `chat_template.jinja`
  (and, for blobs, a same-named section). The compressor's `save_pretrained`
  already writes that sidecar, so the "server reads what the compressor
  writes" invariant (`config.yaml` context) is what the fix RE-ALIGNS with.
- **Affected specs**: `qk-server` — 3 ADDED requirements (loader resolution,
  exporter guarantee, real-artifact regression).
- **Affected code**: `crates/quantibbler-server/src/model/mod.rs` (template
  resolution + `tokenizer_from_json` simplification),
  `crates/quantibbler-qpk1/src/loader.rs` + `src/blob.rs` (expose a
  `chat_template.jinja` raw section), `src/quantibbler/blob.py` (pack the
  section), `tools/quantize_hf_model.py` (exporter self-check),
  `tools/gen_server_fixtures.py` + `tools/gen_chat_golden.py`
  (fixture/golden regeneration), `crates/quantibbler-server/tests/` (fixture
  relocation + real-artifact leg). **No new dependencies.**

## Non-goals

- No fix to artifact QUALITY (Grounds B/C of the verdict — degenerate/looping
  generations, the 14.14× PPL ratio). That is a re-fire/refire question owned
  by @user, not this change.
- No change to the chat-template RENDER semantics (minijinja, the pinned
  render context `add_generation_prompt=true` / `enable_thinking=false`, the
  `python_str_method` hook). Only the template's load LOCATION moves.
- No re-run of the FP baseline, no `qk_bar` re-judgement, no re-fire.
- No handling of multi-template / `additional_chat_templates/` variants —
  out of scope until a benchmark model ships one.
- No archiving of `qk-serve-bf16` (a separate closeout task).
