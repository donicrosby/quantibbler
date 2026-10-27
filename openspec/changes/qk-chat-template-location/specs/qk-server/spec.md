# qk-server delta: qk-chat-template-location

## ADDED Requirements

### Requirement: The loader SHALL resolve the chat template from its canonical location

On load the server SHALL resolve the model's chat template from the artifact's
canonical location `chat_template.jinja` — the directory file, or a blob's
same-named raw section — and MUST fall back to the legacy `chat_template`
string field in `tokenizer_config.json` (then the deprecated `tokenizer.json`
field) only when the canonical file is absent or blank. Resolving no template SHALL
leave it `None`, and `/v1/chat/completions` SHALL then refuse 400 rather than
guess a prompt format.

#### Scenario: CT-1.1 — The canonical `chat_template.jinja` file is read first

- **WHEN** a directory artifact contains a non-empty `chat_template.jinja`
  (the transformers-5.x sidecar no `tokenizer.json` field accompanies)
- **THEN** the loader resolves `chat_template` from that file's text
- **AND** a subsequent chat completion renders the prompt through that template
  without error

#### Scenario: CT-1.2 — Legacy fields are consulted only when the canonical file is absent

- **WHEN** an artifact has no `chat_template.jinja` but its
  `tokenizer_config.json` carries a `chat_template` string field
- **THEN** the loader resolves `chat_template` from `tokenizer_config.json`
- **AND** an artifact with neither a `.jinja` file nor a `tokenizer_config.json`
  field but a `tokenizer.json` field is resolved from that deprecated location
  so the pre-regeneration fixtures keep loading

#### Scenario: CT-1.3 — No template anywhere refuses the chat endpoint

- **WHEN** an artifact has no `chat_template.jinja`, no `tokenizer_config.json`
  field, and no `tokenizer.json` field (all absent, blank, or whitespace-only)
- **THEN** `chat_template` is `None` and `/v1/chat/completions` returns 400 in
  the invalid_request_error envelope naming the absence; no tokenization or
  generation is attempted

#### Scenario: CT-1.4 — A blob's `chat_template.jinja` section resolves identically

- **WHEN** a `.qpk` blob carries a raw `chat_template.jinja` section and the
  same source directory served as a directory artifact
- **THEN** both resolve the identical template string and render identical
  prompts on the same message

### Requirement: The exporter SHALL guarantee a non-empty canonical chat template

The exporter SHALL, after `tokenizer.save_pretrained(out_dir)`, verify the
artifact contains a non-empty `chat_template.jinja` whose trimmed content
equals the trimmed loaded tokenizer's `chat_template` string, and MUST refuse the
export — emitting no artifact — on any missing file, blank content, or
mismatch. It MUST NOT write the `chat_template` string into `tokenizer.json`
or `tokenizer_config.json`.

#### Scenario: CT-2.1 — A matching `/chat_template.jinja` passes the self-check

- **WHEN** an export leaves a `chat_template.jinja` whose trimmed content
  equals the trimmed loaded tokenizer's `chat_template`
- **THEN** the export completes and the artifact is emitted
- **AND** the artifact's `tokenizer.json` carries no `chat_template` field

#### Scenario: CT-2.2 — A missing, blank, or mismatched template refuses the export

- **WHEN** after `save_pretrained` the artifact lacks `chat_template.jinja`, or
  it is empty after trim, or its trimmed content differs from the trimmed tokenizer's
  `chat_template` string
- **THEN** the exporter raises with a diagnostic naming the file and the
  disagreement; no blob is emitted; the failed export's partial output
  directory SHALL NOT be registered as a successful artifact

### Requirement: A real-artifact regression leg SHALL verify chat serves from the canonical location

The suite SHALL include a gated regression leg that loads a REAL exported
artifact directory — activated by `QKB_REAL_ARTIFACT_DIR`, SKIP when unset —
and serves `/v1/chat/completions`, asserting a non-400 response and prompt
parity with the Python HF stack's `apply_chat_template` on the same message.
The committed fixtures SHALL reproduce the real-artifact shape: the template
in `chat_template.jinja` and no `chat_template` field in `tokenizer.json`.

#### Scenario: CT-3.1 — A real artifact serves chat with HF-prompt parity

- **WHEN** the gated leg loads a real exported artifact directory (a
  `chat_template.jinja` present with no `tokenizer.json` field — e.g. the
  v6b artifact) and `QKB_REAL_ARTIFACT_DIR` is set
- **THEN** `/v1/chat/completions` on the pinned probe message
  `[{"role":"user","content":"hello"}]` returns non-400
- **AND** the rendered prompt string equals HF's
  `AutoTokenizer.from_pretrained(<artifact>).apply_chat_template(messages,
  tokenize=False, add_generation_prompt=True)` with thinking off
  (oracle: `tools/gen_chat_golden.py`, `ENABLE_THINKING=False`)

#### Scenario: CT-3.2 — Committed fixtures reproduce the real-artifact shape

- **WHEN** `tools/gen_server_fixtures.py` regenerates the server fixtures and
  `tools/gen_chat_golden.py` regenerates the golden
- **THEN** the fixture's template lives in `chat_template.jinja` with no
  `chat_template` field in `tokenizer.json`
- **AND** the always-run suite asserts that shape loads and renders the
  unchanged golden (same template bytes, relocated)
