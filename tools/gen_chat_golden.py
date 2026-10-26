#!/usr/bin/env python
"""Generate the chat-template golden for qk-serve-bf16 BS-5.1.

The server renders the model's own Jinja2 ``chat_template`` with ``minijinja``;
the parity contract is a golden generated ONCE by the Python HF stack with the
SAME pinned kwargs, committed beside the fixtures:

    messages              = [{"role": "user", "content": "hello"}]
    add_generation_prompt = true
    enable_thinking       = false

The golden records BOTH halves BS-5.1 pins: the rendered prompt STRING (from
``transformers``' ``apply_chat_template``, i.e. Python Jinja2) and the token ids
the committed fixture tokenizer produces for that string (from the ``tokenizers``
crate's Python face — the same implementation the server links).

The chat template itself is MODEL DATA, carried verbatim in the fixture
``tokenizer.json``; its provenance (source model + revision) is recorded in
``tools/gen_server_fixtures.py`` next to the string.

Regeneration is deterministic: no timestamps are written, so a diff here is a
real behavioural change (template, tokenizer, or tokenizer version), not noise.
Run:  ``python tools/gen_chat_golden.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "server-model" / "tokenizer.json"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "chat_golden.json"

#: The pinned probing messages (BS-5.1).
MESSAGES = [{"role": "user", "content": "hello"}]
#: The pinned render context (design S5) — identical on both legs.
ADD_GENERATION_PROMPT = True
ENABLE_THINKING = False


def main() -> int:
    document = json.loads(FIXTURE_TOKENIZER.read_text(encoding="utf-8"))
    template = document.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise SystemExit(f"{FIXTURE_TOKENIZER}: no usable chat_template")

    import jinja2
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    # The `tokenizers` crate (both its Rust and Python faces) REJECTS an unknown
    # top-level key, so the fixture's own document — which carries
    # `chat_template` by design — is not loadable as a BPE definition. Hand the
    # stack the same document with that one key removed; the server's loader
    # does exactly the same (see `model::tokenizer_from_json`).
    stripped = {k: v for k, v in document.items() if k != "chat_template"}
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tokenizer.json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(stripped, handle)
        stripped_path = handle.name

    # 1) The render, through the HF stack (transformers' apply_chat_template).
    hf = PreTrainedTokenizerFast(tokenizer_file=stripped_path, chat_template=template)
    prompt = hf.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=ADD_GENERATION_PROMPT,
        enable_thinking=ENABLE_THINKING,
    )
    # ...cross-checked against the same template rendered by jinja2 directly, so
    # a transformers-side transformation of the string cannot slip through.
    direct = jinja2.Template(template, autoescape=False).render(
        messages=MESSAGES,
        add_generation_prompt=ADD_GENERATION_PROMPT,
        enable_thinking=ENABLE_THINKING,
    )
    if direct != prompt:
        raise SystemExit(
            "transformers' apply_chat_template and a direct jinja2 render "
            f"disagree:\n  hf     = {prompt!r}\n  direct = {direct!r}"
        )

    # 2) The token ids, from the committed fixture tokenizer — the same
    #    implementation the server links (add_special_tokens=False mirrors
    #    `Engine::encode`'s `tokenizer.encode(text, false)`).
    ids = Tokenizer.from_file(stripped_path).encode(prompt, add_special_tokens=False).ids

    golden = {
        # The pinned inputs (BS-5.1 / design S5) — the same context the server
        # renders with, so the contract is readable without the generator.
        "messages": MESSAGES,
        "add_generation_prompt": ADD_GENERATION_PROMPT,
        "enable_thinking": ENABLE_THINKING,
        # The contract itself.
        "prompt": prompt,
        "token_ids": ids,
        "provenance": {
            "generator": "tools/gen_chat_golden.py",
            "tokenizer": "tests/fixtures/server-model/tokenizer.json",
            "chat_template_source": "Qwen/Qwen3-0.6B tokenizer_config.json:chat_template",
            "chat_template_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "python": {
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "jinja2": jinja2.__version__,
            },
            "note": (
                "BS-5.1 parity contract: rendered prompt string and token ids. "
                "Regenerate with `python tools/gen_chat_golden.py`."
            ),
        },
    }
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
    print(f"prompt={prompt!r}")
    print(f"token_ids={ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
