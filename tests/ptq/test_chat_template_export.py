"""The export's canonical chat-template guarantee.

The exporter loads the model's tokenizer, lets `save_pretrained` write the
artifact, and then refuses to emit an artifact whose canonical
`chat_template.jinja` sidecar did not land non-empty and equal to the template
the driver loaded. The sidecar is written by `save_pretrained`, not by the
exporter — the check is what turns "the template is usually there" into "the
artifact provably describes its own prompt format".

These legs drive the check on synthetic post-export directories, so they need
no model and no training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quantize_hf_model as qhf  # noqa: E402

TEMPLATE = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"


def _post_export(root: Path, template: str | None) -> Path:
    """A directory shaped like `save_pretrained` just wrote it."""
    out = root / "artifact"
    out.mkdir(parents=True)
    (out / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    if template is not None:
        (out / qhf.CHAT_TEMPLATE_FILE).write_text(template, encoding="utf-8")
    return out


def test_export_self_check_passes_on_a_matching_sidecar_ct_2_1(tmp_path):
    """A sidecar equal to the loaded template passes, and nothing is rewritten."""
    out = _post_export(tmp_path, TEMPLATE)
    qhf.verify_canonical_chat_template(out, TEMPLATE)
    assert (out / qhf.CHAT_TEMPLATE_FILE).read_text(encoding="utf-8") == TEMPLATE


def test_export_self_check_compares_trimmed_content_ct_2_1(tmp_path):
    """Whitespace around the sidecar's content is not a disagreement."""
    out = _post_export(tmp_path, f"\n{TEMPLATE}\n\n")
    qhf.verify_canonical_chat_template(out, TEMPLATE)


def test_export_self_check_refuses_without_a_usable_sidecar_ct_2_2(tmp_path):
    """A missing, blank, or disagreeing sidecar refuses the export."""
    cases = {
        "missing": _post_export(tmp_path / "missing", None),
        "blank": _post_export(tmp_path / "blank", "  \n\t "),
        "mismatched": _post_export(tmp_path / "mismatched", f"{TEMPLATE}{{# drift #}}"),
    }
    for label, out in cases.items():
        with pytest.raises(RuntimeError) as caught:
            qhf.verify_canonical_chat_template(out, TEMPLATE)
        assert qhf.CHAT_TEMPLATE_FILE in str(caught.value), label
