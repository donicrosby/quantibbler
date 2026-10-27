#!/usr/bin/env python
"""Generate qpk1 model-directory fixtures for the pure-Rust server tests.

Writes a minimal valid qpk1 model directory (the ``qk-server`` acquisition
layout, design D3 of openspec/changes/qk-server) into
``tests/fixtures/server-model/`` plus one corrupted variant per SR-1.2
rejection class into ``tests/fixtures/server-model-bad/<name>/``, and the
bias-carrying version-2 twin into ``tests/fixtures/server-model-bias/``.

Everything is byte-level against the pinned qpk1 layout (qk-representation,
design D1): 24-byte u32 LE header (magic b"QPK1", version, dtype tag,
dout, din, r), then the scale vectors in the tagged dtype (float16 tag 1 /
float32 tag 2), then -- version 2 -- the (dout,) bias vector in that dtype,
then packed uint8 Ub/Vb bitplanes -- row-major, LSB-first within bytes,
+1 -> bit 1, -1 -> bit 0, zero pad to the byte boundary.

Two versions are produced on purpose. ``server-model`` is the committed
PRE-BIAS artifact: it is version 1, and every server test that loads it is
exercising the declared legacy reading (no bias field; the zero map).
``server-model-bias`` is what the compressor writes today: version 2, with a
deterministic non-zero bias on every projection, so the loader and the
engine's forward are exercised on bias-carrying bytes.

Pure Python byte packing (struct + manual bit packing, plus the repo's own
``quantibbler.blob`` writer for the single-file form); the Rust extension is
never imported. Deterministic (fixed seeds); regenerating is idempotent, and
fixture drift is a reviewable diff.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model"
BAD_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model-bad"
BIAS_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model-bias"
#: The bf16-remainder twin, produced by tools/repack_remainder.py from
#: FIXTURE_DIR: same layer streams and tokenizer, sidecars re-encoded at bf16
#: and declared @bf16.
BF16_DIR = REPO_ROOT / "tests" / "fixtures" / "server-model-bf16"
#: The single-file blob form of FIXTURE_DIR (change qk-artifact-blob). The
#: directory stays committed beside it (dual-read, design B7), so both loader
#: paths keep an artifact and the two loads are compared field-for-field.
BLOB_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model.qpk"
#: The bf16-remainder BLOB twin (change qk-serve-bf16, BS-9): the repack
#: utility's blob->blob leg on BLOB_FIXTURE, so every layer stream, the
#: tokenizer and the metadata are byte-identical and only the remainder
#: sections plus their `bf16` declaration move.
BF16_BLOB_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "server-model-bf16.qpk"

QPK1_MAGIC = b"QPK1"
#: Version of the committed pre-bias fixture.
LEGACY_REPRESENTATION_VERSION = 1
#: Version the compressor writes: scales + a (dout,) bias vector.
REPRESENTATION_VERSION = 2
DTYPE_TAG_FLOAT16 = 1
DTYPE_TAG_FLOAT32 = 2
_HEADER = struct.Struct("<4sIIIII")

# Model-level hyperparameters written to config.toml.
CONFIG = {
    "vocab_size": 32,
    "d_model": 4,
    "eos_token_id": 3,
    "bos_token_id": 1,
    # Attention geometry: n_kv*head_dim == d_model,
    # n_heads % n_kv == 0 — the loader rejects anything else fail-closed.
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "head_dim": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
}

# Per-block norm inventory: the fixture exercises the
# Qwen3-class inventory (q/k norms present). Layout arithmetic below derives
# from THIS list, mirroring the loader's inventory-derived count.
NORM_SLOTS = ["input", "post_attention", "q_norm", "k_norm"]

_NORM_SLOT_WIDTH = {
    "input": "d_model",
    "post_attention": "d_model",
    "post_feedforward": "d_model",
    "q_norm": "head_dim",
    "k_norm": "head_dim",
}

# Geometry + inventory: every config.toml built from this string carries a
# VALID norm inventory, so bad-variants reusing it fail on the field they
# corrupt — not on a missing norm_slots.
GEOMETRY_TOML = (
    f"n_layers = {CONFIG['n_layers']}\n"
    f"n_heads = {CONFIG['n_heads']}\n"
    f"n_kv_heads = {CONFIG['n_kv_heads']}\n"
    f"head_dim = {CONFIG['head_dim']}\n"
    f"rms_norm_eps = {CONFIG['rms_norm_eps']}\n"
    f"rope_theta = {CONFIG['rope_theta']}\n"
    "norm_slots = [" + ", ".join(f'"{s}"' for s in NORM_SLOTS) + "]\n"
)


def norm_floats_per_block() -> int:
    total = 0
    for slot in NORM_SLOTS:
        width_key = _NORM_SLOT_WIDTH[slot]
        assert width_key in CONFIG, f"slot {slot}: unknown width key {width_key}"
        total += CONFIG[width_key]
    return total


def write_norms_sidecar(root: Path) -> None:
    """Deterministic norm sidecar: per-block inventory order (NORM_SLOTS) +
    the final norm (d_model), fp32 LE."""
    import array

    count = norm_floats_per_block() * CONFIG["n_layers"] + CONFIG["d_model"]
    norms = array.array("f", [0.1 * (i + 1) for i in range(count)])
    (root / "norms.f32.bin").write_bytes(norms.tobytes())


#: The canonical chat-template sidecar's file name: the location a real
#: export's ``save_pretrained`` writes the model's template to, and the one the
#: serving stack reads it back from.
CHAT_TEMPLATE_FILE = "chat_template.jinja"

#: The model's OWN chat template, the genuine Qwen3 artifact written verbatim
#: to the fixture's canonical ``chat_template.jinja`` sidecar — the file
#: `save_pretrained` emits and the serving stack reads. This is MODEL DATA, not
#: implementation-sourced (clean-room rule). Provenance:
#: Qwen/Qwen3-0.6B tokenizer_config.json `chat_template`, HF revision
#: c1899de289a04d12100db370d81485cdf75e47ca (fetched 2026-10-26).
CHAT_TEMPLATE = (
    "{%- if tools %}\n    {{- '<|im_start|>sys"
    "tem\\n' }}\n    {%- if messages[0].role =="
    " 'system' %}\n        {{- messages[0].con"
    "tent + '\\n\\n' }}\n    {%- endif %}\n    {{"
    '- "# Tools\\n\\nYou may call one or more f'
    "unctions to assist with the user query.\\"
    "n\\nYou are provided with function signat"
    "ures within <tools></tools> XML tags:\\n<"
    'tools>" }}\n    {%- for tool in tools %}\n'
    '        {{- "\\n" }}\n        {{- tool | t'
    'ojson }}\n    {%- endfor %}\n    {{- "\\n</'
    "tools>\\n\\nFor each function call, return"
    " a json object with function name and ar"
    "guments within <tool_call></tool_call> X"
    'ML tags:\\n<tool_call>\\n{\\"name\\": <funct'
    'ion-name>, \\"arguments\\": <args-json-obj'
    'ect>}\\n</tool_call><|im_end|>\\n" }}\n{%- '
    "else %}\n    {%- if messages[0].role == '"
    "system' %}\n        {{- '<|im_start|>syst"
    "em\\n' + messages[0].content + '<|im_end|"
    ">\\n' }}\n    {%- endif %}\n{%- endif %}\n{%"
    "- set ns = namespace(multi_step_tool=tru"
    "e, last_query_index=messages|length - 1)"
    " %}\n{%- for message in messages[::-1] %}"
    "\n    {%- set index = (messages|length - "
    "1) - loop.index0 %}\n    {%- if ns.multi_"
    'step_tool and message.role == "user" and'
    " message.content is string and not(messa"
    "ge.content.startswith('<tool_response>')"
    " and message.content.endswith('</tool_re"
    "sponse>')) %}\n        {%- set ns.multi_s"
    "tep_tool = false %}\n        {%- set ns.l"
    "ast_query_index = index %}\n    {%- endif"
    " %}\n{%- endfor %}\n{%- for message in mes"
    "sages %}\n    {%- if message.content is s"
    "tring %}\n        {%- set content = messa"
    "ge.content %}\n    {%- else %}\n        {%"
    "- set content = '' %}\n    {%- endif %}\n "
    '   {%- if (message.role == "user") or (m'
    'essage.role == "system" and not loop.fir'
    "st) %}\n        {{- '<|im_start|>' + mess"
    "age.role + '\\n' + content + '<|im_end|>'"
    " + '\\n' }}\n    {%- elif message.role == "
    '"assistant" %}\n        {%- set reasoning'
    "_content = '' %}\n        {%- if message."
    "reasoning_content is string %}\n         "
    "   {%- set reasoning_content = message.r"
    "easoning_content %}\n        {%- else %}\n"
    "            {%- if '</think>' in content"
    " %}\n                {%- set reasoning_co"
    "ntent = content.split('</think>')[0].rst"
    "rip('\\n').split('<think>')[-1].lstrip('\\"
    "n') %}\n                {%- set content ="
    " content.split('</think>')[-1].lstrip('\\"
    "n') %}\n            {%- endif %}\n        "
    "{%- endif %}\n        {%- if loop.index0 "
    "> ns.last_query_index %}\n            {%-"
    " if loop.last or (not loop.last and reas"
    "oning_content) %}\n                {{- '<"
    "|im_start|>' + message.role + '\\n<think>"
    "\\n' + reasoning_content.strip('\\n') + '\\"
    "n</think>\\n\\n' + content.lstrip('\\n') }}"
    "\n            {%- else %}\n               "
    " {{- '<|im_start|>' + message.role + '\\n"
    "' + content }}\n            {%- endif %}\n"
    "        {%- else %}\n            {{- '<|i"
    "m_start|>' + message.role + '\\n' + conte"
    "nt }}\n        {%- endif %}\n        {%- i"
    "f message.tool_calls %}\n            {%- "
    "for tool_call in message.tool_calls %}\n "
    "               {%- if (loop.first and co"
    "ntent) or (not loop.first) %}\n          "
    "          {{- '\\n' }}\n                {%"
    "- endif %}\n                {%- if tool_c"
    "all.function %}\n                    {%- "
    "set tool_call = tool_call.function %}\n  "
    "              {%- endif %}\n             "
    '   {{- \'<tool_call>\\n{"name": "\' }}\n    '
    "            {{- tool_call.name }}\n      "
    '          {{- \'", "arguments": \' }}\n    '
    "            {%- if tool_call.arguments i"
    "s string %}\n                    {{- tool"
    "_call.arguments }}\n                {%- e"
    "lse %}\n                    {{- tool_call"
    ".arguments | tojson }}\n                {"
    "%- endif %}\n                {{- '}\\n</to"
    "ol_call>' }}\n            {%- endfor %}\n "
    "       {%- endif %}\n        {{- '<|im_en"
    "d|>\\n' }}\n    {%- elif message.role == \""
    'tool" %}\n        {%- if loop.first or (m'
    'essages[loop.index0 - 1].role != "tool")'
    " %}\n            {{- '<|im_start|>user' }"
    "}\n        {%- endif %}\n        {{- '\\n<t"
    "ool_response>\\n' }}\n        {{- content "
    "}}\n        {{- '\\n</tool_response>' }}\n "
    "       {%- if loop.last or (messages[loo"
    'p.index0 + 1].role != "tool") %}\n       '
    "     {{- '<|im_end|>\\n' }}\n        {%- e"
    "ndif %}\n    {%- endif %}\n{%- endfor %}\n{"
    "%- if add_generation_prompt %}\n    {{- '"
    "<|im_start|>assistant\\n' }}\n    {%- if e"
    "nable_thinking is defined and enable_thi"
    "nking is false %}\n        {{- '<think>\\n"
    "\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- "
    "endif %}"
)

TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 0,
            "content": "<unk>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 1,
            "content": "<bos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 2,
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ],
    "normalizer": None,
    "pre_tokenizer": {
        "type": "Whitespace",
    },
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": "<unk>",
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": True,
        "byte_fallback": False,
        "vocab": {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "a": 3,
            "b": 4,
            "c": 5,
            "d": 6,
            "e": 7,
            "f": 8,
            "g": 9,
            "h": 10,
            "i": 11,
            "j": 12,
            "k": 13,
            "l": 14,
            "m": 15,
            "n": 16,
            "o": 17,
            "p": 18,
            "q": 19,
            "r": 20,
            "s": 21,
            "t": 22,
            "u": 23,
            "v": 24,
            "w": 25,
            "x": 26,
            "y": 27,
            "z": 28,
            "ab": 29,
            "ba": 30,
        },
        "merges": [
            "a b",
            "b a",
        ],
    },
}


def f16_bytes(value: float) -> bytes:
    """Float16 (IEEE 754 binary16) little-endian encoding, stdlib only."""
    return struct.pack("<e", float(value))


def pack_bits(rows: list[list[int]]) -> bytes:
    """Pack a +/-1 matrix per the pinned qpk1 canonical layout (SF-3.1).

    Row-major with ONE continuous bit stream across the whole matrix:
    element ``(i, j)`` lands at bit ``i * r + j`` of the bitplane, i.e.
    byte ``(i*r + j) // 8``, bit ``(i*r + j) % 8`` from the LSB.
    ``+1 -> bit 1``, ``-1 -> bit 0``, pad bits zero. (Note: this is the
    container layout; the kernels rlib's per-row bitplanes are a different,
    working layout the server converts to at load.)
    """
    flat = [entry for row in rows for entry in row]
    out = bytearray((len(flat) + 7) // 8)
    for k, entry in enumerate(flat):
        if entry == 1:
            out[k >> 3] |= 1 << (k & 7)
        elif entry != -1:
            raise ValueError(f"factor entries must be +/-1, got {entry}")
    return bytes(out)


def deterministic_signs(n_rows: int, n_cols: int, seed: int) -> list[list[int]]:
    """splitmix64-driven +/-1 matrix (fixed seed -> fixed fixture bytes)."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    GOLD = 0x9E3779B97F4A7C15
    MIX1 = 0xBF58476D1CE4E5B9
    MIX2 = 0x94D049BB133111EB
    rows: list[list[int]] = []
    for _ in range(n_rows):
        row: list[int] = []
        for _ in range(n_cols):
            state = (state + GOLD) & 0xFFFFFFFFFFFFFFFF
            z = state
            z = ((z ^ (z >> 30)) * MIX1) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * MIX2) & 0xFFFFFFFFFFFFFFFF
            z ^= z >> 31
            row.append(1 if (z >> 11) % 2 == 0 else -1)
        rows.append(row)
    return rows


def qpk1_stream(
    dout: int,
    din: int,
    r: int,
    seed: int,
    dtype_tag: int = DTYPE_TAG_FLOAT16,
    s1: list[float] | None = None,
    s2: list[float] | None = None,
    version: int = LEGACY_REPRESENTATION_VERSION,
    bias: list[float] | None = None,
) -> bytes:
    """Serialize one factorized linear layer to the canonical qpk1 stream.

    ``version`` 1 is the pre-bias layout; version 2 appends the ``(dout,)``
    bias vector between the scales and the bitplanes (all in the tagged
    dtype). A version-2 stream always carries the vector -- ``bias=None``
    writes zeros rather than omitting the field.
    """
    ub = deterministic_signs(dout, r, seed)
    vb = deterministic_signs(din, r, seed + 1)
    if s1 is None:
        s1 = [round(0.5 + 0.25 * ((seed + k) % 5) / 5.0, 4) for k in range(dout)]
    if s2 is None:
        s2 = [round(0.4 + 0.2 * ((seed + k) % 3) / 3.0, 4) for k in range(din)]
    if bias is None:
        bias = [0.0] * dout
    if dtype_tag == DTYPE_TAG_FLOAT16:
        s1_bytes = b"".join(f16_bytes(v) for v in s1)
        s2_bytes = b"".join(f16_bytes(v) for v in s2)
        bias_bytes = b"".join(f16_bytes(v) for v in bias)
    elif dtype_tag == DTYPE_TAG_FLOAT32:
        s1_bytes = struct.pack(f"<{dout}f", *map(float, s1))
        s2_bytes = struct.pack(f"<{din}f", *map(float, s2))
        bias_bytes = struct.pack(f"<{dout}f", *map(float, bias))
    else:
        raise ValueError(f"unknown dtype tag {dtype_tag}")
    if version not in (LEGACY_REPRESENTATION_VERSION, REPRESENTATION_VERSION):
        raise ValueError(f"unknown representation version {version}")
    header = _HEADER.pack(QPK1_MAGIC, version, dtype_tag, dout, din, r)
    body = s1_bytes + s2_bytes
    if version == REPRESENTATION_VERSION:
        body += bias_bytes
    return header + body + pack_bits(ub) + pack_bits(vb), (ub, vb, s1, s2)


def expected_size(dout: int, din: int, r: int, w: int, with_bias: bool = False) -> int:
    """SF-1.1 size formula, at the version-2 float count when ``with_bias``."""
    floats = (2 * dout + din) if with_bias else (dout + din)
    return 24 + floats * w + (dout * r + 7) // 8 + (din * r + 7) // 8


def deterministic_bias(dout: int, seed: int) -> list[float]:
    """splitmix-driven bias vector: non-zero, deterministic, reviewable."""
    return [round(0.05 + 0.1 * ((seed + 3 * k) % 5), 4) for k in range(dout)]


# One real transformer block's projections: the
# engine resolves blocks by these manifest names, so the fixture exercises
# the same resolution + forward path as a driver-exported model. Geometry
# (d_model 4, intermediate 4) makes every projection a 4x4 r4 stream.
BLOCK_PROJECTIONS = [
    "layers.0.inner.self_attn.q_proj",
    "layers.0.inner.self_attn.k_proj",
    "layers.0.inner.self_attn.v_proj",
    "layers.0.inner.self_attn.o_proj",
    "layers.0.inner.mlp.gate_proj",
    "layers.0.inner.mlp.up_proj",
    "layers.0.inner.mlp.down_proj",
]
# The corruption variants target one projection's stream file.
PROBE_LAYER = "layers.0.inner.self_attn.q_proj"
PROBE_FILE = "layers/q_proj.qpk1"


def build_model(root: Path, *, model_id: str, version: int) -> None:
    """Write one fixture model directory (one transformer block).

    ``version`` selects the container version: 1 for the committed pre-bias
    artifact, 2 for the bias-carrying twin, whose every projection carries a
    deterministic non-zero bias.
    """
    if root.exists():
        shutil.rmtree(root)  # regenerate clean: no stale layer files
    (root / "layers").mkdir(parents=True, exist_ok=True)
    dout, din, r = 4, 4, 4
    with_bias = version == REPRESENTATION_VERSION
    expected = expected_size(dout, din, r, 2, with_bias=with_bias)
    entries = []
    for i, name in enumerate(BLOCK_PROJECTIONS):
        stem = name.rsplit(".", 1)[-1]
        rel = f"layers/{stem}.qpk1"
        seed = 20261017 + i
        stream, _ = qpk1_stream(
            dout,
            din,
            r,
            seed=seed,
            version=version,
            bias=deterministic_bias(dout, seed) if with_bias else None,
        )
        layer_path = root / rel
        layer_path.write_bytes(stream)
        actual = layer_path.stat().st_size
        if actual != expected:
            raise AssertionError(f"{layer_path}: {actual} bytes, formula says {expected}")
        entries.append((name, rel))
    manifest_layers = ",\n".join(f'  "{n}"' for n, _ in entries)
    manifest_files = "\n".join(f'"{n}" = "{rel}"' for n, rel in entries)
    (root / "manifest.toml").write_text(
        "# qk-server fixture manifest (design D3)\n"
        f'model_id = "{model_id}"\n'
        f"layers = [\n{manifest_layers},\n]\n"
        "\n"
        "[layer_files]\n"
        "# quoted keys: TOML bare keys with dots would nest tables\n"
        f"{manifest_files}\n"
        "\n"
        "[sidecar]\n"
        "# fp32 embedding / lm_head / norms raw LE float32\n"
        'embedding = "e1.f32.bin"\n'
        'lm_head = "e1.f32.bin"\n'
        'norms = "norms.f32.bin"\n',
        encoding="utf-8",
    )
    (root / "config.toml").write_text(
        "# qk-server fixture config (design D3)\n"
        f"vocab_size = {CONFIG['vocab_size']}\n"
        f"d_model = {CONFIG['d_model']}\n"
        f"eos_token_id = {CONFIG['eos_token_id']}\n"
        f"bos_token_id = {CONFIG['bos_token_id']}\n"
        'weight_tying = "lm_head=e1@fp32"\n' + GEOMETRY_TOML,
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(TOKENIZER_JSON, indent=1) + "\n", encoding="utf-8"
    )
    # The model's chat template lives in its canonical sidecar, not in the
    # tokenizer document: this is the shape `save_pretrained` writes for a real
    # export, so the fixture cannot pass on a location no artifact uses.
    (root / CHAT_TEMPLATE_FILE).write_text(CHAT_TEMPLATE, encoding="utf-8")
    # Sidecar: fp32 embedding/lm_head weights, weight-tied, byte layout in
    # the manifest (OQ2 minimally: raw LE f32, row-major).
    import array

    emb = array.array("f", range(CONFIG["vocab_size"] * CONFIG["d_model"]))
    # deterministic small values, incl. negatives
    for i in range(len(emb)):
        emb[i] = ((i % 7) - 3) * 0.25
    (root / "e1.f32.bin").write_bytes(emb.tobytes())
    write_norms_sidecar(root)


def build_valid_model(root: Path) -> None:
    """The committed PRE-BIAS fixture: version 1, no bias field."""
    build_model(root, model_id="server-model-fixture", version=LEGACY_REPRESENTATION_VERSION)


def build_bias_model(root: Path) -> None:
    """The bias-carrying version-2 twin of the same model directory."""
    build_model(root, model_id="server-model-bias-fixture", version=REPRESENTATION_VERSION)


def build_bf16_model(root: Path) -> None:
    """The bf16-remainder twin, produced BY THE REPACK UTILITY.

    Generating it through the shipped conversion (rather than a second,
    independent writer of the same bytes) means the committed fixture is the
    utility's own output: a drift between the two is a diff here, and the
    fixture cannot prove a contract the utility does not implement.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import repack_remainder

    if root.exists():
        shutil.rmtree(root)
    repack_remainder.repack(FIXTURE_DIR, root)


def build_bad_variants(base: Path) -> dict[str, str]:
    """One corrupted model dir per SR-1.2 rejection class.

    Returns {variant_name: expected diagnostic substring}.
    """
    if base.exists():
        shutil.rmtree(base)
    variants: dict[str, str] = {}

    def derived(name: str, mutate) -> None:
        dest = base / name
        shutil.copytree(FIXTURE_DIR, dest)
        mutate(dest)
        variants[name] = dest

    # 1. missing manifest.toml
    derived("missing_manifest", lambda d: (d / "manifest.toml").unlink())
    # 2. missing tokenizer.json
    derived("missing_tokenizer", lambda d: (d / "tokenizer.json").unlink())
    # 3. malformed tokenizer.json
    derived("malformed_tokenizer", lambda d: (d / "tokenizer.json").write_text("{not json"))
    # 4. manifest names a layer whose file is missing
    derived("missing_layer_file", lambda d: (d / PROBE_FILE).unlink())

    # 5. bad magic
    def _bad_magic(d: Path) -> None:
        p = d / PROBE_FILE
        p.write_bytes(b"XPK1" + p.read_bytes()[4:])

    derived("bad_magic", _bad_magic)

    # 6. wrong version
    def _bad_version(d: Path) -> None:
        p = d / PROBE_FILE
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 4, 7)
        p.write_bytes(bytes(b))

    derived("wrong_version", _bad_version)

    # 7. unknown dtype tag
    def _bad_tag(d: Path) -> None:
        p = d / PROBE_FILE
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 8, 9)
        p.write_bytes(bytes(b))

    derived("unknown_dtype_tag", _bad_tag)

    # 8. rank 0
    def _rank0(d: Path) -> None:
        p = d / PROBE_FILE
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 20, 0)
        p.write_bytes(bytes(b))

    derived("rank_zero", _rank0)

    # 9. r > min(dout, din)
    def _rank_too_big(d: Path) -> None:
        p = d / PROBE_FILE
        b = bytearray(p.read_bytes())
        struct.pack_into("<I", b, 20, 9)
        p.write_bytes(bytes(b))

    derived("rank_infeasible", _rank_too_big)

    # 10. truncated stream (chop the last byte)
    def _truncated(d: Path) -> None:
        p = d / PROBE_FILE
        raw = p.read_bytes()
        p.write_bytes(raw[:-1])

    derived("truncated", _truncated)

    # 11. trailing garbage
    def _trailing(d: Path) -> None:
        p = d / PROBE_FILE
        p.write_bytes(p.read_bytes() + b"\x00")

    derived("trailing_bytes", _trailing)

    # 12. manifest layer path escapes the model dir (review finding:
    #     Path::join absolute/..-replacement made this an arbitrary read)
    def _path_escape(d: Path) -> None:
        m = d / "manifest.toml"
        m.write_text(
            "# qk-server fixture manifest (design D3)\n"
            'model_id = "server-model-fixture"\n'
            'layers = [ "' + PROBE_LAYER + '" ]\n'
            "\n"
            "[layer_files]\n"
            f'"{PROBE_LAYER}" = "../evil.qpk1"\n'
            "\n"
            "[sidecar]\n"
            'embedding = "e1.f32.bin"\n'
            'lm_head = "e1.f32.bin"\n'
            'norms = "norms.f32.bin"\n',
            encoding="utf-8",
        )

    derived("path_escape", _path_escape)

    # 13. layer file is a symlink (escape hatch; loader demands regular files)
    def _symlink_layer(d: Path) -> None:
        target = d / PROBE_FILE
        raw = target.read_bytes()
        target.unlink()
        outside = d / "outside.qpk1"
        outside.write_bytes(raw)
        target.symlink_to(outside)

    derived("symlink_layer", _symlink_layer)

    # 14. sidecar matrix with a wrong (too small) byte length: the
    #     stat-before-read gate must reject on length, pre-read
    def _sidecar_too_big(d: Path) -> None:
        (d / "e1.f32.bin").write_bytes(b"\x00" * 8)

    derived("sidecar_too_big", _sidecar_too_big)

    # 15. undocumented weight_tying value: fail-closed config validation
    def _bad_weight_tying(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            "# qk-server fixture config (design D3)\n"
            "vocab_size = 32\n"
            "d_model = 4\n"
            "eos_token_id = 3\n"
            "bos_token_id = 1\n"
            'weight_tying = "banana"\n' + GEOMETRY_TOML,
            encoding="utf-8",
        )

    derived("bad_weight_tying", _bad_weight_tying)

    # 16. token id at/past vocab_size: unrepresentable row reference
    def _eos_out_of_range(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            "# qk-server fixture config (design D3)\n"
            "vocab_size = 32\n"
            "d_model = 4\n"
            "eos_token_id = 99\n"
            "bos_token_id = 1\n"
            'weight_tying = "lm_head=e1@fp32"\n' + GEOMETRY_TOML,
            encoding="utf-8",
        )

    derived("eos_out_of_range", _eos_out_of_range)

    # --- norm sidecar fail-closed paths ----------------------

    # 17. no [sidecar] block at all
    def _missing_sidecar_block(d: Path) -> None:
        m = d / "manifest.toml"
        m.write_text(m.read_text(encoding="utf-8").split("[sidecar]")[0], encoding="utf-8")

    derived("missing_sidecar_block", _missing_sidecar_block)

    # 18. [sidecar] present but no `norms` key
    def _missing_norms_manifest(d: Path) -> None:
        m = d / "manifest.toml"
        m.write_text(
            m.read_text(encoding="utf-8").replace('norms = "norms.f32.bin"\n', ""),
            encoding="utf-8",
        )

    derived("missing_norms_manifest", _missing_norms_manifest)

    # 19. norms sidecar referenced but absent on disk
    def _missing_norms_file(d: Path) -> None:
        (d / "norms.f32.bin").unlink()

    derived("missing_norms_file", _missing_norms_file)

    # 20. norms sidecar miscounted (5 floats instead of the
    #     inventory-derived count = 16): stat-before-read length gate
    def _norms_wrong_size(d: Path) -> None:
        import array

        (d / "norms.f32.bin").write_bytes(array.array("f", [0.5] * 5).tobytes())

    derived("norms_wrong_size", _norms_wrong_size)

    # --- attention geometry validation paths ---------------------

    # 21. a geometry field missing entirely: the TOML parse itself fails
    def _missing_geometry(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            c.read_text(encoding="utf-8").replace("n_heads = 2\n", ""),
            encoding="utf-8",
        )

    derived("missing_geometry", _missing_geometry)

    # 22. n_kv_heads * head_dim != d_model
    def _gqa_width_mismatch(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            c.read_text(encoding="utf-8").replace("n_kv_heads = 2", "n_kv_heads = 3"),
            encoding="utf-8",
        )

    derived("gqa_width_mismatch", _gqa_width_mismatch)

    # 23. n_heads %% n_kv_heads != 0
    def _gqa_group_mismatch(d: Path) -> None:
        c = d / "config.toml"
        c.write_text(
            c.read_text(encoding="utf-8").replace("n_heads = 2", "n_heads = 3"),
            encoding="utf-8",
        )

    derived("gqa_group_mismatch", _gqa_group_mismatch)
    return variants


def build_blob_fixture() -> None:
    """Pack the committed directory fixture into the single-file blob form.

    The blob is the canonical container (design B7); the directory is kept so
    the dual-read path keeps a fixture of each shape. The header carries every
    declared field, the ``[sidecar]`` dtype rides as ``remainder_dtype``, and
    the layer streams / tokenizer are carried verbatim.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from quantibbler import blob as blob_writer

    blob_writer.write_blob_from_directory(
        FIXTURE_DIR, BLOB_FIXTURE, created_by="quantibbler-gen-server-fixtures"
    )


def build_bf16_blob_fixture() -> None:
    """The bf16-remainder BLOB twin, produced by the shipped repack utility's
    blob->blob leg on the committed fp32 blob (qk-serve-bf16 BS-9). Emitted by
    the same generator so the parity pair cannot drift from the fp32 blob.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import repack_remainder

    if BF16_BLOB_FIXTURE.exists():
        BF16_BLOB_FIXTURE.unlink()
    repack_remainder.repack(BLOB_FIXTURE, BF16_BLOB_FIXTURE, remainder_dtype="bf16")


def main() -> None:
    build_valid_model(FIXTURE_DIR)
    print(f"wrote {FIXTURE_DIR}")
    build_blob_fixture()
    print(f"wrote {BLOB_FIXTURE}")
    build_bias_model(BIAS_DIR)
    print(f"wrote {BIAS_DIR}")
    build_bf16_model(BF16_DIR)
    print(f"wrote {BF16_DIR}")
    build_bf16_blob_fixture()
    print(f"wrote {BF16_BLOB_FIXTURE}")
    variants = build_bad_variants(BAD_DIR)
    for _name, path in sorted(variants.items()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
