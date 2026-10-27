//! Blob container tests: `qk-representation` BF-1..BF-7 and `qk-server`
//! BL-1..BL-4, exercised against the committed single-file fixture
//! (`tests/fixtures/server-model.qpk`) and a spec-derived blob builder.
//!
//! The builder lays out bytes from the PROSE of design B1-B5 independently of
//! `blob::Blob::parse`, so a parser bug cannot make its own test pass.

use quantibbler_qpk1::blob::{Blob, BlobTag, PAGE_SIZE};
use quantibbler_qpk1::loader::ModelDirectory;
use quantibbler_qpk1::{Layer, RemainderDtype};
use serde_json::{Value, json};

const BLOB_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/server-model.qpk"
);
const DIR_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/server-model"
);

// --- test-side blob builder (independent of the parser) --------------------

fn pad_to_page(out: &mut Vec<u8>) {
    let rem = out.len() % PAGE_SIZE as usize;
    if rem != 0 {
        out.extend(std::iter::repeat_n(0u8, PAGE_SIZE as usize - rem));
    }
}

fn build(header_json: &str, sections: &[(&str, u8, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"QKPB");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    pad_to_page(&mut out);
    let table_len = 4 + sections
        .iter()
        .map(|(n, _, _)| 1 + n.len() + 1 + 8 + 8)
        .sum::<usize>();
    let mut first = out.len() + table_len;
    first += (PAGE_SIZE as usize - first % PAGE_SIZE as usize) % PAGE_SIZE as usize;
    let mut offsets = Vec::new();
    let mut cursor = first;
    for (index, (_, _, data)) in sections.iter().enumerate() {
        offsets.push(cursor);
        cursor += data.len();
        if index + 1 < sections.len() {
            cursor += (PAGE_SIZE as usize - cursor % PAGE_SIZE as usize) % PAGE_SIZE as usize;
        }
    }
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for ((name, tag, data), offset) in sections.iter().zip(&offsets) {
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.push(*tag);
        out.extend_from_slice(&(*offset as u64).to_le_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    }
    pad_to_page(&mut out);
    assert_eq!(out.len(), first, "builder table arithmetic");
    for (index, (_, _, data)) in sections.iter().enumerate() {
        out.extend_from_slice(data);
        if index + 1 < sections.len() {
            pad_to_page(&mut out);
        }
    }
    out
}

fn fixture() -> Vec<u8> {
    std::fs::read(BLOB_FIXTURE).expect("committed blob fixture")
}

fn header_len(blob: &[u8]) -> usize {
    u32::from_le_bytes(blob[8..12].try_into().expect("4 bytes")) as usize
}

fn header_json(blob: &[u8]) -> String {
    String::from_utf8(blob[12..12 + header_len(blob)].to_vec()).expect("utf8 header")
}

fn table_start(blob: &[u8]) -> usize {
    let end = 12 + header_len(blob);
    let rem = end % PAGE_SIZE as usize;
    if rem == 0 {
        end
    } else {
        end + PAGE_SIZE as usize - rem
    }
}

/// `(name, name_start, tag_pos, offset_pos, length_pos)` for each entry —
/// located by re-walking the table bytes, not by trusting the parser.
struct Entry {
    name: String,
    tag_pos: usize,
    offset_pos: usize,
    length_pos: usize,
}

fn entries(blob: &[u8]) -> Vec<Entry> {
    let start = table_start(blob);
    let count = u32::from_le_bytes(blob[start..start + 4].try_into().expect("count")) as usize;
    let mut cursor = start + 4;
    let mut out = Vec::new();
    for _ in 0..count {
        let name_len = blob[cursor] as usize;
        let name = String::from_utf8(blob[cursor + 1..cursor + 1 + name_len].to_vec())
            .expect("entry name");
        let tag_pos = cursor + 1 + name_len;
        out.push(Entry {
            name,
            tag_pos,
            offset_pos: tag_pos + 1,
            length_pos: tag_pos + 9,
        });
        cursor = tag_pos + 17;
    }
    out
}

fn entry<'a>(list: &'a [Entry], name: &str) -> &'a Entry {
    list.iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("no section {name:?}"))
}

fn set_u64(blob: &mut [u8], pos: usize, value: u64) {
    blob[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
}

/// The fixture's sections, re-extractable for re-packed variants.
fn parts() -> (String, Vec<(String, u8, Vec<u8>)>) {
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("fixture parses");
    let sections = parsed
        .sections
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.tag.as_u8(),
                parsed.slice(&blob, s).to_vec(),
            )
        })
        .collect();
    (header_json(&blob), sections)
}

fn refs(sections: &[(String, u8, Vec<u8>)]) -> Vec<(&str, u8, &[u8])> {
    sections
        .iter()
        .map(|(n, t, d)| (n.as_str(), *t, d.as_slice()))
        .collect()
}

fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn bf16_bytes(floats: &[f32]) -> Vec<u8> {
    floats
        .iter()
        .flat_map(|v| bf16_word(*v).to_le_bytes())
        .collect()
}

fn f32_floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
        .collect()
}

fn header_value(blob: &[u8]) -> Value {
    serde_json::from_str(&header_json(blob)).expect("header JSON")
}

fn with_header(value: &Value) -> String {
    serde_json::to_string(value).expect("re-serialize header")
}

// --- BF-1: magic + version -------------------------------------------------

#[test]
fn parses_blob_magic_bf1_1() {
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("the committed blob parses");
    assert_eq!(parsed.header.model_id, "server-model-fixture");
    assert_eq!(parsed.header.vocab_size, 32);
    assert_eq!(parsed.header.d_model, 4);
    assert_eq!(parsed.header.layers.len(), 7);
    assert_eq!(parsed.header.norm_slots.len(), 4);
    assert_eq!(parsed.header.remainder_dtype, RemainderDtype::Float32);
}

#[test]
fn rejects_bad_magic_bf1_2() {
    for observed in [b"BLOB".as_slice(), b"QPK1".as_slice(), &[0, 0, 0, 0]] {
        let mut blob = fixture();
        blob[0..4].copy_from_slice(observed);
        let err = Blob::parse_bytes(&blob).expect_err("bad magic must refuse");
        assert!(err.contains("bad blob magic"), "{err}");
        assert!(
            err.contains("QKPB"),
            "the diagnostic names the expected magic: {err}"
        );
    }
    // Truncated to fewer than four bytes.
    let err = Blob::parse_bytes(b"QK").expect_err("a two-byte file refuses");
    assert!(err.contains("bad blob magic"), "{err}");
    // A qpk1 stream must never be readable as a blob.
    let mut stream = vec![0u8; 44];
    stream[0..4].copy_from_slice(b"QPK1");
    assert!(Blob::parse_bytes(&stream).is_err());
}

#[test]
fn rejects_unsupported_version_bf1_3() {
    for version in [0u32, 2, 7] {
        let mut blob = fixture();
        blob[4..8].copy_from_slice(&version.to_le_bytes());
        let err = Blob::parse_bytes(&blob).expect_err("an unsupported version refuses");
        assert!(
            err.contains(&version.to_string()) && err.contains("version"),
            "the refusal names the version: {err}"
        );
    }
}

// --- BF-2: header ----------------------------------------------------------

#[test]
fn header_roundtrips_all_fields_bf2_1() {
    let blob = fixture();
    let header = Blob::parse_bytes(&blob).expect("parses").header;
    assert_eq!(header.model_id, "server-model-fixture");
    assert_eq!(header.vocab_size, 32);
    assert_eq!(header.d_model, 4);
    assert_eq!(header.n_layers, 1);
    assert_eq!(header.n_heads, 2);
    assert_eq!(header.n_kv_heads, 2);
    assert_eq!(header.head_dim, 2);
    assert_eq!(header.eos_token_id, 3);
    assert_eq!(header.bos_token_id, 1);
    assert!((header.rms_norm_eps - 1e-6).abs() < 1e-18);
    assert!((header.rope_theta - 10_000.0).abs() < 1e-9);
    assert_eq!(
        header.norm_slots,
        vec!["input", "post_attention", "q_norm", "k_norm"]
    );
    assert_eq!(header.layers[0], "layers.0.inner.self_attn.q_proj");
    assert_eq!(header.weight_tying.as_deref(), Some("lm_head=e1@fp32"));
    assert_eq!(header.remainder_dtype, RemainderDtype::Float32);
    assert!(!header.head_is_untied(), "the sentinel e1 is the tied form");
}

#[test]
fn no_wallclock_timestamp_bf2_2() {
    let blob = fixture();
    let text = header_json(&blob);
    for banned in ["timestamp", "time", "date", "epoch", "created_at", "mtime"] {
        assert!(
            !text.contains(banned),
            "the header carries no wall-clock field, found {banned:?} in {text}"
        );
    }
    assert_eq!(
        header_value(&blob)["created_by"],
        json!("quantibbler-gen-server-fixtures"),
        "created_by names the producing tool, never a time"
    );
}

#[test]
fn rejects_missing_required_header_field_bf2_4() {
    let (header, sections) = parts();
    let mut value: Value = serde_json::from_str(&header).expect("header");
    for field in [
        "model_id",
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "eos_token_id",
        "bos_token_id",
        "rms_norm_eps",
        "rope_theta",
        "norm_slots",
        "layers",
        "weight_tying",
        "remainder_dtype",
    ] {
        let mut stripped = value.clone();
        stripped.as_object_mut().expect("object").remove(field);
        let blob = build(&with_header(&stripped), &refs(&sections));
        let err = Blob::parse_bytes(&blob).expect_err("a missing required field refuses");
        assert!(
            err.contains(field),
            "the refusal must name {field:?}: {err}"
        );
    }
    // A wrong-typed field: d_model as a string.
    value["d_model"] = json!("four");
    let blob = build(&with_header(&value), &refs(&sections));
    let err = Blob::parse_bytes(&blob).expect_err("a wrong-typed field refuses");
    assert!(err.contains("d_model"), "{err}");
    assert!(
        err.contains("integer"),
        "the diagnostic names the type: {err}"
    );
    // A non-object header, and non-JSON bytes.
    let blob = build("[]", &refs(&sections));
    assert!(Blob::parse_bytes(&blob).is_err());
    let mut bad = fixture();
    let len = header_len(&bad);
    bad[12..14].copy_from_slice(b"{["); // still UTF-8, no longer JSON
    assert!(
        Blob::parse_bytes(&bad).is_err(),
        "header_len {len} bytes of non-JSON"
    );
}

// --- BF-3: section table ---------------------------------------------------

#[test]
fn rejects_unknown_dtype_tag_bf3_1() {
    let mut blob = fixture();
    let list = entries(&blob);
    let tag_pos = entry(&list, "embedding").tag_pos;
    blob[tag_pos] = 9;
    let err = Blob::parse_bytes(&blob).expect_err("an unknown tag refuses");
    assert!(err.contains('9'), "the refusal names the tag: {err}");
}

#[test]
fn rejects_offset_length_past_eof_bf3_2() {
    // Overflow: offset 2^64-100 + length 1000.
    let mut blob = fixture();
    let list = entries(&blob);
    let e = entry(&list, "embedding");
    set_u64(&mut blob, e.offset_pos, u64::MAX - 100);
    set_u64(&mut blob, e.length_pos, 1000);
    let err = Blob::parse_bytes(&blob).expect_err("u64 overflow refuses");
    assert!(err.contains("overflows"), "{err}");

    // Past EOF at a page-aligned offset.
    let mut blob = fixture();
    let list = entries(&blob);
    let e = entry(&list, "embedding");
    set_u64(&mut blob, e.length_pos, 10_000_000);
    let err = Blob::parse_bytes(&blob).expect_err("a range past EOF refuses");
    assert!(err.contains("embedding"), "{err}");
    assert!(err.contains("past the"), "{err}");
}

#[test]
fn rejects_overlapping_sections_bf3_3() {
    let mut blob = fixture();
    let list = entries(&blob);
    let first = entry(&list, "layers.0.inner.self_attn.q_proj");
    let first_offset = u64::from_le_bytes(
        blob[first.offset_pos..first.offset_pos + 8]
            .try_into()
            .unwrap(),
    );
    let emb = entry(&list, "embedding");
    set_u64(&mut blob, emb.offset_pos, first_offset);
    let err = Blob::parse_bytes(&blob).expect_err("overlap refuses");
    assert!(err.contains("overlap"), "{err}");
    assert!(err.contains("embedding"), "names both sections: {err}");
}

#[test]
fn rejects_duplicate_section_name_bf3_4() {
    let (header, sections) = parts();
    let mut doubled = sections.clone();
    let clone = doubled[0].clone();
    doubled.push(clone);
    let blob = build(&header, &refs(&doubled));
    let err = Blob::parse_bytes(&blob).expect_err("a duplicate name refuses");
    assert!(err.contains("duplicate"), "{err}");
    assert!(err.contains("q_proj"), "{err}");
}

#[test]
fn rejects_misaligned_offset_bf3_5() {
    let mut blob = fixture();
    let list = entries(&blob);
    let e = entry(&list, "embedding");
    let offset = u64::from_le_bytes(blob[e.offset_pos..e.offset_pos + 8].try_into().unwrap());
    set_u64(&mut blob, e.offset_pos, offset + 1);
    set_u64(&mut blob, e.length_pos, 64); // keep the range inside EOF
    let err = Blob::parse_bytes(&blob).expect_err("a misaligned offset refuses");
    assert!(err.contains("4096"), "{err}");
    assert!(err.contains("embedding"), "{err}");
}

#[test]
fn rejects_truncated_section_table_bf3_6() {
    let blob = fixture();
    let start = table_start(&blob);
    let truncated = blob[..start + 10].to_vec();
    let err = Blob::parse_bytes(&truncated).expect_err("a table cut mid-entry refuses");
    assert!(err.contains("truncated section table"), "{err}");
}

#[test]
fn rejects_unknown_section_name_bf3_7() {
    let (header, sections) = parts();
    let mut extra = sections.clone();
    extra.insert(0, ("bogus.section".to_string(), 4, vec![1, 2, 3]));
    let blob = build(&header, &refs(&extra));
    let err = Blob::parse_bytes(&blob).expect_err("an unknown name refuses");
    assert!(err.contains("bogus.section"), "{err}");
}

#[test]
fn blob_section_roundtrips_ct_1_4() {
    let (header, sections) = parts();
    let template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
    // The committed fixture already carries a sidecar; these legs craft their
    // own payload, so each starts from the same section list without one.
    let without_sidecar = |sections: &[(String, u8, Vec<u8>)]| -> Vec<(String, u8, Vec<u8>)> {
        sections
            .iter()
            .filter(|(name, _, _)| name != "chat_template.jinja")
            .cloned()
            .collect()
    };
    // The sidecar rides as its own raw section, after the tokenizer sections
    // in canonical order.
    let mut with_template = without_sidecar(&sections);
    with_template.push((
        "chat_template.jinja".to_string(),
        4,
        template.as_bytes().to_vec(),
    ));
    let blob = build(&header, &refs(&with_template));
    let dir = load_variant("template", &blob).expect("the blob carrying a sidecar loads");
    assert_eq!(dir.chat_template_jinja.as_deref(), Some(template));

    // Payload that is not UTF-8 refuses, naming the section.
    let mut bad = without_sidecar(&sections);
    bad.push(("chat_template.jinja".to_string(), 4, vec![0xff, 0xfe, 0xfd]));
    let blob = build(&header, &refs(&bad));
    let err = load_variant("bad-template", &blob).expect_err("non-UTF-8 refuses");
    assert!(
        err.contains("chat_template.jinja"),
        "names the section: {err}"
    );
    assert!(err.contains("UTF-8"), "names the reason: {err}");

    // A blank sidecar reads as "no template declared", never as an empty one.
    let mut blank = without_sidecar(&sections);
    blank.push(("chat_template.jinja".to_string(), 4, b"\n \t\n".to_vec()));
    let blob = build(&header, &refs(&blank));
    let dir = load_variant("blank-template", &blob).expect("a blank sidecar loads");
    assert_eq!(dir.chat_template_jinja, None);

    // And the committed fixture's own sidecar resolves from the blob, exactly
    // as the same model's directory resolves it.
    let dir = load_variant("committed-template", &fixture()).expect("the fixture blob loads");
    let on_disk = std::fs::read_to_string(format!("{DIR_FIXTURE}/chat_template.jinja"))
        .expect("the fixture directory's sidecar");
    assert_eq!(dir.chat_template_jinja.as_deref(), Some(on_disk.as_str()));
}

// --- BF-4 / BF-5 / BF-7 ----------------------------------------------------

#[test]
fn offsets_are_4096_aligned_bf4_1() {
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("parses");
    assert!(!parsed.sections.is_empty());
    for section in &parsed.sections {
        assert_eq!(
            section.offset % PAGE_SIZE,
            0,
            "section {:?} offset {} is not page aligned",
            section.name,
            section.offset
        );
    }
}

#[test]
fn padding_bytes_are_zero_bf4_2() {
    // A valid blob has zero padding by construction; flipping one pad byte
    // must refuse rather than be ignored.
    let mut blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("parses");
    let header_end = parsed.header_end;
    assert_eq!(blob[header_end], 0, "header pad starts zero");
    blob[header_end] = 1;
    let err = Blob::parse_bytes(&blob).expect_err("non-zero header padding refuses");
    assert!(err.contains("padding") || err.contains("non-zero"), "{err}");
}

#[test]
fn rejects_metadata_aliasing_section_bf7_1() {
    let mut blob = fixture();
    let list = entries(&blob);
    let e = entry(&list, "embedding");
    set_u64(&mut blob, e.offset_pos, 0);
    let err = Blob::parse_bytes(&blob).expect_err("a section aliasing the metadata refuses");
    assert!(err.contains("metadata"), "{err}");
    assert!(err.contains("embedding"), "{err}");
}

#[test]
fn rejects_trailing_bytes_bf7_2() {
    let mut blob = fixture();
    blob.push(0);
    let err = Blob::parse_bytes(&blob).expect_err("trailing bytes refuse");
    assert!(err.contains("trailing bytes"), "{err}");
}

#[test]
fn sections_in_canonical_order_bf5_1() {
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("parses");
    let names: Vec<&str> = parsed.sections.iter().map(|s| s.name.as_str()).collect();
    let mut expected: Vec<String> = parsed.header.layers.clone();
    expected.push("embedding".into());
    expected.push("norms".into());
    expected.push("tokenizer.json".into());
    // The chat-template sidecar is the last canonical key: it is neither a
    // layer nor a `tokenizer.*` file, and the fixture carries it.
    expected.push("chat_template.jinja".into());
    assert_eq!(
        names,
        expected.iter().map(|s| s.as_str()).collect::<Vec<_>>()
    );

    // Out of canonical order refuses.
    let (header, sections) = parts();
    let mut shuffled = sections.clone();
    shuffled.swap(0, 1);
    let blob = build(&header, &refs(&shuffled));
    let err = Blob::parse_bytes(&blob).expect_err("a shuffled table refuses");
    assert!(err.contains("canonical order"), "{err}");
}

#[test]
fn pack_twice_hashes_equal_bf6_1() {
    // The production writer is the host tool (its own pack-twice pin is
    // BW-2.1 in tests/ptq/test_blob_export.py); this pins the LAYOUT's
    // determinism — no field depends on wall-clock time or iteration order.
    let (header, sections) = parts();
    let a = build(&header, &refs(&sections));
    let b = build(&header, &refs(&sections));
    assert_eq!(a, b, "two builds of the same inputs are byte-identical");
    assert_eq!(a.len(), fixture().len());
}

#[test]
fn stream_extent_still_one_layer_sf6_1() {
    // The MODIFIED non-goal: a stream is exactly one layer, and bytes beyond
    // the declared extent are rejected (the blob wraps streams, it does not
    // concatenate them into one stream).
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("parses");
    let q = parsed
        .section("layers.0.inner.self_attn.q_proj")
        .expect("section");
    let stream = parsed.slice(&blob, q);
    let mut two = stream.to_vec();
    two.extend_from_slice(stream);
    assert!(
        quantibbler_qpk1::parse(&two).is_err(),
        "two concatenated layers are not one stream"
    );
}

#[test]
fn no_gpu_layout_sf6_2() {
    // The MODIFIED non-goal: no qpk1 field encodes a GPU/device layout — the
    // stream's extent is exactly header + scales + bias + bitplanes.
    let dout = 4usize;
    let din = 4usize;
    let r = 2usize;
    let exact =
        quantibbler_qpk1::expected_size(dout, din, r, quantibbler_qpk1::ScaleDtype::Float32, true);
    let blob = fixture();
    let parsed = Blob::parse_bytes(&blob).expect("parses");
    let q = parsed
        .section("layers.0.inner.self_attn.q_proj")
        .expect("section");
    let stream = parsed.slice(&blob, q);
    // The fixture layer is dout=4, din=4, r=4 float16 (44 bytes); one byte
    // more than its own declared extent is trailing garbage, not a layout.
    let declared =
        quantibbler_qpk1::expected_size(4, 4, 4, quantibbler_qpk1::ScaleDtype::Float16, false);
    assert_eq!(stream.len(), declared);
    let mut longer = stream.to_vec();
    longer.push(0);
    assert!(quantibbler_qpk1::parse(&longer).is_err());
    let _ = exact;
}

// --- BL: the loader's blob path --------------------------------------------

fn assert_same_directory(a: &ModelDirectory, b: &ModelDirectory) {
    assert_eq!(a.model_id, b.model_id);
    assert_eq!(a.layers.len(), b.layers.len());
    for ((na, la), (nb, lb)) in a.layers.iter().zip(&b.layers) {
        assert_eq!(na, nb);
        assert_layer_eq(la, lb);
    }
    assert_eq!(a.eos_token_id, b.eos_token_id);
    assert_eq!(a.bos_token_id, b.bos_token_id);
    assert_eq!(a.vocab_size, b.vocab_size);
    assert_eq!(a.d_model, b.d_model);
    assert_eq!(a.n_layers, b.n_layers);
    assert_eq!(a.n_heads, b.n_heads);
    assert_eq!(a.n_kv_heads, b.n_kv_heads);
    assert_eq!(a.head_dim, b.head_dim);
    assert_eq!(a.norm_slots, b.norm_slots);
    assert_eq!(a.remainder_dtype, b.remainder_dtype);
    assert_eq!(a.embedding, b.embedding);
    assert_eq!(a.lm_head, b.lm_head);
    assert_eq!(a.norm_weights, b.norm_weights);
}

fn assert_layer_eq(a: &Layer, b: &Layer) {
    assert_eq!(a.version, b.version);
    assert_eq!((a.dout, a.din, a.r), (b.dout, b.din, b.r));
    assert_eq!(a.s1, b.s1);
    assert_eq!(a.s2, b.s2);
    assert_eq!(a.bias, b.bias);
    assert_eq!(a.ub_packed, b.ub_packed);
    assert_eq!(a.vb_packed, b.vb_packed);
}

#[test]
fn loads_blob_file_bl1_1_bl2_1() {
    let from_blob =
        ModelDirectory::load(std::path::Path::new(BLOB_FIXTURE)).expect("the committed blob loads");
    let from_dir =
        ModelDirectory::load(std::path::Path::new(DIR_FIXTURE)).expect("the directory loads");
    assert_same_directory(&from_blob, &from_dir);
    // The blob CARRIES its tokenizer as a section (B9); the directory does not.
    assert!(from_blob.tokenizer_json.is_some());
    assert!(from_dir.tokenizer_json.is_none());
    assert_eq!(
        from_blob.tokenizer_json.as_deref(),
        Some(
            std::fs::read(std::path::Path::new(DIR_FIXTURE).join("tokenizer.json"))
                .unwrap()
                .as_slice()
        )
    );
}

#[test]
fn directory_still_loads_bl1_2() {
    let dir = ModelDirectory::load(std::path::Path::new(DIR_FIXTURE)).expect("directory loads");
    assert_eq!(dir.layers.len(), 7);
    assert_eq!(dir.model_id, "server-model-fixture");
}

#[test]
fn rejects_non_blob_regular_file_bl1_3() {
    let path = std::env::temp_dir().join(format!("qpk1-not-a-blob-{}", std::process::id()));
    std::fs::write(&path, b"not a blob at all").expect("write");
    let err = ModelDirectory::load(&path).expect_err("a non-blob regular file refuses");
    assert!(err.contains("not a qpk1 blob"), "{err}");
    assert!(
        err.contains(&*path.display().to_string()),
        "names the path: {err}"
    );
    std::fs::remove_file(&path).expect("cleanup");
}

/// Write a re-packed variant to a temp file and load it.
fn load_variant(tag: &str, blob: &[u8]) -> Result<ModelDirectory, String> {
    let path = std::env::temp_dir().join(format!("qpk1-blob-{tag}-{}", std::process::id()));
    std::fs::write(&path, blob).expect("write variant");
    let result = ModelDirectory::load(&path);
    std::fs::remove_file(&path).expect("cleanup");
    result
}

#[test]
fn rejects_missing_section_bl2_2() {
    let (header, sections) = parts();
    for missing in ["norms", "embedding", "tokenizer.json"] {
        let kept: Vec<_> = sections
            .iter()
            .filter(|(n, _, _)| n != missing)
            .cloned()
            .collect();
        let blob = build(&header, &refs(&kept));
        let err = load_variant("missing", &blob).expect_err("a missing required section refuses");
        assert!(
            err.contains(missing),
            "the refusal names the missing section {missing:?}: {err}"
        );
    }
    // A missing LAYER section too.
    let kept: Vec<_> = sections
        .iter()
        .filter(|(n, _, _)| n != "layers.0.inner.mlp.down_proj")
        .cloned()
        .collect();
    let blob = build(&header, &refs(&kept));
    let err = load_variant("missing-layer", &blob).expect_err("a missing layer refuses");
    assert!(err.contains("down_proj"), "{err}");
}

#[test]
fn rejects_wrong_remainder_length_bl2_3() {
    let (header, sections) = parts();
    let mut shortened = sections.clone();
    for (name, _, data) in shortened.iter_mut() {
        if name == "embedding" {
            data.truncate(data.len() - 4);
        }
    }
    let blob = build(&header, &refs(&shortened));
    let err = load_variant("short-emb", &blob).expect_err("a wrong-length remainder refuses");
    assert!(err.contains("embedding"), "{err}");
    assert!(
        err.contains("expected exactly"),
        "names the expected count: {err}"
    );
}

#[test]
fn dtype_suffix_agrees_bl3_1() {
    let (header, sections) = parts();
    let mut value: Value = serde_json::from_str(&header).expect("header");
    value["remainder_dtype"] = json!("bf16");
    value["weight_tying"] = json!("lm_head=e1@bf16");
    let mut converted = sections.clone();
    for (name, tag, data) in converted.iter_mut() {
        if matches!(name.as_str(), "embedding" | "norms") {
            *tag = BlobTag::Bf16Le.as_u8();
            *data = bf16_bytes(&f32_floats(data));
        }
    }
    let blob = build(&with_header(&value), &refs(&converted));
    let model = load_variant("bf16", &blob).expect("a bf16 blob with an agreeing suffix loads");
    assert_eq!(model.remainder_dtype, RemainderDtype::Bf16);
    assert_eq!(model.embedding.len(), 32 * 4);
}

#[test]
fn rejects_suffix_mismatch_bl3_2() {
    let (header, sections) = parts();
    let mut value: Value = serde_json::from_str(&header).expect("header");
    value["remainder_dtype"] = json!("fp32");
    value["weight_tying"] = json!("lm_head=e1@bf16");
    let blob = build(&with_header(&value), &refs(&sections));
    let err = load_variant("suffix", &blob).expect_err("a suffix disagreement refuses");
    assert!(err.contains("bf16") && err.contains("fp32"), "{err}");
}

#[test]
fn rejects_remainder_tag_mismatch_bl3_3() {
    let (header, sections) = parts();
    let mut retagged = sections.clone();
    for (name, tag, data) in retagged.iter_mut() {
        if name == "embedding" {
            *tag = BlobTag::Bf16Le.as_u8();
            *data = bf16_bytes(&f32_floats(data));
        }
    }
    let blob = build(&header, &refs(&retagged));
    let err =
        load_variant("tag-mismatch", &blob).expect_err("a remainder tag disagreement refuses");
    assert!(err.contains("embedding"), "{err}");
    assert!(err.contains("bf16_le"), "names the tag: {err}");
}

#[test]
fn rejects_layer_version_tag_mismatch_bl3_4() {
    let (header, sections) = parts();
    let mut retagged = sections.clone();
    retagged[0].1 = BlobTag::Qpk1V2.as_u8(); // the fixture's streams are version 1
    let blob = build(&header, &refs(&retagged));
    let err = load_variant("layer-version", &blob).expect_err("a layer tag disagreement refuses");
    assert!(err.contains("qpk1_v2"), "{err}");
    assert!(err.contains("version 1"), "{err}");
}

#[test]
fn null_weight_tying_reads_declared_dtype_bl3_5() {
    let (header, sections) = parts();
    let mut value: Value = serde_json::from_str(&header).expect("header");
    value["remainder_dtype"] = json!("bf16");
    value["weight_tying"] = Value::Null;
    let mut converted = sections.clone();
    for (name, tag, data) in converted.iter_mut() {
        if matches!(name.as_str(), "embedding" | "norms") {
            *tag = BlobTag::Bf16Le.as_u8();
            *data = bf16_bytes(&f32_floats(data));
        }
    }
    // An untied head ships its own "lm_head" section (B8), at bf16.
    let embedding = converted
        .iter()
        .find(|(n, _, _)| n == "embedding")
        .expect("embedding")
        .clone();
    let norms_pos = converted
        .iter()
        .position(|(n, _, _)| n == "norms")
        .expect("norms");
    converted.insert(
        norms_pos,
        (
            "lm_head".to_string(),
            BlobTag::Bf16Le.as_u8(),
            embedding.2.clone(),
        ),
    );
    let blob = build(&with_header(&value), &refs(&converted));
    let model = load_variant("null-tying", &blob).expect("a null weight_tying loads");
    assert_eq!(model.remainder_dtype, RemainderDtype::Bf16);
    let head = model.lm_head.expect("the untied head is read, not tied");
    assert_eq!(head.len(), 32 * 4);
}

#[test]
fn blob_loads_with_mmap_bl4_1() {
    // The blob load path maps the file (load_blob); a successful load proves
    // every section offset passed the alignment assertion first.
    let model = quantibbler_qpk1::loader::load_blob(std::path::Path::new(BLOB_FIXTURE))
        .expect("the blob maps and loads");
    assert_eq!(model.layers.len(), 7);
}

#[test]
fn misaligned_blob_refused_bl4_2() {
    let mut blob = fixture();
    let list = entries(&blob);
    let e = entry(&list, "embedding");
    let offset = u64::from_le_bytes(blob[e.offset_pos..e.offset_pos + 8].try_into().unwrap());
    set_u64(&mut blob, e.offset_pos, offset + 8);
    set_u64(&mut blob, e.length_pos, 32);
    let err = load_variant("misaligned", &blob).expect_err("a misaligned blob refuses");
    assert!(err.contains("4096"), "{err}");
    assert!(err.contains("embedding"), "{err}");
}
