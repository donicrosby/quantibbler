//! qk-serve-bf16 — the server consumes dtype-declared blob artifacts: the
//! blob load path (BS-1), the resolved serving dtype (BS-2), the artifact
//! selector (SR-1.3/1.4/1.5), and the fp32 no-regression contract (BS-8.2).
//!
//! The bf16 fixtures come from the shipped `tools/repack_remainder.py`, so the
//! two blobs differ ONLY in the remainder sections and their declaration.

#![allow(clippy::duplicate_mod)] // design D1: tests embed the binary's modules (no lib target)

#[path = "../src/api/mod.rs"]
pub mod api;
#[path = "../src/cli.rs"]
pub mod cli;
#[path = "../src/engine/mod.rs"]
pub mod engine;
#[path = "../src/model/mod.rs"]
pub mod model;
#[path = "../src/sampler.rs"]
pub mod sampler;

mod common;

use std::path::Path;

use common::*;
use quantibbler_qpk1::Blob;

/// Token ids inside the fixture's 32-token vocabulary.
const PROBE: [u32; 4] = [1, 5, 7, 9];

/// Prefill logits over a fixed probe prompt — the strongest available
/// "identical model" read (bitwise, not just argmax).
fn prefill_logits(artifact: &Path) -> Vec<f32> {
    let loaded = model::load(artifact).unwrap_or_else(|e| panic!("{}: {e}", artifact.display()));
    let w = &loaded.weights;
    let mut cache = engine::KvCache::new(w.n_layers, w.n_kv_heads * w.head_dim);
    w.forward_prefill(&PROBE, &mut cache)
        .expect("transformer forward")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// SR-1.3 + BS-1.1: a `QKPB` file is auto-detected, loaded through mmap, and
/// produces the model a directory load of the same artifact produces.
#[test]
fn blob_auto_detected_via_mmap_and_identical_to_the_directory_sr_1_3() {
    let (loaded, method) = (
        model::load(&blob_fixture()).expect("blob loads"),
        model::load(&blob_fixture())
            .expect("blob loads")
            .blob_load_method,
    );
    let dir = model::load(&fixture_dir()).expect("directory loads");
    assert_eq!(
        method,
        Some(model::BlobLoadMethod::Mmap),
        "a local blob must load through mmap (BS-1.1)"
    );
    assert_eq!(dir.blob_load_method, None, "a directory is never mapped");
    assert_eq!(loaded.id, dir.id);
    assert_eq!(loaded.d_model, dir.d_model);
    assert_eq!(loaded.vocab_size, dir.vocab_size);
    assert_eq!(loaded.weights.layers.len(), dir.weights.layers.len());
    assert_eq!(
        bits(&prefill_logits(&blob_fixture())),
        bits(&prefill_logits(&fixture_dir())),
        "blob and directory loads must be bitwise identical (SR-1.3)"
    );
}

/// SR-1.4: the legacy directory form keeps working through `--artifact`.
#[test]
fn directory_is_still_loadable_sr_1_4() {
    let loaded = model::load(&fixture_dir()).expect("directory loads");
    assert_eq!(loaded.id, FIXTURE_MODEL_ID);
    assert_eq!(loaded.weights.serving_dtype, model::ServingDtype::Fp32);
}

/// SR-1.5: both selectors supplied is refused at parse, naming both flags, and
/// nothing is loaded.
#[test]
fn both_artifact_selectors_are_refused_at_parse_sr_1_5() {
    let exe = env!("CARGO_BIN_EXE_quantibbler-server");
    let out = std::process::Command::new(exe)
        .args([
            "--artifact",
            &blob_fixture().to_string_lossy(),
            "--model-dir",
            &fixture_dir().to_string_lossy(),
            "--port",
            "19999",
        ])
        .output()
        .expect("run the binary");
    assert!(!out.status.success(), "both selectors must be refused");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("--artifact") && text.contains("--model-dir"),
        "the usage error must name both flags: {text}"
    );
}

/// BS-2.1 / BS-2.2 / BS-2.5: the DECLARATION resolves the serving dtype — a
/// blob follows its header, and a bf16-declared DIRECTORY widens and never
/// selects native bf16.
#[test]
fn serving_dtype_follows_the_declaration_bs_2() {
    let fp32_blob = model::load(&blob_fixture()).expect("fp32 blob");
    assert_eq!(fp32_blob.weights.serving_dtype, model::ServingDtype::Fp32);

    let bf16_blob = model::load(&bf16_blob_fixture()).expect("bf16 blob");
    assert_eq!(
        bf16_blob.weights.serving_dtype,
        model::ServingDtype::Bf16,
        "a bf16-declared blob serves natively (BS-2.1)"
    );
    // The bf16 artifact's own declared dtype is bf16...
    let parsed = Blob::parse_bytes(&std::fs::read(bf16_blob_fixture()).expect("fixture bytes"))
        .expect("fixture parses");
    assert_eq!(
        parsed.header.remainder_dtype,
        quantibbler_qpk1::RemainderDtype::Bf16
    );

    // ...while the bf16 DIRECTORY twin declares bf16 but widens (BS-2.5).
    let dir = model::load(&bf16_fixture_dir()).expect("bf16 directory");
    assert_eq!(
        dir.weights.serving_dtype,
        model::ServingDtype::Fp32,
        "native bf16 serving is blob-only (BS-2.5)"
    );
}

// --- a compact test-side blob builder (the negatives BS-2.3/BS-2.4 need) ---

/// Lay out a blob from a header JSON string and named/tagged payloads — the
/// same byte rules the writer follows, written independently here so a parser
/// bug cannot make its own negative test pass.
fn build_blob(header_json: &str, sections: &[(String, u8, Vec<u8>)]) -> Vec<u8> {
    const PAGE: usize = 4096;
    fn pad(out: &mut Vec<u8>) {
        let rem = out.len() % PAGE;
        if rem != 0 {
            out.extend(std::iter::repeat_n(0u8, PAGE - rem));
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"QKPB");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header_json.len() as u32).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    pad(&mut out);
    let table_len = 4 + sections
        .iter()
        .map(|(n, _, _)| 1 + n.len() + 1 + 8 + 8)
        .sum::<usize>();
    let mut offset = out.len() + table_len;
    offset += (PAGE - offset % PAGE) % PAGE;
    let mut offsets = Vec::new();
    let mut cursor = offset;
    for (i, (_, _, data)) in sections.iter().enumerate() {
        offsets.push(cursor);
        cursor += data.len();
        if i + 1 < sections.len() {
            cursor += (PAGE - cursor % PAGE) % PAGE;
        }
    }
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for ((name, tag, data), at) in sections.iter().zip(&offsets) {
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.push(*tag);
        out.extend_from_slice(&(*at as u64).to_le_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    }
    pad(&mut out);
    for (i, (_, _, data)) in sections.iter().enumerate() {
        out.resize(offsets[i], 0);
        out.extend_from_slice(data);
    }
    out
}

/// Re-emit the committed blob with its header JSON mutated, payloads verbatim.
fn reheader(mutate: impl Fn(&mut serde_json::Map<String, serde_json::Value>)) -> Vec<u8> {
    let bytes = std::fs::read(blob_fixture()).expect("fixture bytes");
    let blob = Blob::parse_bytes(&bytes).expect("fixture parses");
    let h = &blob.header;
    let mut map = serde_json::Map::new();
    map.insert("model_id".into(), h.model_id.clone().into());
    for (key, value) in [
        ("vocab_size", h.vocab_size),
        ("d_model", h.d_model),
        ("n_layers", h.n_layers),
        ("n_heads", h.n_heads),
        ("n_kv_heads", h.n_kv_heads),
        ("head_dim", h.head_dim),
        ("eos_token_id", h.eos_token_id as usize),
        ("bos_token_id", h.bos_token_id as usize),
    ] {
        map.insert(key.into(), value.into());
    }
    map.insert("rms_norm_eps".into(), h.rms_norm_eps.into());
    map.insert("rope_theta".into(), h.rope_theta.into());
    map.insert("norm_slots".into(), serde_json::json!(h.norm_slots));
    map.insert("layers".into(), serde_json::json!(h.layers));
    map.insert(
        "weight_tying".into(),
        match &h.weight_tying {
            Some(w) => w.clone().into(),
            None => serde_json::Value::Null,
        },
    );
    map.insert("remainder_dtype".into(), h.remainder_dtype.as_str().into());
    mutate(&mut map);
    let header = serde_json::to_string(&serde_json::Value::Object(map)).expect("header JSON");
    let sections: Vec<(String, u8, Vec<u8>)> = blob
        .sections
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.tag.as_u8(),
                blob.slice(&bytes, s).to_vec(),
            )
        })
        .collect();
    build_blob(&header, &sections)
}

/// Poll a spawned server's captured stdout for `needle` (the drain thread
/// appends asynchronously), then return the log either way so the failure
/// message carries what was actually logged.
fn wait_for_log(server: &ServerProcess, needle: &str) -> String {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let log = server.startup_log();
        if log.contains(needle) || std::time::Instant::now() > deadline {
            return log;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn temp_blob(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("qk-serve-bf16-{}-{name}.qpk", std::process::id()));
    std::fs::write(&path, bytes).expect("write temp blob");
    path
}

/// BS-2.3: a header that parses but omits `remainder_dtype` refuses startup
/// with a diagnostic naming the field.
#[test]
fn missing_remainder_dtype_is_refused_bs_2_3() {
    let bytes = reheader(|map| {
        map.remove("remainder_dtype");
    });
    let path = temp_blob("missing-dtype", &bytes);
    let err = model::load(&path)
        .err()
        .expect("a header without remainder_dtype must refuse");
    assert!(
        err.contains("remainder_dtype"),
        "the refusal must name the missing field: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

/// BS-2.4: a header declaring bf16 whose remainder sections are tagged fp32 is
/// refused, naming the section and both values.
#[test]
fn header_section_dtype_mismatch_is_refused_bs_2_4() {
    // `weight_tying`'s suffix is the second declaration of the same bytes, so
    // it moves WITH the header here; the refusal under test is the SECTION
    // TABLE's, not the two declarations disagreeing with each other.
    let bytes = reheader(|map| {
        map.insert("remainder_dtype".into(), "bf16".into());
        map.insert("weight_tying".into(), "lm_head=e1@bf16".into());
    });
    let path = temp_blob("mismatch", &bytes);
    let err = model::load(&path)
        .err()
        .expect("a header/section dtype mismatch must refuse");
    assert!(
        err.contains("bf16") && err.contains("fp32_le"),
        "the refusal must name both values: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

/// BS-1.2 / BS-1.3: the read fallback (forced through the test-only hatch)
/// yields the same served model as mmap, and the startup log records which
/// source was used.
#[test]
fn read_fallback_matches_mmap_and_is_logged_bs_1_3() {
    let mmap_srv = spawn_server_artifact(&blob_fixture(), &[], &[]);
    assert!(wait_ready(
        mmap_srv.port,
        std::time::Duration::from_secs(30)
    ));
    let mmap_body = post_json(
        mmap_srv.port,
        "/v1/completions",
        &completion_body(FIXTURE_MODEL_ID, false),
    )
    .expect("completion");
    assert_eq!(mmap_body.status, 200);

    let read_srv = spawn_server_artifact(&blob_fixture(), &[], &[(FORCE_READ_FALLBACK_ENV, "1")]);
    assert!(wait_ready(
        read_srv.port,
        std::time::Duration::from_secs(30)
    ));
    let read_body = post_json(
        read_srv.port,
        "/v1/completions",
        &completion_body(FIXTURE_MODEL_ID, false),
    )
    .expect("completion");
    assert_eq!(read_body.status, 200);

    // The log records the byte source each leg used (BS-1.1 / BS-1.2). The
    // drain thread appends asynchronously, so poll for the line instead of
    // racing it.
    let mmap_log = wait_for_log(&mmap_srv, "loaded blob via mmap");
    assert!(
        mmap_log
            .lines()
            .any(|line| line.contains("load_method") && line.contains("mmap")),
        "startup log must record the mmap source: {mmap_log}"
    );
    let read_log = wait_for_log(&read_srv, "fell back to std::fs::read");
    assert!(
        read_log
            .lines()
            .any(|line| line.contains("load_method") && line.contains("fell back to std::fs::read")),
        "the read fallback must be logged at WARN with its reason: {read_log}"
    );

    // The two loaded models are identical: same completion JSON modulo the
    // per-request `created`/`id` fields and the timing block. `timings` is
    // excluded by contract (BS-8.2) — its values jitter with scheduling, so
    // comparing them makes this leg fail whenever the suite runs under load.
    let strip = |body: &str| -> serde_json::Value {
        let mut v: serde_json::Value = serde_json::from_str(body).expect("JSON body");
        if let Some(o) = v.as_object_mut() {
            o.remove("created");
            o.remove("id");
            o.remove("timings");
        }
        v
    };
    assert_eq!(
        strip(&mmap_body.body),
        strip(&read_body.body),
        "mmap and read fallback must serve identically (BS-1.3)"
    );
}

/// BS-8.1 + BS-8.2: the SAME assertion body runs over the committed directory
/// fixture and the committed fp32 blob fixture — the harness parametrization
/// the requirement asks for — and the fp32 blob answers a completion request
/// byte-identically to the directory (timing fields excluded).
#[test]
fn fp32_blob_and_directory_serve_identical_content_bs_8_1_bs_8_2() {
    let bodies: Vec<serde_json::Value> = [fixture_dir(), blob_fixture()]
        .iter()
        .map(|artifact| {
            let (_state, router) = ready_router_for(artifact);
            let port = serve_on_port(router);
            let resp = post_json(
                port,
                "/v1/completions",
                &completion_body(FIXTURE_MODEL_ID, false),
            )
            .expect("completion");
            assert_eq!(resp.status, 200, "{}: {}", artifact.display(), resp.body);
            let mut value: serde_json::Value =
                serde_json::from_str(&resp.body).expect("JSON completion body");
            let object = value.as_object_mut().expect("object body");
            // Per-request identity and the timing block are excluded by
            // contract (BS-8.2).
            object.remove("created");
            object.remove("id");
            object.remove("timings");
            value
        })
        .collect();
    assert_eq!(
        bodies[0], bodies[1],
        "the fp32 blob and the directory fixture must serve identically"
    );
}
