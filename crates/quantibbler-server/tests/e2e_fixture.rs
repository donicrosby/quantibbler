//! Task 6.1 — end-to-end smoke asserting the fixture stand-in assumption
//! (design IA5) at the reader level: the fixture generator's bytes are
//! checked against the authoritative qk-representation SF-1.3 known-layer
//! stream (tests/test_representation.py), then the fixture model is loaded
//! and served end to end.
//!
//! Any byte-layout divergence between the server's qpk1 reader and the
//! pinned representation bytes is a qk-representation finding, not
//! something patched around here.

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

use common::*;

/// The pinned SF-1.3 stream (dout=3, din=5, r=7, float16) from
/// tests/test_representation.py — the compressor lane's byte contract. It is
/// a VERSION-1 stream: reading it is the declared legacy rule (no bias
/// field; the layer's map is the one it always was).
const SF_1_3_STREAM_HEX: &str = "51504b31\
     01000000\
     01000000\
     03000000\
     05000000\
     07000000\
     003c003800c0\
     003400b80040003c0030\
     2b7307\
     55e6413503";

fn hex_bytes(hex: &str) -> Vec<u8> {
    let clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).expect("hex"))
        .collect()
}

/// IA5, leg 1: the server's reader parses the REAL representation lane's
/// known-layer bytes field-for-field.
#[test]
fn e2e_reader_matches_representation_known_layer() {
    let bytes = hex_bytes(SF_1_3_STREAM_HEX);
    assert_eq!(bytes.len(), 48);
    let layer = quantibbler_qpk1::parse(&bytes).expect("SF-1.3 stream parses");
    assert_eq!(layer.dout, 3);
    assert_eq!(layer.din, 5);
    assert_eq!(layer.r, 7);
    assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
    assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
    assert_eq!(layer.ub_packed, vec![0x2b, 0x73, 0x07]);
    assert_eq!(layer.vb_packed, vec![0x55, 0xe6, 0x41, 0x35, 0x03]);
    // A version-1 stream carries no bias, and an empty bias vector is
    // exactly the zero map a biasless linear applies.
    assert_eq!(layer.version, quantibbler_qpk1::LEGACY_VERSION);
    assert!(layer.bias.is_empty());
    // Reader-side rank feasibility (representation deserialize rule).
    assert!(layer.r > layer.dout.min(layer.din));
}

/// The version-2 pinned vector — the same byte string
/// `tests/test_representation.py`'s Python face produces for the same inputs
/// (dout=3, din=5, r=3, float16, bias `[1.0, -0.5, 2.0]`), parsed by the
/// server's reader. One bias-carrying vector, both faces, same bytes.
const V2_BIAS_STREAM_HEX: &str = "51504b31\
     02000000\
     01000000\
     03000000\
     05000000\
     03000000\
     003c003800c0\
     003400b80040003c0030\
     003c00b80040\
     7301\
     e535";

#[test]
fn e2e_reader_recovers_the_v2_bias_vector() {
    let bytes = hex_bytes(V2_BIAS_STREAM_HEX);
    assert_eq!(bytes.len(), 50);
    let layer = quantibbler_qpk1::parse(&bytes).expect("v2 stream parses");
    assert_eq!(layer.version, quantibbler_qpk1::VERSION);
    assert_eq!(layer.dout, 3);
    assert_eq!(layer.din, 5);
    assert_eq!(layer.r, 3);
    assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
    assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
    assert_eq!(layer.bias, vec![1.0, -0.5, 2.0]);
    assert_eq!(layer.ub_packed, vec![0x73, 0x01]);
    assert_eq!(layer.vb_packed, vec![0xe5, 0x35]);
    // The size the reader demands is the version-2 formula.
    assert_eq!(
        quantibbler_qpk1::expected_size(3, 5, 3, layer.dtype, true),
        bytes.len()
    );
}

/// The bias-carrying fixture (version-2 streams, non-zero bias on every
/// projection) and its pre-bias twin are the SAME model apart from the bias
/// vector, so a forward over identical ids must differ: if the engine
/// dropped the bias on the floor the two forwards would agree exactly.
#[test]
fn e2e_bias_carrying_fixture_forward_includes_the_bias() {
    let bias_dir = common::bias_fixture_dir();

    // The stream itself: version 2, a full-width bias vector, not all zero.
    let raw = std::fs::read(bias_dir.join("layers/q_proj.qpk1")).expect("layer bytes");
    let parsed = quantibbler_qpk1::parse(&raw).expect("v2 stream parses");
    assert_eq!(parsed.version, quantibbler_qpk1::VERSION);
    assert_eq!(parsed.bias.len(), parsed.dout);
    assert!(parsed.bias.iter().any(|b| *b != 0.0));

    // The load path carries it: every projection's bias is full-width and
    // non-zero, and the legacy twin's is exactly zero.
    let biased = model::load(&bias_dir).expect("bias fixture loads");
    assert!(
        biased
            .weights
            .layers
            .iter()
            .all(|l| l.bias_f32.len() == l.dout && l.bias_f32.iter().any(|b| *b != 0.0))
    );
    let legacy = model::load(&fixture_dir()).expect("legacy fixture loads");
    assert!(
        legacy
            .weights
            .layers
            .iter()
            .all(|l| l.bias_f32.iter().all(|b| *b == 0.0))
    );

    let ids: Vec<u32> = vec![5, 17, 3, 21];
    let mut cache_b = engine::KvCache::new(
        biased.weights.n_layers,
        biased.weights.n_kv_heads * biased.weights.head_dim,
    );
    let y_bias = biased
        .weights
        .forward_prefill(&ids, &mut cache_b)
        .expect("biased forward");
    let mut cache_l = engine::KvCache::new(
        legacy.weights.n_layers,
        legacy.weights.n_kv_heads * legacy.weights.head_dim,
    );
    let y0 = legacy
        .weights
        .forward_prefill(&ids, &mut cache_l)
        .expect("legacy forward");
    assert_eq!(y0.len(), y_bias.len());
    let max_diff = y0
        .iter()
        .zip(&y_bias)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "the bias never reached the forward: biased and legacy logits agree to {max_diff}"
    );
}

/// IA5, leg 2: the fixture dir's layer stream satisfies the pinned size
/// formula and the server loads + serves it end to end
/// (quantize -> serialize -> serve -> complete, fixture stand-in).
#[test]
#[ignore = "generation rewires onto the transformer forward with qk-attention 2.2/2.3"]
fn e2e_fixture_model_serves_end_to_end() {
    // Size formula (qk-representation SF-1.1).
    let layer_path = fixture_dir().join("layers/q_proj.qpk1");
    let raw = std::fs::read(&layer_path).expect("fixture layer bytes");
    let parsed = quantibbler_qpk1::parse(&raw).expect("fixture stream parses");
    // The crate's own formula, at the stream's OWN version (so a legacy v1
    // fixture and a v2 one are both measured against the right model).
    let expected_size = quantibbler_qpk1::expected_size(
        parsed.dout,
        parsed.din,
        parsed.r,
        parsed.dtype,
        parsed.version >= quantibbler_qpk1::VERSION,
    );
    assert_eq!(
        raw.len(),
        expected_size,
        "fixture stream must match the size formula"
    );

    // Serve the fixture model end to end through the REAL binary, over both
    // committed serving sources (BS-8.1).
    for_each_serving_source(|artifact| {
        let server = spawn_server_artifact(artifact, &[], &[]);
        assert!(wait_ready(server.port, std::time::Duration::from_secs(30)));
        let resp = post_json(
            server.port,
            "/v1/completions",
            &completion_body(FIXTURE_MODEL_ID, false),
        )
        .expect("completion request");
        assert_eq!(resp.status, 200, "{}: {}", artifact.display(), resp.body);
        let body: serde_json::Value = serde_json::from_str(&resp.body).expect("completion JSON");
        assert_eq!(body["object"], serde_json::json!("text_completion"));
        assert!(body["choices"][0]["text"].is_string());
    });
}
